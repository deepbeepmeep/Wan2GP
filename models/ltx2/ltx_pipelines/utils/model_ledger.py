from dataclasses import replace
import json
import os

import torch

from ...ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ...ltx_core.loader.registry import DummyRegistry, Registry
from ...ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ...ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    AudioEncoder,
    AudioEncoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ...ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
    UPCAST_DURING_INFERENCE,
    LTXModelConfigurator,
    X0Model,
)
from ...ltx_core.model.upsampler import LatentUpsampler, LatentUpsamplerConfigurator
from ...ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ...ltx_core.text_encoders.gemma import (
    GEMMA_TEXT_ENCODER_KEY_OPS,
    TEXT_EMBEDDING_PROJECTION_KEY_OPS,
    TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS,
    GemmaTextEmbeddingsConnectorModel,
    GemmaTextEmbeddingsConnectorModelConfigurator,
    GemmaTextEncoderModel,
    GemmaTextEncoderModelConfigurator,
    module_ops_from_gemma_root,
)
from ...ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear


class ModelLedger:
    """
    Central coordinator for loading and building models used in an LTX pipeline.
    The ledger wires together multiple model builders (transformer, video VAE encoder/decoder,
    audio VAE decoder, vocoder, text encoder, and optional latent upsampler) and exposes
    factory methods for constructing model instances.
    ### Model Building
    Each model method (e.g. :meth:`transformer`, :meth:`video_decoder`, :meth:`text_encoder`)
    constructs a new model instance on each call. The builder uses the
    :class:`~ltx_core.loader.registry.Registry` to load weights from the checkpoint,
    instantiates the model with the configured ``dtype``, and moves it to ``self.device``.
    .. note::
        Models are **not cached**. Each call to a model method creates a new instance.
        Callers are responsible for storing references to models they wish to reuse
        and for freeing GPU memory (e.g. by deleting references and calling
        ``torch.cuda.empty_cache()``).
    ### Constructor parameters
    dtype:
        Torch dtype used when constructing all models (e.g. ``torch.bfloat16``).
    device:
        Target device to which models are moved after construction (e.g. ``torch.device("cuda")``).
    checkpoint_path:
        Path to a checkpoint directory or file containing the core model weights
        (transformer, video VAE, audio VAE, text encoder, vocoder). If ``None``, the
        corresponding builders are not created and calling those methods will raise
        a :class:`ValueError`.
    gemma_root_path:
        Base path to Gemma-compatible CLIP/text encoder weights. Required to
        initialize the text encoder builder; if omitted, :meth:`text_encoder` cannot be used.
    spatial_upsampler_path:
        Optional path to a latent upsampler checkpoint. If provided, the
        :meth:`spatial_upsampler` method becomes available; otherwise calling it raises
        a :class:`ValueError`.
    loras:
        Optional collection of LoRA configurations (paths, strengths, and key operations)
        that are applied on top of the base transformer weights when building the model.
    registry:
        Optional :class:`Registry` instance for weight caching across builders.
        Defaults to :class:`DummyRegistry` which performs no cross-builder caching.
    fp8transformer:
        If ``True``, builds the transformer with FP8 quantization and upcasting during inference.
    ### Creating Variants
    Use :meth:`with_loras` to create a new ``ModelLedger`` instance that includes
    additional LoRA configurations while sharing the same registry for weight caching.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        checkpoint_path: str | list[str] | tuple[str, ...] | dict[str, str] | None = None,
        gemma_root_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: LoraPathStrengthAndSDOps | None = None,
        registry: Registry | None = None,
        fp8transformer: bool = False,
        shared_state_dict: dict | None = None,
        shared_quantization_map: dict | None = None,
        shared_config: dict | None = None,
    ):
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.gemma_root_path = gemma_root_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.loras = loras or ()
        self.registry = registry or DummyRegistry()
        self.fp8transformer = fp8transformer
        self._shared_state_dict = shared_state_dict
        self._shared_quantization_map = shared_quantization_map
        self._shared_config = shared_config
        self._component_paths = checkpoint_path if isinstance(checkpoint_path, dict) else None
        if self.checkpoint_path is not None and self._shared_state_dict is None and self._component_paths is None:
            (
                self._shared_state_dict,
                self._shared_quantization_map,
                self._shared_config,
            ) = self._load_checkpoint_state(self.checkpoint_path)
        self.build_model_builders()

    def _get_component_path(self, component: str) -> str | list[str] | tuple[str, ...]:
        if self._component_paths is None:
            raise ValueError("Component paths are not available for this ledger.")
        path = self._component_paths.get(component)
        if not path:
            raise ValueError(f"Missing '{component}' path for multi-file ledger.")
        return path

    def _load_checkpoint_state(self, checkpoint_path: str | list[str] | tuple[str, ...]):
        from mmgp import safetensors2

        state_dict = {}
        quantization_map = None
        config = None
        model_paths = list(checkpoint_path) if isinstance(checkpoint_path, (list, tuple)) else [checkpoint_path]
        for shard_path in model_paths:
            with safetensors2.safe_open(shard_path, framework="pt", device="cpu",writable_tensors=False) as f:
                metadata = f.metadata() or {}
                if config is None:
                    config = metadata.get("config")
                    if isinstance(config, str):
                        config = json.loads(config)
                if quantization_map is None:
                    quantization_map = metadata.get("quantization_map")
                    if isinstance(quantization_map, str):
                        quantization_map = json.loads(quantization_map)
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        if quantization_map is None and model_paths:
            base_path, _ = os.path.splitext(model_paths[0])
            quant_map_path = f"{base_path}_map.json"
            if os.path.isfile(quant_map_path):
                with open(quant_map_path, "r", encoding="utf-8") as reader:
                    quantization_map = json.load(reader)
        return state_dict, quantization_map, config

    def build_model_builders(self) -> None:
        if self.checkpoint_path is not None and self._component_paths is None:
            self.transformer_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=LTXModelConfigurator,
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(self.loras),
                registry=self.registry,
                shared_state_dict=self._shared_state_dict,
                shared_quantization_map=self._shared_quantization_map,
                shared_config=self._shared_config,
                consume_shared_state_dict=True,
            )

            self.vae_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
                shared_state_dict=self._shared_state_dict,
                shared_quantization_map=self._shared_quantization_map,
                shared_config=self._shared_config,
                copy_shared_state_dict=True,
                consume_shared_state_dict=True,
            )

            self.vae_encoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
                shared_state_dict=self._shared_state_dict,
                shared_quantization_map=self._shared_quantization_map,
                shared_config=self._shared_config,
                copy_shared_state_dict=True,
                consume_shared_state_dict=False, #True,
            )

            self.audio_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
                shared_state_dict=self._shared_state_dict,
                shared_quantization_map=self._shared_quantization_map,
                shared_config=self._shared_config,
                consume_shared_state_dict=True,
            )

            self.audio_encoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioEncoderConfigurator,
                model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
                shared_state_dict=self._shared_state_dict,
                shared_quantization_map=self._shared_quantization_map,
                shared_config=self._shared_config,
                copy_shared_state_dict=True,
                consume_shared_state_dict=False,
            )

            self.vocoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VocoderConfigurator,
                model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
                shared_state_dict=self._shared_state_dict,
                shared_quantization_map=self._shared_quantization_map,
                shared_config=self._shared_config,
                consume_shared_state_dict=True,
            )

            if self.gemma_root_path is not None:
                self.text_encoder_builder = Builder(
                    model_path=self.checkpoint_path,
                    model_class_configurator=GemmaTextEncoderModelConfigurator,
                    model_sd_ops=GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=self.registry,
                    module_ops=module_ops_from_gemma_root(self.gemma_root_path),
                    shared_state_dict=self._shared_state_dict,
                    shared_quantization_map=self._shared_quantization_map,
                    shared_config=self._shared_config,
                    ignore_missing_keys=True,
                    consume_shared_state_dict=True,
                )
                self.text_embedding_projection_builder = Builder(
                    model_path=self.checkpoint_path,
                    model_class_configurator=GemmaFeaturesExtractorProjLinear,
                    model_sd_ops=TEXT_EMBEDDING_PROJECTION_KEY_OPS,
                    registry=self.registry,
                    shared_state_dict=self._shared_state_dict,
                    shared_quantization_map=self._shared_quantization_map,
                    shared_config=self._shared_config,
                    copy_shared_state_dict=True,
                    consume_shared_state_dict=True,
                )
                self.text_embeddings_connector_builder = Builder(
                    model_path=self.checkpoint_path,
                    model_class_configurator=GemmaTextEmbeddingsConnectorModelConfigurator,
                    model_sd_ops=TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS,
                    registry=self.registry,
                    shared_state_dict=self._shared_state_dict,
                    shared_quantization_map=self._shared_quantization_map,
                    shared_config=self._shared_config,
                    copy_shared_state_dict=True,
                    consume_shared_state_dict=True,
                )

        if self._component_paths is not None:
            transformer_path = self._get_component_path("transformer")
            video_vae_path = self._get_component_path("video_vae")
            audio_vae_path = self._get_component_path("audio_vae")
            vocoder_path = self._get_component_path("vocoder")
            text_projection_path = self._get_component_path("text_embedding_projection")
            text_connector_path = self._get_component_path("text_embeddings_connector")

            self.transformer_builder = Builder(
                model_path=transformer_path,
                model_class_configurator=LTXModelConfigurator,
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(self.loras),
                registry=self.registry,
            )

            self.vae_decoder_builder = Builder(
                model_path=video_vae_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vae_encoder_builder = Builder(
                model_path=video_vae_path,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.audio_decoder_builder = Builder(
                model_path=audio_vae_path,
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.audio_encoder_builder = Builder(
                model_path=audio_vae_path,
                model_class_configurator=AudioEncoderConfigurator,
                model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vocoder_builder = Builder(
                model_path=vocoder_path,
                model_class_configurator=VocoderConfigurator,
                model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            if self.gemma_root_path is not None:
                empty_text_sd = {}
                empty_text_config = {}
                self.text_encoder_builder = Builder(
                    model_path=transformer_path,
                    model_class_configurator=GemmaTextEncoderModelConfigurator,
                    model_sd_ops=GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=self.registry,
                    module_ops=module_ops_from_gemma_root(self.gemma_root_path),
                    shared_state_dict=empty_text_sd,
                    shared_config=empty_text_config,
                    ignore_missing_keys=True,
                )
                self.text_embedding_projection_builder = Builder(
                    model_path=text_projection_path,
                    model_class_configurator=GemmaFeaturesExtractorProjLinear,
                    model_sd_ops=TEXT_EMBEDDING_PROJECTION_KEY_OPS,
                    registry=self.registry,
                )
                self.text_embeddings_connector_builder = Builder(
                    model_path=text_connector_path,
                    model_class_configurator=GemmaTextEmbeddingsConnectorModelConfigurator,
                    model_sd_ops=TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS,
                    registry=self.registry,
                )

        if self.spatial_upsampler_path is not None:
            self.upsampler_builder = Builder(
                model_path=self.spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self.registry,
            )

    def _target_device(self) -> torch.device:
        if isinstance(self.registry, DummyRegistry) or self.registry is None:
            return self.device
        else:
            return torch.device("cpu")

    def with_loras(self, loras: LoraPathStrengthAndSDOps) -> "ModelLedger":
        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            gemma_root_path=self.gemma_root_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            loras=(*self.loras, *loras),
            registry=self.registry,
            fp8transformer=self.fp8transformer,
            shared_state_dict=self._shared_state_dict,
            shared_quantization_map=self._shared_quantization_map,
            shared_config=self._shared_config,
        )

    def release_shared_state(self) -> None:
        self._shared_state_dict = None
        self._shared_quantization_map = None
        builder_names = (
            "transformer_builder",
            "vae_decoder_builder",
            "vae_encoder_builder",
            "audio_encoder_builder",
            "audio_decoder_builder",
            "vocoder_builder",
            "text_encoder_builder",
            "text_embedding_projection_builder",
            "text_embeddings_connector_builder",
            "upsampler_builder",
        )
        for name in builder_names:
            builder = getattr(self, name, None)
            if builder is None:
                continue
            setattr(
                self,
                name,
                replace(
                    builder,
                    shared_state_dict=None,
                    shared_quantization_map=None,
                ),
            )

    def transformer(self) -> X0Model:
        if not hasattr(self, "transformer_builder"):
            raise ValueError(
                "Transformer not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )
        if self.fp8transformer:
            fp8_builder = replace(
                self.transformer_builder,
                module_ops=(UPCAST_DURING_INFERENCE,),
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
            )
            return X0Model(fp8_builder.build(device=self._target_device())).to(self.device).eval()
        else:
            return (
                X0Model(self.transformer_builder.build(device=self._target_device(), dtype=self.dtype))
                .to(self.device)
                .eval()
            )

    def video_decoder(self) -> VideoDecoder:
        if not hasattr(self, "vae_decoder_builder"):
            raise ValueError(
                "Video decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return self.vae_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def video_encoder(self) -> VideoEncoder:
        if not hasattr(self, "vae_encoder_builder"):
            raise ValueError(
                "Video encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return self.vae_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def audio_encoder(self) -> AudioEncoder:
        if not hasattr(self, "audio_encoder_builder"):
            raise ValueError(
                "Audio encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return self.audio_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def text_encoder(self) -> GemmaTextEncoderModel:
        if not hasattr(self, "text_encoder_builder"):
            raise ValueError(
                "Text encoder not initialized. Please provide a checkpoint path and gemma root path to the "
                "ModelLedger constructor."
            )

        return self.text_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def text_embedding_projection(self) -> GemmaFeaturesExtractorProjLinear:
        if not hasattr(self, "text_embedding_projection_builder"):
            raise ValueError(
                "Text embedding projection not initialized. Please provide a checkpoint path and gemma root path to "
                "the ModelLedger constructor."
            )

        return (
            self.text_embedding_projection_builder.build(device=self._target_device(), dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    def text_embeddings_connector(self) -> GemmaTextEmbeddingsConnectorModel:
        if not hasattr(self, "text_embeddings_connector_builder"):
            raise ValueError(
                "Text embeddings connector not initialized. Please provide a checkpoint path and gemma root path to "
                "the ModelLedger constructor."
            )

        return (
            self.text_embeddings_connector_builder.build(device=self._target_device(), dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    def audio_decoder(self) -> AudioDecoder:
        if not hasattr(self, "audio_decoder_builder"):
            raise ValueError(
                "Audio decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return self.audio_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def vocoder(self) -> Vocoder:
        if not hasattr(self, "vocoder_builder"):
            raise ValueError(
                "Vocoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return self.vocoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

    def spatial_upsampler(self) -> LatentUpsampler:
        if not hasattr(self, "upsampler_builder"):
            raise ValueError("Upsampler not initialized. Please provide upsampler path to the ModelLedger constructor.")

        return self.upsampler_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
