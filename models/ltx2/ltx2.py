import copy
import json
import math
import os
import re
import types
from typing import Callable, Iterator

import torch
import torchaudio
from accelerate import init_empty_weights
from shared.utils import files_locator as fl

from .ltx_core.conditioning import AudioConditionByLatent
from .ltx_core.model.audio_vae import (
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    AudioProcessor,
    VocoderConfigurator,
)
from .ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from .ltx_core.model.upsampler import LatentUpsamplerConfigurator
from .ltx_core.model.video_vae import VideoDecoderConfigurator, VideoEncoderConfigurator
from .ltx_core.text_encoders.gemma import (
    GemmaTextEmbeddingsConnectorModelConfigurator,
    TEXT_EMBEDDING_PROJECTION_KEY_OPS,
    TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS,
    build_gemma_text_encoder,
)
from .ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from .ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from .ltx_core.types import AudioLatentShape, VideoPixelShape
from .ltx_pipelines.distilled import DistilledPipeline
from .ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from .ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DEFAULT_NEGATIVE_PROMPT


_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
_SPATIAL_UPSCALER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
LTX2_USE_FP32_ROPE_FREQS = True #False


def _normalize_config(config_value):
    if isinstance(config_value, dict):
        return config_value
    if isinstance(config_value, (bytes, bytearray, memoryview)):
        try:
            config_value = bytes(config_value).decode("utf-8")
        except Exception:
            return {}
    if isinstance(config_value, str):
        try:
            return json.loads(config_value)
        except json.JSONDecodeError:
            return {}
    return {}


def _load_config_from_checkpoint(path, fallback_config_path: str | None = None):
    from mmgp import quant_router

    if isinstance(path, (list, tuple)):
        if not path:
            return {}
        path = path[0]
    if not path:
        return {}

    def _read_config_metadata(one_path: str) -> dict:
        if not one_path:
            return {}
        _, metadata = quant_router.load_metadata_state_dict(one_path)
        if not metadata:
            return {}
        return _normalize_config(metadata.get("config"))

    config = _read_config_metadata(path)
    if config:
        return config
    if not fallback_config_path:
        return {}
    try:
        with open(fallback_config_path, "r", encoding="utf-8") as reader:
            return _normalize_config(json.load(reader))
    except Exception:
        return {}


def _strip_model_prefix(key: str) -> str:
    for prefix in ("model.", "velocity_model."):
        if key.startswith(prefix):
            return _strip_model_prefix(key[len(prefix) :])
    return key


def _apply_sd_ops(state_dict: dict, quantization_map: dict | None, sd_ops):
    if sd_ops is not None:
        has_match = False
        for key in state_dict.keys():
            key = _strip_model_prefix(key)
            if sd_ops.apply_to_key(key) is not None:
                has_match = True
                break
        if not has_match:
            new_sd = {_strip_model_prefix(k): v for k, v in state_dict.items()}
            new_qm = {}
            if quantization_map:
                new_qm = {_strip_model_prefix(k): v for k, v in quantization_map.items()}
            return new_sd, new_qm

    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        if sd_ops is None:
            new_sd[key] = value
            continue
        else:
            new_key = sd_ops.apply_to_key(key)
            if new_key is None:
                continue
            new_pairs = sd_ops.apply_to_key_value(new_key, value)
        for pair in new_pairs:
            new_sd[pair.new_key] = pair.new_value

    new_qm = {}
    if quantization_map:
        for key, value in quantization_map.items():
            key = _strip_model_prefix(key)
            if sd_ops is None:
                new_key = key
            else:
                new_key = sd_ops.apply_to_key(key)
                if new_key is None:
                    continue
            new_qm[new_key] = value
    return new_sd, new_qm


def _make_sd_postprocess(sd_ops):
    def postprocess(state_dict, quantization_map):
        return _apply_sd_ops(state_dict, quantization_map, sd_ops)

    return postprocess


def _split_vae_state_dict(state_dict: dict, prefix: str):
    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        if key.startswith(prefix):
            key = key[len(prefix) :]
        elif key.startswith(("encoder.", "decoder.", "per_channel_statistics.")):
            key = key
        else:
            continue
        if key.startswith("per_channel_statistics."):
            suffix = key[len("per_channel_statistics.") :]
            new_sd[f"encoder.per_channel_statistics.{suffix}"] = value.clone()
            new_sd[f"decoder.per_channel_statistics.{suffix}"] = value.clone()
        else:
            new_sd[key] = value

    return new_sd, {}


def _make_vae_postprocess(prefix: str):
    def postprocess(state_dict, quantization_map):
        return _split_vae_state_dict(state_dict, prefix)

    return postprocess


class _AudioVAEWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        per_stats = getattr(decoder, "per_channel_statistics", None)
        if per_stats is not None:
            self.per_channel_statistics = per_stats
        self.decoder = decoder


class _VAEContainer(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class _ExternalConnectorWrapper:
    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def __call__(self, *args, **kwargs):
        return self._module(*args, **kwargs)


class LTX2SuperModel(torch.nn.Module):
    def __init__(self, ltx2_model: "LTX2") -> None:
        super().__init__()
        object.__setattr__(self, "_ltx2", ltx2_model)

        transformer = ltx2_model.model
        velocity_model = getattr(transformer, "velocity_model", transformer)
        self.velocity_model = velocity_model
        split_map = getattr(transformer, "split_linear_modules_map", None)
        if split_map is not None:
            self.split_linear_modules_map = split_map

        self.text_embedding_projection = ltx2_model.text_embedding_projection
        self.video_embeddings_connector = ltx2_model.video_embeddings_connector
        self.audio_embeddings_connector = ltx2_model.audio_embeddings_connector

    @property
    def _interrupt(self) -> bool:
        return self._ltx2._interrupt

    @_interrupt.setter
    def _interrupt(self, value: bool) -> None:
        self._ltx2._interrupt = value

    def forward(self, *args, **kwargs):
        return self._ltx2.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._ltx2.generate(*args, **kwargs)

    def get_trans_lora(self):
        return self, None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._ltx2, name)


class _LTX2VAEHelper:
    def __init__(self, block_size: int = 64) -> None:
        self.block_size = block_size

    def get_VAE_tile_size(
        self,
        vae_config: int,
        device_mem_capacity: float,
        mixed_precision: bool,
        output_height: int | None = None,
        output_width: int | None = None,
    ) -> int | tuple[int, int]:
        if vae_config >= 4:
            vae_config = 0

        if vae_config == 0:
            if mixed_precision:
                device_mem_capacity = device_mem_capacity / 1.5
            if device_mem_capacity >= 24000:
                use_vae_config = 1
            elif device_mem_capacity >= 8000:
                use_vae_config = 2
            else:
                use_vae_config = 3
        else:
            use_vae_config = vae_config

        ref_size = output_height if output_height is not None else output_width
        if ref_size is not None and ref_size > 480:
            use_vae_config += 1

        spatial_tile_size = 128
        if use_vae_config <= 1:
            spatial_tile_size = 0
        elif use_vae_config == 2:
            spatial_tile_size = 512
        elif use_vae_config == 3:
            spatial_tile_size = 256

        return spatial_tile_size


def _attach_lora_preprocessor(transformer: torch.nn.Module) -> None:
    def preprocess_loras(self: torch.nn.Module, model_type: str, sd: dict) -> dict:
        if not sd:
            return sd
        module_names = getattr(self, "_lora_module_names", None)
        if module_names is None:
            module_names = {name for name, _ in self.named_modules()}
            self._lora_module_names = module_names

        def split_lora_key(lora_key: str) -> tuple[str | None, str]:
            if lora_key.endswith(".alpha"):
                return lora_key[: -len(".alpha")], ".alpha"
            if lora_key.endswith(".diff"):
                return lora_key[: -len(".diff")], ".diff"
            if lora_key.endswith(".diff_b"):
                return lora_key[: -len(".diff_b")], ".diff_b"
            if lora_key.endswith(".dora_scale"):
                return lora_key[: -len(".dora_scale")], ".dora_scale"
            pos = lora_key.rfind(".lora_")
            if pos > 0:
                return lora_key[:pos], lora_key[pos:]
            return None, ""

        new_sd = {}
        dropped_keys = []
        for key, value in sd.items():
            original_key = key
            if key.startswith("model."):
                key = key[len("model.") :]
            if key.startswith("diffusion_model."):
                key = key[len("diffusion_model.") :]
            if key.startswith("transformer."):
                key = key[len("transformer.") :]
            if key.startswith("embeddings_connector."):
                key = f"video_embeddings_connector.{key[len('embeddings_connector.'):]}"
            if key.startswith("feature_extractor_linear."):
                key = f"text_embedding_projection.{key[len('feature_extractor_linear.'):]}"

            module_name, suffix = split_lora_key(key)
            if not module_name:
                dropped_keys.append(original_key)
                continue
            if module_name not in module_names:
                prefixed_name = f"velocity_model.{module_name}"
                if prefixed_name in module_names:
                    module_name = prefixed_name
                else:
                    dropped_keys.append(original_key)
                    continue
            new_sd[f"{module_name}{suffix}"] = value
        if dropped_keys:
            sample = ", ".join(dropped_keys[:8])
            if len(dropped_keys) > 8:
                sample += ", ..."
            raise ValueError(
                f"LTX2 LoRA preprocessing dropped {len(dropped_keys)} unmatched keys for model '{model_type}': {sample}"
            )
        return new_sd

    transformer.preprocess_loras = types.MethodType(preprocess_loras, transformer)


def _coerce_image_list(image_value):
    if isinstance(image_value, list):
        return image_value[0] if image_value else None
    return image_value


def _to_latent_index(frame_idx: int, stride: int) -> int:
    frame_idx = int(frame_idx)
    stride = int(stride)
    if frame_idx <= 0:
        return 0
    # Causal LTX VAEs keep pixel frame 0 in its own latent slot.
    return (frame_idx - 1) // stride + 1


def _normalize_tiling_size(tile_size: int) -> int:
    tile_size = int(tile_size)
    if tile_size <= 0:
        return 0
    tile_size = max(64, tile_size)
    if tile_size % 32 != 0:
        tile_size = int(math.ceil(tile_size / 32) * 32)
    return tile_size


def _normalize_temporal_tiling_size(tile_frames: int) -> int:
    tile_frames = int(tile_frames)
    if tile_frames <= 0:
        return 0
    tile_frames = max(16, tile_frames)
    if tile_frames % 8 != 0:
        tile_frames = int(math.ceil(tile_frames / 8) * 8)
    return tile_frames


def _normalize_temporal_overlap(overlap_frames: int, tile_frames: int) -> int:
    overlap_frames = max(0, int(overlap_frames))
    if overlap_frames % 8 != 0:
        overlap_frames = int(round(overlap_frames / 8) * 8)
    overlap_frames = max(0, min(overlap_frames, max(0, tile_frames - 8)))
    return overlap_frames


def _build_tiling_config(tile_size: int | tuple | list | None, fps: float | None) -> TilingConfig | None:
    temporal_tiling_divisor = 1
    spatial_config = None
    if isinstance(tile_size, (tuple, list)):
        if len(tile_size) == 0:
            tile_size = None
        else:
            if len(tile_size) > 1:
                temporal_tiling_divisor = max(1, int(tile_size[0] or 1))
            tile_size = tile_size[-1]
    if tile_size is not None:
        tile_size = _normalize_tiling_size(tile_size)
        if tile_size > 0:
            overlap = max(0, tile_size // 4)
            overlap = int(math.floor(overlap / 32) * 32)
            if overlap >= tile_size:
                overlap = max(0, tile_size - 32)
            spatial_config = SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=overlap)

    temporal_config = None
    if fps is not None and fps > 0:
        temporal_tiling_divisor = max(1, temporal_tiling_divisor)
        tile_frames = _normalize_temporal_tiling_size(int(math.ceil(float(fps) * 5.0 / temporal_tiling_divisor)))
        if tile_frames > 0:
            overlap_frames = int(round(tile_frames * 3 / 8))
            overlap_frames = _normalize_temporal_overlap(overlap_frames, tile_frames)
            temporal_config = TemporalTilingConfig(
                tile_size_in_frames=tile_frames,
                tile_overlap_in_frames=overlap_frames,
            )

    if spatial_config is None and temporal_config is None:
        return None
    return TilingConfig(spatial_config=spatial_config, temporal_config=temporal_config)


def _infer_ic_lora_downscale_factor(loras_selected) -> int | None:
    factors = []
    for lora_path in loras_selected or []:
        name = os.path.basename(str(lora_path)).lower()
        if "ic-lora" not in name:
            continue
        match = re.search(r"-ref([0-9]+(?:\.[0-9]+)?)", name)
        if not match:
            factors.append(1)
            continue
        ref_ratio = float(match.group(1))
        if ref_ratio <= 0:
            factors.append(1)
            continue
        factors.append(max(1, int(round(1.0 / ref_ratio))))
    if not factors:
        return None
    unique_factors = sorted(set(factors))
    if len(unique_factors) > 1:
        raise ValueError(f"Conflicting IC-LoRA reference downscale factors in selected LoRAs: {unique_factors}")
    return unique_factors[0]


def _collect_video_chunks(
    video: Iterator[torch.Tensor] | torch.Tensor,
    interrupt_check: Callable[[], bool] | None = None,
) -> torch.Tensor | None:
    if video is None:
        return None
    if torch.is_tensor(video):
        chunks = [video]
    else:
        chunks = []
        for chunk in video:
            if interrupt_check is not None and interrupt_check():
                return None
            if chunk is None:
                continue
            chunks.append(chunk if torch.is_tensor(chunk) else torch.tensor(chunk))
    if not chunks:
        return None
    frames = torch.cat(chunks, dim=0)
    return frames.permute(3, 0, 1, 2)
    # frames = frames.to(dtype=torch.float32).div_(127.5).sub_(1.0)
    # return frames.permute(3, 0, 1, 2).contiguous()


class LTX2:
    def __init__(
        self,
        model_filename,
        model_type: str,
        base_model_type: str,
        model_def: dict,
        dtype: torch.dtype = torch.bfloat16,
        VAE_dtype: torch.dtype = torch.float32,
        text_encoder_filename: str | None = None,
        text_encoder_filepath = None,
        checkpoint_paths: dict | None = None,
    ) -> None:
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.VAE_dtype = VAE_dtype
        self.base_model_type = base_model_type
        self.model_def = model_def
        self._interrupt = False
        self.vae = _LTX2VAEHelper()
        from .ltx_core.model.transformer import rope as rope_utils

        self.use_fp32_rope_freqs = bool(model_def.get("ltx2_rope_freqs_fp32", LTX2_USE_FP32_ROPE_FREQS))
        rope_utils.set_use_fp32_rope_freqs(self.use_fp32_rope_freqs)

        if isinstance(model_filename, (list, tuple)):
            if not model_filename:
                raise ValueError("Missing LTX-2 checkpoint path.")
            transformer_path = list(model_filename)
        else:
            transformer_path = model_filename
        component_paths = checkpoint_paths or {}
        if component_paths:
            transformer_path = component_paths.get("transformer")
            if not transformer_path:
                raise ValueError("Missing transformer path in checkpoint_paths.")

        gemma_root = text_encoder_filepath if text_encoder_filename is None else text_encoder_filename
        if not gemma_root:
            raise ValueError("Missing Gemma text encoder path.")
        if component_paths:
            spatial_upsampler_path = component_paths.get("spatial_upsampler")
        else:
            spatial_upsampler_path = None
        if not spatial_upsampler_path:
            spatial_upsampler_name = model_def.get("ltx2_spatial_upscaler_file", _SPATIAL_UPSCALER_FILENAME)
            spatial_upsampler_path = fl.locate_file(spatial_upsampler_name)

        # Internal FP8 handling is disabled; mmgp manages quantization/dtypes.
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")

        pipeline_models = self._init_models(
            transformer_path=transformer_path,
            component_paths=component_paths,
            gemma_root=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
        )

        if pipeline_kind == "distilled":
            self.pipeline = DistilledPipeline(
                device=self.device,
                models=pipeline_models,
            )
        else:
            self.pipeline = TI2VidTwoStagesPipeline(
                device=self.device,
                stage_1_models=pipeline_models,
                stage_2_models=pipeline_models,
            )
        self._build_diffuser_model()

    def _init_models(
        self,
        transformer_path,
        component_paths: dict,
        gemma_root: str,
        spatial_upsampler_path: str,
    ):
        from mmgp import offload as mmgp_offload

        fallback_config_path = component_paths.get("model_config") if component_paths else None
        base_config = _load_config_from_checkpoint(transformer_path, fallback_config_path=fallback_config_path)
        if not base_config:
            raise ValueError("Missing config in transformer checkpoint.")

        def _component_path(key: str):
            if component_paths:
                path = component_paths.get(key)
                if not path:
                    raise ValueError(f"Missing '{key}' path in checkpoint_paths.")
                return path
            return transformer_path

        def _component_config(path):
            config = _load_config_from_checkpoint(path, fallback_config_path=fallback_config_path)
            return config or base_config

        def _load_component(model, path, sd_ops=None, postprocess=None, ignore_unused_weights=False):
            if postprocess is None and sd_ops is not None:
                postprocess = _make_sd_postprocess(sd_ops)
            mmgp_offload.load_model_data(
                model,
                path,
                postprocess_sd=postprocess,
                default_dtype=self.dtype,
                writable_tensors=False,
                ignore_missing_keys=False,
                ignore_unused_weights=ignore_unused_weights,
            )
            model.eval().requires_grad_(False)
            return model

        transformer_sd_ops = LTXV_MODEL_COMFY_RENAMING_MAP
        with init_empty_weights():
            velocity_model = LTXModelConfigurator.from_config(base_config)
        velocity_model = _load_component(velocity_model, transformer_path, transformer_sd_ops, ignore_unused_weights=True)
        transformer = X0Model(velocity_model)
        transformer.eval().requires_grad_(False)
        VAE_URLs = self.model_def.get("VAE_URLs", None)
        video_vae_path =  fl.locate_file(VAE_URLs[0]) if VAE_URLs is not None and len(VAE_URLs) else _component_path("video_vae")
        video_config = copy.deepcopy(_component_config(video_vae_path))
        video_config_vae = video_config.setdefault("vae", {})
        video_config_vae["spatial_padding_mode"] = "reflect"
        video_config_vae["encoder_spatial_padding_mode"] = "reflect"
        video_config_vae["decoder_spatial_padding_mode"] = "reflect"
        # print("[LTX2 VAE Config] forcing encoder/decoder spatial_padding_mode=reflect")
        with init_empty_weights():
            video_encoder = VideoEncoderConfigurator.from_config(video_config)
            video_decoder = VideoDecoderConfigurator.from_config(video_config)
            video_vae = _VAEContainer(video_encoder, video_decoder)
        video_vae = _load_component(video_vae, video_vae_path, postprocess=_make_vae_postprocess("vae."), ignore_unused_weights=True)
        video_encoder = video_vae.encoder
        video_decoder = video_vae.decoder

        audio_vae_path = _component_path("audio_vae")
        audio_config = _component_config(audio_vae_path)
        with init_empty_weights():
            audio_encoder = AudioEncoderConfigurator.from_config(audio_config)
            audio_decoder = AudioDecoderConfigurator.from_config(audio_config)
            audio_vae = _VAEContainer(audio_encoder, audio_decoder)
        audio_vae = _load_component(audio_vae, audio_vae_path, postprocess=_make_vae_postprocess("audio_vae."))
        audio_encoder = audio_vae.encoder
        audio_decoder = audio_vae.decoder

        vocoder_path = _component_path("vocoder")
        vocoder_config = _component_config(vocoder_path)
        with init_empty_weights():
            vocoder = VocoderConfigurator.from_config(vocoder_config)
        vocoder = _load_component(vocoder, vocoder_path, VOCODER_COMFY_KEYS_FILTER)

        text_projection_path = _component_path("text_embedding_projection")
        text_projection_config = _component_config(text_projection_path)
        with init_empty_weights():
            text_embedding_projection = GemmaFeaturesExtractorProjLinear.from_config(text_projection_config)
        text_embedding_projection = _load_component( text_embedding_projection, text_projection_path, TEXT_EMBEDDING_PROJECTION_KEY_OPS )

        text_connector_path = _component_path("text_embeddings_connector")
        text_connector_config = _component_config(text_connector_path)
        with init_empty_weights():
            text_embeddings_connector = GemmaTextEmbeddingsConnectorModelConfigurator.from_config(text_connector_config)
        text_embeddings_connector = _load_component( text_embeddings_connector, text_connector_path, TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS )

        text_encoder = build_gemma_text_encoder(gemma_root, default_dtype=self.dtype)
        text_encoder.eval().requires_grad_(False)

        upsampler_config = _load_config_from_checkpoint(spatial_upsampler_path)
        with init_empty_weights():
            spatial_upsampler = LatentUpsamplerConfigurator.from_config(upsampler_config)
        spatial_upsampler = _load_component(spatial_upsampler, spatial_upsampler_path, None)

        self.text_encoder = text_encoder
        self.text_embedding_projection = text_embedding_projection
        self.text_embeddings_connector = text_embeddings_connector
        self.video_embeddings_connector = text_embeddings_connector.video_embeddings_connector
        self.audio_embeddings_connector = text_embeddings_connector.audio_embeddings_connector
        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.audio_encoder = audio_encoder
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.spatial_upsampler = spatial_upsampler
        self.model = transformer
        self.model2 = None

        return types.SimpleNamespace(
            text_encoder=self.text_encoder,
            text_embedding_projection=self.text_embedding_projection,
            text_embeddings_connector=self.text_embeddings_connector,
            video_encoder=self.video_encoder,
            video_decoder=self.video_decoder,
            audio_encoder=self.audio_encoder,
            audio_decoder=self.audio_decoder,
            vocoder=self.vocoder,
            spatial_upsampler=self.spatial_upsampler,
            transformer=self.model,
        )

    def _detach_text_encoder_connectors(self) -> None:
        text_encoder = getattr(self, "text_encoder", None)
        if text_encoder is None:
            return
        connectors = {}
        feature_extractor = getattr(self, "text_embedding_projection", None)
        video_connector = getattr(self, "video_embeddings_connector", None)
        audio_connector = getattr(self, "audio_embeddings_connector", None)
        if feature_extractor is not None:
            connectors["feature_extractor_linear"] = feature_extractor
        if video_connector is not None:
            connectors["embeddings_connector"] = video_connector
        if audio_connector is not None:
            connectors["audio_embeddings_connector"] = audio_connector
        if not connectors:
            return
        for name, module in connectors.items():
            if name in text_encoder._modules:
                del text_encoder._modules[name]
            setattr(text_encoder, name, _ExternalConnectorWrapper(module))
        self._text_connectors = connectors

    def _build_diffuser_model(self) -> None:
        self._detach_text_encoder_connectors()
        self.diffuser_model = LTX2SuperModel(self)
        _attach_lora_preprocessor(self.diffuser_model)


    def get_trans_lora(self):
        trans = getattr(self, "diffuser_model", None)
        if trans is None:
            trans = self.model
        return trans, None

    def get_loras_transformer(self, get_model_recursive_prop, model_type, video_prompt_type, base_model_type=None, model_def = None, **kwargs):
        control_map = {
            "P": "pose",
            "D": "depth",
            "E": "canny",
        }
        loras = []
        video_prompt_type = video_prompt_type or ""
        if model_def.get("ltx2_pipeline","two_stage") != "distilled":
            return [], []
        preload_urls = get_model_recursive_prop(model_type, "preload_URLs")
        if (base_model_type or self.base_model_type) == "ltx2_22B":
            if any(letter in video_prompt_type for letter in control_map):
                for file_name in preload_urls:
                    if "union-control" in os.path.basename(file_name):
                        loras.append(fl.locate_file(os.path.basename(file_name)))
                        break
        else:
            for letter, signature in control_map.items():
                if letter in video_prompt_type:
                    for file_name in preload_urls:
                        if signature in file_name:
                            loras.append(fl.locate_file(os.path.basename(file_name)))
                            break
        loras_mult = [1.0] * len(loras)
        return loras, loras_mult

    def generate(
        self,
        input_prompt: str,
        n_prompt: str | None = None,
        image_start=None,
        image_end=None,
        sampling_steps: int = 40,
        guide_scale: float = 4.0,
        alt_guide_scale: float = 1.0,
        input_video=None,
        prefix_frames_count: int = 0,
        input_frames=None,
        input_frames2=None,
        input_ref_images=None,
        input_masks=None,
        input_masks2=None,
        frames_relative_positions_list=None,
        masking_strength: float | None = None,
        input_video_strength: float | None = None,
        return_latent_slice=None,
        video_prompt_type: str = "",
        denoising_strength: float | None = None,
        cfg_star_switch: int = 0,
        apg_switch: int = 0,
        perturbation_switch: int = 0,
        perturbation_layers: list[int] | None = None,
        perturbation_start: float = 0.0,
        perturbation_end: float = 1.0,
        audio_cfg_scale: float | None = None,
        alt_scale: float = 0.0,
        self_refiner_setting: int = 0,
        self_refiner_plan: str = "",
        self_refiner_f_uncertainty: float = 0.1,
        self_refiner_certain_percentage: float = 0.999,
        loras_slists=None,
        loras_selected=None,
        text_connectors=None,
        input_waveform=None,
        input_waveform_sample_rate=None,
        audio_scale: float | None = None,
        masking_source: dict | None = None,
        frame_num: int = 121,
        height: int = 1024,
        width: int = 1536,
        fps: float = 24.0,
        seed: int = 0,
        callback=None,
        VAE_tile_size=None,
        **kwargs,
    ):
        if self._interrupt:
            return None

        image_start = _coerce_image_list(image_start)
        image_end = _coerce_image_list(image_end)
        if input_ref_images is None:
            input_ref_images = []
        elif isinstance(input_ref_images, (list, tuple)):
            input_ref_images = list(input_ref_images)
        else:
            input_ref_images = [input_ref_images]
        if frames_relative_positions_list is None:
            frames_relative_positions_list = []
        elif isinstance(frames_relative_positions_list, (list, tuple)):
            frames_relative_positions_list = list(frames_relative_positions_list)
        else:
            frames_relative_positions_list = [frames_relative_positions_list]

        prefix_frames_count = int(prefix_frames_count or 0)
        video_prompt_type = video_prompt_type or ""
        self_refiner_max_plans = int(self.model_def.get("self_refiner_max_plans", 1))

        def _get_frame_dim(video_tensor: torch.Tensor) -> int | None:
            if video_tensor.dim() < 2:
                return None
            if video_tensor.dim() == 5:
                if video_tensor.shape[1] in (1, 3, 4):
                    return 2
                if video_tensor.shape[-1] in (1, 3, 4):
                    return 1
            if video_tensor.shape[0] in (1, 3, 4):
                return 1
            if video_tensor.shape[-1] in (1, 3, 4):
                return 0
            return 0

        def _frame_count(video_value) -> int | None:
            if not torch.is_tensor(video_value):
                return None
            frame_dim = _get_frame_dim(video_value)
            if frame_dim is None:
                return None
            return int(video_value.shape[frame_dim])

        def _slice_frames(video_value: torch.Tensor, start: int, end: int) -> torch.Tensor:
            frame_dim = _get_frame_dim(video_value)
            if frame_dim == 1:
                return video_value[:, start:end]
            if frame_dim == 2:
                return video_value[:, :, start:end]
            return video_value[start:end]

        def _maybe_trim_control(video_value, target_frames: int):
            if not torch.is_tensor(video_value) or target_frames <= 0:
                return video_value, None
            current_frames = _frame_count(video_value)
            if current_frames is None:
                return video_value, None
            if current_frames > target_frames:
                video_value = _slice_frames(video_value, 0, target_frames)
                current_frames = target_frames
            return video_value, current_frames

        try:
            masking_strength = float(masking_strength) if masking_strength is not None else 0.0
        except (TypeError, ValueError):
            masking_strength = 0.0
        try:
            input_video_strength = float(input_video_strength) if input_video_strength is not None else 1.0
        except (TypeError, ValueError):
            input_video_strength = 1.0
        input_video_strength = max(0.0, min(1.0, input_video_strength))
        if "G" not in video_prompt_type:
            denoising_strength = 1.0
            masking_strength = 0.0
        ic_lora_downscale_factor = None
        if self.model_def.get("ltx2_pipeline", "two_stage") == "distilled":
            ic_lora_downscale_factor = _infer_ic_lora_downscale_factor(loras_selected)
        video_conditioning_downscale_factor = ic_lora_downscale_factor or 1

        video_conditioning = None
        masking_source = None
        if input_frames is not None or input_frames2 is not None:
            control_start_frame = int(prefix_frames_count)
            expected_guide_frames = max(1, int(frame_num) - control_start_frame + (1 if prefix_frames_count > 1 else 0))
            if prefix_frames_count > 1:
                control_start_frame = -control_start_frame
            input_frames, frames_len = _maybe_trim_control(input_frames, expected_guide_frames)
            input_frames2, frames_len2 = _maybe_trim_control(input_frames2, expected_guide_frames)
            input_masks, _ = _maybe_trim_control(input_masks, expected_guide_frames)
            input_masks2, _ = _maybe_trim_control(input_masks2, expected_guide_frames)

            control_strength = 1.0
            if denoising_strength is not None and "G" in video_prompt_type:
                try:
                    control_strength = float(denoising_strength)
                except (TypeError, ValueError):
                    control_strength = 1.0
            control_strength = max(0.0, min(1.0, control_strength))

            conditioning_entries = []
            if input_frames is not None:
                conditioning_entries.append((input_frames, control_start_frame, control_strength))
            if input_frames2 is not None:
                conditioning_entries.append((input_frames2, control_start_frame, control_strength))
            if conditioning_entries:
                video_conditioning = conditioning_entries
            if masking_strength > 0.0:
                if input_masks is not None and input_frames is not None:
                    masking_source = {
                        "video": input_frames,
                        "mask": input_masks,
                        "start_frame": control_start_frame,
                    }
                elif input_masks2 is not None and input_frames2 is not None:
                    masking_source = {
                        "video": input_frames2,
                        "mask": input_masks2,
                        "start_frame": control_start_frame,
                    }

        latent_conditioning_stage2 = None

        latent_stride = 8
        if hasattr(self.pipeline, "pipeline_components"):
            scale_factors = getattr(self.pipeline.pipeline_components, "video_scale_factors", None)
            if scale_factors is not None:
                latent_stride = int(getattr(scale_factors, "time", scale_factors[0]))

        images = []
        guiding_images = []
        guiding_images_stage2 = []
        images_stage2 = []
        stage2_override = False
        has_prefix_frames = input_video is not None and torch.is_tensor(input_video) and prefix_frames_count > 0
        is_start_image_only = image_start is not None and (not has_prefix_frames or prefix_frames_count <= 1)
        use_guiding_latent_for_start_image = bool(self.model_def.get("use_guiding_latent_for_start_image", False))
        use_guiding_start_image = use_guiding_latent_for_start_image and is_start_image_only

        def _append_prefix_entries(target_list, extra_list=None):
            if not has_prefix_frames or is_start_image_only:
                return
            frame_count = min(prefix_frames_count, input_video.shape[1])
            if frame_count <= 0:
                return
            frame_indices = list(range(0, frame_count, latent_stride))
            last_idx = frame_count - 1
            if frame_indices[-1] != last_idx:
                # Ensure the latest prefix frame dominates its latent slot.
                frame_indices.append(last_idx)
            for frame_idx in frame_indices:
                entry = (input_video[:, frame_idx], _to_latent_index(frame_idx, latent_stride), input_video_strength)
                target_list.append(entry)
                if extra_list is not None:
                    extra_list.append(entry)

        def _append_injected_ref_entries(target_list, extra_list=None):
            injected_ref_count = min(len(input_ref_images), len(frames_relative_positions_list))
            for ref_image, frame_idx in zip(input_ref_images[:injected_ref_count], frames_relative_positions_list[:injected_ref_count]):
                entry = (ref_image, int(frame_idx), input_video_strength, "lanczos")
                target_list.append(entry)
                if extra_list is not None:
                    extra_list.append(entry)

        if isinstance(self.pipeline, TI2VidTwoStagesPipeline):
            _append_prefix_entries(images, images_stage2)

            if image_end is not None:
                entry = (image_end, int(frame_num - 1), input_video_strength)
                guiding_images.append(entry)
                guiding_images_stage2.append(entry)

            if image_start is not None:
                entry = (image_start, _to_latent_index(0, latent_stride), input_video_strength, "lanczos")
                if use_guiding_start_image:
                    guiding_images.append(entry)
                    images_stage2.append(entry)
                    stage2_override = True
                else:
                    images.append(entry)
                    images_stage2.append(entry)
            _append_injected_ref_entries(guiding_images, guiding_images_stage2)
        else:
            _append_prefix_entries(images)
            if image_start is not None:
                images.append((image_start, _to_latent_index(0, latent_stride), input_video_strength, "lanczos"))
            if image_end is not None:
                entry = (image_end, int(frame_num - 1), input_video_strength)
                guiding_images.append(entry)
                guiding_images_stage2.append(entry)
            _append_injected_ref_entries(guiding_images, guiding_images_stage2)

        tiling_config = _build_tiling_config(VAE_tile_size, fps)
        interrupt_check = lambda: self._interrupt
        text_connectors = text_connectors or getattr(self, "_text_connectors", None)

        audio_conditionings = None
        if input_waveform is not None:
            if audio_scale is None:
                audio_scale = 1.0
            audio_strength = max(0.0, min(1.0, float(audio_scale)))
            if audio_strength > 0.0:
                if self._interrupt:
                    return None
                waveform, waveform_sample_rate =  torch.from_numpy(input_waveform), input_waveform_sample_rate
                if self._interrupt:
                    return None
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)
                elif waveform.ndim == 2:
                    waveform = waveform.unsqueeze(0)
                target_channels = int(getattr(self.audio_encoder, "in_channels", waveform.shape[1]))
                if target_channels <= 0:
                    target_channels = waveform.shape[1]
                if waveform.shape[1] != target_channels:
                    if waveform.shape[1] == 1 and target_channels > 1:
                        waveform = waveform.repeat(1, target_channels, 1)
                    elif target_channels == 1:
                        waveform = waveform.mean(dim=1, keepdim=True)
                    else:
                        waveform = waveform[:, :target_channels, :]
                        if waveform.shape[1] < target_channels:
                            pad_channels = target_channels - waveform.shape[1]
                            pad = torch.zeros(
                                (waveform.shape[0], pad_channels, waveform.shape[2]),
                                dtype=waveform.dtype,
                            )
                            waveform = torch.cat([waveform, pad], dim=1)

                audio_processor = AudioProcessor(
                    sample_rate=self.audio_encoder.sample_rate,
                    mel_bins=self.audio_encoder.mel_bins,
                    mel_hop_length=self.audio_encoder.mel_hop_length,
                    n_fft=self.audio_encoder.n_fft,
                )
                waveform = waveform.to(device="cpu", dtype=torch.float32)
                audio_processor = audio_processor.to(waveform.device)
                mel = audio_processor.waveform_to_mel(waveform, waveform_sample_rate)
                if self._interrupt:
                    return None
                audio_params = next(self.audio_encoder.parameters(), None)
                audio_device = audio_params.device if audio_params is not None else self.device
                audio_dtype = audio_params.dtype if audio_params is not None else self.dtype
                mel = mel.to(device=audio_device, dtype=audio_dtype)
                with torch.inference_mode():
                    audio_latent = self.audio_encoder(mel)
                if self._interrupt:
                    return None
                audio_downsample = getattr(
                    getattr(self.audio_encoder, "patchifier", None),
                    "audio_latent_downsample_factor",
                    4,
                )
                target_shape = AudioLatentShape.from_video_pixel_shape(
                    VideoPixelShape(
                        batch=audio_latent.shape[0],
                        frames=int(frame_num),
                        width=1,
                        height=1,
                        fps=float(fps),
                    ),
                    channels=audio_latent.shape[1],
                    mel_bins=audio_latent.shape[3],
                    sample_rate=self.audio_encoder.sample_rate,
                    hop_length=self.audio_encoder.mel_hop_length,
                    audio_latent_downsample_factor=audio_downsample,
                )
                target_frames = target_shape.frames
                if audio_latent.shape[2] < target_frames:
                    pad_frames = target_frames - audio_latent.shape[2]
                    pad = torch.zeros(
                        (audio_latent.shape[0], audio_latent.shape[1], pad_frames, audio_latent.shape[3]),
                        device=audio_latent.device,
                        dtype=audio_latent.dtype,
                    )
                    audio_latent = torch.cat([audio_latent, pad], dim=2)
                elif audio_latent.shape[2] > target_frames:
                    audio_latent = audio_latent[:, :, :target_frames, :]
                audio_latent = audio_latent.to(device=self.device, dtype=self.dtype)
                audio_conditionings = [AudioConditionByLatent(audio_latent, audio_strength)]

        target_height = int(height)
        target_width = int(width)
        if target_height % 64 != 0:
            target_height = int(math.ceil(target_height / 64) * 64)
        if target_width % 64 != 0:
            target_width = int(math.ceil(target_width / 64) * 64)

        if latent_conditioning_stage2 is not None:
            expected_lat_h = target_height // 32
            expected_lat_w = target_width // 32
            if (
                latent_conditioning_stage2.shape[3] != expected_lat_h
                or latent_conditioning_stage2.shape[4] != expected_lat_w
            ):
                latent_conditioning_stage2 = None
            else:
                latent_conditioning_stage2 = latent_conditioning_stage2.to(device=self.device, dtype=self.dtype)

        if isinstance(self.pipeline, TI2VidTwoStagesPipeline):
            negative_prompt = n_prompt if n_prompt else DEFAULT_NEGATIVE_PROMPT
            pipeline_output = self.pipeline(
                prompt=input_prompt,
                negative_prompt=negative_prompt,
                seed=int(seed),
                height=target_height,
                width=target_width,
                num_frames=int(frame_num),
                frame_rate=float(fps),
                num_inference_steps=int(sampling_steps),
                cfg_guidance_scale=float(guide_scale),
                audio_cfg_guidance_scale=float(guide_scale if audio_cfg_scale is None else audio_cfg_scale),
                cfg_star_switch=cfg_star_switch,
                apg_switch=apg_switch,
                perturbation_switch=perturbation_switch,
                perturbation_layers=perturbation_layers,
                perturbation_start=perturbation_start,
                perturbation_end=perturbation_end,
                alt_guidance_scale=float(alt_guide_scale),
                alt_scale=float(alt_scale),
                images=images,
                guiding_images=guiding_images or None,
                guiding_images_stage2=guiding_images_stage2 or None,
                images_stage2=images_stage2 if stage2_override else None,
                video_conditioning=video_conditioning,
                video_conditioning_downscale_factor=video_conditioning_downscale_factor,
                latent_conditioning_stage2=latent_conditioning_stage2,
                tiling_config=tiling_config,
                enhance_prompt=False,
                audio_conditionings=audio_conditionings,
                callback=callback,
                interrupt_check=interrupt_check,
                loras_slists=loras_slists,
                text_connectors=text_connectors,
                masking_source=masking_source,
                masking_strength=masking_strength,
                return_latent_slice=return_latent_slice,
                self_refiner_setting=self_refiner_setting,
                self_refiner_plan=self_refiner_plan,
                self_refiner_f_uncertainty=self_refiner_f_uncertainty,
                self_refiner_certain_percentage=self_refiner_certain_percentage,
                self_refiner_max_plans=self_refiner_max_plans,
            )
        else:
            pipeline_output = self.pipeline(
                prompt=input_prompt,
                seed=int(seed),
                height=target_height,
                width=target_width,
                num_frames=int(frame_num),
                frame_rate=float(fps),
                images=images,
                guiding_images=guiding_images or None,
                guiding_images_stage2=guiding_images_stage2 or None,
                alt_guidance_scale=float(alt_guide_scale),
                video_conditioning=video_conditioning,
                video_conditioning_downscale_factor=video_conditioning_downscale_factor,
                latent_conditioning_stage2=latent_conditioning_stage2,
                tiling_config=tiling_config,
                enhance_prompt=False,
                audio_conditionings=audio_conditionings,
                callback=callback,
                interrupt_check=interrupt_check,
                loras_slists=loras_slists,
                text_connectors=text_connectors,
                masking_source=masking_source,
                masking_strength=masking_strength,
                return_latent_slice=return_latent_slice,
                self_refiner_setting=self_refiner_setting,
                self_refiner_plan=self_refiner_plan,
                self_refiner_f_uncertainty=self_refiner_f_uncertainty,
                self_refiner_certain_percentage=self_refiner_certain_percentage,
                self_refiner_max_plans=self_refiner_max_plans,
            )

        latent_slice = None
        if isinstance(pipeline_output, tuple) and len(pipeline_output) == 3:
            video, audio, latent_slice = pipeline_output
        else:
            video, audio = pipeline_output

        if video is None or audio is None:
            return None

        if self._interrupt:
            return None
        video_tensor = _collect_video_chunks(video, interrupt_check=interrupt_check)
        if video_tensor is None:
            return None

        video_tensor = video_tensor[:, :frame_num, :height, :width]
        audio_np = audio.detach().float().cpu().numpy() if audio is not None else None
        if audio_np is not None and audio_np.ndim == 2:
            if audio_np.shape[0] in (1, 2) and audio_np.shape[1] > audio_np.shape[0]:
                audio_np = audio_np.T
        output_audio_sampling_rate = int(getattr(self.vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE))
        result = {
            "x": video_tensor,
            "audio": audio_np,
            "audio_sampling_rate": output_audio_sampling_rate,
        }
        if latent_slice is not None:
            result["latent_slice"] = latent_slice
        return result
