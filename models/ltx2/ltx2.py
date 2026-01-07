import math
import os
import types
from typing import Callable, Iterator

import torch

from shared.utils import files_locator as fl

from .ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from .ltx_pipelines.distilled import DistilledPipeline
from .ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from .ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DEFAULT_NEGATIVE_PROMPT


_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
_SPATIAL_UPSCALER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"


class _AudioVAEWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        per_stats = getattr(decoder, "per_channel_statistics", None)
        if per_stats is not None:
            self.per_channel_statistics = per_stats
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

        transformer = getattr(ltx2_model, "model", None)
        if transformer is not None:
            velocity_model = getattr(transformer, "velocity_model", transformer)
            self.velocity_model = velocity_model
            split_map = getattr(transformer, "split_linear_modules_map", None)
            if split_map is not None:
                self.split_linear_modules_map = split_map

        feature_extractor = getattr(ltx2_model, "text_embedding_projection", None)
        text_connectors = getattr(ltx2_model, "_text_connectors", None) or {}
        if feature_extractor is None:
            feature_extractor = text_connectors.get("feature_extractor_linear")
        if feature_extractor is not None:
            self.text_embedding_projection = feature_extractor

        connectors_model = getattr(ltx2_model, "text_embeddings_connector", None)
        video_connector = None
        audio_connector = None
        if connectors_model is not None:
            video_connector = getattr(connectors_model, "video_embeddings_connector", None)
            audio_connector = getattr(connectors_model, "audio_embeddings_connector", None)
        if video_connector is None:
            video_connector = text_connectors.get("embeddings_connector")
        if audio_connector is None:
            audio_connector = text_connectors.get("audio_embeddings_connector")
        if video_connector is None or audio_connector is None:
            text_encoder = getattr(ltx2_model, "text_encoder", None)
            if text_encoder is not None:
                if video_connector is None:
                    video_connector = getattr(text_encoder, "embeddings_connector", None)
                if audio_connector is None:
                    audio_connector = getattr(text_encoder, "audio_embeddings_connector", None)
        if video_connector is not None:
            self.video_embeddings_connector = video_connector
        if audio_connector is not None:
            self.audio_embeddings_connector = audio_connector

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
    ) -> int:
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

        if use_vae_config <= 1:
            return 0
        if use_vae_config == 2:
            return 512
        if use_vae_config == 3:
            return 256
        return 128


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
        for key, value in sd.items():
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
                continue
            if module_name not in module_names:
                prefixed_name = f"velocity_model.{module_name}"
                if prefixed_name in module_names:
                    module_name = prefixed_name
                else:
                    continue
            new_sd[f"{module_name}{suffix}"] = value
        return new_sd

    transformer.preprocess_loras = types.MethodType(preprocess_loras, transformer)


def _coerce_image_list(image_value):
    if isinstance(image_value, list):
        return image_value[0] if image_value else None
    return image_value


def _to_latent_index(frame_idx: int, stride: int) -> int:
    return int(frame_idx) // int(stride)


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
    spatial_config = None
    if isinstance(tile_size, (tuple, list)):
        if len(tile_size) == 0:
            tile_size = None
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
        tile_frames = _normalize_temporal_tiling_size(int(math.ceil(float(fps) * 5.0)))
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
    frames = frames.to(dtype=torch.float32).div_(127.5).sub_(1.0)
    return frames.permute(3, 0, 1, 2).contiguous()


class LTX2:
    def __init__(
        self,
        model_filename,
        model_type: str,
        base_model_type: str,
        model_def: dict,
        dtype: torch.dtype = torch.bfloat16,
        VAE_dtype: torch.dtype = torch.float32,
        override_text_encoder: str | None = None,
        text_encoder_filepath = None,
    ) -> None:
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.VAE_dtype = VAE_dtype
        self.model_def = model_def
        self._interrupt = False
        self.vae = _LTX2VAEHelper()

        if isinstance(model_filename, (list, tuple)):
            if not model_filename:
                raise ValueError("Missing LTX-2 checkpoint path.")
            checkpoint_path = model_filename[0]
        else:
            checkpoint_path = model_filename

        gemma_root = text_encoder_filepath
        spatial_upsampler_path = fl.locate_file(_SPATIAL_UPSCALER_FILENAME)

        # Keep internal FP8 off by default; mmgp handles quantization transparently.
        fp8transformer = bool(model_def.get("ltx2_internal_fp8", False))
        if fp8transformer:
            fp8transformer = "fp8" in os.path.basename(checkpoint_path).lower()
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")

        if pipeline_kind == "distilled":
            self.pipeline = DistilledPipeline(
                checkpoint_path=checkpoint_path,
                gemma_root=gemma_root,
                spatial_upsampler_path=spatial_upsampler_path,
                loras=[],
                device=self.device,
                fp8transformer=fp8transformer,
                model_device=torch.device("cpu"),
            )
            self._cache_distilled_models()
        else:
            self.pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=checkpoint_path,
                distilled_lora=[],
                spatial_upsampler_path=spatial_upsampler_path,
                gemma_root=gemma_root,
                loras=[],
                device=self.device,
                fp8transformer=fp8transformer,
                model_device=torch.device("cpu"),
            )
            self._cache_two_stage_models()

    def _cache_distilled_models(self) -> None:
        ledger = self.pipeline.model_ledger
        self.text_encoder = ledger.text_encoder()
        self.text_embedding_projection = ledger.text_embedding_projection()
        self.text_embeddings_connector = ledger.text_embeddings_connector()
        self.video_embeddings_connector = self.text_embeddings_connector.video_embeddings_connector
        self.audio_embeddings_connector = self.text_embeddings_connector.audio_embeddings_connector
        self.video_encoder = ledger.video_encoder()
        self.video_decoder = ledger.video_decoder()
        self.audio_decoder = ledger.audio_decoder()
        self.vocoder = ledger.vocoder()
        self.spatial_upsampler = ledger.spatial_upsampler()
        self.model = ledger.transformer()
        self.model2 = None

        ledger.text_encoder = lambda: self.text_encoder
        ledger.text_embedding_projection = lambda: self.text_embedding_projection
        ledger.text_embeddings_connector = lambda: self.text_embeddings_connector
        ledger.video_encoder = lambda: self.video_encoder
        ledger.video_decoder = lambda: self.video_decoder
        ledger.audio_decoder = lambda: self.audio_decoder
        ledger.vocoder = lambda: self.vocoder
        ledger.spatial_upsampler = lambda: self.spatial_upsampler
        ledger.transformer = lambda: self.model
        ledger.release_shared_state()
        self._build_diffuser_model()

    def _cache_two_stage_models(self) -> None:
        ledger_1 = self.pipeline.stage_1_model_ledger
        ledger_2 = self.pipeline.stage_2_model_ledger

        self.text_encoder = ledger_1.text_encoder()
        self.text_embedding_projection = ledger_1.text_embedding_projection()
        self.text_embeddings_connector = ledger_1.text_embeddings_connector()
        self.video_embeddings_connector = self.text_embeddings_connector.video_embeddings_connector
        self.audio_embeddings_connector = self.text_embeddings_connector.audio_embeddings_connector
        self.video_encoder = ledger_1.video_encoder()
        self.video_decoder = ledger_1.video_decoder()
        self.audio_decoder = ledger_1.audio_decoder()
        self.vocoder = ledger_1.vocoder()
        self.spatial_upsampler = ledger_2.spatial_upsampler()
        self.model = ledger_1.transformer()
        self.model2 = None

        ledger_1.text_encoder = lambda: self.text_encoder
        ledger_1.text_embedding_projection = lambda: self.text_embedding_projection
        ledger_1.text_embeddings_connector = lambda: self.text_embeddings_connector
        ledger_1.video_encoder = lambda: self.video_encoder
        ledger_1.video_decoder = lambda: self.video_decoder
        ledger_1.audio_decoder = lambda: self.audio_decoder
        ledger_1.vocoder = lambda: self.vocoder
        ledger_1.transformer = lambda: self.model

        ledger_2.text_encoder = lambda: self.text_encoder
        ledger_2.text_embedding_projection = lambda: self.text_embedding_projection
        ledger_2.text_embeddings_connector = lambda: self.text_embeddings_connector
        ledger_2.video_encoder = lambda: self.video_encoder
        ledger_2.video_decoder = lambda: self.video_decoder
        ledger_2.audio_decoder = lambda: self.audio_decoder
        ledger_2.vocoder = lambda: self.vocoder
        ledger_2.spatial_upsampler = lambda: self.spatial_upsampler
        ledger_2.transformer = lambda: self.model
        ledger_1.release_shared_state()
        if ledger_2 is not ledger_1:
            ledger_2.release_shared_state()
        self._build_diffuser_model()

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

    def generate(
        self,
        input_prompt: str,
        n_prompt: str | None = None,
        image_start=None,
        image_end=None,
        sampling_steps: int = 40,
        guide_scale: float = 4.0,
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

        input_video = kwargs.get("input_video")
        prefix_frames_count = int(kwargs.get("prefix_frames_count") or 0)

        latent_stride = 8
        if hasattr(self.pipeline, "pipeline_components"):
            scale_factors = getattr(self.pipeline.pipeline_components, "video_scale_factors", None)
            if scale_factors is not None:
                latent_stride = int(getattr(scale_factors, "time", scale_factors[0]))

        images = []
        guiding_images = []
        images_stage2 = []
        stage2_override = False
        has_prefix_frames = input_video is not None and torch.is_tensor(input_video) and prefix_frames_count > 0
        is_start_image_only = image_start is not None and (not has_prefix_frames or prefix_frames_count <= 1)

        if isinstance(self.pipeline, TI2VidTwoStagesPipeline):
            if has_prefix_frames and not is_start_image_only:
                frame_count = min(prefix_frames_count, input_video.shape[1])
                for frame_idx in range(0, frame_count, latent_stride):
                    entry = (input_video[:, frame_idx], _to_latent_index(frame_idx, latent_stride), 1.0)
                    images.append(entry)
                    images_stage2.append(entry)

            if image_end is not None:
                entry = (image_end, _to_latent_index(frame_num - 1, latent_stride), 1.0)
                images.append(entry)
                images_stage2.append(entry)

            if image_start is not None:
                entry = (image_start, _to_latent_index(0, latent_stride), 1.0)
                if is_start_image_only:
                    guiding_images.append(entry)
                    images_stage2.append(entry)
                    stage2_override = True
                else:
                    images.append(entry)
                    images_stage2.append(entry)
        else:
            if has_prefix_frames:
                frame_count = min(prefix_frames_count, input_video.shape[1])
                for frame_idx in range(0, frame_count, latent_stride):
                    images.append((input_video[:, frame_idx], _to_latent_index(frame_idx, latent_stride), 1.0))
            if image_start is not None:
                images.append((image_start, _to_latent_index(0, latent_stride), 1.0))
            if image_end is not None:
                images.append((image_end, _to_latent_index(frame_num - 1, latent_stride), 1.0))

        tiling_config = _build_tiling_config(VAE_tile_size, fps)
        interrupt_check = lambda: self._interrupt
        loras_slists = kwargs.get("loras_slists")
        text_connectors = getattr(self, "_text_connectors", None)

        target_height = int(height)
        target_width = int(width)
        if target_height % 64 != 0:
            target_height = int(math.ceil(target_height / 64) * 64)
        if target_width % 64 != 0:
            target_width = int(math.ceil(target_width / 64) * 64)

        if isinstance(self.pipeline, TI2VidTwoStagesPipeline):
            negative_prompt = n_prompt if n_prompt else DEFAULT_NEGATIVE_PROMPT
            video, audio = self.pipeline(
                prompt=input_prompt,
                negative_prompt=negative_prompt,
                seed=int(seed),
                height=target_height,
                width=target_width,
                num_frames=int(frame_num),
                frame_rate=float(fps),
                num_inference_steps=int(sampling_steps),
                cfg_guidance_scale=float(guide_scale),
                images=images,
                guiding_images=guiding_images or None,
                images_stage2=images_stage2 if stage2_override else None,
                tiling_config=tiling_config,
                enhance_prompt=False,
                callback=callback,
                interrupt_check=interrupt_check,
                loras_slists=loras_slists,
                text_connectors=text_connectors,
            )
        else:
            video, audio = self.pipeline(
                prompt=input_prompt,
                seed=int(seed),
                height=target_height,
                width=target_width,
                num_frames=int(frame_num),
                frame_rate=float(fps),
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=False,
                callback=callback,
                interrupt_check=interrupt_check,
                loras_slists=loras_slists,
                text_connectors=text_connectors,
            )

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
        return {
            "x": video_tensor,
            "audio": audio_np,
            "audio_sampling_rate": AUDIO_SAMPLE_RATE,
        }
