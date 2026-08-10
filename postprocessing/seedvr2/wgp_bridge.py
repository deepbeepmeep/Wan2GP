from __future__ import annotations

import os
from typing import Any, Callable

from postprocessing.spatial_upsamplers import SimpleScaleSuffixMixin, UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING


class SeedVR2Bridge(SimpleScaleSuffixMixin):
    UPSAMPLING_VALUE_PREFIX = "seedvr2"
    UPSAMPLING_RATIOS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    TRANSFORMER_FILENAME = "SeedVR2_3B_bf16.safetensors"
    VAE_FILENAME = "SeedVR2_VAE_bf16.safetensors"
    POSITIVE_EMBEDDING_FILENAME = "SeedVR2_pos_emb_bf16.safetensors"
    AUTO_VAE_TILE_SIZE = 256
    MAX_VAE_TILE_SIZE = 512
    WINDOW_SIZE_AUTO = 0
    WINDOW_SIZE_UNLIMITED = -1
    WINDOW_SIZE_VALUES = (WINDOW_SIZE_AUTO, 60, 120, 180, 240, 300, 360, WINDOW_SIZE_UNLIMITED)
    WINDOW_SIZE_CHOICES = [("Auto", WINDOW_SIZE_AUTO), ("60 frames", 60), ("120 frames", 120), ("180 frames", 180), ("240 frames", 240), ("300 frames", 300), ("360 frames", 360), ("Unlimited", WINDOW_SIZE_UNLIMITED)]
    MODEL_FRAME_STEP = 4
    WINDOW_OVERLAP_FRAMES = 3

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"window_size": cls.WINDOW_SIZE_AUTO}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        normalized = cls.default_config()
        normalized.update(config or {})
        try:
            window_size = int(normalized["window_size"])
        except (TypeError, ValueError):
            window_size = cls.WINDOW_SIZE_AUTO
        normalized["window_size"] = window_size if window_size in cls.WINDOW_SIZE_VALUES else cls.WINDOW_SIZE_AUTO
        return normalized

    @classmethod
    def resolve_window_size(cls, value, gpu_memory_gb: float | None = None) -> int:
        value = cls.normalize_config_section({"window_size": value})["window_size"]
        if value != cls.WINDOW_SIZE_AUTO:
            return value
        if gpu_memory_gb is None:
            import torch
            gpu_memory_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1_000_000_000 if torch.cuda.is_available() else 0.0
        return 240 if gpu_memory_gb >= 24 else 120 if gpu_memory_gb >= 12 else 60

    @classmethod
    def resolve_model_window_size(cls, value, gpu_memory_gb: float | None = None) -> int:
        window_size = cls.resolve_window_size(value, gpu_memory_gb)
        return window_size if window_size == cls.WINDOW_SIZE_UNLIMITED else ((window_size - 1) // cls.MODEL_FRAME_STEP) * cls.MODEL_FRAME_STEP + 1

    def config(self) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api
        return upsampler_api.read_config_section(self.server_config, self)

    def window_frames(self) -> int:
        return self.resolve_model_window_size(self.config()["window_size"])

    @staticmethod
    def format_ratio(scale: float) -> str:
        scale = float(scale)
        return str(int(scale)) if scale.is_integer() else f"{scale:g}"

    @classmethod
    def upsampling_value(cls, scale: float) -> str:
        return f"{cls.UPSAMPLING_VALUE_PREFIX}{cls.format_ratio(scale)}"

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "SeedVR2",
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video", "image"),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "seedvr2",
            "pos": 30,
            "method_pos": {cls.UPSAMPLING_VALUE_PREFIX: 30},
            "methods": [("SeedVR2", cls.UPSAMPLING_VALUE_PREFIX)],
            "vae_methods": [],
            "multipliers": {cls.UPSAMPLING_VALUE_PREFIX: cls.UPSAMPLING_RATIOS},
            "default_spatial_upsampling": cls.upsampling_value(2.0),
        }

    def enabled(self) -> bool:
        return True

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        window_size = gr.Dropdown(choices=self.WINDOW_SIZE_CHOICES, value=config["window_size"], label="SeedVR2 Window Size", info="Auto uses 240 frames on GPUs with at least 24 GB, 120 with at least 12 GB, otherwise 60. Finite values are adjusted down to the nearest 4n+1 model window.", interactive=not lock_config)
        return [("window_size", window_size)]

    def config_requires_release(self, old_config: dict[str, Any], new_config: dict[str, Any], changed_keys: set[str]) -> bool:
        return bool({"video_profile", "vae_config"} & changed_keys)

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        if split is None or split[0] != self.UPSAMPLING_VALUE_PREFIX or split[1] not in self.UPSAMPLING_RATIOS:
            return "SeedVR2 only supports x1.0 through x4.0 in 0.5 increments"
        return ""

    def query_download_def(self, enabled_only: bool = True) -> dict[str, Any]:
        return {
            "repoId": "DeepBeepMeep/MiniMax-H3",
            "sourceFolderList": ["SeedVR2"],
            "fileList": [[self.TRANSFORMER_FILENAME, self.VAE_FILENAME, self.POSITIVE_EMBEDDING_FILENAME]],
        }

    def _locate(self, filename: str) -> str:
        return self.files_locator.locate_file(os.path.join("SeedVR2", filename))

    def paths(self):
        from .runtime import SeedVR2Paths
        return SeedVR2Paths(transformer=self._locate(self.TRANSFORMER_FILENAME), vae=self._locate(self.VAE_FILENAME), positive_embedding=self._locate(self.POSITIVE_EMBEDDING_FILENAME))

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        filenames = (self.TRANSFORMER_FILENAME, self.VAE_FILENAME, self.POSITIVE_EMBEDDING_FILENAME)
        if all(self.files_locator.locate_file(os.path.join("SeedVR2", filename), error_if_none=False) is not None for filename in filenames):
            return False
        from shared.utils.download import send_download_status
        send_download_status(send_cmd, status_text)
        process_files(**self.query_download_def())
        return True

    def vae_tile_size(self, vae_config: int, output_height: int, output_width: int) -> int:
        import torch
        from models.wan.modules.vae import WanVAE
        device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576 if torch.cuda.is_available() else 0
        mixed_precision = self.server_config.get("vae_precision", "16") == "32"
        return WanVAE.get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=output_height, output_width=output_width)

    def load_upsampler(self, spatial_upsampling, *, process_files: Callable[..., Any], init_pipe: Callable[..., int], profile, progress_callback=None, **kwargs):
        if self.validate_upsampling(spatial_upsampling, 0):
            raise ValueError(f"Unknown SeedVR2 upsampling mode: {spatial_upsampling}")
        self.download(process_files)
        from .runtime import load_models
        load_models(self.paths(), init_pipe=init_pipe, profile=profile, progress_callback=progress_callback)

    def upscale(self, sample, spatial_upsampling, *, vae_config: int, vae_tile_size=None, seed=0, still_image=False, abort_callback=None, progress_callback=None, **kwargs):
        split = self.split_value(spatial_upsampling)
        if split is None:
            raise ValueError(f"Unknown SeedVR2 upsampling mode: {spatial_upsampling}")
        scale = split[1]
        output_height, output_width = int(sample.shape[-2] * scale), int(sample.shape[-1] * scale)
        vae_tile_size = int(self.vae_tile_size(vae_config, output_height, output_width) or self.AUTO_VAE_TILE_SIZE)
        vae_tile_size = min(vae_tile_size, self.AUTO_VAE_TILE_SIZE if int(vae_config) == 0 else self.MAX_VAE_TILE_SIZE)
        from .runtime import upscale_video
        if still_image:
            sample = sample.expand(-1, 5, -1, -1)
        output, cache = upscale_video(sample, scale, seed=seed, vae_tile_size=vae_tile_size, window_size=self.window_frames(), window_overlap=self.WINDOW_OVERLAP_FRAMES, abort_callback=abort_callback, progress_callback=progress_callback)
        if still_image and output is not None:
            output = output[:, 2:3]
        return output, cache

    def release_vram(self) -> None:
        from .runtime import release_models
        release_models()
