from __future__ import annotations

import os
from typing import Any, Callable

from postprocessing.flashvsr.sparse_backend_config import (
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_SPARGE,
    SPARSE_BACKEND_TRITON_SPARSE,
    normalize_sparse_backend,
)


class FlashVSRBridge:
    MODE_OFF = 0
    MODE_TINY = 1
    MODE_FULL = 2
    MODE_TINY_LONG = 3
    PERSIST_UNLOAD = 1
    PERSIST_RAM = 2
    BACKEND_AUTO = SPARSE_BACKEND_AUTO
    BACKEND_TRITON_SPARSE = SPARSE_BACKEND_TRITON_SPARSE
    BACKEND_SPARGE = SPARSE_BACKEND_SPARGE
    TOPK_RATIO_DEFAULT = 0.0
    TOPK_RATIO_MAX = 4.0
    UPSAMPLING_VALUE_PREFIX = "flashvsr"
    UPSAMPLING_TWO_PASS_VALUE_PREFIX = "flashvsr2pass"
    UPSAMPLING_RATIOS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

    TRANSFORMER_FILENAME = "FlashVSR_v1.1_transformer_bf16.safetensors"
    LQ_PROJ_FILENAME = "FlashVSR_v1.1_lq_proj_bf16.safetensors"
    TCDECODER_FILENAME = "FlashVSR_v1.1_tcdecoder_bf16.safetensors"
    POSI_PROMPT_FILENAME = "FlashVSR_v1.1_posi_prompt_bf16.safetensors"
    VAE_FILENAME = "Wan2.1_VAE.safetensors"

    _VARIANTS = {
        MODE_TINY: "tiny",
        MODE_FULL: "full",
        MODE_TINY_LONG: "tiny-long",
    }

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def normalize_topk_ratio(cls, value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = cls.TOPK_RATIO_DEFAULT
        return max(0.0, min(cls.TOPK_RATIO_MAX, value))

    @classmethod
    def normalize_backend(cls, value: Any) -> str:
        return normalize_sparse_backend(value)

    def normalize_config(self, config: dict[str, Any] | None = None) -> tuple[int, int]:
        config = self.server_config if config is None else config
        mode = config.get("flashvsr_mode", self.MODE_OFF)
        persistence = config.get("flashvsr_persistence", self.PERSIST_UNLOAD)
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            mode = self.MODE_OFF
        try:
            persistence = int(persistence)
        except (TypeError, ValueError):
            persistence = self.PERSIST_UNLOAD
        if mode not in self._VARIANTS and mode != self.MODE_OFF:
            mode = self.MODE_OFF
        if persistence not in (self.PERSIST_UNLOAD, self.PERSIST_RAM):
            persistence = self.PERSIST_UNLOAD
        config["flashvsr_mode"] = mode
        config["flashvsr_persistence"] = persistence
        config["flashvsr_backend"] = self.normalize_backend(config.get("flashvsr_backend", self.BACKEND_AUTO))
        config["flashvsr_topk_ratio"] = self.normalize_topk_ratio(config.get("flashvsr_topk_ratio", self.TOPK_RATIO_DEFAULT))
        return mode, persistence

    def settings(self, config: dict[str, Any] | None = None) -> tuple[bool, str | None, int]:
        mode, persistence = self.normalize_config(config)
        return mode != self.MODE_OFF, self._VARIANTS.get(mode), persistence

    def topk_ratio(self) -> float:
        return self.normalize_topk_ratio(self.server_config.get("flashvsr_topk_ratio", self.TOPK_RATIO_DEFAULT))

    def backend(self) -> str:
        return self.normalize_backend(self.server_config.get("flashvsr_backend", self.BACKEND_AUTO))

    def enabled(self) -> bool:
        return self.settings()[0]

    @classmethod
    def format_ratio(cls, scale: float) -> str:
        scale = float(scale)
        return str(int(scale)) if scale.is_integer() else f"{scale:g}"

    @classmethod
    def format_ratio_label(cls, scale: float) -> str:
        return f"{float(scale):.1f}"

    @classmethod
    def upsampling_value(cls, scale: float) -> str:
        return f"{cls.UPSAMPLING_VALUE_PREFIX}{cls.format_ratio(scale)}"

    @classmethod
    def upsampling_two_pass_value(cls, scale: float) -> str:
        return f"{cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX}{cls.format_ratio(scale)}"

    @classmethod
    def upsampling_choices(cls, include_name: bool = True, include_two_pass: bool = False) -> list[tuple[str, str]]:
        prefix = "FlashVSR " if include_name else ""
        choices = [(f"{prefix}x{cls.format_ratio_label(scale)}", cls.upsampling_value(scale)) for scale in cls.UPSAMPLING_RATIOS]
        return choices + ([(f"{prefix}Two Pass x{cls.format_ratio_label(scale)}", cls.upsampling_two_pass_value(scale)) for scale in cls.UPSAMPLING_RATIOS] if include_two_pass else [])

    @classmethod
    def scale_for_upsampling(cls, spatial_upsampling) -> float | None:
        text = str(spatial_upsampling or "").strip().lower()
        prefix = cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX if text.startswith(cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX) else cls.UPSAMPLING_VALUE_PREFIX
        if not text.startswith(prefix):
            return None
        try:
            scale = float(text[len(prefix):])
        except ValueError:
            return None
        return scale if scale in cls.UPSAMPLING_RATIOS else None

    @classmethod
    def is_two_pass_upsampling(cls, spatial_upsampling) -> bool:
        return str(spatial_upsampling or "").strip().lower().startswith(cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX)

    @classmethod
    def query_edit_mode_def(cls, include_name: bool = True) -> dict[str, Any]:
        return {
            "name": "FlashVSR",
            "spatial_upsampling_choices": cls.upsampling_choices(include_name=include_name, include_two_pass=True),
            "default_spatial_upsampling": cls.upsampling_value(2.0),
        }

    def is_upsampling(self, spatial_upsampling) -> bool:
        return self.scale_for_upsampling(spatial_upsampling) is not None

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        if not self.is_upsampling(spatial_upsampling):
            return ""
        if not self.enabled():
            return "FlashVSR Spatial Upsampling is disabled in Configuration > Extensions"
        return ""

    def query_download_def(self, enabled_only: bool = True) -> dict[str, Any] | None:
        if enabled_only and not self.enabled():
            return None
        return {
            "repoId": "DeepBeepMeep/Wan2.1",
            "sourceFolderList": ["FlashVSR", ""],
            "fileList": [[self.TRANSFORMER_FILENAME, self.LQ_PROJ_FILENAME, self.TCDECODER_FILENAME, self.POSI_PROMPT_FILENAME], [self.VAE_FILENAME]],
        }

    def _locate_flashvsr_file(self, filename: str) -> str:
        return self.files_locator.locate_file(os.path.join("FlashVSR", filename))

    def paths(self, variant: str):
        from postprocessing.flashvsr.runtime import FlashVSRPaths
        return FlashVSRPaths(
            transformer=self._locate_flashvsr_file(self.TRANSFORMER_FILENAME),
            lq_proj=self._locate_flashvsr_file(self.LQ_PROJ_FILENAME),
            posi_prompt=self._locate_flashvsr_file(self.POSI_PROMPT_FILENAME),
            tcdecoder=None if variant == "full" else self._locate_flashvsr_file(self.TCDECODER_FILENAME),
            vae=self.files_locator.locate_file(self.VAE_FILENAME) if variant == "full" else None,
        )

    def vae_tile_size(self, vae_config: int, output_height: int | None = None, output_width: int | None = None) -> int:
        import torch
        from models.wan.modules.vae import WanVAE

        device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576 if torch.cuda.is_available() else 0
        mixed_precision = self.server_config.get("vae_precision", "16") == "32"
        return WanVAE.get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=output_height, output_width=output_width)

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        flashvsr_def = self.query_download_def()
        if flashvsr_def is None:
            return False
        _, variant, _ = self.settings()
        required = [os.path.join("FlashVSR", self.TRANSFORMER_FILENAME), os.path.join("FlashVSR", self.LQ_PROJ_FILENAME), os.path.join("FlashVSR", self.POSI_PROMPT_FILENAME)]
        required.append(self.VAE_FILENAME if variant == "full" else os.path.join("FlashVSR", self.TCDECODER_FILENAME))
        if all(self.files_locator.locate_file(path, error_if_none=False) is not None for path in required):
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text)
        process_files(**flashvsr_def)
        return True

    def upscale(self, sample, spatial_upsampling, *, seed=0, continue_cache=None, return_continue_cache=False, vae_tile_size=None, process_files: Callable[..., Any], vae_config: int, init_pipe: Callable[..., int], profile, still_image=False, abort_callback=None, progress_callback=None):
        scale = self.scale_for_upsampling(spatial_upsampling)
        if scale is None:
            raise ValueError(f"Unknown FlashVSR upsampling mode: {spatial_upsampling}")
        enabled, variant, persistence = self.settings()
        if not enabled:
            raise RuntimeError("FlashVSR spatial upsampling is disabled in Configuration > Extensions.")
        self.download(process_files)
        from postprocessing.flashvsr.attention_backend import set_sparse_backend
        set_sparse_backend(self.backend())
        from postprocessing.flashvsr.runtime import upscale_video

        output_height = int(sample.shape[-2] * scale)
        output_width = int(sample.shape[-1] * scale)
        flashvsr_tile_size = self.vae_tile_size(vae_config, output_height, output_width)
        return upscale_video(
            sample,
            scale,
            self.paths(variant),
            variant=variant,
            seed=seed,
            continue_cache=continue_cache,
            return_continue_cache=return_continue_cache,
            persistent_models=persistence == self.PERSIST_RAM,
            vae_tile_size=flashvsr_tile_size,
            topk_ratio=self.topk_ratio(),
            init_pipe=init_pipe,
            profile=profile,
            attention_mode=self.server_config.get("attention_mode"),
            still_image=still_image,
            two_pass=self.is_two_pass_upsampling(spatial_upsampling),
            abort_callback=abort_callback,
            progress_callback=progress_callback,
        )

    def release_vram(self) -> None:
        from postprocessing.flashvsr.runtime import release_models
        release_models()
