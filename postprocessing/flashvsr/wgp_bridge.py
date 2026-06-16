from __future__ import annotations

import os
from typing import Any, Callable

from postprocessing.flashvsr.sparse_backend_config import (
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_CHOICES,
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
    MODE_CHOICES = [
        ("FlashVSR v1.1 Tiny (Slightly Lower Quality, Faster VAE Decoding, Needs Less RAM)", MODE_TINY),
        ("FlashVSR v1.1 Full (Best Quality, Slower VAE Decoding, Needs More RAM)", MODE_FULL),
    ]
    PERSISTENCE_CHOICES = [("Unload after use", PERSIST_UNLOAD), ("Persistent in RAM", PERSIST_RAM)]

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

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"mode": cls.MODE_OFF, "persistence": cls.PERSIST_UNLOAD, "backend": cls.BACKEND_AUTO, "topk_ratio": cls.TOPK_RATIO_DEFAULT}

    @classmethod
    def legacy_config_keys(cls) -> tuple[str, ...]:
        return ("flashvsr_mode", "flashvsr_persistence", "flashvsr_backend", "flashvsr_topk_ratio")

    @classmethod
    def legacy_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": config.get("flashvsr_mode", cls.MODE_OFF),
            "persistence": config.get("flashvsr_persistence", cls.PERSIST_UNLOAD),
            "backend": config.get("flashvsr_backend", cls.BACKEND_AUTO),
            "topk_ratio": config.get("flashvsr_topk_ratio", cls.TOPK_RATIO_DEFAULT),
        }

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        normalized = cls.default_config()
        normalized.update(config or {})
        try:
            normalized["mode"] = int(normalized.get("mode", cls.MODE_OFF))
        except (TypeError, ValueError):
            normalized["mode"] = cls.MODE_OFF
        try:
            normalized["persistence"] = int(normalized.get("persistence", cls.PERSIST_UNLOAD))
        except (TypeError, ValueError):
            normalized["persistence"] = cls.PERSIST_UNLOAD
        if normalized["mode"] not in cls._VARIANTS and normalized["mode"] != cls.MODE_OFF:
            normalized["mode"] = cls.MODE_OFF
        if normalized["persistence"] not in (cls.PERSIST_UNLOAD, cls.PERSIST_RAM):
            normalized["persistence"] = cls.PERSIST_UNLOAD
        normalized["backend"] = cls.normalize_backend(normalized.get("backend", cls.BACKEND_AUTO))
        normalized["topk_ratio"] = cls.normalize_topk_ratio(normalized.get("topk_ratio", cls.TOPK_RATIO_DEFAULT))
        return normalized

    @classmethod
    def apply_pre_1_1_defaults(cls, config: dict[str, Any]) -> bool:
        if config["mode"] == cls.MODE_OFF:
            config["mode"] = cls.MODE_TINY
            return True
        return False

    def config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api

        return upsampler_api.read_config_section(self.server_config if config is None else config, self)

    def normalize_config(self, config: dict[str, Any] | None = None) -> tuple[int, int]:
        from postprocessing import spatial_upsamplers as upsampler_api

        config = self.server_config if config is None else config
        section = upsampler_api.write_config_section(config, self, self.config(config))
        return section["mode"], section["persistence"]

    def settings(self, config: dict[str, Any] | None = None) -> tuple[bool, str | None, int]:
        mode, persistence = self.normalize_config(config)
        return mode != self.MODE_OFF, self._VARIANTS.get(mode), persistence

    def topk_ratio(self) -> float:
        return self.config()["topk_ratio"]

    def backend(self) -> str:
        return self.config()["backend"]

    def enabled(self) -> bool:
        return self.settings()[0]

    def persistent_models(self) -> bool:
        return self.settings()[2] == self.PERSIST_RAM

    @classmethod
    def format_ratio(cls, scale: float) -> str:
        scale = float(scale)
        return str(int(scale)) if scale.is_integer() else f"{scale:g}"

    @classmethod
    def format_ratio_label(cls, scale: float) -> str:
        return cls.format_ratio(scale)

    @classmethod
    def upsampling_value(cls, scale: float) -> str:
        return f"{cls.UPSAMPLING_VALUE_PREFIX}{cls.format_ratio(scale)}"

    @classmethod
    def upsampling_two_pass_value(cls, scale: float) -> str:
        return f"{cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX}{cls.format_ratio(scale)}"

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
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "FlashVSR",
            "upsampler_types": ("postprocessing",),
            "media": ("video", "image"),
            "profile": "video",
            "config_key": "flashvsr",
            "pos": 20,
            "method_pos": {cls.UPSAMPLING_VALUE_PREFIX: 20, cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX: 21},
            "methods": [("FlashVSR", cls.UPSAMPLING_VALUE_PREFIX), ("FlashVSR Two Pass", cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX)],
            "vae_methods": [],
            "multipliers": {cls.UPSAMPLING_VALUE_PREFIX: cls.UPSAMPLING_RATIOS, cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX: cls.UPSAMPLING_RATIOS},
            "default_spatial_upsampling": cls.upsampling_value(2.0),
        }

    @classmethod
    def split_value(cls, value) -> tuple[str, float] | None:
        scale = cls.scale_for_upsampling(value)
        if scale is None:
            return None
        return (cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX if cls.is_two_pass_upsampling(value) else cls.UPSAMPLING_VALUE_PREFIX), scale

    @classmethod
    def build_value(cls, method, scale) -> str | None:
        scale = float(scale or 2.0)
        if scale not in cls.UPSAMPLING_RATIOS:
            scale = 2.0
        if method == cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX:
            return cls.upsampling_two_pass_value(scale)
        if method == cls.UPSAMPLING_VALUE_PREFIX:
            return cls.upsampling_value(scale)
        return None

    def is_upsampling(self, spatial_upsampling) -> bool:
        return self.scale_for_upsampling(spatial_upsampling) is not None

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        if not self.is_upsampling(spatial_upsampling):
            return ""
        if not self.enabled():
            return "FlashVSR Spatial Upsampling is disabled in Configuration > Extensions"
        return ""

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        with gr.Group():
            with gr.Row():
                mode = gr.Dropdown(choices=self.MODE_CHOICES, value=config["mode"], label="FlashVSR Spatial Upsampling (Needs Triton; SpargeAttn optional)", interactive=not lock_config)
                persistence = gr.Dropdown(choices=self.PERSISTENCE_CHOICES, value=config["persistence"], label="FlashVSR Model Persistence", interactive=not lock_config)
            with gr.Row():
                backend = gr.Dropdown(choices=SPARSE_BACKEND_CHOICES, value=config["backend"], label="Backend", interactive=not lock_config)
                topk_ratio = gr.Slider(0.0, self.TOPK_RATIO_MAX, value=config["topk_ratio"], step=0.05, label="FlashVSR Quality / Sparse Top-K Ratio (0 = Auto)", info="Higher keeps more sparse attention candidates and can improve quality at the cost of speed and memory.", interactive=not lock_config)
        return [("mode", mode), ("persistence", persistence), ("backend", backend), ("topk_ratio", topk_ratio)]

    def validate_config_section(self, config: dict[str, Any]):
        if config["mode"] <= 0:
            return ""
        try:
            from postprocessing.flashvsr.attention_backend import sparse_attention_requirement_message
            return sparse_attention_requirement_message(config["backend"]) or ""
        except Exception as exc:
            return f"FlashVSR sparse attention dependency check failed: {type(exc).__name__}: {exc}"

    def config_requires_release(self, old_config: dict[str, Any], new_config: dict[str, Any], changed_keys: set[str]) -> bool:
        return old_config != new_config or bool({"profile", "video_profile", "vae_config"} & changed_keys)

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

    def load_upsampler(self, spatial_upsampling, *, process_files: Callable[..., Any], init_pipe: Callable[..., int], profile, progress_callback=None, **kwargs):
        scale = self.scale_for_upsampling(spatial_upsampling)
        if scale is None:
            raise ValueError(f"Unknown FlashVSR upsampling mode: {spatial_upsampling}")
        enabled, variant, _ = self.settings()
        if not enabled:
            raise RuntimeError("FlashVSR spatial upsampling is disabled in Configuration > Extensions.")
        self.download(process_files)
        from postprocessing.flashvsr.attention_backend import set_sparse_backend
        set_sparse_backend(self.backend())
        from postprocessing.flashvsr.runtime import load_models

        load_models(self.paths(variant), variant=variant, init_pipe=init_pipe, profile=profile, progress_callback=progress_callback)

    def upscale(self, sample, spatial_upsampling, *, vae_config: int, seed=0, continue_cache=None, return_continue_cache=False, still_image=False, abort_callback=None, progress_callback=None, **kwargs):
        scale = self.scale_for_upsampling(spatial_upsampling)
        if scale is None:
            raise ValueError(f"Unknown FlashVSR upsampling mode: {spatial_upsampling}")
        from postprocessing.flashvsr.runtime import upscale_video

        output_height = int(sample.shape[-2] * scale)
        output_width = int(sample.shape[-1] * scale)
        flashvsr_tile_size = self.vae_tile_size(vae_config, output_height, output_width)
        return upscale_video(
            sample,
            scale,
            seed=seed,
            continue_cache=continue_cache,
            return_continue_cache=return_continue_cache,
            vae_tile_size=flashvsr_tile_size,
            topk_ratio=self.topk_ratio(),
            still_image=still_image,
            two_pass=self.is_two_pass_upsampling(spatial_upsampling),
            abort_callback=abort_callback,
            progress_callback=progress_callback,
        )

    def release_vram(self) -> None:
        from postprocessing.flashvsr.runtime import release_models
        release_models()
