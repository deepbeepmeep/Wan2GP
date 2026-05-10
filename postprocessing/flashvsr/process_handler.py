from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from safetensors.torch import save_file

from postprocessing.flashvsr.wgp_bridge import FlashVSRBridge
from shared.utils.virtual_media import build_virtual_media_path


class FlashVSRProcessHandler:
    system_handler = "flashvsr"
    model_type = "__system_flashvsr"
    model_label = "WanGP System Postprocessing"
    target_control_label = "Upsampling"
    target_control_choices = FlashVSRBridge.upsampling_choices(include_name=False)
    default_target_control = FlashVSRBridge.upsampling_value(2.0)
    default_chunk_size_seconds = 3.0
    frame_step = 1
    minimum_requested_frames = 1
    # FlashVSR's streaming output has an 11-frame tail that must be regenerated with the next source chunk before writing.
    overlap_frames = 11
    hide_sliding_window_overlap = True

    def get_overlap_frames(self, chunk_frames: int) -> int:
        return max(0, min(int(self.overlap_frames), int(chunk_frames) - 1))

    def normalize_target_control(self, value: str | None) -> str:
        values = {choice_value for _label, choice_value in self.target_control_choices}
        value = str(value or "").strip()
        return value if value in values else self.default_target_control

    def output_resolution_token(self, value: str | None) -> str:
        scale = FlashVSRBridge.scale_for_upsampling(self.normalize_target_control(value)) or 2.0
        return f"x{FlashVSRBridge.format_ratio(scale)}"

    def build_queue_settings(self, process_settings: dict, *, source_path: str, start_frame: int, frame_count: int, target_control: str, seed: int, continue_cache: Any, audio_track_no: int | None = None) -> dict:
        video_path = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=start_frame + frame_count - 1, audio_track_no=audio_track_no)
        api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
        api_options.update({"return_media": True, "return_flashvsr_continue_cache": True, "flashvsr_continue_cache": continue_cache, "suppress_source_audio": False, "suppress_metadata_images": True})
        settings = dict(process_settings)
        settings.update({
            "mode": "edit_postprocessing",
            "model_type": self.model_type,
            "prompt": str(settings.get("prompt") or "FlashVSR upsampling"),
            "image_mode": 0,
            "video_source": video_path,
            "video_length": int(frame_count),
            "keep_frames_video_source": str(int(frame_count)),
            "temporal_upsampling": "",
            "spatial_upsampling": self.normalize_target_control(target_control),
            "film_grain_intensity": 0,
            "film_grain_saturation": 0.5,
            "MMAudio_setting": 0,
            "repeat_generation": 1,
            "batch_size": 1,
            "seed": int(seed),
            "_api": api_options,
        })
        return settings

    def supports_continue_cache(self) -> bool:
        return True

    def cache_sidecar_path(self, output_filename: str) -> str:
        output_path = Path(output_filename).resolve()
        return str(output_path.with_suffix(output_path.suffix + ".flashvsr_cache.safetensors"))

    def can_resume_without_output_metadata(self, output_filename: str) -> bool:
        return Path(self.cache_sidecar_path(output_filename)).is_file()

    def move_continue_cache(self, source_output_filename: str, target_output_filename: str) -> bool:
        source_path = Path(self.cache_sidecar_path(source_output_filename))
        if not source_path.is_file():
            return False
        target_path = Path(self.cache_sidecar_path(target_output_filename))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target_path)
        return True

    def delete_continue_cache(self, output_filename: str) -> None:
        cache_path = Path(self.cache_sidecar_path(output_filename))
        if cache_path.is_file():
            cache_path.unlink()

    def save_continue_cache(self, cache: Any, output_filename: str, metadata: dict | None = None) -> str:
        if not isinstance(cache, dict):
            return ""
        tail = cache.get("tail_frames")
        if not torch.is_tensor(tail) or tail.ndim != 4 or int(tail.shape[1]) <= 0:
            return ""
        if tail.dtype != torch.uint8:
            tail = tail.detach().cpu().float().clamp(-1.0, 1.0).add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
        cache_metadata = {
            "version": "1",
            "handler": self.system_handler,
            "scale": str(cache.get("scale", "")),
            "variant": str(cache.get("variant", "")),
            "metadata": json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True),
        }
        sidecar_path = self.cache_sidecar_path(output_filename)
        Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
        save_file({"tail_frames": tail.contiguous()}, sidecar_path, metadata=cache_metadata)
        return sidecar_path

    def load_continue_cache(self, output_filename: str) -> Any:
        sidecar_path = self.cache_sidecar_path(output_filename)
        if not Path(sidecar_path).is_file():
            raise gr.Error(f"FlashVSR continuation cache is missing: {sidecar_path}")
        from safetensors import safe_open
        with safe_open(sidecar_path, framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            tail = handle.get_tensor("tail_frames")
            if not torch.is_tensor(tail) or tail.ndim != 4:
                raise gr.Error(f"FlashVSR continuation cache is invalid: {sidecar_path}")
            if tail.dtype != torch.uint8:
                tail = tail.float().clamp_(-1.0, 1.0).contiguous()
        if tail.dtype != torch.uint8:
            tail = tail.float().clamp_(-1.0, 1.0)
        return {"tail_frames": tail, "scale": _coerce_float(metadata.get("scale"), 0.0), "variant": str(metadata.get("variant") or "")}


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


HANDLER = FlashVSRProcessHandler()
