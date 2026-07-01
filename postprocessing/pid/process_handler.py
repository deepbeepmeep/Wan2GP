from __future__ import annotations

from postprocessing.pid.runtime import PID_FLUX2_POST_UPSAMPLING_METHOD, PID_FLUX2_POST_UPSAMPLING_VALUE, PID_FLUX_POST_UPSAMPLING_METHOD, PID_FLUX_POST_UPSAMPLING_VALUE, is_pid_post_upsampling, pid_backbone_for_upsampling
from postprocessing.pid.wgp_bridge import PiDBridge


class PiDProcessHandler:
    system_handler = "pid"
    model_type = "__system_image_postprocessing"
    model_label = "WanGP System Image Postprocessing"
    target_control_label = "Upscaling"
    default_target_control = PID_FLUX_POST_UPSAMPLING_VALUE
    hide_output_resolution = True
    hide_prompt = True

    def supported_upsampling_ratios(self, process_settings: dict | None = None) -> tuple[float, ...]:
        return tuple(PiDBridge.UPSAMPLING_RATIOS)

    def is_upsampler_process(self, process_settings: dict | None = None) -> bool:
        return True

    def _process_target(self, process_settings: dict | None) -> str:
        settings = process_settings if isinstance(process_settings, dict) else {}
        target = str(settings.get("target_ratio") or settings.get("spatial_upsampling_method") or "").strip().lower()
        return PID_FLUX2_POST_UPSAMPLING_VALUE if target in (PID_FLUX2_POST_UPSAMPLING_METHOD, PID_FLUX2_POST_UPSAMPLING_VALUE) else PID_FLUX_POST_UPSAMPLING_VALUE

    def normalize_target_control(self, value: str | None) -> str:
        value = str(value or "").strip().lower()
        return (PID_FLUX2_POST_UPSAMPLING_VALUE if pid_backbone_for_upsampling(value) == "flux2" else PID_FLUX_POST_UPSAMPLING_VALUE) if is_pid_post_upsampling(value) else self.default_target_control

    def target_control_choices_for_process(self, process_settings: dict) -> list[tuple[str, str]]:
        value = self._process_target(process_settings)
        return [(f"x{_format_ratio_label(scale)}", value) for scale in self.supported_upsampling_ratios(process_settings)]

    def target_control_default_for_process(self, process_settings: dict) -> str:
        return self._process_target(process_settings)

    def normalize_target_control_for_process(self, value: str | None, process_settings: dict) -> str:
        value = str(value or "").strip().lower()
        return (PID_FLUX2_POST_UPSAMPLING_VALUE if pid_backbone_for_upsampling(value) == "flux2" else PID_FLUX_POST_UPSAMPLING_VALUE) if is_pid_post_upsampling(value) else self._process_target(process_settings)

    def output_resolution_token(self, value: str | None) -> str:
        ratios = self.supported_upsampling_ratios()
        return f"x{_format_ratio_label(ratios[0])}" if ratios else "x"

    def build_image_queue_settings(self, process_settings: dict, *, source_path: str, target_control: str, seed: int) -> dict:
        target_control = self.normalize_target_control_for_process(target_control, process_settings)
        api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
        api_options.update({"return_media": True, "suppress_source_audio": True, "suppress_metadata_images": True})
        settings = dict(process_settings)
        settings.update({
            "mode": "edit_postprocessing",
            "model_type": str(settings.get("model_type") or self.model_type),
            "prompt": str(settings.get("prompt") or "PiD upsampling"),
            "image_mode": 1,
            "video_source": source_path,
            "video_length": 1,
            "keep_frames_video_source": "1",
            "temporal_upsampling": "",
            "spatial_upsampling": target_control,
            "film_grain_intensity": 0,
            "film_grain_saturation": 0.5,
            "postprocess_audio": "",
            "repeat_generation": 1,
            "batch_size": 1,
            "seed": int(seed),
            "_api": api_options,
        })
        return settings


def _format_ratio_label(scale: float) -> str:
    value = float(scale)
    return str(int(value)) if value.is_integer() else f"{value:g}"


HANDLER = PiDProcessHandler()
