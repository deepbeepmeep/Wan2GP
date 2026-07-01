from __future__ import annotations

from postprocessing.chain_of_zoom.wgp_bridge import ChainOfZoomBridge


class ChainOfZoomProcessHandler:
    system_handler = "coz"
    model_type = "__system_image_postprocessing"
    model_label = "WanGP System Image Postprocessing"
    target_control_label = "Upscaling"
    default_target_control = f"{ChainOfZoomBridge.UPSAMPLING_VALUE_PREFIX}4"
    hide_output_resolution = True
    hide_prompt = True

    def supported_upsampling_ratios(self, process_settings: dict | None = None) -> tuple[float, ...]:
        return tuple(ChainOfZoomBridge.MULTIPLIERS)

    def is_upsampler_process(self, process_settings: dict | None = None) -> bool:
        return True

    def _value_for_scale(self, scale: float) -> str:
        return f"{ChainOfZoomBridge.UPSAMPLING_VALUE_PREFIX}{_format_ratio_label(scale)}"

    def normalize_target_control(self, value: str | None) -> str:
        value = str(value or "").strip().lower()
        return value if value in [self._value_for_scale(scale) for scale in self.supported_upsampling_ratios()] else self.default_target_control

    def target_control_choices_for_process(self, process_settings: dict) -> list[tuple[str, str]]:
        return [(f"x{_format_ratio_label(scale)}", self._value_for_scale(scale)) for scale in self.supported_upsampling_ratios(process_settings)]

    def target_control_default_for_process(self, process_settings: dict) -> str:
        return self.normalize_target_control(str(process_settings.get("target_ratio") or process_settings.get("spatial_upsampling_method") or ""))

    def normalize_target_control_for_process(self, value: str | None, process_settings: dict) -> str:
        return self.normalize_target_control(value)

    def output_resolution_token(self, value: str | None) -> str:
        return f"x{_format_ratio_label(self.supported_upsampling_ratios()[0])}"

    def build_image_queue_settings(self, process_settings: dict, *, source_path: str, target_control: str, seed: int) -> dict:
        target_control = self.normalize_target_control_for_process(target_control, process_settings)
        api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
        api_options.update({"return_media": True, "suppress_source_audio": True, "suppress_metadata_images": True})
        settings = dict(process_settings)
        settings.update({
            "mode": "edit_postprocessing",
            "model_type": str(settings.get("model_type") or self.model_type),
            "prompt": str(settings.get("prompt") or "Chain-of-Zoom upsampling"),
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


HANDLER = ChainOfZoomProcessHandler()
