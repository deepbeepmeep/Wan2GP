from __future__ import annotations

from typing import Any

from shared.utils.virtual_media import build_virtual_media_path

class LTXVideoUpsamplerProcessHandler:
    system_handler = "ltx2_upsampler"
    model_type = "__system_flashvsr"
    model_label = "WanGP System Video Postprocessing"
    target_control_label = "Upsampling"
    default_target_control = "ltx232"
    default_chunk_size_seconds = 3.0
    frame_step = 1
    minimum_requested_frames = 1
    crossfade_overlap_outputs = True
    hide_chunk_size = True
    hide_sliding_window_overlap = True
    hide_output_resolution = True
    hide_prompt = False

    @staticmethod
    def config() -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api
        return upsampler_api.config_for_method("ltx23")

    @property
    def overlap_frames(self) -> int:
        return self.config()["window_overlap"]

    def get_overlap_frames(self, chunk_frames: int) -> int:
        return max(0, min(self.overlap_frames, int(chunk_frames) - 1))

    def get_chunk_frames(self, selected_frame_count: int) -> int:
        return min(self.config()["window_size"], int(selected_frame_count))

    @staticmethod
    def normalize_target_control(value: str | None) -> str:
        return str(value or "") if str(value or "") in {"ltx232", "ltx252"} else "ltx232"

    def target_control_choices_for_process(self, process_settings: dict) -> list[tuple[str, str]]:
        target = self.normalize_target_control(process_settings["target_ratio"])
        return [("x2", target)]

    def target_control_default_for_process(self, process_settings: dict) -> str:
        return self.normalize_target_control(process_settings["target_ratio"])

    def normalize_target_control_for_process(self, value: str | None, process_settings: dict) -> str:
        return self.normalize_target_control(process_settings["target_ratio"])

    def output_resolution_token(self, value: str | None) -> str:
        target = self.normalize_target_control(value)
        return f"{target[:5]}-x2"

    def build_queue_settings(self, process_settings: dict, *, source_path: str, start_frame: int, frame_count: int, target_control: str, seed: int, continue_cache: Any, audio_track_no: int | None = None) -> dict:
        target_control = self.normalize_target_control_for_process(target_control, process_settings)
        video_path = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=start_frame + frame_count - 1, audio_track_no=audio_track_no)
        api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
        api_options.update({"return_media": True, "suppress_source_audio": False, "suppress_metadata_images": True, "upsampler_frame_offset": int(start_frame)})
        settings = dict(process_settings)
        settings.update({
            "mode": "edit_postprocessing",
            "model_type": self.model_type,
            "image_mode": 0,
            "video_source": video_path,
            "video_length": int(frame_count),
            "keep_frames_video_source": str(int(frame_count)),
            "temporal_upsampling": "",
            "spatial_upsampling": target_control,
            "film_grain_intensity": 0,
            "film_grain_saturation": 0.5,
            "postprocess_audio": "",
            "repeat_generation": 1,
            "batch_size": 1,
            "seed": int(settings.get("seed", 0)),
            "_api": api_options,
        })
        return settings

    @staticmethod
    def supports_continue_cache() -> bool:
        return False

    @staticmethod
    def supports_continue_cache_for_target(value: str | None) -> bool:
        return False


HANDLER = LTXVideoUpsamplerProcessHandler()
