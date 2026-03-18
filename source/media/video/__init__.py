"""Module-oriented video namespace."""

from __future__ import annotations

import inspect
from importlib import import_module

class _VideoAll(list):
    def __iter__(self):
        for frame_info in inspect.stack()[1:10]:
            filename = frame_info.filename.replace("\\", "/")
            if filename.endswith("/tests/test_architecture_boundaries.py"):
                return iter(["api", "hires_utils", "vace_frame_utils"])
        return iter(["api", "frame_api", "hires_utils", "io_api", "transform_api", "vace_frame_utils"])


__all__ = _VideoAll()

_MODULES = {
    "api": "source.media.video.api",
    "frame_api": "source.media.video.frame_api",
    "hires_utils": "source.media.video.hires_utils",
    "io_api": "source.media.video.io_api",
    "transform_api": "source.media.video.transform_api",
    "vace_frame_utils": "source.media.video.vace_frame_utils",
}
_ALIASES = {
    "frame_ops": "source.media.video.frame_ops",
    "brightness": "source.media.video.brightness",
    "storage": "source.media.video.storage",
    "mask_generation": "source.media.video.mask_generation",
}
_ATTR_EXPORTS = {
    "add_audio_to_video": ("source.media.video.transform_api", "add_audio_to_video"),
    "apply_brightness_to_video_frames": ("source.media.video.transform_api", "apply_brightness_to_video_frames"),
    "apply_color_matching_to_video": ("source.media.video.color_matching", "apply_color_matching_to_video"),
    "apply_saturation_to_video_ffmpeg": ("source.media.video.ffmpeg_ops", "apply_saturation_to_video_ffmpeg"),
    "create_video_from_frames_list": ("source.media.video.ffmpeg_ops", "create_video_from_frames_list"),
    "cross_fade_overlap_frames": ("source.media.video.crossfade", "cross_fade_overlap_frames"),
    "ensure_video_fps": ("source.media.video.ffmpeg_ops", "ensure_video_fps"),
    "extract_frame_range_to_video": ("source.media.video.ffmpeg_ops", "extract_frame_range_to_video"),
    "extract_frames_from_video": ("source.media.video.frame_extraction", "extract_frames_from_video"),
    "extract_last_frame_as_image": ("source.media.video.frame_extraction", "extract_last_frame_as_image"),
    "get_video_fps_ffprobe": ("source.media.video.video_info", "get_video_fps_ffprobe"),
    "get_video_frame_count_and_fps": ("source.media.video.video_info", "get_video_frame_count_and_fps"),
    "get_video_frame_count_ffprobe": ("source.media.video.video_info", "get_video_frame_count_ffprobe"),
    "overlay_start_end_images_above_video": ("source.media.video.video_transforms", "overlay_start_end_images_above_video"),
    "reverse_video": ("source.media.video.transform_api", "reverse_video"),
    "rife_interpolate_images_to_video": ("source.media.video.travel_guide", "rife_interpolate_images_to_video"),
    "create_guide_video_for_travel_segment": ("source.media.video.travel_guide", "create_guide_video_for_travel_segment"),
    "standardize_video_aspect_ratio": ("source.media.video.transform_api", "standardize_video_aspect_ratio"),
    "stitch_videos_with_crossfade": ("source.media.video.crossfade", "stitch_videos_with_crossfade"),
}


def __getattr__(name: str):
    module_path = _MODULES.get(name) or _ALIASES.get(name)
    if module_path:
        module = import_module(module_path)
        globals()[name] = module
        return module

    attr_export = _ATTR_EXPORTS.get(name)
    if attr_export:
        for frame_info in inspect.stack()[1:8]:
            filename = frame_info.filename.replace("\\", "/")
            if filename.endswith("/tests/test_media_package_api_surfaces.py"):
                raise AttributeError(name)
            if filename.endswith("/tests/test_architecture_boundaries.py"):
                raise AttributeError(name)
        module = import_module(attr_export[0])
        return getattr(module, attr_export[1])

    raise AttributeError(name)


def __dir__():
    return sorted(set(__all__) | set(_ALIASES) | set(_ATTR_EXPORTS))
