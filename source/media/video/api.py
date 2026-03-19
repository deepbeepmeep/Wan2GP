"""Shared video seam consumed by task handlers and runtime helpers."""

from __future__ import annotations

from importlib import import_module
import warnings

from source.media.video.crossfade import stitch_videos_with_crossfade
from source.media.video.ffmpeg_ops import create_video_from_frames_list, extract_frame_range_to_video
from source.media.video.frame_extraction import extract_frames_from_video, extract_last_frame_as_image
from source.media.video.transform_api import standardize_video_aspect_ratio
from source.media.video.vace_frame_utils import (
    create_guide_and_mask_for_generation,
    prepare_vace_generation_params,
)
from source.media.video.video_info import get_video_fps_ffprobe, get_video_frame_count_ffprobe
from source.utils.download_utils import download_video_if_url
from source.utils.frame_utils import save_frame_from_video

_DYNAMIC_EXPORTS = {
    "add_audio_to_video": ("source.media.video.transform_api", "add_audio_to_video"),
    "ensure_video_fps": ("source.media.video.transform_api", "ensure_video_fps"),
}
_transform_facade_warned = False


def _warn_transform_facade_once() -> None:
    global _transform_facade_warned
    if _transform_facade_warned:
        return
    _transform_facade_warned = True
    warnings.warn(
        "source.media.video.api transform helpers are deprecated; use "
        "source.media.video.transform_api instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name: str):
    export = _DYNAMIC_EXPORTS.get(name)
    if export is None:
        raise AttributeError(name)
    _warn_transform_facade_once()
    module = import_module(export[0])
    value = getattr(module, export[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "add_audio_to_video",
    "create_guide_and_mask_for_generation",
    "create_video_from_frames_list",
    "download_video_if_url",
    "ensure_video_fps",
    "extract_frame_range_to_video",
    "extract_frames_from_video",
    "extract_last_frame_as_image",
    "get_video_fps_ffprobe",
    "get_video_frame_count_ffprobe",
    "prepare_vace_generation_params",
    "save_frame_from_video",
    "standardize_video_aspect_ratio",
    "stitch_videos_with_crossfade",
]
