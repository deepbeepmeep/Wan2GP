"""Typed wrapper layer for transform-style video helpers."""

from __future__ import annotations

from pathlib import Path

from source.media.video.brightness import adjust_frame_brightness
from source.media.video.video_transforms import (
    add_audio_to_video as _add_audio_to_video_impl,
    reverse_video as _reverse_video_impl,
    standardize_video_aspect_ratio as _standardize_video_aspect_ratio_impl,
    apply_brightness_to_video_frames as _apply_brightness_to_video_frames_impl,
)
from source.media.video.ffmpeg_ops import ensure_video_fps


class VideoTransformContractError(RuntimeError):
    """Raised when a transform wrapper receives an invalid implementation result."""


def _require_result(name: str, result):
    if result is None:
        raise VideoTransformContractError(f"{name} returned None")
    return result


def add_audio_to_video(**kwargs):
    return _require_result("add_audio_to_video", _add_audio_to_video_impl(**kwargs))


def reverse_video(**kwargs):
    return _require_result("reverse_video", _reverse_video_impl(**kwargs))


def standardize_video_aspect_ratio(**kwargs):
    return _require_result(
        "standardize_video_aspect_ratio",
        _standardize_video_aspect_ratio_impl(**kwargs),
    )


def apply_brightness_to_video_frames(**kwargs):
    return _require_result(
        "apply_brightness_to_video_frames",
        _apply_brightness_to_video_frames_impl(**kwargs),
    )


__all__ = [
    "VideoTransformContractError",
    "add_audio_to_video",
    "adjust_frame_brightness",
    "apply_brightness_to_video_frames",
    "ensure_video_fps",
    "reverse_video",
    "standardize_video_aspect_ratio",
]
