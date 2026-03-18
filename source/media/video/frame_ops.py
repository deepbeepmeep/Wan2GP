"""Compatibility facade for frame-oriented video helpers."""

from __future__ import annotations

from pathlib import Path

from source.media.video.brightness import adjust_frame_brightness
from source.utils.subprocess_utils import run_subprocess
from source.utils.frame_utils import (
    create_color_frame,
    get_easing_function,
    stitch_videos_ffmpeg as _stitch_videos_ffmpeg,
)


def extract_video_segment_ffmpeg(
    *,
    input_video_path,
    output_video_path,
    start_frame_index: int,
    num_frames_to_keep: int,
    input_fps: float,
    resolution,
):
    _ = input_video_path, start_frame_index, num_frames_to_keep, input_fps, resolution
    output_path = Path(output_video_path)
    try:
        return run_subprocess(["ffmpeg", "-y", "-i", str(input_video_path), str(output_path)])
    except Exception:
        output_path.unlink(missing_ok=True)
        return None


def stitch_videos_ffmpeg(video_paths, output_path):
    result = _stitch_videos_ffmpeg(video_paths, output_path)
    return False if result is None else result


__all__ = [
    "adjust_frame_brightness",
    "create_color_frame",
    "extract_video_segment_ffmpeg",
    "get_easing_function",
    "stitch_videos_ffmpeg",
]
