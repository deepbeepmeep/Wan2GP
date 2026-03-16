"""Video preprocessing helpers for join-clip orchestration."""

from __future__ import annotations

from pathlib import Path

from source.media.video.video_info import VideoMetadataError, get_video_frame_count_and_fps


def detect_aspect_ratio_string(width: int, height: int) -> str:
    """Return a friendly aspect-ratio label for common video sizes."""
    if width <= 0 or height <= 0:
        return f"{width}:{height}"

    ratio = width / height
    known = {
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
        "1:1": 1.0,
    }
    for label, value in known.items():
        if abs(ratio - value) < 0.02:
            return label
    return f"{width}:{height}"


def require_video_frame_count_and_fps(
    input_video_path: str | Path,
    *,
    context: str = "video metadata",
) -> tuple[int, float]:
    """Return required frame-count/FPS metadata or raise a structured error."""
    frame_count, fps = get_video_frame_count_and_fps(str(input_video_path))
    if not frame_count or frame_count <= 0 or not fps or fps <= 0:
        raise VideoMetadataError(f"{context}: could not determine frame count")
    return int(frame_count), float(fps)

