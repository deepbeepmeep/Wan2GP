"""Shared video seam consumed by task handlers and runtime helpers."""

from __future__ import annotations

from source.media.video.crossfade import stitch_videos_with_crossfade
from source.media.video.ffmpeg_ops import (
    create_video_from_frames_list,
    ensure_video_fps,
    extract_frame_range_to_video,
)
from source.media.video.frame_extraction import extract_frames_from_video, extract_last_frame_as_image
from source.media.video.transform_api import add_audio_to_video, standardize_video_aspect_ratio
from source.media.video.vace_frame_utils import (
    create_guide_and_mask_for_generation,
    prepare_vace_generation_params,
)
from source.media.video.video_info import get_video_fps_ffprobe, get_video_frame_count_ffprobe
from source.utils.download_utils import download_video_if_url
from source.utils.frame_utils import save_frame_from_video

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
