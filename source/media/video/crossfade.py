"""Crossfade and blending functions for video stitching."""
import math
from pathlib import Path

import cv2
import numpy as np

from source.core.log import generation_logger
from source.media.video.frame_extraction import extract_frames_from_video
from source.media.video.ffmpeg_ops import create_video_from_frames_list

__all__ = [
    "crossfade_ease",
    "cross_fade_overlap_frames",
    "stitch_videos_with_crossfade",
]


def crossfade_ease(alpha_lin: float) -> float:
    """Cosine ease-in-out function (maps 0..1 to 0..1).
    Used to determine the blending alpha for crossfades.
    """
    return (1 - math.cos(alpha_lin * math.pi)) / 2.0

def _blend_linear(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return cv2.addWeighted(a, 1.0-t, b, t, 0)

def _blend_linear_sharp(a: np.ndarray, b: np.ndarray, t: float, amt: float) -> np.ndarray:
    base = _blend_linear(a,b,t)
    if amt<=0: return base
    blur = cv2.GaussianBlur(base,(0,0),3)
    return cv2.addWeighted(base, 1.0+amt*t, blur, -amt*t, 0)

def cross_fade_overlap_frames(
    segment1_frames: list[np.ndarray],
    segment2_frames: list[np.ndarray],
    overlap_count: int,
    mode: str = "linear_sharp",
    sharp_amt: float = 0.3
) -> list[np.ndarray]:
    """
    Cross-fades the overlapping frames between two segments using various modes.

    Args:
        segment1_frames: Frames from the first segment (video ending)
        segment2_frames: Frames from the second segment (video starting)
        overlap_count: Number of frames to cross-fade
        mode: Blending mode ("linear", "linear_sharp")
        sharp_amt: Sharpening amount for "linear_sharp" mode (0-1)

    Returns:
        List of cross-faded frames for the overlap region
    """
    if overlap_count <= 0:
        return []

    n = min(overlap_count, len(segment1_frames), len(segment2_frames))
    if n <= 0:
        return []

    # Determine target resolution from segment2 (the newer generated video)
    if not segment2_frames:
        return []

    target_height, target_width = segment2_frames[0].shape[:2]
    target_resolution = (target_width, target_height)

    # Log dimension information for debugging
    seg1_height, seg1_width = segment1_frames[0].shape[:2] if segment1_frames else (0, 0)
    generation_logger.debug(f"CrossFade: Segment1 resolution: {seg1_width}x{seg1_height}, Segment2 resolution: {target_width}x{target_height}")
    generation_logger.debug(f"CrossFade: Target resolution set to: {target_width}x{target_height} (from segment2)")
    generation_logger.debug(f"CrossFade: Processing {n} overlap frames")

    out_frames = []
    for i in range(n):
        t_linear = (i + 1) / float(n)
        alpha = crossfade_ease(t_linear)

        frame_a_np = segment1_frames[-n+i].astype(np.float32)
        frame_b_np = segment2_frames[i].astype(np.float32)

        # Log original shapes before any resizing
        original_a_shape = frame_a_np.shape[:2]
        original_b_shape = frame_b_np.shape[:2]

        # Ensure both frames have the same dimensions
        if frame_a_np.shape[:2] != (target_height, target_width):
            frame_a_np = cv2.resize(frame_a_np, target_resolution, interpolation=cv2.INTER_AREA).astype(np.float32)
            if i == 0:  # Log only for first frame to avoid spam
                generation_logger.debug(f"CrossFade: Resized segment1 frame from {original_a_shape} to {frame_a_np.shape[:2]}")
        if frame_b_np.shape[:2] != (target_height, target_width):
            frame_b_np = cv2.resize(frame_b_np, target_resolution, interpolation=cv2.INTER_AREA).astype(np.float32)
            if i == 0:  # Log only for first frame to avoid spam
                generation_logger.debug(f"CrossFade: Resized segment2 frame from {original_b_shape} to {frame_b_np.shape[:2]}")

        blended_float: np.ndarray
        if mode == "linear_sharp":
            blended_float = _blend_linear_sharp(frame_a_np, frame_b_np, alpha, sharp_amt)
        elif mode == "linear":
            blended_float = _blend_linear(frame_a_np, frame_b_np, alpha)
        else:
            generation_logger.warning(f"Unknown crossfade mode '{mode}'. Defaulting to linear.")
            blended_float = _blend_linear(frame_a_np, frame_b_np, alpha)

        blended_uint8 = np.clip(blended_float, 0, 255).astype(np.uint8)
        out_frames.append(blended_uint8)

    return out_frames

def stitch_videos_with_crossfade(
    video_paths: list[Path | str],
    blend_frame_counts: list[int],
    output_video_path: Path | str,
    fps: float,
    crossfade_mode: str = "linear_sharp",
    crossfade_sharp_amt: float = 0.3,
) -> Path:
    """
    Stitch multiple videos together with crossfade blending at boundaries.

    This is a generalized version of the stitching logic from the travel handlers
    that can be used for any video concatenation with smooth crossfades.

    Args:
        video_paths: List of video file paths to stitch (in order)
        blend_frame_counts: List of blend frame counts between each pair of videos
                           Length must be len(video_paths) - 1
                           blend_frame_counts[0] = blend between video 0 and video 1
        output_video_path: Path for the output stitched video
        fps: Frames per second for output video
        crossfade_mode: Blending mode for crossfade ("linear_sharp" or "linear")
        crossfade_sharp_amt: Sharpening amount for linear_sharp mode (0-1)

    Returns:
        Path to the created output video

    Example:
        # Stitch 3 videos with 3-frame blends between each
        stitch_videos_with_crossfade(
            video_paths=[clip1, transition, clip2],
            blend_frame_counts=[3, 3],  # 3 frames blend between clip1->transition, transition->clip2
            output_video_path=output.mp4,
            fps=16
        )
    """
    if len(video_paths) < 2:
        raise ValueError("Need at least 2 videos to stitch")

    if len(blend_frame_counts) != len(video_paths) - 1:
        raise ValueError(f"blend_frame_counts must have length {len(video_paths) - 1}, got {len(blend_frame_counts)}")

    generation_logger.debug_anomaly("STITCH_VIDEOS", f"Stitching {len(video_paths)} videos with crossfade blending")

    # Extract frames from all videos
    all_video_frames = []
    for i, video_path in enumerate(video_paths):
        frames = extract_frames_from_video(str(video_path))
        if not frames:
            raise ValueError(f"Failed to extract frames from video {i}: {video_path}")
        generation_logger.debug_anomaly("STITCH_VIDEOS", f"Video {i}: {len(frames)} frames extracted")
        all_video_frames.append(frames)

    # Stitch videos together with crossfading
    final_stitched_frames = []
    overlap_frames_for_next_blend = []  # Store overlap frames from previous video for blending

    for i, frames_curr_segment in enumerate(all_video_frames):
        if i == 0:
            # First video: add all frames except those that will be blended with next video
            blend_with_next = blend_frame_counts[0] if i < len(blend_frame_counts) else 0

            if blend_with_next > 0:
                frames_to_add = frames_curr_segment[:-blend_with_next]
                overlap_frames_for_next_blend = frames_curr_segment[-blend_with_next:]
                final_stitched_frames.extend(frames_to_add)
                generation_logger.debug_anomaly("STITCH_VIDEOS", f"Video 0: Added {len(frames_to_add)} frames (keeping {blend_with_next} for blend)")
            else:
                final_stitched_frames.extend(frames_curr_segment)
                generation_logger.debug_anomaly("STITCH_VIDEOS", f"Video 0: Added {len(frames_curr_segment)} frames (no blend)")
        else:
            # Subsequent videos: crossfade with previous, then add remaining
            blend_count = blend_frame_counts[i - 1]

            if blend_count > 0 and overlap_frames_for_next_blend:
                # Use the overlap frames we saved from previous video
                frames_prev_for_fade = overlap_frames_for_next_blend
                generation_logger.debug_anomaly("STITCH_VIDEOS", f"Using {len(frames_prev_for_fade)} overlap frames from previous video for blend")

                # Get frames for crossfade from current segment
                frames_curr_for_fade = frames_curr_segment[:blend_count]

                # Perform crossfade
                faded_frames = cross_fade_overlap_frames(
                    frames_prev_for_fade,
                    frames_curr_for_fade,
                    blend_count,
                    crossfade_mode,
                    crossfade_sharp_amt
                )
                final_stitched_frames.extend(faded_frames)
                generation_logger.debug_anomaly("STITCH_VIDEOS", f"Added {len(faded_frames)} crossfaded frames at boundary {i-1}->{i}")

                # Calculate remaining frames to add from current segment
                # Skip the blended frames and also frames that will be blended with next video (if any)
                blend_with_next = blend_frame_counts[i] if i < len(blend_frame_counts) else 0
                start_idx = blend_count

                if blend_with_next > 0:
                    end_idx = len(frames_curr_segment) - blend_with_next
                    overlap_frames_for_next_blend = frames_curr_segment[-blend_with_next:]
                else:
                    end_idx = len(frames_curr_segment)
                    overlap_frames_for_next_blend = []

                if end_idx > start_idx:
                    frames_to_add = frames_curr_segment[start_idx:end_idx]
                    final_stitched_frames.extend(frames_to_add)
                    generation_logger.debug_anomaly("STITCH_VIDEOS", f"Video {i}: Added {len(frames_to_add)} non-overlapping frames (keeping {blend_with_next} for next blend)")
            else:
                # No blend: just add all frames (minus those for next blend if any)
                blend_with_next = blend_frame_counts[i] if i < len(blend_frame_counts) else 0

                if blend_with_next > 0:
                    frames_to_add = frames_curr_segment[:-blend_with_next]
                    overlap_frames_for_next_blend = frames_curr_segment[-blend_with_next:]
                    final_stitched_frames.extend(frames_to_add)
                    generation_logger.debug_anomaly("STITCH_VIDEOS", f"Video {i}: Added {len(frames_to_add)} frames (no blend, keeping {blend_with_next} for next)")
                else:
                    final_stitched_frames.extend(frames_curr_segment)
                    generation_logger.debug_anomaly("STITCH_VIDEOS", f"Video {i}: Added {len(frames_curr_segment)} frames (no blend)")

    if not final_stitched_frames:
        raise ValueError("No frames produced after stitching")

    generation_logger.debug_anomaly("STITCH_VIDEOS", f"Total stitched frames: {len(final_stitched_frames)}")

    # Create output video
    output_video_path = Path(output_video_path)
    height, width = final_stitched_frames[0].shape[:2]
    resolution_wh = (width, height)

    created_video = create_video_from_frames_list(
        final_stitched_frames,
        output_video_path,
        fps,
        resolution_wh,
    )

    # create_video_from_frames_list raises on failure; these are extra safety checks
    if not created_video.exists():
        raise ValueError(f"Stitched video file does not exist: {created_video}")

    if created_video.stat().st_size == 0:
        raise ValueError(f"Stitched video file is empty: {created_video}")

    generation_logger.debug_anomaly("STITCH_VIDEOS", f"Created stitched video: {created_video} ({created_video.stat().st_size} bytes)")

    return created_video
