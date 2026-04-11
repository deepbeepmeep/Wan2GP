"""Mask video creation utilities."""

from pathlib import Path

import numpy as np
import source.media.video.api as video_api

from source.core.log import headless_logger

__all__ = [
    "create_mask_video_from_inactive_indices",
    "create_simple_first_frame_mask_video",
]


def _api_create_mask_video(
    *,
    total_frames: int,
    resolution_wh: tuple[int, int],
    inactive_frame_indices: set[int],
    output_path: Path | str,
    fps: int,
    task_id_for_logging: str,
):
    h, w = resolution_wh[1], resolution_wh[0]
    frames = [
        np.full((h, w, 3), 0 if idx in inactive_frame_indices else 255, dtype=np.uint8)
        for idx in range(total_frames)
    ]
    return video_api.create_video_from_frames_list(frames, Path(output_path), fps, resolution_wh)


def create_mask_video_from_inactive_indices(
    total_frames: int,
    resolution_wh: tuple[int, int],
    inactive_frame_indices: set[int] | list[int],
    output_path: Path | str,
    fps: int = 16,
    task_id_for_logging: str = "unknown",
    *,
    create_video_from_frames_list_fn=None,
) -> Path | None:
    """
    Create a mask video where:
    - Black frames (0) = inactive/keep original - don't edit these frames
    - White frames (255) = active/generate - model should generate these frames

    Args:
        total_frames: Total number of frames in the video
        resolution_wh: (width, height) tuple for video resolution
        inactive_frame_indices: Set or list of frame indices that should be black (inactive)
        output_path: Where to save the mask video
        fps: Frames per second for the output video
        task_id_for_logging: Task ID for debug logging

    Returns:
        Path to created mask video, or None if creation failed
    """
    try:
        if total_frames <= 0:
            headless_logger.warning(f"Task {task_id_for_logging}: Cannot create mask video with {total_frames} frames", task_id=task_id_for_logging)
            return None

        h, w = resolution_wh[1], resolution_wh[0]  # height, width
        inactive_set = {
            idx
            for idx in (inactive_frame_indices if isinstance(inactive_frame_indices, set) else set(inactive_frame_indices))
            if isinstance(idx, int) and idx >= 0
        }
        normalized_fps = max(int(fps), 1)

        # Create mask frames: 0 (black) for inactive, 255 (white) for active
        mask_frames_buf: list[np.ndarray] = [
            np.full((h, w, 3), 0 if idx in inactive_set else 255, dtype=np.uint8)
            for idx in range(total_frames)
        ]

        if create_video_from_frames_list_fn is None:
            created_mask_video = _api_create_mask_video(
                total_frames=total_frames,
                resolution_wh=resolution_wh,
                inactive_frame_indices=inactive_set,
                output_path=Path(output_path),
                fps=normalized_fps,
                task_id_for_logging=task_id_for_logging,
            )
        else:
            created_mask_video = create_video_from_frames_list_fn(
                mask_frames_buf,
                Path(output_path),
                normalized_fps,
                resolution_wh,
            )
        return created_mask_video

    except (OSError, ValueError, RuntimeError) as e:
        headless_logger.error(f"Task {task_id_for_logging}: Mask video creation failed: {e}", task_id=task_id_for_logging)
        return None

def create_simple_first_frame_mask_video(
    total_frames: int,
    resolution_wh: tuple[int, int],
    output_path: Path | str,
    fps: int = 16,
    task_id_for_logging: str = "unknown") -> Path | None:
    """
    Convenience function to create a mask video where only the first frame is inactive (black).
    This is useful for workflows where you want to keep the first frame unchanged
    and generate the rest.

    Returns:
        Path to created mask video, or None if creation failed
    """
    return create_mask_video_from_inactive_indices(
        total_frames=total_frames,
        resolution_wh=resolution_wh,
        inactive_frame_indices={0},  # Only first frame is inactive
        output_path=output_path,
        fps=fps,
        task_id_for_logging=task_id_for_logging)
