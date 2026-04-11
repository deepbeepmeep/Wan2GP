"""
Clip validation and frame calculation for join operations.

Validates that clips have sufficient frames for safe joining
and calculates minimum frame requirements.
"""

from pathlib import Path
from typing import Tuple, List

from source.core.log import task_logger
from source.media.video.video_info import get_video_frame_count_and_fps
from source.utils.download_utils import download_video_if_url

__all__ = [
    "calculate_min_clip_frames",
    "validate_clip_frames_for_join",
]

def calculate_min_clip_frames(gap_frame_count: int, context_frame_count: int, replace_mode: bool) -> int:
    """
    Calculate the minimum number of frames a clip must have to safely join.

    In REPLACE mode, we need:
        gap_frame_count + 2 * context_frame_count <= min_clip_frames

    This ensures that context frames don't overlap with previously blended regions
    in chained joins, avoiding the "double-blending" artifact.

    Args:
        gap_frame_count: Number of frames in the generated gap/transition
        context_frame_count: Number of context frames from each clip boundary
        replace_mode: Whether REPLACE mode is enabled

    Returns:
        Minimum required frames for each clip
    """
    if replace_mode:
        return gap_frame_count + 2 * context_frame_count
    else:
        return 2 * context_frame_count

def validate_clip_frames_for_join(
    clip_list: List[dict],
    gap_frame_count: int,
    context_frame_count: int,
    replace_mode: bool,
    temp_dir: Path,
    orchestrator_task_id: str,
    **_kwargs
) -> Tuple[bool, str, List[int]]:
    """
    Validate that all clips have enough frames for safe joining.

    Args:
        clip_list: List of clip dicts with 'url' keys
        gap_frame_count: Gap frames for transition
        context_frame_count: Context frames from each boundary
        replace_mode: Whether REPLACE mode is enabled
        temp_dir: Directory to download clips for frame counting
        orchestrator_task_id: Task ID for logging

    Returns:
        Tuple of (is_valid, error_message, frame_counts_per_clip)
    """
    min_frames = calculate_min_clip_frames(gap_frame_count, context_frame_count, replace_mode)

    frame_counts = []
    violations = []

    for idx, clip in enumerate(clip_list):
        clip_url = clip.get("url")
        if not clip_url:
            return False, f"Clip {idx} missing 'url' field", []

        # Download clip to count frames
        local_path = download_video_if_url(
            clip_url,
            download_target_dir=temp_dir,
            task_id_for_logging=orchestrator_task_id,
            descriptive_name=f"validate_clip_{idx}"
        )

        if not local_path or not Path(local_path).exists():
            return False, f"Failed to download clip {idx} for validation: {clip_url}", []

        # Get frame count
        frames, fps = get_video_frame_count_and_fps(str(local_path))
        if not frames:
            return False, f"Could not determine frame count for clip {idx}", []

        frame_counts.append(frames)

        # First and last clips only need half the minimum (only one boundary)
        is_first = (idx == 0)
        is_last = (idx == len(clip_list) - 1)

        if is_first or is_last:
            if replace_mode:
                gap_from_side = gap_frame_count // 2 if is_first else (gap_frame_count - gap_frame_count // 2)
                required = context_frame_count + gap_from_side
            else:
                required = context_frame_count
        else:
            required = min_frames

        if frames < required:
            violations.append({
                "idx": idx,
                "frames": frames,
                "required": required,
                "shortfall": required - frames
            })

    if violations:
        min_available = min(frame_counts)
        total_needed = gap_frame_count + 2 * context_frame_count
        ratio = min_available / total_needed
        reduced_gap = max(1, int(gap_frame_count * ratio))
        reduced_context = max(1, int(context_frame_count * ratio))

        warning_parts = []
        for v in violations:
            warning_parts.append(f"Clip {v['idx']}: {v['frames']} frames < {v['required']} required")

        warning_msg = (
            f"[PROPORTIONAL_REDUCTION] Some clips are shorter than ideal:\n  "
            + "\n  ".join(warning_parts)
            + f"\n  Original settings: gap={gap_frame_count}, context={context_frame_count}"
            + f"\n  Will reduce to approximately: gap\u2248{reduced_gap}, context\u2248{reduced_context} ({ratio:.0%} of original)"
            + f"\n  Transitions will be shorter but still generated successfully."
        )
        task_logger.debug_anomaly("VALIDATE_CLIPS", f"{warning_msg}")

        return True, warning_msg, frame_counts

    return True, "", frame_counts
