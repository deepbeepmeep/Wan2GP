"""
VACE Frame Utilities - Shared logic for frame-based VACE generation tasks

This module provides shared functionality for tasks that use VACE to generate
video frames with guide and mask videos. Used by:
- join_clips: Bridge two video clips with smooth transition
- inpaint_frames: Regenerate a range of frames within a single video

Key Features:
- Guide video creation with context frames + gray gap
- Mask video creation (black=preserve, white=generate)
- Consistent VACE parameter handling
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import numpy as np

from source.core.log import generation_logger
from source.utils.mask_utils import create_mask_video_from_inactive_indices
from source.utils.frame_utils import create_color_frame
from .ffmpeg_ops import create_video_from_frames_list


@dataclass(frozen=True)
class GuideMaskArtifacts:
    guide_video_path: Path
    mask_video_path: Path
    total_frames: int

    def __iter__(self):
        yield self.guide_video_path
        yield self.mask_video_path
        yield self.total_frames


def create_guide_and_mask_for_generation(
    context_frames_before: List[np.ndarray],
    context_frames_after: List[np.ndarray],
    gap_frame_count: int,
    resolution_wh: Tuple[int, int],
    fps: int,
    output_dir: Path,
    task_id: str,
    filename_prefix: str = "vace_gen",
    regenerate_anchors: bool = False,
    num_anchor_frames: int = 3,
    replace_mode: bool = False,
    gap_inserted_frames: dict = None,
    total_frames: int = None,
) -> Tuple[Path, Path]:
    """
    Create guide and mask videos for VACE generation.

    This shared function is used by both join_clips and inpaint_frames to create
    the guide video (context + gray gap + context) and mask video (black=keep, white=generate).

    Args:
        context_frames_before: Frames to preserve before the gap (numpy arrays)
        context_frames_after: Frames to preserve after the gap (numpy arrays)
        gap_frame_count: Number of frames to generate in the gap (INSERT mode) or replace (REPLACE mode)
        resolution_wh: (width, height) tuple for video resolution
        fps: Target frames per second
        output_dir: Directory to save guide and mask videos
        task_id: Task ID for logging
        filename_prefix: Prefix for output filenames (default: "vace_gen")
        regenerate_anchors: If True, exclude anchor frames from guide and regenerate them
        num_anchor_frames: Number of anchor frames to regenerate on each side (default: 3)
        replace_mode: If True, gap frames REPLACE boundary frames instead of being inserted (default: False)
        gap_inserted_frames: Optional dict mapping {relative_gap_index: image_array} to insert and preserve in the gap

    Returns:
        Tuple of (guide_video_path, mask_video_path, total_frame_count)

    Raises:
        ValueError: If context frames are empty or resolution is invalid
        RuntimeError: If video creation fails
    """
    # Validate inputs
    if not context_frames_before and not context_frames_after:
        raise ValueError("At least one context frame set must be provided")

    if gap_frame_count < 0:
        raise ValueError(f"gap_frame_count must be non-negative, got {gap_frame_count}")

    if resolution_wh[0] <= 0 or resolution_wh[1] <= 0:
        raise ValueError(f"Invalid resolution: {resolution_wh}")

    # === GUIDE/MASK CREATION DIAGNOSTICS ===
    generation_logger.debug(f"[VACE_GUIDE_MASK] Task {task_id}: === Guide/Mask Creation Input ===")
    generation_logger.debug(f"[VACE_GUIDE_MASK]   context_frames_before: {len(context_frames_before)} frames")
    generation_logger.debug(f"[VACE_GUIDE_MASK]   context_frames_after: {len(context_frames_after)} frames")
    generation_logger.debug(f"[VACE_GUIDE_MASK]   gap_frame_count: {gap_frame_count}")
    generation_logger.debug(f"[VACE_GUIDE_MASK]   replace_mode: {replace_mode}")
    generation_logger.debug(f"[VACE_GUIDE_MASK]   regenerate_anchors: {regenerate_anchors}, num_anchor_frames: {num_anchor_frames}")
    generation_logger.debug(f"[VACE_GUIDE_MASK]   resolution_wh: {resolution_wh}, fps: {fps}")
    if gap_inserted_frames:
        generation_logger.debug(f"[VACE_GUIDE_MASK]   gap_inserted_frames at relative indices: {list(gap_inserted_frames.keys())}")

    # Calculate total frames accounting for regenerate_anchors and replace_mode
    num_context_before = len(context_frames_before)
    num_context_after = len(context_frames_after)

    # Both REPLACE and INSERT modes now work similarly for guide/mask creation:
    # - Keep all context frames as preserved (black mask)
    # - Add gap_frame_count grey frames in the middle (white mask = generate)
    # The difference is in how the final video is stitched (replace removes original frames)

    if replace_mode:
        # REPLACE MODE: context + gap + context
        # All context frames are preserved, gap frames are generated
        num_anchor_frames_before = 0
        num_anchor_frames_after = 0

        if total_frames is None:
            total_frames = num_context_before + gap_frame_count + num_context_after

        generation_logger.debug(f"[VACE_UTILS] Task {task_id}: Creating guide and mask videos (REPLACE MODE)")
        generation_logger.debug(f"[VACE_UTILS]   Context before: {num_context_before} frames (preserved)")
        generation_logger.debug(f"[VACE_UTILS]   Gap: {gap_frame_count} frames (to generate)")
        generation_logger.debug(f"[VACE_UTILS]   Context after: {num_context_after} frames (preserved)")
        generation_logger.debug(f"[VACE_UTILS]   Total: {total_frames} frames")
    else:
        # INSERT MODE (original behavior)
        # If regenerate_anchors, we'll exclude N anchor frames from each side
        # and add them as gray placeholders to be generated
        num_anchor_frames_before = 0
        num_anchor_frames_after = 0
        if regenerate_anchors:
            if num_context_before > 0:
                num_anchor_frames_before = min(num_anchor_frames, num_context_before)
            if num_context_after > 0:
                num_anchor_frames_after = min(num_anchor_frames, num_context_after)

        total_frames = num_context_before + gap_frame_count + num_context_after

        generation_logger.debug(f"[VACE_UTILS] Task {task_id}: Creating guide and mask videos (INSERT MODE)")
        generation_logger.debug(f"[VACE_UTILS]   Context before: {num_context_before} frames")
        generation_logger.debug(f"[VACE_UTILS]   Gap: {gap_frame_count} frames")
        generation_logger.debug(f"[VACE_UTILS]   Context after: {num_context_after} frames")
        generation_logger.debug(f"[VACE_UTILS]   Total: {total_frames} frames")
        if regenerate_anchors:
            generation_logger.debug(f"[VACE_UTILS]   Regenerate anchors: {num_anchor_frames_before} frames at end of before context, {num_anchor_frames_after} frames at start of after context")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filenames
    timestamp_short = datetime.now().strftime("%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    guide_filename = f"{filename_prefix}_guide_{timestamp_short}_{unique_suffix}.mp4"
    mask_filename = f"{filename_prefix}_mask_{timestamp_short}_{unique_suffix}.mp4"

    guide_video_path = output_dir / guide_filename
    mask_video_path = output_dir / mask_filename

    # --- 1. Build Guide Video ---
    generation_logger.debug(f"[VACE_UTILS] Task {task_id}: Building guide video...")

    guide_frames = []
    gray_frame = create_color_frame(resolution_wh, (128, 128, 128))

    # Track indices of inserted frames (absolute index in guide_frames)
    inserted_frame_indices = []

    # Normalize gap_inserted_frames to empty dict if None
    gap_inserted_frames = gap_inserted_frames or {}

    if replace_mode:
        # REPLACE MODE: Build guide with context + gap + context
        # All context frames preserved, gap frames generated

        # Add all context frames from before
        guide_frames.extend(context_frames_before)
        generation_logger.debug(f"[VACE_UTILS]   Added {num_context_before} context frames (before, preserved)")

        # Add gray placeholders for the gap (or inserted frames)
        for i in range(gap_frame_count):
            if i in gap_inserted_frames:
                inserted_frame_indices.append(len(guide_frames))
                guide_frames.append(gap_inserted_frames[i])
            else:
                guide_frames.append(gray_frame.copy())

        if inserted_frame_indices:
            generation_logger.debug(f"[VACE_UTILS]   Inserted {len(inserted_frame_indices)} frames into gap at relative indices: {list(gap_inserted_frames.keys())}")
        generation_logger.debug(f"[VACE_UTILS]   Added {gap_frame_count} grey frames for gap (to generate)")

        # Add all context frames from after
        guide_frames.extend(context_frames_after)
        generation_logger.debug(f"[VACE_UTILS]   Added {num_context_after} context frames (after, preserved)")

    else:
        # INSERT MODE (original behavior)
        # Add context frames before gap
        if regenerate_anchors and num_anchor_frames_before > 0:
            # Exclude the last N anchor frames and add them as gray placeholders later
            num_preserved_before = num_context_before - num_anchor_frames_before
            guide_frames.extend(context_frames_before[:num_preserved_before])
            generation_logger.debug(f"[VACE_UTILS]   Added {num_preserved_before} context frames (before, excluding {num_anchor_frames_before} anchors)")
        else:
            guide_frames.extend(context_frames_before)
            generation_logger.debug(f"[VACE_UTILS]   Added {len(context_frames_before)} context frames (before)")

        # Add gray placeholders for regenerated anchors (last N frames of before context)
        if regenerate_anchors and num_anchor_frames_before > 0:
            for _ in range(num_anchor_frames_before):
                guide_frames.append(gray_frame.copy())
            generation_logger.debug(f"[VACE_UTILS]   Added {num_anchor_frames_before} gray placeholders for regenerated anchors (end of before context)")

        # Add gray placeholder frames for the gap (or inserted frames)
        for i in range(gap_frame_count):
            if i in gap_inserted_frames:
                inserted_frame_indices.append(len(guide_frames))
                guide_frames.append(gap_inserted_frames[i])
            else:
                guide_frames.append(gray_frame.copy())

        if inserted_frame_indices:
            generation_logger.debug(f"[VACE_UTILS]   Inserted {len(inserted_frame_indices)} frames into gap at relative indices: {list(gap_inserted_frames.keys())}")
        generation_logger.debug(f"[VACE_UTILS]   Added {gap_frame_count} frames for gap")

        # Add gray placeholders for regenerated anchors (first N frames of after context)
        if regenerate_anchors and num_anchor_frames_after > 0:
            for _ in range(num_anchor_frames_after):
                guide_frames.append(gray_frame.copy())
            generation_logger.debug(f"[VACE_UTILS]   Added {num_anchor_frames_after} gray placeholders for regenerated anchors (start of after context)")

        # Add context frames after gap
        if regenerate_anchors and num_anchor_frames_after > 0:
            # Exclude the first N anchor frames since we added them as gray placeholders above
            guide_frames.extend(context_frames_after[num_anchor_frames_after:])
            generation_logger.debug(f"[VACE_UTILS]   Added {len(context_frames_after) - num_anchor_frames_after} context frames (after, excluding {num_anchor_frames_after} anchors)")
        else:
            guide_frames.extend(context_frames_after)
            generation_logger.debug(f"[VACE_UTILS]   Added {len(context_frames_after)} context frames (after)")

    # Determine final total frame count before writing videos/masks
    guide_frame_count = len(guide_frames)
    if guide_frame_count <= 0:
        raise ValueError("Guide video cannot be empty")

    if total_frames is None:
        total_frames = guide_frame_count
    elif total_frames != guide_frame_count:
        generation_logger.debug(f"[VACE_UTILS] Task {task_id}: total_frames override ({total_frames}) "
               f"does not match constructed guide ({guide_frame_count}). Using guide frame count.")
        total_frames = guide_frame_count

    generation_logger.debug(f"[VACE_UTILS]   Final guide frame count: {guide_frame_count}")

    # Create guide video
    try:
        created_guide_video = create_video_from_frames_list(
            guide_frames,
            guide_video_path,
            fps,
            resolution_wh
        )
        generation_logger.debug(f"[VACE_UTILS] Task {task_id}: Guide video created: {created_guide_video}")
    except (OSError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"Failed to create guide video: {e}") from e

    # --- 2. Build Mask Video ---
    generation_logger.debug(f"[VACE_UTILS] Task {task_id}: Building mask video...")

    # Determine inactive (black) frame indices
    # Inactive = context frames we want to preserve
    # Active (white) = gap frames + anchor frames (if regenerate_anchors) we want to generate
    inactive_indices = set()

    if replace_mode:
        # REPLACE MODE: Mark context frames as inactive (black/preserve), gap frames as active (white/generate)

        # BLACK: first num_context_before frames (preserved context)
        for i in range(num_context_before):
            inactive_indices.add(i)
        generation_logger.debug(f"[VACE_UTILS]   Marked {num_context_before} frames as inactive (preserved from before context)")

        # WHITE: next gap_frame_count frames (the gap to generate)
        # These are automatically active (not added to inactive_indices)

        # BLACK: last num_context_after frames (preserved context)
        start_of_after_context = num_context_before + gap_frame_count
        for i in range(start_of_after_context, total_frames):
            inactive_indices.add(i)
        generation_logger.debug(f"[VACE_UTILS]   Marked {num_context_after} frames as inactive (preserved from after context)")

    else:
        # INSERT MODE (original behavior)
        # Mark context frames before gap as inactive
        if regenerate_anchors and num_anchor_frames_before > 0:
            # Exclude the last N anchor frames - they should be white/generate
            num_inactive_before = num_context_before - num_anchor_frames_before
            for i in range(num_inactive_before):
                inactive_indices.add(i)
            generation_logger.debug(f"[VACE_UTILS]   Marked {num_inactive_before} frames as inactive (before context, excluding {num_anchor_frames_before} anchors)")
        else:
            for i in range(num_context_before):
                inactive_indices.add(i)
            generation_logger.debug(f"[VACE_UTILS]   Marked {num_context_before} frames as inactive (before context)")

        # Mark context frames after gap as inactive
        start_of_after_context = num_context_before + gap_frame_count
        if regenerate_anchors and num_anchor_frames_after > 0:
            # Exclude the first N anchor frames - they should be white/generate
            # The anchors start at index start_of_after_context, so start marking inactive from start_of_after_context + num_anchor_frames_after
            for i in range(start_of_after_context + num_anchor_frames_after, total_frames):
                inactive_indices.add(i)
            num_inactive_after = total_frames - (start_of_after_context + num_anchor_frames_after)
            generation_logger.debug(f"[VACE_UTILS]   Marked {num_inactive_after} frames as inactive (after context, excluding {num_anchor_frames_after} anchors)")
        else:
            for i in range(start_of_after_context, total_frames):
                inactive_indices.add(i)
            generation_logger.debug(f"[VACE_UTILS]   Marked {total_frames - start_of_after_context} frames as inactive (after context)")

    # Active frames are everything not in inactive_indices
    active_indices = [i for i in range(total_frames) if i not in inactive_indices]

    # If frames were inserted into the gap, ensure they are inactive (BLACK/KEEP)
    if inserted_frame_indices:
        for idx in inserted_frame_indices:
            inactive_indices.add(idx)
            if idx in active_indices:
                active_indices.remove(idx)
        generation_logger.debug(f"[VACE_UTILS]   Marked {len(inserted_frame_indices)} inserted frames as inactive (black/keep) at indices: {inserted_frame_indices}")

    generation_logger.debug(f"[VACE_UTILS]   Inactive frame indices (black/keep): {sorted(inactive_indices)}")
    generation_logger.debug(f"[VACE_UTILS]   Active frame indices (white/generate): {active_indices}")

    # === FINAL STRUCTURE SUMMARY ===
    num_inactive = len(inactive_indices)
    num_active = len(active_indices)
    generation_logger.debug(f"[VACE_GUIDE_MASK] Task {task_id}: === Final Guide/Mask Structure ===")
    generation_logger.debug(f"[VACE_GUIDE_MASK]   Total frames in guide: {total_frames}")
    generation_logger.debug(f"[VACE_GUIDE_MASK]   Preserved (black mask): {num_inactive} frames")
    generation_logger.debug(f"[VACE_GUIDE_MASK]   Generated (white mask): {num_active} frames")
    # Check 4N+1 constraint
    is_valid_4n1 = (total_frames - 1) % 4 == 0
    generation_logger.debug(f"[VACE_GUIDE_MASK]   Valid 4N+1: {is_valid_4n1} ({total_frames} = 4*{(total_frames-1)//4}+1)")
    if not is_valid_4n1:
        nearest_valid = ((total_frames - 1) // 4) * 4 + 1
        generation_logger.warning(f"[VACE_GUIDE_MASK]   VACE may quantize to {nearest_valid} frames!")

    # Create mask video
    try:
        created_mask_video = create_mask_video_from_inactive_indices(
            total_frames=total_frames,
            resolution_wh=resolution_wh,
            inactive_frame_indices=inactive_indices,
            output_path=mask_video_path,
            fps=fps,
            task_id_for_logging=task_id,
        )

        if not created_mask_video:
            raise RuntimeError("Mask video creation failed")

        generation_logger.debug(f"[VACE_UTILS] Task {task_id}: Mask video created: {created_mask_video}")

    except (OSError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"Failed to create mask video: {e}") from e

    return created_guide_video, created_mask_video, total_frames


def validate_frame_range(
    total_frame_count: int,
    start_frame: int,
    end_frame: int,
    context_frame_count: int,
    task_id: str = "unknown",
) -> Tuple[bool, str]:
    """
    Validate that a frame range has sufficient context frames on both sides.

    Used by inpaint_frames to ensure the requested range can be processed.

    Args:
        total_frame_count: Total frames in the source video
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (exclusive)
        context_frame_count: Required context frames on each side
        task_id: Task ID for logging

    Returns:
        Tuple of (is_valid: bool, error_message: str or None)
    """
    generation_logger.debug(f"[VACE_UTILS] Task {task_id}: Validating frame range")
    generation_logger.debug(f"[VACE_UTILS]   Total frames: {total_frame_count}")
    generation_logger.debug(f"[VACE_UTILS]   Range: [{start_frame}, {end_frame})")
    generation_logger.debug(f"[VACE_UTILS]   Context required: {context_frame_count} frames on each side")

    # Check if range is valid
    if start_frame < 0:
        return False, f"start_frame ({start_frame}) must be non-negative"

    if end_frame > total_frame_count:
        return False, f"end_frame ({end_frame}) exceeds total frame count ({total_frame_count})"

    if start_frame >= end_frame:
        return False, f"start_frame ({start_frame}) must be less than end_frame ({end_frame})"

    # Check if there's enough context before
    if start_frame < context_frame_count:
        return False, f"Need {context_frame_count} context frames before start_frame ({start_frame}), but only {start_frame} available"

    # Check if there's enough context after
    frames_after = total_frame_count - end_frame
    if frames_after < context_frame_count:
        return False, f"Need {context_frame_count} context frames after end_frame ({end_frame}), but only {frames_after} available"

    generation_logger.debug(f"[VACE_UTILS] Task {task_id}: Frame range validation passed")
    return True, None


def prepare_vace_generation_params(
    guide_video_path: Path,
    mask_video_path: Path,
    total_frames: int,
    resolution_wh: Tuple[int, int],
    prompt: str,
    negative_prompt: str,
    model: str = "wan_2_2_vace_lightning_baseline_2_2_2",
    seed: int = -1,
    task_params: dict = None
) -> dict:
    """
    Prepare standardized VACE generation parameters.

    Creates a consistent parameter dict for VACE generation with guide and mask.

    Args:
        guide_video_path: Path to guide video
        mask_video_path: Path to mask video
        total_frames: Total number of frames to generate
        resolution_wh: (width, height) tuple
        prompt: Generation prompt
        negative_prompt: Negative prompt
        model: Model name (default: wan_2_2_vace_lightning_baseline_2_2_2)
        seed: Random seed (-1 for random)
        task_params: Additional task-specific parameters to merge

    Returns:
        Dict of generation parameters ready for GenerationTask
    """
    generation_params = {
        "video_guide": str(guide_video_path.resolve()),
        "video_mask": str(mask_video_path.resolve()),
        "video_prompt_type": "VM",  # Video + Mask for VACE
        "video_length": total_frames,
        "resolution": f"{resolution_wh[0]}x{resolution_wh[1]}",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }

    # Merge additional task-specific parameters if provided
    if task_params:
        # Add optional parameters only if explicitly provided
        optional_params = [
            "num_inference_steps", "guidance_scale", "embedded_guidance_scale",
            "flow_shift", "audio_guidance_scale", "cfg_zero_step",
            "guidance2_scale", "guidance3_scale", "guidance_phases",
            "switch_threshold", "switch_threshold2", "model_switch_phase",
            "sample_solver", "additional_loras", "phase_config",
            "latent_noise_mask_strength",  # 0.0 = disabled, 1.0 = full latent noise masking
            "vid2vid_init_video",  # Path to video for vid2vid initialization
            "vid2vid_init_strength"  # 0.0 = keep original, 1.0 = random noise
        ]

        for param in optional_params:
            if param in task_params:
                generation_params[param] = task_params[param]

    return generation_params
