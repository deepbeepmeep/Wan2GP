"""
Mask Video Builder for Travel Segments

Handles mask video creation for frame control, including:
- Inactive frame index calculation (overlap, anchors, keyframes)
- Independent segment mode handling (chain_segments=False)
- Mask video creation and verification
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

from source.core.log import travel_logger
from source.media.video.video_info import get_video_frame_count_and_fps
from source.utils.mask_utils import create_mask_video_from_inactive_indices
from source.utils.output_paths import prepare_output_path

def create_mask_video(proc: Any) -> Optional[Path]:
    """
    Create mask video for frame control.

    Returns:
        Path to created mask video, or None if not created/failed
    """
    ctx = proc.ctx

    if not ctx.mask_active_frames:
        travel_logger.debug(f"Task {ctx.task_id}: mask_active_frames disabled, skipping mask video creation", task_id=ctx.task_id)
        return None

    try:
        # Determine which frame indices should be kept (inactive = black)
        inactive_indices = set()

        # Define overlap_count for consistent logging
        frame_overlap_from_previous = ctx.segment_params.get("frame_overlap_from_previous", 0)
        overlap_count = max(0, int(frame_overlap_from_previous))

        # 1) Frames reused from the previous segment (overlap)
        # If chain_segments is False, treat overlap as 0 for masking purposes
        chain_segments = ctx.orchestrator_details.get("chain_segments", True)

        if not chain_segments:
            overlap_count = 0
            travel_logger.debug(f"[INDEPENDENT_SEGMENTS] Seg {ctx.segment_idx}: chain_segments=False: Forcing mask overlap_count=0 (independent mode)", task_id=ctx.task_id)
        elif overlap_count > 0:
            overlap_indices = set(range(overlap_count))
            inactive_indices.update(overlap_indices)
            travel_logger.debug(f"Seg {ctx.segment_idx}: Adding {len(overlap_indices)} overlap frames to inactive set: {sorted(overlap_indices)}", task_id=ctx.task_id)
        else:
            travel_logger.debug(f"Seg {ctx.segment_idx}: No overlap frames to mark as inactive", task_id=ctx.task_id)

        # 2) First frame when this is the very first segment from scratch OR independent segments
        # In independent mode (chain_segments=False), every segment starts from a fixed keyframe image, so frame 0 must be anchored.
        is_first_segment_val = ctx.segment_params.get("is_first_segment", False)
        is_continue_scenario = ctx.orchestrator_details.get("continue_from_video_resolved_path") is not None

        if (is_first_segment_val and not is_continue_scenario) or not chain_segments:
            inactive_indices.add(0)
            travel_logger.debug(f"Seg {ctx.segment_idx}: Marking frame 0 as inactive (anchor start image)", task_id=ctx.task_id)

        # 3) Last frame for multi-image segments - each segment travels TO a target image
        # For single image journeys, we don't anchor the end, let the model generate freely
        is_single_image_journey = proc._detect_single_image_journey()
        if not is_single_image_journey:
            inactive_indices.add(ctx.total_frames_for_segment - 1)
            travel_logger.debug(f"Seg {ctx.segment_idx}: Multi-image journey - marking last frame {ctx.total_frames_for_segment - 1} as inactive (target image)", task_id=ctx.task_id)
        else:
            travel_logger.debug(f"Seg {ctx.segment_idx}: Single image journey - NOT marking last frame as inactive, letting model generate freely", task_id=ctx.task_id)

        # 4) Consolidated keyframe positions (frame consolidation optimization)
        consolidated_keyframe_positions = ctx.segment_params.get("consolidated_keyframe_positions")
        if consolidated_keyframe_positions:
            # Mark all keyframe positions as inactive since they should show exact keyframe images
            for frame_pos in consolidated_keyframe_positions:
                if 0 <= frame_pos < ctx.total_frames_for_segment:
                    inactive_indices.add(frame_pos)
            travel_logger.debug(f"Seg {ctx.segment_idx}: CONSOLIDATED SEGMENT - marking keyframe positions as inactive: {consolidated_keyframe_positions}", task_id=ctx.task_id)

        # --- DEBUG LOGGING (restored from original) ---
        travel_logger.debug(f"[MASK_DEBUG] Segment {ctx.segment_idx}: frame_overlap_from_previous={frame_overlap_from_previous}", task_id=ctx.task_id)
        travel_logger.debug(f"[MASK_DEBUG] Segment {ctx.segment_idx}: inactive (masked) frame indices: {sorted(list(inactive_indices))}", task_id=ctx.task_id)
        travel_logger.debug(f"[MASK_DEBUG] Segment {ctx.segment_idx}: active (unmasked) frame indices: {[i for i in range(ctx.total_frames_for_segment) if i not in inactive_indices]}", task_id=ctx.task_id)
        # --- END DEBUG LOGGING ---

        # Create mask video output path
        timestamp_short = datetime.now().strftime("%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        mask_filename = f"{ctx.task_id}_seg{ctx.segment_idx:02d}_mask_{timestamp_short}_{unique_suffix}.mp4"

        # Use prepare_output_path to ensure mask video goes to task_type directory
        mask_out_path_tmp, _ = prepare_output_path(
            task_id=ctx.task_id,
            filename=mask_filename,
            main_output_dir_base=ctx.main_output_dir_base,
            task_type="travel_segment"
        )

        travel_logger.debug(f"Seg {ctx.segment_idx}: Creating mask video with {len(inactive_indices)} inactive frames: {sorted(inactive_indices)}", task_id=ctx.task_id)

        # Mask video is required for VACE frame anchoring and for overlap-masked continuation.
        if not ctx.debug_enabled and not (proc.is_vace_model or proc.generation_policy.continuation.uses_mask_video):
            travel_logger.debug(f"Task {ctx.task_id}: Debug mode disabled and non-VACE model, skipping mask video creation", task_id=ctx.task_id)
            return None
        else:
            if (proc.is_vace_model or proc.generation_policy.continuation.uses_mask_video) and not ctx.debug_enabled:
                travel_logger.debug(
                    f"Task {ctx.task_id}: Creating mask video for required travel frame anchoring",
                    task_id=ctx.task_id,
                )

            # Use the generalized mask creation function
            created_mask_vid = create_mask_video_from_inactive_indices(
                total_frames=ctx.total_frames_for_segment,
                resolution_wh=ctx.parsed_res_wh,
                inactive_frame_indices=inactive_indices,
                output_path=mask_out_path_tmp,
                fps=ctx.orchestrator_details.get("fps_helpers", 16),
                task_id_for_logging=ctx.task_id)

            if created_mask_vid and created_mask_vid.exists():
                # Verify mask video properties match guide video
                try:
                    mask_frames, mask_fps = get_video_frame_count_and_fps(str(created_mask_vid))
                    travel_logger.debug(f"Seg {ctx.segment_idx}: Mask video generated - {mask_frames} frames @ {mask_fps}fps -> {created_mask_vid}", task_id=ctx.task_id)

                    # Warn if frame count mismatch
                    if mask_frames != ctx.total_frames_for_segment:
                        travel_logger.warning(f"Seg {ctx.segment_idx}: Mask frame count ({mask_frames}) != target ({ctx.total_frames_for_segment})", task_id=ctx.task_id)
                except (OSError, ValueError, RuntimeError) as e_verify:
                    travel_logger.warning(f"Seg {ctx.segment_idx}: Could not verify mask video properties: {e_verify}", task_id=ctx.task_id)

                return created_mask_vid
            else:
                travel_logger.error(f"[MASK_ERROR] Seg {ctx.segment_idx}: Mask video creation failed", task_id=ctx.task_id)
                return None

    except (OSError, ValueError, RuntimeError) as e_mask:
        travel_logger.error(f"[MASK_ERROR] Seg {ctx.segment_idx}: Mask video creation failed: {e_mask}", task_id=ctx.task_id, exc_info=True)
        return None
