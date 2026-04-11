"""
Join clips generation handler - bridge two video clips using VACE generation.

This module provides the main handle_join_clips_task function that:
1. Optionally standardizes both videos to a target aspect ratio (via center-crop)
2. Extracts context frames from the end of the first clip
3. Extracts context frames from the beginning of the second clip
4. Generates transition frames between them using VACE
5. Uses mask video to preserve the context frames and only generate the gap

The staged workflow seam is represented by _JoinWorkflowStageOutputs and
_join_run_workflow_stages(...).
"""

import json
import time
from pathlib import Path
from typing import Tuple
import source.media.video.api as video_api

# Import shared utilities
from ...utils import (
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    get_video_frame_count_and_fps,
    download_video_if_url,
    save_frame_from_video,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
    upload_intermediate_file_to_storage
)
from ...media.video import (
    extract_frames_from_video,
    extract_frame_range_to_video,
    ensure_video_fps,
    standardize_video_aspect_ratio,
    stitch_videos_with_crossfade,
    create_video_from_frames_list,
    get_video_fps_ffprobe,
    add_audio_to_video)
from ... import db_operations as db_ops
from ...core.log import headless_logger, orchestrator_logger

from .vace_quantization import _calculate_vace_quantization

def handle_join_clips_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str,
    task_queue = None) -> Tuple[bool, str]:
    """
    Handle join_clips task: bridge two video clips using VACE generation.

    Args:
        task_params_from_db: Task parameters including:
            - starting_video_path: Path to first video clip
            - ending_video_path: Path to second video clip
            - context_frame_count: Number of frames to extract from each clip
            - gap_frame_count: Number of frames to generate between clips (INSERT mode) or replace (REPLACE mode)
            - replace_mode: Optional bool (default False). If True, gap frames REPLACE boundary frames instead of being inserted
            - prompt: Generation prompt for the transition
            - aspect_ratio: Optional aspect ratio (e.g., "16:9", "9:16", "1:1") to standardize both videos
            - model: Optional model override (defaults to wan_2_2_vace_lightning_baseline_2_2_2)
            - resolution: Optional [width, height] override
            - use_input_video_resolution: Optional bool (default False). If True, uses the detected resolution from input video instead of resolution override
            - fps: Optional FPS override (defaults to 16)
            - use_input_video_fps: Optional bool (default False). If True, uses input video's FPS. If False, downsamples to fps param (default 16)
            - max_wait_time: Optional timeout in seconds for generation (defaults to 1800s / 30 minutes)
            - additional_loras: Optional dict of additional LoRAs {name: weight}
            - Other standard VACE parameters (guidance_scale, flow_shift, etc.)
        main_output_dir_base: Base output directory
        task_id: Task ID for logging and status updates
        task_queue: HeadlessTaskQueue instance for generation

    Returns:
        Tuple of (success: bool, output_path_or_message: str)
    """
    try:
        # --- 1. Extract and Validate Parameters ---
        starting_video_path = task_params_from_db.get("starting_video_path")
        ending_video_path = task_params_from_db.get("ending_video_path")
        context_frame_count = task_params_from_db.get("context_frame_count", 8)
        gap_frame_count = task_params_from_db.get("gap_frame_count", 53)
        replace_mode = task_params_from_db.get("replace_mode", False)  # If True, gap REPLACES frames instead of inserting
        prompt = task_params_from_db.get("prompt", "")
        aspect_ratio = task_params_from_db.get("aspect_ratio")  # Optional: e.g., "16:9", "9:16", "1:1"

        # Extract keep_bridging_images param
        keep_bridging_images = task_params_from_db.get("keep_bridging_images", False)

        # transition_only mode: generate transition video without stitching
        # Used by parallel join architecture where final stitch happens in a separate task
        transition_only = task_params_from_db.get("transition_only", False)

        mode_label = "REPLACE" if replace_mode else "INSERT"

        # Check if this is part of an orchestrator and starting_video_path needs to be fetched
        orchestrator_task_id_ref = task_params_from_db.get("orchestrator_task_id_ref")
        is_first_join = task_params_from_db.get("is_first_join", False)
        is_last_join = task_params_from_db.get("is_last_join", False)
        audio_url = task_params_from_db.get("audio_url")  # Audio to add to final output (only used on last join)

        if not starting_video_path:
            # Check if this is an orchestrator child task that needs to fetch predecessor output
            if orchestrator_task_id_ref:
                if is_first_join:
                    error_msg = "First join in orchestrator must have starting_video_path explicitly set"
                    orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                    return False, error_msg

                # Fetch predecessor output using edge function
                predecessor_id, predecessor_output = db_ops.get_predecessor_output_via_edge_function(task_id)

                if not predecessor_output:
                    error_msg = f"Failed to fetch predecessor output (predecessor_id={predecessor_id})"
                    orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                    return False, error_msg

                starting_video_path = predecessor_output
            else:
                # Standalone join_clips task without starting_video_path
                error_msg = "starting_video_path is required"
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                return False, error_msg

        if not ending_video_path:
            error_msg = "ending_video_path is required"
            orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        # Validate task queue
        if task_queue is None:
            error_msg = "task_queue is required for join_clips"
            orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        # Create working directory - use custom dir if provided by orchestrator
        if "join_output_dir" in task_params_from_db:
            join_clips_dir = Path(task_params_from_db["join_output_dir"])
        else:
            join_clips_dir = main_output_dir_base / "join_clips" / task_id

        join_clips_dir.mkdir(parents=True, exist_ok=True)

        # Download videos if they are URLs (e.g., Supabase storage URLs)
        starting_video_path = download_video_if_url(
            starting_video_path,
            download_target_dir=join_clips_dir,
            task_id_for_logging=task_id,
            descriptive_name="starting_video"
        )
        ending_video_path = download_video_if_url(
            ending_video_path,
            download_target_dir=join_clips_dir,
            task_id_for_logging=task_id,
            descriptive_name="ending_video"
        )

        # Convert to Path objects and validate existence
        starting_video = Path(starting_video_path)
        ending_video = Path(ending_video_path)

        if not starting_video.exists():
            error_msg = f"Starting video not found: {starting_video_path}"
            orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        if not ending_video.exists():
            error_msg = f"Ending video not found: {ending_video_path}"
            orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        # --- 2a. Determine and apply target FPS ---
        # This single FPS value is used for BOTH resampling and all downstream
        # calculations (VACE guide, gap indices, output encoding, metadata).
        use_input_video_fps = task_params_from_db.get("use_input_video_fps", False)
        explicit_fps = task_params_from_db.get("fps")

        if use_input_video_fps:
            # Detect from source clips — use min to avoid upsampling
            start_native_fps = get_video_fps_ffprobe(str(starting_video)) or 16
            end_native_fps = get_video_fps_ffprobe(str(ending_video)) or 16
            resolved_fps = min(start_native_fps, end_native_fps)
            fps_source = f"input_video(min start={start_native_fps}, end={end_native_fps})"
        elif explicit_fps:
            resolved_fps = explicit_fps
            fps_source = "explicit"
        else:
            resolved_fps = 16
            fps_source = "default"

        if not use_input_video_fps:
            starting_video_before = starting_video
            try:
                starting_video = ensure_video_fps(
                    input_video_path=starting_video,
                    target_fps=resolved_fps,
                    output_dir=join_clips_dir)
            except (OSError, ValueError, RuntimeError) as e:
                error_msg = f"Failed to ensure starting video is at {resolved_fps} fps: {e}"
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                return False, error_msg
            ending_video_before = ending_video
            try:
                ending_video = ensure_video_fps(
                    input_video_path=ending_video,
                    target_fps=resolved_fps,
                    output_dir=join_clips_dir)
            except (OSError, ValueError, RuntimeError) as e:
                error_msg = f"Failed to ensure ending video is at {resolved_fps} fps: {e}"
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                return False, error_msg
            fps_resampled = {
                "start": Path(starting_video_before) != Path(starting_video),
                "end": Path(ending_video_before) != Path(ending_video),
            }
        else:
            fps_resampled = {"start": False, "end": False}

        aspect_ratio_decision = "unchanged"
        aspect_ratio_reference = None

        # --- 2b. Standardize Videos to Target Aspect Ratio (if specified) ---
        if aspect_ratio:
            # Standardize starting video
            standardized_start_path = join_clips_dir / f"start_standardized_{task_id}.mp4"
            result = standardize_video_aspect_ratio(
                input_video_path=starting_video,
                output_video_path=standardized_start_path,
                target_aspect_ratio=aspect_ratio,
                task_id_for_logging=task_id)
            if result is None:
                error_msg = f"Failed to standardize starting video to aspect ratio {aspect_ratio}"
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                return False, error_msg
            starting_video = standardized_start_path

            # Standardize ending video
            standardized_end_path = join_clips_dir / f"end_standardized_{task_id}.mp4"
            result = standardize_video_aspect_ratio(
                input_video_path=ending_video,
                output_video_path=standardized_end_path,
                target_aspect_ratio=aspect_ratio,
                task_id_for_logging=task_id)
            if result is None:
                error_msg = f"Failed to standardize ending video to aspect ratio {aspect_ratio}"
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                return False, error_msg
            ending_video = standardized_end_path
            aspect_ratio_decision = "explicit_standardize"
            aspect_ratio_reference = aspect_ratio
        else:
            # Get dimensions of both videos
            try:
                import subprocess

                def get_video_dimensions(video_path):
                    probe_cmd = [
                        'ffprobe', '-v', 'error',
                        '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height',
                        '-of', 'csv=p=0',
                        str(video_path)
                    ]
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode != 0:
                        return None, None
                    width_str, height_str = result.stdout.strip().split(',')
                    return int(width_str), int(height_str)

                start_w, start_h = get_video_dimensions(starting_video)
                end_w, end_h = get_video_dimensions(ending_video)

                if start_w and start_h and end_w and end_h:
                    start_aspect = start_w / start_h
                    end_aspect = end_w / end_h

                    # If aspect ratios differ by more than 1%, standardize ending video to match starting video
                    if abs(start_aspect - end_aspect) > 0.01:
                        # Calculate aspect ratio string from starting video
                        # Use common aspect ratios or create from dimensions
                        if abs(start_aspect - 16/9) < 0.01:
                            auto_aspect_ratio = "16:9"
                        elif abs(start_aspect - 9/16) < 0.01:
                            auto_aspect_ratio = "9:16"
                        elif abs(start_aspect - 1.0) < 0.01:
                            auto_aspect_ratio = "1:1"
                        elif abs(start_aspect - 4/3) < 0.01:
                            auto_aspect_ratio = "4:3"
                        elif abs(start_aspect - 21/9) < 0.01:
                            auto_aspect_ratio = "21:9"
                        else:
                            # Use exact dimensions as ratio
                            auto_aspect_ratio = f"{start_w}:{start_h}"

                        standardized_end_path = join_clips_dir / f"end_standardized_{task_id}.mp4"
                        result = standardize_video_aspect_ratio(
                            input_video_path=ending_video,
                            output_video_path=standardized_end_path,
                            target_aspect_ratio=auto_aspect_ratio,
                            task_id_for_logging=task_id)
                        if result is None:
                            orchestrator_logger.debug_anomaly("JOIN_CLIPS_WARNING", f"Task {task_id}: Failed to auto-standardize ending video, proceeding with original")
                        else:
                            ending_video = standardized_end_path
                            aspect_ratio_decision = "auto_standardize"
                            aspect_ratio_reference = auto_aspect_ratio
                    else:
                        aspect_ratio_decision = "keep_original"
                        aspect_ratio_reference = f"{start_w}x{start_h}"

            except (OSError, ValueError, RuntimeError) as e:
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_WARNING", f"Task {task_id}: Could not check video dimensions: {e}")
                aspect_ratio_decision = "skip_check_error"

        # --- 3. Extract Video Properties ---
        try:
            start_frame_count, start_fps = get_video_frame_count_and_fps(str(starting_video))
            end_frame_count, end_fps = get_video_frame_count_and_fps(str(ending_video))

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Failed to extract video properties: {e}"
            orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        # Validate that frame counts were detected (WebM and some codecs may fail)
        if start_frame_count is None:
            error_msg = f"Could not detect frame count for starting video: {starting_video}. The video may be corrupt, empty, or in an unsupported format."
            orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        if end_frame_count is None:
            error_msg = f"Could not detect frame count for ending video: {ending_video}. The video may be corrupt, empty, or in an unsupported format."
            orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        # Auto-adjust context frame count if it exceeds available frames
        original_context_frame_count = context_frame_count
        max_available_context = min(start_frame_count, end_frame_count)

        if context_frame_count > max_available_context:
            context_frame_count = max_available_context
            orchestrator_logger.debug_anomaly(
                "JOIN_CLIPS",
                (
                    f"context_frame_count reduced from {original_context_frame_count} to {context_frame_count} "
                    f"for available frames start={start_frame_count}, end={end_frame_count}"
                ),
                task_id=task_id,
            )
            headless_logger.warning(
                f"[JOIN_CLIPS] Task {task_id}: context_frame_count reduced from {original_context_frame_count} to {context_frame_count} to fit available frames",
                task_id=task_id
            )

        # Use the single resolved_fps for all downstream operations
        target_fps = resolved_fps

        # --- 4. Calculate gap sizes first (needed for REPLACE mode context extraction) ---
        # Calculate VACE quantization adjustments early so we know exact gap sizes
        quantization_result = _calculate_vace_quantization(
            context_frame_count=context_frame_count,
            gap_frame_count=gap_frame_count,
            replace_mode=replace_mode
        )
        gap_for_guide = quantization_result['gap_for_guide']
        quantization_shift = quantization_result['quantization_shift']
        gap_split_adjustments = []

        # Calculate gap split for REPLACE mode
        # Start with even split, then adjust if one clip is too short
        gap_from_clip1 = gap_for_guide // 2 if replace_mode else 0
        gap_from_clip2 = (gap_for_guide - gap_from_clip1) if replace_mode else 0

        if replace_mode:
            # Dynamically adjust gap split if one clip is too short
            # Each clip needs: gap_from_clip + at least 1 frame for context
            min_frames_needed_clip1 = gap_from_clip1 + 1
            _min_frames_needed_clip2 = gap_from_clip2 + 1

            if start_frame_count < min_frames_needed_clip1:
                # Clip1 too short - shift gap frames to clip2
                max_gap_from_clip1 = max(0, start_frame_count - 1)  # Leave at least 1 frame for context
                shift_amount = gap_from_clip1 - max_gap_from_clip1
                gap_from_clip1 = max_gap_from_clip1
                gap_from_clip2 = gap_from_clip2 + shift_amount
                gap_split_adjustments.append(
                    f"clip1_shift={shift_amount}->{gap_from_clip1}/{gap_from_clip2}"
                )

            if end_frame_count < gap_from_clip2 + 1:
                # Clip2 too short - shift gap frames to clip1
                max_gap_from_clip2 = max(0, end_frame_count - 1)
                shift_amount = gap_from_clip2 - max_gap_from_clip2
                gap_from_clip2 = max_gap_from_clip2
                gap_from_clip1 = gap_from_clip1 + shift_amount
                gap_split_adjustments.append(
                    f"clip2_shift={shift_amount}->{gap_from_clip1}/{gap_from_clip2}"
                )

            # Final validation - check if total gap is still achievable
            total_available = (start_frame_count - 1) + (end_frame_count - 1)  # -1 for minimum 1 context frame each
            if gap_for_guide > total_available:
                error_msg = (
                    f"Videos too short for requested gap: need {gap_for_guide} gap frames but only "
                    f"{total_available} available (start: {start_frame_count}, end: {end_frame_count}). "
                    f"Try reducing gap_frame_count or using longer source clips."
                )
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                return False, error_msg
        orchestrator_logger.debug_block(
            "SETUP",
            {
                "task": task_id,
                "orchestrator_task": orchestrator_task_id_ref,
                "start_video": starting_video,
                "end_video": ending_video,
                "mode": mode_label,
                "transition_only": transition_only,
                "fps": resolved_fps,
                "fps_source": fps_source,
                "fps_resampled": fps_resampled,
                "aspect_ratio": aspect_ratio or "auto",
                "aspect_ratio_decision": aspect_ratio_decision,
                "aspect_ratio_reference": aspect_ratio_reference,
                "clip_properties": {
                    "start": f"{start_frame_count}f@{start_fps}",
                    "end": f"{end_frame_count}f@{end_fps}",
                },
                "context_frame_count": context_frame_count,
                "requested_gap": gap_frame_count,
                "gap_for_guide": gap_for_guide,
                "gap_split": (gap_from_clip1, gap_from_clip2),
                "gap_split_adjustments": gap_split_adjustments or None,
            },
            task_id=task_id,
        )

        # --- 5. Extract Context Frames ---
        try:
            # Extract all frames from both videos
            start_all_frames = extract_frames_from_video(str(starting_video))
            if not start_all_frames or len(start_all_frames) < context_frame_count:
                error_msg = f"Failed to extract {context_frame_count} frames from starting video"
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                return False, error_msg

            end_all_frames = extract_frames_from_video(str(ending_video))
            if not end_all_frames or len(end_all_frames) < context_frame_count:
                error_msg = f"Failed to extract {context_frame_count} frames from ending video"
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                return False, error_msg

            # Initialize vid2vid source video path (will be set in REPLACE mode if enabled)
            vid2vid_source_video_path = None

            if replace_mode:
                # REPLACE mode: Context comes from OUTSIDE the gap region
                # Gap is removed from boundary, context is adjacent to (but outside) the gap
                #
                # clip1: [...][context 8][gap N removed]
                # clip2: [gap M removed][context 8][...]
                #
                clip1_available = len(start_all_frames)
                clip2_available = len(end_all_frames)
                min_clip_frames = min(clip1_available, clip2_available)

                # Calculate total frames needed: gap + context on each side
                total_needed = gap_frame_count + 2 * context_frame_count

                # --- PROPORTIONAL REDUCTION if clips are too short ---
                if min_clip_frames < total_needed:
                    # Calculate reduction ratio
                    # Leave at least 5 frames for minimum VACE generation (context + gap + context >= 5)
                    usable_frames = max(5, min_clip_frames - 2)  # Reserve 2 frames for safety margin
                    ratio = usable_frames / total_needed

                    # Apply proportional reduction to both gap and context
                    adjusted_gap = max(1, int(gap_frame_count * ratio))
                    adjusted_context = max(1, int(context_frame_count * ratio))

                    # Ensure we have at least 5 total frames for VACE (minimum 4n+1 = 5)
                    adjusted_total = adjusted_gap + 2 * adjusted_context
                    if adjusted_total < 5:
                        # Force minimum viable settings
                        adjusted_gap = 1
                        adjusted_context = 2
                        adjusted_total = 5

                    # Update the working values
                    gap_frame_count = adjusted_gap
                    context_frame_count = adjusted_context

                    # Recalculate gap splits with new gap value
                    gap_from_clip1 = gap_frame_count // 2
                    gap_from_clip2 = gap_frame_count - gap_from_clip1

                    # Recalculate VACE quantization with new values
                    quantization_result = _calculate_vace_quantization(
                        context_frame_count=context_frame_count,
                        gap_frame_count=gap_frame_count,
                        replace_mode=replace_mode
                    )
                    gap_for_guide = quantization_result['gap_for_guide']
                    quantization_shift = quantization_result['quantization_shift']
                    orchestrator_logger.debug_anomaly(
                        "JOIN_CLIPS",
                        (
                            f"proportional reduction applied ratio={ratio:.2%}, gap={gap_frame_count}, "
                            f"context={context_frame_count}, total={adjusted_total}, "
                            f"gap_splits=({gap_from_clip1},{gap_from_clip2})"
                        ),
                        task_id=task_id,
                    )

                # Now calculate available context with (potentially adjusted) gap values
                clip1_max_context = clip1_available - gap_from_clip1
                clip2_max_context = clip2_available - gap_from_clip2

                # Validate clips have enough frames (should pass after proportional reduction)
                if clip1_max_context < 1:
                    error_msg = f"Starting video too short: need at least {gap_from_clip1 + 1} frames, have {clip1_available}"
                    orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                    return False, error_msg

                if clip2_max_context < 1:
                    error_msg = f"Ending video too short: need at least {gap_from_clip2 + 1} frames, have {clip2_available}"
                    orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                    return False, error_msg

                # Final context adjustment (may further reduce if clips are asymmetric)
                context_from_clip1 = min(context_frame_count, clip1_max_context)
                context_from_clip2 = min(context_frame_count, clip2_max_context)

                needs_asymmetric_context = context_from_clip1 < context_frame_count or context_from_clip2 < context_frame_count
                if needs_asymmetric_context:
                    # Recalculate quantization with actual asymmetric context counts
                    quantization_result = _calculate_vace_quantization(
                        context_frame_count=context_frame_count,
                        gap_frame_count=gap_frame_count,
                        replace_mode=replace_mode,
                        context_before=context_from_clip1,
                        context_after=context_from_clip2
                    )
                    gap_for_guide = quantization_result['gap_for_guide']
                    quantization_shift = quantization_result['quantization_shift']
                    orchestrator_logger.debug_anomaly(
                        "JOIN_CLIPS",
                        (
                            f"asymmetric context clip1={context_from_clip1}/{clip1_max_context}, "
                            f"clip2={context_from_clip2}/{clip2_max_context}, gap_for_guide={gap_for_guide}"
                        ),
                        task_id=task_id,
                    )

                # Context from clip1: frames BEFORE the gap (not the last frames)
                # If removing last N frames, context is the N frames before that
                context_start_idx = len(start_all_frames) - gap_from_clip1 - context_from_clip1
                context_end_idx = len(start_all_frames) - gap_from_clip1
                start_context_frames = start_all_frames[context_start_idx:context_end_idx]

                # Context from clip2: frames AFTER the gap (not the first frames)
                # If removing first M frames, context is the N frames after that
                end_context_frames = end_all_frames[gap_from_clip2:gap_from_clip2 + context_from_clip2]

                # --- VID2VID SOURCE: Extract gap frames for vid2vid initialization ---
                # If vid2vid_init_strength is set, create a source video from the gap frames
                vid2vid_init_strength = task_params_from_db.get("vid2vid_init_strength")
                if vid2vid_init_strength is not None and vid2vid_init_strength < 1.0:
                    # Extract gap frames that are being replaced
                    # Gap from clip1: last N frames of starting video
                    gap_frames_clip1 = start_all_frames[context_end_idx:]  # frames from context_end_idx to end
                    # Gap from clip2: first M frames of ending video
                    gap_frames_clip2 = end_all_frames[:gap_from_clip2]  # frames 0 to gap_from_clip2-1

                    # Build vid2vid source video with same structure as guide: context + gap + context
                    vid2vid_frames = []
                    vid2vid_frames.extend(start_context_frames)  # Context before
                    vid2vid_frames.extend(gap_frames_clip1)      # Gap from clip1
                    vid2vid_frames.extend(gap_frames_clip2)      # Gap from clip2
                    vid2vid_frames.extend(end_context_frames)    # Context after

                    # Get resolution from first context frame
                    first_frame = start_context_frames[0]
                    vid2vid_res_wh = (first_frame.shape[1], first_frame.shape[0])  # (width, height)

                    # Create vid2vid source video file
                    vid2vid_source_video_path = join_clips_dir / f"vid2vid_source_{task_id}.mp4"
                    try:
                        create_video_from_frames_list(
                            vid2vid_frames,
                            vid2vid_source_video_path,
                            target_fps,
                            vid2vid_res_wh
                        )
                    except (OSError, ValueError, RuntimeError) as v2v_err:
                        orchestrator_logger.debug_anomaly("JOIN_CLIPS_WARNING", f"Task {task_id}: Error creating vid2vid source video: {v2v_err}")
                        vid2vid_source_video_path = None
            else:
                # INSERT mode: Context is at the boundary (last/first frames)
                # No frames are removed, we're just inserting new frames between clips
                start_context_frames = start_all_frames[-context_frame_count:]
                end_context_frames = end_all_frames[:context_frame_count]

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Failed to extract context frames: {e}"
            orchestrator_logger.error(error_msg, task_id=task_id, exc_info=True)
            return False, error_msg

        # Log expected guide structure
        expected_guide_frames = len(start_context_frames) + gap_for_guide + len(end_context_frames)

        # Get resolution from first frame or task params
        first_frame = start_context_frames[0]
        frame_height, frame_width = first_frame.shape[:2]
        detected_res_wh = (frame_width, frame_height)

        # Determine resolution: check use_input_video_resolution flag first, then explicit resolution param
        use_input_video_resolution = task_params_from_db.get("use_input_video_resolution", False)
        resolution_override = task_params_from_db.get("resolution")  # May be None or a list

        if use_input_video_resolution:
            parsed_res_wh = detected_res_wh
            resolution_source = "input_video"
        elif resolution_override is not None:
            # resolution_override should be a list [width, height]
            parsed_res_wh = (resolution_override[0], resolution_override[1])
            resolution_source = "override"
        else:
            parsed_res_wh = detected_res_wh
            resolution_source = "detected"
        # --- 6. Build Guide and Mask Videos (using shared helper) ---
        # (quantization already calculated above in step 4)
        quantized_total_frames = quantization_result['total_frames']

        actual_ctx_before = len(start_context_frames)
        actual_ctx_after = len(end_context_frames)
        orchestrator_logger.debug_block(
            "SETUP",
            {
                "context_summary": {
                    "start_total": len(start_all_frames),
                    "end_total": len(end_all_frames),
                    "start_context": actual_ctx_before,
                    "end_context": actual_ctx_after,
                    "expected_guide_frames": expected_guide_frames,
                },
                "resolution": parsed_res_wh,
                "resolution_source": resolution_source,
                "detected_resolution": detected_res_wh,
                "quantized_gap": gap_for_guide,
                "quantized_total_frames": quantized_total_frames,
                "quantization_shift": quantization_shift,
            },
            task_id=task_id,
        )

        # Determine inserted frames for gap preservation (if enabled)
        # Both modes now use the same logic: insert boundary frames at 1/3 and 2/3 of gap
        gap_inserted_frames = {}

        if keep_bridging_images:
            if len(start_context_frames) > 0 and len(end_context_frames) > 0:
                # Anchor 1: End of first video (last frame of start context)
                anchor1 = start_context_frames[-1]
                idx1 = gap_for_guide // 3

                # Anchor 2: Start of second video (first frame of end context)
                anchor2 = end_context_frames[0]
                idx2 = (gap_for_guide * 2) // 3

                # Only insert if gap is large enough to separate them
                if gap_for_guide >= 3 and idx1 < idx2:
                    gap_inserted_frames[idx1] = anchor1
                    gap_inserted_frames[idx2] = anchor2
                elif gap_for_guide < 3 or idx1 >= idx2:
                    orchestrator_logger.debug_anomaly(
                        "JOIN_CLIPS",
                        f"keep_bridging_images requested but gap {gap_for_guide} is too small for anchors",
                        task_id=task_id,
                    )
            else:
                orchestrator_logger.debug_anomaly("JOIN_CLIPS_WARNING", f"Task {task_id}: keep_bridging_images=True but contexts empty")

        # Create guide/mask with adjusted gap
        try:
            created_guide_video, created_mask_video, guide_frame_count = video_api.create_guide_and_mask_for_generation(
                context_frames_before=start_context_frames,
                context_frames_after=end_context_frames,
                gap_frame_count=gap_for_guide,  # Use quantization-adjusted gap
                resolution_wh=parsed_res_wh,
                fps=target_fps,
                output_dir=join_clips_dir,
                task_id=task_id,
                filename_prefix="join",
                replace_mode=replace_mode,
                gap_inserted_frames=gap_inserted_frames)
        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Failed to create guide/mask videos: {e}"
            orchestrator_logger.error(error_msg, task_id=task_id, exc_info=True)
            return False, error_msg

        total_frames = guide_frame_count
        is_valid_4n1 = (total_frames - 1) % 4 == 0

        # Determine model (default to Lightning baseline for fast generation)
        model = task_params_from_db.get("model", "wan_2_2_vace_lightning_baseline_2_2_2")

        # Ensure prompt is valid
        prompt = ensure_valid_prompt(prompt)
        negative_prompt = ensure_valid_negative_prompt(
            task_params_from_db.get("negative_prompt", "")
        )

        additional_loras = task_params_from_db.get("additional_loras", {})
        phase_config = task_params_from_db.get("phase_config")

        # Use shared helper to prepare standardized VACE parameters
        generation_params = video_api.prepare_vace_generation_params(
            guide_video_path=created_guide_video,
            mask_video_path=created_mask_video,
            total_frames=total_frames,
            resolution_wh=parsed_res_wh,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            seed=task_params_from_db.get("seed", -1),
            task_params=task_params_from_db  # Pass through for optional param merging (includes additional_loras)
        )

        # Add vid2vid source video if we created one in replace mode
        if vid2vid_source_video_path is not None:
            generation_params["vid2vid_init_video"] = str(vid2vid_source_video_path.resolve())
            # vid2vid_init_strength should already be in generation_params from task_params

        # --- 7. Submit to Generation Queue ---
        is_valid_4n1 = (total_frames - 1) % 4 == 0
        orchestrator_logger.debug_block(
            "SETUP",
            {
                "model": model,
                "video_length": total_frames,
                "valid_4n1": is_valid_4n1,
                "resolution": parsed_res_wh,
                "guide": created_guide_video.name,
                "mask": created_mask_video.name,
                "additional_loras": len(additional_loras or {}),
                "phase_config": bool(phase_config),
                "vid2vid": vid2vid_source_video_path is not None,
            },
            task_id=task_id,
        )

        try:
            # Import GenerationTask from correct location
            from source.task_handlers.queue.task_queue import GenerationTask

            generation_task = GenerationTask(
                id=task_id,
                model=model,
                prompt=prompt,
                parameters=generation_params,
                priority=task_params_from_db.get("priority", 0)
            )

            # Submit task using correct method
            submitted_task_id = task_queue.submit_task(generation_task)

            # Wait for completion using polling pattern (same as direct queue tasks)
            # Allow timeout override via task params (default: 30 minutes to handle slow model loading)
            max_wait_time = task_params_from_db.get("max_wait_time", 1800)  # 30 minute default timeout
            wait_interval = 2  # Check every 2 seconds
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                status = task_queue.get_task_status(task_id)

                if status is None:
                    error_msg = "Task status became None during processing"
                    orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
                    return False, error_msg

                if status.status == "completed":
                    transition_video_path = status.result_path
                    processing_time = status.processing_time or 0

                    # IMPORTANT: Check actual frame count vs expected
                    # VACE may generate fewer frames than requested (e.g., 45 instead of 48)
                    actual_transition_frames, _ = get_video_frame_count_and_fps(transition_video_path)

                    # Calculate safe blend_frames based on actual transition length
                    # Transition structure: [context_before][gap][context_after]
                    # We need at least blend_frames at each end for crossfading
                    expected_total = total_frames

                    if actual_transition_frames != expected_total:
                        orchestrator_logger.debug_anomaly(
                            "JOIN_CLIPS",
                            f"transition frame count mismatch expected={expected_total}, actual={actual_transition_frames}",
                            task_id=task_id,
                        )

                        # Calculate the difference
                        frame_diff = expected_total - actual_transition_frames

                        if frame_diff > 0:
                            # VACE generated fewer frames than expected
                            # This could cause misalignment - we need to adjust blend_frames
                            orchestrator_logger.debug_anomaly(
                                "JOIN_CLIPS",
                                f"VACE generated {frame_diff} fewer frames than expected",
                                task_id=task_id,
                            )

                            # Maximum safe blend = half of actual transition (to leave room for gap)
                            max_safe_blend = actual_transition_frames // 4  # Conservative: 1/4 of total at each end

                            if context_frame_count > max_safe_blend:
                                orchestrator_logger.debug_anomaly(
                                    "JOIN_CLIPS",
                                    f"reducing blend_frames from {context_frame_count} to {max_safe_blend} for safety",
                                    task_id=task_id,
                                )

                        total_frames = actual_transition_frames  # Use actual count
                    else:
                        max_safe_blend = context_frame_count

                    # --- 8. Handle transition_only mode (early return) ---
                    if transition_only:
                        # Upload transition video to storage FIRST (before returning JSON)
                        # This is critical: we return JSON with metadata, so we need to upload
                        # the file ourselves rather than relying on the completion logic
                        # (which expects a simple file path, not JSON)
                        transition_output_path, _ = prepare_output_path_with_upload(
                            task_id=task_id,
                            filename=f"{task_id}_transition.mp4",
                            main_output_dir_base=main_output_dir_base,
                            task_type="join_clips_segment")

                        # Copy transition to output path
                        import shutil
                        shutil.copy2(transition_video_path, transition_output_path)

                        # Upload to Supabase storage and get public URL
                        # Must use upload_intermediate_file_to_storage because we're returning JSON,
                        # not a simple file path that the completion logic can handle
                        storage_url = upload_intermediate_file_to_storage(
                            local_file_path=transition_output_path,
                            task_id=task_id,
                            filename=f"{task_id}_transition.mp4")

                        if not storage_url:
                            return False, f"Failed to upload transition video to storage"

                        # Return transition metadata as JSON
                        # The completion logic in db_operations handles JSON outputs specially:
                        # it extracts the storage_path from the URL and stores the full JSON in output_location

                        # GROUND TRUTH: Context is PRESERVED (black mask), gap is GENERATED (white mask)
                        # We put exactly context_from_clip1 + gap_for_guide + context_from_clip2 in the guide
                        # The mask preserves context frames, so they should match what we put in
                        # If actual frame count differs, it's the GAP that changed, not context
                        actual_ctx_clip1 = context_from_clip1 if replace_mode else context_frame_count
                        actual_ctx_clip2 = context_from_clip2 if replace_mode else context_frame_count

                        # Calculate actual gap from ground truth: total - context
                        actual_gap = actual_transition_frames - actual_ctx_clip1 - actual_ctx_clip2

                        # Sanity check: gap should be positive and reasonable
                        if actual_gap < 0:
                            orchestrator_logger.debug_anomaly(
                                "JOIN_CLIPS",
                                (
                                    f"invalid negative gap {actual_gap} from actual_frames={actual_transition_frames}, "
                                    f"ctx1={actual_ctx_clip1}, ctx2={actual_ctx_clip2}"
                                ),
                                task_id=task_id,
                            )
                            # Fall back to requested values - something is very wrong
                            actual_gap = gap_for_guide

                        # Log if gap differs from what we requested (indicates VACE quantization)
                        if actual_gap != gap_for_guide:
                            orchestrator_logger.debug_anomaly(
                                "JOIN_CLIPS",
                                f"gap adjusted by VACE from {gap_for_guide} to {actual_gap}",
                                task_id=task_id,
                            )
                            # Recalculate gap splits proportionally
                            if gap_for_guide > 0:
                                ratio = gap_from_clip1 / gap_for_guide
                                actual_gap_from_clip1 = round(actual_gap * ratio)
                                actual_gap_from_clip2 = actual_gap - actual_gap_from_clip1
                            else:
                                actual_gap_from_clip1 = actual_gap // 2
                                actual_gap_from_clip2 = actual_gap - actual_gap_from_clip1
                            gap_from_clip1 = actual_gap_from_clip1
                            gap_from_clip2 = actual_gap_from_clip2

                        actual_blend = min(actual_ctx_clip1, actual_ctx_clip2, max_safe_blend)

                        orchestrator_logger.debug_block(
                            "OUTPUT",
                            {
                                "mode": "transition_only",
                                "submitted_task_id": submitted_task_id,
                                "processing_time_s": round(processing_time, 1),
                                "storage_url": storage_url,
                                "frames": actual_transition_frames,
                                "structure": f"[{actual_ctx_clip1} ctx] + [{actual_gap} gap] + [{actual_ctx_clip2} ctx]",
                                "gap_splits": (gap_from_clip1, gap_from_clip2),
                                "blend_frames": actual_blend,
                            },
                            task_id=task_id,
                        )

                        try:
                            import numpy as np
                            transition_frames = extract_frames_from_video(str(transition_output_path))
                            if transition_frames and start_all_frames and end_all_frames:
                                trans_first = transition_frames[0].astype(float)
                                start_diffs = {}
                                for offset in range(-2, 3):
                                    test_idx = context_start_idx + offset
                                    if 0 <= test_idx < len(start_all_frames):
                                        diff = np.abs(trans_first - start_all_frames[test_idx].astype(float)).mean()
                                        start_diffs[offset] = diff

                                expected_clip2_last = gap_from_clip2 + context_from_clip2 - 1
                                trans_last = transition_frames[-1].astype(float)
                                end_diffs = {}
                                for offset in range(-2, 3):
                                    test_idx = expected_clip2_last + offset
                                    if 0 <= test_idx < len(end_all_frames):
                                        diff = np.abs(trans_last - end_all_frames[test_idx].astype(float)).mean()
                                        end_diffs[offset] = diff

                                if start_diffs and end_diffs:
                                    best_start = min(start_diffs, key=start_diffs.get)
                                    best_end = min(end_diffs, key=end_diffs.get)
                                    if best_start != 0 or best_end != 0:
                                        orchestrator_logger.debug_anomaly(
                                            "JOIN_CLIPS",
                                            (
                                                f"source alignment offsets start={best_start:+d} "
                                                f"(diff={start_diffs[best_start]:.2f}), "
                                                f"end={best_end:+d} (diff={end_diffs[best_end]:.2f})"
                                            ),
                                            task_id=task_id,
                                        )
                        except (OSError, ValueError, RuntimeError) as e:
                            orchestrator_logger.debug_anomaly(
                                "JOIN_CLIPS",
                                f"source alignment check error: {e}",
                                task_id=task_id,
                            )

                        # Build comprehensive debugging data for the output_location
                        # This helps diagnose alignment issues in the final stitch

                        # Calculate clip1 context indices (REPLACE mode: before gap, INSERT mode: at end)
                        if replace_mode:
                            clip1_ctx_start = start_frame_count - gap_from_clip1 - actual_ctx_clip1
                            clip1_ctx_end = start_frame_count - gap_from_clip1
                            clip2_ctx_start = gap_from_clip2
                            clip2_ctx_end = gap_from_clip2 + actual_ctx_clip2
                        else:
                            clip1_ctx_start = start_frame_count - actual_ctx_clip1
                            clip1_ctx_end = start_frame_count
                            clip2_ctx_start = 0
                            clip2_ctx_end = actual_ctx_clip2

                        return True, json.dumps({
                            # --- Core transition data ---
                            "transition_url": storage_url,
                            "transition_index": task_params_from_db.get("transition_index", 0),
                            "frames": actual_transition_frames,

                            # --- Gap data (ground truth from VACE) ---
                            "gap_frames": actual_gap,  # Actual generated gap (ground truth)
                            "gap_from_clip1": gap_from_clip1,  # Frames trimmed from clip1 END
                            "gap_from_clip2": gap_from_clip2,  # Frames trimmed from clip2 START
                            "requested_gap": gap_frame_count,  # Original requested gap
                            "quantized_gap": gap_for_guide,    # After VACE 4n+1 quantization

                            # --- Context data ---
                            "context_from_clip1": actual_ctx_clip1,
                            "context_from_clip2": actual_ctx_clip2,
                            "context_frame_count": context_frame_count,  # Original requested
                            "blend_frames": actual_blend,

                            # --- Source clip info (for alignment verification) ---
                            "clip1_total_frames": start_frame_count,
                            "clip2_total_frames": end_frame_count,

                            # --- Frame indices showing exactly what was used ---
                            # Clip1: context extracted from [clip1_ctx_start:clip1_ctx_end)
                            # Clip1: frames [clip1_ctx_end:clip1_total_frames) are gap (trimmed)
                            "clip1_context_start_idx": clip1_ctx_start,
                            "clip1_context_end_idx": clip1_ctx_end,

                            # Clip2: frames [0:gap_from_clip2) are gap (trimmed)
                            # Clip2: context extracted from [clip2_ctx_start:clip2_ctx_end)
                            "clip2_context_start_idx": clip2_ctx_start,
                            "clip2_context_end_idx": clip2_ctx_end,

                            # --- Transition structure (for debugging) ---
                            # transition[0:ctx1] = clip1[clip1_ctx_start:clip1_ctx_end] (context before)
                            # transition[ctx1:ctx1+gap] = generated gap frames
                            # transition[ctx1+gap:end] = clip2[clip2_ctx_start:clip2_ctx_end] (context after)
                            "transition_structure": f"[{actual_ctx_clip1} ctx1] + [{actual_gap} gap] + [{actual_ctx_clip2} ctx2]",

                            # --- FPS and resolution (so stitch handler can match clips) ---
                            "fps": start_fps,
                            "resolution": list(parsed_res_wh),  # [width, height]

                            # --- Final stitch guidance ---
                            # When stitching: trim clip1 end by gap_from_clip1, trim clip2 start by gap_from_clip2
                            # Then crossfade: clip1_trimmed[-ctx1:] with transition[0:ctx1]
                            #                 transition[-ctx2:] with clip2_trimmed[0:ctx2]
                            "mode": "replace" if replace_mode else "insert",
                        })

                    # --- 9. Concatenate Full Clips with Transition ---
                    try:
                        import subprocess
                        import tempfile

                        # Create trimmed versions of the original clips
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=join_clips_dir) as clip1_trimmed_file:
                            clip1_trimmed_path = Path(clip1_trimmed_file.name)

                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=join_clips_dir) as clip2_trimmed_file:
                            clip2_trimmed_path = Path(clip2_trimmed_file.name)

                        # Trimming uses gap_from_clip1 and gap_from_clip2 calculated earlier
                        # REPLACE mode: Remove gap frames from boundary, context remains and blends
                        # INSERT mode: Don't remove any frames, just insert transition

                        # For proper blending, blend over the full context region (or max safe if VACE returned fewer frames)
                        blend_frames = min(context_frame_count, max_safe_blend)

                        # Use pre-calculated gap sizes (gap_from_clip1, gap_from_clip2 from step 4)
                        frames_to_remove_clip1 = gap_from_clip1  # 0 for INSERT mode
                        frames_to_keep_clip1 = start_frame_count - frames_to_remove_clip1

                        # Use common frame extraction: frames 0 to (frames_to_keep_clip1 - 1)
                        extract_frame_range_to_video(
                            input_video_path=starting_video,
                            output_video_path=clip1_trimmed_path,
                            start_frame=0,
                            end_frame=frames_to_keep_clip1 - 1,
                            fps=start_fps)

                        # Clip2 trimming uses pre-calculated gap_from_clip2
                        frames_to_skip_clip2 = gap_from_clip2  # 0 for INSERT mode

                        frames_remaining_clip2 = end_frame_count - frames_to_skip_clip2

                        # Log net frame change summary
                        total_gap_removed = frames_to_remove_clip1 + frames_to_skip_clip2
                        # Transition = context + gap + context, but context regions overlap with clips via blend
                        # Effective new frames = gap_for_guide (the middle portion)
                        effective_frames_added = gap_for_guide
                        net_frame_change = effective_frames_added - total_gap_removed

                        # Use common frame extraction: skip first frames_to_skip_clip2 frames
                        extract_frame_range_to_video(
                            input_video_path=ending_video,
                            output_video_path=clip2_trimmed_path,
                            start_frame=frames_to_skip_clip2,
                            end_frame=None,  # All remaining frames
                            fps=end_fps)

                        # Final concatenated output - use standardized path
                        final_output_path, initial_db_location = prepare_output_path_with_upload(
                            task_id=task_id,
                            filename=f"{task_id}_joined.mp4",
                            main_output_dir_base=main_output_dir_base,
                            task_type="join_clips_segment")

                        # Use generalized stitch function with frame-level crossfade blending
                        # This matches the approach used in the travel handlers

                        # Calculate expected final frame count
                        # Stitch: clip1[:-blend] + crossfade(clip1[-blend:], trans[:blend]) + trans[blend:-blend] + crossfade(trans[-blend:], clip2[:blend]) + clip2[blend:]
                        # = (frames_to_keep_clip1 - blend_frames) + blend_frames + (actual_transition_frames - 2*blend_frames) + blend_frames + (frames_remaining_clip2 - blend_frames)
                        # = frames_to_keep_clip1 + actual_transition_frames + frames_remaining_clip2 - 2*blend_frames
                        expected_final_frames = frames_to_keep_clip1 + actual_transition_frames + frames_remaining_clip2 - 2 * blend_frames

                        # Compare to original
                        original_total = start_frame_count + end_frame_count

                        video_paths = [
                            clip1_trimmed_path,
                            Path(transition_video_path),
                            clip2_trimmed_path
                        ]

                        # Blend between clip1->transition and transition->clip2
                        blend_frame_counts = [blend_frames, blend_frames]

                        try:
                            stitch_videos_with_crossfade(
                                video_paths=video_paths,
                                blend_frame_counts=blend_frame_counts,
                                output_video_path=final_output_path,
                                fps=target_fps,
                                crossfade_mode="linear_sharp",
                                crossfade_sharp_amt=0.3)
                        except (OSError, ValueError, RuntimeError) as e:
                            raise ValueError(f"Failed to stitch videos with crossfade: {e}") from e

                        # Verify final output exists and is valid
                        if not final_output_path.exists():
                            raise ValueError(f"Final concatenated video does not exist: {final_output_path}")

                        file_size = final_output_path.stat().st_size
                        if file_size == 0:
                            raise ValueError(f"Final concatenated video is empty (0 bytes)")

                        # === FINAL OUTPUT VERIFICATION ===
                        final_actual_frames, final_actual_fps = get_video_frame_count_and_fps(str(final_output_path))
                        if final_actual_frames and abs(final_actual_frames - expected_final_frames) > 3:
                            orchestrator_logger.debug_anomaly(
                                "JOIN_CLIPS",
                                f"final frame count mismatch diff={final_actual_frames - expected_final_frames}",
                                task_id=task_id,
                            )

                        # Extract poster image/thumbnail from the final video
                        poster_output_path = final_output_path.with_suffix('.jpg')
                        try:
                            # Extract first frame as poster
                            poster_frame_index = 0

                            if save_frame_from_video(
                                final_output_path,
                                poster_frame_index,
                                poster_output_path,
                                parsed_res_wh
                            ):
                                pass
                        except (OSError, ValueError, RuntimeError) as poster_error:
                            orchestrator_logger.debug_anomaly("JOIN_CLIPS", f"Task {task_id}: Warning: Poster extraction failed: {poster_error}")

                        # Clean up temporary files (unless debug mode is enabled)
                        debug_mode = task_params_from_db.get("debug", False)
                        if not debug_mode:
                            try:
                                clip1_trimmed_path.unlink()
                                clip2_trimmed_path.unlink()
                                Path(transition_video_path).unlink()  # Remove transition-only video
                                # Clean up vid2vid source video if it was created
                                if vid2vid_source_video_path is not None and vid2vid_source_video_path.exists():
                                    vid2vid_source_video_path.unlink()
                            except OSError as cleanup_error:
                                orchestrator_logger.debug_anomaly("JOIN_CLIPS", f"Warning: Cleanup failed: {cleanup_error}")

                        # Add audio to final output if this is the last join and audio_url is provided
                        if is_last_join and audio_url:
                            # Create path for video with audio
                            video_with_audio_path = final_output_path.with_name(
                                final_output_path.stem + "_with_audio.mp4"
                            )

                            result_with_audio = add_audio_to_video(
                                input_video_path=final_output_path,
                                audio_url=audio_url,
                                output_video_path=video_with_audio_path,
                                temp_dir=join_clips_dir)

                            if result_with_audio and result_with_audio.exists():
                                # Replace the final output path with the audio version
                                # Remove the silent version
                                try:
                                    final_output_path.unlink()
                                except OSError as e_unlink_silent:
                                    headless_logger.debug_anomaly("JOIN_CLIPS", f"Task {task_id}: Could not remove silent video {final_output_path}: {e_unlink_silent}")

                                # Rename audio version to final path
                                result_with_audio.rename(final_output_path)

                        # Handle upload and get final DB location
                        final_db_location = upload_and_get_final_output_location(
                            local_file_path=final_output_path,
                            initial_db_location=initial_db_location)
                        orchestrator_logger.debug_block(
                            "OUTPUT",
                            {
                                "mode": "joined_video",
                                "processing_time_s": round(processing_time, 1),
                                "output": final_db_location,
                                "expected_frames": expected_final_frames,
                                "actual_frames": final_actual_frames,
                                "fps": final_actual_fps,
                                "file_size": file_size,
                            },
                            task_id=task_id,
                        )

                        return True, final_db_location

                    except (OSError, ValueError, RuntimeError) as concat_error:
                        error_msg = f"Failed to concatenate full clips: {concat_error}"
                        orchestrator_logger.error(error_msg, task_id=task_id, exc_info=True)
                        # Return the transition video as fallback
                        orchestrator_logger.debug_anomaly(
                            "JOIN_CLIPS",
                            "returning transition video as fallback after concat failure",
                            task_id=task_id,
                        )
                        return True, transition_video_path

                elif status.status == "failed":
                    error_msg = status.error_message or "Generation failed without specific error message"
                    orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: Generation failed - {error_msg}")
                    return False, error_msg

                else:
                    # Still processing
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval

            # Timeout reached
            error_msg = f"Processing timeout after {max_wait_time} seconds"
            orchestrator_logger.debug_anomaly("JOIN_CLIPS_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        except (RuntimeError, ValueError, OSError) as e:
            error_msg = f"Failed to submit/complete generation task: {e}"
            orchestrator_logger.error(error_msg, task_id=task_id, exc_info=True)
            return False, error_msg

    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        error_msg = f"Unexpected error in join_clips handler: {e}"
        orchestrator_logger.error(error_msg, task_id=task_id, exc_info=True)
        return False, error_msg
