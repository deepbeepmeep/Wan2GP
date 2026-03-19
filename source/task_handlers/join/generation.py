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
    get_video_frame_count_ffprobe,
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
    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Starting join_clips handler")

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

        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Mode: {'REPLACE' if replace_mode else 'INSERT'}")
        if transition_only:
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: transition_only=True - will output transition video only (no stitching)")
        if replace_mode:
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: gap_frame_count={gap_frame_count} frames will REPLACE boundary frames (no insertion)")

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
                    orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg

                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Part of orchestrator {orchestrator_task_id_ref}, fetching predecessor output")

                # Fetch predecessor output using edge function
                predecessor_id, predecessor_output = db_ops.get_predecessor_output_via_edge_function(task_id)

                if not predecessor_output:
                    error_msg = f"Failed to fetch predecessor output (predecessor_id={predecessor_id})"
                    orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg

                starting_video_path = predecessor_output
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Fetched predecessor output: {predecessor_output}")
            else:
                # Standalone join_clips task without starting_video_path
                error_msg = "starting_video_path is required"
                orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

        if not ending_video_path:
            error_msg = "ending_video_path is required"
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Validate task queue
        if task_queue is None:
            error_msg = "task_queue is required for join_clips"
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Create working directory - use custom dir if provided by orchestrator
        if "join_output_dir" in task_params_from_db:
            join_clips_dir = Path(task_params_from_db["join_output_dir"])
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Using orchestrator output dir: {join_clips_dir}")
        else:
            join_clips_dir = main_output_dir_base / "join_clips" / task_id

        join_clips_dir.mkdir(parents=True, exist_ok=True)

        # Download videos if they are URLs (e.g., Supabase storage URLs)
        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Checking if videos need to be downloaded...")
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
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        if not ending_video.exists():
            error_msg = f"Ending video not found: {ending_video_path}"
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Parameters validated")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Starting video: {starting_video}")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Ending video: {ending_video}")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Context frames: {context_frame_count}")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Gap frames: {gap_frame_count}")
        if aspect_ratio:
            orchestrator_logger.debug(f"[JOIN_CLIPS]   Target aspect ratio: {aspect_ratio}")

        # === FRAME/FPS DIAGNOSTICS (BEFORE ensure_video_fps) ===
        try:
            start_ff_frames_pre = get_video_frame_count_ffprobe(str(starting_video))
            end_ff_frames_pre = get_video_frame_count_ffprobe(str(ending_video))
            start_ff_fps_pre = get_video_fps_ffprobe(str(starting_video))
            end_ff_fps_pre = get_video_fps_ffprobe(str(ending_video))
            start_cv_frames_pre, start_cv_fps_pre = get_video_frame_count_and_fps(str(starting_video))
            end_cv_frames_pre, end_cv_fps_pre = get_video_frame_count_and_fps(str(ending_video))
            orchestrator_logger.debug(
                f"[JOIN_CLIPS] Task {task_id}: Pre-ensure_fps stats: "
                f"start(ff_frames={start_ff_frames_pre}, ff_fps={start_ff_fps_pre}, cv_frames={start_cv_frames_pre}, cv_fps={start_cv_fps_pre}); "
                f"end(ff_frames={end_ff_frames_pre}, ff_fps={end_ff_fps_pre}, cv_frames={end_cv_frames_pre}, cv_fps={end_cv_fps_pre})"
            )
        except (OSError, ValueError, RuntimeError) as e_diag:
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Pre-ensure_fps diagnostics error: {e_diag}")

        # --- 2a. Ensure Videos are at Target FPS ---
        # Strategy: use the minimum FPS of the two source clips so we never upsample.
        # An explicit fps param overrides this (e.g., if the frontend requests a specific FPS).
        use_input_video_fps = task_params_from_db.get("use_input_video_fps", False)

        if use_input_video_fps:
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: use_input_video_fps=True, keeping original video FPS")
            # Don't convert - will use input video's FPS
        else:
            explicit_fps = task_params_from_db.get("fps")
            if explicit_fps:
                target_fps_param = explicit_fps
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Using explicit fps param: {target_fps_param}")
            else:
                # Auto-detect: use the minimum FPS of the two clips
                start_fps = get_video_fps_ffprobe(str(starting_video)) or 16
                end_fps = get_video_fps_ffprobe(str(ending_video)) or 16
                target_fps_param = min(start_fps, end_fps)
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Auto-detected target FPS: {target_fps_param} (start={start_fps}, end={end_fps})")
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Ensuring videos are at {target_fps_param} FPS...")

            starting_video_before = starting_video
            try:
                starting_video = ensure_video_fps(
                    input_video_path=starting_video,
                    target_fps=target_fps_param,
                    output_dir=join_clips_dir)
            except (OSError, ValueError, RuntimeError) as e:
                error_msg = f"Failed to ensure starting video is at {target_fps_param} fps: {e}"
                orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg
            if Path(starting_video_before) != Path(starting_video):
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: ensure_video_fps RESAMPLED starting clip -> {starting_video.name}")

            ending_video_before = ending_video
            try:
                ending_video = ensure_video_fps(
                    input_video_path=ending_video,
                    target_fps=target_fps_param,
                    output_dir=join_clips_dir)
            except (OSError, ValueError, RuntimeError) as e:
                error_msg = f"Failed to ensure ending video is at {target_fps_param} fps: {e}"
                orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg
            if Path(ending_video_before) != Path(ending_video):
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: ensure_video_fps RESAMPLED ending clip -> {ending_video.name}")

        # === FRAME/FPS DIAGNOSTICS (AFTER ensure_video_fps) ===
        try:
            start_ff_frames_post = get_video_frame_count_ffprobe(str(starting_video))
            end_ff_frames_post = get_video_frame_count_ffprobe(str(ending_video))
            start_ff_fps_post = get_video_fps_ffprobe(str(starting_video))
            end_ff_fps_post = get_video_fps_ffprobe(str(ending_video))
            start_cv_frames_post, start_cv_fps_post = get_video_frame_count_and_fps(str(starting_video))
            end_cv_frames_post, end_cv_fps_post = get_video_frame_count_and_fps(str(ending_video))
            orchestrator_logger.debug(
                f"[JOIN_CLIPS] Task {task_id}: Post-ensure_fps stats: "
                f"start(ff_frames={start_ff_frames_post}, ff_fps={start_ff_fps_post}, cv_frames={start_cv_frames_post}, cv_fps={start_cv_fps_post}); "
                f"end(ff_frames={end_ff_frames_post}, ff_fps={end_ff_fps_post}, cv_frames={end_cv_frames_post}, cv_fps={end_cv_fps_post})"
            )
        except (OSError, ValueError, RuntimeError) as e_diag2:
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Post-ensure_fps diagnostics error: {e_diag2}")

        # --- 2b. Standardize Videos to Target Aspect Ratio (if specified) ---
        if aspect_ratio:
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Standardizing videos to aspect ratio {aspect_ratio}...")

            # Standardize starting video
            standardized_start_path = join_clips_dir / f"start_standardized_{task_id}.mp4"
            result = standardize_video_aspect_ratio(
                input_video_path=starting_video,
                output_video_path=standardized_start_path,
                target_aspect_ratio=aspect_ratio,
                task_id_for_logging=task_id)
            if result is None:
                error_msg = f"Failed to standardize starting video to aspect ratio {aspect_ratio}"
                orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg
            starting_video = standardized_start_path
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Starting video standardized")

            # Standardize ending video
            standardized_end_path = join_clips_dir / f"end_standardized_{task_id}.mp4"
            result = standardize_video_aspect_ratio(
                input_video_path=ending_video,
                output_video_path=standardized_end_path,
                target_aspect_ratio=aspect_ratio,
                task_id_for_logging=task_id)
            if result is None:
                error_msg = f"Failed to standardize ending video to aspect ratio {aspect_ratio}"
                orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg
            ending_video = standardized_end_path
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Ending video standardized")
        else:
            # No explicit aspect ratio specified - auto-standardize ending video to match starting video
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Checking if videos have matching aspect ratios...")

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

                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Starting video: {start_w}x{start_h} (aspect: {start_aspect:.3f})")
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Ending video: {end_w}x{end_h} (aspect: {end_aspect:.3f})")

                    # If aspect ratios differ by more than 1%, standardize ending video to match starting video
                    if abs(start_aspect - end_aspect) > 0.01:
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Aspect ratios differ - standardizing ending video to match starting video")

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

                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Auto-detected aspect ratio: {auto_aspect_ratio}")

                        standardized_end_path = join_clips_dir / f"end_standardized_{task_id}.mp4"
                        result = standardize_video_aspect_ratio(
                            input_video_path=ending_video,
                            output_video_path=standardized_end_path,
                            target_aspect_ratio=auto_aspect_ratio,
                            task_id_for_logging=task_id)
                        if result is None:
                            orchestrator_logger.debug(f"[JOIN_CLIPS_WARNING] Task {task_id}: Failed to auto-standardize ending video, proceeding with original")
                        else:
                            ending_video = standardized_end_path
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Ending video standardized to match starting video")
                    else:
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Videos have matching aspect ratios, no standardization needed")

            except (OSError, ValueError, RuntimeError) as e:
                orchestrator_logger.debug(f"[JOIN_CLIPS_WARNING] Task {task_id}: Could not check video dimensions: {e}")
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Proceeding with original videos")

        # --- 3. Extract Video Properties ---
        try:
            start_frame_count, start_fps = get_video_frame_count_and_fps(str(starting_video))
            end_frame_count, end_fps = get_video_frame_count_and_fps(str(ending_video))

            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Starting video - {start_frame_count} frames @ {start_fps} fps")
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Ending video - {end_frame_count} frames @ {end_fps} fps")

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Failed to extract video properties: {e}"
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Validate that frame counts were detected (WebM and some codecs may fail)
        if start_frame_count is None:
            error_msg = f"Could not detect frame count for starting video: {starting_video}. The video may be corrupt, empty, or in an unsupported format."
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        if end_frame_count is None:
            error_msg = f"Could not detect frame count for ending video: {ending_video}. The video may be corrupt, empty, or in an unsupported format."
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        # Auto-adjust context frame count if it exceeds available frames
        original_context_frame_count = context_frame_count
        max_available_context = min(start_frame_count, end_frame_count)

        if context_frame_count > max_available_context:
            context_frame_count = max_available_context
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u26a0\ufe0f Auto-adjusted context_frame_count from {original_context_frame_count} to {context_frame_count} (limited by shortest video: start={start_frame_count}, end={end_frame_count})")
            headless_logger.warning(
                f"[JOIN_CLIPS] Task {task_id}: context_frame_count reduced from {original_context_frame_count} to {context_frame_count} to fit available frames",
                task_id=task_id
            )

        # Set target FPS based on use_input_video_fps setting
        if use_input_video_fps:
            target_fps = start_fps
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Using input video FPS: {target_fps}")
        else:
            target_fps = task_params_from_db.get("fps") or 16
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Using target FPS: {target_fps}")

        # --- 4. Calculate gap sizes first (needed for REPLACE mode context extraction) ---
        # Calculate VACE quantization adjustments early so we know exact gap sizes
        quantization_result = _calculate_vace_quantization(
            context_frame_count=context_frame_count,
            gap_frame_count=gap_frame_count,
            replace_mode=replace_mode
        )
        gap_for_guide = quantization_result['gap_for_guide']
        quantization_shift = quantization_result['quantization_shift']

        # Calculate gap split for REPLACE mode
        # Start with even split, then adjust if one clip is too short
        gap_from_clip1 = gap_for_guide // 2 if replace_mode else 0
        gap_from_clip2 = (gap_for_guide - gap_from_clip1) if replace_mode else 0

        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Mode={'REPLACE' if replace_mode else 'INSERT'}, gap_for_guide={gap_for_guide}")
        if replace_mode:
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: REPLACE initial split: {gap_from_clip1} from clip1, {gap_from_clip2} from clip2")

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
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u26a0\ufe0f Clip1 too short ({start_frame_count} frames) - shifted {shift_amount} gap frames to clip2")
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Adjusted split: {gap_from_clip1} from clip1, {gap_from_clip2} from clip2")

            if end_frame_count < gap_from_clip2 + 1:
                # Clip2 too short - shift gap frames to clip1
                max_gap_from_clip2 = max(0, end_frame_count - 1)
                shift_amount = gap_from_clip2 - max_gap_from_clip2
                gap_from_clip2 = max_gap_from_clip2
                gap_from_clip1 = gap_from_clip1 + shift_amount
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u26a0\ufe0f Clip2 too short ({end_frame_count} frames) - shifted {shift_amount} gap frames to clip1")
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Adjusted split: {gap_from_clip1} from clip1, {gap_from_clip2} from clip2")

            # Final validation - check if total gap is still achievable
            total_available = (start_frame_count - 1) + (end_frame_count - 1)  # -1 for minimum 1 context frame each
            if gap_for_guide > total_available:
                error_msg = (
                    f"Videos too short for requested gap: need {gap_for_guide} gap frames but only "
                    f"{total_available} available (start: {start_frame_count}, end: {end_frame_count}). "
                    f"Try reducing gap_frame_count or using longer source clips."
                )
                orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg
        orchestrator_logger.debug(
            f"[JOIN_CLIPS] Task {task_id}: gap split summary: "
            f"requested_gap={gap_frame_count}, gap_for_guide={gap_for_guide}, "
            f"gap_from_clip1={gap_from_clip1}, gap_from_clip2={gap_from_clip2}, "
            f"context_frame_count={context_frame_count}"
        )

        # --- 5. Extract Context Frames ---
        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Extracting context frames...")

        try:
            # Extract all frames from both videos
            start_all_frames = extract_frames_from_video(str(starting_video))
            if not start_all_frames or len(start_all_frames) < context_frame_count:
                error_msg = f"Failed to extract {context_frame_count} frames from starting video"
                orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                return False, error_msg

            end_all_frames = extract_frames_from_video(str(ending_video))
            if not end_all_frames or len(end_all_frames) < context_frame_count:
                error_msg = f"Failed to extract {context_frame_count} frames from ending video"
                orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
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
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Clips too short, applying proportional reduction")
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Original: gap={gap_frame_count}, context={context_frame_count}, total_needed={total_needed}")
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Available: clip1={clip1_available}, clip2={clip2_available}, min={min_clip_frames}")

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
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Forced minimum VACE settings (gap=1, context=2)")

                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Adjusted: gap={adjusted_gap}, context={adjusted_context}, total={adjusted_total}")
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Reduction ratio: {ratio:.2%}")

                    # Update the working values
                    gap_frame_count = adjusted_gap
                    context_frame_count = adjusted_context

                    # Recalculate gap splits with new gap value
                    gap_from_clip1 = gap_frame_count // 2
                    gap_from_clip2 = gap_frame_count - gap_from_clip1
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   New gap splits: gap_from_clip1={gap_from_clip1}, gap_from_clip2={gap_from_clip2}")

                    # Recalculate VACE quantization with new values
                    quantization_result = _calculate_vace_quantization(
                        context_frame_count=context_frame_count,
                        gap_frame_count=gap_frame_count,
                        replace_mode=replace_mode
                    )
                    gap_for_guide = quantization_result['gap_for_guide']
                    quantization_shift = quantization_result['quantization_shift']
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Recalculated quantization: gap_for_guide={gap_for_guide}, shift={quantization_shift}")

                    # Summary
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u2713 PROPORTIONAL REDUCTION COMPLETE")
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   FINAL: gap={gap_frame_count}, context={context_frame_count}, gap_splits=({gap_from_clip1},{gap_from_clip2})")
                else:
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Clips have sufficient frames ({min_clip_frames} >= {total_needed}), no reduction needed")

                # Now calculate available context with (potentially adjusted) gap values
                clip1_max_context = clip1_available - gap_from_clip1
                clip2_max_context = clip2_available - gap_from_clip2

                # Validate clips have enough frames (should pass after proportional reduction)
                if clip1_max_context < 1:
                    error_msg = f"Starting video too short: need at least {gap_from_clip1 + 1} frames, have {clip1_available}"
                    orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg

                if clip2_max_context < 1:
                    error_msg = f"Ending video too short: need at least {gap_from_clip2 + 1} frames, have {clip2_available}"
                    orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg

                # Final context adjustment (may further reduce if clips are asymmetric)
                context_from_clip1 = min(context_frame_count, clip1_max_context)
                context_from_clip2 = min(context_frame_count, clip2_max_context)

                needs_asymmetric_context = context_from_clip1 < context_frame_count or context_from_clip2 < context_frame_count
                if needs_asymmetric_context:
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Adjusting context frames to fit clip lengths")
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   clip1: {clip1_available} frames, max context: {clip1_max_context} -> using {context_from_clip1}")
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   clip2: {clip2_available} frames, max context: {clip2_max_context} -> using {context_from_clip2}")

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
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Recalculated quantization with asymmetric context: gap_for_guide={gap_for_guide}")

                # Context from clip1: frames BEFORE the gap (not the last frames)
                # If removing last N frames, context is the N frames before that
                context_start_idx = len(start_all_frames) - gap_from_clip1 - context_from_clip1
                context_end_idx = len(start_all_frames) - gap_from_clip1
                start_context_frames = start_all_frames[context_start_idx:context_end_idx]

                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode - clip1 context from frames [{context_start_idx}:{context_end_idx}] ({context_from_clip1} frames)")
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   (frames {context_end_idx} to {len(start_all_frames)-1} will be removed as gap)")
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: REPLACE clip1: context_idx=[{context_start_idx}:{context_end_idx}), removed_tail=[{context_end_idx}:{len(start_all_frames)})")

                # Context from clip2: frames AFTER the gap (not the first frames)
                # If removing first M frames, context is the N frames after that
                end_context_frames = end_all_frames[gap_from_clip2:gap_from_clip2 + context_from_clip2]

                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode - clip2 context from frames [{gap_from_clip2}:{gap_from_clip2 + context_from_clip2}] ({context_from_clip2} frames)")
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   (frames 0 to {gap_from_clip2-1} will be removed as gap)")
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: REPLACE clip2: removed_head=[0:{gap_from_clip2}), context_idx=[{gap_from_clip2}:{gap_from_clip2 + context_from_clip2})")

                # --- VID2VID SOURCE: Extract gap frames for vid2vid initialization ---
                # If vid2vid_init_strength is set, create a source video from the gap frames
                vid2vid_init_strength = task_params_from_db.get("vid2vid_init_strength")
                if vid2vid_init_strength is not None and vid2vid_init_strength < 1.0:
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Vid2vid mode enabled (strength={vid2vid_init_strength})")

                    # Extract gap frames that are being replaced
                    # Gap from clip1: last N frames of starting video
                    gap_frames_clip1 = start_all_frames[context_end_idx:]  # frames from context_end_idx to end
                    # Gap from clip2: first M frames of ending video
                    gap_frames_clip2 = end_all_frames[:gap_from_clip2]  # frames 0 to gap_from_clip2-1

                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Extracted {len(gap_frames_clip1)} gap frames from clip1")
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Extracted {len(gap_frames_clip2)} gap frames from clip2")

                    # Build vid2vid source video with same structure as guide: context + gap + context
                    vid2vid_frames = []
                    vid2vid_frames.extend(start_context_frames)  # Context before
                    vid2vid_frames.extend(gap_frames_clip1)      # Gap from clip1
                    vid2vid_frames.extend(gap_frames_clip2)      # Gap from clip2
                    vid2vid_frames.extend(end_context_frames)    # Context after

                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Vid2vid source video structure:")
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Context before: {len(start_context_frames)} frames")
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Gap (clip1): {len(gap_frames_clip1)} frames")
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Gap (clip2): {len(gap_frames_clip2)} frames")
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Context after: {len(end_context_frames)} frames")
                    orchestrator_logger.debug(f"[JOIN_CLIPS]   Total: {len(vid2vid_frames)} frames")

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
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Created vid2vid source video: {vid2vid_source_video_path}")
                    except (OSError, ValueError, RuntimeError) as v2v_err:
                        orchestrator_logger.debug(f"[JOIN_CLIPS_WARNING] Task {task_id}: Error creating vid2vid source video: {v2v_err}")
                        vid2vid_source_video_path = None
            else:
                # INSERT mode: Context is at the boundary (last/first frames)
                # No frames are removed, we're just inserting new frames between clips
                start_context_frames = start_all_frames[-context_frame_count:]
                end_context_frames = end_all_frames[:context_frame_count]

                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: INSERT mode - clip1 context from last {context_frame_count} frames")
                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: INSERT mode - clip2 context from first {context_frame_count} frames")
                orchestrator_logger.debug(
                    f"[JOIN_CLIPS] Task {task_id}: INSERT indices: "
                    f"clip1 context_idx=[{len(start_all_frames)-context_frame_count}:{len(start_all_frames)}), "
                    f"clip2 context_idx=[0:{context_frame_count})"
                )

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Failed to extract context frames: {e}"
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}", exc_info=True)
            return False, error_msg

        # === CONTEXT EXTRACTION SUMMARY ===
        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: === Context Extraction Summary ===")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Starting video: {len(start_all_frames)} total frames")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Ending video: {len(end_all_frames)} total frames")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Context extracted from start: {len(start_context_frames)} frames")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Context extracted from end: {len(end_context_frames)} frames")
        if replace_mode:
            orchestrator_logger.debug(f"[JOIN_CLIPS]   Mode: REPLACE (context from OUTSIDE gap region)")
            orchestrator_logger.debug(f"[JOIN_CLIPS]   Gap frames removed from clip1 tail: {gap_from_clip1}")
            orchestrator_logger.debug(f"[JOIN_CLIPS]   Gap frames removed from clip2 head: {gap_from_clip2}")
        else:
            orchestrator_logger.debug(f"[JOIN_CLIPS]   Mode: INSERT (context at boundaries, no removal)")
        # Log expected guide structure
        expected_guide_frames = len(start_context_frames) + gap_for_guide + len(end_context_frames)
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Expected guide structure: {len(start_context_frames)} ctx + {gap_for_guide} gap + {len(end_context_frames)} ctx = {expected_guide_frames} total")

        # Get resolution from first frame or task params
        first_frame = start_context_frames[0]
        frame_height, frame_width = first_frame.shape[:2]
        detected_res_wh = (frame_width, frame_height)

        # Determine resolution: check use_input_video_resolution flag first, then explicit resolution param
        use_input_video_resolution = task_params_from_db.get("use_input_video_resolution", False)
        resolution_override = task_params_from_db.get("resolution")  # May be None or a list

        # DEBUG: Log the raw values to diagnose resolution issues
        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Resolution decision debug:")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   detected_res_wh (from frames): {detected_res_wh}")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   use_input_video_resolution raw value: {use_input_video_resolution!r} (type: {type(use_input_video_resolution).__name__})")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   resolution_override raw value: {resolution_override!r}")

        if use_input_video_resolution:
            parsed_res_wh = detected_res_wh
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: use_input_video_resolution=True, using detected resolution: {parsed_res_wh}")
        elif resolution_override is not None:
            # resolution_override should be a list [width, height]
            parsed_res_wh = (resolution_override[0], resolution_override[1])
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Using resolution override: {parsed_res_wh}")
        else:
            parsed_res_wh = detected_res_wh
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Using detected resolution: {parsed_res_wh}")

        # --- 6. Build Guide and Mask Videos (using shared helper) ---
        # (quantization already calculated above in step 4)
        quantized_total_frames = quantization_result['total_frames']

        # === QUANTIZATION DIAGNOSTIC SUMMARY ===
        actual_ctx_before = len(start_context_frames)
        actual_ctx_after = len(end_context_frames)
        orchestrator_logger.debug(f"[FRAME_COUNTS] Task {task_id}: ========== QUANTIZATION SUMMARY ==========")
        if actual_ctx_before != actual_ctx_after or actual_ctx_before != context_frame_count:
            orchestrator_logger.debug(f"[FRAME_COUNTS] Task {task_id}: Context frames: {actual_ctx_before} before, {actual_ctx_after} after (requested: {context_frame_count})")
        else:
            orchestrator_logger.debug(f"[FRAME_COUNTS] Task {task_id}: Context frames: {context_frame_count} (each side)")
        orchestrator_logger.debug(f"[FRAME_COUNTS] Task {task_id}: Requested gap: {gap_frame_count}")
        orchestrator_logger.debug(f"[FRAME_COUNTS] Task {task_id}: Actual total: {actual_ctx_before + gap_for_guide + actual_ctx_after}")
        orchestrator_logger.debug(f"[FRAME_COUNTS] Task {task_id}: Quantized total: {quantized_total_frames} (4N+1 enforced)")
        if quantization_shift > 0:
            orchestrator_logger.debug(f"[FRAME_COUNTS] Task {task_id}: Gap adjusted: {gap_frame_count} \u2192 {gap_for_guide} (shift: -{quantization_shift})")
        else:
            orchestrator_logger.debug(f"[FRAME_COUNTS] Task {task_id}: No quantization needed (already valid 4N+1)")
        orchestrator_logger.debug(f"[FRAME_COUNTS] Task {task_id}: ===========================================")

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
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: keep_bridging_images=True: Using start_clip[-1] at gap[{idx1}] and end_clip[0] at gap[{idx2}]")
                else:
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Gap too small ({gap_for_guide}) for equidistant anchors, skipping")
            else:
                orchestrator_logger.debug(f"[JOIN_CLIPS_WARNING] Task {task_id}: keep_bridging_images=True but contexts empty")

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
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}", exc_info=True)
            return False, error_msg

        total_frames = guide_frame_count

        # === GUIDE/MASK VERIFICATION ===
        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: === Guide/Mask Verification ===")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Expected total frames: {quantized_total_frames}")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Actual guide frames: {guide_frame_count}")
        is_valid_4n1 = (total_frames - 1) % 4 == 0
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Valid 4N+1: {is_valid_4n1} ({total_frames} = 4*{(total_frames-1)//4}+1)")
        if guide_frame_count != quantized_total_frames:
            orchestrator_logger.debug(f"[JOIN_CLIPS]   \u26a0\ufe0f  MISMATCH: Guide ({guide_frame_count}) != Expected ({quantized_total_frames})")
        else:
            orchestrator_logger.debug(f"[JOIN_CLIPS]   \u2713 Frame counts match")

        # --- 6. Prepare Generation Parameters (using shared helper) ---
        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Preparing generation parameters...")

        # Determine model (default to Lightning baseline for fast generation)
        model = task_params_from_db.get("model", "wan_2_2_vace_lightning_baseline_2_2_2")

        # Ensure prompt is valid
        prompt = ensure_valid_prompt(prompt)
        negative_prompt = ensure_valid_negative_prompt(
            task_params_from_db.get("negative_prompt", "")
        )

        # Extract additional_loras for logging (if present)
        additional_loras = task_params_from_db.get("additional_loras", {})
        if additional_loras:
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Found {len(additional_loras)} additional LoRAs: {list(additional_loras.keys())}")

        # Extract phase_config for logging (if present)
        phase_config = task_params_from_db.get("phase_config")
        if phase_config:
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Found phase_config: {json.dumps(phase_config, default=str)[:100]}...")

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
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Added vid2vid source video to generation params")

        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Generation parameters prepared")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Model: {model}")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Video length: {total_frames} frames")
        orchestrator_logger.debug(f"[JOIN_CLIPS]   Resolution: {parsed_res_wh}")

        # Log LoRA settings
        if generation_params.get("additional_loras"):
            orchestrator_logger.debug(f"[JOIN_CLIPS]   Additional LoRAs: {len(generation_params['additional_loras'])} configured")

        # --- 7. Submit to Generation Queue ---
        # === WGP SUBMISSION DIAGNOSTIC SUMMARY ===
        orchestrator_logger.debug(f"[WGP_SUBMIT] Task {task_id}: ========== WGP GENERATION REQUEST ==========")
        orchestrator_logger.debug(f"[WGP_SUBMIT] Task {task_id}: video_length (target frames): {total_frames}")
        is_valid_4n1 = (total_frames - 1) % 4 == 0
        _valid_marker = '\u2713' if is_valid_4n1 else '\u2717 WARNING'
        orchestrator_logger.debug(f"[WGP_SUBMIT] Task {task_id}: Valid 4N+1: {is_valid_4n1} {_valid_marker}")
        orchestrator_logger.debug(f"[WGP_SUBMIT] Task {task_id}: video_guide: {created_guide_video}")
        orchestrator_logger.debug(f"[WGP_SUBMIT] Task {task_id}: video_mask: {created_mask_video}")
        orchestrator_logger.debug(f"[WGP_SUBMIT] Task {task_id}: model: {model}")
        orchestrator_logger.debug(f"[WGP_SUBMIT] Task {task_id}: resolution: {parsed_res_wh}")
        orchestrator_logger.debug(f"[WGP_SUBMIT] Task {task_id}: =============================================")

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
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Submitted to generation queue as {submitted_task_id}")

            # Wait for completion using polling pattern (same as direct queue tasks)
            # Allow timeout override via task params (default: 30 minutes to handle slow model loading)
            max_wait_time = task_params_from_db.get("max_wait_time", 1800)  # 30 minute default timeout
            wait_interval = 2  # Check every 2 seconds
            elapsed_time = 0
            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Waiting for generation (timeout: {max_wait_time}s)")

            while elapsed_time < max_wait_time:
                status = task_queue.get_task_status(task_id)

                if status is None:
                    error_msg = "Task status became None during processing"
                    orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
                    return False, error_msg

                if status.status == "completed":
                    transition_video_path = status.result_path
                    processing_time = status.processing_time or 0
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Generation completed successfully in {processing_time:.1f}s")
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Transition video: {transition_video_path}")

                    # IMPORTANT: Check actual frame count vs expected
                    # VACE may generate fewer frames than requested (e.g., 45 instead of 48)
                    actual_transition_frames, _ = get_video_frame_count_and_fps(transition_video_path)

                    # Calculate safe blend_frames based on actual transition length
                    # Transition structure: [context_before][gap][context_after]
                    # We need at least blend_frames at each end for crossfading
                    expected_total = total_frames

                    if actual_transition_frames != expected_total:
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u26a0\ufe0f  Frame count mismatch! Expected {expected_total}, got {actual_transition_frames}")

                        # Calculate the difference
                        frame_diff = expected_total - actual_transition_frames

                        if frame_diff > 0:
                            # VACE generated fewer frames than expected
                            # This could cause misalignment - we need to adjust blend_frames
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: VACE generated {frame_diff} fewer frames than expected")

                            # Maximum safe blend = half of actual transition (to leave room for gap)
                            max_safe_blend = actual_transition_frames // 4  # Conservative: 1/4 of total at each end

                            if context_frame_count > max_safe_blend:
                                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u26a0\ufe0f  Reducing blend_frames from {context_frame_count} to {max_safe_blend} for safety")

                        total_frames = actual_transition_frames  # Use actual count
                    else:
                        max_safe_blend = context_frame_count

                    # --- 8. Handle transition_only mode (early return) ---
                    if transition_only:
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: transition_only mode - uploading transition video directly")

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

                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: transition_only complete - {storage_url}")

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
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u26a0\ufe0f INVALID: Calculated negative gap ({actual_gap}). This shouldn't happen!")
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   actual_frames={actual_transition_frames}, ctx1={actual_ctx_clip1}, ctx2={actual_ctx_clip2}")
                            # Fall back to requested values - something is very wrong
                            actual_gap = gap_for_guide

                        # Log if gap differs from what we requested (indicates VACE quantization)
                        if actual_gap != gap_for_guide:
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u26a0\ufe0f Gap adjusted by VACE from {gap_for_guide} to {actual_gap}")
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
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Gap splits adjusted to ({gap_from_clip1}, {gap_from_clip2})")

                        actual_blend = min(actual_ctx_clip1, actual_ctx_clip2, max_safe_blend)

                        # Log ground truth values that will be reported
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: GROUND TRUTH for transition_only output:")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   frames={actual_transition_frames} (actual from VACE)")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   structure: [{actual_ctx_clip1} ctx] + [{actual_gap} gap] + [{actual_ctx_clip2} ctx]")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   gap_splits: ({gap_from_clip1}, {gap_from_clip2})")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   blend_frames={actual_blend}")

                        # === OFFSET DETECTION AT SOURCE: Verify transition context matches original clips ===
                        # Compare transition frames against original clip frames at different offsets
                        # to detect if VACE shifted the alignment
                        try:
                            import numpy as np
                            transition_frames = extract_frames_from_video(str(transition_output_path))
                            if transition_frames and start_all_frames and end_all_frames:
                                orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE] Task {task_id}: Verifying transition alignment at source...")

                                # Check START context: trans[0] should match clip1[context_start_idx]
                                orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE]   Comparing trans[0] vs clip1 frames around index {context_start_idx}")
                                trans_first = transition_frames[0].astype(float)
                                for offset in range(-2, 3):
                                    test_idx = context_start_idx + offset
                                    if 0 <= test_idx < len(start_all_frames):
                                        test_frame = start_all_frames[test_idx].astype(float)
                                        diff = np.abs(trans_first - test_frame).mean()
                                        marker = " <<<" if offset == 0 else ""
                                        orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE]     offset={offset:+d} (clip1[{test_idx}]): diff={diff:.2f}{marker}")

                                # Check END context: trans[-1] should match clip2[gap_from_clip2 + context_from_clip2 - 1]
                                expected_clip2_last = gap_from_clip2 + context_from_clip2 - 1
                                orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE]   Comparing trans[-1] vs clip2 frames around index {expected_clip2_last}")
                                trans_last = transition_frames[-1].astype(float)
                                for offset in range(-2, 3):
                                    test_idx = expected_clip2_last + offset
                                    if 0 <= test_idx < len(end_all_frames):
                                        test_frame = end_all_frames[test_idx].astype(float)
                                        diff = np.abs(trans_last - test_frame).mean()
                                        marker = " <<<" if offset == 0 else ""
                                        orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE]     offset={offset:+d} (clip2[{test_idx}]): diff={diff:.2f}{marker}")

                                # Find best offset for both START and END
                                start_diffs = {}
                                for offset in range(-2, 3):
                                    test_idx = context_start_idx + offset
                                    if 0 <= test_idx < len(start_all_frames):
                                        diff = np.abs(trans_first - start_all_frames[test_idx].astype(float)).mean()
                                        start_diffs[offset] = diff
                                end_diffs = {}
                                for offset in range(-2, 3):
                                    test_idx = expected_clip2_last + offset
                                    if 0 <= test_idx < len(end_all_frames):
                                        diff = np.abs(trans_last - end_all_frames[test_idx].astype(float)).mean()
                                        end_diffs[offset] = diff

                                if start_diffs:
                                    best_start = min(start_diffs, key=start_diffs.get)
                                    orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE]   START best match: offset={best_start:+d} (diff={start_diffs[best_start]:.2f})")
                                if end_diffs:
                                    best_end = min(end_diffs, key=end_diffs.get)
                                    orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE]   END best match: offset={best_end:+d} (diff={end_diffs[best_end]:.2f})")

                                if start_diffs and end_diffs:
                                    best_start = min(start_diffs, key=start_diffs.get)
                                    best_end = min(end_diffs, key=end_diffs.get)
                                    if best_start != 0 or best_end != 0:
                                        orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE]   >>> ALIGNMENT ISSUE DETECTED AT SOURCE! start_offset={best_start:+d}, end_offset={best_end:+d}")
                                    else:
                                        orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE]   >>> Alignment looks correct at source (both offsets = 0)")
                        except (OSError, ValueError, RuntimeError) as e:
                            orchestrator_logger.debug(f"[OFFSET_DETECT_SOURCE] Task {task_id}: Error during offset detection: {e}")

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

                            # --- FPS (so stitch handler can resample clips to match) ---
                            "fps": start_fps,

                            # --- Final stitch guidance ---
                            # When stitching: trim clip1 end by gap_from_clip1, trim clip2 start by gap_from_clip2
                            # Then crossfade: clip1_trimmed[-ctx1:] with transition[0:ctx1]
                            #                 transition[-ctx2:] with clip2_trimmed[0:ctx2]
                            "mode": "replace" if replace_mode else "insert",
                        })

                    # --- 9. Concatenate Full Clips with Transition ---
                    orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Concatenating full clips with transition...")

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
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Blend frames: {blend_frames} (context={context_frame_count}, max_safe={max_safe_blend})")

                        # Use pre-calculated gap sizes (gap_from_clip1, gap_from_clip2 from step 4)
                        frames_to_remove_clip1 = gap_from_clip1  # 0 for INSERT mode
                        frames_to_keep_clip1 = start_frame_count - frames_to_remove_clip1

                        if replace_mode:
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode - removing {gap_from_clip1} gap frames from clip1")
                        else:
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: INSERT mode - keeping all frames from clip1")
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   Keeping {frames_to_keep_clip1}/{start_frame_count} frames from clip1")

                        # Use common frame extraction: frames 0 to (frames_to_keep_clip1 - 1)
                        extract_frame_range_to_video(
                            input_video_path=starting_video,
                            output_video_path=clip1_trimmed_path,
                            start_frame=0,
                            end_frame=frames_to_keep_clip1 - 1,
                            fps=start_fps)

                        # Clip2 trimming uses pre-calculated gap_from_clip2
                        frames_to_skip_clip2 = gap_from_clip2  # 0 for INSERT mode

                        if replace_mode:
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: REPLACE mode - skipping {gap_from_clip2} gap frames from clip2")
                        else:
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: INSERT mode - keeping all frames from clip2")

                        frames_remaining_clip2 = end_frame_count - frames_to_skip_clip2
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   Keeping {frames_remaining_clip2}/{end_frame_count} frames from clip2")

                        # Log net frame change summary
                        total_gap_removed = frames_to_remove_clip1 + frames_to_skip_clip2
                        # Transition = context + gap + context, but context regions overlap with clips via blend
                        # Effective new frames = gap_for_guide (the middle portion)
                        effective_frames_added = gap_for_guide
                        net_frame_change = effective_frames_added - total_gap_removed
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: === NET FRAME CHANGE ===")
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   Gap frames removed from clips: {total_gap_removed}")
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   Gap frames generated by VACE: {effective_frames_added}")
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}:   Net change: {net_frame_change:+d} frames")

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

                        # === FINAL STITCHING DIAGNOSTICS ===
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: === Final Stitching Plan ===")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   Clip1 trimmed: frames 0-{frames_to_keep_clip1-1} ({frames_to_keep_clip1} frames)")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   Transition: {actual_transition_frames} frames (context+gap+context)")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   Clip2 trimmed: frames {frames_to_skip_clip2}-end ({frames_remaining_clip2} frames)")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   Blend at each boundary: {blend_frames} frames")

                        # Calculate expected final frame count
                        # Stitch: clip1[:-blend] + crossfade(clip1[-blend:], trans[:blend]) + trans[blend:-blend] + crossfade(trans[-blend:], clip2[:blend]) + clip2[blend:]
                        # = (frames_to_keep_clip1 - blend_frames) + blend_frames + (actual_transition_frames - 2*blend_frames) + blend_frames + (frames_remaining_clip2 - blend_frames)
                        # = frames_to_keep_clip1 + actual_transition_frames + frames_remaining_clip2 - 2*blend_frames
                        expected_final_frames = frames_to_keep_clip1 + actual_transition_frames + frames_remaining_clip2 - 2 * blend_frames
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   Expected final frame count: {expected_final_frames}")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]     = {frames_to_keep_clip1} + {actual_transition_frames} + {frames_remaining_clip2} - 2*{blend_frames}")

                        # Compare to original
                        original_total = start_frame_count + end_frame_count
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   Original clips total: {original_total} ({start_frame_count} + {end_frame_count})")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   Net frame delta: {expected_final_frames - original_total:+d}")

                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Stitching videos with {blend_frames}-frame crossfade at each boundary")

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

                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Successfully stitched videos with crossfade blending")
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
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: === Final Output Verification ===")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   Expected frames: {expected_final_frames}")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   Actual frames: {final_actual_frames}")
                        orchestrator_logger.debug(f"[JOIN_CLIPS]   FPS: {final_actual_fps}")
                        if final_actual_frames and abs(final_actual_frames - expected_final_frames) > 3:
                            orchestrator_logger.debug(f"[JOIN_CLIPS]   \u26a0\ufe0f  FRAME COUNT MISMATCH: diff={final_actual_frames - expected_final_frames}")
                        elif final_actual_frames:
                            orchestrator_logger.debug(f"[JOIN_CLIPS]   \u2713 Frame count within tolerance")

                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Final video validated: {file_size} bytes")

                        # Extract poster image/thumbnail from the final video
                        poster_output_path = final_output_path.with_suffix('.jpg')
                        try:
                            # Extract first frame as poster
                            poster_frame_index = 0

                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Extracting poster image (first frame)")

                            if save_frame_from_video(
                                final_output_path,
                                poster_frame_index,
                                poster_output_path,
                                parsed_res_wh
                            ):
                                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Poster image saved: {poster_output_path}")
                            else:
                                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Warning: Failed to extract poster image")
                        except (OSError, ValueError, RuntimeError) as poster_error:
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Warning: Poster extraction failed: {poster_error}")

                        # Clean up temporary files (unless debug mode is enabled)
                        debug_mode = task_params_from_db.get("debug", False)
                        if debug_mode:
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Debug mode enabled - preserving intermediate files:")
                            orchestrator_logger.debug(f"[JOIN_CLIPS]   Clip1 trimmed: {clip1_trimmed_path}")
                            orchestrator_logger.debug(f"[JOIN_CLIPS]   Transition: {transition_video_path}")
                            orchestrator_logger.debug(f"[JOIN_CLIPS]   Clip2 trimmed: {clip2_trimmed_path}")
                            if vid2vid_source_video_path is not None:
                                orchestrator_logger.debug(f"[JOIN_CLIPS]   Vid2vid source: {vid2vid_source_video_path}")
                        else:
                            try:
                                clip1_trimmed_path.unlink()
                                clip2_trimmed_path.unlink()
                                Path(transition_video_path).unlink()  # Remove transition-only video
                                # Clean up vid2vid source video if it was created
                                if vid2vid_source_video_path is not None and vid2vid_source_video_path.exists():
                                    vid2vid_source_video_path.unlink()
                                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Cleaned up temporary files")
                            except OSError as cleanup_error:
                                orchestrator_logger.debug(f"[JOIN_CLIPS] Warning: Cleanup failed: {cleanup_error}")

                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Successfully created final joined video")
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Output: {final_output_path}")

                        # Add audio to final output if this is the last join and audio_url is provided
                        if is_last_join and audio_url:
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: This is the last join - adding audio...")

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
                                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u2705 Audio added successfully")

                                # Remove the silent version
                                try:
                                    final_output_path.unlink()
                                except OSError as e_unlink_silent:
                                    headless_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Could not remove silent video {final_output_path}: {e_unlink_silent}")

                                # Rename audio version to final path
                                result_with_audio.rename(final_output_path)
                                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Final output now includes audio")
                            else:
                                orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: \u26a0\ufe0f  Failed to add audio - returning silent video")
                        elif audio_url and not is_last_join:
                            orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: audio_url provided but not last join - skipping audio")

                        # Handle upload and get final DB location
                        final_db_location = upload_and_get_final_output_location(
                            local_file_path=final_output_path,
                            initial_db_location=initial_db_location)

                        return True, final_db_location

                    except (OSError, ValueError, RuntimeError) as concat_error:
                        error_msg = f"Failed to concatenate full clips: {concat_error}"
                        orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}", exc_info=True)
                        # Return the transition video as fallback
                        orchestrator_logger.debug(f"[JOIN_CLIPS] Task {task_id}: Returning transition video as fallback")
                        return True, transition_video_path

                elif status.status == "failed":
                    error_msg = status.error_message or "Generation failed without specific error message"
                    orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: Generation failed - {error_msg}")
                    return False, error_msg

                else:
                    # Still processing
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval

            # Timeout reached
            error_msg = f"Processing timeout after {max_wait_time} seconds"
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}")
            return False, error_msg

        except (RuntimeError, ValueError, OSError) as e:
            error_msg = f"Failed to submit/complete generation task: {e}"
            orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}", exc_info=True)
            return False, error_msg

    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        error_msg = f"Unexpected error in join_clips handler: {e}"
        orchestrator_logger.debug(f"[JOIN_CLIPS_ERROR] Task {task_id}: {error_msg}", exc_info=True)
        return False, error_msg
