"""
Inpaint Frames - Regenerate a range of frames within a single video

This module provides functionality to replace/regenerate a specific range of frames
within a video clip using VACE generation with guide and mask videos.

Use cases:
- Fix corrupted frames in a video
- Regenerate a problematic section
- Change content in a specific time range
- Smooth out jarring transitions within a clip

The task extracts context frames before and after the target range, then uses
VACE to generate smooth replacement frames that blend naturally with the surrounding content.
"""

import time
from pathlib import Path
from typing import Tuple

from ..utils import (
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    get_video_frame_count_and_fps
)
from ..media.video import (
    extract_frames_from_video)
from source.media.video.vace_frame_utils import (
    create_guide_and_mask_for_generation,
    prepare_vace_generation_params,
    validate_frame_range
)
from source.core.log import task_logger
from source.core.params.task_result import normalize_task_result

def _handle_inpaint_frames_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str,
    task_queue = None) -> Tuple[bool, str]:
    """
    Handle inpaint_frames task: regenerate a range of frames within a single video.

    Args:
        task_params_from_db: Task parameters including:
            - video_path: Path to video clip
            - inpaint_start_frame: Start frame index (inclusive)
            - inpaint_end_frame: End frame index (exclusive)
            - context_frame_count: Frames to preserve on each side (default: 8)
            - prompt: Generation prompt for the inpainted frames
            - model: Optional model override (defaults to wan_2_2_vace_lightning_baseline_2_2_2)
            - Other standard VACE parameters
        main_output_dir_base: Base output directory
        task_id: Task ID for logging and status updates
        task_queue: HeadlessTaskQueue instance for generation

    Returns:
        Tuple of (success: bool, output_path_or_message: str)
    """
    try:
        # --- 1. Extract and Validate Parameters ---
        video_path = task_params_from_db.get("video_path")
        inpaint_start_frame = task_params_from_db.get("inpaint_start_frame")
        inpaint_end_frame = task_params_from_db.get("inpaint_end_frame")
        context_frame_count = task_params_from_db.get("context_frame_count", 8)
        prompt = task_params_from_db.get("prompt", "")

        # Validate required parameters
        if not video_path:
            error_msg = "video_path is required"
            task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        if inpaint_start_frame is None:
            error_msg = "inpaint_start_frame is required"
            task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        if inpaint_end_frame is None:
            error_msg = "inpaint_end_frame is required"
            task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        # Convert to Path object and validate existence
        video = Path(video_path)

        if not video.exists():
            error_msg = f"Video not found: {video_path}"
            task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        # Validate task queue
        if task_queue is None:
            error_msg = "task_queue is required for inpaint_frames"
            task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        # --- 2. Extract Video Properties and All Frames ---
        try:
            total_frame_count, video_fps = get_video_frame_count_and_fps(str(video))

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Failed to extract video properties: {e}"
            task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        # Use FPS from task params, or default to video FPS
        target_fps = task_params_from_db.get("fps", video_fps)
        task_logger.debug_block(
            "SETUP",
            {
                "task": task_id,
                "video": video,
                "video_frames": total_frame_count,
                "video_fps": video_fps,
                "target_fps": target_fps,
                "inpaint_range": (inpaint_start_frame, inpaint_end_frame),
                "context_frame_count": context_frame_count,
            },
            task_id=task_id,
        )

        # Validate frame range has sufficient context
        is_valid, validation_error = validate_frame_range(
            total_frame_count=total_frame_count,
            start_frame=inpaint_start_frame,
            end_frame=inpaint_end_frame,
            context_frame_count=context_frame_count,
            task_id=task_id)

        if not is_valid:
            task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {validation_error}")
            return False, validation_error

        # --- 3. Extract All Frames from Video ---
        try:
            all_frames = extract_frames_from_video(str(video))
            if not all_frames or len(all_frames) != total_frame_count:
                error_msg = f"Failed to extract frames: expected {total_frame_count}, got {len(all_frames) if all_frames else 0}"
                task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {error_msg}")
                return False, error_msg

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Failed to extract frames from video: {e}"
            task_logger.error(error_msg, task_id=task_id, exc_info=True)
            return False, error_msg

        # --- 4. Extract Context Frames ---
        # Context before: [start - context_count : start]
        # Context after: [end : end + context_count]
        context_start_idx = inpaint_start_frame - context_frame_count
        context_end_idx = inpaint_end_frame + context_frame_count

        context_before = all_frames[context_start_idx:inpaint_start_frame]
        context_after = all_frames[inpaint_end_frame:context_end_idx]
        gap_count = inpaint_end_frame - inpaint_start_frame

        # Get resolution from first frame or task params
        first_frame = all_frames[0]
        frame_height, frame_width = first_frame.shape[:2]

        # Allow resolution override from task params
        if "resolution" in task_params_from_db:
            resolution_list = task_params_from_db["resolution"]
            parsed_res_wh = (resolution_list[0], resolution_list[1])
        else:
            parsed_res_wh = (frame_width, frame_height)
        task_logger.debug_block(
            "SETUP",
            {
                "context_before": len(context_before),
                "gap_frames": gap_count,
                "context_after": len(context_after),
                "resolution": parsed_res_wh,
            },
            task_id=task_id,
        )

        # --- 5. Build Guide and Mask Videos (using shared helper) ---
        # Create working directory
        inpaint_dir = main_output_dir_base / "inpaint_frames" / task_id
        inpaint_dir.mkdir(parents=True, exist_ok=True)

        try:
            created_guide_video, created_mask_video, guide_frame_count = create_guide_and_mask_for_generation(
                context_frames_before=context_before,
                context_frames_after=context_after,
                gap_frame_count=gap_count,
                resolution_wh=parsed_res_wh,
                fps=target_fps,
                output_dir=inpaint_dir,
                task_id=task_id,
                filename_prefix="inpaint")
        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Failed to create guide/mask videos: {e}"
            task_logger.error(error_msg, task_id=task_id, exc_info=True)
            return False, error_msg

        expected_total_frames = context_frame_count * 2 + gap_count
        if guide_frame_count != expected_total_frames:
            task_logger.debug_anomaly(
                "INPAINT_FRAMES",
                f"guide/mask total frame count {guide_frame_count} differs from expected {expected_total_frames}",
                task_id=task_id,
            )
        total_frames = guide_frame_count

        # --- 6. Prepare Generation Parameters (using shared helper) ---
        # Determine model (default to Lightning baseline for fast generation)
        model = task_params_from_db.get("model", "wan_2_2_vace_lightning_baseline_2_2_2")

        # Ensure prompt is valid
        prompt = ensure_valid_prompt(prompt)
        negative_prompt = ensure_valid_negative_prompt(
            task_params_from_db.get("negative_prompt", "")
        )

        # Use shared helper to prepare standardized VACE parameters
        generation_params = prepare_vace_generation_params(
            guide_video_path=created_guide_video,
            mask_video_path=created_mask_video,
            total_frames=total_frames,
            resolution_wh=parsed_res_wh,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            seed=task_params_from_db.get("seed", -1),
            task_params=task_params_from_db  # Pass through for optional param merging
        )

        task_logger.debug_block(
            "SETUP",
            {
                "model": model,
                "video_length": total_frames,
                "resolution": parsed_res_wh,
                "gap_frames": gap_count,
                "seed": task_params_from_db.get("seed", -1),
            },
            task_id=task_id,
        )

        # --- 7. Submit to Generation Queue ---
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
            max_wait_time = 600  # 10 minute timeout
            wait_interval = 2  # Check every 2 seconds
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                status = task_queue.get_task_status(task_id)

                if status is None:
                    error_msg = "Task status became None during processing"
                    task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {error_msg}")
                    return False, error_msg

                if status.status == "completed":
                    output_path = status.result_path
                    processing_time = status.processing_time or 0
                    task_logger.debug_block(
                        "OUTPUT",
                        {
                            "submitted_task_id": submitted_task_id,
                            "processing_time_s": round(processing_time, 1),
                            "output": output_path,
                            "frames_processed": gap_count,
                        },
                        task_id=task_id,
                    )
                    return True, output_path

                elif status.status == "failed":
                    error_msg = status.error_message or "Generation failed without specific error message"
                    task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: Generation failed - {error_msg}")
                    return False, error_msg

                else:
                    # Still processing
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval

            # Timeout reached
            error_msg = f"Processing timeout after {max_wait_time} seconds"
            task_logger.debug_anomaly("INPAINT_FRAMES_ERROR", f"Task {task_id}: {error_msg}")
            return False, error_msg

        except (RuntimeError, ValueError, OSError) as e:
            error_msg = f"Failed to submit/complete generation task: {e}"
            task_logger.error(error_msg, task_id=task_id, exc_info=True)
            return False, error_msg

    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        error_msg = f"Unexpected error in inpaint_frames handler: {e}"
        task_logger.error(error_msg, task_id=task_id, exc_info=True)
        return False, error_msg


def handle_inpaint_frames_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str,
    task_queue=None,
):
    """Typed public entrypoint that normalizes the legacy tuple result."""
    return normalize_task_result(
        _handle_inpaint_frames_task(
            task_params_from_db=task_params_from_db,
            main_output_dir_base=main_output_dir_base,
            task_id=task_id,
            task_queue=task_queue,
        )
    )
