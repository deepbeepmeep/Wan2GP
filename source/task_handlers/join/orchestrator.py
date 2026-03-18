"""
Join Clips Orchestrator - Main handler for sequentially joining video clips.

Takes a list of video clips and creates a chain of join_clips_child
tasks to progressively build them into a single seamless video.

__all__ is defined to ensure underscore-prefixed names are exported via import *.

Pattern:
    Input: [clip_A, clip_B, clip_C, clip_D]

    Creates:
        join_0: clip_A + clip_B -> AB.mp4 (no dependency)
        join_1: AB.mp4 + clip_C -> ABC.mp4 (depends on join_0)
        join_2: ABC.mp4 + clip_D -> ABCD.mp4 (depends on join_1)
"""

import json
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional

from source.core.db.task_polling import get_task_output_location_from_db
from source.core.params.task_result import TaskResult
from source.media.video import reverse_video
from source.core.log import orchestrator_logger
from source.utils.download_utils import download_video_if_url
from source.utils.output_paths import upload_intermediate_file_to_storage

from source.task_handlers.join.shared import (
    _check_existing_join_tasks,
    _extract_join_settings_from_payload)
from source.task_handlers.join.clip_validator import validate_clip_frames_for_join
from source.task_handlers.join.task_builder import (
    _create_join_chain_tasks,
    _create_parallel_join_tasks)
from source.task_handlers.join.vlm_enhancement import (
    _extract_boundary_frames_for_vlm,
    _generate_vlm_prompts_for_joins)

__all__ = ["_get_video_resolution", "_handle_join_clips_orchestrator_task", "_parse_join_orchestrator_inputs"]


def _parse_join_orchestrator_inputs(
    task_params_from_db: dict,
    *,
    orchestrator_task_id_str: str,
):
    if "orchestrator_details" not in task_params_from_db:
        return TaskResult.failed("orchestrator_details missing")

    orchestrator_payload = dict(task_params_from_db["orchestrator_details"])
    orchestrator_payload.setdefault("orchestrator_task_id_ref", orchestrator_task_id_str)
    orchestrator_payload.setdefault("clip_list", [])
    orchestrator_payload.setdefault("loop_first_clip", False)
    run_id = orchestrator_payload.get("run_id")
    if not run_id:
        return TaskResult.failed("run_id is required")
    return (
        orchestrator_payload,
        orchestrator_payload["clip_list"],
        run_id,
        bool(orchestrator_payload["loop_first_clip"]),
    )

def _get_video_resolution(video_path: str | Path) -> Tuple[int, int] | None:
    """
    Get video resolution (width, height) using ffprobe.

    Returns:
        (width, height) tuple, or None if detection fails
    """
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            orchestrator_logger.debug(f"[GET_RESOLUTION] ffprobe failed: {result.stderr}")
            return None
        width_str, height_str = result.stdout.strip().split(',')
        return (int(width_str), int(height_str))
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        orchestrator_logger.debug(f"[GET_RESOLUTION] Error detecting resolution: {e}")
        return None

def _handle_join_clips_orchestrator_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None) -> Tuple[bool, str]:
    """
    Handle join_clips_orchestrator task - creates chained join_clips_child tasks.

    Args:
        task_params_from_db: Task parameters containing orchestrator_details
        main_output_dir_base: Base output directory
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization

    Returns:
        (success: bool, message: str)
    """
    orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Starting orchestrator task {orchestrator_task_id_str}")

    try:
        # === 1. PARSE ORCHESTRATOR PAYLOAD ===
        if 'orchestrator_details' not in task_params_from_db:
            orchestrator_logger.debug("[JOIN_ORCHESTRATOR] ERROR: orchestrator_details missing")
            return False, "orchestrator_details missing"

        orchestrator_payload = task_params_from_db['orchestrator_details']
        orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Orchestrator payload keys: {list(orchestrator_payload.keys())}")

        # Extract required fields
        clip_list = orchestrator_payload.get("clip_list", [])
        run_id = orchestrator_payload.get("run_id")
        loop_first_clip = orchestrator_payload.get("loop_first_clip", False)

        # === DYNAMIC CLIP_LIST FROM SEGMENT TASKS ===
        segment_task_ids = orchestrator_payload.get("segment_task_ids", [])
        if segment_task_ids and not clip_list:
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Building clip_list from {len(segment_task_ids)} segment task outputs")

            built_clip_list = []
            for i, task_id in enumerate(segment_task_ids):
                output_url = get_task_output_location_from_db(task_id)
                if not output_url:
                    return False, f"Segment task {task_id} has no output_location (may not be complete)"

                if output_url.startswith('{'):
                    try:
                        output_data = json.loads(output_url)
                        output_url = output_data.get("output_location") or output_data.get("url") or output_url
                    except json.JSONDecodeError:
                        pass

                built_clip_list.append({
                    "url": output_url,
                    "name": f"Segment {i + 1}"
                })
                orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR]   Segment {i}: {output_url[:80]}...")

            clip_list = built_clip_list
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Built clip_list with {len(clip_list)} clips from segment outputs")

        if not run_id:
            return False, "run_id is required"

        # Handle loop_first_clip: reverse the first clip and use it as the second clip
        if loop_first_clip:
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] loop_first_clip=True - will reverse first clip to create looping effect")

            if not clip_list or len(clip_list) < 1:
                return False, "clip_list must contain at least 1 clip when loop_first_clip=True"

            loop_temp_dir = Path(main_output_dir_base) / f"join_clips_run_{run_id}" / "loop_temp"
            loop_temp_dir.mkdir(parents=True, exist_ok=True)

            first_clip = clip_list[0]
            first_clip_url = first_clip.get("url")
            first_clip_name = first_clip.get("name", "clip_0")

            if not first_clip_url:
                return False, "First clip in clip_list is missing 'url' field"

            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Downloading first clip for reversal: {first_clip_url[:80]}...")
            local_first_clip_path = download_video_if_url(
                first_clip_url,
                download_target_dir=loop_temp_dir,
                task_id_for_logging=orchestrator_task_id_str,
                descriptive_name="first_clip_for_loop"
            )

            if not local_first_clip_path or not Path(local_first_clip_path).exists():
                return False, f"Failed to download first clip: {first_clip_url}"

            reversed_clip_path = loop_temp_dir / f"{first_clip_name}_reversed.mp4"
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Reversing first clip...")

            reversed_path = reverse_video(
                local_first_clip_path,
                reversed_clip_path)

            if not reversed_path or not reversed_path.exists():
                return False, f"Failed to reverse first clip: {local_first_clip_path}"

            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] \u2705 Created reversed clip locally: {reversed_path}")

            reversed_filename = f"{first_clip_name}_reversed.mp4"
            reversed_url = upload_intermediate_file_to_storage(
                local_file_path=reversed_path,
                task_id=orchestrator_task_id_str,
                filename=reversed_filename)

            if not reversed_url:
                orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] \u26a0\ufe0f  Upload failed, using local path (may fail on multi-worker)")
                reversed_url = str(reversed_path.resolve())
            else:
                orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] \u2705 Reversed clip uploaded: {reversed_url}")

            reversed_clip_dict = {
                "url": reversed_url,
                "name": f"{first_clip_name}_reversed"
            }

            clip_list = [first_clip, reversed_clip_dict]
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] clip_list overridden: [{first_clip_name}, {reversed_clip_dict['name']}]")

        if not clip_list or len(clip_list) < 2:
            return False, "clip_list must contain at least 2 clips"

        num_joins = len(clip_list) - 1
        orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Processing {len(clip_list)} clips = {num_joins} join tasks")

        # === EARLY IDEMPOTENCY CHECK (before expensive VLM work) ===
        idempotency_check = _check_existing_join_tasks(orchestrator_task_id_str, num_joins)
        if idempotency_check is not None:
            return idempotency_check

        # Extract join settings using shared helper
        join_settings = _extract_join_settings_from_payload(orchestrator_payload)
        per_join_settings = orchestrator_payload.get("per_join_settings", [])
        output_base_dir = orchestrator_payload.get("output_base_dir", str(main_output_dir_base.resolve()))

        # Create run-specific output directory
        current_run_output_dir = Path(output_base_dir) / f"join_clips_run_{run_id}"
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Run output directory: {current_run_output_dir}")

        # === VALIDATE CLIP FRAME COUNTS (optional, enabled by default) ===
        skip_validation = orchestrator_payload.get("skip_frame_validation", False)
        if not skip_validation:
            validation_temp_dir = current_run_output_dir / "validation_temp"
            validation_temp_dir.mkdir(parents=True, exist_ok=True)

            is_valid, validation_message, frame_counts = validate_clip_frames_for_join(
                clip_list=clip_list,
                gap_frame_count=join_settings.get("gap_frame_count", 53),
                context_frame_count=join_settings.get("context_frame_count", 8),
                replace_mode=join_settings.get("replace_mode", False),
                temp_dir=validation_temp_dir,
                orchestrator_task_id=orchestrator_task_id_str)

            if not is_valid:
                orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] VALIDATION FAILED: {validation_message}")
                return False, f"Clip frame validation failed: {validation_message}"

            if validation_message:
                orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] {validation_message}")

            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Clip frame validation complete, frame counts: {frame_counts}")
        else:
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Frame validation skipped (skip_frame_validation=True)")

        # === DETECT RESOLUTION FROM INPUT VIDEO (when use_input_video_resolution=True) ===
        if join_settings.get("use_input_video_resolution", False):
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] use_input_video_resolution=True, detecting resolution from first clip...")

            first_clip_url = clip_list[0].get("url")
            if first_clip_url:
                resolution_temp_dir = current_run_output_dir / "resolution_temp"
                resolution_temp_dir.mkdir(parents=True, exist_ok=True)

                local_first_clip = download_video_if_url(
                    first_clip_url,
                    download_target_dir=resolution_temp_dir,
                    task_id_for_logging=orchestrator_task_id_str,
                    descriptive_name="detect_resolution"
                )

                if local_first_clip and Path(local_first_clip).exists():
                    detected_res = _get_video_resolution(local_first_clip)
                    if detected_res:
                        join_settings["resolution"] = list(detected_res)
                        orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] \u2713 Detected resolution from input video: {detected_res}")
                    else:
                        orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] \u26a0 Could not detect resolution, segments will detect from frames")
                else:
                    orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] \u26a0 Could not download first clip for resolution detection")

        # === VLM PROMPT ENHANCEMENT (optional) ===
        enhance_prompt = orchestrator_payload.get("enhance_prompt", False)
        vlm_enhanced_prompts: List[Optional[str]] = [None] * num_joins

        if enhance_prompt:
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] enhance_prompt=True, generating VLM-enhanced prompts for {num_joins} joins")

            vlm_device = orchestrator_payload.get("vlm_device", "cuda")
            vlm_temp_dir = current_run_output_dir / "vlm_temp"
            vlm_temp_dir.mkdir(parents=True, exist_ok=True)
            base_prompt = join_settings.get("prompt", "")

            try:
                orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Extracting 4 frames per join from {len(clip_list)} clips...")
                image_quads = _extract_boundary_frames_for_vlm(
                    clip_list=clip_list,
                    temp_dir=vlm_temp_dir,
                    orchestrator_task_id=orchestrator_task_id_str)

                orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Running VLM batch on {len(image_quads)} quads (4 images each)...")
                vlm_enhanced_prompts = _generate_vlm_prompts_for_joins(
                    image_quads=image_quads,
                    base_prompt=base_prompt,
                    vlm_device=vlm_device)

                valid_count = sum(1 for p in vlm_enhanced_prompts if p is not None)
                orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] VLM enhancement complete: {valid_count}/{num_joins} prompts generated")

            except (RuntimeError, ValueError, OSError) as vlm_error:
                orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] VLM enhancement failed, using base prompts: {vlm_error}", exc_info=True)
                vlm_enhanced_prompts = [None] * num_joins
        else:
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] enhance_prompt=False, using base prompt for all joins")

        # === CREATE JOIN TASKS ===
        use_parallel = orchestrator_payload.get("use_parallel_joins", True)

        parent_generation_id = (
            task_params_from_db.get("parent_generation_id")
            or orchestrator_payload.get("parent_generation_id")
        )

        if use_parallel:
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Using PARALLEL pattern (transitions in parallel + final stitch)")
            success, message = _create_parallel_join_tasks(
                clip_list=clip_list,
                run_id=run_id,
                join_settings=join_settings,
                per_join_settings=per_join_settings,
                vlm_enhanced_prompts=vlm_enhanced_prompts,
                current_run_output_dir=current_run_output_dir,
                orchestrator_task_id_str=orchestrator_task_id_str,
                orchestrator_project_id=orchestrator_project_id,
                orchestrator_payload=orchestrator_payload,
                parent_generation_id=parent_generation_id)
        else:
            orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] Using CHAIN pattern (legacy sequential)")
            success, message = _create_join_chain_tasks(
                clip_list=clip_list,
                run_id=run_id,
                join_settings=join_settings,
                per_join_settings=per_join_settings,
                vlm_enhanced_prompts=vlm_enhanced_prompts,
                current_run_output_dir=current_run_output_dir,
                orchestrator_task_id_str=orchestrator_task_id_str,
                orchestrator_project_id=orchestrator_project_id,
                orchestrator_payload=orchestrator_payload,
                parent_generation_id=parent_generation_id)

        orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] {message}")
        return success, message

    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        msg = f"Failed during join orchestration: {e}"
        orchestrator_logger.debug(f"[JOIN_ORCHESTRATOR] ERROR: {msg}", exc_info=True)
        return False, msg


# Public alias for cross-module use.
handle_join_clips_orchestrator_task = _handle_join_clips_orchestrator_task
