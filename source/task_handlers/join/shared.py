"""
Shared orchestrator helpers for join operations.

Used by both join_clips_orchestrator and edit_video_orchestrator.
Provides cancellation checks, settings extraction, and idempotency checks.
Also documents that enqueue gating active to avoid duplicate child creation.
"""

import json
from typing import Optional

from source.core.db.dependencies.task_dependencies_children import (
    cancel_orchestrator_children,
    get_task_current_status,
)
from source.core.db.task_dependencies import get_orchestrator_child_tasks
from source.core.log import task_logger
from source.core.params.task_result import TaskResult

__all__ = [
    "_check_orchestrator_cancelled",
    "_extract_join_settings_from_payload",
    "_check_existing_join_tasks",
    "_check_parallel_join_tasks",
    "_check_chain_join_tasks",
    "check_orchestrator_cancelled",
    "extract_join_settings_from_payload",
    "check_existing_join_tasks",
]


def _partial_join_child_state_result(prefix: str) -> TaskResult:
    return TaskResult.orchestrating(f"{prefix} Partial join child state detected; waiting for existing children to settle.")


def _check_parallel_join_tasks(
    orchestrator_task_id_str: str,
    num_joins: int,
    existing_joins: list[dict],
    existing_final_stitch: list[dict],
) -> Optional[TaskResult]:
    if not existing_joins and not existing_final_stitch:
        return None
    if len(existing_joins) < num_joins or not existing_final_stitch:
        return _partial_join_child_state_result("Parallel:")
    return None


def _check_chain_join_tasks(
    orchestrator_task_id_str: str,
    num_joins: int,
    existing_joins: list[dict],
    existing_final_stitch: list[dict],
) -> Optional[TaskResult]:
    _ = orchestrator_task_id_str, existing_final_stitch
    if existing_joins and len(existing_joins) < num_joins:
        return _partial_join_child_state_result("Chain:")
    return None


def _check_orchestrator_cancelled(orchestrator_task_id: str, context_msg: str, **_kwargs) -> str | None:
    """Check if orchestrator was cancelled and cancel children if so.

    Returns an error message string if cancelled, None if still active.
    """
    status = get_task_current_status(orchestrator_task_id)
    if status and status.lower() in ('cancelled', 'canceled'):
        task_logger.debug_anomaly("CANCELLATION", f"Join orchestrator {orchestrator_task_id} was cancelled - {context_msg}", task_id=orchestrator_task_id)
        cancel_orchestrator_children(orchestrator_task_id, reason="Orchestrator cancelled by user")
        return f"Orchestrator cancelled: {context_msg}"
    return None


def _extract_join_settings_from_payload(orchestrator_payload: dict) -> dict:
    """
    Extract standardized join settings from an orchestrator payload.

    Used by both join_clips_orchestrator and edit_video_orchestrator.

    Args:
        orchestrator_payload: The orchestrator_details dict

    Returns:
        Dict of join settings for join_clips_segment tasks
    """
    use_input_res = orchestrator_payload.get("use_input_video_resolution", False)

    return {
        "context_frame_count": orchestrator_payload.get("context_frame_count", 8),
        "gap_frame_count": orchestrator_payload.get("gap_frame_count", 53),
        "replace_mode": orchestrator_payload.get("replace_mode", False),
        "prompt": orchestrator_payload.get("prompt", "smooth transition"),
        "negative_prompt": orchestrator_payload.get("negative_prompt", ""),
        "model": orchestrator_payload.get("model", "wan_2_2_vace_lightning_baseline_2_2_2"),
        "aspect_ratio": orchestrator_payload.get("aspect_ratio"),
        "resolution": None if use_input_res else orchestrator_payload.get("resolution"),
        "use_input_video_resolution": use_input_res,
        "fps": orchestrator_payload.get("fps"),
        "use_input_video_fps": orchestrator_payload.get("use_input_video_fps", False),
        "phase_config": orchestrator_payload.get("phase_config"),
        "num_inference_steps": orchestrator_payload.get("num_inference_steps"),
        "guidance_scale": orchestrator_payload.get("guidance_scale"),
        "seed": orchestrator_payload.get("seed", -1),
        # LoRA parameters
        "additional_loras": orchestrator_payload.get("additional_loras", {}),
        # Keep bridging image param
        "keep_bridging_images": orchestrator_payload.get("keep_bridging_images", False),
        # Vid2vid initialization for replace mode
        "vid2vid_init_strength": orchestrator_payload.get("vid2vid_init_strength"),
        # Audio to add to final output (only used by last join)
        "audio_url": orchestrator_payload.get("audio_url"),
    }


def _check_existing_join_tasks(
    orchestrator_task_id_str: str,
    num_joins: int,
    **_kwargs
) -> Optional[TaskResult]:
    """
    Check for existing child tasks (idempotency check).

    Handles both patterns:
    - Chain pattern: num_joins join_clips_segment tasks
    - Parallel pattern: num_joins join_clips_segment (transitions) + 1 join_final_stitch

    Returns:
        None if no existing tasks or should proceed with creation.
        TaskResult if should return early (complete/failed/in-progress).
    """
    existing_child_tasks = get_orchestrator_child_tasks(orchestrator_task_id_str)
    existing_joins = existing_child_tasks.get('join_clips_segment', [])
    existing_final_stitch = existing_child_tasks.get('join_final_stitch', [])

    # Determine which pattern was used
    is_parallel_pattern = len(existing_final_stitch) > 0

    if not existing_joins and not existing_final_stitch:
        return None

    # Check completion status helper
    def is_complete(task):
        return (task.get('status', '') or '').lower() == 'complete'

    def is_terminal_failure(task):
        status = task.get('status', '').lower()
        return status in ('failed', 'cancelled', 'canceled', 'error')

    if is_parallel_pattern:
        # === PARALLEL PATTERN ===
        partial_result = _check_parallel_join_tasks(
            orchestrator_task_id_str=orchestrator_task_id_str,
            num_joins=num_joins,
            existing_joins=existing_joins,
            existing_final_stitch=existing_final_stitch,
        )
        if partial_result is not None:
            return partial_result

        all_tasks = existing_joins + existing_final_stitch
        any_failed = any(is_terminal_failure(t) for t in all_tasks)

        if any_failed:
            failed_tasks = [t for t in all_tasks if is_terminal_failure(t)]
            error_msg = f"{len(failed_tasks)} task(s) failed/cancelled"
            task_logger.debug_anomaly("JOIN_CORE", f"FAILED: {error_msg}", task_id=orchestrator_task_id_str)
            return TaskResult.failed(error_msg)

        if existing_final_stitch and is_complete(existing_final_stitch[0]):
            final_stitch = existing_final_stitch[0]
            final_output = final_stitch.get('output_location', 'Completed via idempotency')
            return TaskResult.orchestrator_complete(output_path=final_output, thumbnail_url="")

        trans_complete = sum(1 for j in existing_joins if is_complete(j))
        stitch_status = "complete" if existing_final_stitch and is_complete(existing_final_stitch[0]) else "pending"
        return TaskResult.orchestrating(f"Parallel: {trans_complete}/{num_joins} transitions complete, stitch: {stitch_status}")

    else:
        # === CHAIN PATTERN (legacy) ===
        partial_result = _check_chain_join_tasks(
            orchestrator_task_id_str=orchestrator_task_id_str,
            num_joins=num_joins,
            existing_joins=existing_joins,
            existing_final_stitch=existing_final_stitch,
        )
        if partial_result is not None:
            return partial_result

        all_joins_complete = all(is_complete(join) for join in existing_joins)
        any_join_failed = any(is_terminal_failure(join) for join in existing_joins)

        if any_join_failed:
            failed_joins = [j for j in existing_joins if is_terminal_failure(j)]
            error_msg = f"{len(failed_joins)} join task(s) failed/cancelled"
            task_logger.debug_anomaly("JOIN_CORE", f"FAILED: {error_msg}", task_id=orchestrator_task_id_str)
            return TaskResult.failed(error_msg)

        if all_joins_complete:
            def get_join_index(task):
                params = task.get('task_params', {})
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except (json.JSONDecodeError, ValueError):
                        return 0
                return params.get('join_index', 0)

            sorted_joins = sorted(existing_joins, key=get_join_index)
            final_join = sorted_joins[-1]
            final_output = final_join.get('output_location', 'Completed via idempotency')

            final_params = final_join.get('task_params', {})
            if isinstance(final_params, str):
                try:
                    final_params = json.loads(final_params)
                except (json.JSONDecodeError, ValueError):
                    final_params = {}

            final_thumbnail = final_params.get('thumbnail_url', '')

            return TaskResult.orchestrator_complete(output_path=final_output, thumbnail_url=final_thumbnail)

        complete_count = sum(1 for j in existing_joins if is_complete(j))
        return TaskResult.orchestrating(f"Join tasks in progress: {complete_count}/{num_joins} complete")


# Public aliases for cross-module use.
check_orchestrator_cancelled = _check_orchestrator_cancelled
extract_join_settings_from_payload = _extract_join_settings_from_payload
check_existing_join_tasks = _check_existing_join_tasks
