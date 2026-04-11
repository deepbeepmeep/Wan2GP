"""
Join task creation for chain and parallel patterns.

Creates join_clips_segment and join_final_stitch tasks in the database,
supporting both the legacy sequential chain pattern and the newer parallel pattern.
"""

from pathlib import Path
from typing import Tuple, List, Optional

from source.core.db.task_completion import add_task_to_db
from source.core.log import orchestrator_logger
from source.task_handlers.join.shared import _check_orchestrator_cancelled

__all__ = [
    "_create_join_chain_tasks",
    "_create_parallel_join_tasks",
    "create_join_chain_tasks",
    "create_parallel_join_tasks",
]

def _create_join_chain_tasks(
    clip_list: List[dict],
    run_id: str,
    join_settings: dict,
    per_join_settings: List[dict],
    vlm_enhanced_prompts: List[Optional[str]],
    current_run_output_dir: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
    orchestrator_payload: dict,
    parent_generation_id: str | None) -> Tuple[bool, str]:
    """
    Core logic: Create chained join_clips_segment tasks (LEGACY - sequential pattern).

    DEPRECATED: Use _create_parallel_join_tasks for better quality (avoids re-encoding).

    Args:
        clip_list: List of clip dicts with 'url' and optional 'name' keys
        run_id: Unique run identifier
        join_settings: Base settings for all join tasks
        per_join_settings: Per-join overrides (list, one per join)
        vlm_enhanced_prompts: VLM-generated prompts (or None for each join)
        current_run_output_dir: Output directory for this run
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization
        orchestrator_payload: Full orchestrator payload for reference
        parent_generation_id: Parent generation ID for variant linking

    Returns:
        (success: bool, message: str)
    """
    num_joins = len(clip_list) - 1

    if num_joins < 1:
        return False, "clip_list must contain at least 2 clips"

    previous_join_task_id = None
    joins_created = 0

    for idx in range(num_joins):
        clip_start = clip_list[idx]
        clip_end = clip_list[idx + 1]

        # Merge global settings with per-join overrides
        task_join_settings = join_settings.copy()
        if idx < len(per_join_settings):
            task_join_settings.update(per_join_settings[idx])

        # Apply VLM-enhanced prompt if available (overrides base prompt)
        if idx < len(vlm_enhanced_prompts) and vlm_enhanced_prompts[idx] is not None:
            task_join_settings["prompt"] = vlm_enhanced_prompts[idx]

        # Build join payload
        join_payload = {
            "orchestrator_task_id_ref": orchestrator_task_id_str,
            "orchestrator_run_id": run_id,
            "project_id": orchestrator_project_id,
            "join_index": idx,
            "is_first_join": (idx == 0),
            "is_last_join": (idx == num_joins - 1),
            "child_order": idx,
            "skip_generation": True,
            "starting_video_path": clip_start.get("url") if idx == 0 else None,
            "ending_video_path": clip_end.get("url"),
            **task_join_settings,
            "current_run_base_output_dir": str(current_run_output_dir.resolve()),
            "join_output_dir": str((current_run_output_dir / f"join_{idx}").resolve()),
            "orchestrator_details": orchestrator_payload,
        }

        # === CANCELLATION CHECK ===
        cancel_msg = _check_orchestrator_cancelled(
            orchestrator_task_id_str,
            f"aborting join creation at index {idx} ({joins_created} joins already created)")
        if cancel_msg:
            return False, cancel_msg

        actual_db_row_id = add_task_to_db(
            task_payload=join_payload,
            task_type_str="join_clips_segment",
            dependant_on=previous_join_task_id
        )

        previous_join_task_id = actual_db_row_id
        joins_created += 1

    # === Create join_final_stitch that depends on the last join ===
    context_frame_count = join_settings.get("context_frame_count", 8)

    final_stitch_payload = {
        "orchestrator_task_id_ref": orchestrator_task_id_str,
        "orchestrator_run_id": run_id,
        "project_id": orchestrator_project_id,
        "parent_generation_id": parent_generation_id,
        "clip_list": clip_list,
        "transition_task_ids": [previous_join_task_id],
        "chain_mode": True,
        "blend_frames": min(context_frame_count, 15),
        "fps": join_settings.get("fps") or orchestrator_payload.get("fps", 16),
        "audio_url": orchestrator_payload.get("audio_url"),
        "current_run_base_output_dir": str(current_run_output_dir.resolve()),
    }

    cancel_msg = _check_orchestrator_cancelled(
        orchestrator_task_id_str,
        f"aborting before final stitch ({joins_created} joins cancelled)")
    if cancel_msg:
        return False, cancel_msg

    final_stitch_task_id = add_task_to_db(
        task_payload=final_stitch_payload,
        task_type_str="join_final_stitch",
        dependant_on=previous_join_task_id
    )

    return True, f"Successfully enqueued {joins_created} chain joins + 1 final stitch for run {run_id}"

def _create_parallel_join_tasks(
    clip_list: List[dict],
    run_id: str,
    join_settings: dict,
    per_join_settings: List[dict],
    vlm_enhanced_prompts: List[Optional[str]],
    current_run_output_dir: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
    orchestrator_payload: dict,
    parent_generation_id: str | None) -> Tuple[bool, str]:
    """
    Create parallel join tasks with a final stitch (NEW - parallel pattern).

    This pattern avoids quality loss from re-encoding:
    1. Create N-1 transition tasks in parallel (no dependencies between them)
    2. Create a single join_final_stitch task that depends on ALL transition tasks

    Args:
        clip_list: List of clip dicts with 'url' and optional 'name' keys
        run_id: Unique run identifier
        join_settings: Base settings for all join tasks
        per_join_settings: Per-join overrides (list, one per join)
        vlm_enhanced_prompts: VLM-generated prompts (or None for each join)
        current_run_output_dir: Output directory for this run
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization
        orchestrator_payload: Full orchestrator payload for reference
        parent_generation_id: Parent generation ID for variant linking

    Returns:
        (success: bool, message: str)
    """
    num_joins = len(clip_list) - 1

    if num_joins < 1:
        return False, "clip_list must contain at least 2 clips"

    transition_task_ids = []

    # === Phase 1: Create transition tasks in parallel (no dependencies) ===
    for idx in range(num_joins):
        clip_start = clip_list[idx]
        clip_end = clip_list[idx + 1]

        task_join_settings = join_settings.copy()
        if idx < len(per_join_settings):
            task_join_settings.update(per_join_settings[idx])

        if idx < len(vlm_enhanced_prompts) and vlm_enhanced_prompts[idx] is not None:
            task_join_settings["prompt"] = vlm_enhanced_prompts[idx]

        transition_payload = {
            "orchestrator_task_id_ref": orchestrator_task_id_str,
            "orchestrator_run_id": run_id,
            "project_id": orchestrator_project_id,
            "join_index": idx,
            "transition_index": idx,
            "is_first_join": False,
            "is_last_join": False,
            "child_order": idx,
            "skip_generation": True,
            "starting_video_path": clip_start.get("url"),
            "ending_video_path": clip_end.get("url"),
            "transition_only": True,
            **task_join_settings,
            "current_run_base_output_dir": str(current_run_output_dir.resolve()),
            "join_output_dir": str((current_run_output_dir / f"transition_{idx}").resolve()),
            "orchestrator_details": orchestrator_payload,
        }

        cancel_msg = _check_orchestrator_cancelled(
            orchestrator_task_id_str,
            f"aborting transition creation at index {idx} ({len(transition_task_ids)} transitions already created)")
        if cancel_msg:
            return False, cancel_msg

        trans_task_id = add_task_to_db(
            task_payload=transition_payload,
            task_type_str="join_clips_segment",
            dependant_on=None
        )

        transition_task_ids.append(trans_task_id)

    # === Phase 2: Create final stitch task that depends on ALL transitions ===
    context_frame_count = join_settings.get("context_frame_count", 8)

    final_stitch_payload = {
        "orchestrator_task_id_ref": orchestrator_task_id_str,
        "orchestrator_run_id": run_id,
        "project_id": orchestrator_project_id,
        "parent_generation_id": parent_generation_id,
        "clip_list": clip_list,
        "transition_task_ids": transition_task_ids,
        "blend_frames": min(context_frame_count, 15),
        "fps": join_settings.get("fps") or orchestrator_payload.get("fps", 16),
        "audio_url": orchestrator_payload.get("audio_url"),
        "current_run_base_output_dir": str(current_run_output_dir.resolve()),
    }

    cancel_msg = _check_orchestrator_cancelled(
        orchestrator_task_id_str,
        f"aborting before final stitch ({len(transition_task_ids)} transitions cancelled)")
    if cancel_msg:
        return False, cancel_msg

    final_stitch_task_id = add_task_to_db(
        task_payload=final_stitch_payload,
        task_type_str="join_final_stitch",
        dependant_on=transition_task_ids
    )

    return True, f"Successfully enqueued {num_joins} parallel transitions + 1 final stitch for run {run_id}"


# Public aliases for cross-module use.
create_join_chain_tasks = _create_join_chain_tasks
create_parallel_join_tasks = _create_parallel_join_tasks
