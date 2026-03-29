"""
Task insertion into the database via Edge Functions.
"""
import os

from source.core.log import headless_logger

__all__ = [
    "add_task_to_db",
]

from . import config as _cfg
from .edge_helpers import _call_edge_function_with_retry


def add_task_to_db(task_payload: dict, task_type_str: str, dependant_on: str | list[str] | None = None, db_path: str | None = None) -> str:
    """
    Adds a new task to the Supabase database via Edge Function.

    Args:
        task_payload: Task parameters dictionary
        task_type_str: Type of task being created
        dependant_on: Optional dependency - single task ID string or list of task IDs.
                      When list is provided, task will only run when ALL dependencies complete.
        db_path: Ignored (kept for API compatibility)

    Returns:
        str: The database row ID (UUID) assigned to the task
    """
    # Generate a new UUID for the database row ID
    import uuid
    actual_db_row_id = str(uuid.uuid4())

    # Sanitize payload and get project_id
    params_for_db = task_payload.copy()
    params_for_db.pop("task_type", None)  # Ensure task_type is not duplicated in params
    project_id = task_payload.get("project_id", "default_project_id")

    # Build Edge URL
    edge_url = (
        _cfg.SUPABASE_EDGE_CREATE_TASK_URL
    ) or os.getenv("SUPABASE_EDGE_CREATE_TASK_URL") or (
        f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/create-task" if _cfg.SUPABASE_URL else None
    )

    if not edge_url:
        raise ValueError("Edge Function URL for create-task is not configured")

    headers = {"Content-Type": "application/json"}
    if _cfg.SUPABASE_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"

    # Normalize dependant_on to list format for consistency
    # Edge Function expects: null, or JSON array of UUIDs
    dependant_on_list: list[str] | None = None
    if dependant_on:
        if isinstance(dependant_on, str):
            dependant_on_list = [dependant_on]
        else:
            dependant_on_list = list(dependant_on)

    # Dependency validation is handled server-side by the create-task edge function.
    # No client-side pre-check needed.

    # Map legacy task_type to the new family-based resolver format.
    # The create-task edge function requires { family, project_id, input }.
    _TASK_TYPE_TO_FAMILY = {
        "travel_segment": "individual_travel_segment",
        "individual_travel_segment": "individual_travel_segment",
        "travel_stitch": "crossfade_join",
        "join_clips_orchestrator": "join_clips",
        "join_final_stitch": "crossfade_join",
        "image_upscale": "image_upscale",
        "image-upscale": "image_upscale",
        "video_enhance": "video_enhance",
        "animate_character": "character_animate",
        "travel_orchestrator": "travel_between_images",
        "edit_video_orchestrator": "edit_video_orchestrator",
        "magic_edit": "magic_edit",
        "image_inpaint": "masked_edit",
        "image_edit": "masked_edit",
        "annotated_image_edit": "masked_edit",
        "qwen_image_edit": "masked_edit",
        "single_image": "image_generation",
        "z_image_turbo_i2i": "z_image_turbo_i2i",
    }
    family = _TASK_TYPE_TO_FAMILY.get(task_type_str, task_type_str)

    payload_edge = {
        "family": family,
        "project_id": project_id,
        "input": {
            **params_for_db,
            "task_id": actual_db_row_id,
            "dependant_on": dependant_on_list,
        },
    }

    headless_logger.debug(f"Supabase Edge call >>> POST {edge_url} payload={str(payload_edge)[:120]}\u2026")

    # Use standardized retry helper (handles 5xx, timeout, and network errors)
    resp, edge_error = _call_edge_function_with_retry(
        edge_url=edge_url,
        payload=payload_edge,
        headers=headers,
        function_name="create-task",
        context_id=f"{task_type_str}:{actual_db_row_id[:8]}",
        timeout=45,  # Base timeout
        max_retries=3)

    # Handle failure cases
    if edge_error:
        raise RuntimeError(f"Edge Function create-task failed: {edge_error}")
    if resp is None:
        raise RuntimeError(f"Edge Function create-task returned no response for {actual_db_row_id}")

    if resp.status_code == 200:
        headless_logger.essential(f"Task {actual_db_row_id} (Type: {task_type_str}) queued via Edge Function.", task_id=actual_db_row_id)
        # Edge function returned 200 — task creation confirmed server-side.
        # No client-side verification needed.
        return actual_db_row_id
    else:
        error_msg = f"Edge Function create-task failed: {resp.status_code} - {resp.text}"
        headless_logger.error(error_msg, task_id=actual_db_row_id)
        raise RuntimeError(error_msg)
