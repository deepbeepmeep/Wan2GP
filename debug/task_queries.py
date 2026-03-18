"""Task-specific query helpers extracted from DebugClient."""

from __future__ import annotations

import json


def get_task_info(context, task_id: str, *, follow_cascade: bool = True):
    """Compatibility wrapper that delegates to the bound debug client API."""
    return context.get_task_info(task_id, follow_cascade=follow_cascade)


def get_orchestrator_child_tasks(
    context,
    *,
    orchestrator_id: str,
    orchestrator_state: dict,
    unavailable_error_cls,
):
    params = orchestrator_state.get("params", {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError as exc:
            raise unavailable_error_cls("Malformed orchestrator params") from exc

    project_id = orchestrator_state.get("project_id") or params.get("project_id")
    rows = context.supabase.table("tasks").select(
        "id, task_type, status, error_message, output_location, created_at, "
        "generation_started_at, generation_processed_at, worker_id, attempts, project_id, params"
    ).eq("project_id", project_id).execute().data or []

    child_rows = []
    for row in rows:
        raw_params = row.get("params", {})
        if isinstance(raw_params, str):
            try:
                raw_params = json.loads(raw_params)
            except json.JSONDecodeError as exc:
                raise unavailable_error_cls("Malformed child-task params") from exc
        ref = raw_params.get("orchestrator_task_id_ref")
        if ref == orchestrator_id:
            child_rows.append(row)
    return child_rows


__all__ = ["get_orchestrator_child_tasks", "get_task_info"]
