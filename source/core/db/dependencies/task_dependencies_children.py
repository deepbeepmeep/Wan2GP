"""Compatibility wrappers for orchestrator-child dependency helpers."""

from __future__ import annotations

from source.core.db import task_dependencies as _task_dependencies
from source.core.db.config import allow_direct_query_fallback, resolve_edge_request

_cfg = _task_dependencies._cfg


def cancel_orchestrator_children(
    orchestrator_task_id: str,
    reason: str = "Orchestrator cancelled",
    *,
    child_fetcher=None,
    status_updater=None,
) -> int:
    """Cancel non-terminal child tasks, optionally using injected collaborators."""
    if child_fetcher is None and status_updater is None:
        return _task_dependencies.cancel_orchestrator_children(orchestrator_task_id, reason)

    child_fetcher = child_fetcher or _task_dependencies.get_orchestrator_child_tasks
    status_updater = status_updater or _task_dependencies.update_task_status
    terminal_statuses = {"complete", "failed", "cancelled", "canceled", "error"}

    child_tasks = child_fetcher(orchestrator_task_id)
    all_children = []
    for category in child_tasks.values():
        if isinstance(category, list):
            all_children.extend(category)

    cancelled_count = 0
    for child in all_children:
        child_id = child.get("id")
        child_status = (child.get("status") or "").lower()
        if child_status in terminal_statuses:
            continue
        status_updater(child_id, "Cancelled", output_location=reason)
        cancelled_count += 1
    return cancelled_count


def get_task_current_status(task_id: str):
    """Compatibility wrapper for task status lookup."""
    request = resolve_edge_request("get-task-status")
    if not request.url and allow_direct_query_fallback() and _cfg.SUPABASE_CLIENT:
        response = (
            _cfg.SUPABASE_CLIENT.table(_cfg.PG_TABLE_NAME)
            .select("status")
            .eq("id", task_id)
            .single()
            .execute()
        )
        if response.data:
            return response.data.get("status")
        return None

    return _task_dependencies.get_task_current_status(task_id)


__all__ = ["_cfg", "resolve_edge_request", "cancel_orchestrator_children", "get_task_current_status"]
