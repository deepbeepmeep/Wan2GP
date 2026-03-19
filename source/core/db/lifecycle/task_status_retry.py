"""Compatibility wrappers for task-status retry helpers."""

from __future__ import annotations

from source.core.db import task_status as _task_status
from source.core.db.config import STATUS_QUEUED
from source.core.db.lifecycle.task_status_runtime import (
    resolve_runtime_config,
    resolve_update_status_request,
)

call_edge_function_with_retry = _task_status._call_edge_function_with_retry


def requeue_task_direct_db(
    task_id: str,
    attempts: int,
    details: str,
    *,
    runtime_config=None,
) -> bool:
    """Requeue a task using the direct DB client contract."""
    runtime = resolve_runtime_config(runtime_config)
    client = getattr(runtime, "supabase_client", None)
    table_name = getattr(runtime, "pg_table_name", _task_status._cfg.PG_TABLE_NAME)
    if not client:
        return False

    result = (
        client.table(table_name)
        .update(
            {
                "status": STATUS_QUEUED,
                "worker_id": None,
                "attempts": attempts,
                "error_details": details,
                "generation_started_at": None,
            }
        )
        .eq("id", task_id)
        .execute()
    )
    return bool(getattr(result, "data", None))


def requeue_task_for_retry(
    task_id: str,
    error_message: str,
    current_attempts: int,
    error_category: str | None = None,
) -> bool:
    """Requeue through the edge function and fall back to direct DB when needed."""
    runtime = resolve_runtime_config(None)
    new_attempts = current_attempts + 1
    details = f"Retry {new_attempts}"
    if error_category:
        details += f" ({error_category})"
    if error_message:
        details += f": {error_message[:500]}"

    request = resolve_update_status_request(runtime)
    if not getattr(request, "url", None):
        return requeue_task_direct_db(
            task_id,
            new_attempts,
            details,
            runtime_config=runtime,
        )

    response, edge_error = call_edge_function_with_retry(
        edge_url=request.url,
        payload={
            "task_id": task_id,
            "status": STATUS_QUEUED,
            "attempts": new_attempts,
            "error_details": details,
            "clear_worker": True,
        },
        headers=getattr(request, "headers", {}),
        function_name="update-task-status",
        context_id=task_id,
        timeout=30,
        max_retries=3,
    )
    if response and response.status_code == 200 and not edge_error:
        return True
    return requeue_task_direct_db(
        task_id,
        new_attempts,
        details,
        runtime_config=runtime,
    )


__all__ = [
    "call_edge_function_with_retry",
    "resolve_runtime_config",
    "resolve_update_status_request",
    "requeue_task_direct_db",
    "requeue_task_for_retry",
]
