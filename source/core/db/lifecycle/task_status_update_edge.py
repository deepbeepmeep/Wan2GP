"""Compatibility wrappers for edge-backed task-status helpers."""

from __future__ import annotations

from source.core.db import task_status as _task_status
from source.core.db.lifecycle.task_status_runtime import resolve_update_status_request

call_edge_function_with_retry = _task_status._call_edge_function_with_retry


def update_status_via_edge(
    task_id: str,
    status: str,
    *,
    output_location_val: str | None = None,
    thumbnail_url_val: str | None = None,
    runtime_config=None,
) -> bool:
    """Update task status through the edge-function contract."""
    request = resolve_update_status_request(runtime_config)
    if not getattr(request, "url", None):
        return False

    payload = {"task_id": task_id, "status": status}
    if output_location_val is not None:
        payload["output_location"] = output_location_val
    if thumbnail_url_val is not None:
        payload["thumbnail_url"] = thumbnail_url_val

    response, edge_error = call_edge_function_with_retry(
        edge_url=request.url,
        payload=payload,
        headers=getattr(request, "headers", {}),
        function_name="update-task-status",
        context_id=task_id,
        timeout=30,
        max_retries=3,
    )
    return bool(response and response.status_code == 200 and not edge_error)


__all__ = [
    "call_edge_function_with_retry",
    "resolve_update_status_request",
    "update_status_via_edge",
]
