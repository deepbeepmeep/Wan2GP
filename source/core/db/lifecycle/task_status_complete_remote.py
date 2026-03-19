"""Compatibility wrappers for remote task-status completion helpers."""

from __future__ import annotations

from source.core.db import task_status as _task_status
from source.core.db.config import STATUS_FAILED
from source.core.db.lifecycle.task_status_runtime import resolve_update_status_request

call_edge_function_with_retry = _task_status._call_edge_function_with_retry


def mark_task_failed_via_edge_function(
    task_id: str,
    message: str,
    *,
    runtime_config=None,
) -> bool:
    """Mark a task failed through the shared update-status edge contract."""
    request = resolve_update_status_request(runtime_config)
    if not getattr(request, "url", None):
        return False

    response, edge_error = call_edge_function_with_retry(
        edge_url=request.url,
        payload={
            "task_id": task_id,
            "status": STATUS_FAILED,
            "output_location": message,
        },
        headers=getattr(request, "headers", {}),
        function_name="update-task-status",
        context_id=task_id,
        timeout=30,
        max_retries=3,
    )
    return bool(response and response.status_code == 200 and not edge_error)


def _complete_with_payload(
    task_id_str: str,
    payload: dict,
    *,
    complete_request,
    runtime=None,
):
    """Submit a completion payload to the complete_task edge endpoint."""
    return call_edge_function_with_retry(
        edge_url=complete_request.url,
        payload=payload,
        headers=getattr(complete_request, "headers", {}),
        function_name="complete_task",
        context_id=task_id_str,
        timeout=60,
        max_retries=3,
        fallback_url=None,
        retry_on_404_patterns=["Task not found", "not found"],
    )


def complete_task_with_remote_output(
    task_id_str: str,
    output_location_val: str,
    *,
    thumbnail_url_val: str | None = None,
    complete_request,
    runtime=None,
):
    """Complete a task using an existing remote storage path or raw output URL."""
    storage_marker = "/storage/v1/object/public/image_uploads/"
    payload = {"task_id": task_id_str}

    if storage_marker in output_location_val:
        payload["storage_path"] = output_location_val.split(storage_marker, 1)[1]
    else:
        payload["output_location"] = output_location_val

    if thumbnail_url_val and storage_marker in thumbnail_url_val:
        payload["thumbnail_storage_path"] = thumbnail_url_val.split(storage_marker, 1)[1]

    completion_result = _complete_with_payload(
        task_id_str,
        payload,
        complete_request=complete_request,
        runtime=runtime,
    )
    if isinstance(completion_result, tuple):
        response, edge_error = completion_result
    else:
        response, edge_error = completion_result, None
    if response and response.status_code == 200 and not edge_error:
        return response.json()

    error_message = edge_error or (
        f"HTTP_{response.status_code}: {response.text}" if response else "no response"
    )
    mark_task_failed_via_edge_function(
        task_id_str,
        f"Completion failed: {error_message}",
        runtime_config=runtime,
    )
    return False


__all__ = [
    "call_edge_function_with_retry",
    "mark_task_failed_via_edge_function",
    "_complete_with_payload",
    "complete_task_with_remote_output",
]
