"""Compatibility wrappers for local task-status completion helpers."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from source.core.constants import BYTES_PER_MB
from source.core.db import task_status as _task_status
from source.core.db.lifecycle.task_status_complete_remote import (
    mark_task_failed_via_edge_function,
)
from source.core.db.lifecycle.task_status_runtime import (
    resolve_generate_upload_url_request,
)

call_edge_function_with_retry = _task_status._call_edge_function_with_retry
FILE_SIZE_THRESHOLD_MB = 2.0


def complete_task_with_local_file(
    task_id_str: str,
    output_file: Path,
    *,
    complete_request,
    runtime=None,
):
    """Complete a task from a local artifact, using base64 or presigned uploads."""
    file_size_mb = output_file.stat().st_size / BYTES_PER_MB
    if file_size_mb < FILE_SIZE_THRESHOLD_MB:
        response, edge_error = call_edge_function_with_retry(
            edge_url=complete_request.url,
            payload={
                "task_id": task_id_str,
                "filename": output_file.name,
                "file_data": base64.b64encode(output_file.read_bytes()).decode("utf-8"),
            },
            headers=getattr(complete_request, "headers", {}),
            function_name="complete_task",
            context_id=task_id_str,
            timeout=60,
            max_retries=3,
            fallback_url=None,
            retry_on_404_patterns=["Task not found", "not found"],
        )
        if response and response.status_code == 200 and not edge_error:
            return response.json()
        error_message = edge_error or (
            f"HTTP_{response.status_code}: {response.text}" if response else "no response"
        )
        mark_task_failed_via_edge_function(
            task_id_str,
            f"Upload failed: {error_message}",
            runtime_config=runtime,
        )
        return False

    upload_request = resolve_generate_upload_url_request(runtime)
    if not getattr(upload_request, "url", None):
        mark_task_failed_via_edge_function(
            task_id_str,
            "Upload failed: missing generate-upload-url endpoint",
            runtime_config=runtime,
        )
        return False

    content_type = mimetypes.guess_type(str(output_file))[0] or "application/octet-stream"
    upload_response, upload_error = call_edge_function_with_retry(
        edge_url=upload_request.url,
        payload={
            "task_id": task_id_str,
            "filename": output_file.name,
            "content_type": content_type,
        },
        headers=getattr(upload_request, "headers", {}),
        function_name="generate-upload-url",
        context_id=task_id_str,
        timeout=30,
        max_retries=3,
    )
    if upload_error or not upload_response or upload_response.status_code != 200:
        error_message = upload_error or (
            f"HTTP_{upload_response.status_code}: {upload_response.text}"
            if upload_response
            else "no response"
        )
        mark_task_failed_via_edge_function(
            task_id_str,
            f"Upload failed: {error_message}",
            runtime_config=runtime,
        )
        return False

    upload_data = upload_response.json()
    put_response, put_error = call_edge_function_with_retry(
        edge_url=upload_data["upload_url"],
        payload=output_file,
        headers={"Content-Type": content_type},
        function_name="storage-upload-file",
        context_id=task_id_str,
        timeout=600,
        max_retries=3,
        method="PUT",
    )
    if put_error or not put_response or put_response.status_code not in (200, 201):
        error_message = put_error or (
            f"HTTP_{put_response.status_code}: {put_response.text}"
            if put_response
            else "no response"
        )
        mark_task_failed_via_edge_function(
            task_id_str,
            f"Upload failed: {error_message}",
            runtime_config=runtime,
        )
        return False

    complete_response, complete_error = call_edge_function_with_retry(
        edge_url=complete_request.url,
        payload={"task_id": task_id_str, "storage_path": upload_data["storage_path"]},
        headers=getattr(complete_request, "headers", {}),
        function_name="complete_task",
        context_id=task_id_str,
        timeout=60,
        max_retries=3,
        fallback_url=None,
        retry_on_404_patterns=["Task not found", "not found"],
    )
    if complete_response and complete_response.status_code == 200 and not complete_error:
        return complete_response.json()

    error_message = complete_error or (
        f"HTTP_{complete_response.status_code}: {complete_response.text}"
        if complete_response
        else "no response"
    )
    mark_task_failed_via_edge_function(
        task_id_str,
        f"Upload failed: {error_message}",
        runtime_config=runtime,
    )
    return False


__all__ = [
    "FILE_SIZE_THRESHOLD_MB",
    "call_edge_function_with_retry",
    "resolve_generate_upload_url_request",
    "mark_task_failed_via_edge_function",
    "complete_task_with_local_file",
]
