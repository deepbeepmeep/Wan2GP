"""Video storage upload compatibility helpers."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from source.core.db.config import has_required_edge_credentials, resolve_edge_request
from source.core.db.edge.retry import call_edge_function_with_retry

__all__ = [
    "call_edge_function_with_retry",
    "resolve_edge_request",
    "resolve_final_output_location",
    "upload_and_get_final_output_location",
    "upload_intermediate_file_to_storage",
]


def resolve_final_output_location(
    local_file_path: Path,
    initial_db_location: str | None,
) -> str:
    """Prefer the DB-facing location when one was already assigned."""
    if initial_db_location:
        return initial_db_location
    return str(local_file_path.resolve())


def upload_and_get_final_output_location(
    local_file_path: Path,
    initial_db_location: str | None,
) -> str:
    """Return the final output location without mutating storage state."""
    return resolve_final_output_location(local_file_path, initial_db_location)


def upload_intermediate_file_to_storage(
    local_file_path: Path,
    task_id: str,
    filename: str,
    runtime_config=None,
) -> str | None:
    """Upload an intermediate file through the edge-function contract."""
    if not local_file_path.exists():
        return None

    request = resolve_edge_request("generate-upload-url", runtime_config=runtime_config)
    if not request.url:
        return None
    if not has_required_edge_credentials(request.headers):
        return None

    content_type = mimetypes.guess_type(str(local_file_path))[0] or "application/octet-stream"
    response, edge_error = call_edge_function_with_retry(
        edge_url=request.url,
        payload={
            "task_id": task_id,
            "filename": filename,
            "content_type": content_type,
        },
        headers=request.headers,
        function_name="generate-upload-url",
    )
    if edge_error or response is None or response.status_code != 200:
        return None

    payload = response.json() or {}
    upload_url = payload.get("upload_url")
    storage_path = payload.get("storage_path")
    if not upload_url or not storage_path:
        return None

    upload_response, upload_error = call_edge_function_with_retry(
        edge_url=upload_url,
        payload={
            "local_file_path": str(local_file_path),
            "content_type": content_type,
        },
        headers={"Content-Type": content_type},
        function_name="storage-upload",
        method="PUT",
        timeout_seconds=300,
    )
    if upload_error or upload_response is None or upload_response.status_code not in {200, 201}:
        return None

    supabase_url = (getattr(runtime_config, "supabase_url", "") or "").rstrip("/")
    if not supabase_url:
        return storage_path
    return f"{supabase_url}/storage/v1/object/public/image_uploads/{storage_path}"
