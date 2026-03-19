"""Compatibility wrappers for runtime task-status helpers."""

from __future__ import annotations

from source.core.db import task_status as _task_status
from source.core.db.config import resolve_edge_request

_cfg = _task_status._cfg


def resolve_runtime_config(runtime_config=None):
    """Prefer an explicit runtime config and fall back to the shared DB runtime."""
    return runtime_config or _cfg.get_db_runtime_config()


def resolve_update_status_request(runtime_config=None):
    """Build the update-task-status request contract for the provided runtime."""
    runtime = resolve_runtime_config(runtime_config)
    return resolve_edge_request("update-task-status", runtime_config=runtime)


def resolve_generate_upload_url_request(runtime_config=None):
    """Build the generate-upload-url request contract for the provided runtime."""
    runtime = resolve_runtime_config(runtime_config)
    return resolve_edge_request("generate-upload-url", runtime_config=runtime)


def resolve_complete_task_request(runtime_config=None):
    """Build the complete_task request contract for the provided runtime."""
    runtime = resolve_runtime_config(runtime_config)
    return resolve_edge_request("complete_task", runtime_config=runtime)


__all__ = [
    "_cfg",
    "resolve_edge_request",
    "resolve_runtime_config",
    "resolve_update_status_request",
    "resolve_generate_upload_url_request",
    "resolve_complete_task_request",
]
