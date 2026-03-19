"""Compatibility wrappers for task-status completion helpers."""

from __future__ import annotations

from pathlib import Path

from source.core.db import task_status as _task_status
from source.core.db.config import STATUS_COMPLETE
from source.core.db.lifecycle.task_status_complete_local import complete_task_with_local_file
from source.core.db.lifecycle.task_status_complete_remote import complete_task_with_remote_output
from source.core.db.lifecycle.task_status_runtime import (
    resolve_complete_task_request,
    resolve_runtime_config,
)
from source.core.db.lifecycle.task_status_update_edge import update_status_via_edge


def update_task_status_supabase_legacy(
    task_id_str,
    status_str,
    output_location_val=None,
    thumbnail_url_val=None,
):
    """Route completion updates through split helpers and everything else through update edge."""
    runtime = resolve_runtime_config(None)
    if status_str != STATUS_COMPLETE or output_location_val is None:
        return update_status_via_edge(
            task_id_str,
            status_str,
            output_location_val=output_location_val,
            thumbnail_url_val=thumbnail_url_val,
            runtime_config=runtime,
        )

    complete_request = resolve_complete_task_request(runtime)
    output_path = Path(output_location_val)
    if output_path.exists() and output_path.is_file():
        return complete_task_with_local_file(
            task_id_str,
            output_path,
            complete_request=complete_request,
            runtime=runtime,
        )
    return complete_task_with_remote_output(
        task_id_str,
        output_location_val,
        thumbnail_url_val=thumbnail_url_val,
        complete_request=complete_request,
        runtime=runtime,
    )


__all__ = [
    "resolve_runtime_config",
    "update_status_via_edge",
    "update_task_status_supabase_legacy",
]
