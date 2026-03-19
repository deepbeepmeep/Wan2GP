"""Compatibility wrappers for task-polling helper seams."""

from __future__ import annotations

from source.core.db import task_polling as _task_polling

_cfg = _task_polling._cfg


def query_task_status(task_id: str) -> tuple[str | None, str | None]:
    """Query task status using the legacy direct-client contract."""
    if not _cfg.SUPABASE_CLIENT:
        return None, None

    response = (
        _cfg.SUPABASE_CLIENT.table(_cfg.PG_TABLE_NAME)
        .select("status, output_location")
        .eq("id", task_id)
        .single()
        .execute()
    )
    if not response.data:
        return None, None
    return response.data.get("status"), response.data.get("output_location")


evaluate_polled_status = _task_polling.evaluate_polled_status


__all__ = ["_cfg", "query_task_status", "evaluate_polled_status"]
