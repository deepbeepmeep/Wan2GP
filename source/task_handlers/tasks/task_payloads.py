"""Task payload coercion helpers."""

from __future__ import annotations


def coerce_db_task_payload(payload, *, task_id: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"task {task_id}: payload must be a mapping")
    return dict(payload)


__all__ = ["coerce_db_task_payload"]
