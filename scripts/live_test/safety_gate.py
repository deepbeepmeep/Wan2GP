"""Takeover safety checks for Variant B live-test runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


class UnsafeTakeoverError(RuntimeError):
    """Raised when a pod or user is not safe to take over for live testing."""


def _coerce_rows(result: Any) -> list[dict[str, Any]]:
    data = getattr(result, "data", None)
    if not data:
        return []
    if isinstance(data, dict):
        return [data]
    return [row for row in data if isinstance(row, dict)]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def assert_safe_to_take_over(
    db,
    pod_id: str,
    user_id: str,
    *,
    allow_fresh_heartbeat: bool = False,
) -> None:
    """Reject unsafe takeovers.

    B-existing passes ``allow_fresh_heartbeat=False`` because an active heartbeat on the target pod
    means someone is already using it. B-fresh-takeover passes ``True`` because the caller owns the
    newly spawned pod. The PAT-work guard still runs in both modes.
    """
    recent_task_cutoff = datetime.now(timezone.utc) - timedelta(seconds=90)
    recent_user_work = _coerce_rows(
        db.supabase.table("tasks")
        .select("id, generation_started_at, status, task_type, project_id, projects!inner(user_id)")
        .eq("status", "In Progress")
        .eq("projects.user_id", user_id)
        .execute()
    )
    active_recent_rows = []
    for row in recent_user_work:
        started_at = _parse_timestamp(row.get("generation_started_at"))
        if started_at is not None and started_at > recent_task_cutoff:
            active_recent_rows.append(row)
    if active_recent_rows:
        task_ids = ", ".join(str(row.get("id")) for row in active_recent_rows if row.get("id"))
        raise UnsafeTakeoverError(
            "Target user still has fresh in-progress PAT work; refusing takeover. "
            f"Tasks: {task_ids}"
        )

    if allow_fresh_heartbeat:
        return

    worker_rows = _coerce_rows(
        db.supabase.table("workers").select("id, last_heartbeat").eq("id", pod_id).execute()
    )
    worker = worker_rows[0] if worker_rows else None
    heartbeat = _parse_timestamp(worker.get("last_heartbeat") if worker else None)
    heartbeat_cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
    if heartbeat is not None and heartbeat >= heartbeat_cutoff:
        raise UnsafeTakeoverError(f"Pod {pod_id} still has a fresh worker heartbeat")


__all__ = ["UnsafeTakeoverError", "assert_safe_to_take_over"]
