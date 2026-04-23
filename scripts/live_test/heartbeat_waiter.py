"""Heartbeat-based readiness detection for live worker tests."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any


class WorkerReadyTimeoutError(TimeoutError):
    """Raised when the worker never establishes a stable heartbeat."""


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


def wait_until_ready(
    db,
    worker_id: str,
    *,
    timeout_sec: int = 900,
    interval_sec: int = 10,
    dwell_polls: int = 2,
) -> dict[str, Any]:
    """Wait until the workers row has a recent heartbeat for N consecutive polls."""
    if dwell_polls < 1:
        raise ValueError("dwell_polls must be >= 1")

    deadline = time.monotonic() + timeout_sec
    consecutive_fresh_polls = 0

    while time.monotonic() < deadline:
        rows = _coerce_rows(
            db.supabase.table("workers").select("id, last_heartbeat").eq("id", worker_id).execute()
        )
        worker = rows[0] if rows else None
        last_heartbeat = _parse_timestamp(worker.get("last_heartbeat") if worker else None)

        # Only the startup template writes startup_phase metadata. No code under reigh-worker/source/
        # updates it, so gating Variant Fresh on startup_phase would make the custom PAT launch path
        # impossible to satisfy. Heartbeat freshness plus dwell is the reliable signal here.
        is_fresh = False
        if last_heartbeat is not None:
            freshness_cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
            is_fresh = last_heartbeat > freshness_cutoff

        if is_fresh:
            consecutive_fresh_polls += 1
            if consecutive_fresh_polls >= dwell_polls:
                return worker or {"id": worker_id, "last_heartbeat": None}
        else:
            consecutive_fresh_polls = 0

        time.sleep(interval_sec)

    raise WorkerReadyTimeoutError(
        f"Worker {worker_id} did not maintain a fresh heartbeat for {dwell_polls} consecutive polls"
    )


__all__ = ["WorkerReadyTimeoutError", "wait_until_ready"]
