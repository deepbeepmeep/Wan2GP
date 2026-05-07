"""Heartbeat-based readiness detection for live worker tests."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.live_test.logger import get_logger


class WorkerReadyTimeoutError(TimeoutError):
    """Raised when the worker never establishes a stable heartbeat."""


log = get_logger(__name__)


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
    progress_every_sec: int | None = None,
) -> dict[str, Any]:
    """Wait until the workers row has a recent heartbeat and worker-ready preflight."""
    if dwell_polls < 1:
        raise ValueError("dwell_polls must be >= 1")

    deadline = time.monotonic() + timeout_sec
    started_at = time.monotonic()
    next_progress_at = started_at
    consecutive_fresh_polls = 0

    while time.monotonic() < deadline:
        rows = _coerce_rows(
            db.supabase.table("workers").select("id, last_heartbeat, metadata").eq("id", worker_id).execute()
        )
        worker = rows[0] if rows else None
        last_heartbeat = _parse_timestamp(worker.get("last_heartbeat") if worker else None)
        metadata = worker.get("metadata") if worker else None
        ready_for_tasks = bool(metadata.get("ready_for_tasks")) if isinstance(metadata, dict) else False

        # Heartbeat alone starts before WGP import and task-queue startup finish. The worker
        # publishes ready_for_tasks only after backend preflight and queue startup pass.
        is_fresh = False
        if last_heartbeat is not None:
            freshness_cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
            is_fresh = last_heartbeat > freshness_cutoff

        if is_fresh and ready_for_tasks:
            consecutive_fresh_polls += 1
            if consecutive_fresh_polls >= dwell_polls:
                if progress_every_sec:
                    log.info(
                        "worker ready for tasks",
                        worker_id=worker_id,
                        elapsed_sec=round(time.monotonic() - started_at, 1),
                    )
                return worker or {"id": worker_id, "last_heartbeat": None}
        else:
            consecutive_fresh_polls = 0

        now = time.monotonic()
        if progress_every_sec and now >= next_progress_at:
            log.info(
                "waiting for worker heartbeat",
                worker_id=worker_id,
                elapsed_sec=round(now - started_at, 1),
                timeout_sec=timeout_sec,
                fresh_polls=consecutive_fresh_polls,
                heartbeat_fresh=is_fresh,
                ready_for_tasks=ready_for_tasks,
            )
            next_progress_at = now + progress_every_sec

        time.sleep(interval_sec)

    raise WorkerReadyTimeoutError(
        f"Worker {worker_id} did not maintain fresh heartbeat and ready_for_tasks for {dwell_polls} consecutive polls"
    )


__all__ = ["WorkerReadyTimeoutError", "wait_until_ready"]
