"""Worker-focused runtime query helpers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from debug.models import WorkerInfo, WorkersSummary


def _query_logs(context, **kwargs):
    return context.log_client.get_logs(context.log_query_class(**kwargs))


def get_worker_info(context, *, worker_id: str, hours: int, unavailable_error_cls):
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        logs = _query_logs(context, worker_id=worker_id, start_time=cutoff, limit=5000, order_desc=False)
        state_rows = context.supabase.table("workers").select("*").eq("id", worker_id).execute().data
        state = state_rows[0] if state_rows else None
        tasks = context.supabase.table("tasks").select("*").eq("worker_id", worker_id).order("created_at", desc=True).limit(20).execute().data
    except Exception as exc:
        raise unavailable_error_cls(str(exc)) from exc
    return WorkerInfo(worker_id=worker_id, state=state, logs=logs or [], tasks=tasks or [])


def check_worker_logging(context, *, worker_id: str):
    logs = _query_logs(context, worker_id=worker_id, limit=10, order_desc=True)
    return SimpleNamespace(is_logging=bool(logs), log_count=len(logs), recent_logs=logs[:5])


def query_check_worker_disk_space(*, worker_id: str, unavailable_error_cls):
    raise unavailable_error_cls(f"SSH disk check not supported for worker {worker_id}")


check_worker_disk_space = query_check_worker_disk_space


def get_workers_by_statuses(context, *, statuses: list[str], unavailable_error_cls):
    try:
        return context.supabase.table("workers").select("*").in_("status", statuses).execute().data or []
    except Exception as exc:
        raise unavailable_error_cls(str(exc)) from exc


def get_active_workers_grouped_by_storage(context, *, unavailable_error_cls):
    workers = get_workers_by_statuses(context, statuses=["active"], unavailable_error_cls=unavailable_error_cls)
    grouped: dict[str, list[dict]] = {}
    for worker in workers:
        storage = ((worker.get("metadata") or {}).get("storage_volume")) or "unknown"
        grouped.setdefault(storage, []).append(worker)
    return grouped


def get_workers_summary(context, *, hours: int, unavailable_error_cls):
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        workers = context.supabase.table("workers").select("*").gte("created_at", cutoff.isoformat()).order("created_at", desc=True).execute().data or []
    except Exception as exc:
        raise unavailable_error_cls(str(exc)) from exc

    now = datetime.now(timezone.utc)
    status_counts = Counter(worker.get("status") for worker in workers)
    active_healthy = 0
    active_stale = 0
    recent_failures = []

    for worker in workers:
        status = worker.get("status")
        last_hb = worker.get("last_heartbeat")
        if status == "active":
            if last_hb:
                try:
                    age = (now - datetime.fromisoformat(last_hb.replace("Z", "+00:00"))).total_seconds()
                    if age < 60:
                        active_healthy += 1
                    else:
                        active_stale += 1
                except Exception:
                    active_stale += 1
            else:
                active_stale += 1
        if status in {"error", "terminated"}:
            recent_failures.append(worker)

    failure_rate = None if not workers else len(recent_failures) / len(workers)
    return WorkersSummary(
        workers=workers,
        total_count=len(workers),
        status_counts=dict(status_counts),
        active_healthy=active_healthy,
        active_stale=active_stale,
        recent_failures=recent_failures[:10],
        failure_rate=failure_rate,
    )


__all__ = [
    "check_worker_disk_space",
    "get_active_workers_grouped_by_storage",
    "get_worker_info",
    "get_workers_by_statuses",
    "get_workers_summary",
    "query_check_worker_disk_space",
]
