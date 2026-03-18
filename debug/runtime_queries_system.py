"""System/orchestrator runtime query helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from source.core.db.config import DBRuntimeContractError
from debug.models import OrchestratorStatus, SystemHealth


def _query_logs(context, **kwargs):
    return context.log_client.get_logs(context.log_query_class(**kwargs))


def query_system_health(context):
    try:
        return get_system_health(context, unavailable_error_cls=DBRuntimeContractError)
    except Exception as exc:
        if isinstance(exc, DBRuntimeContractError):
            raise
        raise DBRuntimeContractError(str(exc)) from exc


def get_system_health(context, *, unavailable_error_cls):
    try:
        now = datetime.now(timezone.utc)
        workers = context.supabase.table("workers").select("*").neq("status", "terminated").execute().data or []
        tasks = context.supabase.table("tasks").select("status").execute().data or []
        errors = _query_logs(
            context,
            start_time=now - timedelta(hours=1),
            log_level="ERROR",
            limit=50,
            order_desc=True,
        )
    except Exception as exc:
        raise unavailable_error_cls(str(exc)) from exc

    healthy = 0
    for worker in workers:
        if worker.get("status") != "active" or not worker.get("last_heartbeat"):
            continue
        try:
            last_hb = datetime.fromisoformat(str(worker["last_heartbeat"]).replace("Z", "+00:00"))
            if (now - last_hb).total_seconds() < 60:
                healthy += 1
        except Exception:
            pass

    return SystemHealth(
        timestamp=now,
        workers_active=sum(1 for worker in workers if worker.get("status") == "active"),
        workers_spawning=sum(1 for worker in workers if worker.get("status") == "spawning"),
        workers_healthy=healthy,
        tasks_queued=sum(1 for task in tasks if task.get("status") == "Queued"),
        tasks_in_progress=sum(1 for task in tasks if task.get("status") == "In Progress"),
        recent_errors=errors[:10],
        failure_rate=None,
        failure_rate_status="OK",
    )


def get_orchestrator_status(context, *, hours: int):
    logs = _query_logs(
        context,
        start_time=datetime.now(timezone.utc) - timedelta(hours=hours),
        source_type="orchestrator_gpu",
        limit=1000,
        order_desc=True,
    )
    parsed_logs = []
    malformed = 0
    for log in logs:
        try:
            ts = datetime.fromisoformat(str(log.get("timestamp")).replace("Z", "+00:00"))
            parsed_logs.append((ts, log))
        except Exception:
            malformed += 1
    if malformed:
        context._debug(f"ignored {malformed} malformed orchestrator log timestamps")

    last_activity = parsed_logs[0][0] if parsed_logs else None
    last_cycle = parsed_logs[0][1].get("cycle_number") if parsed_logs else None
    if last_activity is None:
        status = "NO_LOGS"
    else:
        age_minutes = (datetime.now(timezone.utc) - last_activity).total_seconds() / 60.0
        status = "HEALTHY" if age_minutes < 5 else "WARNING" if age_minutes < 15 else "STALE"

    cycle_logs = [log for _ts, log in parsed_logs if "Starting orchestrator cycle" in str(log.get("message", ""))]
    return OrchestratorStatus(
        last_activity=last_activity,
        last_cycle=last_cycle,
        status=status,
        recent_cycles=cycle_logs[:10],
        recent_logs=[log for _ts, log in parsed_logs[:50]],
    )


__all__ = ["get_orchestrator_status", "get_system_health", "query_system_health"]
