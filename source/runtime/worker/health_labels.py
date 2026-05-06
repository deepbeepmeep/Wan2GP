"""Safe worker health labels shared by local health and heartbeat payloads."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from source.runtime.worker.preflight import read_preflight_state
from source.runtime.worker.resource_pressure import read_resource_pressure_state


_SECRET_KEY_PARTS = ("key", "secret", "token", "password", "authorization")
_SAFE_LABEL_KEYS = {
    "route_key",
    "claimed_route_key",
    "parent_route_key",
    "selector_key",
}


def safe_route_labels() -> dict[str, Any]:
    return _route_labels(None)


def worker_route_state_path(worker_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in worker_id)
    return Path(os.environ.get("REIGH_ROUTE_STATE_DIR", os.environ.get("REIGH_PREFLIGHT_STATE_DIR", "/tmp"))) / (
        f"reigh_worker_route_{safe}.json"
    )


def write_worker_route_state(worker_id: str, task_data: dict[str, Any]) -> Path:
    route_state = _task_route_state(task_data)
    path = worker_route_state_path(worker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_redact(route_state), sort_keys=True), encoding="utf-8")
    return path


def read_worker_route_state(worker_id: str | None) -> dict[str, Any] | None:
    if not worker_id:
        return None
    try:
        data = json.loads(worker_route_state_path(worker_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _redact(data)


def _route_labels(worker_id: str | None) -> dict[str, Any]:
    route_state = read_worker_route_state(worker_id)
    return {
        "backend": _env("REIGH_BACKEND", "WORKER_BACKEND", default="wgp"),
        "profile": _env("REIGH_WORKER_PROFILE", "WGP_PROFILE", default="1"),
        "pool": _env("REIGH_WORKER_POOL", "WORKER_POOL", default="gpu-wgp-production"),
        "selector_namespace": _env("REIGH_SELECTOR_NAMESPACE", "ROUTE_SELECTOR_NAMESPACE", default="production"),
        "selector_version": _env("REIGH_SELECTOR_VERSION", "ROUTE_SELECTOR_VERSION", default=""),
        "worker_contract_version": _env("REIGH_WORKER_CONTRACT_VERSION", default="1"),
        "run_id": _env("REIGH_WORKER_RUN_ID", "WORKER_RUN_ID", default=""),
        "route_key": _first_non_empty(
            route_state,
            "route_key",
            env_names=("REIGH_ROUTE_KEY", "ROUTE_KEY"),
            default="",
        ),
        "template_id": _first_non_empty(
            route_state,
            "template_id",
            env_names=("REIGH_TEMPLATE_ID", "ROUTE_TEMPLATE_ID"),
            default="",
        ),
        "current_task_id": str(route_state.get("task_id") or "") if route_state else "",
        "current_task_type": str(route_state.get("task_type") or "") if route_state else "",
    }


def disk_labels(paths: list[Path] | None = None) -> dict[str, Any]:
    threshold = _float_env("REIGH_DISK_NEAR_FULL_PCT", 90.0)
    selected_paths = paths or _disk_paths_from_env()
    volumes: list[dict[str, Any]] = []
    worst_used_pct = 0.0
    for path in selected_paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        used_pct = round(((usage.total - usage.free) / usage.total) * 100, 2) if usage.total else 0.0
        worst_used_pct = max(worst_used_pct, used_pct)
        volumes.append(
            {
                "path": str(path),
                "used_pct": used_pct,
                "free_mb": int(usage.free / (1024 * 1024)),
            }
        )
    return {
        "status": "near_full" if worst_used_pct >= threshold else "ok",
        "threshold_pct": threshold,
        "worst_used_pct": worst_used_pct,
        "volumes": volumes,
    }


def warm_cache_state_path(worker_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in worker_id)
    return Path(os.environ.get("REIGH_WARM_CACHE_STATE_DIR", os.environ.get("REIGH_PREFLIGHT_STATE_DIR", "/tmp"))) / (
        f"reigh_worker_warm_cache_{safe}.json"
    )


def write_warm_cache_state(worker_id: str, metadata: dict[str, Any]) -> Path:
    path = warm_cache_state_path(worker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_redact(metadata), sort_keys=True), encoding="utf-8")
    return path


def read_warm_cache_state(worker_id: str | None) -> dict[str, Any] | None:
    if not worker_id:
        return None
    try:
        data = json.loads(warm_cache_state_path(worker_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _redact(data)


def build_health_payload(
    *,
    worker_id: str,
    version: str,
    disk_paths: list[Path] | None = None,
) -> dict[str, Any]:
    preflight = read_preflight_state(worker_id)
    warm_cache = read_warm_cache_state(worker_id)
    resource_pressure = read_resource_pressure_state(worker_id)
    payload: dict[str, Any] = {
        "ok": True,
        "worker_id": worker_id,
        "version": version,
        "labels": {
            "route": _route_labels(worker_id),
            "preflight": _preflight_labels(preflight),
            "disk": disk_labels(disk_paths),
            "warm_cache": _warm_cache_labels(warm_cache),
            "resource_pressure": _resource_pressure_labels(resource_pressure),
        },
    }
    if preflight is not None:
        payload["preflight"] = _redact(preflight)
    if warm_cache is not None:
        payload["warm_cache"] = _redact(warm_cache)
    if resource_pressure is not None:
        payload["resource_pressure"] = _redact(resource_pressure)
    return payload


def guardian_label_log(worker_id: str) -> dict[str, Any]:
    telemetry = queryable_telemetry_labels(worker_id)
    return {
        "level": "info",
        "message": "worker_health_labels",
        "metadata": _redact({
            **telemetry,
            "route": _route_labels(worker_id),
            "preflight": _preflight_labels(read_preflight_state(worker_id)),
            "disk": disk_labels(),
            "warm_cache": _warm_cache_labels(read_warm_cache_state(worker_id)),
            "resource_pressure": _resource_pressure_labels(read_resource_pressure_state(worker_id)),
        }),
    }


def queryable_telemetry_labels(worker_id: str) -> dict[str, Any]:
    route = _route_labels(worker_id)
    preflight = _preflight_labels(read_preflight_state(worker_id))
    disk = disk_labels()
    resource_pressure = _resource_pressure_labels(read_resource_pressure_state(worker_id))
    warm_cache = _warm_cache_labels(read_warm_cache_state(worker_id))
    return {
        "backend": route["backend"],
        "profile": route["profile"],
        "pool": route["pool"],
        "route_key": route["route_key"],
        "template_id": route["template_id"],
        "run_id": route["run_id"],
        "selector_namespace": route["selector_namespace"],
        "selector_version": route["selector_version"],
        "worker_contract_version": route["worker_contract_version"],
        "current_task_id": route["current_task_id"],
        "current_task_type": route["current_task_type"],
        "preflight_status": preflight["status"],
        "preflight_ok": preflight["ok"],
        "disk_status": disk["status"],
        "disk_worst_used_pct": disk["worst_used_pct"],
        "resource_pressure_status": resource_pressure["status"],
        "resource_pressure_action": resource_pressure["action"],
        "quota_alert": resource_pressure["quota_alert"],
        "warm_cache_status": warm_cache["status"],
        "warm_cache_model": warm_cache["model"],
    }


def _preflight_labels(preflight: dict[str, Any] | None) -> dict[str, Any]:
    preflight = preflight or {}
    return {
        "status": preflight.get("preflight_status") or "unknown",
        "ok": bool(preflight.get("preflight_ok")),
        "failed_checks": list(preflight.get("preflight_failed_checks") or []),
    }


def _warm_cache_labels(warm_cache: dict[str, Any] | None) -> dict[str, Any]:
    warm_cache = warm_cache or {}
    return {
        "status": warm_cache.get("warm_cache_status") or "unknown",
        "model": warm_cache.get("warm_cache_model") or "",
        "source": warm_cache.get("warm_cache_source") or "",
        "skip_reason": warm_cache.get("warm_cache_skip_reason") or "",
    }


def _resource_pressure_labels(resource_pressure: dict[str, Any] | None) -> dict[str, Any]:
    resource_pressure = resource_pressure or {}
    return {
        "status": resource_pressure.get("resource_pressure_status") or "unknown",
        "action": resource_pressure.get("resource_pressure_action") or "",
        "quota_alert": bool(resource_pressure.get("resource_pressure_quota_alert")),
        "allow_work": bool(resource_pressure.get("resource_pressure_allow_work", True)),
        "reason": resource_pressure.get("resource_pressure_reason") or "",
    }


def _disk_paths_from_env() -> list[Path]:
    configured = os.environ.get("REIGH_DISK_HEALTH_PATHS")
    if configured:
        return [Path(part) for part in configured.split(":") if part]
    paths = [Path("/")]
    workspace = Path("/workspace")
    if workspace.exists():
        paths.append(workspace)
    return paths


def _task_route_state(task_data: dict[str, Any]) -> dict[str, Any]:
    params = task_data.get("params") if isinstance(task_data.get("params"), dict) else {}
    contract = params.get("route_contract") if isinstance(params.get("route_contract"), dict) else {}
    snapshot = contract.get("route_selection_snapshot") if isinstance(contract.get("route_selection_snapshot"), dict) else {}
    return {
        "task_id": task_data.get("task_id") or task_data.get("id") or "",
        "task_type": task_data.get("task_type") or "",
        "route_key": _coalesce(
            task_data.get("route_key"),
            task_data.get("claimed_route_key"),
            contract.get("route_key"),
            snapshot.get("route_key"),
        ),
        "template_id": _coalesce(
            task_data.get("selected_template_id"),
            contract.get("selected_template_id"),
            snapshot.get("template_id"),
        ),
        "backend": _coalesce(
            task_data.get("selected_backend"),
            task_data.get("claimed_backend"),
            contract.get("selected_backend"),
            snapshot.get("selected_backend"),
        ),
        "profile": _coalesce(contract.get("selected_profile"), snapshot.get("selected_profile")),
        "selector_namespace": _coalesce(
            task_data.get("selector_namespace"),
            task_data.get("claimed_selector_namespace"),
            contract.get("selector_namespace"),
            snapshot.get("selector_namespace"),
        ),
        "selector_version": _coalesce(
            task_data.get("selector_version"),
            task_data.get("claimed_selector_version"),
            contract.get("selector_version"),
            snapshot.get("selector_version"),
        ),
        "run_id": _coalesce(contract.get("route_run_id"), snapshot.get("route_run_id")),
        "worker_contract_version": _coalesce(
            task_data.get("claimed_capability_version"),
            contract.get("worker_contract_version"),
            snapshot.get("worker_contract_version"),
        ),
    }


def _coalesce(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def _first_non_empty(
    route_state: dict[str, Any] | None,
    key: str,
    *,
    env_names: tuple[str, ...],
    default: str,
) -> str:
    if route_state and route_state.get(key) not in (None, ""):
        return str(route_state[key])
    for name in env_names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _env(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            lowered = str(key).lower()
            should_redact = lowered not in _SAFE_LABEL_KEYS and any(part in lowered for part in _SECRET_KEY_PARTS)
            result[key] = "[redacted]" if should_redact else _redact(child)
        return result
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value
