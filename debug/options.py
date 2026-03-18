"""Typed CLI/debug option coercion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _mapping(value: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _int_or_none(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TaskOptions:
    format: str = "text"
    debug: bool = False
    task_id: str | None = None
    logs_only: bool = False


@dataclass(frozen=True)
class WorkerOptions:
    format: str = "text"
    debug: bool = False
    worker_id: str | None = None
    hours: int = 24
    logs_only: bool = False
    startup: bool = False
    check_logging: bool = False
    check_disk: bool = False


@dataclass(frozen=True)
class TasksOptions:
    format: str = "text"
    debug: bool = False
    limit: int = 50
    status: str | None = None
    task_type: str | None = None
    worker_id: str | None = None
    hours: int | None = None


@dataclass(frozen=True)
class HealthOptions:
    format: str = "text"
    debug: bool = False


@dataclass(frozen=True)
class ConfigOptions:
    explain: bool = False


@dataclass(frozen=True)
class RunpodOptions:
    terminate: bool = False
    debug: bool = False


@dataclass(frozen=True)
class StorageOptions:
    expand: str | None = None
    debug: bool = False


def coerce_task_options(value) -> TaskOptions:
    data = _mapping(value)
    return TaskOptions(
        format=data.get("format", "text"),
        debug=bool(data.get("debug", False)),
        task_id=data.get("task_id"),
        logs_only=bool(data.get("logs_only", False)),
    )


def coerce_worker_options(value) -> WorkerOptions:
    data = _mapping(value)
    return WorkerOptions(
        format=data.get("format", "text"),
        debug=bool(data.get("debug", False)),
        worker_id=data.get("worker_id"),
        hours=_int_or_none(data.get("hours")) or 24,
        logs_only=bool(data.get("logs_only", False)),
        startup=bool(data.get("startup", False)),
        check_logging=bool(data.get("check_logging", False)),
        check_disk=bool(data.get("check_disk", False)),
    )


def coerce_tasks_options(value) -> TasksOptions:
    data = _mapping(value)
    return TasksOptions(
        format=data.get("format", "text"),
        debug=bool(data.get("debug", False)),
        limit=_int_or_none(data.get("limit")) or 50,
        status=data.get("status"),
        task_type=data.get("task_type") or data.get("type"),
        worker_id=data.get("worker_id") or data.get("worker"),
        hours=_int_or_none(data.get("hours")),
    )


def coerce_health_options(value) -> HealthOptions:
    data = _mapping(value)
    return HealthOptions(format=data.get("format", "text"), debug=bool(data.get("debug", False)))


def coerce_config_options(value) -> ConfigOptions:
    data = _mapping(value)
    return ConfigOptions(explain=bool(data.get("explain", False)))


def coerce_runpod_options(value) -> RunpodOptions:
    data = _mapping(value)
    return RunpodOptions(terminate=bool(data.get("terminate", False)), debug=bool(data.get("debug", False)))


def coerce_storage_options(value) -> StorageOptions:
    data = _mapping(value)
    return StorageOptions(expand=data.get("expand"), debug=bool(data.get("debug", False)))


__all__ = [
    "ConfigOptions",
    "HealthOptions",
    "RunpodOptions",
    "StorageOptions",
    "TaskOptions",
    "TasksOptions",
    "WorkerOptions",
    "coerce_config_options",
    "coerce_health_options",
    "coerce_runpod_options",
    "coerce_storage_options",
    "coerce_task_options",
    "coerce_tasks_options",
    "coerce_worker_options",
]
