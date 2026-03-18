"""Travel orchestrator detail coercion helpers."""

from __future__ import annotations

from types import MappingProxyType

from source.core.params.contracts import REQUIRED_ORCHESTRATOR_KEYS


def coerce_orchestrator_details(
    payload,
    *,
    context: str,
    task_id: str,
    mutable: bool = False,
    validate_required: bool = True,
):
    if not isinstance(payload, dict):
        raise ValueError(f"{context} (task {task_id}): orchestrator_details must be a mapping")
    normalized = dict(payload)
    if validate_required:
        missing = sorted(key for key in REQUIRED_ORCHESTRATOR_KEYS if key not in normalized)
        if missing:
            raise ValueError(f"{context} (task {task_id}): missing required keys {missing}")
    return normalized if mutable else MappingProxyType(normalized)


__all__ = ["coerce_orchestrator_details"]
