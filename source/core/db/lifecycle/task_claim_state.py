"""State helpers for claim-flow deferral logic."""

from __future__ import annotations

import os

ORCHESTRATOR_DEFERRAL_COUNTS: dict[str, int] = {}


def has_required_edge_credentials(headers: dict[str, str] | None) -> bool:
    if not headers:
        return False
    lowered = {str(key).lower(): value for key, value in headers.items()}
    return bool(lowered.get("authorization"))


def max_orchestrator_deferrals() -> int:
    raw = os.getenv("WAN2GP_ORCHESTRATOR_MAX_DEFERRALS", "5")
    try:
        value = int(raw)
    except ValueError:
        return 5
    return max(1, value)


def register_orchestrator_deferral(task_id: str) -> tuple[bool, int]:
    count = ORCHESTRATOR_DEFERRAL_COUNTS.get(task_id, 0) + 1
    ORCHESTRATOR_DEFERRAL_COUNTS[task_id] = count
    return count >= max_orchestrator_deferrals(), count


def clear_orchestrator_deferral(task_id: str) -> None:
    ORCHESTRATOR_DEFERRAL_COUNTS.pop(task_id, None)
