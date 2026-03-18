"""Dispatch payload normalization helpers."""

from __future__ import annotations


def normalize_task_dispatch_payload(payload, *, task_id: str) -> dict:
    normalized = dict(payload) if isinstance(payload, dict) else {}
    normalized["task_id"] = task_id
    orchestrator_details = normalized.get("orchestrator_details")
    if isinstance(orchestrator_details, dict):
        normalized["orchestrator_details"] = dict(orchestrator_details)
        normalized["orchestrator_details"].setdefault("orchestrator_task_id", task_id)
    return normalized


__all__ = ["normalize_task_dispatch_payload"]
