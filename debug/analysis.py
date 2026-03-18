"""Small analysis helpers for debug summary views."""

from __future__ import annotations

from collections import Counter
from datetime import datetime


def _parse_ts(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_task_summary_metrics(tasks: list[dict]):
    status_dist = dict(Counter(task.get("status") for task in tasks))
    type_dist = dict(Counter(task.get("task_type") for task in tasks))

    processing_times: list[float] = []
    queue_times: list[float] = []
    for task in tasks:
        try:
            started = _parse_ts(task["generation_started_at"])
            processed = _parse_ts(task["generation_processed_at"])
            processing_times.append((processed - started).total_seconds())
        except Exception:
            pass
        try:
            created = _parse_ts(task["created_at"])
            started = _parse_ts(task["generation_started_at"])
            queue_times.append((started - created).total_seconds())
        except Exception:
            pass

    return (
        status_dist,
        type_dist,
        {
            "avg_processing_seconds": (
                sum(processing_times) / len(processing_times) if processing_times else None
            ),
            "avg_queue_seconds": sum(queue_times) / len(queue_times) if queue_times else None,
            "total_with_timing": len(processing_times),
        },
    )


__all__ = ["build_task_summary_metrics"]
