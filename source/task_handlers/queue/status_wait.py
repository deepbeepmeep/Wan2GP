"""Shared queue-task waiting semantics."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class QueueTaskWaitResult:
    status: str
    result_path: str | None = None
    processing_time: float | None = None
    error_message: str | None = None


def wait_for_queue_task(
    *,
    task_queue,
    task_id: str,
    max_wait_time: float,
    wait_interval: float,
) -> QueueTaskWaitResult:
    start = time.time()
    while (time.time() - start) < max_wait_time:
        status = task_queue.get_task_status(task_id)
        if status is None:
            return QueueTaskWaitResult(status="missing", error_message=f"Task {task_id} became None")
        current = getattr(status, "status", None)
        if current in {"completed", "failed"}:
            return QueueTaskWaitResult(
                status=current,
                result_path=getattr(status, "result_path", None),
                processing_time=getattr(status, "processing_time", None),
                error_message=getattr(status, "error_message", None),
            )
        time.sleep(wait_interval)
    return QueueTaskWaitResult(status="timeout", error_message=f"Task {task_id} timeout")
