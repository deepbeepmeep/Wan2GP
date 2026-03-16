"""Queue-side bootstrap gate helpers."""

from __future__ import annotations

import os

from source.task_handlers.queue.wgp_init import ensure_orchestrator_bootstrap_state


def bootstrap_gate_allows_task_dequeue(queue, worker_name: str, *, sleep_fn=None) -> bool:
    state = ensure_orchestrator_bootstrap_state(queue)
    if state.state == "ready":
        setattr(queue, "_bootstrap_fatal_pause_count", 0)
        return True

    sleeper = sleep_fn or (lambda seconds: None)
    pause_seconds = state.retry_after_seconds or 2.0

    if state.state == "failed_fatal":
        current = int(getattr(queue, "_bootstrap_fatal_pause_count", 0) or 0) + 1
        queue._bootstrap_fatal_pause_count = current
        max_pauses = int(os.environ.get("WAN2GP_BOOTSTRAP_FATAL_MAX_PAUSES", "3"))
        if current >= max_pauses:
            if hasattr(queue, "running"):
                queue.running = False
            shutdown_event = getattr(queue, "shutdown_event", None)
            if shutdown_event is not None:
                shutdown_event.set()
            return False
        sleeper(pause_seconds)
    else:
        queue._bootstrap_fatal_pause_count = 0
        sleeper(pause_seconds)

    return False
