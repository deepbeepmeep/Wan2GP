"""Worker-side bootstrap gate helpers."""

from __future__ import annotations

from source.task_handlers.queue.wgp_init import ensure_orchestrator_bootstrap_state


def bootstrap_gate_allows_task_claim(*, task_queue, lifecycle, poll_interval: float) -> bool:
    state = ensure_orchestrator_bootstrap_state(task_queue)
    if state.state == "ready":
        return True
    lifecycle.sleep_for(poll_interval)
    return False
