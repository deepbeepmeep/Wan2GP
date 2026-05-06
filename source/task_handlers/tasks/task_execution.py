"""Execution helpers for backend-neutral resolved direct tasks."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Protocol, Tuple
import time

from source.core.log import headless_logger
from source.task_handlers.tasks.template_routing import (
    ResolvedTask,
    RouteSupportState,
    WorkerBackend,
    routing_telemetry_fields,
)


class _TaskQueue(Protocol):
    def submit_task(self, task: Any) -> Any: ...
    def get_task_status(self, task_id: str) -> Any: ...


WgpTaskBuilder = Callable[[], Any]
VibeComfyHandler = Callable[[ResolvedTask, Any], tuple[bool, str | None]]


def execute_resolved_direct_task(
    *,
    resolved: ResolvedTask,
    context: Mapping[str, Any],
    build_wgp_generation_task: WgpTaskBuilder,
    vibecomfy_handler: VibeComfyHandler | None = None,
    max_wait_time: int = 3600,
    wait_interval: int = 2,
) -> Tuple[bool, Optional[str]]:
    """Execute a resolved direct task through WGP or VibeComfy.

    WGP behavior intentionally mirrors the existing direct queue path: build the
    WGP GenerationTask, submit it to the queue, and poll for completion.
    """

    if resolved.backend == WorkerBackend.WGP:
        _log_routing_card(resolved, decision="wgp_queue")
        return _execute_wgp_direct_task(
            resolved=resolved,
            context=context,
            build_wgp_generation_task=build_wgp_generation_task,
            max_wait_time=max_wait_time,
            wait_interval=wait_interval,
        )

    if resolved.fail_closed_reason:
        _log_routing_card(
            resolved,
            decision="fail_closed",
            fail_closed_reason=resolved.fail_closed_reason,
        )
        return False, _fail_closed_message(resolved)

    if resolved.support_state != RouteSupportState.VIBECOMFY_SUPPORTED:
        _log_routing_card(resolved, decision="fail_closed")
        return False, _fail_closed_message(resolved)

    _log_routing_card(resolved, decision="vibecomfy_adapter")
    handler = vibecomfy_handler or _load_vibecomfy_handler()
    return handler(resolved, context["main_output_dir_base"])


def _execute_wgp_direct_task(
    *,
    resolved: ResolvedTask,
    context: Mapping[str, Any],
    build_wgp_generation_task: WgpTaskBuilder,
    max_wait_time: int,
    wait_interval: int,
) -> Tuple[bool, Optional[str]]:
    task_queue = _require_task_queue(context)
    generation_task = build_wgp_generation_task()
    task_queue.submit_task(generation_task)

    elapsed = 0
    while elapsed < max_wait_time:
        status = task_queue.get_task_status(resolved.task_id)
        if not status:
            return False, "Task status became None"

        if status.status == "completed":
            return True, status.result_path
        if status.status == "failed":
            return False, status.error_message or "Failed without message"

        time.sleep(wait_interval)
        elapsed += wait_interval

    return False, "Timeout"


def _require_task_queue(context: Mapping[str, Any]) -> _TaskQueue:
    task_queue = context.get("task_queue")
    if task_queue is None:
        raise ValueError("Resolved direct WGP execution requires task_queue")
    return task_queue


def _fail_closed_message(resolved: ResolvedTask) -> str:
    reason = resolved.fail_closed_reason or (
        f"Route {resolved.route_key!r} is {resolved.support_state.value}"
    )
    return (
        f"VibeComfy backend fail-closed for task {resolved.task_id} "
        f"({resolved.task_type}): {reason}"
    )


def _log_routing_card(
    resolved: ResolvedTask,
    *,
    decision: str,
    fail_closed_reason: str | None = None,
) -> None:
    payload = routing_telemetry_fields(resolved)
    payload["decision"] = decision
    if fail_closed_reason:
        payload["fail_closed_reason"] = fail_closed_reason
    headless_logger.debug_block(
        "VIBECOMFY_ROUTING",
        payload,
        task_id=resolved.task_id,
    )


def _load_vibecomfy_handler() -> VibeComfyHandler:
    from source.models.comfy.vibecomfy_adapter import handle_vibecomfy_resolved_task

    return handle_vibecomfy_resolved_task


__all__ = [
    "VibeComfyHandler",
    "WgpTaskBuilder",
    "execute_resolved_direct_task",
]
