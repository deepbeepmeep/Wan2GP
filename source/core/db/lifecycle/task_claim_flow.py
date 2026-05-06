"""Injected claim-flow helper used by tests and worker orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

from source.task_handlers.tasks.template_routing import parse_worker_backend


def _worker_contract_version() -> int:
    try:
        return int(os.getenv("REIGH_WORKER_CONTRACT_VERSION", "1"))
    except (TypeError, ValueError):
        return 1


@dataclass(frozen=True)
class TaskClaimFlowDependencies:
    check_my_assigned_tasks: Callable[..., Any]
    check_task_counts_supabase: Callable[..., Any]
    orchestrator_has_incomplete_children: Callable[..., bool]
    register_orchestrator_deferral: Callable[[str], tuple[bool, int]]
    clear_orchestrator_deferral: Callable[[str], None]
    resolve_edge_request: Callable[..., Any]
    call_edge_function_with_retry: Callable[..., Any]
    has_required_edge_credentials: Callable[[dict[str, str]], bool]


def claim_oldest_queued_task(*, worker_id: str, runtime, deps: TaskClaimFlowDependencies):
    assigned = deps.check_my_assigned_tasks(worker_id=worker_id, runtime=runtime)
    deferred = None
    if assigned:
        task_type = (assigned.get("task_type") or "").lower()
        if task_type.endswith("_orchestrator"):
            deferred = assigned
        else:
            return assigned

    counts = deps.check_task_counts_supabase(runtime_config=runtime, run_type="gpu")
    queued_only = ((counts or {}).get("totals") or {}).get("queued_only", 0)
    if queued_only <= 0:
        if deferred is None:
            return None
        task_id = deferred.get("task_id")
        if task_id and deps.orchestrator_has_incomplete_children(task_id):
            return None
        return deferred

    request = deps.resolve_edge_request("claim-next-task", runtime_config=runtime)
    if not request.url or not deps.has_required_edge_credentials(request.headers):
        return None

    response, _error = deps.call_edge_function_with_retry(
        edge_url=request.url,
        payload={
            "worker_id": worker_id,
            "run_type": "gpu",
            "worker_backend": parse_worker_backend().value,
            "worker_profile": (
                os.getenv("REIGH_WORKER_PROFILE")
                or os.getenv("WGP_PROFILE")
                or os.getenv("WORKER_PROFILE")
                or "default"
            ),
            "selector_namespace": (
                os.getenv("REIGH_SELECTOR_NAMESPACE")
                or os.getenv("ROUTE_SELECTOR_NAMESPACE")
                or "production"
            ),
            "selector_version": (
                os.getenv("REIGH_SELECTOR_VERSION")
                or os.getenv("ROUTE_SELECTOR_VERSION")
                or None
            ),
            "worker_contract_version": _worker_contract_version(),
        },
        headers=request.headers,
        function_name="claim-next-task",
    )
    if response and response.status_code == 200:
        return response.json()
    return None
