"""
Task claiming and assignment recovery functions.
"""
from enum import Enum
import os
import sys
import traceback

import httpx

from source.core.log import headless_logger
from source.task_handlers.tasks.template_routing import WorkerBackend, parse_worker_backend

__all__ = [
    "ClaimPollOutcome",
    "init_db",
    "init_db_supabase",
    "check_task_counts_supabase",
    "check_my_assigned_tasks",
    "poll_next_task",
]

from . import config as _cfg


class ClaimPollOutcome(str, Enum):
    CLAIMED = "claimed"
    EMPTY = "empty"
    ERROR = "error"


def _selector_namespace() -> str:
    value = (
        os.getenv("REIGH_SELECTOR_NAMESPACE")
        or os.getenv("ROUTE_SELECTOR_NAMESPACE")
        or "production"
    ).strip()
    return value or "production"


def _route_context(task_data: dict | None) -> dict:
    data = task_data or {}
    return {
        "route_key": data.get("route_key"),
        "selected_backend": data.get("selected_backend"),
        "selector_namespace": data.get("selector_namespace"),
        "selector_version": data.get("selector_version"),
        "claimed_backend": data.get("claimed_backend"),
        "claimed_selector_namespace": data.get("claimed_selector_namespace"),
        "claimed_route_key": data.get("claimed_route_key"),
        "claimed_selector_version": data.get("claimed_selector_version"),
        "claimed_capability_version": data.get("claimed_capability_version"),
        "claim_decision_reason": data.get("claim_decision_reason"),
    }


def _int_attempts(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _requeue_backend_mismatch(task_id: str, message: str, attempts: int) -> bool:
    from source.core.db.lifecycle.task_status_retry import requeue_task_for_retry

    return requeue_task_for_retry(
        task_id,
        message,
        attempts,
        error_category="backend_mismatch",
    )


def _fail_closed_claim_decision(task_id: str, message: str, attempts: int) -> bool:
    from source.core.db.lifecycle.task_status_complete_remote import mark_task_failed_via_edge_function
    from source.core.db.lifecycle.task_status_retry import requeue_task_for_retry

    failed = mark_task_failed_via_edge_function(task_id, message)
    if failed:
        return True

    return requeue_task_for_retry(
        task_id,
        message,
        attempts,
        error_category="route_decision_fail_closed",
    )


def _claim_route_guard(task_data: dict, expected_backend: WorkerBackend) -> bool:
    claimed_backend = task_data.get("claimed_backend")
    selected_backend = task_data.get("selected_backend")
    claim_reason = task_data.get("claim_decision_reason")
    task_id = str(task_data.get("task_id") or task_data.get("id") or "unknown")
    context = _route_context(task_data)

    # Older task responses do not include route decision fields. Let them run
    # until the route-aware DB migration is fully deployed.
    if claimed_backend is None and selected_backend is None and claim_reason is None:
        return True

    backend_to_validate = claimed_backend if claimed_backend is not None else selected_backend
    try:
        returned_backend = parse_worker_backend(str(backend_to_validate))
    except ValueError:
        message = (
            "Malformed route/backend claim decision before execution: "
            f"expected={expected_backend.value}, context={context}"
        )
        cleared = _fail_closed_claim_decision(
            task_id,
            message,
            _int_attempts(task_data.get("attempts")),
        )
        headless_logger.error(
            "[CLAIM] Malformed route/backend claim decision; failing closed "
            f"task_id={task_id} cleared={cleared} {message}"
        )
        return False

    if returned_backend != expected_backend:
        message = (
            "Claimed task backend mismatch before execution: "
            f"expected={expected_backend.value}, claimed={returned_backend.value}, context={context}"
        )
        requeued = _requeue_backend_mismatch(task_id, message, _int_attempts(task_data.get("attempts")))
        headless_logger.error(
            "[CLAIM] Backend mismatch guard stopped execution "
            f"task_id={task_id} requeued={requeued} {message}"
        )
        return False

    allowed_reasons = {"eligible", "missing_selector_wgp_capability_supported"}
    if claim_reason is not None and claim_reason not in allowed_reasons:
        message = (
            "Unsupported claim decision reason before execution: "
            f"expected={expected_backend.value}, context={context}"
        )
        cleared = _fail_closed_claim_decision(
            task_id,
            message,
            _int_attempts(task_data.get("attempts")),
        )
        headless_logger.error(
            "[CLAIM] Unsupported claim decision reason; failing closed "
            f"task_id={task_id} cleared={cleared} {message}"
        )
        return False

    return True


def _resolve_runtime_config(runtime_config=None):
    return runtime_config or _cfg.get_db_runtime_config()


def init_db():
    """Initializes the Supabase database connection."""
    return init_db_supabase()

def init_db_supabase():
    """Verify Supabase configuration is present.

    Connectivity is validated implicitly when the first edge function call
    (claim-next-task) succeeds, so we no longer need a direct table query here.
    """
    if not _cfg.SUPABASE_URL or not _cfg.SUPABASE_ACCESS_TOKEN:
        headless_logger.error("Supabase URL or access token not configured. Cannot operate.")
        sys.exit(1)
    headless_logger.essential(f"Supabase: Configuration present (URL={_cfg.SUPABASE_URL[:40]}…)")
    return True

def check_task_counts_supabase(run_type: str = "gpu", runtime_config=None) -> dict | None:
    """Check task counts via Supabase Edge Function before attempting to claim tasks."""
    runtime = _resolve_runtime_config(runtime_config)
    access_token = getattr(runtime, "supabase_access_token", None)
    supabase_url = getattr(runtime, "supabase_url", None)

    if not access_token or not supabase_url:
        headless_logger.error("[TASK_COUNTS] access_token or supabase_url not configured")
        return None

    # Build task-counts edge function URL using same pattern as other functions
    edge_url = (
        os.getenv('SUPABASE_EDGE_TASK_COUNTS_URL')
        or (f"{supabase_url.rstrip('/')}/functions/v1/task-counts" if supabase_url else None)
    )

    if not edge_url:
        headless_logger.error("[TASK_COUNTS] No edge function URL available")
        return None

    try:
        # Use the configured bearer credential for edge access.
        # The edge endpoint determines how the credential is interpreted.
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }

        worker_backend = parse_worker_backend()
        selector_namespace = _selector_namespace()
        payload = {
            "run_type": run_type,
            "include_active": True,
            "worker_backend": worker_backend.value,
            "selector_namespace": selector_namespace,
        }

        headless_logger.debug(f"DEBUG check_task_counts_supabase: Calling task-counts at {edge_url}")
        resp = httpx.post(edge_url, json=payload, headers=headers, timeout=10)
        headless_logger.debug(f"Task-counts response status: {resp.status_code}")

        if resp.status_code == 200:
            counts_data = resp.json()
            # Always log a concise summary so we can observe behavior without enabling debug
            try:
                totals = counts_data.get('totals', {})
                headless_logger.debug_anomaly(
                    "TASK_COUNTS",
                    f"totals={totals} run_type={payload.get('run_type')} "
                    f"worker_backend={worker_backend.value} selector_namespace={selector_namespace}",
                )
            except (ValueError, KeyError, TypeError):
                # Fall back to raw text if JSON structure unexpected
                headless_logger.debug_anomaly("TASK_COUNTS", f"raw_response={resp.text[:500]}")
            headless_logger.debug(f"Task-counts result: {counts_data.get('totals', {})}")
            return counts_data
        else:
            headless_logger.error(f"[TASK_COUNTS] Edge function returned {resp.status_code}: {resp.text[:500]}")
            return None

    except (httpx.HTTPError, OSError, ValueError) as e_counts:
        headless_logger.error(f"[TASK_COUNTS] Call failed: {e_counts}")
        return None

def check_my_assigned_tasks(worker_id: str) -> dict | None:
    """
    Check if this worker has any tasks already assigned to it (In Progress).
    This handles the case where a claim succeeded but the response was lost.

    Returns task data dict if found, None otherwise.

    NOTE: This requires direct DB access which is not available with PAT auth.
    Tasks that lose their HTTP response are recovered by heartbeat timeout instead.
    """
    # Direct DB query not available with PAT auth — heartbeat timeout handles recovery
    headless_logger.debug_anomaly("RECOVERY", f"Skipping assigned-task check (PAT auth, heartbeat handles recovery)")
    return None

def _orchestrator_has_incomplete_children(orchestrator_task_id: str) -> bool:
    """
    Check if an orchestrator has child tasks that are not yet complete.
    Used to prevent re-running orchestrators that are waiting for children.
    """
    def _check_children(children_data: list) -> bool:
        """Check if any child is not complete."""
        for child in children_data:
            status = (child.get("status") or "").lower()
            if status not in ("complete", "failed", "cancelled", "canceled", "error"):
                headless_logger.debug_anomaly("RECOVERY_CHECK", f"Orchestrator {orchestrator_task_id} has incomplete child {child['id']} (status={status})")
                return True
        return False

    # Try the edge route first (works for local workers without direct DB credentials)
    edge_url = (
        os.getenv("SUPABASE_EDGE_GET_ORCHESTRATOR_CHILDREN_URL")
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/get-orchestrator-children" if _cfg.SUPABASE_URL else None)
    )

    if edge_url and _cfg.SUPABASE_ACCESS_TOKEN:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"
        }
        payload = {"orchestrator_task_id": orchestrator_task_id}

        try:
            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                tasks = data.get("tasks", [])
                if not tasks:
                    return False  # No children found
                return _check_children(tasks)
        except (httpx.HTTPError, OSError, ValueError) as e:
            headless_logger.debug_anomaly("RECOVERY_CHECK", f"Edge function failed: {e}")

    # Edge function failed or unavailable — assume incomplete (safe default: don't re-run orchestrator)
    headless_logger.debug_anomaly("RECOVERY_CHECK", f"Edge function unavailable for orchestrator {orchestrator_task_id}, assuming incomplete children")
    return True

def poll_next_task(
    worker_id: str | None,
    same_model_only: bool,
    max_task_wait_minutes: int | None,
) -> tuple[ClaimPollOutcome, dict | None]:
    if not _cfg.SUPABASE_URL or not _cfg.SUPABASE_ACCESS_TOKEN:
        headless_logger.error("Supabase URL or access token not configured. Cannot get task.")
        return ClaimPollOutcome.ERROR, None

    # Worker ID is required
    if not worker_id:
        headless_logger.error("No worker_id provided to get_oldest_queued_task_supabase")
        return ClaimPollOutcome.ERROR, None

    headless_logger.debug(f"DEBUG: Using worker_id: {worker_id}")

    try:
        worker_backend = parse_worker_backend()
    except ValueError as backend_error:
        headless_logger.error(f"[CLAIM] {backend_error}")
        return ClaimPollOutcome.ERROR, None
    selector_namespace = _selector_namespace()

    # OPTIMIZATION: Check task counts first to avoid unnecessary claim attempts
    headless_logger.debug("Checking task counts before attempting to claim...")
    task_counts = check_task_counts_supabase("gpu")

    if task_counts is None:
        headless_logger.debug("WARNING: Could not check task counts, proceeding with direct claim attempt")
    else:
        totals = task_counts.get('totals', {})
        # Gate claim by queued_only to avoid claiming when only active tasks exist
        available_tasks = totals.get('queued_only', 0)
        eligible_queued = totals.get('eligible_queued', 0)
        active_only = totals.get('active_only', 0)

        headless_logger.debug_anomaly("CLAIM_DEBUG", f"Task counts: queued_only={available_tasks}, eligible_queued={eligible_queued}, active_only={active_only}")

        # Log warning if counts are inconsistent
        if eligible_queued > 0 and available_tasks == 0:
            headless_logger.warning(f"Task count inconsistency detected: eligible_queued={eligible_queued} but queued_only={available_tasks}")
            headless_logger.warning(f"This suggests tasks exist but aren't visible as 'Queued' status - possible replication lag or status corruption")
            # Proceed with claim attempt despite queued_only=0 since eligible_queued>0
            headless_logger.debug_anomaly("CLAIM_DEBUG", f"Proceeding with claim attempt despite queued_only=0 because eligible_queued={eligible_queued}")
        elif available_tasks <= 0:
            headless_logger.debug("No non-orchestrator queued tasks according to task-counts, still attempting claim (orchestrators excluded from counts)")
            # Fall through to claim attempt — task-counts excludes orchestrators
        else:
            headless_logger.debug(f"Found {available_tasks} queued tasks, proceeding with claim")

    # Use Edge Function exclusively
    edge_url = (
        _cfg.SUPABASE_EDGE_CLAIM_TASK_URL
        or os.getenv('SUPABASE_EDGE_CLAIM_TASK_URL')
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/claim-next-task" if _cfg.SUPABASE_URL else None)
    )

    if edge_url and _cfg.SUPABASE_ACCESS_TOKEN:
        try:
            headless_logger.debug(f"DEBUG get_oldest_queued_task_supabase: Calling Edge Function at {edge_url}")
            headless_logger.debug(f"DEBUG: Using worker_id: {worker_id}")

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {_cfg.SUPABASE_ACCESS_TOKEN}'
            }

            # Pass worker_id, run_type, and model affinity params to edge function
            payload = {
                "worker_id": worker_id,
                "run_type": "gpu",
                "same_model_only": same_model_only,
                "worker_backend": worker_backend.value,
                "selector_namespace": selector_namespace,
            }
            if max_task_wait_minutes is not None:
                payload["max_task_wait_minutes"] = max_task_wait_minutes

            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=15)
            headless_logger.debug(f"Edge Function response status: {resp.status_code}")

            if resp.status_code == 200:
                task_data = resp.json()
                task_id = task_data.get('task_id', 'unknown')
                task_type = task_data.get('task_type', 'unknown')

                route_context = _route_context(task_data)
                headless_logger.debug(
                    f"Claimed task {task_id} (type={task_type}) route_context={route_context}",
                    task_id=task_id,
                )
                headless_logger.debug_anomaly("CLAIM_DEBUG", f"Full task data: {task_data}")
                if not _claim_route_guard(task_data, worker_backend):
                    return ClaimPollOutcome.ERROR, None
                return ClaimPollOutcome.CLAIMED, task_data
            if resp.status_code == 204:
                headless_logger.debug("Edge Function: No queued tasks available")
                return ClaimPollOutcome.EMPTY, None

            headless_logger.error(f"[CLAIM] Edge Function returned {resp.status_code}: {resp.text[:500]}")
            return ClaimPollOutcome.ERROR, None
        except (httpx.HTTPError, OSError, ValueError) as e_edge:
            # Log visibly - this is a critical failure that can cause orphaned tasks
            headless_logger.error(f"[CLAIM] Edge Function call failed: {e_edge}")
            headless_logger.debug_anomaly("CLAIM_DEBUG", f"Exception type: {type(e_edge).__name__}")
            headless_logger.debug_anomaly("CLAIM_DEBUG", f"Full traceback: {traceback.format_exc()}")
            return ClaimPollOutcome.ERROR, None

    headless_logger.error("[CLAIM] No edge function URL or auth configuration available for task claiming")
    return ClaimPollOutcome.ERROR, None
