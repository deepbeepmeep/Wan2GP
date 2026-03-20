"""
Task claiming and assignment recovery functions.
"""
import os
import sys
import traceback

import httpx

from source.core.log import headless_logger

__all__ = [
    "init_db",
    "init_db_supabase",
    "check_task_counts_supabase",
    "check_my_assigned_tasks",
    "get_oldest_queued_task",
    "get_oldest_queued_task_supabase",
]

from . import config as _cfg


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

        payload = {
            "run_type": run_type,
            "include_active": True
        }

        headless_logger.debug(f"DEBUG check_task_counts_supabase: Calling task-counts at {edge_url}")
        resp = httpx.post(edge_url, json=payload, headers=headers, timeout=10)
        headless_logger.debug(f"Task-counts response status: {resp.status_code}")

        if resp.status_code == 200:
            counts_data = resp.json()
            # Always log a concise summary so we can observe behavior without enabling debug
            try:
                totals = counts_data.get('totals', {})
                headless_logger.debug(f"[TASK_COUNTS] totals={totals} run_type={payload.get('run_type')}")
            except (ValueError, KeyError, TypeError):
                # Fall back to raw text if JSON structure unexpected
                headless_logger.debug(f"[TASK_COUNTS] raw_response={resp.text[:500]}")
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
    headless_logger.debug(f"[RECOVERY] Skipping assigned-task check (PAT auth, heartbeat handles recovery)")
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
                headless_logger.debug(f"[RECOVERY_CHECK] Orchestrator {orchestrator_task_id} has incomplete child {child['id']} (status={status})")
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
            headless_logger.debug(f"[RECOVERY_CHECK] Edge function failed: {e}")

    # Edge function failed or unavailable — assume incomplete (safe default: don't re-run orchestrator)
    headless_logger.debug(f"[RECOVERY_CHECK] Edge function unavailable for orchestrator {orchestrator_task_id}, assuming incomplete children")
    return True

def get_oldest_queued_task():
    """Gets the oldest queued task from Supabase."""
    return get_oldest_queued_task_supabase()

def get_oldest_queued_task_supabase(worker_id: str = None):
    """Fetches the oldest task via Supabase Edge Function. First checks task counts to avoid unnecessary claim attempts."""
    if not _cfg.SUPABASE_URL or not _cfg.SUPABASE_ACCESS_TOKEN:
        headless_logger.error("Supabase URL or access token not configured. Cannot get task.")
        return None

    # Worker ID is required
    if not worker_id:
        headless_logger.error("No worker_id provided to get_oldest_queued_task_supabase")
        return None

    headless_logger.debug(f"DEBUG: Using worker_id: {worker_id}")

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

        headless_logger.debug(f"[CLAIM_DEBUG] Task counts: queued_only={available_tasks}, eligible_queued={eligible_queued}, active_only={active_only}")

        # Log warning if counts are inconsistent
        if eligible_queued > 0 and available_tasks == 0:
            headless_logger.warning(f"Task count inconsistency detected: eligible_queued={eligible_queued} but queued_only={available_tasks}")
            headless_logger.warning(f"This suggests tasks exist but aren't visible as 'Queued' status - possible replication lag or status corruption")
            # Proceed with claim attempt despite queued_only=0 since eligible_queued>0
            headless_logger.debug(f"[CLAIM_DEBUG] Proceeding with claim attempt despite queued_only=0 because eligible_queued={eligible_queued}")
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

            # Pass worker_id and run_type in the request body for edge function to use
            payload = {"worker_id": worker_id, "run_type": "gpu"}

            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=15)
            headless_logger.debug(f"Edge Function response status: {resp.status_code}")

            if resp.status_code == 200:
                task_data = resp.json()
                task_id = task_data.get('task_id', 'unknown')
                task_type = task_data.get('task_type', 'unknown')
                params = task_data.get('params', {})
                segment_index = params.get('segment_index') if isinstance(params, dict) else None

                headless_logger.essential(f"[CLAIM] Claimed task {task_id} (type={task_type}, segment_index={segment_index})", task_id=task_id)
                headless_logger.debug(f"[CLAIM_DEBUG] Full task data: {task_data}")
                return task_data  # Already in the expected format
            elif resp.status_code == 204:
                headless_logger.debug("Edge Function: No queued tasks available")
                return None
            else:
                headless_logger.error(f"[CLAIM] Edge Function returned {resp.status_code}: {resp.text[:500]}")
                return None
        except (httpx.HTTPError, OSError, ValueError) as e_edge:
            # Log visibly - this is a critical failure that can cause orphaned tasks
            headless_logger.error(f"[CLAIM] Edge Function call failed: {e_edge}")
            headless_logger.debug(f"[CLAIM_DEBUG] Exception type: {type(e_edge).__name__}")
            headless_logger.debug(f"[CLAIM_DEBUG] Full traceback: {traceback.format_exc()}")
            return None
    else:
        headless_logger.error("[CLAIM] No edge function URL or auth configuration available for task claiming")
        return None
