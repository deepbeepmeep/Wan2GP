"""
Task dependency, orchestrator child management, and cross-task queries.
"""
import os
import httpx

from source.core.log import headless_logger

__all__ = [
    "get_task_dependency",
    "get_orchestrator_child_tasks",
    "get_task_current_status",
    "cancel_orchestrator_children",
    "cleanup_duplicate_child_tasks",
    "get_predecessor_output_via_edge_function",
    "get_segment_predecessor_output",
    "get_completed_segment_outputs_for_stitch",
]

from . import config as _cfg
from .edge_helpers import _call_edge_function_with_retry
from .task_status import update_task_status
from .task_polling import get_task_output_location_from_db

def get_task_dependency(task_id: str, max_retries: int = 3, retry_delay: float = 0.5) -> str | None:
    """
    Gets the dependency task ID for a given task ID via edge function.

    Includes retry logic to handle race conditions where a newly created task
    may not be immediately visible in the database.

    Args:
        task_id: Task ID to get dependency for
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Seconds to wait between retries (default: 0.5)

    Returns:
        Dependency task ID or None if no dependency
    """
    headless_logger.debug(f"Fetching dependency for task: {task_id}")

    # Build edge function URL
    edge_url = (
        os.getenv("SUPABASE_EDGE_GET_TASK_OUTPUT_URL")
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/get-task-output" if _cfg.SUPABASE_URL else None)
    )

    if not edge_url or not _cfg.SUPABASE_ACCESS_TOKEN:
        headless_logger.error(f"No edge function URL or access token for get_task_dependency({task_id})")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"
    }

    payload = {"task_id": task_id}

    # Use consistent retry pattern with retry_on_404_patterns for race conditions
    resp, edge_error = _call_edge_function_with_retry(
        edge_url=edge_url,
        payload=payload,
        headers=headers,
        function_name="get-task-output",
        context_id=task_id,
        timeout=30,
        max_retries=max_retries,
        method="POST",
        retry_on_404_patterns=["not found", "Task not found"],  # Handle race conditions
    )

    if resp and resp.status_code == 200:
        data = resp.json()
        return data.get("dependant_on")
    elif edge_error:
        headless_logger.debug(f"Error fetching dependency for {task_id}: {edge_error}")

    return None

def get_orchestrator_child_tasks(orchestrator_task_id: str) -> dict:
    """
    Gets all child tasks for a given orchestrator task ID via edge function.
    Returns dict with task type lists: 'segments', 'stitch', 'join_clips_segment',
    'join_clips_orchestrator', 'join_final_stitch'.
    """
    empty_result = {'segments': [], 'stitch': [], 'join_clips_segment': [], 'join_clips_orchestrator': [], 'join_final_stitch': []}
    headless_logger.debug(f"Fetching child tasks for orchestrator: {orchestrator_task_id}")

    # Build edge function URL
    edge_url = (
        os.getenv("SUPABASE_EDGE_GET_ORCHESTRATOR_CHILDREN_URL")
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/get-orchestrator-children" if _cfg.SUPABASE_URL else None)
    )

    def _categorize_tasks(tasks_data: list) -> dict:
        """Categorize tasks by type."""
        segments = []
        stitch = []
        join_clips_segment = []
        join_clips_orchestrator = []
        join_final_stitch = []

        for task in tasks_data:
            task_data = {
                'id': task['id'],
                'task_type': task['task_type'],
                'status': task['status'],
                'params': task.get('params', {}),
                'task_params': task.get('params', {}),
                'output_location': task.get('output_location', '')
            }
            if task['task_type'] == 'travel_segment':
                segments.append(task_data)
            elif task['task_type'] == 'travel_stitch':
                stitch.append(task_data)
            elif task['task_type'] == 'join_clips_segment':
                join_clips_segment.append(task_data)
            elif task['task_type'] == 'join_clips_orchestrator':
                join_clips_orchestrator.append(task_data)
            elif task['task_type'] == 'join_final_stitch':
                join_final_stitch.append(task_data)

        return {
            'segments': segments,
            'stitch': stitch,
            'join_clips_segment': join_clips_segment,
            'join_clips_orchestrator': join_clips_orchestrator,
            'join_final_stitch': join_final_stitch,
        }

    # Try edge function first
    if edge_url and _cfg.SUPABASE_ACCESS_TOKEN:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"
        }
        payload = {"orchestrator_task_id": orchestrator_task_id}

        try:
            resp, edge_error = _call_edge_function_with_retry(
                edge_url=edge_url,
                payload=payload,
                headers=headers,
                function_name="get-orchestrator-children",
                context_id=orchestrator_task_id,
                timeout=30,
                max_retries=3,
                method="POST",
                retry_on_404_patterns=["not found"],  # Handle race conditions
            )

            if resp and resp.status_code == 200:
                data = resp.json()
                tasks = data.get("tasks", [])
                return _categorize_tasks(tasks)
            elif edge_error:
                headless_logger.debug(f"get-orchestrator-children failed: {edge_error}")
        except (httpx.HTTPError, OSError, ValueError) as e:
            headless_logger.debug(f"Error calling get-orchestrator-children: {e}")

    # No fallback — edge function is the only path
    headless_logger.error(f"Edge function unavailable for get_orchestrator_child_tasks({orchestrator_task_id})", task_id=orchestrator_task_id)
    return empty_result

def get_task_current_status(task_id: str) -> str | None:
    """
    Lightweight status lookup for a single task via the get-task-status edge function.
    Used by orchestrator handlers to detect cancellation before creating child tasks.

    Returns the status string (e.g. "Queued", "In Progress", "Cancelled") or None on error.
    """
    # Build edge function URL
    edge_url = (
        os.getenv("SUPABASE_EDGE_GET_TASK_STATUS_URL")
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/get-task-status" if _cfg.SUPABASE_URL else None)
    )

    # Try edge function first (works for local workers without service key)
    if edge_url and _cfg.SUPABASE_ACCESS_TOKEN:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"
        }
        payload = {"task_id": task_id}

        try:
            resp, edge_error = _call_edge_function_with_retry(
                edge_url=edge_url,
                payload=payload,
                headers=headers,
                function_name="get-task-status",
                context_id=task_id,
                timeout=15,
                max_retries=2,
                method="POST")

            if resp and resp.status_code == 200:
                data = resp.json()
                return data.get("status")
            elif edge_error:
                headless_logger.debug(f"[GET_TASK_STATUS] Edge function failed for {task_id}: {edge_error}")
        except (httpx.HTTPError, OSError, ValueError) as e:
            headless_logger.debug(f"[GET_TASK_STATUS] Error calling get-task-status for {task_id}: {e}")

    # No fallback — edge function is the only path
    return None

def cancel_orchestrator_children(orchestrator_task_id: str, reason: str = "Orchestrator cancelled") -> int:
    """
    Cancel all non-terminal child tasks of an orchestrator.

    Fetches all children via get_orchestrator_child_tasks(), then sets any that are
    still in a non-terminal state (Queued, In Progress) to "Cancelled" via update-task-status.

    Args:
        orchestrator_task_id: The orchestrator task ID whose children should be cancelled.
        reason: Human-readable reason stored in output_location.

    Returns:
        Number of child tasks that were cancelled.
    """
    TERMINAL_STATUSES = {'complete', 'failed', 'cancelled', 'canceled', 'error'}

    child_tasks = get_orchestrator_child_tasks(orchestrator_task_id)

    # Flatten all child task categories into a single list
    all_children = []
    for category in child_tasks.values():
        if isinstance(category, list):
            all_children.extend(category)

    if not all_children:
        headless_logger.debug(f"[CANCEL_CHILDREN] No child tasks found for orchestrator {orchestrator_task_id}")
        return 0

    cancelled_count = 0
    for child in all_children:
        child_id = child.get('id')
        child_status = (child.get('status') or '').lower()

        if child_status in TERMINAL_STATUSES:
            headless_logger.debug(f"[CANCEL_CHILDREN] Skipping child {child_id} (already {child_status})")
            continue

        headless_logger.debug(f"[CANCEL_CHILDREN] Cancelling child task {child_id} (was {child_status})")
        try:
            update_task_status(child_id, "Cancelled", output_location=reason)
            cancelled_count += 1
        except (httpx.HTTPError, OSError, ValueError) as e:
            headless_logger.error(f"[CANCEL_CHILDREN] Failed to cancel child {child_id}: {e}", task_id=child_id)

    if cancelled_count > 0:
        headless_logger.essential(f"[CANCEL_CHILDREN] Cancelled {cancelled_count}/{len(all_children)} child tasks for orchestrator {orchestrator_task_id}", task_id=orchestrator_task_id)
    else:
        headless_logger.debug(f"[CANCEL_CHILDREN] All {len(all_children)} children already in terminal state for orchestrator {orchestrator_task_id}")

    return cancelled_count

def cleanup_duplicate_child_tasks(orchestrator_task_id: str, expected_segments: int) -> dict:
    """
    Detects and removes duplicate child tasks for an orchestrator.
    Returns summary of cleanup actions.
    """
    child_tasks = get_orchestrator_child_tasks(orchestrator_task_id)
    segments = child_tasks['segments']
    stitch_tasks = child_tasks['stitch']

    cleanup_summary = {
        'duplicate_segments_removed': 0,
        'duplicate_stitch_removed': 0,
        'errors': []
    }

    try:
        # Remove duplicate segments (keep the oldest for each segment_index)
        segment_by_index = {}
        for segment in segments:
            segment_idx = segment['params'].get('segment_index', -1)
            if segment_idx in segment_by_index:
                # We have a duplicate - keep the older one (first created)
                existing = segment_by_index[segment_idx]
                duplicate_id = segment['id']

                headless_logger.debug(f"[IDEMPOTENCY] Found duplicate segment {segment_idx}: keeping {existing['id']}, removing {duplicate_id}")

                # Remove the duplicate
                if _delete_task_by_id(duplicate_id):
                    cleanup_summary['duplicate_segments_removed'] += 1
                else:
                    cleanup_summary['errors'].append(f"Failed to delete duplicate segment {duplicate_id}")
            else:
                segment_by_index[segment_idx] = segment

        # Remove duplicate stitch tasks (should only be 1)
        if len(stitch_tasks) > 1:
            # Keep the oldest stitch task, remove others
            stitch_sorted = sorted(stitch_tasks, key=lambda x: x.get('created_at', ''))
            for stitch in stitch_sorted[1:]:  # Remove all but first
                duplicate_id = stitch['id']
                headless_logger.debug(f"[IDEMPOTENCY] Found duplicate stitch task: removing {duplicate_id}")

                if _delete_task_by_id(duplicate_id):
                    cleanup_summary['duplicate_stitch_removed'] += 1
                else:
                    cleanup_summary['errors'].append(f"Failed to delete duplicate stitch {duplicate_id}")

    except (ValueError, KeyError, TypeError) as e:
        cleanup_summary['errors'].append(f"Cleanup error: {str(e)}")
        headless_logger.debug(f"Error during duplicate cleanup: {e}", exc_info=True)

    return cleanup_summary

def _delete_task_by_id(task_id: str) -> bool:
    """Helper to delete a task by ID from Supabase.

    No edge function equivalent exists for deletion. This is only used for rare
    orchestrator duplicate cleanup. With PAT auth, log and skip.
    """
    headless_logger.warning(f"[CLEANUP] Task deletion skipped (no direct DB access): {task_id}", task_id=task_id)
    return False


def _fallback_predecessor_output_lookup(task_id: str) -> tuple[str | None, str | None]:
    predecessor_id = get_task_dependency(task_id)
    if predecessor_id:
        output_location = get_task_output_location_from_db(predecessor_id)
        return predecessor_id, output_location
    return None, None


def _resolve_segment_branch_value(params_obj: dict, key: str) -> str | None:
    for source in (
        params_obj,
        params_obj.get("individual_segment_params") or {},
        params_obj.get("orchestrator_details") or {},
    ):
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None

def get_predecessor_output_via_edge_function(
    task_id: str,
    *,
    parent_generation_id: str | None = None,
    child_order: int | None = None,
) -> tuple[str | None, str | None]:
    """
    Gets predecessor task/generation ID and output location via edge function.

    Supports two lookup paths (handled server-side):
    1. dependant_on chain — for orchestrator-created tasks
    2. Generation sibling lookup — for individual segment regens (needs parent_generation_id + child_order)

    Returns:
        (predecessor_id, output_location) or (None, None)
    """
    if not _cfg.SUPABASE_URL or not _cfg.SUPABASE_ACCESS_TOKEN:
        headless_logger.error("Supabase configuration incomplete. Falling back to direct queries.", task_id=task_id)
        return _fallback_predecessor_output_lookup(task_id)

    edge_url = f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/get-predecessor-output"

    payload: dict[str, object] = {"task_id": task_id}
    if parent_generation_id:
        payload["parent_generation_id"] = parent_generation_id
    if child_order is not None:
        payload["child_order"] = child_order

    try:
        headless_logger.debug(f"Calling Edge Function: {edge_url} for task {task_id}")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {_cfg.SUPABASE_ACCESS_TOKEN}'
        }

        resp = httpx.post(edge_url, json=payload, headers=headers, timeout=15)
        headless_logger.debug(f"Edge Function response status: {resp.status_code}")

        if resp.status_code == 200:
            result = resp.json()
            headless_logger.debug(f"Edge Function result: {result}")

            if result is None:
                return None, None

            predecessor_id = result.get("predecessor_id")
            output_location = result.get("output_location")
            return predecessor_id, output_location

        if resp.status_code == 404:
            headless_logger.debug(f"Edge Function: Task {task_id} not found")
            return None, None

        headless_logger.debug(
            f"Edge Function returned {resp.status_code}: {resp.text}. Falling back to direct queries."
        )
        return _fallback_predecessor_output_lookup(task_id)

    except (httpx.HTTPError, OSError, ValueError) as e_edge:
        headless_logger.debug(f"Edge Function call failed: {e_edge}. Falling back to direct queries.")
        return _fallback_predecessor_output_lookup(task_id)


def get_segment_predecessor_output(
    *,
    task_id: str,
    parent_generation_id: str | None = None,
    child_generation_id: str | None = None,
    child_order: int | None = None,
    segment_index: int | None = None,
) -> tuple[str | None, str | None]:
    """Resolve the previous segment output for chained travel generation.

    Uses the get-predecessor-output edge function which supports two paths:
    1. dependant_on chain — for orchestrator-created tasks
    2. Generation sibling lookup — for individual segment regens (parent_generation_id + child_order)

    All lookups go through the edge function to work with PAT auth.
    """
    # Resolve the effective child_order for the edge function
    effective_child_order = child_order if isinstance(child_order, int) else (
        segment_index if isinstance(segment_index, int) else None
    )

    return get_predecessor_output_via_edge_function(
        task_id,
        parent_generation_id=parent_generation_id,
        child_order=effective_child_order,
    )


def get_completed_segment_outputs_for_stitch(run_id: str, project_id: str | None = None) -> list:
    """Gets completed travel_segment outputs for a given run_id for stitching from Supabase."""
    if not _cfg.SUPABASE_URL or not _cfg.SUPABASE_ACCESS_TOKEN:
        headless_logger.error("Supabase configuration incomplete. Cannot get completed segments.")
        return []

    edge_url = f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/get-completed-segments"
    try:
        headless_logger.debug(f"Calling Edge Function: {edge_url} for run_id {run_id}, project_id {project_id}")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {_cfg.SUPABASE_ACCESS_TOKEN}'
        }
        payload = {"run_id": run_id}
        if project_id:
            payload["project_id"] = project_id

        resp = httpx.post(edge_url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            results = resp.json()
            sorted_results = sorted(results, key=lambda x: x['segment_index'])
            return [(r['segment_index'], r['output_location']) for r in sorted_results]
        else:
            headless_logger.debug(f"Edge Function returned {resp.status_code}: {resp.text}. Falling back to direct query.")
    except (httpx.HTTPError, OSError, ValueError) as e:
        headless_logger.debug(f"Edge Function failed: {e}. Falling back to direct query.")

    # No fallback — edge function is the only path for completed segment queries
    headless_logger.error(f"Edge function failed for get_completed_segment_outputs_for_stitch(run_id={run_id})")
    return []
