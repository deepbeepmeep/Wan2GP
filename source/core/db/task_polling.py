"""
Task polling, output queries, and parameter retrieval.
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from postgrest.exceptions import APIError

from source.core.log import headless_logger

__all__ = [
    "poll_task_status",
    "poll_task_status_result",
    "get_task_output_location_from_db",
    "get_task_output_location_from_db_result",
    "get_task_params",
    "get_task_params_result",
    "get_abs_path_from_db_path",
    "query_task_status",
    "evaluate_polled_status",
    "_TaskPollOptions",
]

from . import config as _cfg
from .config import (
    DBRuntimeContractError,
    STATUS_QUEUED,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETE,
    STATUS_FAILED,
    allow_direct_query_fallback,
    build_edge_headers,
    resolve_edge_auth_token,
    resolve_edge_function_url)
from .edge_helpers import _call_edge_function_with_retry


@dataclass(frozen=True)
class _TaskPollOptions:
    poll_interval_seconds: int = 10
    timeout_seconds: int = 1800
    db_path: str | None = None


@dataclass(frozen=True)
class TaskPollResult:
    outcome: str
    output_location: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class TaskOutputLocationResult:
    outcome: str
    output_location: str | None = None
    error: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class TaskParamsResult:
    outcome: str
    params: dict | None = None
    error: str | None = None
    source: str | None = None


def _resolve_runtime_config(runtime_config=None):
    return runtime_config or _cfg.get_db_runtime_config()


def _normalize_task_params(params_payload):
    if params_payload is None:
        return None
    if isinstance(params_payload, dict):
        return params_payload
    if isinstance(params_payload, str):
        try:
            parsed = json.loads(params_payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def query_task_status(task_id: str, runtime_config=None) -> tuple[str | None, str | None]:
    runtime = _resolve_runtime_config(runtime_config)
    client = getattr(runtime, "supabase_client", None)
    table_name = getattr(runtime, "pg_table_name", _cfg.PG_TABLE_NAME)
    if not client:
        return None, None

    response = client.table(table_name).select("status, output_location").eq("id", task_id).single().execute()
    if not response.data:
        return None, None
    return response.data.get("status"), response.data.get("output_location")


def evaluate_polled_status(
    *,
    task_id: str,
    status: str | None,
    output_location: str | None,
    current_time: float,
    last_status_print_time: float,
    poll_interval_seconds: int,
):
    next_print = last_status_print_time
    if current_time - last_status_print_time > poll_interval_seconds * 2:
        if status:
            headless_logger.essential(
                f"Task {task_id}: Status = {status} (Output: {output_location if output_location else 'N/A'})",
                task_id=task_id,
            )
        else:
            headless_logger.essential(
                f"Task {task_id}: Not found in DB yet or status pending...",
                task_id=task_id,
            )
        next_print = current_time

    if status == STATUS_COMPLETE:
        if output_location:
            headless_logger.essential(f"Task {task_id} completed successfully. Output: {output_location}", task_id=task_id)
            return True, output_location, next_print
        headless_logger.error(f"Task {task_id} is COMPLETE but output_location is missing. Assuming failure.", task_id=task_id)
        return True, None, next_print

    if status == STATUS_FAILED:
        headless_logger.error(f"Task {task_id} failed. Error details: {output_location}", task_id=task_id)
        return True, None, next_print

    if status and status not in [STATUS_QUEUED, STATUS_IN_PROGRESS]:
        headless_logger.warning(f"Task {task_id} has unknown status '{status}'. Treating as error.", task_id=task_id)
        return True, None, next_print

    return False, None, next_print

def poll_task_status(task_id: str, poll_interval_seconds: int = 10, timeout_seconds: int = 1800, db_path: str | None = None) -> str | None:
    """
    Polls Supabase for task completion and returns the output_location.

    Args:
        task_id: Task ID to poll
        poll_interval_seconds: Seconds between polls
        timeout_seconds: Maximum time to wait
        db_path: Ignored (kept for API compatibility)

    Returns:
        Output location string if successful, None otherwise
    """
    headless_logger.essential(f"Polling for completion of task {task_id} (timeout: {timeout_seconds}s)...", task_id=task_id)
    start_time = time.time()
    last_status_print_time = 0

    while True:
        current_time = time.time()
        if current_time - start_time > timeout_seconds:
            headless_logger.error(f"Timeout polling for task {task_id} after {timeout_seconds} seconds.", task_id=task_id)
            return None

        try:
            status, output_location = query_task_status(task_id)
        except (APIError, httpx.HTTPError, OSError, ValueError, KeyError) as e:
            headless_logger.error(f"Supabase error while polling task {task_id}: {e}. Retrying...", task_id=task_id)
            status, output_location = None, None

        done, result, last_status_print_time = evaluate_polled_status(
            task_id=task_id,
            status=status,
            output_location=output_location,
            current_time=current_time,
            last_status_print_time=last_status_print_time,
            poll_interval_seconds=poll_interval_seconds,
        )
        if done:
            return result

        time.sleep(poll_interval_seconds)


def poll_task_status_result(task_id: str, *, options: _TaskPollOptions | None = None, runtime_config=None) -> TaskPollResult:
    options = options or _TaskPollOptions()
    if options.db_path is not None:
        raise TypeError("poll_task_status_result no longer accepts db_path")

    start_time = time.time()
    last_status_print_time = 0.0
    while True:
        current_time = time.time()
        if current_time - start_time > options.timeout_seconds:
            return TaskPollResult(outcome="timeout", error=f"timed_out:{task_id}")
        try:
            status, output_location = query_task_status(task_id, runtime_config=runtime_config)
        except DBRuntimeContractError as exc:
            return TaskPollResult(outcome="precondition_failed", error=str(exc))
        except (APIError, httpx.HTTPError, OSError, ValueError, KeyError) as exc:
            return TaskPollResult(outcome="error", error=str(exc))

        done, result, last_status_print_time = evaluate_polled_status(
            task_id=task_id,
            status=status,
            output_location=output_location,
            current_time=current_time,
            last_status_print_time=last_status_print_time,
            poll_interval_seconds=options.poll_interval_seconds,
        )
        if done:
            return TaskPollResult(
                outcome="completed" if result else "failed",
                output_location=result,
                error=None if result else f"task_not_complete:{task_id}",
            )

        time.sleep(options.poll_interval_seconds)

# Helper to query DB for a specific task's output (needed by segment handler)
def get_task_output_location_from_db(task_id_to_find: str, runtime_config=None) -> str | None:
    """
    Fetches a task's output location via the get-task-output Edge Function.

    This uses an edge function instead of direct DB query to work with
    workers that only have anon key access (RLS would block direct queries).

    Args:
        task_id_to_find: Task ID to look up

    Returns:
        Output location string if task is complete, None otherwise
    """
    headless_logger.debug(f"Fetching output location for task: {task_id_to_find}")

    # Build edge function URL
    runtime = _resolve_runtime_config(runtime_config)
    edge_url = os.getenv("SUPABASE_EDGE_GET_TASK_OUTPUT_URL") or resolve_edge_function_url(
        "get-task-output",
        runtime_config=runtime,
    )

    if not edge_url:
        headless_logger.error(f"No edge function URL available for get-task-output", task_id=task_id_to_find)
        return None

    token = resolve_edge_auth_token(runtime_config=runtime)
    if not token:
        headless_logger.error(f"No auth configuration available for get-task-output", task_id=task_id_to_find)
        return None

    headers = build_edge_headers(token, include_apikey=False)

    payload = {"task_id": task_id_to_find}

    try:
        resp, edge_error = _call_edge_function_with_retry(
            edge_url=edge_url,
            payload=payload,
            headers=headers,
            function_name="get-task-output",
            context_id=task_id_to_find,
            timeout=30,
            max_retries=3,
            method="POST",
            retry_on_404_patterns=["not found", "Task not found"],  # Handle race conditions
        )

        if edge_error:
            headless_logger.error(f"get-task-output failed for {task_id_to_find}: {edge_error}", task_id=task_id_to_find)
            return None

        if resp and resp.status_code == 200:
            data = resp.json()
            status = data.get("status")
            output_location = data.get("output_location")

            if status == STATUS_COMPLETE and output_location:
                headless_logger.debug(f"Task {task_id_to_find} output fetched successfully")
                return output_location
            else:
                headless_logger.debug(f"Task {task_id_to_find} not complete or no output. Status: {status}")
                return None
        elif resp and resp.status_code == 404:
            headless_logger.debug(f"Task {task_id_to_find} not found")
            return None
        else:
            status_code = resp.status_code if resp else "no response"
            headless_logger.error(f"get-task-output unexpected response for {task_id_to_find}: {status_code}", task_id=task_id_to_find)
            return None

    except (httpx.HTTPError, OSError, ValueError) as e:
        headless_logger.error(f"get-task-output exception for {task_id_to_find}: {e}", task_id=task_id_to_find, exc_info=True)
        return None

def get_task_output_location_from_db_result(task_id_to_find: str, runtime_config=None) -> TaskOutputLocationResult:
    runtime = _resolve_runtime_config(runtime_config)
    edge_url = os.getenv("SUPABASE_EDGE_GET_TASK_OUTPUT_URL") or resolve_edge_function_url(
        "get-task-output",
        runtime_config=runtime,
    )
    if not edge_url:
        return TaskOutputLocationResult(
            outcome="unavailable",
            error="No edge function URL available for get-task-output",
        )

    output_location = get_task_output_location_from_db(task_id_to_find, runtime_config=runtime)
    if output_location:
        return TaskOutputLocationResult(outcome="ok", output_location=output_location, source="edge")
    return TaskOutputLocationResult(outcome="missing", error=f"task_output_missing:{task_id_to_find}", source="edge")


def get_task_params(task_id: str, runtime_config=None) -> dict | None:
    """Gets and normalizes params for a given task ID."""
    return get_task_params_result(task_id, runtime_config=runtime_config).params


def get_task_params_result(task_id: str, runtime_config=None) -> TaskParamsResult:
    """Gets the normalized params payload for a given task ID."""
    headless_logger.debug(f"Fetching params for task: {task_id}")
    runtime = _resolve_runtime_config(runtime_config)
    client = getattr(runtime, "supabase_client", None)
    table_name = getattr(runtime, "pg_table_name", _cfg.PG_TABLE_NAME)

    # Build edge function URL
    edge_url = os.getenv("SUPABASE_EDGE_GET_TASK_OUTPUT_URL") or resolve_edge_function_url(
        "get-task-output",
        runtime_config=runtime,
    )
    token = resolve_edge_auth_token(runtime_config=runtime)

    if edge_url and token:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        payload = {"task_id": task_id}
        try:
            resp, edge_error = _call_edge_function_with_retry(
                edge_url=edge_url,
                payload=payload,
                headers=headers,
                function_name="get-task-output",
                context_id=task_id,
                timeout=30,
                max_retries=3,
                method="POST",
                retry_on_404_patterns=["not found", "Task not found"],
            )

            if edge_error:
                headless_logger.debug(f"get-task-output (params) failed for {task_id}: {edge_error}")
                return TaskParamsResult(outcome="error", error=edge_error, source="edge")

            if resp and resp.status_code == 200:
                data = resp.json()
                params = _normalize_task_params(data.get("params"))
                if params is None:
                    return TaskParamsResult(outcome="error", error="invalid_params_payload", source="edge")
                return TaskParamsResult(outcome="ok", params=params, source="edge")
            return TaskParamsResult(outcome="missing", error=f"task_params_missing:{task_id}", source="edge")

        except (httpx.HTTPError, OSError, ValueError) as e:
            headless_logger.debug(f"Error getting task params for {task_id}: {e}")
            return TaskParamsResult(outcome="error", error=str(e), source="edge")

    if client and (allow_direct_query_fallback() or edge_url is None):
        try:
            resp = client.table(table_name).select("params").eq("id", task_id).single().execute()
            params = _normalize_task_params(resp.data.get("params") if resp.data else None)
            if params is None:
                return TaskParamsResult(outcome="error", error="invalid_params_payload", source="direct")
            return TaskParamsResult(outcome="ok", params=params, source="direct")
        except (APIError, RuntimeError, ValueError, OSError) as e:
            headless_logger.debug(f"Error getting task params for {task_id}: {e}")
            return TaskParamsResult(outcome="error", error=str(e), source="direct")

    raise DBRuntimeContractError("No supported query path available for get_task_params")

def get_abs_path_from_db_path(db_path: str) -> Path | None:
    """
    Helper to resolve a path from the DB to a usable absolute path.
    Assumes paths from Supabase are already absolute or valid URLs.
    """
    if not db_path:
        return None

    if str(db_path).startswith(("http://", "https://")):
        return None

    # Path from DB is assumed to be absolute (Supabase) or a URL
    resolved_path = Path(db_path).resolve()

    if resolved_path and resolved_path.exists():
        return resolved_path
    else:
        headless_logger.debug(f"Warning: Resolved path '{resolved_path}' from DB path '{db_path}' does not exist.")
        return None
