"""Compatibility wrappers for dependency query-helper tests and imports."""

from __future__ import annotations

import json
import time
from typing import Any

from postgrest.exceptions import APIError

from source.core.db import task_dependencies as _task_dependencies

_cfg = _task_dependencies._cfg


def _get_task_dependency_direct(
    task_id: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> str | None:
    """Read `dependant_on` directly from the configured task table."""
    if not _cfg.SUPABASE_CLIENT:
        return None

    for attempt in range(max_retries):
        try:
            response = (
                _cfg.SUPABASE_CLIENT.table(_cfg.PG_TABLE_NAME)
                .select("dependant_on")
                .eq("id", task_id)
                .single()
                .execute()
            )
            if response.data:
                return response.data.get("dependant_on")
            return None
        except (APIError, RuntimeError, ValueError, OSError) as exc:
            if "0 rows" in str(exc) and attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None


def _parse_predecessor_edge_response(
    response: Any,
    task_id: str,
) -> tuple[str | None, str | None]:
    """Normalize the edge-function predecessor response contract."""
    if response.status_code == 200:
        payload = response.json()
        if payload is None:
            return None, None
        return payload.get("predecessor_id"), payload.get("output_location")

    if response.status_code == 404:
        return None, None

    raise ValueError(f"get-predecessor-output failed for {task_id}: {response.status_code}")


def _completed_segments_direct_query(run_id: str) -> list[tuple[int | None, str | None]]:
    """Fallback direct query for completed travel segments for one orchestrator run."""
    if not _cfg.SUPABASE_CLIENT:
        return []

    response = (
        _cfg.SUPABASE_CLIENT.table(_cfg.PG_TABLE_NAME)
        .select("params, output_location")
        .execute()
    )

    results: list[tuple[int | None, str | None]] = []
    for row in response.data or []:
        params_raw = row.get("params")
        if params_raw is None:
            continue
        try:
            params_obj = params_raw if isinstance(params_raw, dict) else json.loads(params_raw)
        except (TypeError, json.JSONDecodeError, ValueError):
            continue

        if str(params_obj.get("orchestrator_run_id")) != str(run_id):
            continue

        results.append((params_obj.get("segment_index"), row.get("output_location")))

    return sorted(results, key=lambda item: item[0] if item[0] is not None else 0)


get_task_dependency = _task_dependencies.get_task_dependency
get_predecessor_output_via_edge_function = _task_dependencies.get_predecessor_output_via_edge_function
get_segment_predecessor_output = _task_dependencies.get_segment_predecessor_output
get_completed_segment_outputs_for_stitch = _task_dependencies.get_completed_segment_outputs_for_stitch


__all__ = [
    "_cfg",
    "_get_task_dependency_direct",
    "_parse_predecessor_edge_response",
    "_completed_segments_direct_query",
    "get_task_dependency",
    "get_predecessor_output_via_edge_function",
    "get_segment_predecessor_output",
    "get_completed_segment_outputs_for_stitch",
]
