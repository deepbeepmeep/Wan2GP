"""Compatibility wrappers for dependency-query helpers."""

from __future__ import annotations

from source.core.db import task_dependencies as _task_dependencies
from source.core.db.config import allow_direct_query_fallback, resolve_edge_request

_cfg = _task_dependencies._cfg


def get_task_dependency(task_id: str, max_retries: int = 3, retry_delay: float = 0.5) -> str | None:
    request = resolve_edge_request("get-task-output")
    if not request.url and allow_direct_query_fallback() and _cfg.SUPABASE_CLIENT:
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

    return _task_dependencies.get_task_dependency(
        task_id,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


def get_predecessor_output_via_edge_function(
    task_id: str,
    *,
    dependency_lookup=None,
    output_lookup=None,
) -> tuple[str | None, str | None]:
    """Return predecessor/output, optionally using injected fallback lookups."""
    if dependency_lookup is None and output_lookup is None:
        request = resolve_edge_request("get-predecessor-output")
        if not request.url and allow_direct_query_fallback():
            dependency_lookup = _task_dependencies.get_task_dependency
            output_lookup = _task_dependencies.get_task_output_location_from_db
        else:
            return _task_dependencies.get_predecessor_output_via_edge_function(task_id)

    if dependency_lookup is None and output_lookup is None:
        return _task_dependencies.get_predecessor_output_via_edge_function(task_id)

    dependency_lookup = dependency_lookup or _task_dependencies.get_task_dependency
    output_lookup = output_lookup or _task_dependencies.get_task_output_location_from_db
    predecessor_id = dependency_lookup(task_id)
    if predecessor_id:
        return predecessor_id, output_lookup(predecessor_id)
    return None, None


__all__ = [
    "_cfg",
    "resolve_edge_request",
    "get_task_dependency",
    "get_predecessor_output_via_edge_function",
]
