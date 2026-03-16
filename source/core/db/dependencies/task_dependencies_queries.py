"""Compatibility wrappers for dependency-query helpers."""

from __future__ import annotations

from source.core.db import task_dependencies as _task_dependencies

_cfg = _task_dependencies._cfg


def get_task_dependency(task_id: str, max_retries: int = 3, retry_delay: float = 0.5) -> str | None:
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
        return _task_dependencies.get_predecessor_output_via_edge_function(task_id)

    dependency_lookup = dependency_lookup or _task_dependencies.get_task_dependency
    output_lookup = output_lookup or _task_dependencies.get_task_output_location_from_db
    predecessor_id = dependency_lookup(task_id)
    if predecessor_id:
        return predecessor_id, output_lookup(predecessor_id)
    return None, None


__all__ = ["_cfg", "get_task_dependency", "get_predecessor_output_via_edge_function"]
