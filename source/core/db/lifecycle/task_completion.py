"""Compatibility shim for the split lifecycle task-completion module."""

from source.core.db import task_completion as _task_completion
from source.core.db.task_completion import *  # noqa: F401,F403

_cfg = _task_completion._cfg
_call_edge_function_with_retry = _task_completion._call_edge_function_with_retry
STATUS_QUEUED = _task_completion.STATUS_QUEUED


def add_task_to_db(task_payload: dict, task_type_str: str, dependant_on=None, db_path: str | None = None) -> str:
    """Mirror the flat helper while honoring monkeypatches on this shim module."""
    original_retry = _task_completion._call_edge_function_with_retry
    try:
        _task_completion._call_edge_function_with_retry = _call_edge_function_with_retry
        return _task_completion.add_task_to_db(
            task_payload=task_payload,
            task_type_str=task_type_str,
            dependant_on=dependant_on,
            db_path=db_path,
        )
    finally:
        _task_completion._call_edge_function_with_retry = original_retry
