"""Compatibility shim for the split lifecycle task-claim module."""

from source.core.db import task_claim as _task_claim
from source.core.db.task_claim import *  # noqa: F401,F403

_resolve_runtime_config = _task_claim._resolve_runtime_config


def check_task_counts_supabase(*args, **kwargs):
    original = _task_claim._resolve_runtime_config
    try:
        _task_claim._resolve_runtime_config = _resolve_runtime_config
        return _task_claim.check_task_counts_supabase(*args, **kwargs)
    finally:
        _task_claim._resolve_runtime_config = original
