"""Compatibility shim for the split lifecycle task-status module."""

from source.core.db import task_status as _task_status
from source.core.db.task_status import *  # noqa: F401,F403

_cfg = _task_status._cfg
STATUS_QUEUED = _task_status.STATUS_QUEUED
STATUS_IN_PROGRESS = _task_status.STATUS_IN_PROGRESS
STATUS_COMPLETE = _task_status.STATUS_COMPLETE
STATUS_FAILED = _task_status.STATUS_FAILED
_update_task_status_supabase_legacy = _task_status._update_task_status_supabase_legacy


def update_task_status_supabase(*args, **kwargs):
    original = _task_status._update_task_status_supabase_legacy
    try:
        _task_status._update_task_status_supabase_legacy = _update_task_status_supabase_legacy
        return _task_status.update_task_status_supabase(*args, **kwargs)
    finally:
        _task_status._update_task_status_supabase_legacy = original
