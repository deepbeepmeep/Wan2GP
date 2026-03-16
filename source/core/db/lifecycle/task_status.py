"""Compatibility shim for the split lifecycle task-status module."""

from source.core.db import task_status as _task_status
from source.core.db.task_status import *  # noqa: F401,F403

_cfg = _task_status._cfg
STATUS_QUEUED = _task_status.STATUS_QUEUED
