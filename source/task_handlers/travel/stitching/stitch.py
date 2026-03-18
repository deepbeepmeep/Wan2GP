"""Compatibility shim for the split travel stitch path."""

from source.task_handlers.travel.stitch import *  # noqa: F401,F403
from source.task_handlers.travel.stitch import _handle_travel_stitch_task

__all__ = ["_handle_travel_stitch_task"]
