"""Canonical travel-segment queue entrypoint."""

from source.task_handlers.tasks.task_registry import _handle_travel_segment_via_queue_impl

__all__ = ["handle_travel_segment_via_queue"]


def handle_travel_segment_via_queue(*args, **kwargs):
    return _handle_travel_segment_via_queue_impl(*args, **kwargs)
