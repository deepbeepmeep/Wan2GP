"""Queue-runtime logging facade."""

from source.core.log.core import (
    flush_log_buffer as _flush_log_buffer,
    queue_logger,
    set_current_task_context as _set_current_task_context,
)


def flush_log_buffer() -> None:
    _flush_log_buffer()


def set_current_task_context(task_id: str | None) -> None:
    _set_current_task_context(task_id)


__all__ = [
    "_flush_log_buffer",
    "_set_current_task_context",
    "flush_log_buffer",
    "queue_logger",
    "set_current_task_context",
]
