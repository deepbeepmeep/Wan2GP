"""Tasks-summary formatter wrapper."""

from debug.formatters import Formatter


def format_tasks_summary(summary):
    return Formatter.format_tasks_summary(summary, "text")


__all__ = ["format_tasks_summary"]
