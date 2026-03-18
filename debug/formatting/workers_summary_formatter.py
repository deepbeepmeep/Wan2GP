"""Workers-summary formatter wrapper."""

from debug.formatters import Formatter


def format_workers_summary(summary):
    return Formatter.format_workers_summary(summary, "text")


__all__ = ["format_workers_summary"]
