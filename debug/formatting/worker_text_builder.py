"""Compatibility wrapper for worker text building."""

from debug.formatters import Formatter


def build_worker_text(info):
    return Formatter.format_worker(info, "text", False)


__all__ = ["build_worker_text"]
