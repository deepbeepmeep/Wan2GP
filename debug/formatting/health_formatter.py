"""Health formatter wrapper."""

from debug.formatters import Formatter


def format_health(health):
    return Formatter.format_health(health, "text")


__all__ = ["format_health"]
