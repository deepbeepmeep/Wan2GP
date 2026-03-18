"""Orchestrator formatter wrapper."""

from debug.formatters import Formatter


def format_orchestrator(status):
    return Formatter.format_orchestrator(status, "text")


__all__ = ["format_orchestrator"]
