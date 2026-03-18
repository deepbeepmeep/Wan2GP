"""Lazy command registry for the debug CLI."""

from __future__ import annotations

from importlib import import_module

_COMMAND_MODULE_PATHS = {
    "task": "debug.commands.task",
    "worker": "debug.commands.worker",
    "tasks": "debug.commands.tasks",
    "workers": "debug.commands.workers",
    "health": "debug.commands.health",
    "orchestrator": "debug.commands.orchestrator",
    "config": "debug.commands.config",
    "runpod": "debug.commands.runpod",
    "storage": "debug.commands.storage",
}


def _load_command_modules():
    return {name: import_module(path) for name, path in _COMMAND_MODULE_PATHS.items()}


COMMAND_MODULES = _load_command_modules()

__all__ = ["COMMAND_MODULES", "_COMMAND_MODULE_PATHS", "_load_command_modules"]
