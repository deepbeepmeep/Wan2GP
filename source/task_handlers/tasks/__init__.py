"""Task definitions, conversion, and registry."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "dispatch_manifest",
    "specialized_dispatch",
    "task_conversion",
    "task_execution",
    "task_registry",
    "task_types",
    "template_routing",
    "travel_segment_types",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"source.task_handlers.tasks.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
