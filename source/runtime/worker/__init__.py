"""Runtime worker package."""

from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    if name in {"main", "parse_args"}:
        module = import_module("source.runtime.worker.server")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
