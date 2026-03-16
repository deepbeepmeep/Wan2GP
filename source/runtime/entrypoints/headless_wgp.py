"""Runtime entrypoint exposing WanOrchestrator."""

from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    module = import_module("source.models.wgp.orchestrator")
    if hasattr(module, name):
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
