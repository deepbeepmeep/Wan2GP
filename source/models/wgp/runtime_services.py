"""Compatibility runtime-service definitions for WGP orchestrator integration."""

from __future__ import annotations


class DefaultModelRuntime:
    """Minimal default model-runtime collaborator."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator


__all__ = ["DefaultModelRuntime"]
