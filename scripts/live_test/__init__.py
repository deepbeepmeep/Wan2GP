"""Shared package bootstrap for the live-test harness."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
WORKER_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_ROOT = WORKSPACE_ROOT / "reigh-worker-orchestrator"


def _add_path(path: Path) -> None:
    resolved = str(path.resolve())
    if path.exists() and resolved not in sys.path:
        sys.path.insert(0, resolved)


def ensure_orchestrator_imports() -> None:
    _add_path(ORCHESTRATOR_ROOT)


ensure_orchestrator_imports()

_LAZY_EXPORTS = {
    "DatabaseClient": ("gpu_orchestrator.database", "DatabaseClient"),
    "RunPodConfig": ("runpod_lifecycle", "RunPodConfig"),
    "SSHClient": ("runpod_lifecycle.ssh", "SSHClient"),
    "get_pod_ssh_details": ("runpod_lifecycle.api", "get_pod_ssh_details"),
    "terminate_pod": ("runpod_lifecycle.api", "terminate_pod"),
    "launch": ("runpod_lifecycle", "launch"),
    "get_network_volumes": ("runpod_lifecycle", "get_network_volumes"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "DatabaseClient",
    "ORCHESTRATOR_ROOT",
    "RunPodConfig",
    "SSHClient",
    "WORKER_ROOT",
    "WORKSPACE_ROOT",
    "ensure_orchestrator_imports",
    "get_network_volumes",
    "get_pod_ssh_details",
    "launch",
    "terminate_pod",
]
