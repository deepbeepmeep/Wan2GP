"""Shared subprocess helpers with normalized binary resolution."""

from __future__ import annotations

import shutil
import subprocess

CalledProcessError = subprocess.CalledProcessError
TimeoutExpired = subprocess.TimeoutExpired


def _normalize_command(command):
    if not command:
        raise ValueError("empty command")
    normalized = list(command)
    resolved = shutil.which(normalized[0])
    if resolved:
        normalized[0] = resolved
    return normalized


def run_subprocess(command, timeout=None, **kwargs):
    return subprocess.run(_normalize_command(command), timeout=timeout, **kwargs)


def popen_subprocess(command, **kwargs):
    return subprocess.Popen(_normalize_command(command), **kwargs)


__all__ = [
    "CalledProcessError",
    "TimeoutExpired",
    "_normalize_command",
    "popen_subprocess",
    "run_subprocess",
]
