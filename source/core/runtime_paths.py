"""Runtime path helpers for Wan2GP bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

from source.runtime.process_globals import get_bootstrap_controller, run_bootstrap_once


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_wan2gp_path() -> Path:
    return get_repo_root() / "Wan2GP"


def ensure_wan2gp_on_path() -> str:
    wan2gp_path = str(get_wan2gp_path().resolve())

    def _initializer() -> None:
        if wan2gp_path in sys.path:
            sys.path.remove(wan2gp_path)
        sys.path.insert(0, wan2gp_path)

    run_bootstrap_once(
        "wan2gp.sys_path",
        _initializer,
        version="2026-03-16",
        controller=get_bootstrap_controller("wan2gp.paths"),
    )
    return wan2gp_path
