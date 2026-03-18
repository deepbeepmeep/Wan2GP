"""Shared helpers for example scripts."""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


__all__ = ["repo_root"]
