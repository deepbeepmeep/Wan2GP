"""Minimal stub for optional decord dependency."""

from __future__ import annotations


class _Bridge:
    def set_bridge(self, _name: str) -> None:
        return None


bridge = _Bridge()


def cpu(_index: int = 0):
    return _index


class VideoReader:
    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError("decord backend is not installed in this environment")
