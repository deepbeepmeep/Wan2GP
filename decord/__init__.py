"""Stub that delegates to the real decord package if installed, otherwise provides minimal stubs."""

from __future__ import annotations

import importlib
import os
import sys

# Find and import the real decord from site-packages by temporarily
# hiding this stub's parent directory from sys.path.
_this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_real_decord = None

_original_path = sys.path[:]
try:
    sys.path = [p for p in sys.path if os.path.abspath(p) != _this_dir]
    # Also remove ourselves from sys.modules so the real one can load
    _self = sys.modules.pop("decord", None)
    try:
        _real_decord = importlib.import_module("decord")
    except ImportError:
        pass
    finally:
        # Put ourselves back regardless
        sys.modules["decord"] = _self if _self is not None else sys.modules.get("decord")
finally:
    sys.path[:] = _original_path

if _real_decord is not None:
    # Re-export everything from the real decord
    for _attr in dir(_real_decord):
        if not _attr.startswith("_"):
            globals()[_attr] = getattr(_real_decord, _attr)
    # Ensure key symbols are always available
    bridge = getattr(_real_decord, "bridge", bridge) if "bridge" in dir(_real_decord) else globals().get("bridge")
    cpu = getattr(_real_decord, "cpu", None)
    VideoReader = getattr(_real_decord, "VideoReader", None)
else:
    # Fallback stubs when decord is truly not installed
    class _Bridge:
        def set_bridge(self, _name: str) -> None:
            return None

    bridge = _Bridge()

    def cpu(_index: int = 0):
        return _index

    class VideoReader:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("decord backend is not installed in this environment")
