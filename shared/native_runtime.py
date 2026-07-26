from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

_PRELOADED_LIBSTDCXX: str | None = None


def _loaded_libstdcxx_paths() -> list[str]:
    if not sys.platform.startswith("linux"):
        return []
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        return []

    paths: list[str] = []
    try:
        with maps_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                if "libstdc++.so.6" not in line:
                    continue
                path = line.rsplit(None, 1)[-1]
                if path not in paths:
                    paths.append(path)
    except OSError:
        return []
    return paths


def _candidate_libstdcxx_paths() -> list[str]:
    prefixes = [sys.prefix, sys.exec_prefix, os.environ.get("CONDA_PREFIX")]
    paths: list[str] = []
    seen: set[str] = set()
    for prefix in prefixes:
        if not prefix:
            continue
        path = str(Path(prefix) / "lib" / "libstdc++.so.6")
        norm_path = os.path.normcase(os.path.abspath(path))
        if norm_path in seen or not os.path.isfile(path):
            continue
        seen.add(norm_path)
        paths.append(path)
    return paths


def _prepend_library_path(path: str) -> None:
    directory = os.path.dirname(path)
    if not directory:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if directory in parts:
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([directory, *parts])


def preload_preferred_libstdcxx() -> str | None:
    global _PRELOADED_LIBSTDCXX
    if _PRELOADED_LIBSTDCXX is not None:
        return _PRELOADED_LIBSTDCXX
    if not sys.platform.startswith("linux"):
        return None

    loaded_paths = _loaded_libstdcxx_paths()
    if loaded_paths:
        return None

    for path in _candidate_libstdcxx_paths():
        try:
            mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
            ctypes.CDLL(path, mode=mode)
        except OSError:
            continue
        _prepend_library_path(path)
        _PRELOADED_LIBSTDCXX = path
        return path
    return None
