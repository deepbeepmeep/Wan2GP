"""Module-oriented visualization namespace."""

from __future__ import annotations

from importlib import import_module

__all__ = ["api"]

_MODULES = {
    "api": "source.media.visualization.api",
    "timeline": "source.media.visualization.timeline",
    "comparison": "source.media.visualization.comparison",
    "layouts": "source.media.visualization.layouts",
}


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(name)
    module = import_module(_MODULES[name])
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(__all__) | set(_MODULES))
