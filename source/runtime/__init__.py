"""Canonical runtime boundary package."""

from importlib import import_module

__all__ = ["bootstrap_service", "wgp_bridge"]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"source.runtime.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
