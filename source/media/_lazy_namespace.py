"""Shared helpers for module-oriented lazy namespaces."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


def resolve_lazy_submodule(module_map: dict[str, str], name: str):
    """Resolve and import a lazily-mapped submodule."""
    module_path = module_map.get(name)
    if not module_path:
        raise AttributeError(name)
    return import_module(module_path)


def lazy_namespace_dir(module_map: dict[str, str], extra_names: set[str] | None = None):
    """Return a stable sorted directory surface for a lazy namespace."""
    return sorted(set(module_map) | set(extra_names or set()))


MODULE_ROOT = Path(__file__).resolve().parent

__all__ = ["MODULE_ROOT", "resolve_lazy_submodule", "lazy_namespace_dir"]
