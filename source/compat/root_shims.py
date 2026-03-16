"""Helpers for thin root-level compatibility shims."""

from __future__ import annotations

from importlib import import_module

from source.compat._deprecation import warn_compat_deprecation


def load_root_entrypoint(*, legacy_path: str, replacement_path: str):
    warn_compat_deprecation(legacy_path=legacy_path, replacement_path=replacement_path)
    return import_module(replacement_path)


def resolve_root_attr(*, module_name: str, entry_module, module_globals: dict[str, object], name: str):
    if hasattr(entry_module, name):
        value = getattr(entry_module, name)
        module_globals[name] = value
        return value
    raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
