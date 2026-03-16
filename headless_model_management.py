"""Root compatibility shim for headless model management."""

from __future__ import annotations

from importlib import import_module

from source.compat._deprecation import warn_compat_deprecation
from source.compat.root_shims import load_root_entrypoint, resolve_root_attr

warn_compat_deprecation(
    legacy_path="headless_model_management",
    replacement_path="source.runtime.entrypoints.headless_model_management",
)


def _runtime_entry_module():
    return load_root_entrypoint(
        legacy_path="headless_model_management",
        replacement_path="source.runtime.entrypoints.headless_model_management",
    )


def _legacy_exports_module():
    warn_compat_deprecation(
        legacy_path="headless_model_management",
        replacement_path="source.task_handlers.queue.task_queue",
    )
    return import_module("source.task_handlers.queue.task_queue")


def main():
    return _runtime_entry_module().main()


def __getattr__(name: str):
    return resolve_root_attr(
        module_name=__name__,
        entry_module=_legacy_exports_module(),
        module_globals=globals(),
        name=name,
    )
