"""Root compatibility shim for `source.runtime.entrypoints.heartbeat_guardian`."""

from __future__ import annotations

from importlib import import_module

from source.compat._deprecation import warn_compat_deprecation
from source.compat.root_shims import load_root_entrypoint, resolve_root_attr

warn_compat_deprecation(
    legacy_path="heartbeat_guardian",
    replacement_path="source.runtime.entrypoints.heartbeat_guardian",
)


def _entry_module():
    return load_root_entrypoint(
        legacy_path="heartbeat_guardian",
        replacement_path="source.runtime.entrypoints.heartbeat_guardian",
    )


def _legacy_exports_module():
    return import_module("source.runtime.worker.guardian")


def main():
    return _entry_module().main()


def __getattr__(name: str):
    return resolve_root_attr(
        module_name=__name__,
        entry_module=_legacy_exports_module(),
        module_globals=globals(),
        name=name,
    )
