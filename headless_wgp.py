"""Root compatibility shim for `source.runtime.entrypoints.headless_wgp`."""

from __future__ import annotations

from source.compat._deprecation import warn_compat_deprecation
from source.compat.root_shims import load_root_entrypoint, resolve_root_attr

warn_compat_deprecation(
    legacy_path="headless_wgp",
    replacement_path="source.runtime.entrypoints.headless_wgp",
)


def _entry_module():
    return load_root_entrypoint(
        legacy_path="headless_wgp",
        replacement_path="source.runtime.entrypoints.headless_wgp",
    )


def main() -> int:
    warn_compat_deprecation(
        legacy_path="headless_wgp",
        replacement_path="source.runtime.entrypoints.headless_wgp",
    )
    print("headless_wgp.py is a compatibility shim. Use source.runtime.entrypoints.headless_wgp instead.")
    return 1


def __getattr__(name: str):
    return resolve_root_attr(
        module_name=__name__,
        entry_module=_entry_module(),
        module_globals=globals(),
        name=name,
    )
