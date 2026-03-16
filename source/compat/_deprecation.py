"""Compatibility deprecation helpers."""

from __future__ import annotations

import os
import warnings


def warn_compat_deprecation(*, legacy_path: str, replacement_path: str) -> None:
    if os.environ.get("HEADLESS_COMPAT_DISABLE_SHIMS") == "1":
        raise RuntimeError(f"{legacy_path} is disabled by HEADLESS_COMPAT_DISABLE_SHIMS")
    warnings.warn(
        f"{legacy_path} is deprecated; use {replacement_path} instead. Compatibility shim sunset target is the next major cleanup.",
        DeprecationWarning,
        stacklevel=2,
    )
