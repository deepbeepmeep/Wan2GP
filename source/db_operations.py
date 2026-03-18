"""Database operations - re-export facade for backward compatibility.

All mutable config (SUPABASE_URL, etc.) lives in source.core.db.config.
Worker.py writes config there directly at startup. Read-only constants and
functions are re-exported here for backward compatibility.
"""

from __future__ import annotations

import inspect
import warnings
from importlib import import_module

warnings.warn(
    "source.db_operations is deprecated; import source.core.db submodules directly.",
    DeprecationWarning,
    stacklevel=2,
)

_COMPAT_MODULES = (
    "source.core.db.config",
    "source.core.db.edge_helpers",
    "source.core.db.task_claim",
    "source.core.db.task_status",
    "source.core.db.task_completion",
    "source.core.db.task_polling",
    "source.core.db.task_dependencies",
)


def _called_from_importlib() -> bool:
    for frame_info in inspect.stack()[1:8]:
        filename = frame_info.filename.replace("\\", "/")
        if "importlib/_bootstrap" in filename or "importlib/_bootstrap_external" in filename:
            return True
        if frame_info.function == "_handle_fromlist":
            return True
    return False


def __getattr__(name: str):
    for module_path in _COMPAT_MODULES:
        module = import_module(module_path)
        if hasattr(module, name):
            value = getattr(module, name)
            if _called_from_importlib():
                return value
            return value
    raise AttributeError(name)


def __dir__():
    names = set()
    for module_path in _COMPAT_MODULES:
        module = import_module(module_path)
        names.update(getattr(module, "__all__", ()))
    return sorted(name for name in names if not str(name).startswith("_"))
