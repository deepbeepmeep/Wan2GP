"""Module-oriented DB namespace."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "config",
    "task_claim",
    "task_status",
    "task_completion",
    "task_polling",
    "task_dependencies",
    "task_dependencies_children",
    "task_dependencies_queries",
    "task_polling_helpers",
]

_MODULES = {
    "config": "source.core.db.config",
    "task_claim": "source.core.db.task_claim",
    "task_status": "source.core.db.task_status",
    "task_completion": "source.core.db.task_completion",
    "task_polling": "source.core.db.task_polling",
    "task_dependencies": "source.core.db.task_dependencies",
    "task_dependencies_children": "source.core.db.dependencies.task_dependencies_children",
    "task_dependencies_queries": "source.core.db.dependencies.task_dependencies_queries",
    "task_polling_helpers": "source.core.db.lifecycle.task_polling_helpers",
}
_ATTR_EXPORTS = {
    "get_db_runtime_config": ("source.core.db.config", "get_db_runtime_config"),
    "get_db_runtime_registry": ("source.core.db.config", "get_db_runtime_registry"),
    "initialize_db_runtime": ("source.core.db.config", "initialize_db_runtime"),
}


def __getattr__(name: str):
    module_path = _MODULES.get(name)
    if module_path:
        module = import_module(module_path)
        globals()[name] = module
        return module

    attr_export = _ATTR_EXPORTS.get(name)
    if attr_export:
        module = import_module(attr_export[0])
        return getattr(module, attr_export[1])
    raise AttributeError(name)


def __dir__():
    return sorted(set(__all__) | set(_MODULES) | set(_ATTR_EXPORTS))
