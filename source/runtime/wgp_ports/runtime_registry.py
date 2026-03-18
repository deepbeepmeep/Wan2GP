"""Runtime registry helpers for the vendored WGP module."""

from __future__ import annotations

import sys
import time
from importlib import import_module
from types import MappingProxyType
from typing import Any

from source.core.runtime_paths import ensure_wan2gp_on_path

_STATE: dict[str, Any] = {
    "module_name": "wgp",
    "module_id": None,
    "import_count": 0,
    "last_imported_at": None,
    "last_force_reload": False,
}


class _ReadonlyRuntimeProxy:
    def __init__(self, runtime):
        object.__setattr__(self, "_runtime", runtime)

    def __getattr__(self, name: str):
        value = getattr(object.__getattribute__(self, "_runtime"), name)
        if isinstance(value, dict):
            return MappingProxyType(value)
        return value

    def __setattr__(self, name: str, value):
        raise AttributeError(f"{type(self).__name__} is read-only")


def get_wgp_runtime_state() -> dict[str, Any]:
    return dict(_STATE)


def reset_wgp_runtime_module() -> None:
    sys.modules.pop("wgp", None)
    _STATE.update({
        "module_id": None,
        "last_imported_at": None,
        "last_force_reload": True,
    })


def get_wgp_runtime_module(*, force_reload: bool = False):
    ensure_wan2gp_on_path()
    if force_reload:
        sys.modules.pop("wgp", None)
    module = sys.modules.get("wgp")
    if module is None:
        module = import_module("wgp")
    _STATE["module_id"] = id(module)
    _STATE["import_count"] = int(_STATE["import_count"]) + 1
    _STATE["last_imported_at"] = time.time()
    _STATE["last_force_reload"] = bool(force_reload)
    return _ReadonlyRuntimeProxy(module)


def get_wgp_runtime_module_mutable(*, force_reload: bool = False):
    ensure_wan2gp_on_path()
    if force_reload:
        sys.modules.pop("wgp", None)
    module = sys.modules.get("wgp")
    if module is None:
        module = import_module("wgp")
    _STATE["module_id"] = id(module)
    _STATE["import_count"] = int(_STATE["import_count"]) + 1
    _STATE["last_imported_at"] = time.time()
    _STATE["last_force_reload"] = bool(force_reload)
    return module


def get_wgp_models_def(*, force_reload: bool = False):
    return get_wgp_runtime_module(force_reload=force_reload).models_def


def set_wgp_model_def(model_name: str, model_def: dict[str, Any]) -> dict[str, Any]:
    runtime = get_wgp_runtime_module_mutable()
    runtime.models_def[model_name] = model_def
    if getattr(runtime, "transformer_type", None) == model_name and getattr(runtime, "wan_model", None) is not None:
        runtime.wan_model.model_def = model_def
    return model_def


def set_wgp_reload_needed(value: bool) -> bool:
    runtime = get_wgp_runtime_module_mutable()
    runtime.reload_needed = bool(value)
    return runtime.reload_needed


def upsert_wgp_model_definition(
    model_name: str,
    model_def: dict[str, Any],
    *,
    initialize: bool = False,
) -> dict[str, Any]:
    runtime = get_wgp_runtime_module_mutable()
    final_model_def = dict(model_def)
    if initialize:
        final_model_def = runtime.init_model_def(model_name, final_model_def)
    runtime.models_def[model_name] = final_model_def
    if getattr(runtime, "transformer_type", None) == model_name and getattr(runtime, "wan_model", None) is not None:
        runtime.wan_model.model_def = dict(final_model_def)
    return final_model_def


def load_wgp_runtime_model(*, force_reload: bool = False):
    return get_wgp_runtime_module(force_reload=force_reload)


class _RuntimePatchTransaction:
    def __init__(self, runtime, model_name: str, keys: tuple[str, ...]):
        self.runtime = runtime
        self.model_name = model_name
        self.keys = tuple(keys)
        self.models_def_target = runtime.models_def.setdefault(model_name, {})
        self.loaded_target = None
        if getattr(runtime, "transformer_type", None) == model_name and getattr(runtime, "wan_model", None) is not None:
            self.loaded_target = runtime.wan_model.model_def
        self.snapshot = {}
        for target_name, target in (("models_def", self.models_def_target), ("loaded_model", self.loaded_target)):
            if target is None:
                continue
            for key in self.keys:
                self.snapshot[(target_name, key)] = (key in target, target.get(key))


def begin_runtime_model_patch(model_name: str, *, keys: tuple[str, ...]):
    runtime = get_wgp_runtime_module_mutable()
    return _RuntimePatchTransaction(runtime, model_name, keys)


def apply_runtime_model_patch(transaction: _RuntimePatchTransaction, patch_values: dict[str, Any]) -> None:
    for target in (transaction.models_def_target, transaction.loaded_target):
        if target is None:
            continue
        target.update(patch_values)


def rollback_runtime_model_patch(transaction: _RuntimePatchTransaction) -> None:
    for target_name, target in (("models_def", transaction.models_def_target), ("loaded_model", transaction.loaded_target)):
        if target is None:
            continue
        for key in transaction.keys:
            existed, previous = transaction.snapshot.get((target_name, key), (False, None))
            if existed:
                target[key] = previous
            else:
                target.pop(key, None)
