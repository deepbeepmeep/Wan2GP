"""Module-oriented VLM namespace."""

from __future__ import annotations

from importlib import import_module

__all__ = ["api"]

_MODULES = {
    "api": "source.media.vlm.api",
    "model": "source.media.vlm.model",
    "debug_artifacts": "source.media.vlm.debug_artifacts",
}


# Re-export commonly used functions so callers can do:
#   from source.media.vlm import generate_transition_prompts_batch
_FUNCTION_REEXPORTS = {
    "generate_transition_prompts_batch": ("source.media.vlm.api", "generate_transition_prompts_batch"),
    "generate_transition_prompt": ("source.media.vlm.api", "generate_transition_prompt"),
}


def __getattr__(name: str):
    if name in _MODULES:
        module = import_module(_MODULES[name])
        globals()[name] = module
        return module
    if name in _FUNCTION_REEXPORTS:
        mod_path, func_name = _FUNCTION_REEXPORTS[name]
        module = import_module(mod_path)
        func = getattr(module, func_name)
        globals()[name] = func
        return func
    raise AttributeError(name)


def __dir__():
    return sorted(set(__all__) | set(_MODULES) | set(_FUNCTION_REEXPORTS))
