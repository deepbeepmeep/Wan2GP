"""Single runtime boundary for Wan2GP/WGP access."""

from __future__ import annotations

from importlib import import_module

from source.core.runtime_paths import ensure_wan2gp_on_path
from source.runtime.wgp_ports import runtime_registry, vendor_imports

get_wgp_runtime_module = runtime_registry.get_wgp_runtime_module
get_wgp_runtime_module_mutable = runtime_registry.get_wgp_runtime_module_mutable
get_wgp_runtime_state = runtime_registry.get_wgp_runtime_state
reset_wgp_runtime_module = runtime_registry.reset_wgp_runtime_module
set_wgp_model_def = runtime_registry.set_wgp_model_def
set_wgp_reload_needed = runtime_registry.set_wgp_reload_needed
upsert_wgp_model_definition = runtime_registry.upsert_wgp_model_definition
begin_runtime_model_patch = runtime_registry.begin_runtime_model_patch
apply_runtime_model_patch = runtime_registry.apply_runtime_model_patch
rollback_runtime_model_patch = runtime_registry.rollback_runtime_model_patch
load_wgp_runtime_model = runtime_registry.load_wgp_runtime_model

create_qwen_prompt_expander = vendor_imports.create_qwen_prompt_expander
load_uni3c_controlnet = vendor_imports.load_uni3c_controlnet
run_rife_temporal_interpolation = vendor_imports.run_rife_temporal_interpolation


def get_qwen_prompt_expander_class():
    return vendor_imports.get_qwen_prompt_expander_class()


def get_rife_temporal_interpolation():
    return vendor_imports.get_rife_temporal_interpolation()


def get_wan2gp_save_video_callable():
    return vendor_imports.get_wan2gp_save_video_callable()


def clear_uni3c_cache_if_unused() -> bool:
    return vendor_imports.clear_uni3c_cache_if_unused()


def get_model_min_frames_and_step(model_name: str):
    return get_wgp_runtime_module().get_model_min_frames_and_step(model_name)


def get_model_fps(model_name: str):
    return get_wgp_runtime_module().get_model_fps(model_name)


__all__ = [
    "apply_runtime_model_patch",
    "begin_runtime_model_patch",
    "clear_uni3c_cache_if_unused",
    "create_qwen_prompt_expander",
    "get_model_fps",
    "get_model_min_frames_and_step",
    "get_qwen_prompt_expander_class",
    "get_rife_temporal_interpolation",
    "get_wan2gp_save_video_callable",
    "get_wgp_runtime_module",
    "get_wgp_runtime_module_mutable",
    "get_wgp_runtime_state",
    "load_uni3c_controlnet",
    "load_wgp_runtime_model",
    "reset_wgp_runtime_module",
    "rollback_runtime_model_patch",
    "run_rife_temporal_interpolation",
    "set_wgp_model_def",
    "set_wgp_reload_needed",
    "upsert_wgp_model_definition",
]
