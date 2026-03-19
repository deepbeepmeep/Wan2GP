"""Single runtime boundary for Wan2GP/WGP access."""

from __future__ import annotations

from importlib import import_module

from source.core.runtime_paths import ensure_wan2gp_on_path
from source.runtime.wgp_ports import runtime_registry, vendor_imports

get_wgp_runtime_module = runtime_registry.get_wgp_runtime_module
get_wgp_runtime_module_mutable = runtime_registry.get_wgp_runtime_module_mutable
get_wgp_runtime_state = runtime_registry.get_wgp_runtime_state
reset_wgp_runtime_module = runtime_registry.reset_wgp_runtime_module
clear_wgp_loaded_model_state = runtime_registry.clear_wgp_loaded_model_state
set_wgp_model_def = runtime_registry.set_wgp_model_def
set_wgp_loaded_model_state = runtime_registry.set_wgp_loaded_model_state
set_wgp_reload_needed = runtime_registry.set_wgp_reload_needed
upsert_wgp_model_definition = runtime_registry.upsert_wgp_model_definition
begin_runtime_model_patch = runtime_registry.begin_runtime_model_patch
apply_runtime_model_patch = runtime_registry.apply_runtime_model_patch
rollback_runtime_model_patch = runtime_registry.rollback_runtime_model_patch
load_wgp_runtime_model = runtime_registry.load_wgp_runtime_model

create_qwen_prompt_expander = vendor_imports.create_qwen_prompt_expander
get_canny_video_annotator_class = vendor_imports.get_canny_video_annotator_class
get_depth_v2_video_annotator_class = vendor_imports.get_depth_v2_video_annotator_class
get_flow_annotator_class = vendor_imports.get_flow_annotator_class
get_flow_viz_module = vendor_imports.get_flow_viz_module
get_pose_body_face_video_annotator_class = vendor_imports.get_pose_body_face_video_annotator_class
get_qwen_family_handler = vendor_imports.get_qwen_family_handler
get_qwen_main_module = vendor_imports.get_qwen_main_module
get_shared_lora_utils_module = vendor_imports.get_shared_lora_utils_module
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


def get_model_def(model_name: str):
    return get_wgp_runtime_module().get_model_def(model_name)


def get_default_settings(model_name: str):
    return get_wgp_runtime_module().get_default_settings(model_name)


def get_lora_dir(model_name: str):
    return get_wgp_runtime_module().get_lora_dir(model_name)


def get_model_name(model_name: str):
    return get_wgp_runtime_module().get_model_name(model_name)


def get_model_min_frames_and_step(model_name: str):
    return get_wgp_runtime_module().get_model_min_frames_and_step(model_name)


def get_model_fps(model_name: str):
    return get_wgp_runtime_module().get_model_fps(model_name)


def get_model_recursive_prop(model_name: str, prop_name: str, **kwargs):
    return get_wgp_runtime_module().get_model_recursive_prop(model_name, prop_name, **kwargs)


def parse_loras_multipliers(*args, **kwargs):
    return get_wgp_runtime_module().parse_loras_multipliers(*args, **kwargs)


def preparse_loras_multipliers(*args, **kwargs):
    return get_wgp_runtime_module().preparse_loras_multipliers(*args, **kwargs)


def setup_loras(*args, **kwargs):
    return get_wgp_runtime_module().setup_loras(*args, **kwargs)


__all__ = [
    "apply_runtime_model_patch",
    "begin_runtime_model_patch",
    "clear_uni3c_cache_if_unused",
    "clear_wgp_loaded_model_state",
    "create_qwen_prompt_expander",
    "get_canny_video_annotator_class",
    "get_depth_v2_video_annotator_class",
    "get_default_settings",
    "get_lora_dir",
    "get_flow_annotator_class",
    "get_flow_viz_module",
    "get_model_def",
    "get_pose_body_face_video_annotator_class",
    "get_model_fps",
    "get_model_min_frames_and_step",
    "get_model_name",
    "get_model_recursive_prop",
    "get_qwen_prompt_expander_class",
    "get_qwen_family_handler",
    "get_qwen_main_module",
    "get_rife_temporal_interpolation",
    "get_shared_lora_utils_module",
    "get_wan2gp_save_video_callable",
    "get_wgp_runtime_module",
    "get_wgp_runtime_module_mutable",
    "get_wgp_runtime_state",
    "load_uni3c_controlnet",
    "load_wgp_runtime_model",
    "parse_loras_multipliers",
    "preparse_loras_multipliers",
    "reset_wgp_runtime_module",
    "rollback_runtime_model_patch",
    "run_rife_temporal_interpolation",
    "set_wgp_model_def",
    "set_wgp_loaded_model_state",
    "set_wgp_reload_needed",
    "setup_loras",
    "upsert_wgp_model_definition",
]
