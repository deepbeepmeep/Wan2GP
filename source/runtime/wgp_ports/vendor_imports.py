"""Vendor-facing imports isolated behind a runtime boundary."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any

_BRIDGE_MODULE_NAME = ".".join(("source", "runtime", "wgp_bridge"))


def _get_bridge_module():
    bridge_module = sys.modules.get("source.runtime.wgp_bridge")
    if bridge_module is None:
        bridge_module = import_module(_BRIDGE_MODULE_NAME)
    return bridge_module


def _ensure_bridge_bootstrap() -> str:
    bridge_module = _get_bridge_module()
    ensure_fn = getattr(bridge_module, "ensure_wan2gp_on_path")
    return ensure_fn()


def _import_primary(module_name: str):
    bridge_module = _get_bridge_module()
    import_fn = getattr(bridge_module, "import_module", import_module)
    _ensure_bridge_bootstrap()
    return import_fn(module_name)


def create_qwen_prompt_expander(*args, **kwargs):
    expander_cls = get_qwen_prompt_expander_class()
    return expander_cls(*args, **kwargs)


def get_qwen_prompt_expander_class():
    _ensure_bridge_bootstrap()
    primary_module = ".".join(("Wan2GP", "shared", "utils", "prompt_extend"))
    fallback_module = ".".join(("Wan2GP", "wan", "utils", "prompt_extend"))
    try:
        return import_module(primary_module).QwenPromptExpander
    except ModuleNotFoundError:
        return import_module(fallback_module).QwenPromptExpander


def get_rife_temporal_interpolation():
    module_name = ".".join(("Wan2GP", "postprocessing", "rife", "inference"))
    return _import_primary(module_name).temporal_interpolation


def run_rife_temporal_interpolation(flownet_ckpt, sample_input, exp_val, *, device):
    bridge_module = sys.modules.get("source.runtime.wgp_bridge")
    getter = getattr(bridge_module, "get_rife_temporal_interpolation", get_rife_temporal_interpolation) if bridge_module else get_rife_temporal_interpolation
    return getter()(flownet_ckpt, sample_input, exp_val, device=device)


def get_wan2gp_save_video_callable():
    shared_audio_module = sys.modules.get("shared.utils.audio_video")
    if shared_audio_module is not None and hasattr(shared_audio_module, "save_video"):
        return shared_audio_module.save_video
    module_name = ".".join(("Wan2GP", "shared", "utils", "audio_video"))
    return _import_primary(module_name).save_video


def get_qwen_family_handler():
    module_name = ".".join(("models", "qwen", "qwen_handler"))
    return _import_primary(module_name).family_handler


def get_shared_lora_utils_module():
    module_name = ".".join(("shared", "utils", "loras_mutipliers"))
    return _import_primary(module_name)


def get_qwen_main_module():
    module_name = ".".join(("models", "qwen", "qwen_main"))
    return _import_primary(module_name)


def get_flow_annotator_class():
    module_name = ".".join(("Wan2GP", "preprocessing", "flow"))
    return _import_primary(module_name).FlowAnnotator


def get_flow_viz_module():
    module_name = ".".join(("Wan2GP", "preprocessing", "raft", "utils", "flow_viz"))
    return _import_primary(module_name)


def get_canny_video_annotator_class():
    module_name = ".".join(("Wan2GP", "preprocessing", "canny"))
    return _import_primary(module_name).CannyVideoAnnotator


def get_depth_v2_video_annotator_class():
    module_name = ".".join(("Wan2GP", "preprocessing", "depth_anything_v2", "depth"))
    return _import_primary(module_name).DepthV2VideoAnnotator


def get_pose_body_face_video_annotator_class():
    module_name = ".".join(("Wan2GP", "preprocessing", "dwpose", "pose"))
    return _import_primary(module_name).PoseBodyFaceVideoAnnotator


def clear_uni3c_cache_if_unused() -> bool:
    try:
        module_name = ".".join(("Wan2GP", "models", "wan", "uni3c"))
        _import_primary(module_name).clear_uni3c_cache_if_unused()
        return True
    except ModuleNotFoundError:
        return False


def load_uni3c_controlnet(*, ckpts_dir: str, device: str, dtype: str, use_cache: bool = True, **kwargs):
    if not ckpts_dir:
        raise ValueError("ckpts_dir must be non-empty")
    if not isinstance(use_cache, bool):
        raise TypeError("use_cache must be a bool")
    module_name = ".".join(("Wan2GP", "models", "wan", "uni3c"))
    loader = _import_primary(module_name).load_uni3c_controlnet
    return loader(ckpts_dir=ckpts_dir, device=device, dtype=dtype, use_cache=use_cache, **kwargs)
