"""Vendor-facing imports isolated behind a runtime boundary."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any

from source.core.runtime_paths import ensure_wan2gp_on_path


def _import_primary(module_name: str):
    bridge_module = sys.modules.get("source.runtime.wgp_bridge")
    ensure_fn = getattr(bridge_module, "ensure_wan2gp_on_path", ensure_wan2gp_on_path) if bridge_module else ensure_wan2gp_on_path
    import_fn = getattr(bridge_module, "import_module", import_module) if bridge_module else import_module
    ensure_fn()
    return import_fn(module_name)


def create_qwen_prompt_expander(*args, **kwargs):
    expander_cls = get_qwen_prompt_expander_class()
    return expander_cls(*args, **kwargs)


def get_qwen_prompt_expander_class():
    ensure_wan2gp_on_path()
    try:
        return import_module("Wan2GP.shared.utils.prompt_extend").QwenPromptExpander
    except ModuleNotFoundError:
        return import_module("Wan2GP.wan.utils.prompt_extend").QwenPromptExpander


def get_rife_temporal_interpolation():
    return _import_primary("Wan2GP.postprocessing.rife.inference").temporal_interpolation


def run_rife_temporal_interpolation(flownet_ckpt, sample_input, exp_val, *, device):
    bridge_module = sys.modules.get("source.runtime.wgp_bridge")
    getter = getattr(bridge_module, "get_rife_temporal_interpolation", get_rife_temporal_interpolation) if bridge_module else get_rife_temporal_interpolation
    return getter()(flownet_ckpt, sample_input, exp_val, device=device)


def get_wan2gp_save_video_callable():
    return _import_primary("Wan2GP.shared.utils.audio_video").save_video


def clear_uni3c_cache_if_unused() -> bool:
    try:
        _import_primary("Wan2GP.models.wan.uni3c").clear_uni3c_cache_if_unused()
        return True
    except ModuleNotFoundError:
        return False


def load_uni3c_controlnet(*, ckpts_dir: str, device: str, dtype: str, use_cache: bool = True, **kwargs):
    if not ckpts_dir:
        raise ValueError("ckpts_dir must be non-empty")
    if not isinstance(use_cache, bool):
        raise TypeError("use_cache must be a bool")
    loader = _import_primary("Wan2GP.models.wan.uni3c").load_uni3c_controlnet
    return loader(ckpts_dir=ckpts_dir, device=device, dtype=dtype, use_cache=use_cache, **kwargs)
