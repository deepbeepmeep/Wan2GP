"""MiniMax H3 tensor-layout transforms for generic ConvRot conversion and loading."""

import json
from functools import partial

import torch


def _reorder_qkv_rows(tensor, heads, head_dim, grouped):
    tail = tuple(tensor.shape[1:])
    if grouped:
        return tensor.reshape(heads, 3, head_dim, *tail).permute(1, 0, 2, *range(3, 3 + len(tail))).reshape_as(tensor).contiguous()
    return tensor.reshape(3, heads, head_dim, *tail).permute(1, 0, 2, *range(3, 3 + len(tail))).reshape_as(tensor).contiguous()


def group_qkv_rows(tensor, heads, head_dim):
    return _reorder_qkv_rows(tensor, heads, head_dim, grouped=True)


def interleave_qkv_rows(tensor, heads, head_dim):
    return _reorder_qkv_rows(tensor, heads, head_dim, grouped=False)


def get_convrot_layout(model):
    """Return H3's optional source-to-ConvRot tensor layout mapping."""
    return {
        f"{name}.qkv_proj.weight": partial(group_qkv_rows, heads=module.heads, head_dim=module.head_dim)
        for name, module in model.named_modules() if hasattr(module, "heads") and hasattr(module, "head_dim")
    }


def _comfy_quant_config(tensor):
    if not torch.is_tensor(tensor):
        return {}
    try:
        config = json.loads(bytes(tensor.detach().cpu().to(torch.uint8).reshape(-1).tolist()).decode("utf-8"))
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def restore_interleaved_h3_qkv(state_dict):
    """Restore official H3 head-interleaved QKV rows from grouped checkpoint layouts."""
    has_convrot = any(key.endswith(".comfy_quant") and _comfy_quant_config(value).get("convrot") for key, value in state_dict.items())
    has_nvfp4 = any(key.endswith(".qkv_proj.comfy_quant") and _comfy_quant_config(value).get("format") == "nvfp4"
                     for key, value in state_dict.items())
    has_w4a8 = any(key.endswith(".weight_s_rel") and key.removesuffix(".weight_s_rel") + ".weight_s_channel" in state_dict
                    for key in state_dict)
    if not has_convrot and not has_nvfp4 and not has_w4a8:
        return state_dict
    norm_key = next(key for key in state_dict if key.endswith("blocks.0.attn.q_norm.weight"))
    head_dim = state_dict[norm_key].shape[0]
    qkv_key = next(key for key in state_dict if key.endswith("blocks.0.attn.qkv_proj.weight"))
    heads = state_dict[qkv_key].shape[0] // (3 * head_dim)
    for key in [key for key in state_dict if key.endswith(".qkv_proj.weight")]:
        base = key[:-len(".weight")]
        quant_key = base + ".comfy_quant"
        if quant_key in state_dict:
            if _comfy_quant_config(state_dict[quant_key]).get("format") == "nvfp4":
                # NVFP4 scales stay grouped for the quantization handler's contiguous Q/K/V split.
                state_dict[key] = interleave_qkv_rows(state_dict[key], heads, head_dim)
                state_dict.pop(quant_key)
            continue
        if base + ".weight_s_rel" in state_dict:
            continue
        state_dict[key] = interleave_qkv_rows(state_dict[key], heads, head_dim)
        scale_key = base + ".weight_scale"
        if scale_key in state_dict:
            state_dict[scale_key] = interleave_qkv_rows(state_dict[scale_key], heads, head_dim)
    return state_dict
