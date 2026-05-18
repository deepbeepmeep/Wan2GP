from __future__ import annotations

import importlib
import torch
from contextlib import nullcontext
from transformers.masking_utils import create_causal_mask as hf_create_causal_mask


class _InitCompat:
    @staticmethod
    def ones_(tensor):
        return torch.nn.init.ones_(tensor)

    @staticmethod
    def zeros_(tensor):
        return torch.nn.init.zeros_(tensor)

    @staticmethod
    def copy_(tensor, value):
        with torch.no_grad():
            tensor.copy_(value)
        return tensor


init = _InitCompat()


def use_kernelized_func(_func):
    def decorator(obj):
        return obj

    return decorator


def torch_compilable_check(condition, message):
    if not bool(condition):
        raise ValueError(message)


def is_flash_attention_requested(config):
    return getattr(config, "_attn_implementation", None) == "flash_attention_2"


def maybe_autocast(device_type=None, enabled=True):
    if enabled and torch.cuda.is_available() and device_type == "cuda":
        return torch.autocast("cuda")
    return nullcontext()


def merge_with_config_defaults(func=None, **_kwargs):
    if func is None:
        def decorator(inner):
            return inner

        return decorator
    return func


def is_causal_conv1d_available():
    return False
    if not torch.cuda.is_available():
        return False
    try:
        causal_conv1d = importlib.import_module("causal_conv1d")
    except Exception:
        return False
    return all(hasattr(causal_conv1d, name) for name in ("causal_conv1d_fn", "causal_conv1d_update"))


def is_flash_linear_attention_available():
    if not torch.cuda.is_available():
        return False
    try:
        fla_modules = importlib.import_module("fla.modules")
        fla_ops = importlib.import_module("fla.ops.gated_delta_rule")
    except Exception:
        return False
    return all(
        hasattr(fla_modules, name)
        for name in ("FusedRMSNormGated",)
    ) and all(
        hasattr(fla_ops, name)
        for name in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")
    )


def capture_outputs(func=None, **_kwargs):
    if func is None:
        def decorator(inner):
            return inner

        return decorator
    return func


def auto_docstring(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(obj):
        return obj

    return decorator


def can_return_tuple(func=None, **_kwargs):
    if func is None:
        def decorator(inner):
            return inner

        return decorator
    return func


class _MaskCompatCache:
    def __init__(self, cache):
        self._cache = cache

    def __getattr__(self, name):
        return getattr(self._cache, name)

    def get_mask_sizes(self, query_length, layer_idx):
        if torch.is_tensor(query_length):
            if query_length.ndim == 0:
                query_length = 1
            else:
                query_length = int(query_length.shape[0])
        else:
            query_length = int(query_length)
        return self._cache.get_mask_sizes(query_length, layer_idx)


def _normalize_cache_position(cache_position: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(cache_position):
        return cache_position
    if cache_position.ndim == 0:
        return cache_position.reshape(1)
    if cache_position.ndim > 1:
        cache_position = cache_position.reshape(-1)
    return cache_position.to(dtype=torch.long)


def create_causal_mask(*args, **kwargs):
    if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
        kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
    if "cache_position" not in kwargs:
        input_embeds = kwargs.get("input_embeds")
        position_ids = kwargs.get("position_ids")
        past_key_values = kwargs.get("past_key_values")
        if position_ids is not None:
            if position_ids.ndim == 0:
                cache_position = position_ids.reshape(1)
            elif position_ids.ndim == 1:
                cache_position = position_ids
            else:
                cache_position = position_ids[0]
        elif input_embeds is not None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + input_embeds.shape[1],
                device=input_embeds.device,
            )
        else:
            raise TypeError("create_causal_mask() requires either position_ids or input_embeds to infer cache_position")
        kwargs["cache_position"] = cache_position
    kwargs["cache_position"] = _normalize_cache_position(kwargs["cache_position"])
    past_key_values = kwargs.get("past_key_values")
    if past_key_values is not None and hasattr(past_key_values, "get_mask_sizes"):
        kwargs["past_key_values"] = _MaskCompatCache(past_key_values)
    return hf_create_causal_mask(*args, **kwargs)


__all__ = [
    "init",
    "auto_docstring",
    "can_return_tuple",
    "capture_outputs",
    "create_causal_mask",
    "is_causal_conv1d_available",
    "is_flash_attention_requested",
    "is_flash_linear_attention_available",
    "maybe_autocast",
    "merge_with_config_defaults",
    "torch_compilable_check",
    "use_kernelized_func",
]
