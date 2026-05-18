from __future__ import annotations

from collections import OrderedDict
import re
from typing import Mapping


MODALITY_NAMES = ("video", "audio", "text")
MM_LAYERS = frozenset((0, 1, 2, 3, 36, 37, 38, 39))
Q_SIZE = 5120
KV_SIZE = 8 * 128
G_SIZE = 5120 // 128
EXPERT_LINEAR_NAMES = frozenset((
    "attention.linear_q",
    "attention.linear_k",
    "attention.linear_v",
    "attention.linear_g",
    "attention.linear_proj",
    "mlp.up_gate_proj",
    "mlp.down_proj",
))
_EXPERT_KEY_RE = re.compile(r"^(block\.layers\.(\d+)\.(" + "|".join(re.escape(name) for name in sorted(EXPERT_LINEAR_NAMES)) + r"))(\..+)$")
_EXPERT_SUFFIX_RE = re.compile(r"^(block\.layers\.(\d+)\.(" + "|".join(re.escape(name) for name in sorted(EXPERT_LINEAR_NAMES)) + r"))_(video|audio|text)(\..+)$")
_QKVG_KEY_RE = re.compile(r"^(block\.layers\.(\d+)\.attention\.linear_qkv)(\..+)$")
_ROW_SPLIT_SUFFIXES = (".weight", ".bias", ".lora_B.weight", ".lora_up.weight", ".diff_b")
_SHARED_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".alpha", ".dora_scale", ".diff")


def _match_expert_key(key: str):
    match = _EXPERT_KEY_RE.match(key)
    if match is None:
        return None
    layer_idx = int(match.group(2))
    if layer_idx not in MM_LAYERS:
        return None
    return match


def _is_already_split(key: str) -> bool:
    return _EXPERT_SUFFIX_RE.match(key) is not None


def _split_tensor_rows(value, parts: int):
    if value.shape[0] % parts != 0:
        raise ValueError(f"Cannot split tensor with shape {tuple(value.shape)} into {parts} equal row chunks.")
    return value.chunk(parts, dim=0)


def _iter_qkvg_targets(layer_idx: int, suffix: str, value):
    num_modality = len(MODALITY_NAMES) if layer_idx in MM_LAYERS else 1
    offset = 0
    for modality_idx in range(num_modality):
        modality_name = MODALITY_NAMES[modality_idx]
        modality_suffix = f"_{modality_name}" if num_modality > 1 else ""
        for proj_name, proj_size in (("linear_q", Q_SIZE), ("linear_k", KV_SIZE), ("linear_v", KV_SIZE), ("linear_g", G_SIZE)):
            chunk = value.narrow(0, offset, proj_size)
            offset += proj_size
            yield f"block.layers.{layer_idx}.attention.{proj_name}{modality_suffix}{suffix}", chunk


def convert_transformer_state_dict_to_split_experts(state_dict: Mapping[str, object]) -> OrderedDict[str, object]:
    new_state_dict: OrderedDict[str, object] = OrderedDict()
    for key, value in state_dict.items():
        qkvg_match = _QKVG_KEY_RE.match(key)
        if qkvg_match is not None:
            layer_idx = int(qkvg_match.group(2))
            suffix = qkvg_match.group(3)
            if suffix in (".weight", ".bias"):
                for new_key, chunk in _iter_qkvg_targets(layer_idx, suffix, value):
                    new_state_dict[new_key] = chunk
                continue
        match = _match_expert_key(key)
        if match is None or _is_already_split(key):
            new_state_dict[key] = value
            continue
        base_key, suffix = match.group(1), match.group(4)
        if suffix not in (".weight", ".bias"):
            new_state_dict[key] = value
            continue
        chunks = _split_tensor_rows(value, len(MODALITY_NAMES))
        for modality_name, chunk in zip(MODALITY_NAMES, chunks):
            new_state_dict[f"{base_key}_{modality_name}{suffix}"] = chunk
    return new_state_dict


def preprocess_magi_lora_state_dict(state_dict: Mapping[str, object]) -> OrderedDict[str, object]:
    first_key = next(iter(state_dict), None)
    if first_key is None or _is_already_split(first_key):
        return OrderedDict(state_dict.items())

    new_state_dict: OrderedDict[str, object] = OrderedDict()
    for key, value in state_dict.items():
        qkvg_match = _QKVG_KEY_RE.match(key)
        if qkvg_match is not None:
            layer_idx = int(qkvg_match.group(2))
            suffix = qkvg_match.group(3)
            if suffix in _ROW_SPLIT_SUFFIXES:
                for new_key, chunk in _iter_qkvg_targets(layer_idx, suffix, value):
                    new_state_dict[new_key] = chunk
                continue
            if suffix in _SHARED_SUFFIXES:
                num_modality = len(MODALITY_NAMES) if layer_idx in MM_LAYERS else 1
                for modality_idx in range(num_modality):
                    modality_name = MODALITY_NAMES[modality_idx]
                    modality_suffix = f"_{modality_name}" if num_modality > 1 else ""
                    for proj_name in ("linear_q", "linear_k", "linear_v", "linear_g"):
                        new_state_dict[f"block.layers.{layer_idx}.attention.{proj_name}{modality_suffix}{suffix}"] = value
                continue
        match = _match_expert_key(key)
        if match is None:
            new_state_dict[key] = value
            continue
        base_key, suffix = match.group(1), match.group(4)
        if suffix in _ROW_SPLIT_SUFFIXES:
            chunks = _split_tensor_rows(value, len(MODALITY_NAMES))
            for modality_name, chunk in zip(MODALITY_NAMES, chunks):
                new_state_dict[f"{base_key}_{modality_name}{suffix}"] = chunk
        elif suffix in _SHARED_SUFFIXES:
            for modality_name in MODALITY_NAMES:
                new_state_dict[f"{base_key}_{modality_name}{suffix}"] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
