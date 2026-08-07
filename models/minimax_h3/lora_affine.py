"""Affine conversion between full-width and rank-8 MiniMax H3 AdaLN LoRAs."""

from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file


FULL_TIME_DIM = 2688
_MAP_DIR = Path(__file__).with_name("lora_affine_maps")
_ARCHITECTURES = {
    "minimax_h3_fl2va": "fl2va",
    "minimax_h3_fl2va_pruned": "fl2va",
    "minimax_h3_ref2va": "ref2va",
    "minimax_h3_ref2va_pruned": "ref2va",
}
_LORA_SUFFIXES = (
    ("lora_A.weight", "lora_B.weight"),
    ("lora_down.weight", "lora_up.weight"),
    ("lora.A.weight", "lora.B.weight"),
    ("lora.down.weight", "lora.up.weight"),
)


def _architecture(model_type):
    try:
        return _ARCHITECTURES[model_type]
    except KeyError as error:
        raise ValueError(f"Unsupported MiniMax H3 architecture for AdaLN LoRA conversion: {model_type}") from error


@lru_cache(maxsize=2)
def _load_affine_package(architecture):
    tensors = load_file(str(_MAP_DIR / f"{architecture}_rank8.sft"), device="cpu")
    table, affine = tensors["adaln_t_table"].float(), tensors["adaln_affine_map"].float()
    if table.shape != (1025, 8) or affine.shape != (9, FULL_TIME_DIM):
        raise ValueError(f"Invalid MiniMax H3 {architecture} AdaLN affine package")
    return table, affine


def _aligned_affine_map(architecture, target_table):
    canonical_table, canonical_affine = _load_affine_package(architecture)
    target_table = target_table.detach().to(device="cpu", dtype=torch.float64)
    if target_table.shape != canonical_table.shape:
        raise ValueError(f"MiniMax H3 {architecture} LoRA conversion expects an AdaLN table shaped {tuple(canonical_table.shape)}, found {tuple(target_table.shape)}")
    ones = target_table.new_ones(target_table.shape[0], 1)
    target_h = torch.cat((target_table, ones), dim=1)
    canonical_h = torch.cat((canonical_table.double(), ones), dim=1)
    fit = torch.linalg.lstsq(target_h, canonical_h, rcond=1e-14)
    if int(fit.rank) != target_h.shape[1]:
        raise ValueError(f"MiniMax H3 {architecture} checkpoint has a rank-deficient AdaLN table")
    relative_error = torch.linalg.vector_norm(target_h @ fit.solution - canonical_h) / torch.linalg.vector_norm(canonical_h)
    if relative_error > 1e-5:
        raise ValueError(f"MiniMax H3 {architecture} checkpoint AdaLN table is incompatible with the canonical LoRA map (relative error {relative_error:.3g})")
    return (fit.solution @ canonical_affine.double()).float()


@lru_cache(maxsize=2)
def _canonical_encoder(architecture):
    _, affine = _load_affine_package(architecture)
    return torch.linalg.pinv(affine[:8].double(), rtol=1e-14).T.float()


def _add_bias_delta(state_dict, key, delta):
    existing = state_dict.get(key)
    if existing is not None:
        if existing.shape != delta.shape:
            raise ValueError(f"MiniMax H3 LoRA bias delta shape mismatch for {key}: {tuple(existing.shape)} != {tuple(delta.shape)}")
        delta.add_(existing.float())
    state_dict[key] = delta


def convert_adaln_loras(model_type, state_dict, target_table=None):
    """Convert AdaLN LoRA factors to the loaded full or pruned H3 representation."""
    architecture = _architecture(model_type)
    pruned_target = target_table is not None
    source_width, target_width = (FULL_TIME_DIM, 8) if pruned_target else (8, FULL_TIME_DIM)
    candidates = []

    for down_suffix, up_suffix in _LORA_SUFFIXES:
        marker = "." + down_suffix
        for down_key in [key for key in state_dict if key.endswith(marker) and ".adaln_proj.linear." in key]:
            down = state_dict[down_key]
            if down.ndim != 2 or down.shape[1] != source_width:
                continue
            module_name = down_key[:-len(marker)]
            up_key = module_name + "." + up_suffix
            if up_key not in state_dict:
                raise ValueError(f"MiniMax H3 LoRA is missing {up_key}")
            up = state_dict[up_key]
            if up.ndim != 2 or up.shape[1] != down.shape[0]:
                raise ValueError(f"MiniMax H3 LoRA factors are incompatible for {module_name}: A={tuple(down.shape)}, B={tuple(up.shape)}")
            candidates.append((module_name, down_key, up_key))

    if not candidates:
        return 0, architecture, source_width, target_width

    target_affine = _aligned_affine_map(architecture, target_table) if pruned_target else None
    encoder = None if pruned_target else _canonical_encoder(architecture)

    for module_name, down_key, up_key in candidates:
        down, up = state_dict[down_key], state_dict[up_key]
        if pruned_target:
            mapped = down.float() @ target_affine.T
            state_dict[down_key] = mapped[:, :target_width]
            inner_bias = mapped[:, target_width]
        else:
            mapped = down.float() @ encoder
            state_dict[down_key] = mapped
            inner_bias = -(mapped @ _load_affine_package(architecture)[1][8])
        _add_bias_delta(state_dict, module_name + ".diff_b", up.float() @ inner_bias)

    return len(candidates), architecture, source_width, target_width
