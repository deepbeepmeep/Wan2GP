#!/usr/bin/env python3
"""
Convert a Flux model from Diffusers (folder or single-file) into the original
single-file Flux transformer checkpoint used by Black Forest Labs / ComfyUI.

Input  : /path/to/diffusers   (root or .../transformer)  OR  /path/to/*.safetensors (single file)
Output : /path/to/flux1-your-model.safetensors  (transformer only)

Usage:
  python diffusers_to_flux_transformer.py /path/to/diffusers /out/flux1-dev.safetensors
  python diffusers_to_flux_transformer.py /path/to/diffusion_pytorch_model.safetensors /out/flux1-dev.safetensors
  # optional quantization:
  #   --fp8           (float8_e4m3fn, simple)
  #   --fp8-scaled    (scaled float8 for 2D weights; adds .scale_weight tensors)
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors import safe_open
import safetensors.torch
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("diffusers_path", type=str,
                    help="Path to Diffusers checkpoint folder OR a single .safetensors file.")
    ap.add_argument("output_path", type=str,
                    help="Output .safetensors path for the Flux transformer.")
    ap.add_argument("--fp8", action="store_true",
                    help="Experimental: write weights as float8_e4m3fn via stochastic rounding (transformer only).")
    ap.add_argument("--fp8-scaled", action="store_true",
                    help="Experimental: scaled float8_e4m3fn for 2D weight tensors; adds .scale_weight tensors.")
    return ap.parse_args()


# Mapping from original Flux keys -> list of Diffusers keys (per block where applicable).
DIFFUSERS_MAP = {
    # global embeds
    "time_in.in_layer.weight": ["time_text_embed.timestep_embedder.linear_1.weight"],
    "time_in.in_layer.bias":   ["time_text_embed.timestep_embedder.linear_1.bias"],
    "time_in.out_layer.weight": ["time_text_embed.timestep_embedder.linear_2.weight"],
    "time_in.out_layer.bias":   ["time_text_embed.timestep_embedder.linear_2.bias"],

    "vector_in.in_layer.weight": ["time_text_embed.text_embedder.linear_1.weight"],
    "vector_in.in_layer.bias":   ["time_text_embed.text_embedder.linear_1.bias"],
    "vector_in.out_layer.weight": ["time_text_embed.text_embedder.linear_2.weight"],
    "vector_in.out_layer.bias":   ["time_text_embed.text_embedder.linear_2.bias"],

    "guidance_in.in_layer.weight": ["time_text_embed.guidance_embedder.linear_1.weight"],
    "guidance_in.in_layer.bias":   ["time_text_embed.guidance_embedder.linear_1.bias"],
    "guidance_in.out_layer.weight": ["time_text_embed.guidance_embedder.linear_2.weight"],
    "guidance_in.out_layer.bias":   ["time_text_embed.guidance_embedder.linear_2.bias"],

    "txt_in.weight": ["context_embedder.weight"],
    "txt_in.bias":   ["context_embedder.bias"],
    "img_in.weight": ["x_embedder.weight"],
    "img_in.bias":   ["x_embedder.bias"],

    # dual-stream (image/text) blocks
    "double_blocks.().img_mod.lin.weight": ["norm1.linear.weight"],
    "double_blocks.().img_mod.lin.bias":   ["norm1.linear.bias"],
    "double_blocks.().txt_mod.lin.weight": ["norm1_context.linear.weight"],
    "double_blocks.().txt_mod.lin.bias":   ["norm1_context.linear.bias"],

    "double_blocks.().img_attn.qkv.weight": [
        ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
        ["qkv_proj.weight"],
    ],
    "double_blocks.().img_attn.qkv.bias": [
        ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"],
        ["qkv_proj.bias"],
    ],
    "double_blocks.().txt_attn.qkv.weight": [
        ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
        ["qkv_proj_context.weight"],
    ],
    "double_blocks.().txt_attn.qkv.bias": [
        ["attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"],
        ["qkv_proj_context.bias"],
    ],

    "double_blocks.().img_attn.norm.query_norm.scale": [
        ["attn.norm_q.weight"],
        ["norm_q.weight"],
    ],
    "double_blocks.().img_attn.norm.key_norm.scale": [
        ["attn.norm_k.weight"],
        ["norm_k.weight"],
    ],
    "double_blocks.().txt_attn.norm.query_norm.scale": [
        ["attn.norm_added_q.weight"],
        ["norm_added_q.weight"],
    ],
    "double_blocks.().txt_attn.norm.key_norm.scale": [
        ["attn.norm_added_k.weight"],
        ["norm_added_k.weight"],
    ],

    "double_blocks.().img_mlp.0.weight": [
        ["ff.net.0.proj.weight"],
        ["mlp_fc1.weight"],
    ],
    "double_blocks.().img_mlp.0.bias": [
        ["ff.net.0.proj.bias"],
        ["mlp_fc1.bias"],
    ],
    "double_blocks.().img_mlp.2.weight": [
        ["ff.net.2.weight"],
        ["mlp_fc2.weight"],
    ],
    "double_blocks.().img_mlp.2.bias": [
        ["ff.net.2.bias"],
        ["mlp_fc2.bias"],
    ],

    "double_blocks.().txt_mlp.0.weight": [
        ["ff_context.net.0.proj.weight"],
        ["mlp_context_fc1.weight"],
    ],
    "double_blocks.().txt_mlp.0.bias": [
        ["ff_context.net.0.proj.bias"],
        ["mlp_context_fc1.bias"],
    ],
    "double_blocks.().txt_mlp.2.weight": [
        ["ff_context.net.2.weight"],
        ["mlp_context_fc2.weight"],
    ],
    "double_blocks.().txt_mlp.2.bias": [
        ["ff_context.net.2.bias"],
        ["mlp_context_fc2.bias"],
    ],

    "double_blocks.().img_attn.proj.weight": [
        ["attn.to_out.0.weight"],
        ["out_proj.weight"],
    ],
    "double_blocks.().img_attn.proj.bias": [
        ["attn.to_out.0.bias"],
        ["out_proj.bias"],
    ],
    "double_blocks.().txt_attn.proj.weight": [
        ["attn.to_add_out.weight"],
        ["out_proj_context.weight"],
    ],
    "double_blocks.().txt_attn.proj.bias": [
        ["attn.to_add_out.bias"],
        ["out_proj_context.bias"],
    ],

    # single-stream blocks
    "single_blocks.().modulation.lin.weight": ["norm.linear.weight"],
    "single_blocks.().modulation.lin.bias":   ["norm.linear.bias"],
    "single_blocks.().linear1.weight": [
        ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "proj_mlp.weight"],
        ["qkv_proj.weight", "mlp_fc1.weight"],
    ],
    "single_blocks.().linear1.bias": [
        ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"],
        ["qkv_proj.bias", "mlp_fc1.bias"],
    ],
    "single_blocks.().norm.query_norm.scale": [
        ["attn.norm_q.weight"],
        ["norm_q.weight"],
    ],
    "single_blocks.().norm.key_norm.scale": [
        ["attn.norm_k.weight"],
        ["norm_k.weight"],
    ],
    "single_blocks.().linear2.weight": [
        ["proj_out.weight"],
        ["out_proj.weight", "mlp_fc2.weight"],
    ],
    "single_blocks.().linear2.bias": [
        ["proj_out.bias"],
        ["out_proj.bias", "mlp_fc2.bias"],
    ],

    # final
    "final_layer.linear.weight":              ["proj_out.weight"],
    "final_layer.linear.bias":                ["proj_out.bias"],
    # these two are built from norm_out.linear.{weight,bias} by swapping [shift,scale] -> [scale,shift]
    "final_layer.adaLN_modulation.1.weight":  ["norm_out.linear.weight"],
    "final_layer.adaLN_modulation.1.bias":    ["norm_out.linear.bias"],
}

_TARGET_SUFFIXES = {}
for _tgt_key in DIFFUSERS_MAP:
    _base, _suffix = _tgt_key.rsplit(".", 1)
    _TARGET_SUFFIXES.setdefault(_base, set()).add("." + _suffix)


def _strip_prefix(key: str) -> str:
    return key[6:] if key.startswith("model.") else key


class StateDictSource:
    """
    Provide DiffusersSource-like access over an in-memory state dict.
    """

    POSSIBLE_PREFIXES = ["", "model."]

    def __init__(self, state_dict: dict):
        self._state_dict = state_dict
        self._all_keys = list(state_dict.keys())

    def _resolve(self, want: str):
        for pref in self.POSSIBLE_PREFIXES:
            key = pref + want
            if key in self._state_dict:
                return key
        return None

    def has(self, want: str) -> bool:
        return self._resolve(want) is not None

    def get(self, want: str) -> torch.Tensor:
        real_key = self._resolve(want)
        if real_key is None:
            raise KeyError(f"Missing key: {want}")
        return self._state_dict[real_key]

    @property
    def base_keys(self):
        return [_strip_prefix(k) for k in self._all_keys]


def detect_diffusers_state_dict(state_dict: dict) -> bool:
    base_keys = [_strip_prefix(k) for k in state_dict.keys()]
    if any(k.startswith(("double_blocks.", "single_blocks.")) for k in base_keys):
        return False
    return any(k.startswith(("transformer_blocks.", "single_transformer_blocks.")) for k in base_keys)


def convert_state_dict(state_dict: dict, *, verbose: bool = False) -> dict:
    if not detect_diffusers_state_dict(state_dict):
        return state_dict
    converted = _convert_from_source(StateDictSource(state_dict), verbose=verbose)
    return converted if converted else state_dict


def _count_blocks(base_keys):
    num_dual = 0
    num_single = 0
    for key in base_keys:
        if key.startswith("transformer_blocks."):
            try:
                idx = int(key.split(".")[1])
                num_dual = max(num_dual, idx + 1)
            except Exception:
                pass
        elif key.startswith("single_transformer_blocks."):
            try:
                idx = int(key.split(".")[1])
                num_single = max(num_single, idx + 1)
            except Exception:
                pass
    return num_dual, num_single


def _swap_scale_shift(vec: torch.Tensor) -> torch.Tensor:
    if vec is None or vec.ndim != 1 or vec.numel() % 2 != 0:
        return vec
    shift, scale = vec.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def _swap_scale_shift_matrix(mat: torch.Tensor) -> torch.Tensor:
    if mat is None or mat.ndim != 2 or mat.size(0) % 2 != 0:
        return mat
    shift, scale = mat.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def _collect_suffixes(base_keys, base):
    prefix = base + "."
    suffixes = set()
    for key in base_keys:
        if key.startswith(prefix):
            suffixes.add("." + key[len(prefix):])
    return suffixes


def _normalize_suffix(suffix: str) -> str:
    if suffix == ".smooth":
        return ".smooth_factor"
    if suffix == ".smooth_orig":
        return ".smooth_factor_orig"
    if suffix == ".lora_down":
        return ".proj_down"
    if suffix == ".lora_up":
        return ".proj_up"
    return suffix


def _match_dtype_device(ref: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if ref.dtype != other.dtype or ref.device != other.device:
        return other.to(device=ref.device, dtype=ref.dtype)
    return other


def _concat(values, dim=0):
    if any(v is None for v in values):
        return None
    ref = values[0]
    if ref.ndim == 0:
        return ref
    merged = [ref] + [_match_dtype_device(ref, v) for v in values[1:]]
    return torch.cat(merged, dim=dim)


def _maybe_load_nunchaku():
    try:
        from shared.qtypes import nunchaku_int4 as _nunchaku_int4  # pylint: disable=import-outside-toplevel
    except Exception:
        return None
    return _nunchaku_int4


def _get_qweight_dims(qweight: torch.Tensor | None) -> tuple[int | None, int | None]:
    if not torch.is_tensor(qweight):
        return None, None
    if qweight.dtype == torch.int8:
        out_features = qweight.size(0)
    else:
        out_features = qweight.size(0) * 4
    in_features = qweight.size(1) * 2
    return out_features, in_features


def _merge_packed_scales_out(values, qweights, group_size: int = 64):
    if any(v is None for v in values):
        return None
    nunchaku = _maybe_load_nunchaku()
    if nunchaku is None or any(q is None for q in qweights):
        return _concat(values, dim=1)
    unpacked = []
    out_total = 0
    in_features = None
    for value, qweight in zip(values, qweights):
        out_i, in_i = _get_qweight_dims(qweight)
        if out_i is None or in_i is None:
            return _concat(values, dim=1)
        if in_features is None:
            in_features = in_i
        if in_i != in_features:
            return _concat(values, dim=1)
        unpacked_i = nunchaku._unpack_nunchaku_wscales(value, out_i, in_i, group_size)
        if not torch.is_tensor(unpacked_i):
            return _concat(values, dim=1)
        unpacked.append(unpacked_i)
        out_total += out_i
    merged = [unpacked[0]] + [_match_dtype_device(unpacked[0], u) for u in unpacked[1:]]
    merged = torch.cat(merged, dim=1)
    return nunchaku._pack_nunchaku_wscales(merged, out_total, in_features, group_size)


def _merge_packed_scales_in(values, qweights, group_size: int = 64):
    if any(v is None for v in values):
        return None
    nunchaku = _maybe_load_nunchaku()
    if nunchaku is None or any(q is None for q in qweights):
        return _concat(values, dim=0)
    out_a, in_a = _get_qweight_dims(qweights[0])
    out_b, in_b = _get_qweight_dims(qweights[1])
    if out_a is None or in_a is None or out_b is None or in_b is None:
        return _concat(values, dim=0)
    if out_a != out_b:
        return _concat(values, dim=0)
    unpack_a = nunchaku._unpack_nunchaku_wscales(values[0], out_a, in_a, group_size)
    unpack_b = nunchaku._unpack_nunchaku_wscales(values[1], out_b, in_b, group_size)
    if not torch.is_tensor(unpack_a) or not torch.is_tensor(unpack_b):
        return _concat(values, dim=0)
    if unpack_a.dtype != unpack_b.dtype or unpack_a.device != unpack_b.device:
        unpack_b = unpack_b.to(device=unpack_a.device, dtype=unpack_a.dtype)
    merged = torch.cat([unpack_a, unpack_b], dim=0)
    return nunchaku._pack_nunchaku_wscales(merged, out_a, in_a + in_b, group_size)


def _pad_to_multiple(tensor: torch.Tensor | None, divisor: int | tuple[int, int]):
    if tensor is None:
        return None
    if isinstance(divisor, int):
        div0 = div1 = divisor
    else:
        div0, div1 = divisor
    height, width = tensor.shape
    new_h = ((height + div0 - 1) // div0) * div0
    new_w = ((width + div1 - 1) // div1) * div1
    if new_h == height and new_w == width:
        return tensor
    padded = torch.zeros((new_h, new_w), dtype=tensor.dtype, device=tensor.device)
    padded[:height, :width] = tensor
    return padded


def _pack_lowrank_weight(weight: torch.Tensor | None, down: bool):
    if weight is None or weight.ndim != 2:
        return weight
    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    weight = _pad_to_multiple(weight, (frag_n, frag_k))
    if weight is None:
        return None
    if down:
        rows, cols = weight.shape
        r_frags, c_frags = rows // frag_n, cols // frag_k
        weight = weight.view(r_frags, frag_n, c_frags, frag_k).permute(2, 0, 1, 3)
    else:
        cols, rows = weight.shape
        c_frags, r_frags = cols // frag_n, rows // frag_k
        weight = weight.view(c_frags, frag_n, r_frags, frag_k).permute(0, 2, 1, 3)
    weight = weight.reshape(c_frags, r_frags, n_pack_size, num_n_lanes, k_pack_size, num_k_lanes, lane_k)
    weight = weight.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return weight.view(cols, rows)


def _pack_nunchaku_w4a4_weight(qvals, out_features, in_features):
    if qvals is None or qvals.ndim != 2:
        return qvals
    if qvals.dtype not in (torch.int8, torch.int16, torch.int32):
        qvals = qvals.to(torch.int32)
    if qvals.dtype != torch.int32:
        qvals = qvals.to(torch.int32)
    if qvals.shape != (out_features, in_features):
        return None
    mem_n = 128
    mem_k = 64
    num_k_unrolls = 2
    if out_features % mem_n != 0 or in_features % (mem_k * num_k_unrolls) != 0:
        return None
    n_pack_size = 2
    k_pack_size = 2
    num_n_lanes = 8
    num_k_lanes = 4
    reg_n = 1
    reg_k = 8
    num_n_packs = mem_n // (n_pack_size * num_n_lanes * reg_n)
    num_k_packs = mem_k // (k_pack_size * num_k_lanes * reg_k)
    n_tiles = out_features // mem_n
    k_tiles = in_features // mem_k
    weight = qvals.reshape(
        n_tiles,
        num_n_packs,
        n_pack_size,
        num_n_lanes,
        reg_n,
        k_tiles,
        num_k_packs,
        k_pack_size,
        num_k_lanes,
        reg_k,
    )
    weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
    weight = weight.bitwise_and_(0xF)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
    weight = weight.bitwise_left_shift_(shifts)
    weight = weight.sum(dim=-1, dtype=torch.int32)
    return weight.view(dtype=torch.int8).view(out_features, -1)


def _unpack_lowrank_weight(weight: torch.Tensor | None, down: bool):
    if weight is None:
        return None
    nunchaku = _maybe_load_nunchaku()
    if nunchaku is None:
        return weight
    return nunchaku._unpack_lowrank_weight(weight, down)


def _block_diag_out(mats):
    if not mats:
        return None
    ref = mats[0]
    total_rows = sum(m.size(0) for m in mats)
    total_cols = sum(m.size(1) for m in mats)
    out = torch.zeros((total_rows, total_cols), dtype=ref.dtype, device=ref.device)
    row = 0
    col = 0
    for mat in mats:
        mat = _match_dtype_device(ref, mat)
        out[row : row + mat.size(0), col : col + mat.size(1)] = mat
        row += mat.size(0)
        col += mat.size(1)
    return out


def _merge_lowrank_down(values):
    if any(v is None for v in values):
        return None
    unpacked = [_unpack_lowrank_weight(v, down=True) for v in values]
    ref = unpacked[0]
    merged = [ref] + [_match_dtype_device(ref, v) for v in unpacked[1:]]
    merged = torch.cat(merged, dim=0)
    return _pack_lowrank_weight(merged, down=True)


def _merge_lowrank_up(values):
    if any(v is None for v in values):
        return None
    unpacked = [_unpack_lowrank_weight(v, down=False) for v in values]
    merged = _block_diag_out(unpacked)
    return _pack_lowrank_weight(merged, down=False)


def _merge_lowrank_down_block_diag(values):
    if any(v is None for v in values):
        return None
    unpacked = [_unpack_lowrank_weight(v, down=True) for v in values]
    merged = _block_diag_out(unpacked)
    return _pack_lowrank_weight(merged, down=True)


def _merge_lowrank_up_concat(values):
    if any(v is None for v in values):
        return None
    unpacked = [_unpack_lowrank_weight(v, down=False) for v in values]
    ref = unpacked[0]
    merged = [ref] + [_match_dtype_device(ref, v) for v in unpacked[1:]]
    merged = torch.cat(merged, dim=1)
    return _pack_lowrank_weight(merged, down=False)


def _merge_qweight_in(values):
    if any(v is None for v in values):
        return None
    a, b = values
    out_a, in_a = _get_qweight_dims(a)
    out_b, in_b = _get_qweight_dims(b)
    if out_a is None or in_a is None or out_b is None or in_b is None:
        return _concat(values, dim=1)
    if out_a != out_b:
        return _concat(values, dim=1)
    nunchaku = _maybe_load_nunchaku()
    if nunchaku is None or a.dtype != torch.int8 or b.dtype != torch.int8:
        return _concat(values, dim=1)
    unpack_a = nunchaku._unpack_nunchaku_w4a4_weight(a, out_a, in_a)
    unpack_b = nunchaku._unpack_nunchaku_w4a4_weight(b, out_b, in_b)
    if not torch.is_tensor(unpack_a) or not torch.is_tensor(unpack_b):
        return _concat(values, dim=1)
    if unpack_a.dtype != torch.int32:
        unpack_a = unpack_a.to(torch.int32)
    if unpack_b.dtype != torch.int32:
        unpack_b = unpack_b.to(torch.int32)
    if unpack_a.dtype != unpack_b.dtype or unpack_a.device != unpack_b.device:
        unpack_b = unpack_b.to(device=unpack_a.device, dtype=unpack_a.dtype)
    merged = torch.cat([unpack_a, unpack_b], dim=1)
    packed = _pack_nunchaku_w4a4_weight(merged, out_a, in_a + in_b)
    if packed is None:
        return _concat(values, dim=1)
    return packed


def _merge_multi(values, suffix, qweights):
    if any(v is None for v in values):
        return None
    if suffix in (".wscales", ".wzeros"):
        return _merge_packed_scales_out(values, qweights)
    if suffix == ".proj_down":
        return _merge_lowrank_down(values)
    if suffix == ".proj_up":
        return _merge_lowrank_up(values)
    if suffix in (".smooth_factor", ".smooth_factor_orig", ".input_scale", ".output_scale", ".scale_weight"):
        return values[0]
    return _concat(values, dim=0)


def _merge_multi_in(values, suffix, qweights):
    if any(v is None for v in values):
        return None
    if suffix == ".qweight":
        return _merge_qweight_in(values)
    if suffix in (".wscales", ".wzeros"):
        return _merge_packed_scales_in(values, qweights)
    if suffix == ".proj_down":
        return _merge_lowrank_down_block_diag(values)
    if suffix == ".proj_up":
        return _merge_lowrank_up_concat(values)
    if suffix in (".smooth_factor", ".smooth_factor_orig"):
        return _concat(values, dim=0)
    if suffix == ".bias":
        ref = values[0]
        total = ref
        for val in values[1:]:
            total = total + _match_dtype_device(ref, val)
        return total
    if suffix in (".input_scale", ".output_scale", ".scale_weight"):
        return values[0]
    return _concat(values, dim=1)


def _convert_from_source(src, *, verbose: bool = False) -> dict:
    base_keys = src.base_keys
    num_dual, num_single = _count_blocks(base_keys)
    if verbose:
        print(f"Found {num_dual} dual-stream blocks, {num_single} single-stream blocks")

    out = {}
    quant_suffixes = {
        ".qweight",
        ".wscales",
        ".wzeros",
        ".smooth",
        ".smooth_orig",
        ".lora_down",
        ".lora_up",
        ".bias",
        ".input_scale",
        ".output_scale",
        ".scale_weight",
    }

    def _map_entry(src_prefix, tgt_template, dvals):
        tgt_base_template, tgt_suffix = tgt_template.rsplit(".", 1)
        tgt_suffix = "." + tgt_suffix
        candidates = dvals if isinstance(dvals[0], (list, tuple)) else [dvals]

        for candidate in candidates:
            src_suffix = "." + candidate[0].rsplit(".", 1)[1]
            if any(d.rsplit(".", 1)[1] != src_suffix.lstrip(".") for d in candidate):
                continue
            src_bases = [d.rsplit(".", 1)[0] for d in candidate]
            suffix_sets = [_collect_suffixes(base_keys, src_prefix + base) for base in src_bases]
            if any(not suffix_set for suffix_set in suffix_sets):
                continue
            common_suffixes = set.intersection(*suffix_sets)
            if not common_suffixes:
                continue

            explicit_suffixes = _TARGET_SUFFIXES.get(tgt_base_template, set())
            allow_extra = src_suffix == tgt_suffix
            if src_suffix in common_suffixes:
                suffixes = {src_suffix}
                if allow_extra and tgt_suffix == ".weight":
                    suffixes |= (common_suffixes - explicit_suffixes)
            else:
                suffixes = common_suffixes & quant_suffixes
                if not suffixes:
                    continue

            tgt_base = (
                tgt_base_template.replace("()", str(block_idx))
                if "()" in tgt_base_template
                else tgt_base_template
            )
            merge_in_features = tgt_base_template.endswith(".linear2") and len(src_bases) > 1

            for suffix in suffixes:
                values = [src.get(src_prefix + base + suffix) for base in src_bases]
                if len(values) == 1:
                    merged = values[0]
                else:
                    qweights = [
                        src.get(src_prefix + base + ".qweight") if src.has(src_prefix + base + ".qweight") else None
                        for base in src_bases
                    ]
                    norm_suffix = _normalize_suffix(suffix)
                    if merge_in_features:
                        merged = _merge_multi_in(values, norm_suffix, qweights)
                    else:
                        merged = _merge_multi(values, norm_suffix, qweights)
                if merged is None:
                    continue
                out_suffix = tgt_suffix if suffix == src_suffix else _normalize_suffix(suffix)
                out[tgt_base + out_suffix] = merged
            break

    for block_idx in range(num_dual):
        prefix = f"transformer_blocks.{block_idx}."
        for tgt_key, dvals in DIFFUSERS_MAP.items():
            if not tgt_key.startswith("double_blocks."):
                continue
            _map_entry(prefix, tgt_key, dvals)

    for block_idx in range(num_single):
        prefix = f"single_transformer_blocks.{block_idx}."
        for tgt_key, dvals in DIFFUSERS_MAP.items():
            if not tgt_key.startswith("single_blocks."):
                continue
            _map_entry(prefix, tgt_key, dvals)

    block_idx = None
    for tgt_key, dvals in DIFFUSERS_MAP.items():
        if tgt_key.startswith(("double_blocks.", "single_blocks.")):
            continue
        _map_entry("", tgt_key, dvals)

    if "final_layer.adaLN_modulation.1.weight" in out:
        out["final_layer.adaLN_modulation.1.weight"] = _swap_scale_shift_matrix(
            out["final_layer.adaLN_modulation.1.weight"]
        )
    if "final_layer.adaLN_modulation.1.bias" in out:
        out["final_layer.adaLN_modulation.1.bias"] = _swap_scale_shift(
            out["final_layer.adaLN_modulation.1.bias"]
        )

    return out


class DiffusersSource:
    """
    Uniform interface over:
      1) Folder with index JSON + shards
      2) Folder with exactly one .safetensors (no index)
      3) Single .safetensors file
    Provides .has(key), .get(key)->Tensor, .base_keys (keys with 'model.' stripped for scanning)
    """

    POSSIBLE_PREFIXES = ["", "model."]  # try in this order

    def __init__(self, path: Path):
        p = Path(path)
        if p.is_dir():
            # use 'transformer' subfolder if present
            if (p / "transformer").is_dir():
                p = p / "transformer"
            self._init_from_dir(p)
        elif p.is_file() and p.suffix == ".safetensors":
            self._init_from_single_file(p)
        else:
            raise FileNotFoundError(f"Invalid path: {p}")

    # ---------- common helpers ----------

    @staticmethod
    def _strip_prefix(k: str) -> str:
        return k[6:] if k.startswith("model.") else k

    def _resolve(self, want: str):
        """
        Return the actual stored key matching `want` by trying known prefixes.
        """
        for pref in self.POSSIBLE_PREFIXES:
            k = pref + want
            if k in self._all_keys:
                return k
        return None

    def has(self, want: str) -> bool:
        return self._resolve(want) is not None

    def get(self, want: str) -> torch.Tensor:
        real_key = self._resolve(want)
        if real_key is None:
            raise KeyError(f"Missing key: {want}")
        return self._get_by_real_key(real_key).to("cpu")

    @property
    def base_keys(self):
        # keys without 'model.' prefix for scanning
        return [self._strip_prefix(k) for k in self._all_keys]

    # ---------- modes ----------

    def _init_from_single_file(self, file_path: Path):
        self._mode = "single"
        self._file = file_path
        self._handle = safe_open(file_path, framework="pt", device="cpu")
        self._all_keys = list(self._handle.keys())

        def _get_by_real_key(real_key: str):
            return self._handle.get_tensor(real_key)

        self._get_by_real_key = _get_by_real_key

    def _init_from_dir(self, dpath: Path):
        index_json = dpath / "diffusion_pytorch_model.safetensors.index.json"
        if index_json.exists():
            with open(index_json, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index["weight_map"]  # full mapping
            self._mode = "sharded"
            self._dpath = dpath
            self._weight_map = {k: dpath / v for k, v in weight_map.items()}
            self._all_keys = list(self._weight_map.keys())
            self._open_handles = {}

            def _get_by_real_key(real_key: str):
                fpath = self._weight_map[real_key]
                h = self._open_handles.get(fpath)
                if h is None:
                    h = safe_open(fpath, framework="pt", device="cpu")
                    self._open_handles[fpath] = h
                return h.get_tensor(real_key)

            self._get_by_real_key = _get_by_real_key
            return

        # no index: try exactly one safetensors in folder
        files = sorted(dpath.glob("*.safetensors"))
        if len(files) != 1:
            raise FileNotFoundError(
                f"No index found and {dpath} does not contain exactly one .safetensors file."
            )
        self._init_from_single_file(files[0])


def main():
    args = parse_args()
    src = DiffusersSource(Path(args.diffusers_path))
    orig = _convert_from_source(src, verbose=True)

    # Optional FP8 variants (experimental; not required for ComfyUI/BFL)
    if args.fp8 or args.fp8_scaled:
        dtype = torch.float8_e4m3fn  # noqa
        minv, maxv = torch.finfo(dtype).min, torch.finfo(dtype).max

        def stochastic_round_to(t):
            t = t.float().clamp(minv, maxv)
            lower = torch.floor(t * 256) / 256
            upper = torch.ceil(t * 256) / 256
            prob = torch.where(upper != lower, (t - lower) / (upper - lower), torch.zeros_like(t))
            rnd = torch.rand_like(t)
            out = torch.where(rnd < prob, upper, lower)
            return out.to(dtype)

        def scale_to_8bit(weight, target_max=416.0):
            absmax = weight.abs().max()
            scale = absmax / target_max if absmax > 0 else torch.tensor(1.0)
            scaled = (weight / scale).clamp(minv, maxv).to(dtype)
            return scaled, scale

        scales = {}
        for k in tqdm(list(orig.keys()), desc="Quantizing to fp8"):
            t = orig[k]
            if args.fp8:
                orig[k] = stochastic_round_to(t)
            else:
                if k.endswith(".weight") and t.dim() == 2:
                    qt, s = scale_to_8bit(t)
                    orig[k] = qt
                    scales[k[:-len(".weight")] + ".scale_weight"] = s
                else:
                    orig[k] = t.clamp(minv, maxv).to(dtype)
        if args.fp8_scaled:
            orig.update(scales)
            orig["scaled_fp8"] = torch.tensor([], dtype=dtype)
    else:
        # Default: save in bfloat16
        for k in list(orig.keys()):
            orig[k] = orig[k].to(torch.bfloat16).cpu()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = OrderedDict()
    meta["format"] = "pt"
    meta["modelspec.date"] = __import__("datetime").date.today().strftime("%Y-%m-%d")
    print(f"Saving transformer to: {out_path}")
    safetensors.torch.save_file(orig, str(out_path), metadata=meta)
    print("Done.")


if __name__ == "__main__":
    main()
