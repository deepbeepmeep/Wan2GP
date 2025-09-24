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

    "double_blocks.().img_attn.qkv.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    "double_blocks.().img_attn.qkv.bias":   ["attn.to_q.bias",   "attn.to_k.bias",   "attn.to_v.bias"],
    "double_blocks.().txt_attn.qkv.weight": ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
    "double_blocks.().txt_attn.qkv.bias":   ["attn.add_q_proj.bias",   "attn.add_k_proj.bias",   "attn.add_v_proj.bias"],

    "double_blocks.().img_attn.norm.query_norm.scale": ["attn.norm_q.weight"],
    "double_blocks.().img_attn.norm.key_norm.scale":   ["attn.norm_k.weight"],
    "double_blocks.().txt_attn.norm.query_norm.scale": ["attn.norm_added_q.weight"],
    "double_blocks.().txt_attn.norm.key_norm.scale":   ["attn.norm_added_k.weight"],

    "double_blocks.().img_mlp.0.weight": ["ff.net.0.proj.weight"],
    "double_blocks.().img_mlp.0.bias":   ["ff.net.0.proj.bias"],
    "double_blocks.().img_mlp.2.weight": ["ff.net.2.weight"],
    "double_blocks.().img_mlp.2.bias":   ["ff.net.2.bias"],

    "double_blocks.().txt_mlp.0.weight": ["ff_context.net.0.proj.weight"],
    "double_blocks.().txt_mlp.0.bias":   ["ff_context.net.0.proj.bias"],
    "double_blocks.().txt_mlp.2.weight": ["ff_context.net.2.weight"],
    "double_blocks.().txt_mlp.2.bias":   ["ff_context.net.2.bias"],

    "double_blocks.().img_attn.proj.weight": ["attn.to_out.0.weight"],
    "double_blocks.().img_attn.proj.bias":   ["attn.to_out.0.bias"],
    "double_blocks.().txt_attn.proj.weight": ["attn.to_add_out.weight"],
    "double_blocks.().txt_attn.proj.bias":   ["attn.to_add_out.bias"],

    # single-stream blocks
    "single_blocks.().modulation.lin.weight": ["norm.linear.weight"],
    "single_blocks.().modulation.lin.bias":   ["norm.linear.bias"],
    "single_blocks.().linear1.weight":        ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "proj_mlp.weight"],
    "single_blocks.().linear1.bias":          ["attn.to_q.bias",   "attn.to_k.bias",   "attn.to_v.bias",   "proj_mlp.bias"],
    "single_blocks.().norm.query_norm.scale": ["attn.norm_q.weight"],
    "single_blocks.().norm.key_norm.scale":   ["attn.norm_k.weight"],
    "single_blocks.().linear2.weight":        ["proj_out.weight"],
    "single_blocks.().linear2.bias":          ["proj_out.bias"],

    # final
    "final_layer.linear.weight":              ["proj_out.weight"],
    "final_layer.linear.bias":                ["proj_out.bias"],
    # these two are built from norm_out.linear.{weight,bias} by swapping [shift,scale] -> [scale,shift]
    "final_layer.adaLN_modulation.1.weight":  ["norm_out.linear.weight"],
    "final_layer.adaLN_modulation.1.bias":    ["norm_out.linear.bias"],
}


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

    # Count blocks by scanning base keys (with any 'model.' prefix removed)
    num_dual = 0
    num_single = 0
    for k in src.base_keys:
        if k.startswith("transformer_blocks."):
            try:
                i = int(k.split(".")[1])
                num_dual = max(num_dual, i + 1)
            except Exception:
                pass
        elif k.startswith("single_transformer_blocks."):
            try:
                i = int(k.split(".")[1])
                num_single = max(num_single, i + 1)
            except Exception:
                pass
    print(f"Found {num_dual} dual-stream blocks, {num_single} single-stream blocks")

    # Swap [shift, scale] -> [scale, shift] (weights are concatenated along dim=0)
    def swap_scale_shift(vec: torch.Tensor) -> torch.Tensor:
        shift, scale = vec.chunk(2, dim=0)
        return torch.cat([scale, shift], dim=0)

    orig = {}

    # Per-block (dual)
    for b in range(num_dual):
        prefix = f"transformer_blocks.{b}."
        for okey, dvals in DIFFUSERS_MAP.items():
            if not okey.startswith("double_blocks."):
                continue
            dkeys = [prefix + v for v in dvals]
            if not all(src.has(k) for k in dkeys):
                continue
            if len(dkeys) == 1:
                orig[okey.replace("()", str(b))] = src.get(dkeys[0])
            else:
                orig[okey.replace("()", str(b))] = torch.cat([src.get(k) for k in dkeys], dim=0)

    # Per-block (single)
    for b in range(num_single):
        prefix = f"single_transformer_blocks.{b}."
        for okey, dvals in DIFFUSERS_MAP.items():
            if not okey.startswith("single_blocks."):
                continue
            dkeys = [prefix + v for v in dvals]
            if not all(src.has(k) for k in dkeys):
                continue
            if len(dkeys) == 1:
                orig[okey.replace("()", str(b))] = src.get(dkeys[0])
            else:
                orig[okey.replace("()", str(b))] = torch.cat([src.get(k) for k in dkeys], dim=0)

    # Globals (non-block)
    for okey, dvals in DIFFUSERS_MAP.items():
        if okey.startswith(("double_blocks.", "single_blocks.")):
            continue
        dkeys = dvals
        if not all(src.has(k) for k in dkeys):
            continue
        if len(dkeys) == 1:
            orig[okey] = src.get(dkeys[0])
        else:
            orig[okey] = torch.cat([src.get(k) for k in dkeys], dim=0)

    # Fix final_layer.adaLN_modulation.1.{weight,bias} by swapping scale/shift halves
    if "final_layer.adaLN_modulation.1.weight" in orig:
        orig["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(
            orig["final_layer.adaLN_modulation.1.weight"]
        )
    if "final_layer.adaLN_modulation.1.bias" in orig:
        orig["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(
            orig["final_layer.adaLN_modulation.1.bias"]
        )

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
