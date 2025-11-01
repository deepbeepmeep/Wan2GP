"""
wan_convert_universal — Diffusers → Wan (T2V/I2V) canonical naming, single format.

- No model detection or flags.
- Shared keys get the same canonical names used by both T2V and I2V loaders.
- I2V extras are mapped only if present in the source.

API:
    new_sd = convert_state_dict_universal(state_dict, cast_dtype=None)
    convert_files_universal([...safetensors...], "out/model.safetensors", cast_dtype=None)
"""

import os, re
from typing import Dict, List, Optional
import torch
from safetensors.torch import safe_open, save_file

_RE_BLOCK = r"^blocks\.(\d+)\."

def _common_unet(src: str) -> str:
    s = src
    # attn1/attn2 → self_attn/cross_attn
    s = re.sub(rf"{_RE_BLOCK}attn1\.", r"blocks.\1.self_attn.", s)
    s = re.sub(rf"{_RE_BLOCK}attn2\.", r"blocks.\1.cross_attn.", s)
    # to_q/k/v/out.0 → q/k/v/o (any *attn* path)
    s = re.sub(r"(\b[^.\s]*attn[^.\s]*\b.*?\.)to_q\.", r"\1q.", s)
    s = re.sub(r"(\b[^.\s]*attn[^.\s]*\b.*?\.)to_k\.", r"\1k.", s)
    s = re.sub(r"(\b[^.\s]*attn[^.\s]*\b.*?\.)to_v\.", r"\1v.", s)
    s = re.sub(r"(\b[^.\s]*attn[^.\s]*\b.*?\.)to_out\.0\.", r"\1o.", s)
    # ffn.net.0.proj → ffn.0 ; ffn.net.2 → ffn.2
    s = re.sub(rf"{_RE_BLOCK}ffn\.net\.0\.proj\.", r"blocks.\1.ffn.0.", s)
    s = re.sub(rf"{_RE_BLOCK}ffn\.net\.2\.", r"blocks.\1.ffn.2.", s)
    return s

def rename_key_universal(src: str) -> str:
    """
    Canonical Wan naming (works for both T2V and I2V shared keys).
    """
    s = _common_unet(src)

    # Cross-attn image projections (only hit if present in the source)
    s = re.sub(rf"{_RE_BLOCK}cross_attn\.add_k_proj\.", r"blocks.\1.cross_attn.k_img.", s)
    s = re.sub(rf"{_RE_BLOCK}cross_attn\.add_v_proj\.", r"blocks.\1.cross_attn.v_img.", s)
    s = re.sub(rf"{_RE_BLOCK}cross_attn\.norm_added_k\.", r"blocks.\1.cross_attn.norm_k_img.", s)

    # Block-level canonical names
    s = re.sub(rf"{_RE_BLOCK}scale_shift_table$", r"blocks.\1.modulation", s)  # shared canonical
    s = re.sub(rf"{_RE_BLOCK}norm2\b", r"blocks.\1.norm3", s)                  # shared canonical

    # Conditioning MLPs (canonical heads)
    s = re.sub(r"^condition_embedder\.text_embedder\.linear_1\.", r"text_embedding.0.", s)
    s = re.sub(r"^condition_embedder\.text_embedder\.linear_2\.", r"text_embedding.2.", s)

    s = re.sub(r"^condition_embedder\.time_embedder\.linear_1\.", r"time_embedding.0.", s)
    s = re.sub(r"^condition_embedder\.time_embedder\.linear_2\.", r"time_embedding.2.", s)

    # time projection single linear → time_projection.1.*
    s = re.sub(r"^condition_embedder\.time_proj\.", r"time_projection.1.", s)

    # Image conditioner "head" (corrected ordering!)
    s = re.sub(r"^condition_embedder\.image_embedder\.norm1\.", r"img_emb.proj.0.", s)           # LN (vector)
    s = re.sub(r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.", r"img_emb.proj.1.", s) # Linear (matrix)
    s = re.sub(r"^condition_embedder\.image_embedder\.ff\.net\.2\.", r"img_emb.proj.3.", s)       # Linear (matrix)
    s = re.sub(r"^condition_embedder\.image_embedder\.norm2\.", r"img_emb.proj.4.", s)            # LN (vector)

    # Output head: canonical head.head.* ; top-level scale_shift_table → head.modulation
    s = re.sub(r"^proj_out\.", r"head.head.", s)
    if s == "scale_shift_table":
        s = "head.modulation"

    # patch_embedding.* remains unchanged
    return s

# ---------- Public helpers (single canonical format) ----------

def convert_state_dict_universal(
    state_dict: Dict[str, torch.Tensor],
    cast_dtype: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Map a Diffusers-style state_dict to the canonical Wan naming (T2V/I2V shared).
    """
    dtype = None
    if cast_dtype:
        cd = cast_dtype.lower().strip()
        if cd in ("float16", "fp16", "half"): dtype = torch.float16
        elif cd in ("bfloat16", "bf16"):      dtype = torch.bfloat16
        elif cd in ("float32", "fp32"):       dtype = torch.float32
        else: raise ValueError(f"Unsupported cast_dtype: {cast_dtype}")

    out: Dict[str, torch.Tensor] = {}
    for k, t in state_dict.items():
        out[rename_key_universal(k)] = t.to(dtype) if dtype is not None else t
    return out

def convert_files_universal(
    input_paths: List[str],
    out_safetensors: str,
    cast_dtype: Optional[str] = None,
) -> None:
    """
    Stream one/many Diffusers .safetensors into a single canonical Wan .safetensors.
    """
    dtype = None
    if cast_dtype:
        cd = cast_dtype.lower().strip()
        if cd in ("float16", "fp16", "half"): dtype = torch.float16
        elif cd in ("bfloat16", "bf16"):      dtype = torch.bfloat16
        elif cd in ("float32", "fp32"):       dtype = torch.float32
        else: raise ValueError(f"Unsupported cast_dtype: {cast_dtype}")

    os.makedirs(os.path.dirname(out_safetensors) or ".", exist_ok=True)
    out: Dict[str, torch.Tensor] = {}
    for p in input_paths:
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if dtype is not None: t = t.to(dtype)
                out[rename_key_universal(k)] = t

    save_file(out, out_safetensors, metadata={
        "format": "wan_universal",
        "converted_from": "diffusers",
        "script": "wan_convert_universal",
        "cast_dtype": str(dtype) if dtype is not None else "unchanged",
    })



def main():


    from mmgp import safetensors2
    files = [f"c:/temp/chrono/diffusion_pytorch_model-{i:05d}-of-00014.safetensors" for i in range(1, 15)]
    new_sd = {}
    for file in files:
        sd = safetensors2.torch_load_file(file)
        conv_sd = convert_state_dict_universal(sd) #, cast_dtype="bf16"
        sd = None
        new_sd.update(conv_sd)

    safetensors2.torch_write_file(new_sd, "chrono.safetensors")

if __name__ == "__main__":
    main()

