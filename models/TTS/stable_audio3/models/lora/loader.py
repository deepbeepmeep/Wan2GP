"""Helpers for loading pre-trained LoRA checkpoints onto a live model."""
import os
from functools import partial

import torch

from .model import LoRAParametrization, add_lora
from .utils import (
    get_lora_layers,
    infer_global_rank,
    load_lora_checkpoint,
    prepare_dora_state_dict,
    remap_lora_state_dict,
    resolve_adapter_type,
)
from ...verbose import vprint


def load_and_apply_loras(model, lora_ckpt_paths, model_type, svd_bases_path=None):
    """Load LoRA checkpoints from disk and attach them to `model`.

    - Uses a two-pass approach: first resolves each LoRA's adapter type, then
      loads SVD bases (only if any LoRA is -XS), then applies each LoRA.
    - Handles legacy "dora" adapter_type via `resolve_adapter_type`.
    - Only passes svd_bases to -XS LoRAs to avoid spurious "no -XS layers"
      messages for non-XS LoRAs.
    - Sets `model.use_lora = True` and `model.lora_names = [...]` as a
      convenience for downstream UI code.

    Args:
        model: The loaded model. For `diffusion_cond` / `diffusion_cond_inpaint`
            types, LoRAs are applied to `model.model` and `model.conditioner`;
            otherwise they are applied to `model` directly.
        lora_ckpt_paths: List of paths to `.ckpt` or `.safetensors` files.
        model_type: String from the model config, e.g. "diffusion_cond".
        svd_bases_path: Optional path to a precomputed SVD bases .pt file
            (used only by -XS adapter types).

    Returns:
        List of display names (file stems) in the same order as the input paths.
    """
    if not lora_ckpt_paths:
        model.use_lora = False
        model.lora_names = []
        return []

    is_cond = model_type in ("diffusion_cond", "diffusion_cond_inpaint")

    # Pass 1: load each checkpoint and resolve adapter type.
    lora_entries = []  # (path, state_dict, config_dict, adapter_type)
    for i, lora_path in enumerate(lora_ckpt_paths):
        print(f"Loading LoRA {i} from {lora_path}")
        state_dict, config_dict = load_lora_checkpoint(lora_path)
        adapter_type_raw = config_dict.get("adapter_type", "lora")
        adapter_type = resolve_adapter_type(adapter_type_raw, state_dict)
        if adapter_type != adapter_type_raw:
            vprint(f"Resolved legacy '{adapter_type_raw}' -> {adapter_type}")
        lora_entries.append((lora_path, state_dict, config_dict, adapter_type))

    # Load SVD bases only if at least one LoRA is -XS.
    model_svd_bases = None
    cond_svd_bases = None
    any_xs = any(t.endswith("-xs") for _, _, _, t in lora_entries)
    if any_xs:
        if svd_bases_path is not None:
            vprint(f"Loading SVD bases from {svd_bases_path}")
            svd_bases_all = torch.load(svd_bases_path, map_location="cpu", weights_only=True)
            model_svd_bases = {k[len("model."):]: v for k, v in svd_bases_all.items() if k.startswith("model.")}
            cond_svd_bases = {k[len("conditioner."):]: v for k, v in svd_bases_all.items() if k.startswith("conditioner.")}
            vprint(f"  model bases: {len(model_svd_bases)}, conditioner bases: {len(cond_svd_bases)}")
        else:
            print("WARNING: -XS adapter type present without svd_bases_path -- SVD will be computed on current device")

    # Pass 2: apply each LoRA.
    lora_names = []
    for i, (lora_path, state_dict, config_dict, adapter_type) in enumerate(lora_entries):
        rank = config_dict.get("rank", infer_global_rank(state_dict))
        alpha = config_dict.get("alpha", rank)
        include = config_dict.get("include", None)
        exclude = config_dict.get("exclude", None)
        is_xs = adapter_type.endswith("-xs")

        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=alpha, adapter_type=adapter_type, lora_index=i),
            },
            torch.nn.Conv1d: {
                "weight": partial(LoRAParametrization.from_conv1d, rank=rank, lora_alpha=alpha, adapter_type=adapter_type, lora_index=i),
            },
        }
        if is_cond:
            add_lora(model.model, lora_config, include=include, exclude=exclude,
                     svd_bases=model_svd_bases if is_xs else None)
            add_lora(model.conditioner, lora_config, include=include, exclude=exclude,
                     svd_bases=cond_svd_bases if is_xs else None)
        else:
            add_lora(model, lora_config, include=include, exclude=exclude,
                     svd_bases=model_svd_bases if is_xs else None)

        prepare_dora_state_dict(state_dict)
        remapped_sd = remap_lora_state_dict(state_dict, i)
        if is_cond:
            model.model.load_state_dict(remapped_sd, strict=False)
            model.conditioner.load_state_dict(remapped_sd, strict=False)
        else:
            model.load_state_dict(remapped_sd, strict=False)
        lora_names.append(os.path.splitext(os.path.basename(lora_path))[0])

    vprint("lora layers:", len(get_lora_layers(model)))
    model.use_lora = True
    model.lora_names = lora_names
    return lora_names
