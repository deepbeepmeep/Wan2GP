"""Shared adapter format helpers for LoRA and linear LoKr checkpoints.

``convert_lora_keys`` changes names only; tensor values and architecture layouts
are unchanged. ``compose_linear_lokr_factors`` is an opt-in conversion for
decomposed linear LoKr factors. Underscore-separated module names are resolved
against the actual architecture instead of splitting names such as ``to_out``.
"""

import torch


_DECOMPOSED_LOKR = ("lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b")


def compose_linear_lokr_factors(state_dict):
    """Fold AI Toolkit's linear LoKr matrix products into full w1/w2 factors.

    Call only for linear targets. The result is a new dictionary; source
    tensors are untouched. This deliberately does not handle convolutional CP
    factors, which need a different tensor transformation.
    """
    state_dict = dict(state_dict)
    groups = {}
    for key in state_dict:
        module, dot, suffix = key.rpartition(".")
        if dot and suffix in _DECOMPOSED_LOKR:
            groups.setdefault(module, set()).add(suffix)

    for module, suffixes in groups.items():
        rank = None
        factors = {}
        for part in ("w1", "w2"):
            a_name, b_name = f"lokr_{part}_a", f"lokr_{part}_b"
            full_name = f"{module}.lokr_{part}"
            if (a_name in suffixes) != (b_name in suffixes):
                raise ValueError(f"Incomplete AI Toolkit LoKr {part} factors for {module}")
            if a_name in suffixes:
                if full_name in state_dict:
                    raise ValueError(f"Both full and decomposed LoKr {part} factors for {module}")
                a = state_dict.pop(f"{module}.{a_name}")
                b = state_dict.pop(f"{module}.{b_name}")
                if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                    raise ValueError(f"Invalid AI Toolkit LoKr {part} matrix shapes for {module}")
                if rank is not None and rank != a.shape[1]:
                    raise ValueError(f"Inconsistent AI Toolkit LoKr ranks for {module}")
                factors[full_name] = a @ b
                rank = a.shape[1]

        if rank is None:
            continue
        if rank <= 0:
            raise ValueError(f"Invalid AI Toolkit LoKr rank for {module}")
        if not (f"{module}.lokr_w1" in factors or f"{module}.lokr_w1" in state_dict):
            raise ValueError(f"Missing AI Toolkit LoKr w1 factor for {module}")
        if not (f"{module}.lokr_w2" in factors or f"{module}.lokr_w2" in state_dict):
            raise ValueError(f"Missing AI Toolkit LoKr w2 factor for {module}")
        alpha_key = f"{module}.alpha"
        if alpha_key in state_dict:
            alpha = state_dict.pop(alpha_key)
            alpha = float(alpha.item()) if torch.is_tensor(alpha) else float(alpha)
            scale = alpha / rank
            w1_key = f"{module}.lokr_w1"
            if w1_key in factors:
                factors[w1_key] = factors[w1_key] * scale
            elif w1_key in state_dict:
                state_dict[w1_key] = state_dict[w1_key] * scale
        for key, value in factors.items():
            if key in state_dict:
                raise ValueError(f"LoKr keys collide at {key}")
            state_dict[key] = value
    return state_dict


def convert_lora_keys(state_dict, module_names, target="wangp", fused_split_map=None, split_linear_modules_map=None):
    if target not in ("wangp", "diffusers", "kohya"):
        raise ValueError(f"Unknown LoRA naming format: {target}")
    module_names = set(module_names)
    for name in tuple(module_names):
        for fused, spec in (fused_split_map or {}).items():
            if name == fused or name.endswith("." + fused):
                prefix = name[:-len(fused)]
                module_names.update(prefix + alias for alias in spec["mapped_modules"])
        # LoRAs can also target a fused source projection when the model keeps
        # its outputs separate. MMGP applies this map to the adapter tensors.
        for fused, spec in (split_linear_modules_map or {}).items():
            for split in spec["mapped_modules"]:
                if name == split or name.endswith("." + split):
                    module_names.add(name[:-len(split)] + fused)
    encoded = {}
    for name in module_names:
        key = name.replace(".", "_")
        if key in encoded and encoded[key] != name:
            raise ValueError(f"Ambiguous underscore module name: {key}")
        encoded[key] = name
    result = {}
    suffixes = (".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight", ".alpha", ".dora_scale", ".lokr_w1", ".lokr_w2")
    for key, value in state_dict.items():
        normalized = key.replace(".lora.", ".lora_").replace(".default.weight", ".weight")
        suffix = next((suffix for suffix in suffixes if normalized.endswith(suffix)), None)
        if suffix is None:
            result[key] = value
            continue
        module = normalized[:-len(suffix)]
        if module.startswith("lora_unet_"):
            encoded_name = module[len("lora_unet_"):]
            if encoded_name not in encoded:
                raise ValueError(f"Unknown Kohya LoRA module: {encoded_name}")
            module = encoded[encoded_name]
        else:
            for prefix in ("base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer."):
                if module.startswith(prefix):
                    module = module[len(prefix):]
        if target == "kohya":
            suffix = suffix.replace(".lora_A.", ".lora_down.").replace(".lora_B.", ".lora_up.")
            destination = "lora_unet_" + module.replace(".", "_") + suffix
        else:
            suffix = suffix.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
            destination = ("diffusion_model." if target == "wangp" else "transformer.") + module + suffix
        if destination in result:
            raise ValueError(f"LoRA keys collide at {destination}")
        result[destination] = value
    return result
