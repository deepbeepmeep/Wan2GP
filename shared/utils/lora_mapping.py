"""Lossless LoRA key conversion between Diffusers/PEFT, WanGP and Kohya names.

Tensor values and architecture layouts are unchanged. Underscore-separated module
names are resolved against the actual architecture instead of splitting underscores
inside names such as ``transformer_blocks`` or ``to_out``.
"""


def convert_lora_keys(state_dict, module_names, target="wangp", fused_split_map=None):
    if target not in ("wangp", "diffusers", "kohya"):
        raise ValueError(f"Unknown LoRA naming format: {target}")
    module_names = set(module_names)
    for name in tuple(module_names):
        for fused, spec in (fused_split_map or {}).items():
            if name == fused or name.endswith("." + fused):
                prefix = name[:-len(fused)]
                module_names.update(prefix + alias for alias in spec["mapped_modules"])
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
