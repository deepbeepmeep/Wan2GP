"""Translate Ming transformer adapters into the MMGP wrapper namespace."""

from shared.utils.lora_mapping import compose_linear_lokr_factors, convert_lora_keys


def preprocess_ming_loras(model, state_dict):
    if any(key.endswith(".lokr_t2") for key in state_dict):
        raise ValueError("Convolutional CP LoKr factors are not supported by Ming's linear transformer")
    state_dict = compose_linear_lokr_factors(state_dict)
    # Include both the upstream inner names and their WanGP wrapper names so
    # Diffusers/PEFT and flattened Kohya adapters use the shared converter.
    inner_names = {name for name, _ in model.transformer.named_modules() if name}
    names = inner_names | {name for name, _ in model.named_modules() if name}
    converted = convert_lora_keys(state_dict, names, target="wangp")
    result = {}
    for key, value in converted.items():
        if not key.startswith("diffusion_model."):
            raise ValueError(f"Unsupported Ming adapter key: {key}")
        module_key = key[len("diffusion_model."):]
        if not module_key.startswith(("transformer.", "mlp.")):
            module_key = "transformer." + module_key
        # MMGP removes the leading diffusion_model namespace before looking
        # up modules in ConditionedTransformer.named_modules().
        destination = "diffusion_model." + module_key
        if destination in result:
            raise ValueError(f"Ming LoRA keys collide at {destination}")
        result[destination] = value
    return result
