from __future__ import annotations


_STANDARD_GGUF_BLOCK_PREFIXES = ("blk.", "enc.blk.", "dec.blk.")


def has_standard_gguf_tensor_names(state_dict) -> bool:
    if not state_dict or "token_embd.weight" not in state_dict:
        return False
    return any(name.startswith(_STANDARD_GGUF_BLOCK_PREFIXES) for name in state_dict)


def remap_named_mapping(mapping, name_map, *, keep_unmapped=True):
    if mapping is None:
        return None
    remapped = mapping.__class__()
    for name, value in mapping.items():
        mapped_name = name_map.get(name)
        if mapped_name is None:
            if not keep_unmapped:
                continue
            mapped_name = name
        if mapped_name in remapped:
            raise ValueError(f"Duplicate weight after GGUF mapping: {mapped_name}")
        if isinstance(value, list):
            value = [name_map.get(item, item) for item in value]
        remapped[mapped_name] = value
    return remapped


def remap_state_dict_triplet(state_dict, quantization_map, tied_weights_map, name_map, *, keep_unmapped=True):
    return (
        remap_named_mapping(state_dict, name_map, keep_unmapped=keep_unmapped),
        remap_named_mapping(quantization_map, name_map, keep_unmapped=keep_unmapped),
        remap_named_mapping(tied_weights_map, name_map, keep_unmapped=keep_unmapped),
    )

