import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file as _st_save_file
from torch import nn

from .model import LoRAParametrization, _expand
from ...verbose import vprint

def apply_to_lora(fn):
    """apply a function to LoRAParametrization layers, designed to be used with model.apply"""

    def apply_fn(layer):
        if isinstance(layer, LoRAParametrization):
            fn(layer)

    return apply_fn

def enable_lora(model, lora_index=None):
    """Enable LoRA layers. If lora_index is None, enables all. If specified, enables only that index."""
    def _enable(layer):
        if isinstance(layer, LoRAParametrization):
            if lora_index is None or layer.lora_index == lora_index:
                layer.enable_lora()
    model.apply(_enable)

def disable_lora(model, lora_index=None):
    """Disable LoRA layers. If lora_index is None, disables all. If specified, disables only that index."""
    def _disable(layer):
        if isinstance(layer, LoRAParametrization):
            if lora_index is None or layer.lora_index == lora_index:
                layer.disable_lora()
    model.apply(_disable)

def get_lora_layers(model):
    layers = []
    for name, m in model.named_modules():
        plist = getattr(getattr(m, "parametrizations", None), "weight", None)
        if plist is None:
            continue
        for p in plist:
            if isinstance(p, LoRAParametrization):
                layers.append((name, p))
    return layers

def has_lora(model):
    """Return True if the model has at least one LoRAParametrization on any weight."""
    return len(get_lora_layers(model))>=1

def filter_lora_layers(model, lora_layer_filter, lora_index=None):
    """Enable/disable LoRA layers by name filter. If lora_index is specified, only affects that index."""
    lora_layer_filter = (lora_layer_filter or "").strip()
    # lora_layer_filter: logical OR on comma separated substrings
    # e.g. ".transformer.layers, .to_global_embed."
    def is_filtered(layer_name):
        filters = [x.lower().strip() for x in lora_layer_filter.split(",") if len(x)]
        for f in filters:
            for _f in _expand(f):
                if _f in layer_name:
                    return True
        return False
    for name, p in get_lora_layers(model):
        if lora_index is not None and p.lora_index != lora_index:
            continue
        if is_filtered(name):
            p.disable_lora()
        else:
            p.enable_lora()



# ------------------- helper function for collecting parameters for training/saving -------------------


def name_is_lora(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["lora_A", "lora_B", "M_xs", "magnitude", "magnitude_r", "magnitude_c"]
    )


def name_is_bias(name):
    return name.split(".")[-1] == "bias"


def get_params_by_name(model, print_shapes=False, name_filter=None):
    for n, p in model.named_parameters():
        if name_filter is None or name_filter(n):
            if print_shapes:
                print(n, p.shape)
            yield p


def get_lora_params(model, print_shapes=False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_lora)


def get_bias_params(model, print_shapes=False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_bias)


def get_lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if name_is_lora(k)}


def cast_base_to_precision(model, precision):
    """Cast frozen base weights to lower precision, keeping LoRA params in fp32.

    Args:
        model: Model with LoRA parametrizations applied.
        precision: One of "bf16", "bfloat16", "fp16", "float16".
    """
    import torch

    dtype_map = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "fp16": torch.float16, "float16": torch.float16}
    target_dtype = dtype_map.get(precision)
    if target_dtype is None:
        print(f"Unknown base_precision={precision!r}, keeping fp32")
        return
    vprint(f"Casting frozen base weights to {target_dtype}")
    model.to(target_dtype)
    # Restore LoRA params to fp32 â€” optimizer needs full precision
    n_restored = 0
    for p in get_lora_params(model):
        p.data = p.data.to(torch.float32)
        n_restored += 1
    vprint(f"  Restored {n_restored} LoRA params to fp32")


# ------------------- helper function for loading LoRa at inference -------------------

def infer_global_rank(lora_sd: dict) -> int:
    candidates = []
    for k, v in lora_sd.items():
        if k.endswith(".lora_A"):
            candidates.append(v.shape[0])
        elif k.endswith(".lora_B"):
            candidates.append(v.shape[1])
        elif k.endswith(".M_xs"):
            candidates.append(v.shape[0])
    if not candidates:
        raise ValueError("No LoRA/LoRA-XS tensors found to infer rank")
    r = candidates[0]
    # sanity check they all match (most checkpoints use one global r)
    if any(c != r for c in candidates):
        print(f"Multiple ranks detected in ckpt: {sorted(set(candidates))}. "
                         "Youâ€™ll need per-layer ranks.")
    return r


def detect_dora_variant(state_dict):
    """Detect DoRA variant from a state dict.

    Returns "dora-cols" (dim=0, per-input-feature) or "dora-rows" (dim=1,
    per-output-neuron) based on the shape of the first .magnitude tensor
    found. Returns None if no magnitude tensors exist or axis is ambiguous.
    """
    for key, val in state_dict.items():
        if not key.endswith(".magnitude"):
            continue
        if val.dim() == 2:
            if val.shape[0] == 1:
                return "dora-cols"   # old format: (1, fan_in)
            elif val.shape[1] == 1:
                return "dora-rows"   # (fan_out, 1)
        # 1D magnitude â€” can't determine axis without model context
        return None
    return None


def prepare_dora_state_dict(state_dict):
    """Squeeze 2D DoRA magnitude tensors to 1D for loading. Modifies in-place."""
    for key in list(state_dict.keys()):
        if key.endswith(".magnitude") and state_dict[key].dim() == 2:
            state_dict[key] = state_dict[key].squeeze()


def resolve_adapter_type(adapter_type, state_dict=None):
    """Resolve legacy "dora" adapter_type to the correct variant.

    If adapter_type is "dora" and a checkpoint state_dict is provided,
    attempts to detect dora-rows vs dora-cols from magnitude tensor shapes.
    Falls back to "dora-rows" (paper-correct default).
    All other adapter types are returned unchanged.
    """
    if adapter_type != "dora":
        return adapter_type
    if state_dict is not None:
        detected = detect_dora_variant(state_dict)
        if detected:
            return detected
    return "dora-rows"


# ------------------- helper functions for safetensors LoRA checkpoints -------------------


def save_lora_safetensors(state_dict, lora_config, path):
    """Save a LoRA checkpoint in safetensors format with config as metadata.

    The lora_config dict is JSON-serialized and stored under the metadata key
    ``"lora_config"``, which safetensors stores in the file header as a string.

    Args:
        state_dict: Dict of LoRA tensors (from get_lora_state_dict).
        lora_config: Dict with keys like rank, alpha, adapter_type, include, exclude.
        path: Output file path (should end in .safetensors).
    """
    metadata = {"lora_config": json.dumps(lora_config)}
    fp16_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    _st_save_file(fp16_dict, str(path), metadata=metadata)


def load_lora_checkpoint(path):
    """Load a LoRA checkpoint from either .safetensors or .ckpt format.

    Returns:
        Tuple of (state_dict, lora_config) where lora_config may be an empty
        dict if the checkpoint has no embedded config.
    """
    path = str(path)
    if path.endswith(".safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
            metadata = f.metadata()
        lora_config = {}
        if metadata and "lora_config" in metadata:
            lora_config = json.loads(metadata["lora_config"])
        return state_dict, lora_config
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        lora_config = ckpt.get("lora_config", {})
        return state_dict, lora_config


def convert_lora_ckpt_to_safetensors(ckpt_path, output_path=None):
    """Convert a .ckpt LoRA checkpoint to .safetensors format.

    Args:
        ckpt_path: Path to the input .ckpt file.
        output_path: Path for the output .safetensors file.
            If None, replaces .ckpt extension with .safetensors.

    Returns:
        The output path.
    """
    ckpt_path = str(ckpt_path)
    if output_path is None:
        if ckpt_path.endswith(".ckpt"):
            output_path = ckpt_path[:-5] + ".safetensors"
        else:
            output_path = ckpt_path + ".safetensors"
    state_dict, lora_config = load_lora_checkpoint(ckpt_path)
    save_lora_safetensors(state_dict, lora_config, output_path)
    return output_path


# ------------------- helper functions for multi-LoRA stacking -------------------


def get_lora_count(model):
    """Return the number of stacked LoRAs (distinct lora_index values)."""
    indices = set()
    for _, p in get_lora_layers(model):
        indices.add(p.lora_index)
    return len(indices)


def remap_lora_state_dict(state_dict, target_index):
    """Remap LoRA state dict keys from parametrization index 0 to target_index.

    Checkpoints always save keys with index 0 (e.g. *.parametrizations.weight.0.lora_A).
    When loading the Nth stacked LoRA, we need to remap to index N.
    """
    if target_index == 0:
        return state_dict
    remapped = {}
    for k, v in state_dict.items():
        new_key = k.replace(
            ".parametrizations.weight.0.",
            ".parametrizations.weight.{}.".format(target_index)
        )
        remapped[new_key] = v
    return remapped


# ------------------- helper function for inferencing with multiple lora -------------------


def _get_adapter_param_names(lora_layer):
    """Return the list of trainable parameter names for this adapter type."""
    if lora_layer.adapter_type == "lora":
        return ["lora_A", "lora_B"]
    elif lora_layer.adapter_type in ("dora", "dora-rows", "dora-cols"):
        return ["lora_A", "lora_B", "magnitude"]
    elif lora_layer.adapter_type == "bora":
        return ["lora_A", "lora_B", "magnitude_r", "magnitude_c"]
    elif lora_layer.adapter_type == "lora-xs":
        return ["M_xs"]
    elif lora_layer.adapter_type in ("dora-rows-xs", "dora-cols-xs"):
        return ["M_xs", "magnitude"]
    elif lora_layer.adapter_type == "bora-xs":
        return ["M_xs", "magnitude_r", "magnitude_c"]
    return []


def _prepare_for_multiple_lora(lora_layer):
    lora_layer._multi_lora_slots = []


def _append_lora(lora_layer):
    slot = {}
    for name in _get_adapter_param_names(lora_layer):
        slot[name] = nn.Parameter(getattr(lora_layer, name).clone())
    lora_layer._multi_lora_slots.append(slot)


def load_multiple_lora(model, lora_state_dicts):
    model.apply(apply_to_lora(_prepare_for_multiple_lora))
    for state_dict in lora_state_dicts:
        _ = model.load_state_dict(state_dict, strict=False)
        model.apply(apply_to_lora(_append_lora))
    return model


def _select_lora(lora_layer, index):
    slot = lora_layer._multi_lora_slots[index]
    for name, param in slot.items():
        setattr(lora_layer, name, param)


def select_lora(model, index):
    model.apply(apply_to_lora(lambda x: _select_lora(x, index)))
    return model

# ------------------- helper function for merging multiple LoRAs -------------------

def merge_loras_into_base_model(model, lora_configurations):
    """
    Merges multiple LoRAs with specified application weights directly into the base model's weights.
    Works with all adapter types (LoRA, DoRA, LoRA-XS) by using the parametrization's own forward.
    After merging, the LoRA parametrization on each layer is disabled.

    Args:
        model: The PyTorch model with LoRAParametrization already applied (via add_lora).
        lora_configurations: A list of dicts, each specifying a LoRA to merge:
            [
                {
                    'name': 'style1',
                    'state_dict': loaded_torch_state_dict,
                    'application_weight': 0.7
                },
                ...
            ]
    """
    import torch

    if not lora_configurations:
        return

    # Collect all parametrized layers with their LoRA instances
    parametrized_layers = []
    for name, module in model.named_modules():
        if not hasattr(module, "parametrizations"):
            continue
        for attr_name in list(module.parametrizations.keys()):
            plist = module.parametrizations[attr_name]
            for p in plist:
                if isinstance(p, LoRAParametrization):
                    parametrized_layers.append((name, module, attr_name, p, plist.original))
                    break

    # Initialize per-layer delta accumulators
    deltas = {id(orig): torch.zeros_like(orig.data) for _, _, _, _, orig in parametrized_layers}

    for lora_cfg in lora_configurations:
        lora_sd = lora_cfg['state_dict']
        app_weight = lora_cfg['application_weight']

        # Load this LoRA's adapter weights into the model's parametrizations
        model.load_state_dict(lora_sd, strict=False)

        # Compute each layer's delta using the parametrization's own forward
        with torch.no_grad():
            for name, module, attr_name, lora_p, original in parametrized_layers:
                merged = lora_p(original)
                deltas[id(original)] += app_weight * (merged - original)

    # Apply accumulated deltas to original weights and disable LoRA
    with torch.no_grad():
        for name, module, attr_name, lora_p, original in parametrized_layers:
            original.data.add_(deltas[id(original)])
            lora_p.disable_lora()

# ------------------- helper function for tying and untieing weights -------------------


def tie_weights(linear: nn.Linear, embedding: nn.Embedding):
    """tie the weights of the linear layer and the embedding layer both with the same lora"""
    lora_p = embedding.parametrizations.weight[0]
    if lora_p.adapter_type != "lora":
        raise NotImplementedError(f"tie_weights only supports standard LoRA, got adapter_type='{lora_p.adapter_type}'")
    # this line below is optional if the original is already tied
    embedding.parametrizations.weight.original = linear.parametrizations.weight.original
    embedding.parametrizations.weight[0].lora_A = linear.parametrizations.weight[0].lora_B
    embedding.parametrizations.weight[0].lora_B = linear.parametrizations.weight[0].lora_A


def untie_weights(linear: nn.Linear, embedding: nn.Embedding):
    """untie the weights of the linear layer and the embedding layer"""
    lora_p = embedding.parametrizations.weight[0]
    if lora_p.adapter_type != "lora":
        raise NotImplementedError(f"untie_weights only supports standard LoRA, got adapter_type='{lora_p.adapter_type}'")
    embedding.parametrizations.weight.original = nn.Parameter(embedding.weight.original.clone())
    embedding.parametrizations.weight[0].lora_A = nn.Parameter(embedding.parametrizations.weight[0].lora_A.clone())
    embedding.parametrizations.weight[0].lora_B = nn.Parameter(embedding.parametrizations.weight[0].lora_B.clone())
