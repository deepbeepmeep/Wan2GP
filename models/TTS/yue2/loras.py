"""Route YuE2 AR and acoustic adapters without changing their tensor deltas."""

import re

import torch
from torch import nn

from shared.lora_mapper import LoraKeyMapper


class YuE2LoraTarget(nn.Module):
    def __init__(self, ar, acoustic):
        super().__init__()
        self.ar = ar
        self.acoustic = acoustic
        self._subloras = {"ar": ar, "acoustic": acoustic}
        self._bound = False

    def bind(self):
        # MMGP profiles the components independently. Share their existing adapter
        # dictionaries, not their residency or forward hooks. Avoid parent-module
        # ownership cycles when exposing both children to the shared loader.
        if self._bound:
            return
        self._loras_model_data = {}
        self._loras_model_shortcuts = {}
        for prefix, model in self._subloras.items():
            self._loras_model_data.update(model._loras_model_data)
            self._loras_model_shortcuts.update({prefix + "." + key: value for key, value in model._loras_model_shortcuts.items()})
        # Shared/fused LoRA factors can be aliases of fully qualified keys.
        for model in self._subloras.values():
            model._loras_model_shortcuts.update(self._loras_model_shortcuts)
        self._bound = True

    def preprocess_loras(self, model_type, state_dict):
        modules = dict(self.named_modules())
        mapper = LoraKeyMapper(modules)
        acoustic_mapper = LoraKeyMapper(dict(self.acoustic.named_modules()))
        mapped = {}
        for source, tensor in state_dict.items():
            key = source
            if key.startswith("text_encoders."):
                key = "ar." + key[len("text_encoders."):]
            elif key.startswith("text_encoder."):
                key = "ar." + key[len("text_encoder."):]
            elif key.startswith(("diffusion_model.", "transformer.")):
                key = "acoustic." + key.split(".", 1)[1]
            elif not key.startswith(("ar.", "acoustic.")):
                if re.match(r"(?:model\.)?layers\.\d+\.(?:self_attn|mlp)\.", key):
                    key = "ar." + key
                else:
                    # Preserve the existing acoustic-only namespace, including
                    # flattened Kohya names resolved against the acoustic model.
                    key = acoustic_mapper.map_key(key)
                    key = "acoustic." + key
            key = re.sub(r"^(ar|acoustic)\.layers\.", r"\1.model.layers.", key)
            if key.startswith("acoustic."):
                key = key.replace(".self_attn.", ".nar_self_attn.").replace(".mlp.", ".nar_mlp.")
            key = key.replace(".lora_down.weight", ".lora_A.weight").replace(".lora_up.weight", ".lora_B.weight")
            if key.endswith((".lora_A", ".lora_B")):
                key += ".weight"
            key = mapper.map_key(key)
            fused = re.match(r"(.+)\.(qkv_proj|gate_up_proj)\.(lora_[AB]\.weight|alpha)$", key)
            entries = {key: tensor}
            if fused:
                parent, projection, suffix = fused.groups()
                names = ("q_proj", "k_proj", "v_proj") if projection == "qkv_proj" else ("gate_proj", "up_proj")
                destinations = [parent + "." + name for name in names]
                if not all(name in modules for name in destinations):
                    raise ValueError(f"Unknown YuE2 fused LoRA target: {source}")
                # Splitting B's output rows and sharing A preserves B @ A,
                # including ComfyUI's block-diagonal packed adapters and alpha/rank.
                values = torch.split(tensor, [modules[name].out_features for name in destinations], dim=0) if suffix == "lora_B.weight" else [tensor] * len(names)
                entries = {name + "." + suffix: value for name, value in zip(destinations, values)}
            for destination, value in entries.items():
                if destination in mapped:
                    raise ValueError(f"YuE2 LoRA keys collide at {destination}")
                if destination.endswith((".weight", ".bias")) and ".lora_" not in destination:
                    raise ValueError(f"YuE2 LoRA contains full replacement weights ({source}); use an adapter with base-relative .diff/.diff_b tensors.")
                mapped[destination] = value
        return mapped

    def validate_ar_scaling(self, slists):
        if slists is None:
            return
        for data in self.ar._loras_model_data.values():
            for adapter in self.ar._loras_active_adapters:
                if adapter not in data:
                    continue
                values = slists["phase1"][int(adapter)]
                if isinstance(values, list) and any(value != values[0] for value in values):
                    raise ValueError("YuE2 AR LoRAs require a constant multiplier for composition and acoustic conditioning. Use a separate acoustic LoRA for denoising-step schedules.")


def ar_lora_signature(model):
    """Include Python capture decisions and adapter storage missed by the runner."""
    signature = []
    for adapter in getattr(model, "_loras_active_adapters", []):
        scaling = model._loras_scaling[adapter]
        if isinstance(scaling, list):
            if any(value != scaling[0] for value in scaling):
                raise ValueError("YuE2 AR LoRAs require a constant multiplier.")
            scaling = scaling[0]
        for data in model._loras_model_data.values():
            weights = data.get(adapter + "_GPU")
            if weights is not None:
                storage = []
                for value in weights:
                    if torch.is_tensor(value):
                        storage.append((value.data_ptr(), None if value.is_inference() else value._version))
                    else:
                        storage.append(str(value))
                signature.append((adapter, scaling, tuple(storage)))
    return tuple(signature)
