from __future__ import annotations

from .configuration_acestep_v15 import AceStepConfig
from .modeling_acestep_v15_turbo import AceStepConditionGenerationModel as _AceStepHFModel


class AceStepConditionGenerationModel(_AceStepHFModel):
    @staticmethod
    def _split_lora_key(lora_key):
        if lora_key.endswith(".alpha"):
            return lora_key[: -len(".alpha")], ".alpha"
        if lora_key.endswith(".diff"):
            return lora_key[: -len(".diff")], ".diff"
        if lora_key.endswith(".diff_b"):
            return lora_key[: -len(".diff_b")], ".diff_b"
        pos = lora_key.rfind(".lora_")
        if pos > 0:
            return lora_key[:pos], lora_key[pos:]
        return None, ""

    def preprocess_loras(self, model_type, sd):
        if not sd:
            return sd
        module_names = getattr(self, "_lora_module_names", None)
        if module_names is None:
            module_names = {name for name, _ in self.named_modules()}
            self._lora_module_names = module_names

        new_sd = {}
        for key, value in sd.items():
            key = key.replace(".lora.", ".lora_").replace(".default.weight", ".weight")
            for prefix in ("base_model.model.", "base_model.", "model.", "diffusion_model.", "transformer."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            module_name, suffix = self._split_lora_key(key)
            if not module_name:
                continue
            candidates = [module_name]
            if module_name.startswith("layers."):
                candidates.append(f"decoder.{module_name}")
            elif not module_name.startswith("decoder."):
                candidates.append(f"decoder.{module_name}")
            if module_name.startswith("decoder.model."):
                candidates.append(f"decoder.{module_name[len('decoder.model.'):]}")
            resolved = next((name for name in candidates if name in module_names), None)
            if resolved is not None:
                new_sd[f"transformer.{resolved}{suffix}"] = value
        return new_sd if len(new_sd) > 0 else sd

    @classmethod
    def from_config(cls, config):
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        else:
            config = dict(config)
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        return cls(AceStepConfig(**config))
