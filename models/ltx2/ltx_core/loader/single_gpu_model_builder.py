import logging
from dataclasses import dataclass, field, replace
from typing import Generic

import torch

from .module_ops import ModuleOps
from .primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from .registry import DummyRegistry, Registry
from .sd_ops import SDOps
from .sft_loader import SafetensorsModelStateDictLoader
from ..model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry, repr=False)
    shared_state_dict: dict | None = field(default=None, repr=False)
    shared_quantization_map: dict | None = field(default=None, repr=False)
    shared_config: dict | None = field(default=None, repr=False)
    ignore_missing_keys: bool = False
    copy_shared_state_dict: bool = False
    consume_shared_state_dict: bool = False

    def lora(self, lora_path: str, strength: float = 1.0, sd_ops: SDOps | None = None) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def model_config(self) -> dict:
        if self.shared_config is not None:
            return self.shared_config
        first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
        return self.model_loader.metadata(first_shard_path)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        with torch.device("meta"):
            model = self.model_class_configurator.from_config(config)
        for module_op in module_ops:
            if module_op.matcher(model):
                model = module_op.mutator(model)
        return model

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        state_dict = registry.get(paths, sd_ops)
        if state_dict is None:
            state_dict = self.model_loader.load(paths, sd_ops=sd_ops, device=device)
            registry.add(paths, sd_ops=sd_ops, state_dict=state_dict)
        return state_dict

    def _filter_state_dict(self, state_dict: dict, sd_ops: SDOps | None) -> dict:
        if sd_ops is None:
            return dict(state_dict)
        filtered = {}
        if self.consume_shared_state_dict:
            for key in list(state_dict.keys()):
                expected_name = sd_ops.apply_to_key(key)
                if expected_name is None:
                    continue
                value = state_dict.pop(key)
                key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                for new_key, new_value in key_value_pairs:
                    filtered[new_key] = new_value
        else:
            for key, value in state_dict.items():
                expected_name = sd_ops.apply_to_key(key)
                if expected_name is None:
                    continue
                key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                for new_key, new_value in key_value_pairs:
                    filtered[new_key] = new_value
        return filtered

    def _filter_quantization_map(self, quantization_map: dict | None, sd_ops: SDOps | None) -> dict | None:
        if not quantization_map:
            return None
        if sd_ops is None:
            return dict(quantization_map)
        filtered = {}
        if self.consume_shared_state_dict:
            for key in list(quantization_map.keys()):
                expected_name = sd_ops.apply_to_key(key)
                if expected_name is None:
                    continue
                value = quantization_map.pop(key)
                if expected_name.endswith(".weight"):
                    expected_name = expected_name[: -len(".weight")]
                filtered[expected_name] = value
        else:
            for key, value in quantization_map.items():
                expected_name = sd_ops.apply_to_key(key)
                if expected_name is None:
                    continue
                if expected_name.endswith(".weight"):
                    expected_name = expected_name[: -len(".weight")]
                filtered[expected_name] = value
        return filtered or None

    def _return_model(self, meta_model: ModelType, device: torch.device) -> ModelType:
        uninitialized_params = [name for name, param in meta_model.named_parameters() if str(param.device) == "meta"]
        uninitialized_buffers = [name for name, buffer in meta_model.named_buffers() if str(buffer.device) == "meta"]
        if uninitialized_params or uninitialized_buffers:
            logger.warning(f"Uninitialized parameters or buffers: {uninitialized_params + uninitialized_buffers}")
            return meta_model
        retval = meta_model.to(device)
        return retval

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> ModelType:
        device = torch.device("cuda") if device is None else device
        config = self.model_config()
        meta_model = self.meta_model(config, self.module_ops)
        from mmgp import offload as mmgp_offload

        if self.shared_state_dict is not None:
            sd = self._filter_state_dict(self.shared_state_dict, self.model_sd_ops)
            quantization_map = self._filter_quantization_map(self.shared_quantization_map, self.model_sd_ops)

            if self.copy_shared_state_dict:
                sd = {key: value.clone() if torch.is_tensor(value) else value for key, value in sd.items()}
            if len(sd):

                mmgp_offload.load_model_data(
                    meta_model,
                    [(sd, quantization_map)],
                    default_dtype=dtype or torch.bfloat16,
                    ignore_missing_keys=self.ignore_missing_keys,
                )
            return self._return_model(meta_model, device)

        def preprocess_sd(sd):
            new_sd = {}
            prefixes = ["vae", "audio_vae", "vocoder", "text_embedding_projection", "diffusion_model"] #, ""
            for k,v in sd.items():
                if k.startswith("model"):
                    k = k[len("model")+1:]
                for prefix in prefixes:
                    if k.startswith(prefix):
                        k = k[len(prefix)+1:]
                        break
                new_sd[k] = v
            return new_sd
        
        model_paths = self.model_path if isinstance(self.model_path, tuple) else [self.model_path]
        mmgp_offload.load_model_data(
            meta_model,
            model_paths,
            default_dtype=dtype or torch.bfloat16,
            preprocess_sd=preprocess_sd,
            ignore_missing_keys=self.ignore_missing_keys,
        )
        return self._return_model(meta_model, device)


        model_state_dict = self.load_sd(model_paths, sd_ops=self.model_sd_ops, registry=self.registry, device=device)

        sd = model_state_dict.sd

        quantization_map = {}
        post_load_hooks = []
        from mmgp import offload as mmgp_offload
        from mmgp.quant_router import apply_pre_quantization, detect_and_convert

        conv_result = detect_and_convert(sd, default_dtype=dtype or torch.bfloat16, verboseLevel=0)
        sd = conv_result.get("state_dict", sd)
        quantization_map = conv_result.get("quant_map", {}) or {}
        if quantization_map:
            quantization_map, post_load_hooks = apply_pre_quantization(
                meta_model,
                sd,
                quantization_map,
                default_dtype=dtype,
                verboseLevel=0,
            )
            if quantization_map:
                mmgp_offload._requantize(meta_model, sd, quantization_map, default_dtype=dtype)

        if dtype is not None and not quantization_map:
            sd = {key: value.to(dtype=dtype) for key, value in sd.items()}

        meta_model.load_state_dict(sd, strict=False, assign=True)
        model = self._return_model(meta_model, device)
        if post_load_hooks:
            for hook in post_load_hooks:
                try:
                    hook(model)
                except Exception as exc:
                    logger.warning("Post-load hook skipped: %s", exc)
        return model
