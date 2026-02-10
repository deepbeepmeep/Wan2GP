import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _get_parameter_safe(model: nn.Module, weight_name: str):
    """
    Try to get parameter from model, handling name mismatches.

    Some models have nested structure (e.g., Qwen3ForCausalLM has model.embed_tokens)
    but weight files may have flat names (embed_tokens.weight).
    """
    # Try direct access first
    try:
        return model.get_parameter(weight_name)
    except AttributeError:
        pass

    # Try with 'model.' prefix (for nested model structure)
    try:
        prefixed_name = f"model.{weight_name}"
        return model.get_parameter(prefixed_name)
    except AttributeError:
        pass

    # Try removing 'model.' prefix
    if weight_name.startswith("model."):
        try:
            unprefixed_name = weight_name[6:]  # Remove 'model.' prefix
            return model.get_parameter(unprefixed_name)
        except AttributeError:
            pass

    return None


def _get_submodule_safe(model: nn.Module, module_name: str):
    try:
        return model.get_submodule(module_name)
    except Exception:
        pass
    try:
        return model.get_submodule(f"model.{module_name}")
    except Exception:
        pass
    if module_name.startswith("model."):
        try:
            return model.get_submodule(module_name[6:])
        except Exception:
            pass
    return None


def _list_safetensor_files(path: str) -> list[str]:
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return [path]
    return glob(os.path.join(path, "*.safetensors"))


class WeightStore:
    def __init__(self, path: str, mode: str = "lazy"):
        self.path = path
        self.mode = (mode or "lazy").lower()
        self.files = _list_safetensor_files(path)
        if not self.files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")
        self._file_handles = {}
        self._weight_to_file = {}
        self._pinned_weights = {}
        self.is_quanto_int8 = False
        for file in self.files:
            with safe_open(file, "pt", "cpu") as f:
                for key in f.keys():
                    self._weight_to_file[key] = file
                    if key.endswith(".weight._data"):
                        self.is_quanto_int8 = True
        if self.mode == "pinned":
            self._preload_pinned()

    def _get_handle(self, file_path: str):
        handle = self._file_handles.get(file_path)
        if handle is None:
            handle = safe_open(file_path, "pt", "cpu")
            self._file_handles[file_path] = handle
        return handle

    def _preload_pinned(self):
        for key, file in self._weight_to_file.items():
            handle = self._get_handle(file)
            tensor = handle.get_tensor(key)
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
            self._pinned_weights[key] = tensor

    def get_tensor(self, key: str) -> torch.Tensor:
        if self.mode == "pinned":
            return self._pinned_weights[key]
        handle = self._get_handle(self._weight_to_file[key])
        return handle.get_tensor(key)


def load_model(model: nn.Module, path: str, weight_store: WeightStore | None = None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    if weight_store is None:
        safetensor_files = _list_safetensor_files(path)
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")
        for file in safetensor_files:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    tensor = f.get_tensor(weight_name)
                    _apply_weight(model, packed_modules_mapping, weight_name, tensor)
        _finalize_quantized_modules(model)
        return

    for weight_name in weight_store._weight_to_file.keys():
        tensor = weight_store.get_tensor(weight_name)
        _apply_weight(model, packed_modules_mapping, weight_name, tensor)
    _finalize_quantized_modules(model)


def _apply_weight(model: nn.Module, packed_modules_mapping, weight_name: str, tensor: torch.Tensor):
    quant_suffix = None
    if weight_name.endswith(".weight._data"):
        quant_suffix = "qdata"
    elif weight_name.endswith(".weight._scale"):
        quant_suffix = "qscale"
    elif weight_name.endswith(".input_scale"):
        quant_suffix = "input_scale"
    elif weight_name.endswith(".output_scale"):
        quant_suffix = "output_scale"

    for k in packed_modules_mapping:
        if k in weight_name:
            v, shard_id = packed_modules_mapping[k]
            mapped_name = weight_name.replace(k, v)
            if quant_suffix is not None:
                module_name = mapped_name
                if quant_suffix == "qdata":
                    module_name = mapped_name[: -len(".weight._data")]
                elif quant_suffix == "qscale":
                    module_name = mapped_name[: -len(".weight._scale")]
                elif quant_suffix == "input_scale":
                    module_name = mapped_name[: -len(".input_scale")]
                elif quant_suffix == "output_scale":
                    module_name = mapped_name[: -len(".output_scale")]
                module = _get_submodule_safe(model, module_name)
                if module is None:
                    print(f"[loader] Warning: Module not found: {module_name}")
                    return
                if quant_suffix == "qdata":
                    loader_fn = getattr(module, "quant_weight_data_loader", None)
                elif quant_suffix == "qscale":
                    loader_fn = getattr(module, "quant_weight_scale_loader", None)
                elif quant_suffix == "input_scale":
                    loader_fn = getattr(module, "quant_input_scale_loader", None)
                else:
                    loader_fn = getattr(module, "quant_output_scale_loader", None)
                if loader_fn is None:
                    print(f"[loader] Warning: Quant loader not found on module: {module_name}")
                    return
                if quant_suffix in ("qdata", "qscale"):
                    loader_fn(tensor, shard_id)
                else:
                    loader_fn(tensor)
                return
            param_name = mapped_name
            param = _get_parameter_safe(model, param_name)
            if param is None:
                print(f"[loader] Warning: Parameter not found: {param_name}")
                return
            weight_loader = getattr(param, "weight_loader")
            weight_loader(param, tensor, shard_id)
            return
    if quant_suffix is not None:
        if quant_suffix == "qdata":
            module_name = weight_name[: -len(".weight._data")]
        elif quant_suffix == "qscale":
            module_name = weight_name[: -len(".weight._scale")]
        elif quant_suffix == "input_scale":
            module_name = weight_name[: -len(".input_scale")]
        else:
            module_name = weight_name[: -len(".output_scale")]
        module = _get_submodule_safe(model, module_name)
        if module is None:
            print(f"[loader] Warning: Module not found: {module_name}")
            return
        if quant_suffix == "qdata":
            loader_fn = getattr(module, "quant_weight_data_loader", None)
        elif quant_suffix == "qscale":
            loader_fn = getattr(module, "quant_weight_scale_loader", None)
        elif quant_suffix == "input_scale":
            loader_fn = getattr(module, "quant_input_scale_loader", None)
        else:
            loader_fn = getattr(module, "quant_output_scale_loader", None)
        if loader_fn is None:
            print(f"[loader] Warning: Quant loader not found on module: {module_name}")
            return
        loader_fn(tensor)
        return
    param = _get_parameter_safe(model, weight_name)
    if param is None:
        print(f"[loader] Warning: Parameter not found: {weight_name}")
        return
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, tensor)


def _finalize_quantized_modules(model: nn.Module):
    for module in model.modules():
        finalize = getattr(module, "finalize_quantized", None)
        if callable(finalize):
            finalize()
