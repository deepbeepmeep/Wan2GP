import gc
import os
from collections import OrderedDict
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    if param.is_meta:
        raise RuntimeError("default_weight_loader cannot copy into meta parameter directly.")
    if loaded_weight.device != param.device or loaded_weight.dtype != param.dtype:
        loaded_weight = loaded_weight.to(
            device=param.device,
            dtype=param.dtype,
            non_blocking=True,
        )
    param.data.copy_(loaded_weight)


def _coerce_loaded_weight_to_param(param: nn.Parameter, loaded_weight: torch.Tensor) -> torch.Tensor:
    target_device = param.device if param.device.type != "meta" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loaded_weight.device != target_device:
        loaded_weight = loaded_weight.to(device=target_device, non_blocking=True)
    if loaded_weight.dtype != param.dtype:
        loaded_weight = loaded_weight.to(dtype=param.dtype, non_blocking=True)
    return loaded_weight


def _restore_param_attributes(new_param: nn.Parameter, src_param: nn.Parameter) -> None:
    for attr_name, attr_value in src_param.__dict__.items():
        setattr(new_param, attr_name, attr_value)


def _replace_meta_parameter(
    parent: nn.Module,
    param_name: str,
    loaded_weight: torch.Tensor,
    src_param: nn.Parameter,
) -> nn.Parameter:
    new_param = nn.Parameter(
        loaded_weight,
        requires_grad=src_param.requires_grad,
    )
    parent._parameters[param_name] = new_param
    _restore_param_attributes(new_param, src_param)
    return new_param


def _replace_tied_meta_parameter(
    model: nn.Module,
    old_param: nn.Parameter,
    loaded_weight: torch.Tensor,
) -> nn.Parameter:
    new_param = nn.Parameter(
        loaded_weight,
        requires_grad=old_param.requires_grad,
    )
    _restore_param_attributes(new_param, old_param)
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is old_param:
                module._parameters[name] = new_param
    return new_param


def _should_replace_param_globally(model: nn.Module, param: nn.Parameter) -> bool:
    tied_ids = getattr(model, "_vllm_tied_param_ids", None)
    return bool(tied_ids) and id(param) in tied_ids


def _candidate_weight_names(weight_name: str):
    candidates = [weight_name]
    if weight_name.startswith("model."):
        candidates.append(weight_name[6:])
    else:
        candidates.append(f"model.{weight_name}")
    return candidates


def _get_parameter_target(model: nn.Module, weight_name: str):
    for candidate in _candidate_weight_names(weight_name):
        if "." in candidate:
            parent_name, param_name = candidate.rsplit(".", 1)
        else:
            parent_name, param_name = "", candidate
        try:
            parent = model if parent_name == "" else model.get_submodule(parent_name)
        except Exception:
            continue
        param = parent._parameters.get(param_name)
        if param is not None and isinstance(param, nn.Parameter):
            return parent, param_name, param
    return None


def _get_submodule_safe(model: nn.Module, module_name: str):
    for candidate in _candidate_weight_names(module_name):
        if candidate == "":
            return model
        try:
            return model.get_submodule(candidate)
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
        self._weight_to_file = {}
        self._file_to_weight_names = OrderedDict()
        self._pinned_weights = {}
        self.is_quanto_int8 = False
        for file in self.files:
            with safe_open(file, "pt", "cpu", writable_tensors=False) as f:
                for key in f.keys():
                    self._weight_to_file[key] = file
                    self._file_to_weight_names.setdefault(file, []).append(key)
                    if key.endswith(".weight._data"):
                        self.is_quanto_int8 = True
        if self.mode == "pinned":
            self._preload_pinned()

    def _preload_pinned(self):
        for file, keys in self._file_to_weight_names.items():
            with safe_open(file, "pt", "cpu", writable_tensors=False) as handle:
                for key in keys:
                    tensor = handle.get_tensor(key)
                    if tensor.device.type != "cpu":
                        tensor = tensor.cpu()
                    if not tensor.is_pinned():
                        tensor = tensor.pin_memory()
                    self._pinned_weights[key] = tensor

    def close(self):
        self._weight_to_file.clear()
        self._file_to_weight_names.clear()
        self._pinned_weights.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_tensor(self, key: str) -> torch.Tensor:
        if self.mode == "pinned":
            return self._pinned_weights[key]
        file = self._weight_to_file.get(key)
        if file is None:
            raise KeyError(key)
        with safe_open(file, "pt", "cpu", writable_tensors=False) as handle:
            return handle.get_tensor(key)


def load_model(
    model: nn.Module,
    path: str,
    weight_store: WeightStore | None = None,
    clone_loaded_tensors: bool = False,
):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    if weight_store is None:
        safetensor_files = _list_safetensor_files(path)
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")
        for file in safetensor_files:
            with safe_open(file, "pt", "cpu", writable_tensors=False) as f:
                for weight_name in f.keys():
                    tensor = f.get_tensor(weight_name)
                    _apply_weight(model, packed_modules_mapping, weight_name, tensor)
        _finalize_quantized_modules(model)
        return
    for file in weight_store.files:
        file_weights = weight_store._file_to_weight_names.get(file, [])
        if not file_weights:
            continue
        with safe_open(file, "pt", "cuda", writable_tensors = False, streaming = True) as f:
            file_weights = f.reorder_for_streaming(file_weights)
            for weight_name in file_weights:
                tensor = f.get_tensor(weight_name)
                if clone_loaded_tensors:
                    tensor = tensor.clone()
                try:
                    _apply_weight(model, packed_modules_mapping, weight_name, tensor)
                finally:
                    del tensor
            f = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    gc.collect()
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

    mapped_name = weight_name
    shard_id = None
    for src_name, mapped in packed_modules_mapping.items():
        if src_name in weight_name:
            mapped_target = mapped[0] if isinstance(mapped, tuple) else mapped
            mapped_name = weight_name.replace(src_name, mapped_target)
            shard_id = mapped[1] if isinstance(mapped, tuple) and len(mapped) > 1 else None
            break

    if quant_suffix is not None:
        if quant_suffix == "qdata":
            module_name = mapped_name[:-len(".weight._data")]
            loader_name = "quant_weight_data_loader"
        elif quant_suffix == "qscale":
            module_name = mapped_name[:-len(".weight._scale")]
            loader_name = "quant_weight_scale_loader"
        elif quant_suffix == "input_scale":
            module_name = mapped_name[:-len(".input_scale")]
            loader_name = "quant_input_scale_loader"
        else:
            module_name = mapped_name[:-len(".output_scale")]
            loader_name = "quant_output_scale_loader"
        module = _get_submodule_safe(model, module_name)
        if module is None:
            print(f"[loader] Warning: Module not found: {module_name}")
            return
        loader_fn = getattr(module, loader_name, None)
        if loader_fn is None:
            print(f"[loader] Warning: Quant loader not found on module: {module_name}")
            return
        if shard_id is not None and quant_suffix in ("qdata", "qscale"):
            loader_fn(tensor, shard_id)
        else:
            loader_fn(tensor)
        return

    target = _get_parameter_target(model, mapped_name)
    if target is None:
        print(f"[loader] Warning: Parameter not found: {mapped_name}")
        return
    parent, param_name, param = target
    weight_loader = getattr(param, "weight_loader", None)

    if shard_id is not None and weight_loader is not None:
        if param.is_meta:
            replacement_device = (
                param.device if param.device.type != "meta" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            replacement = torch.empty(param.shape, device=replacement_device, dtype=param.dtype)
            if _should_replace_param_globally(model, param):
                param = _replace_tied_meta_parameter(model, param, replacement)
            else:
                param = _replace_meta_parameter(parent, param_name, replacement, param)
        try:
            weight_loader(param, tensor, shard_id)
        except TypeError:
            weight_loader(param, tensor)
        return

    if param.is_meta:
        loaded_weight = _coerce_loaded_weight_to_param(param, tensor)
        if loaded_weight.shape != param.shape:
            raise RuntimeError(
                f"Weight shape mismatch for '{mapped_name}': expected {tuple(param.shape)}, got {tuple(loaded_weight.shape)}"
            )
        if _should_replace_param_globally(model, param):
            _replace_tied_meta_parameter(model, param, loaded_weight)
        else:
            _replace_meta_parameter(parent, param_name, loaded_weight, param)
        return
    if weight_loader is None:
        weight_loader = default_weight_loader
    weight_loader(param, tensor)


def _finalize_quantized_modules(model: nn.Module):
    for module in model.modules():
        finalize = getattr(module, "finalize_quantized", None)
        if callable(finalize):
            finalize()
