import ast
import json
import math
import os

import torch
from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes
from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor


HANDLER_NAME = "int8_convrot"
HANDLER_PRIORITY = 11

_QTYPE_NAME = "qint8_convrot"
if _QTYPE_NAME not in _quanto_qtypes:
    _quanto_qtypes[_QTYPE_NAME] = _quanto_qtype(
        _QTYPE_NAME,
        is_floating_point=False,
        bits=8,
        dtype=torch.int8,
        qmin=-128,
        qmax=127,
    )
_QINT8_CONVROT_QTYPE = _quanto_qtypes[_QTYPE_NAME]

_HADAMARD_CACHE = {}
_DTYPE_DEBUG_COUNT = 0
_FUSED_SPLIT_MARKER_SUFFIX = ".qweight"

try:
    from torch._subclasses.fake_tensor import is_fake as _torch_is_fake_tensor
except Exception:  # pragma: no cover
    _torch_is_fake_tensor = None


def _is_fake_tensor(tensor):
    return bool(torch.is_tensor(tensor) and _torch_is_fake_tensor is not None and _torch_is_fake_tensor(tensor))


def _normalize_compute_dtype(dtype):
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        return dtype
    return torch.bfloat16


class Int8ConvRotWeightTensor(WeightQBytesTensor):
    @staticmethod
    def create(qtype, axis, size, stride, data, scale, activation_qtype=None, requires_grad=False, compute_dtype=None):
        return Int8ConvRotWeightTensor(qtype, axis, size, stride, data, scale, activation_qtype, requires_grad, compute_dtype)

    @staticmethod
    def __new__(cls, qtype, axis, size, stride, data, scale, activation_qtype=None, requires_grad=False, compute_dtype=None):
        assert data.device == scale.device
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            dtype=_normalize_compute_dtype(compute_dtype),
            device=data.device,
            requires_grad=requires_grad,
        )

    def __init__(self, qtype, axis, size, stride, data, scale, activation_qtype=None, requires_grad=False, compute_dtype=None):
        super().__init__(qtype, axis, size, stride, data, scale, activation_qtype, requires_grad)
        self._compute_dtype = _normalize_compute_dtype(compute_dtype)

    def optimize(self):
        return self

    def weight_qbytes_tensor(self):
        return WeightQBytesTensor(self.qtype, self.axis, self.size(), self.stride(), self._data, self._scale, self.activation_qtype, self.requires_grad)

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "activation_qtype": "none" if self.activation_qtype is None else self.activation_qtype.name,
            "compute_dtype": str(self.dtype),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        qtype = _quanto_qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        activation_qtype = None if meta["activation_qtype"] == "none" else _quanto_qtypes[meta["activation_qtype"]]
        dtype_name = meta.get("compute_dtype", "torch.bfloat16").replace("torch.", "")
        return Int8ConvRotWeightTensor(qtype, axis, size, stride, inner_tensors["_data"], inner_tensors["_scale"], activation_qtype, compute_dtype=getattr(torch, dtype_name, torch.bfloat16))

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        packet = op.overloadpacket
        kwargs = kwargs or {}
        if packet is torch.ops.aten.detach:
            t = args[0]
            return Int8ConvRotWeightTensor.create(
                t.qtype,
                t.axis,
                t.size(),
                t.stride(),
                packet(t._data),
                packet(t._scale),
                activation_qtype=t.activation_qtype,
                requires_grad=t.requires_grad,
                compute_dtype=t.dtype,
            )
        if packet in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.pop("dtype", t.dtype)
            device = kwargs.pop("device", t.device)
            if dtype != t.dtype:
                raise ValueError("The dtype of a weights Tensor cannot be changed")
            out_data = packet(t._data, device=device, **kwargs)
            out_scale = packet(t._scale, device=device, **kwargs)
            return Int8ConvRotWeightTensor.create(
                t.qtype,
                t.axis,
                t.size(),
                t.stride(),
                out_data,
                out_scale,
                activation_qtype=t.activation_qtype,
                requires_grad=t.requires_grad,
                compute_dtype=t.dtype,
            )
        return WeightQBytesTensor.__torch_dispatch__.__func__(WeightQBytesTensor, op, types, args, kwargs)


def _wrap_int8_convrot_weight(weight, compute_dtype):
    if not isinstance(weight, WeightQBytesTensor) or getattr(weight, "qtype", None) != _QINT8_CONVROT_QTYPE:
        return weight
    compute_dtype = _normalize_compute_dtype(compute_dtype)
    if isinstance(weight, Int8ConvRotWeightTensor) and weight.dtype == compute_dtype:
        return weight
    return Int8ConvRotWeightTensor.create(
        weight.qtype,
        weight.axis,
        weight.size(),
        weight.stride(),
        weight._data,
        weight._scale,
        activation_qtype=weight.activation_qtype,
        requires_grad=weight.requires_grad,
        compute_dtype=compute_dtype,
    )


def _decode_json_tensor(tensor):
    if not torch.is_tensor(tensor):
        return {}
    try:
        data = tensor.detach().cpu().to(torch.uint8).reshape(-1).tolist()
        return json.loads(bytes(data).decode("utf-8"))
    except Exception:
        return {}


def _convrot_group_size(config):
    if not isinstance(config, dict) or not bool(config.get("convrot", False)):
        return 0
    return int(config.get("convrot_groupsize", 256) or 256)


def _regular_hadamard(size, device, dtype):
    size = int(size)
    device = torch.device(device)
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    h = h.mul(size ** -0.5)
    _HADAMARD_CACHE[key] = h
    return h


def _rotate_activation(input, group_size):
    group_size = int(group_size or 0)
    if group_size <= 0:
        return input
    features = int(input.shape[-1])
    if features % group_size != 0:
        raise RuntimeError(f"ConvRot input features {features} not divisible by group size {group_size}")
    h = _regular_hadamard(group_size, input.device, input.dtype)
    shape = input.shape
    return torch.matmul(input.reshape(-1, features // group_size, group_size), h).reshape(shape)


def _dtype_debug_enabled():
    return os.environ.get("WAN2GP_INT8_CONVROT_DTYPE_DEBUG", "").lower() in ("1", "true", "yes", "on")


def _debug_forward_dtype(module, input, rotated, qweight):
    global _DTYPE_DEBUG_COUNT
    if not _dtype_debug_enabled() or _DTYPE_DEBUG_COUNT >= 80:
        return
    _DTYPE_DEBUG_COUNT += 1
    name = getattr(module, "name", None) or getattr(module, "_mm_name", None) or module.__class__.__name__
    scale = getattr(qweight, "_scale", None)
    data = getattr(qweight, "_data", None)
    bias = getattr(module, "bias", None)
    print(
        "[WAN2GP][INT8][dtype] "
        f"{name}: input={getattr(input, 'dtype', None)} rotated={getattr(rotated, 'dtype', None)} "
        f"qweight={getattr(qweight, 'dtype', None)} data={getattr(data, 'dtype', None)} "
        f"scale={getattr(scale, 'dtype', None)} bias={getattr(bias, 'dtype', None)} "
        f"default={getattr(module, '_convrot_default_dtype', None) or getattr(module, '_router_default_dtype', None)}"
    )


def _collect_specs(state_dict):
    specs = []
    for key, tensor in state_dict.items():
        if not key.endswith(".weight") or getattr(tensor, "dtype", None) != torch.int8:
            continue
        base = key[:-7]
        scale = state_dict.get(base + ".weight_scale")
        quant_config = state_dict.get(base + ".comfy_quant")
        if not torch.is_tensor(scale) or not torch.is_tensor(quant_config):
            continue
        config = _decode_json_tensor(quant_config)
        if not isinstance(config, dict):
            continue
        specs.append({"name": base, "weight": tensor, "scale": scale, "config": config})
    return specs


def detect(state_dict, verboseLevel=1):
    specs = _collect_specs(state_dict)
    if not specs:
        return {"matched": False, "kind": "none", "details": {}}
    names = [spec["name"] for spec in specs[:8]]
    convrot_count = sum(1 for spec in specs if _convrot_group_size(spec["config"]) > 0)
    return {"matched": True, "kind": HANDLER_NAME, "details": {"count": len(specs), "convrot_count": convrot_count, "names": names}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    specs = _collect_specs(state_dict)
    if not specs:
        return {"state_dict": state_dict, "quant_map": {}}
    quant_map = {}
    for spec in specs:
        base = spec["name"]
        weight = state_dict.pop(base + ".weight")
        scale = state_dict.pop(base + ".weight_scale")
        state_dict.pop(base + ".comfy_quant", None)
        if scale.numel() == weight.shape[0]:
            scale = scale.reshape(weight.shape[0], 1)
        if scale.numel() != weight.shape[0]:
            raise RuntimeError(f"INT8 ConvRot weight scale for '{base}' has {scale.numel()} values, expected {weight.shape[0]}")
        state_dict[base + ".weight._data"] = weight
        state_dict[base + ".weight._scale"] = scale.to(torch.float32)
        state_dict[base + _FUSED_SPLIT_MARKER_SUFFIX] = torch.empty(0, dtype=torch.uint8, device=weight.device)
        state_dict[base + ".input_scale"] = torch.ones((), dtype=torch.float32, device=weight.device)
        state_dict[base + ".output_scale"] = torch.ones((), dtype=torch.float32, device=weight.device)
        state_dict[base + ".convrot_group_size"] = torch.tensor(_convrot_group_size(spec["config"]), dtype=torch.int32, device=weight.device)
        qcfg = {"weights": _QTYPE_NAME, "activations": "none"}
        quant_map[base] = qcfg
        quant_map[base + ".weight"] = qcfg
    return {"state_dict": state_dict, "quant_map": quant_map}


def _split_scale(src, *, dim, split_sizes, context):
    if src is None or not torch.is_tensor(src):
        return None
    if src.numel() == 1:
        return [src] * len(split_sizes)
    total = sum(split_sizes)
    if src.ndim > dim and src.size(dim) == total:
        return torch.split(src, split_sizes, dim=dim)
    if src.ndim == 2 and src.shape[0] == total and src.shape[1] == 1:
        return torch.split(src, split_sizes, dim=0)
    return None


def split_fused_weights(state_dict, fused_split_map, quantization_map=None, allowed_bases=None, default_dtype=None, verboseLevel=1):
    from mmgp import offload

    state_dict, split_bases = offload.sd_split_linear(
        state_dict,
        fused_split_map,
        split_fields={"weight._data": 0, "weight._scale": 0, "bias": 0},
        share_fields=("input_scale", "output_scale", "convrot_group_size"),
        split_handlers={"weight._scale": _split_scale},
        verboseLevel=verboseLevel,
        allowed_bases=allowed_bases,
        return_split_bases=True,
    )
    for base in split_bases:
        state_dict.pop(base + _FUSED_SPLIT_MARKER_SUFFIX, None)
    return state_dict, split_bases


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    for key in [key for key, value in state_dict.items() if key.endswith(_FUSED_SPLIT_MARKER_SUFFIX) and torch.is_tensor(value) and value.numel() == 0]:
        state_dict.pop(key, None)
    return quantization_map or {}, []


def detect_quantization_label_from_filename(filename, verboseLevel=0):
    if filename and "convrot" in os.path.basename(str(filename)).lower() and "int8" in os.path.basename(str(filename)).lower():
        return "INT8 ConvRot"
    return ""


class QLinearInt8ConvRot(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(cls, module, weights, activations=None, optimizer=None, device=None):
        if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point:
            weight_dtype = module.weight.dtype
        elif torch.is_tensor(getattr(module, "bias", None)) and module.bias.dtype.is_floating_point:
            weight_dtype = module.bias.dtype
        else:
            weight_dtype = torch.float16
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=device,
            dtype=weight_dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        weights=None,
        activations=None,
        optimizer=None,
        quantize_input=True,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=quantize_input,
        )
        self._convrot_group_size = 0
        self._convrot_default_dtype = dtype
        self._mm_requires_native_linear_forward = True

    def set_default_dtype(self, dtype):
        self._convrot_default_dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qweight = self.qweight
        if _is_fake_tensor(input):
            return input.new_empty((*input.shape[:-1], qweight.shape[0]))
        original_input = input
        if self.weight_qtype == _QINT8_CONVROT_QTYPE:
            input = _rotate_activation(input, self._convrot_group_size)
        _debug_forward_dtype(self, original_input, input, qweight)
        return torch.nn.functional.linear(input, qweight, bias=self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.weight_qtype != _QINT8_CONVROT_QTYPE:
            return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        group_key = prefix + "convrot_group_size"
        group_size = state_dict.pop(group_key, None)
        if group_size is None:
            missing_keys.append(group_key)
        else:
            self._convrot_group_size = int(group_size.item() if torch.is_tensor(group_size) else group_size)
        result = super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        compute_dtype = self._convrot_default_dtype
        if compute_dtype is None and torch.is_tensor(getattr(self, "bias", None)) and self.bias.dtype.is_floating_point:
            compute_dtype = self.bias.dtype
        wrapped = _wrap_int8_convrot_weight(self.weight, compute_dtype)
        if wrapped is not self.weight:
            self.weight = torch.nn.Parameter(wrapped, requires_grad=self.weight.requires_grad)
        return result
