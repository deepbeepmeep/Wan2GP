import ast
import json
import os

import torch
from torch.utils import _pytree as pytree

from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes

try:
    from torch._subclasses.fake_tensor import FakeTensor as _TorchFakeTensor
except Exception:  # pragma: no cover
    _TorchFakeTensor = ()

try:
    from bitsandbytes.autograd._functions import matmul_4bit as _bnb_matmul_4bit
    from bitsandbytes.functional import dequantize_4bit as _bnb_dequantize_4bit
    from bitsandbytes.functional import QuantState as _BNBQuantState
except Exception:  # pragma: no cover
    _bnb_matmul_4bit = None
    _bnb_dequantize_4bit = None
    _BNBQuantState = None


HANDLER_NAME = "bnb_nf4"
HANDLER_PRIORITY = 3

_BNB_NF4_QTYPE_NAME = "nf4"
if _BNB_NF4_QTYPE_NAME not in _quanto_qtypes:
    _quanto_qtypes[_BNB_NF4_QTYPE_NAME] = _quanto_qtype(
        _BNB_NF4_QTYPE_NAME,
        is_floating_point=True,
        bits=4,
        dtype=torch.uint8,
        qmin=-1.0,
        qmax=1.0,
    )
_BNB_NF4_QTYPE = _quanto_qtypes[_BNB_NF4_QTYPE_NAME]

_BNB_NF4_SPLIT_FIELDS = {
    "weight": 0,
    "bias": 0,
    "weight.absmax": 0,
    "weight.quant_state.bitsandbytes__nf4": 0,
}
_BNB_NF4_SHARE_FIELDS = ("weight.quant_map",)

_BNB_NF4_KERNEL_LOGGED = False
_BNB_NF4_KERNEL_FAILED_LOGGED = False
_BNB_NF4_FALLBACK_LOGGED = False
_BNB_NF4_DEQUANT_FALLBACK_LOGGED = False


def _is_fake_tensor(tensor):
    return isinstance(tensor, _TorchFakeTensor)


def _note_fallback():
    global _BNB_NF4_FALLBACK_LOGGED
    if not _BNB_NF4_FALLBACK_LOGGED:
        print("BNB NF4: using dequantized linear fallback.")
        _BNB_NF4_FALLBACK_LOGGED = True


def _note_bnb_dequant_fallback():
    global _BNB_NF4_DEQUANT_FALLBACK_LOGGED
    if not _BNB_NF4_DEQUANT_FALLBACK_LOGGED:
        print("BNB NF4: using bitsandbytes dequantized linear fallback.")
        _BNB_NF4_DEQUANT_FALLBACK_LOGGED = True


def _note_kernel():
    global _BNB_NF4_KERNEL_LOGGED
    if not _BNB_NF4_KERNEL_LOGGED:
        print("BNB NF4: using bitsandbytes 4-bit kernels.")
        _BNB_NF4_KERNEL_LOGGED = True


def _note_kernel_failed(exc):
    global _BNB_NF4_KERNEL_FAILED_LOGGED
    if not _BNB_NF4_KERNEL_FAILED_LOGGED:
        print(f"BNB NF4: bitsandbytes kernel unavailable ({type(exc).__name__}: {exc}); using fallback.")
        _BNB_NF4_KERNEL_FAILED_LOGGED = True


def _state_from_tensor(tensor):
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return {}
    data = tensor.detach().cpu().to(torch.uint8).flatten().tolist()
    try:
        return json.loads(bytes(data).decode("utf-8"))
    except Exception:
        return {}


def _state_to_tensor(state, like):
    encoded = json.dumps(state, separators=(",", ": ")).encode("utf-8")
    device = like.device if torch.is_tensor(like) else torch.device("cpu")
    return torch.tensor(list(encoded), dtype=torch.uint8, device=device)


def _state_shape(state):
    shape = state.get("shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return None


def _ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def _nf4_context_shape(context):
    qstate = context.get("field_tensors", {}).get("weight.quant_state.bitsandbytes__nf4")
    state = _state_from_tensor(qstate)
    shape = _state_shape(state)
    if shape is not None:
        return state, shape
    total = int(context.get("total", 0))
    weight = context.get("field_tensors", {}).get("weight")
    if torch.is_tensor(weight) and total > 0:
        packed = int(weight.numel())
        in_features = packed * 2 // total
        return state, (total, in_features)
    return state, None


def _split_nf4_weight(src, *, dim, split_sizes, context):
    if src is None or not torch.is_tensor(src):
        return None
    _, shape = _nf4_context_shape(context)
    if shape is None:
        return None
    _, in_features = shape
    if in_features % 2 != 0:
        return None
    packed_per_row = in_features // 2
    packed_sizes = [int(size) * packed_per_row for size in split_sizes]
    if src.numel() != sum(packed_sizes):
        return None
    flat = src.reshape(-1)
    return [chunk.reshape(-1, 1) for chunk in torch.split(flat, packed_sizes, dim=0)]


def _split_nf4_absmax(src, *, dim, split_sizes, context):
    if src is None or not torch.is_tensor(src):
        return None
    state, shape = _nf4_context_shape(context)
    if shape is None:
        return None
    _, in_features = shape
    blocksize = int(state.get("blocksize", 64) or 64)
    blocks_per_row = _ceil_div(in_features, blocksize)
    absmax_sizes = [int(size) * blocks_per_row for size in split_sizes]
    if src.numel() != sum(absmax_sizes):
        return None
    return list(torch.split(src.reshape(-1), absmax_sizes, dim=0))


def _split_nf4_quant_state(src, *, dim, split_sizes, context):
    if src is None or not torch.is_tensor(src):
        return None
    state, shape = _nf4_context_shape(context)
    if shape is None:
        return None
    _, in_features = shape
    chunks = []
    for size in split_sizes:
        split_state = dict(state)
        split_state["shape"] = [int(size), int(in_features)]
        chunks.append(_state_to_tensor(split_state, src))
    return chunks


def split_fused_weights(state_dict, fused_split_map, quantization_map=None, allowed_bases=None, default_dtype=None, verboseLevel=1):
    from mmgp import offload

    return offload.sd_split_linear(
        state_dict,
        fused_split_map,
        split_fields=dict(_BNB_NF4_SPLIT_FIELDS),
        share_fields=_BNB_NF4_SHARE_FIELDS,
        split_handlers={
            "weight": _split_nf4_weight,
            "weight.absmax": _split_nf4_absmax,
            "weight.quant_state.bitsandbytes__nf4": _split_nf4_quant_state,
        },
        verboseLevel=verboseLevel,
        allowed_bases=allowed_bases,
        return_split_bases=True,
    )


def _collect_nf4_specs(state_dict):
    specs = []
    suffix = ".weight.quant_state.bitsandbytes__nf4"
    for key, tensor in state_dict.items():
        if not key.endswith(suffix):
            continue
        base = key[:-len(suffix)]
        weight_key = base + ".weight"
        absmax_key = base + ".weight.absmax"
        quant_map_key = base + ".weight.quant_map"
        weight = state_dict.get(weight_key)
        absmax = state_dict.get(absmax_key)
        quant_map = state_dict.get(quant_map_key)
        if getattr(weight, "dtype", None) != torch.uint8:
            continue
        if getattr(absmax, "dtype", None) is None or getattr(quant_map, "dtype", None) is None:
            continue
        specs.append({"name": base, "weight": weight})
    return specs


def detect(state_dict, verboseLevel=1):
    specs = _collect_nf4_specs(state_dict)
    if not specs:
        return {"matched": False, "kind": "none", "details": {}}
    names = [spec["name"] for spec in specs[:8]]
    return {"matched": True, "kind": "bnb_nf4", "details": {"count": len(specs), "names": names}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    specs = _collect_nf4_specs(state_dict)
    if not specs:
        return {"state_dict": state_dict, "quant_map": {}}
    quant_map = {}
    for spec in specs:
        qcfg = {"weights": _BNB_NF4_QTYPE.name, "activations": "none"}
        quant_map[spec["name"]] = qcfg
        quant_map[spec["name"] + ".weight"] = qcfg
    return {"state_dict": state_dict, "quant_map": quant_map}


def detect_quantization_label_from_filename(filename, verboseLevel=0):
    if filename and "nf4" in os.path.basename(str(filename)).lower():
        return "NF4"
    return ""


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    return quantization_map, []


def _dtype_from_state(value, default=torch.bfloat16):
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        name = value.removeprefix("torch.")
        return getattr(torch, name, default)
    return default


def _bnb_nf4_qfallback(callable, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(BNBNF4WeightTensor, lambda x: x.dequantize(), (args, kwargs or {}))
    return callable(*args, **kwargs)


def _dequantize_bnb_nf4_weight(weight_u8, absmax, quant_map, shape, blocksize, dtype, device):
    if weight_u8.device != device:
        weight_u8 = weight_u8.to(device)
    if absmax.device != device:
        absmax = absmax.to(device)
    if quant_map.device != device:
        quant_map = quant_map.to(device)

    out_features, in_features = int(shape[0]), int(shape[1])
    total_values = out_features * in_features
    packed = weight_u8.reshape(-1).to(torch.uint8)
    codes = torch.empty(packed.numel() * 2, dtype=torch.long, device=device)
    codes[0::2] = (packed >> 4).to(torch.long)
    codes[1::2] = (packed & 0x0F).to(torch.long)
    compute_dtype = torch.float32 if dtype != torch.float32 else dtype
    values = quant_map.to(device=device, dtype=compute_dtype).index_select(0, codes[:total_values])
    values = values.reshape(-1, blocksize)
    scales = absmax.reshape(-1).to(device=device, dtype=compute_dtype).reshape(-1, 1)
    values.mul_(scales)
    return values.reshape(out_features, in_features).to(dtype)


def _bnb_nf4_linear(input, weight, bias=None):
    if torch.is_tensor(input) and _is_fake_tensor(input):
        return input.new_empty((*input.shape[:-1], weight.size(0)))
    if torch.is_tensor(input) and _bnb_matmul_4bit is not None:
        bias_arg = bias
        if bias_arg is not None and torch.is_tensor(bias_arg):
            if bias_arg.device != input.device or bias_arg.dtype != input.dtype:
                bias_arg = bias_arg.to(device=input.device, dtype=input.dtype)
        quant_state = weight.bnb_quant_state(input.device)
        packed_weight = weight._data if weight._data.device == input.device else weight._data.to(input.device)
        if input.device.type == "cuda" and input.numel() == input.shape[-1] and input.shape[-1] % weight._blocksize == 0:
            try:
                _note_kernel()
                return _bnb_matmul_4bit(input, packed_weight.t(), quant_state=quant_state, bias=bias_arg)
            except Exception as exc:
                _note_kernel_failed(exc)
        if _bnb_dequantize_4bit is not None:
            _note_bnb_dequant_fallback()
            qweight = _bnb_dequantize_4bit(packed_weight, quant_state).to(input.dtype)
            return torch.nn.functional.linear(input, qweight, bias=bias_arg)
    _note_fallback()
    if torch.is_tensor(input):
        qweight = weight.dequantize(dtype=input.dtype, device=input.device)
        bias_arg = bias
        if bias_arg is not None and torch.is_tensor(bias_arg):
            if bias_arg.device != input.device or bias_arg.dtype != input.dtype:
                bias_arg = bias_arg.to(device=input.device, dtype=input.dtype)
        return torch.nn.functional.linear(input, qweight, bias=bias_arg)
    return torch.nn.functional.linear(input, weight.dequantize(), bias=bias)


class BNBNF4WeightTensor(QTensor):
    @staticmethod
    def create(weight_u8, absmax, quant_map, quant_state, size, stride, dtype, device=None, requires_grad=False):
        state = _state_from_tensor(quant_state)
        shape = _state_shape(state) or tuple(size)
        blocksize = int(state.get("blocksize", 64) or 64)
        qdtype = _dtype_from_state(state.get("dtype", dtype), dtype)
        if dtype is None:
            dtype = qdtype
        device = weight_u8.device if device is None else device
        if weight_u8.device != device:
            weight_u8 = weight_u8.to(device)
        if absmax.device != device:
            absmax = absmax.to(device)
        if quant_map.device != device:
            quant_map = quant_map.to(device)
        if quant_state.device != device:
            quant_state = quant_state.to(device)
        return BNBNF4WeightTensor(
            qtype=_BNB_NF4_QTYPE,
            axis=0,
            size=size,
            stride=stride,
            weight_u8=weight_u8,
            absmax=absmax,
            quant_map=quant_map,
            quant_state=quant_state,
            original_shape=tuple(shape),
            blocksize=blocksize,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    @staticmethod
    def __new__(cls, qtype, axis, size, stride, weight_u8, absmax, quant_map, quant_state, original_shape, blocksize, dtype, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            dtype=dtype,
            device=weight_u8.device,
            requires_grad=requires_grad,
        )

    def __init__(self, qtype, axis, size, stride, weight_u8, absmax, quant_map, quant_state, original_shape, blocksize, dtype, requires_grad=False):
        super().__init__(qtype, axis)
        self._data = weight_u8
        self._absmax = absmax
        self._quant_map = quant_map
        self._quant_state = quant_state
        self._original_shape = tuple(original_shape)
        self._blocksize = int(blocksize)

    def __repr__(self):
        return f"BNBNF4WeightTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"

    __str__ = __repr__

    def dequantize(self, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return _dequantize_bnb_nf4_weight(
            weight_u8=self._data,
            absmax=self._absmax,
            quant_map=self._quant_map,
            shape=self._original_shape,
            blocksize=self._blocksize,
            dtype=dtype,
            device=device,
        )

    def bnb_quant_state(self, device):
        if _BNBQuantState is None:
            raise RuntimeError("bitsandbytes is not available")
        device = torch.device(device)
        absmax = self._absmax if self._absmax.device == device else self._absmax.to(device)
        quant_map = self._quant_map if self._quant_map.device == device else self._quant_map.to(device)
        state = _state_from_tensor(self._quant_state)
        qdtype = _dtype_from_state(state.get("dtype", self.dtype), self.dtype)
        return _BNBQuantState(
            absmax=absmax,
            shape=torch.Size(self._original_shape),
            code=quant_map,
            blocksize=self._blocksize,
            quant_type="nf4",
            dtype=qdtype,
        )

    def get_quantized_subtensors(self):
        return [
            ("weight_u8", self._data),
            ("absmax", self._absmax),
            ("quant_map", self._quant_map),
            ("quant_state", self._quant_state),
        ]

    def set_quantized_subtensors(self, sub_tensors):
        sub_map = sub_tensors if isinstance(sub_tensors, dict) else {name: tensor for name, tensor in sub_tensors}
        if sub_map.get("weight_u8") is not None:
            self._data = sub_map["weight_u8"]
        if sub_map.get("absmax") is not None:
            self._absmax = sub_map["absmax"]
        if sub_map.get("quant_map") is not None:
            self._quant_map = sub_map["quant_map"]
        if sub_map.get("quant_state") is not None:
            self._quant_state = sub_map["quant_state"]
            state = _state_from_tensor(self._quant_state)
            self._original_shape = _state_shape(state) or self._original_shape
            self._blocksize = int(state.get("blocksize", self._blocksize) or self._blocksize)

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_absmax", "_quant_map", "_quant_state"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "dtype": str(self.dtype),
            "original_shape": str(list(self._original_shape)),
            "blocksize": str(self._blocksize),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        qtype = _quanto_qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        dtype_name = meta.get("dtype", "torch.bfloat16").removeprefix("torch.")
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        original_shape = ast.literal_eval(meta.get("original_shape", str(size)))
        blocksize = int(meta.get("blocksize", "64"))
        return BNBNF4WeightTensor(
            qtype=qtype,
            axis=axis,
            size=size,
            stride=stride,
            weight_u8=inner_tensors["_data"],
            absmax=inner_tensors["_absmax"],
            quant_map=inner_tensors["_quant_map"],
            quant_state=inner_tensors["_quant_state"],
            original_shape=original_shape,
            blocksize=blocksize,
            dtype=dtype,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            input = args[0] if len(args) > 0 else kwargs.get("input", None)
            weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
            bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
            if isinstance(weight, BNBNF4WeightTensor):
                return _bnb_nf4_linear(input, weight, bias=bias)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        op = op.overloadpacket
        if op is torch.ops.aten.linear:
            input = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None
            if isinstance(weight, BNBNF4WeightTensor):
                return _bnb_nf4_linear(input, weight, bias=bias)
        if op is torch.ops.aten.detach:
            t = args[0]
            return BNBNF4WeightTensor.create(
                weight_u8=op(t._data),
                absmax=op(t._absmax),
                quant_map=op(t._quant_map),
                quant_state=op(t._quant_state),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=t.device,
                requires_grad=t.requires_grad,
            )
        if op in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.pop("dtype", t.dtype) if kwargs else t.dtype
            device = kwargs.pop("device", t.device) if kwargs else t.device
            if dtype != t.dtype:
                return t.dequantize(dtype=dtype, device=device)
            return BNBNF4WeightTensor.create(
                weight_u8=op(t._data, device=device, **(kwargs or {})),
                absmax=op(t._absmax, device=device, **(kwargs or {})),
                quant_map=op(t._quant_map, device=device, **(kwargs or {})),
                quant_state=op(t._quant_state, device=device, **(kwargs or {})),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=device,
                requires_grad=t.requires_grad,
            )
        return _bnb_nf4_qfallback(op, *args, **(kwargs or {}))


class QLinearBNBNF4(QModuleMixin, torch.nn.Linear):
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
        self._bnb_nf4_default_dtype = dtype

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

    def set_default_dtype(self, dtype):
        self._bnb_nf4_default_dtype = dtype

    @property
    def qweight(self):
        if self.weight_qtype == _BNB_NF4_QTYPE:
            return self.weight
        return super().qweight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.qweight, bias=self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.weight_qtype != _BNB_NF4_QTYPE:
            return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        weight_key = prefix + "weight"
        absmax_key = prefix + "weight.absmax"
        quant_map_key = prefix + "weight.quant_map"
        quant_state_key = prefix + "weight.quant_state.bitsandbytes__nf4"
        bias_key = prefix + "bias"
        input_scale_key = prefix + "input_scale"
        output_scale_key = prefix + "output_scale"

        weight_u8 = state_dict.pop(weight_key, None)
        absmax = state_dict.pop(absmax_key, None)
        quant_map = state_dict.pop(quant_map_key, None)
        quant_state = state_dict.pop(quant_state_key, None)
        bias = state_dict.pop(bias_key, None)
        input_scale = state_dict.pop(input_scale_key, None)
        output_scale = state_dict.pop(output_scale_key, None)

        if weight_u8 is None:
            missing_keys.append(weight_key)
        if absmax is None:
            missing_keys.append(absmax_key)
        if quant_map is None:
            missing_keys.append(quant_map_key)
        if quant_state is None:
            missing_keys.append(quant_state_key)

        target_dtype = self._bnb_nf4_default_dtype or self.weight.dtype
        if weight_u8 is not None and absmax is not None and quant_map is not None and quant_state is not None:
            qweight = BNBNF4WeightTensor.create(
                weight_u8=weight_u8,
                absmax=absmax,
                quant_map=quant_map,
                quant_state=quant_state,
                size=self.weight.size(),
                stride=self.weight.stride(),
                dtype=target_dtype,
                device=weight_u8.device,
                requires_grad=False,
            )
            self.weight = torch.nn.Parameter(qweight, requires_grad=False)

        if bias is not None:
            if target_dtype is not None and bias.dtype != target_dtype:
                bias = bias.to(target_dtype)
            self.bias = torch.nn.Parameter(bias)

        scale_device = weight_u8.device if torch.is_tensor(weight_u8) else torch.device("cpu")
        if input_scale is not None:
            self.input_scale = input_scale.to(scale_device)
        elif not hasattr(self, "input_scale") or self.input_scale.is_meta:
            self.input_scale = torch.ones((), dtype=torch.float32, device=scale_device)
        if output_scale is not None:
            self.output_scale = output_scale.to(scale_device)
        elif not hasattr(self, "output_scale") or self.output_scale.is_meta:
            self.output_scale = torch.ones((), dtype=torch.float32, device=scale_device)
        return
