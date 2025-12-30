import ast
import torch
from torch.utils import _pytree as pytree

from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes

try:
    from lightx2v_kernel.gemm import scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm
    _KERNEL_AVAILABLE = True
except Exception:
    scaled_nvfp4_quant = None
    cutlass_scaled_nvfp4_mm = None
    _KERNEL_AVAILABLE = False

_NVFP4_QTYPE_NAME = "nvfp4"
if _NVFP4_QTYPE_NAME not in _quanto_qtypes:
    _quanto_qtypes[_NVFP4_QTYPE_NAME] = _quanto_qtype(
        _NVFP4_QTYPE_NAME,
        is_floating_point=True,
        bits=4,
        dtype=torch.uint8,
        qmin=-6.0,
        qmax=6.0,
    )
_NVFP4_QTYPE = _quanto_qtypes[_NVFP4_QTYPE_NAME]

def _supports_nvfp4_kernel(device):
    if not _KERNEL_AVAILABLE:
        return False
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 12


def _is_float8_dtype(dtype):
    return "float8" in str(dtype).lower() or "f8" in str(dtype).lower()

_FP4_LUT_BASE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_FP4_LUT_CACHE = {}
_FP4_BYTE_LUT_CACHE = {}


def _get_fp4_lut(device, dtype):
    key = (device, dtype)
    lut = _FP4_LUT_CACHE.get(key)
    if lut is None:
        lut = _FP4_LUT_BASE.to(device=device, dtype=dtype)
        _FP4_LUT_CACHE[key] = lut
    return lut


def _get_fp4_byte_lut(device, dtype):
    key = (device, dtype)
    byte_lut = _FP4_BYTE_LUT_CACHE.get(key)
    if byte_lut is None:
        lut16 = _get_fp4_lut(device, dtype)
        b = torch.arange(256, device=device, dtype=torch.int32)
        byte_lut = torch.empty((256, 2), device=device, dtype=dtype)
        byte_lut[:, 0] = lut16[b & 0x0F]
        byte_lut[:, 1] = lut16[b >> 4]
        _FP4_BYTE_LUT_CACHE[key] = byte_lut
    return byte_lut


def _deswizzle_nvfp4_scale(scale, in_features, block_size=16, dtype=None):
    k_groups = in_features // block_size
    if scale.shape[1] < k_groups:
        raise RuntimeError(
            f"NVFP4 scale shape mismatch: expected at least {k_groups} groups, got {scale.shape[1]}"
        )
    if scale.shape[1] > k_groups:
        scale = scale[:, :k_groups]

    m, _ = scale.shape
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (in_features + f - 1) // f
    tmp = scale if dtype is None else scale.to(dtype)
    tmp = tmp.reshape(1, m_tiles, k_tiles, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    out = tmp.reshape(m_tiles * 128, k_tiles * 4)
    return out[:m, :k_groups]


def _dequantize_nvfp4_weight(
    weight_u8,
    weight_scale,
    input_global_scale,
    alpha,
    dtype,
    device,
    block_size=16,
):
    if weight_u8.device != device:
        weight_u8 = weight_u8.to(device)
    scale = weight_scale if weight_scale.device == device else weight_scale.to(device)
    if alpha.device != device:
        alpha = alpha.to(device)
    if input_global_scale.device != device:
        input_global_scale = input_global_scale.to(device)

    m, k_bytes = weight_u8.shape
    byte_lut = _get_fp4_byte_lut(device, dtype)
    idx = weight_u8.to(torch.int32)
    out = byte_lut[idx].reshape(m, k_bytes * 2)

    scale = _deswizzle_nvfp4_scale(scale, out.shape[1], block_size=block_size, dtype=dtype)
    out = out.view(out.shape[0], scale.shape[1], block_size)
    out.mul_(scale.unsqueeze(-1))
    out = out.view(out.shape[0], -1)

    scale_factor = alpha.to(dtype) * input_global_scale.to(dtype)
    out.mul_(scale_factor)
    return out


def _collect_nvfp4_specs(state_dict):
    specs = []
    for key, tensor in state_dict.items():
        if not key.endswith(".weight"):
            continue
        if tensor.dtype != torch.uint8:
            continue
        base = key[:-7]
        scale_key = base + ".weight_scale"
        input_global_key = base + ".input_global_scale"
        alpha_key = base + ".alpha"
        if scale_key not in state_dict or input_global_key not in state_dict or alpha_key not in state_dict:
            continue
        if not _is_float8_dtype(state_dict[scale_key].dtype):
            continue
        specs.append(
            {
                "name": base,
                "weight": tensor,
                "weight_scale": state_dict[scale_key],
                "input_global_scale": state_dict[input_global_key],
                "alpha": state_dict[alpha_key],
                "bias": state_dict.get(base + ".bias", None),
            }
        )
    return specs


def detect_nvfp4_state_dict(state_dict):
    return len(_collect_nvfp4_specs(state_dict)) > 0


def describe_nvfp4_state_dict(state_dict, max_names=8):
    specs = _collect_nvfp4_specs(state_dict)
    names = [spec["name"] for spec in specs]
    return {"count": len(names), "names": names[:max_names]}


def convert_nvfp4_to_quanto(state_dict, default_dtype=None, verboseLevel=1):
    specs = _collect_nvfp4_specs(state_dict)
    if not specs:
        return {"state_dict": state_dict, "quant_map": {}}
    quant_map = {spec["name"]: {"weights": "nvfp4", "activations": "none"} for spec in specs}
    return {"state_dict": state_dict, "quant_map": quant_map}


def detect(state_dict, verboseLevel=1):
    matched = detect_nvfp4_state_dict(state_dict)
    details = describe_nvfp4_state_dict(state_dict) if matched else {}
    return {"matched": matched, "kind": "nvfp4" if matched else "none", "details": details}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    return convert_nvfp4_to_quanto(state_dict, default_dtype=default_dtype, verboseLevel=verboseLevel)


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    return quantization_map, []


def _nvfp4_qfallback(callable, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(NVFP4WeightTensor, lambda x: x.dequantize(), (args, kwargs or {}))
    return callable(*args, **kwargs)


class NVFP4WeightTensor(QTensor):
    @staticmethod
    def create(
        weight_u8,
        weight_scale,
        input_global_scale,
        alpha,
        size,
        stride,
        dtype,
        device=None,
        requires_grad=False,
    ):
        device = weight_u8.device if device is None else device
        if weight_u8.device != device:
            weight_u8 = weight_u8.to(device)
        if weight_scale.device != device:
            weight_scale = weight_scale.to(device)
        if input_global_scale.device != device:
            input_global_scale = input_global_scale.to(device)
        if alpha.device != device:
            alpha = alpha.to(device)
        return NVFP4WeightTensor(
            qtype=_NVFP4_QTYPE,
            axis=0,
            size=size,
            stride=stride,
            weight_u8=weight_u8,
            weight_scale=weight_scale,
            input_global_scale=input_global_scale,
            alpha=alpha,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    @staticmethod
    def __new__(
        cls,
        qtype,
        axis,
        size,
        stride,
        weight_u8,
        weight_scale,
        input_global_scale,
        alpha,
        dtype,
        requires_grad=False,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            dtype=dtype,
            device=weight_u8.device,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        qtype,
        axis,
        size,
        stride,
        weight_u8,
        weight_scale,
        input_global_scale,
        alpha,
        dtype,
        requires_grad=False,
    ):
        super().__init__(qtype, axis)
        self._data = weight_u8
        self._scale = weight_scale
        self._input_global_scale = input_global_scale
        self._alpha = alpha
        self._block_size = 16

    def dequantize(self, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return _dequantize_nvfp4_weight(
            weight_u8=self._data,
            weight_scale=self._scale,
            input_global_scale=self._input_global_scale,
            alpha=self._alpha,
            dtype=dtype,
            device=device,
            block_size=self._block_size,
        )

    def get_quantized_subtensors(self):
        return [
            ("weight_u8", self._data),
            ("weight_scale", self._scale),
            ("input_global_scale", self._input_global_scale),
            ("alpha", self._alpha),
        ]

    def set_quantized_subtensors(self, sub_tensors):
        if isinstance(sub_tensors, dict):
            sub_map = sub_tensors
        else:
            sub_map = {name: tensor for name, tensor in sub_tensors}
        data = sub_map.get("weight_u8", sub_map.get("data"))
        if data is not None:
            self._data = data
        if "weight_scale" in sub_map and sub_map["weight_scale"] is not None:
            self._scale = sub_map["weight_scale"]
        if "input_global_scale" in sub_map and sub_map["input_global_scale"] is not None:
            self._input_global_scale = sub_map["input_global_scale"]
        if "alpha" in sub_map and sub_map["alpha"] is not None:
            self._alpha = sub_map["alpha"]

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_input_global_scale", "_alpha"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "dtype": str(self.dtype),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        qtype = _quanto_qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        dtype_str = meta.get("dtype", "torch.float16")
        if dtype_str.startswith("torch."):
            dtype_name = dtype_str.split(".", 1)[1]
            dtype = getattr(torch, dtype_name, torch.float16)
        else:
            dtype = getattr(torch, dtype_str, torch.float16)
        return NVFP4WeightTensor(
            qtype=qtype,
            axis=axis,
            size=size,
            stride=stride,
            weight_u8=inner_tensors["_data"],
            weight_scale=inner_tensors["_scale"],
            input_global_scale=inner_tensors["_input_global_scale"],
            alpha=inner_tensors["_alpha"],
            dtype=dtype,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            input = args[0] if len(args) > 0 else kwargs.get("input", None)
            weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
            bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
            if isinstance(weight, NVFP4WeightTensor):
                if torch.is_tensor(input) and _supports_nvfp4_kernel(input.device):
                    x2d = input.reshape(-1, input.shape[-1])
                    if not x2d.is_floating_point():
                        x2d = x2d.to(torch.float16)
                    orig_dtype = x2d.dtype
                    input_quant, input_scale = scaled_nvfp4_quant(x2d, weight._input_global_scale)
                    if bias is not None and torch.is_tensor(bias) and bias.dtype != orig_dtype:
                        bias = bias.to(orig_dtype)
                    out = cutlass_scaled_nvfp4_mm(
                        input_quant,
                        weight._data,
                        input_scale,
                        weight._scale,
                        alpha=weight._alpha,
                        bias=bias,
                    )
                    if out.dtype != orig_dtype:
                        out = out.to(orig_dtype)
                    return out.reshape(*input.shape[:-1], weight.size(0))
                dtype = input.dtype if torch.is_tensor(input) else weight.dtype
                device = input.device if torch.is_tensor(input) else weight.device
                w = weight.dequantize(dtype=dtype, device=device)
                if bias is not None and torch.is_tensor(bias) and bias.dtype != dtype:
                    bias = bias.to(dtype)
                return torch.nn.functional.linear(input, w, bias)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        op = op.overloadpacket
        if op is torch.ops.aten.linear:
            input = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None
            if isinstance(weight, NVFP4WeightTensor):
                if torch.is_tensor(input) and _supports_nvfp4_kernel(input.device):
                    x2d = input.reshape(-1, input.shape[-1])
                    if not x2d.is_floating_point():
                        x2d = x2d.to(torch.float16)
                    orig_dtype = x2d.dtype
                    input_quant, input_scale = scaled_nvfp4_quant(x2d, weight._input_global_scale)
                    if bias is not None and torch.is_tensor(bias) and bias.dtype != orig_dtype:
                        bias = bias.to(orig_dtype)
                    out = cutlass_scaled_nvfp4_mm(
                        input_quant,
                        weight._data,
                        input_scale,
                        weight._scale,
                        alpha=weight._alpha,
                        bias=bias,
                    )
                    if out.dtype != orig_dtype:
                        out = out.to(orig_dtype)
                    return out.reshape(*input.shape[:-1], weight.size(0))
                dtype = input.dtype if torch.is_tensor(input) else weight.dtype
                device = input.device if torch.is_tensor(input) else weight.device
                w = weight.dequantize(dtype=dtype, device=device)
                if bias is not None and torch.is_tensor(bias) and bias.dtype != dtype:
                    bias = bias.to(dtype)
                return op(input, w, bias)
        if op is torch.ops.aten.detach:
            t = args[0]
            return NVFP4WeightTensor.create(
                weight_u8=op(t._data),
                weight_scale=op(t._scale),
                input_global_scale=op(t._input_global_scale),
                alpha=op(t._alpha),
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
            out_data = op(t._data, device=device, **(kwargs or {}))
            out_scale = op(t._scale, device=device, **(kwargs or {}))
            out_igs = op(t._input_global_scale, device=device, **(kwargs or {}))
            out_alpha = op(t._alpha, device=device, **(kwargs or {}))
            return NVFP4WeightTensor.create(
                weight_u8=out_data,
                weight_scale=out_scale,
                input_global_scale=out_igs,
                alpha=out_alpha,
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=device,
                requires_grad=t.requires_grad,
            )
        return _nvfp4_qfallback(op, *args, **(kwargs or {}))


class QLinearNVFP4(QModuleMixin, torch.nn.Linear):
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
        self._nvfp4_default_dtype = dtype

    @classmethod
    def qcreate(
        cls,
        module,
        weights,
        activations=None,
        optimizer=None,
        device=None,
    ):
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
        self._nvfp4_default_dtype = dtype

    @property
    def qweight(self):
        if self.weight_qtype == _NVFP4_QTYPE:
            return self.weight
        return super().qweight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.qweight, bias=self.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.weight_qtype != _NVFP4_QTYPE:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        weight_key = prefix + "weight"
        scale_key = prefix + "weight_scale"
        igs_key = prefix + "input_global_scale"
        alpha_key = prefix + "alpha"
        bias_key = prefix + "bias"
        input_scale_key = prefix + "input_scale"
        output_scale_key = prefix + "output_scale"

        weight_u8 = state_dict.pop(weight_key, None)
        weight_scale = state_dict.pop(scale_key, None)
        input_global_scale = state_dict.pop(igs_key, None)
        alpha = state_dict.pop(alpha_key, None)
        bias = state_dict.pop(bias_key, None)
        input_scale = state_dict.pop(input_scale_key, None)
        output_scale = state_dict.pop(output_scale_key, None)

        if weight_u8 is None:
            missing_keys.append(weight_key)
        if weight_scale is None:
            missing_keys.append(scale_key)
        if input_global_scale is None:
            missing_keys.append(igs_key)
        if alpha is None:
            missing_keys.append(alpha_key)

        target_dtype = self._nvfp4_default_dtype or self.weight.dtype
        if weight_u8 is not None and weight_scale is not None and input_global_scale is not None and alpha is not None:
            nvfp4_weight = NVFP4WeightTensor.create(
                weight_u8=weight_u8,
                weight_scale=weight_scale,
                input_global_scale=input_global_scale,
                alpha=alpha,
                size=self.weight.size(),
                stride=self.weight.stride(),
                dtype=target_dtype,
                device=weight_u8.device,
                requires_grad=False,
            )
            self.weight = torch.nn.Parameter(nvfp4_weight, requires_grad=False)

        if bias is not None:
            if target_dtype is not None and bias.dtype != target_dtype:
                bias = bias.to(target_dtype)
            self.bias = torch.nn.Parameter(bias)

        if torch.is_tensor(weight_u8):
            scale_device = weight_u8.device
        elif torch.is_tensor(self.weight):
            scale_device = self.weight.device
        elif torch.is_tensor(bias):
            scale_device = bias.device
        else:
            scale_device = torch.device("cpu")

        if input_scale is not None:
            self.input_scale = input_scale.to(scale_device)
        else:
            if not hasattr(self, "input_scale") or self.input_scale.is_meta:
                scale_dtype = self.input_scale.dtype if hasattr(self, "input_scale") else torch.float32
                self.input_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        if output_scale is not None:
            self.output_scale = output_scale.to(scale_device)
        else:
            if not hasattr(self, "output_scale") or self.output_scale.is_meta:
                scale_dtype = self.output_scale.dtype if hasattr(self, "output_scale") else torch.float32
                self.output_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        return
