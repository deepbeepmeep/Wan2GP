import ast
import os
import torch
from torch.utils import _pytree as pytree

from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes

def _maybe_add_nvfp4_cu13_dll_dir():
    if os.name != "nt":
        return
    try:
        import nvidia.cu13
        dll_dir = os.path.join(nvidia.cu13.__path__[0], "bin", "x86_64")
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
    except Exception:
        pass

try:
    from comfy_kitchen.backends import cuda as _ck_cuda
    _ck_cuda_available = getattr(_ck_cuda, "_EXT_AVAILABLE", False)
except Exception:
    _ck_cuda = None
    _ck_cuda_available = False

try:
    _maybe_add_nvfp4_cu13_dll_dir()
    from lightx2v_kernel import gemm as _lx_gemm
    _lx_gemm_available = True
except Exception:
    _lx_gemm = None
    _lx_gemm_available = False

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
HANDLER_PRIORITY = 1

_NVFP4_LAYOUT_LEGACY = "legacy"
_NVFP4_LAYOUT_TENSORCORE = "tensorcore"

_NVFP4_BACKEND_AUTO = "auto"
_NVFP4_BACKEND_COMFY = "comfy"
_NVFP4_BACKEND_LIGHTX2V = "lightx2v"

_NVFP4_KERNEL_LOGGED = False
_NVFP4_FALLBACK_LOGGED = False
_NVFP4_LOAD_LOGGED = False
_NVFP4_KERNEL_AVAILABLE = False
_NVFP4_KERNEL_CHECKED = False
_NVFP4_KERNEL_BACKEND = None
_NVFP4_ACT_SCALE_CACHE = {}

_NVFP4_SPLIT_FIELDS = {
    "weight": 0,
    "bias": 0,
    "weight_scale": 0,
    "weight_scale_2": 0,
    "input_scale": 0,
    "input_global_scale": 0,
    "alpha": 0,
    "input_absmax": 0,
    "weight_global_scale": 0,
    "output_scale": 0,
}

_NVFP4_BACKEND = os.environ.get("WGP_NVFP4_BACKEND", _NVFP4_BACKEND_AUTO).strip().lower()
_NVFP4_BACKEND = _NVFP4_BACKEND_LIGHTX2V

def _normalize_nvfp4_backend(name):
    if name is None:
        return _NVFP4_BACKEND_AUTO
    norm = str(name).strip().lower()
    if norm in ("", "auto", "default"):
        return _NVFP4_BACKEND_AUTO
    if norm in ("comfy", "comfy-kitchen", "comfy_kitchen", "ck"):
        return _NVFP4_BACKEND_COMFY
    if norm in ("lightx2v", "lightx2v_kernel", "lightx2v-kernel", "lx"):
        return _NVFP4_BACKEND_LIGHTX2V
    if norm in ("off", "none", "fallback", "disable", "disabled"):
        return "fallback"
    return norm


def _split_or_share_nvfp4_scale(src, *, dim, split_sizes, context):
    if src is None or not torch.is_tensor(src):
        return None
    total = sum(split_sizes)
    if src.numel() == 1:
        return [src] * len(split_sizes)
    if src.dim() > dim and src.size(dim) == total:
        return torch.split(src, split_sizes, dim=dim)
    if src.ndim > 1 and src.size(1) == total:
        return torch.split(src, split_sizes, dim=1)
    return [src] * len(split_sizes)


def split_fused_weights(state_dict, fused_split_map, quantization_map=None, allowed_bases=None, default_dtype=None, verboseLevel=1):
    from mmgp import offload
    return offload.sd_split_linear(
        state_dict,
        fused_split_map,
        split_fields=dict(_NVFP4_SPLIT_FIELDS),
        split_handlers={
            "weight_scale": _split_or_share_nvfp4_scale,
            "weight_scale_2": _split_or_share_nvfp4_scale,
            "input_scale": _split_or_share_nvfp4_scale,
            "input_global_scale": _split_or_share_nvfp4_scale,
            "alpha": _split_or_share_nvfp4_scale,
            "input_absmax": _split_or_share_nvfp4_scale,
            "weight_global_scale": _split_or_share_nvfp4_scale,
            "output_scale": _split_or_share_nvfp4_scale,
        },
        verboseLevel=verboseLevel,
        allowed_bases=allowed_bases,
        return_split_bases=True,
    )


_NVFP4_BACKEND = _normalize_nvfp4_backend(_NVFP4_BACKEND)


def _nvfp4_backend_candidates():
    if _NVFP4_BACKEND == _NVFP4_BACKEND_AUTO:
        return [_NVFP4_BACKEND_COMFY, _NVFP4_BACKEND_LIGHTX2V]
    if _NVFP4_BACKEND in (_NVFP4_BACKEND_COMFY, _NVFP4_BACKEND_LIGHTX2V):
        return [_NVFP4_BACKEND]
    return []


def _nvfp4_backend_label(backend):
    if backend == _NVFP4_BACKEND_LIGHTX2V:
        return "lightx2v"
    if backend == _NVFP4_BACKEND_COMFY:
        return "comfy-kitchen"
    return backend


def _nvfp4_lightx2v_device_ok(device):
    force = os.environ.get("WGP_NVFP4_LIGHTX2V_FORCE", "").strip().lower()
    if force in ("1", "true", "yes", "y"):
        return True
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return False
    return props.major >= 12


def set_nvfp4_backend(name):
    global _NVFP4_BACKEND, _NVFP4_KERNEL_CHECKED, _NVFP4_KERNEL_AVAILABLE, _NVFP4_KERNEL_BACKEND
    global _NVFP4_KERNEL_LOGGED, _NVFP4_LOAD_LOGGED
    _NVFP4_BACKEND = _normalize_nvfp4_backend(name)
    _NVFP4_KERNEL_CHECKED = False
    _NVFP4_KERNEL_AVAILABLE = False
    _NVFP4_KERNEL_BACKEND = None
    _NVFP4_KERNEL_LOGGED = False
    _NVFP4_LOAD_LOGGED = False
    _init_nvfp4_kernel_support()


def _nvfp4_note_kernel():
    global _NVFP4_KERNEL_LOGGED
    if not _NVFP4_KERNEL_LOGGED:
        label = _nvfp4_backend_label(_NVFP4_KERNEL_BACKEND) if _NVFP4_KERNEL_BACKEND else "CUDA"
        print(f"NVFP4: using {label} kernel")
        _NVFP4_KERNEL_LOGGED = True


def _nvfp4_note_fallback():
    global _NVFP4_FALLBACK_LOGGED
    global _NVFP4_KERNEL_LOGGED
    if not _NVFP4_FALLBACK_LOGGED:
        if _NVFP4_KERNEL_LOGGED:
            print("NVFP4: linear fallback needed on some weights")
        else:
            print("NVFP4: linear fallback")
        _NVFP4_FALLBACK_LOGGED = True

def _nvfp4_note_reset():
    global _NVFP4_FALLBACK_LOGGED
    global _NVFP4_KERNEL_LOGGED
    global _NVFP4_LOAD_LOGGED
    _NVFP4_KERNEL_LOGGED = False
    _NVFP4_FALLBACK_LOGGED = False
    _NVFP4_LOAD_LOGGED = False

def _nvfp4_note_load_backend():
    global _NVFP4_LOAD_LOGGED
    if _NVFP4_LOAD_LOGGED:
        return
    _NVFP4_LOAD_LOGGED = True
    if _NVFP4_KERNEL_AVAILABLE:
        label = _nvfp4_backend_label(_NVFP4_KERNEL_BACKEND) if _NVFP4_KERNEL_BACKEND else "unknown"
        print(f"NVFP4: kernels available ({label}); optimized path will be used when compatible.")
    else:
        print("NVFP4: kernels unavailable; using fallback.")


def _check_nvfp4_kernel_support(device, backend):
    if device.type != "cuda":
        return False
    if backend == _NVFP4_BACKEND_COMFY:
        if not _ck_cuda_available:
            return False
        if not hasattr(_ck_cuda, "scaled_mm_nvfp4"):
            return False
        if not hasattr(_ck_cuda, "quantize_nvfp4"):
            return False
        if not (hasattr(torch.ops, "comfy_kitchen") and hasattr(torch.ops.comfy_kitchen, "scaled_mm_nvfp4")):
            return False
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor) >= (10, 0)
    if backend == _NVFP4_BACKEND_LIGHTX2V:
        if not _lx_gemm_available:
            return False
        if not _nvfp4_lightx2v_device_ok(device):
            return False
        if not (hasattr(torch.ops, "lightx2v_kernel") and hasattr(torch.ops.lightx2v_kernel, "cutlass_scaled_nvfp4_mm_sm120")):
            return False
        if not hasattr(torch.ops.lightx2v_kernel, "scaled_nvfp4_quant_sm120"):
            return False
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor) >= (12, 0)
    return False


def _init_nvfp4_kernel_support():
    global _NVFP4_KERNEL_AVAILABLE, _NVFP4_KERNEL_CHECKED, _NVFP4_KERNEL_BACKEND
    if _NVFP4_KERNEL_CHECKED:
        return
    _NVFP4_KERNEL_CHECKED = True
    _NVFP4_KERNEL_AVAILABLE = False
    _NVFP4_KERNEL_BACKEND = None
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    for backend in _nvfp4_backend_candidates():
        try:
            if _check_nvfp4_kernel_support(device, backend):
                _NVFP4_KERNEL_AVAILABLE = True
                _NVFP4_KERNEL_BACKEND = backend
                break
        except Exception:
            continue


def _supports_nvfp4_kernel(device):
    if device.type != "cuda":
        return False
    if not _NVFP4_KERNEL_CHECKED:
        _init_nvfp4_kernel_support()
    return _NVFP4_KERNEL_AVAILABLE


_init_nvfp4_kernel_support()


def _nvfp4_layout(weight):
    return getattr(weight, "_layout", _NVFP4_LAYOUT_LEGACY)


def _nvfp4_can_use_kernel(input, weight):
    if not torch.is_tensor(input):
        return False
    if not getattr(weight, "_allow_kernel", True):
        return False
    if not _supports_nvfp4_kernel(input.device):
        return False
    backend = _NVFP4_KERNEL_BACKEND
    if backend is None:
        return False
    layout = _nvfp4_layout(weight)
    if backend == _NVFP4_BACKEND_LIGHTX2V:
        if input.shape[-1] % 32 != 0:
            return False
        if weight.size(0) % 32 != 0:
            return False
    else:
        if layout == _NVFP4_LAYOUT_LEGACY:
            if input.shape[-1] % 64 != 0:
                return False
        else:
            if input.shape[-1] % 16 != 0:
                return False
        if weight.size(0) % 8 != 0:
            return False
    if weight._data.shape[1] * 2 != input.shape[-1]:
        return False
    if weight._block_size != 16:
        return False
    if not torch.is_tensor(weight._input_global_scale) or not torch.is_tensor(weight._alpha):
        return False
    if getattr(weight._input_global_scale, "is_meta", False):
        return False
    try:
        if not torch.isfinite(weight._input_global_scale).all():
            return False
    except Exception:
        return False
    return True


def _nvfp4_get_act_scale(device):
    act_scale = _NVFP4_ACT_SCALE_CACHE.get(device)
    if act_scale is None:
        act_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        _NVFP4_ACT_SCALE_CACHE[device] = act_scale
    return act_scale


def _nvfp4_swap_nibbles(tensor):
    return ((tensor & 0x0F) << 4) | ((tensor & 0xF0) >> 4)


def _nvfp4_linear_cuda_comfy(input, weight, bias=None):
    _nvfp4_note_kernel()
    x2d = input.reshape(-1, input.shape[-1])
    if not x2d.is_floating_point():
        x2d = x2d.to(torch.float16)
    orig_dtype = x2d.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        x2d = x2d.to(torch.float16)
        out_dtype = torch.float16
    else:
        out_dtype = orig_dtype
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    weight_fp4 = weight._data
    weight_scale = weight._scale
    input_scale = weight._input_global_scale
    alpha = weight._alpha
    layout = _nvfp4_layout(weight)
    device = x2d.device
    if weight_fp4.device != device:
        weight_fp4 = weight_fp4.to(device)
    if weight_scale.device != device:
        weight_scale = weight_scale.to(device)
    if input_scale.device != device:
        input_scale = input_scale.to(device)
    if alpha.device != device:
        alpha = alpha.to(device)
    if bias is not None and torch.is_tensor(bias) and bias.dtype != out_dtype:
        bias = bias.to(out_dtype)
    orig_rows = x2d.shape[0]
    pad_16x = (orig_rows % 16) != 0
    if layout == _NVFP4_LAYOUT_TENSORCORE:
        input_scale = input_scale.to(torch.float32)
        tensor_scale = alpha.to(torch.float32)
        qx, qx_scale = _ck_cuda.quantize_nvfp4(x2d, input_scale, 0.0, pad_16x)
        out = _ck_cuda.scaled_mm_nvfp4(
            qx,
            weight_fp4,
            tensor_scale_a=input_scale,
            tensor_scale_b=tensor_scale,
            block_scale_a=qx_scale,
            block_scale_b=weight_scale,
            bias=bias,
            out_dtype=out_dtype,
        )
    else:
        alpha = alpha * input_scale
        if alpha.dtype != torch.float32:
            alpha = alpha.to(torch.float32)
        act_scale = _nvfp4_get_act_scale(device)
        qx, qx_scale = _ck_cuda.quantize_nvfp4(x2d, act_scale, 0.0, pad_16x)
        weight_fp4 = _nvfp4_swap_nibbles(weight_fp4)
        out = _ck_cuda.scaled_mm_nvfp4(
            qx,
            weight_fp4,
            act_scale,
            input_scale,
            qx_scale,
            weight_scale,
            bias=bias,
            out_dtype=out_dtype,
            alpha=alpha,
        )
    if pad_16x:
        out = out[:orig_rows]
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out.reshape(*input.shape[:-1], weight.size(0))


def _nvfp4_linear_cuda_lightx2v(input, weight, bias=None):
    _nvfp4_note_kernel()
    x2d = input.reshape(-1, input.shape[-1])
    if not x2d.is_floating_point():
        x2d = x2d.to(torch.float16)
    orig_dtype = x2d.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        x2d = x2d.to(torch.float16)
        out_dtype = torch.float16
    else:
        out_dtype = orig_dtype
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    weight_fp4 = weight._data
    weight_scale = weight._scale
    input_scale = weight._input_global_scale
    alpha = weight._alpha
    layout = _nvfp4_layout(weight)
    device = x2d.device
    if weight_fp4.device != device:
        weight_fp4 = weight_fp4.to(device)
    if weight_scale.device != device:
        weight_scale = weight_scale.to(device)
    if not weight_fp4.is_contiguous():
        weight_fp4 = weight_fp4.contiguous()
    if not weight_scale.is_contiguous():
        weight_scale = weight_scale.contiguous()
    if input_scale.device != device:
        input_scale = input_scale.to(device)
    if alpha.device != device:
        alpha = alpha.to(device)
    if input_scale.dtype != torch.float32:
        input_scale = input_scale.to(torch.float32)
    if alpha.dtype != torch.float32:
        alpha = alpha.to(torch.float32)
    if bias is not None and torch.is_tensor(bias):
        if bias.dtype != torch.bfloat16:
            bias = bias.to(torch.bfloat16)
        if not bias.is_contiguous():
            bias = bias.contiguous()
    if layout == _NVFP4_LAYOUT_TENSORCORE:
        quant_scale = torch.reciprocal(torch.clamp(input_scale, min=1e-8))
        alpha = alpha * input_scale
    else:
        quant_scale = input_scale
    qx, qx_scale = _lx_gemm.scaled_nvfp4_quant(x2d, quant_scale)
    if layout == _NVFP4_LAYOUT_TENSORCORE:
        qx = _nvfp4_swap_nibbles(qx)
    if not qx.is_contiguous():
        qx = qx.contiguous()
    if not qx_scale.is_contiguous():
        qx_scale = qx_scale.contiguous()
    out = _lx_gemm.cutlass_scaled_nvfp4_mm(
        qx,
        weight_fp4,
        qx_scale,
        weight_scale,
        alpha=alpha,
        bias=bias,
    )
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out.reshape(*input.shape[:-1], weight.size(0))


def _nvfp4_linear_cuda(input, weight, bias=None):
    if _NVFP4_KERNEL_BACKEND == _NVFP4_BACKEND_LIGHTX2V:
        return _nvfp4_linear_cuda_lightx2v(input, weight, bias=bias)
    return _nvfp4_linear_cuda_comfy(input, weight, bias=bias)


@torch.compiler.disable()
def _nvfp4_linear(input, weight, bias=None, op=None):
    if _nvfp4_can_use_kernel(input, weight):
        return _nvfp4_linear_cuda(input, weight, bias=bias)
    _nvfp4_note_fallback()
    dtype = input.dtype if torch.is_tensor(input) else weight.dtype
    device = input.device if torch.is_tensor(input) else weight.device
    w = weight.dequantize(dtype=dtype, device=device)
    if bias is not None and torch.is_tensor(bias) and bias.dtype != dtype:
        bias = bias.to(dtype)
    if op is not None:
        return op(input, w, bias)
    return torch.nn.functional.linear(input, w, bias)


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
    layout=_NVFP4_LAYOUT_LEGACY,
):
    if weight_u8.device != device:
        weight_u8 = weight_u8.to(device)
    scale = weight_scale if weight_scale.device == device else weight_scale.to(device)
    if alpha.device != device:
        alpha = alpha.to(device)
    if input_global_scale.device != device:
        input_global_scale = input_global_scale.to(device)
    if layout == _NVFP4_LAYOUT_TENSORCORE and device.type == "cuda" and _ck_cuda_available:
        try:
            return _ck_cuda.dequantize_nvfp4(weight_u8, alpha.to(torch.float32), scale, output_type=dtype)
        except Exception:
            pass

    m, k_bytes = weight_u8.shape
    byte_lut = _get_fp4_byte_lut(device, dtype)
    if layout == _NVFP4_LAYOUT_TENSORCORE:
        idx = _nvfp4_swap_nibbles(weight_u8).to(torch.int32)
    else:
        idx = weight_u8.to(torch.int32)
    out = byte_lut[idx].reshape(m, k_bytes * 2)

    scale = _deswizzle_nvfp4_scale(scale, out.shape[1], block_size=block_size, dtype=dtype)
    out = out.view(out.shape[0], scale.shape[1], block_size)
    out.mul_(scale.unsqueeze(-1))
    out = out.view(out.shape[0], -1)

    if layout == _NVFP4_LAYOUT_TENSORCORE:
        scale_factor = alpha.to(dtype)
    else:
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
        if scale_key not in state_dict:
            continue
        if not _is_float8_dtype(state_dict[scale_key].dtype):
            continue

        weight_scale_2_key = base + ".weight_scale_2"
        input_scale_key = base + ".input_scale"
        if weight_scale_2_key in state_dict:
            specs.append(
                {
                    "name": base,
                    "weight": tensor,
                    "weight_scale": state_dict[scale_key],
                    "weight_scale_2": state_dict[weight_scale_2_key],
                    "input_scale": state_dict.get(input_scale_key, None),
                    "bias": state_dict.get(base + ".bias", None),
                    "layout": _NVFP4_LAYOUT_TENSORCORE,
                }
            )
            continue

        input_global_key = base + ".input_global_scale"
        alpha_key = base + ".alpha"
        input_absmax_key = base + ".input_absmax"
        weight_global_scale_key = base + ".weight_global_scale"
        if input_global_key not in state_dict or alpha_key not in state_dict:
            if input_absmax_key not in state_dict or weight_global_scale_key not in state_dict:
                continue
            input_absmax = state_dict[input_absmax_key]
            weight_global_scale = state_dict[weight_global_scale_key]
            input_global_scale = (2688.0 / input_absmax).to(torch.float32)
            alpha = 1.0 / (input_global_scale * weight_global_scale.to(torch.float32))
        else:
            input_global_scale = state_dict[input_global_key]
            alpha = state_dict[alpha_key]
        specs.append(
            {
                "name": base,
                "weight": tensor,
                "weight_scale": state_dict[scale_key],
                "input_global_scale": input_global_scale,
                "alpha": alpha,
                "bias": state_dict.get(base + ".bias", None),
                "layout": _NVFP4_LAYOUT_LEGACY,
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
    _nvfp4_note_load_backend()
    quant_map = {}
    for spec in specs:
        qcfg = {"weights": "nvfp4", "activations": "none"}
        quant_map[spec["name"]] = qcfg
        quant_map[spec["name"] + ".weight"] = qcfg
    return {"state_dict": state_dict, "quant_map": quant_map}


def detect(state_dict, verboseLevel=1):
    matched = detect_nvfp4_state_dict(state_dict)
    details = describe_nvfp4_state_dict(state_dict) if matched else {}
    return {"matched": matched, "kind": "nvfp4" if matched else "none", "details": details}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    _nvfp4_note_reset()    
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
        size,
        stride,
        dtype,
        input_global_scale=None,
        alpha=None,
        input_scale=None,
        weight_scale_2=None,
        device=None,
        requires_grad=False,
        layout=_NVFP4_LAYOUT_LEGACY,
        allow_kernel=True,
    ):
        if input_global_scale is None and input_scale is not None:
            input_global_scale = input_scale
        if alpha is None and weight_scale_2 is not None:
            alpha = weight_scale_2
        if layout == _NVFP4_LAYOUT_LEGACY and (weight_scale_2 is not None or input_scale is not None):
            layout = _NVFP4_LAYOUT_TENSORCORE
        if input_global_scale is None or alpha is None:
            raise ValueError("NVFP4WeightTensor.create requires input_global_scale/alpha or input_scale/weight_scale_2")
        if torch.is_tensor(input_global_scale):
            try:
                if not torch.isfinite(input_global_scale).all():
                    allow_kernel = False
            except Exception:
                allow_kernel = False
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
            allow_kernel=allow_kernel,
            dtype=dtype,
            requires_grad=requires_grad,
            layout=layout,
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
        allow_kernel=True,
        requires_grad=False,
        layout=_NVFP4_LAYOUT_LEGACY,
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
        layout=_NVFP4_LAYOUT_LEGACY,
        allow_kernel=True,
    ):
        super().__init__(qtype, axis)
        self._data = weight_u8
        self._scale = weight_scale
        self._input_global_scale = input_global_scale
        self._alpha = alpha
        self._block_size = 16
        self._layout = layout
        self._allow_kernel = allow_kernel

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
            layout=self._layout,
        )

    def get_quantized_subtensors(self):
        if self._layout == _NVFP4_LAYOUT_TENSORCORE:
            return [
                ("weight_u8", self._data),
                ("weight_scale", self._scale),
                ("weight_scale_2", self._alpha),
                ("input_scale", self._input_global_scale),
            ]
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
        if "input_scale" in sub_map and sub_map["input_scale"] is not None:
            self._input_global_scale = sub_map["input_scale"]
        elif "input_global_scale" in sub_map and sub_map["input_global_scale"] is not None:
            self._input_global_scale = sub_map["input_global_scale"]
        if "weight_scale_2" in sub_map and sub_map["weight_scale_2"] is not None:
            self._alpha = sub_map["weight_scale_2"]
        elif "alpha" in sub_map and sub_map["alpha"] is not None:
            self._alpha = sub_map["alpha"]

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_input_global_scale", "_alpha"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "dtype": str(self.dtype),
            "layout": self._layout,
            "allow_kernel": "1" if self._allow_kernel else "0",
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
        layout = meta.get("layout", _NVFP4_LAYOUT_LEGACY)
        allow_kernel = str(meta.get("allow_kernel", "1")).strip().lower() not in ("0", "false", "no")
        return NVFP4WeightTensor(
            qtype=qtype,
            axis=axis,
            size=size,
            stride=stride,
            weight_u8=inner_tensors["_data"],
            weight_scale=inner_tensors["_scale"],
            input_global_scale=inner_tensors["_input_global_scale"],
            alpha=inner_tensors["_alpha"],
            allow_kernel=allow_kernel,
            dtype=dtype,
            layout=layout,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            input = args[0] if len(args) > 0 else kwargs.get("input", None)
            weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
            bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
            if isinstance(weight, NVFP4WeightTensor):
                return _nvfp4_linear(input, weight, bias=bias)
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
                return _nvfp4_linear(input, weight, bias=bias, op=op)
        if op is torch.ops.aten.detach:
            t = args[0]
            return NVFP4WeightTensor.create(
                weight_u8=op(t._data),
                weight_scale=op(t._scale),
                input_global_scale=op(t._input_global_scale),
                alpha=op(t._alpha),
                allow_kernel=getattr(t, "_allow_kernel", True),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=t.device,
                requires_grad=t.requires_grad,
                layout=t._layout,
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
                allow_kernel=getattr(t, "_allow_kernel", True),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=device,
                requires_grad=t.requires_grad,
                layout=t._layout,
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
        scale2_key = prefix + "weight_scale_2"
        igs_key = prefix + "input_global_scale"
        alpha_key = prefix + "alpha"
        input_absmax_key = prefix + "input_absmax"
        weight_global_scale_key = prefix + "weight_global_scale"
        bias_key = prefix + "bias"
        input_scale_key = prefix + "input_scale"
        output_scale_key = prefix + "output_scale"

        weight_u8 = state_dict.pop(weight_key, None)
        weight_scale = state_dict.pop(scale_key, None)
        weight_scale_2 = state_dict.pop(scale2_key, None)
        input_global_scale = state_dict.pop(igs_key, None)
        alpha = state_dict.pop(alpha_key, None)
        input_absmax = state_dict.pop(input_absmax_key, None)
        weight_global_scale = state_dict.pop(weight_global_scale_key, None)
        bias = state_dict.pop(bias_key, None)
        input_scale = state_dict.pop(input_scale_key, None)
        output_scale = state_dict.pop(output_scale_key, None)

        if weight_u8 is None:
            missing_keys.append(weight_key)
        if weight_scale is None:
            missing_keys.append(scale_key)
        layout = _NVFP4_LAYOUT_LEGACY
        allow_kernel = True
        if weight_scale_2 is not None or input_scale is not None:
            layout = _NVFP4_LAYOUT_TENSORCORE
            if weight_scale_2 is None:
                missing_keys.append(scale2_key)
            if input_scale is None:
                allow_kernel = False
                if torch.is_tensor(weight_scale_2):
                    input_scale = torch.full(
                        (),
                        float("nan"),
                        dtype=weight_scale_2.dtype,
                        device=weight_scale_2.device,
                    )
                elif torch.is_tensor(weight_u8):
                    input_scale = torch.full((), float("nan"), dtype=torch.float32, device=weight_u8.device)
                else:
                    input_scale = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            if input_global_scale is None or alpha is None:
                if input_absmax is not None and weight_global_scale is not None:
                    input_global_scale = (2688.0 / input_absmax).to(torch.float32)
                    alpha = 1.0 / (input_global_scale * weight_global_scale.to(torch.float32))
                else:
                    if input_global_scale is None:
                        missing_keys.append(igs_key)
                    if alpha is None:
                        missing_keys.append(alpha_key)

        target_dtype = self._nvfp4_default_dtype or self.weight.dtype
        if layout == _NVFP4_LAYOUT_TENSORCORE:
            if weight_u8 is not None and weight_scale is not None and weight_scale_2 is not None and input_scale is not None:
                nvfp4_weight = NVFP4WeightTensor.create(
                    weight_u8=weight_u8,
                    weight_scale=weight_scale,
                    input_global_scale=input_scale,
                    alpha=weight_scale_2,
                    allow_kernel=allow_kernel,
                    size=self.weight.size(),
                    stride=self.weight.stride(),
                    dtype=target_dtype,
                    device=weight_u8.device,
                    requires_grad=False,
                    layout=layout,
                )
                self.weight = torch.nn.Parameter(nvfp4_weight, requires_grad=False)
        else:
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
                    layout=layout,
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


def validate_nvfp4_kernel(
    state_dict=None,
    checkpoint_path=None,
    device=None,
    max_layers=4,
    seed=0,
    batch_size=2,
    dtype=torch.bfloat16,
    verbose=True,
):
    """Compare kernel vs fallback outputs for a few NVFP4 layers."""
    if state_dict is None:
        if checkpoint_path is None:
            raise ValueError("state_dict or checkpoint_path is required")
        from mmgp import safetensors2

        state_dict = {}
        with safetensors2.safe_open(checkpoint_path, framework="pt", device="cpu", writable_tensors=False) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    specs = _collect_nvfp4_specs(state_dict)
    if not specs:
        return {"ok": False, "reason": "no nvfp4 weights found"}

    candidates = sorted(specs, key=lambda spec: spec["weight"].numel())
    if isinstance(max_layers, int) and max_layers > 0:
        candidates = candidates[:max_layers]

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    results = []
    with torch.no_grad():
        for spec in candidates:
            weight = spec["weight"]
            layout = spec.get("layout", _NVFP4_LAYOUT_LEGACY)
            in_features = weight.shape[1] * 2
            bias = spec.get("bias")

            if layout == _NVFP4_LAYOUT_TENSORCORE:
                input_scale = spec.get("input_scale")
                tensor_scale = spec.get("weight_scale_2")
            else:
                input_scale = spec.get("input_global_scale")
                tensor_scale = spec.get("alpha")

            if input_scale is None or tensor_scale is None:
                results.append({"name": spec["name"], "layout": layout, "kernel": False, "reason": "missing scales"})
                continue

            nvfp4_weight = NVFP4WeightTensor.create(
                weight_u8=weight,
                weight_scale=spec["weight_scale"],
                input_global_scale=input_scale,
                alpha=tensor_scale,
                size=(weight.shape[0], in_features),
                stride=(in_features, 1),
                dtype=dtype,
                device=device,
                requires_grad=False,
                layout=layout,
            )

            x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
            if bias is not None:
                bias = bias.to(device=device, dtype=dtype)

            kernel_ok = _nvfp4_can_use_kernel(x, nvfp4_weight)
            y_kernel = _nvfp4_linear_cuda(x, nvfp4_weight, bias=bias) if kernel_ok else None
            x_ref = x
            if (
                layout == _NVFP4_LAYOUT_TENSORCORE
                and _ck_cuda_available
                and device.type == "cuda"
                and torch.is_tensor(input_scale)
            ):
                input_scale_fp32 = input_scale.to(device=device, dtype=torch.float32)
                pad_16x = (x.shape[0] % 16) != 0
                qx, qx_scale = _ck_cuda.quantize_nvfp4(x, input_scale_fp32, 0.0, pad_16x)
                x_ref = _ck_cuda.dequantize_nvfp4(qx, input_scale_fp32, qx_scale, output_type=dtype)
                if pad_16x:
                    x_ref = x_ref[: x.shape[0]]
            y_ref = torch.nn.functional.linear(
                x_ref,
                nvfp4_weight.dequantize(dtype=dtype, device=device),
                bias,
            )

            if y_kernel is None:
                results.append({"name": spec["name"], "layout": layout, "kernel": False})
                continue

            diff = (y_kernel - y_ref).float()
            results.append(
                {
                    "name": spec["name"],
                    "layout": layout,
                    "kernel": True,
                    "max_abs": diff.abs().max().item(),
                    "mean_abs": diff.abs().mean().item(),
                }
            )

    if verbose:
        print("NVFP4 kernel validation:")
        for entry in results:
            if not entry.get("kernel"):
                print(f"  {entry['name']}: kernel skipped ({entry.get('reason', 'incompatible')})")
                continue
            print(
                f"  {entry['name']}: max_abs={entry['max_abs']:.6f} mean_abs={entry['mean_abs']:.6f}"
            )

    return {"ok": True, "results": results}
