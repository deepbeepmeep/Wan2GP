from __future__ import annotations

import importlib
import os
import atexit
import traceback
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

try:
    from torch._subclasses.fake_tensor import is_fake as _torch_is_fake_tensor
except Exception:  # pragma: no cover
    _torch_is_fake_tensor = None

# Env toggles
_ENV_ENABLE = "WAN2GP_QUANTO_INT8_KERNEL"
_ENV_DEBUG = "WAN2GP_QUANTO_INT8_DEBUG"
_ENV_ALLOW_RUNTIME_FALLBACK = "WAN2GP_QUANTO_INT8_ALLOW_RUNTIME_FALLBACK"
_ENV_NATIVE_FALLBACK_MAX_M = "WAN2GP_QUANTO_INT8_NATIVE_FALLBACK_MAX_M"
_ENV_PROFILE_SHAPES = "WAN2GP_QUANTO_INT8_PROFILE_SHAPES"
_ENV_PROFILE_TIME = "WAN2GP_QUANTO_INT8_PROFILE_TIME"

_STARTUP_PRINTED = False
_RUNTIME_DISABLED = False
_RUNTIME_DISABLE_REASON = ""
_RUNTIME_DISABLE_PRINTED = False
_TRITON_MODULE = None
_TRITON_DIRECT_FUSED_READY = False
_TRITON_DIRECT_SCALED_READY = False
_KERNEL_USED_PRINTED = False
_SHAPE_PROFILE_ON = False
_SHAPE_COUNTS_FUSED = {}
_SHAPE_COUNTS_SCALED = {}
_TIME_PROFILE_ON = False
_TIME_PROFILE_EVENTS = []
_TIME_PROFILE_CPU_MS = 0.0
_TIME_PROFILE_CALLS = 0
_DEBUG_OVERRIDE: Optional[bool] = None

_PATCH_STATE = SimpleNamespace(enabled=False, orig_forward=None)
_OPS_REGISTERED = False
_OPS_NAMESPACE = "wan2gp_int8"
_OPS_LIBS = []
_FUSED_LAUNCH_CACHE_MAX = 4096
_FUSED_LAUNCH_CACHE = {}
_FUSED_LAUNCH_CACHE_FIFO = []
_SCALED_LAUNCH_CACHE_MAX = 4096
_SCALED_LAUNCH_CACHE = {}
_SCALED_LAUNCH_CACHE_FIFO = []
_QBYTES_TENSOR_CLS = None
_WEIGHT_QBYTES_CLS = None
_NATIVE_FALLBACK_MAX_M = 0


def _encode_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 1
    if dtype == torch.float32:
        return 2
    return 0


def _decode_dtype(code: int, fallback: torch.dtype = torch.bfloat16) -> torch.dtype:
    if int(code) == 1:
        return torch.float16
    if int(code) == 2:
        return torch.float32
    return torch.bfloat16 if fallback not in (torch.bfloat16, torch.float16, torch.float32) else fallback


def _env_flag(name: str, default: str = "1") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _log(msg: str) -> None:
    print(f"[WAN2GP][INT8][quanto] {msg}")


def _debug(msg: str) -> None:
    if _DEBUG_OVERRIDE is None:
        debug_on = _env_flag(_ENV_DEBUG, "0")
    else:
        debug_on = bool(_DEBUG_OVERRIDE)
    if debug_on:
        _log(msg)


def _format_exception_detail(exc: Exception) -> str:
    try:
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    except Exception:
        return str(exc)


def _summarize_kernel_error(exc_or_text: Exception | str, max_chars: int = 480) -> str:
    text = str(exc_or_text)
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
    if len(lines) == 0:
        return "Unknown Triton kernel failure"
    keywords = (
        "CompilationError",
        "shape mismatch",
        "tl.dot",
        "K >=",
        "M >=",
        "N >=",
        "Triton",
        "unsupported",
        "invalid",
        "at ",
    )
    picked = [ln for ln in lines if any(kw in ln for kw in keywords)]
    if len(picked) == 0:
        picked = [lines[-1]]
    unique: list[str] = []
    seen = set()
    for ln in picked:
        if ln in seen:
            continue
        seen.add(ln)
        unique.append(ln)
    summary = " | ".join(unique[-4:])
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


def set_kernel_debug(enabled: Optional[bool] = None) -> None:
    global _DEBUG_OVERRIDE
    _DEBUG_OVERRIDE = None if enabled is None else bool(enabled)


def _allow_runtime_fallback() -> bool:
    return _env_flag(_ENV_ALLOW_RUNTIME_FALLBACK, "1")


def _startup_status(enabled: bool, detail: str) -> None:
    global _STARTUP_PRINTED
    if _STARTUP_PRINTED:
        return
    _STARTUP_PRINTED = True
    if enabled:
        _log(f"Injected int8 kernels ACTIVE (backend=triton).")
    else:
        _log(f"Injected int8 kernels INACTIVE. {detail}")


def _disable_runtime(reason: str) -> None:
    global _RUNTIME_DISABLED, _RUNTIME_DISABLE_REASON, _RUNTIME_DISABLE_PRINTED
    _RUNTIME_DISABLED = True
    _RUNTIME_DISABLE_REASON = _summarize_kernel_error(reason)
    if not _RUNTIME_DISABLE_PRINTED:
        _RUNTIME_DISABLE_PRINTED = True
        _log(
            "Runtime fallback to non-injected Quanto path is now active. Reason: "
            f"{_RUNTIME_DISABLE_REASON}"
        )


def _init_quanto_tensor_types() -> bool:
    global _QBYTES_TENSOR_CLS, _WEIGHT_QBYTES_CLS
    if _QBYTES_TENSOR_CLS is not None and _WEIGHT_QBYTES_CLS is not None:
        return True
    try:
        from optimum.quanto.tensor.qbytes import QBytesTensor
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
    except Exception:
        return False
    _QBYTES_TENSOR_CLS = QBytesTensor
    _WEIGHT_QBYTES_CLS = WeightQBytesTensor
    return True


def _refresh_triton_direct_kernel_flags() -> None:
    global _TRITON_DIRECT_FUSED_READY, _TRITON_DIRECT_SCALED_READY
    mod = _TRITON_MODULE
    triton_ns = getattr(mod, "triton", None) if mod is not None else None
    has_common = bool(mod is not None and triton_ns is not None and hasattr(triton_ns, "cdiv") and hasattr(mod, "_select_triton_int8_config"))
    _TRITON_DIRECT_FUSED_READY = bool(has_common and hasattr(mod, "_fused_dynamic_int8_blockscale_gemm_kernel"))
    _TRITON_DIRECT_SCALED_READY = bool(has_common and hasattr(mod, "_scaled_int8_gemm_kernel"))


def _is_qbytes_tensor(t: torch.Tensor) -> bool:
    if not _init_quanto_tensor_types():
        return False
    return isinstance(t, _QBYTES_TENSOR_CLS)


def _is_weight_qbytes(t: torch.Tensor) -> bool:
    if not _init_quanto_tensor_types():
        return False
    return isinstance(t, _WEIGHT_QBYTES_CLS)


def _flatten_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2 and scale.shape[1] == 1:
        return scale.view(-1)
    if scale.ndim == 1:
        return scale
    return scale.reshape(-1)


def _expand_scale_to_rows(scale: torch.Tensor, rows: int, dtype: torch.dtype, device: Optional[torch.device] = None) -> torch.Tensor:
    scale = _flatten_scale(scale)
    if scale.numel() == 1:
        scale = scale.reshape(1).expand(rows)
    elif scale.numel() != rows:
        raise RuntimeError(f"Activation scale length mismatch: expected {rows}, got {scale.numel()}")
    if device is None:
        return scale.contiguous().to(dtype=dtype)
    return scale.contiguous().to(device=device, dtype=dtype, non_blocking=True)


def _prepare_weight_scale(scale: torch.Tensor, out_features: int, device: torch.device) -> torch.Tensor:
    flat_scale = _flatten_scale(scale)
    if flat_scale.numel() != out_features:
        raise RuntimeError("Weight scale length does not match output features")
    if flat_scale.device != device:
        flat_scale = flat_scale.to(device=device, non_blocking=True)
    if flat_scale.dtype != torch.float32:
        flat_scale = flat_scale.to(torch.float32)
    if not flat_scale.is_contiguous():
        flat_scale = flat_scale.contiguous()
    return flat_scale


def _cache_launch_params(cache: dict, fifo: list, max_size: int, key: tuple[int, int, int, int], params: tuple[int, int, int, int, int, int, int]) -> tuple[int, int, int, int, int, int, int]:
    if key in cache:
        return cache[key]
    cache[key] = params
    fifo.append(key)
    if len(fifo) > max_size:
        stale_key = fifo.pop(0)
        cache.pop(stale_key, None)
    return params


def _fused_launch_params(m: int, k: int, n: int, device: torch.device) -> tuple[int, int, int, int, int, int, int]:
    device_index = int(device.index if device.type == "cuda" else -1)
    key = (device_index, m, k, n)
    cached = _FUSED_LAUNCH_CACHE.get(key)
    if cached is not None:
        return cached
    mod = _TRITON_MODULE
    if mod is None:
        raise RuntimeError("Triton backend not initialized")
    block_m, block_n, block_k, num_warps, num_stages = mod._select_triton_int8_config(m, k, n, device=device, kernel_kind="fused")
    grid_m = mod.triton.cdiv(m, block_m)
    grid_n = mod.triton.cdiv(n, block_n)
    params = (block_m, block_n, block_k, num_warps, num_stages, grid_m, grid_n)
    return _cache_launch_params(_FUSED_LAUNCH_CACHE, _FUSED_LAUNCH_CACHE_FIFO, _FUSED_LAUNCH_CACHE_MAX, key, params)


def _scaled_launch_params(m: int, k: int, n: int, device: torch.device) -> tuple[int, int, int, int, int, int, int]:
    device_index = int(device.index if device.type == "cuda" else -1)
    key = (device_index, m, k, n)
    cached = _SCALED_LAUNCH_CACHE.get(key)
    if cached is not None:
        return cached
    mod = _TRITON_MODULE
    if mod is None:
        raise RuntimeError("Triton backend not initialized")
    block_m, block_n, block_k, num_warps, num_stages = mod._select_triton_int8_config(m, k, n, device=device, kernel_kind="scaled")
    grid_m = mod.triton.cdiv(m, block_m)
    grid_n = mod.triton.cdiv(n, block_n)
    params = (block_m, block_n, block_k, num_warps, num_stages, grid_m, grid_n)
    return _cache_launch_params(_SCALED_LAUNCH_CACHE, _SCALED_LAUNCH_CACHE_FIFO, _SCALED_LAUNCH_CACHE_MAX, key, params)


def _is_compiling_graph() -> bool:
    try:
        if bool(torch.compiler.is_compiling()):
            return True
    except Exception:
        pass
    try:
        import torch._dynamo as _dynamo

        if bool(_dynamo.is_compiling()):
            return True
    except Exception:
        pass
    return False


def _is_fake_tensor(t: object) -> bool:
    if not torch.is_tensor(t):
        return False
    if _torch_is_fake_tensor is not None:
        return bool(_torch_is_fake_tensor(t))
    return False


def _resolve_output_dtype(input: torch.Tensor, other: torch.Tensor) -> torch.dtype:
    other_scale = getattr(other, "_scale", None)
    if torch.is_tensor(other_scale) and other_scale.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return other_scale.dtype
    if _is_qbytes_tensor(input):
        input_scale = getattr(input, "_scale", None)
        if torch.is_tensor(input_scale) and input_scale.dtype in (torch.bfloat16, torch.float16, torch.float32):
            return input_scale.dtype
    if isinstance(input, torch.Tensor) and input.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return input.dtype
    return torch.bfloat16


def _probe_triton_backend() -> Tuple[Optional[object], str]:
    try:
        mod = importlib.import_module("shared.kernels.quanto_int8_triton")
    except Exception as exc:
        return None, f"failed to import shared.kernels.quanto_int8_triton ({exc})"

    if not hasattr(mod, "is_available"):
        return None, "shared.kernels.quanto_int8_triton.is_available() missing"
    try:
        if not bool(mod.is_available()):
            return None, "Triton backend unavailable on this runtime/GPU"
    except Exception as exc:
        return None, f"Triton availability check failed ({exc})"
    return mod, "ok"


def _register_int8_ops_for_namespace(ns: str, lib: torch.library.Library) -> None:
    lib.define("fused_quant_scaled_mm(Tensor x2d, Tensor qweight, Tensor qweight_scale, int out_dtype_code=0) -> Tensor")
    lib.define("scaled_int8_mm(Tensor a_int8, Tensor b_int8, Tensor a_scale, Tensor b_scale, int out_dtype_code=0) -> Tensor")

    @torch.library.impl(f"{ns}::fused_quant_scaled_mm", "CUDA")
    def _fused_quant_scaled_mm_cuda(x2d: torch.Tensor, qweight: torch.Tensor, qweight_scale: torch.Tensor, out_dtype_code: int = 0):
        if _TRITON_MODULE is None:
            raise RuntimeError("Triton backend not initialized")
        out_dtype = _decode_dtype(out_dtype_code, x2d.dtype)
        return _TRITON_MODULE.fused_quant_scaled_mm(x2d, qweight, qweight_scale, out_dtype=out_dtype)

    @torch.library.impl(f"{ns}::scaled_int8_mm", "CUDA")
    def _scaled_int8_mm_cuda(a_int8: torch.Tensor, b_int8: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, out_dtype_code: int = 0):
        if _TRITON_MODULE is None:
            raise RuntimeError("Triton backend not initialized")
        out_dtype = _decode_dtype(out_dtype_code, torch.bfloat16)
        return _TRITON_MODULE.scaled_int8_mm(a_int8, b_int8, a_scale, b_scale, out_dtype=out_dtype)

    @torch.library.register_fake(f"{ns}::fused_quant_scaled_mm")
    def _fused_quant_scaled_mm_fake(x2d: torch.Tensor, qweight: torch.Tensor, qweight_scale: torch.Tensor, out_dtype_code: int = 0):
        if x2d.ndim != 2 or qweight.ndim != 2:
            raise RuntimeError("fused_quant_scaled_mm expects 2D tensors")
        out_dtype = _decode_dtype(out_dtype_code, x2d.dtype)
        return x2d.new_empty((x2d.shape[0], qweight.shape[0]), dtype=out_dtype)

    @torch.library.register_fake(f"{ns}::scaled_int8_mm")
    def _scaled_int8_mm_fake(a_int8: torch.Tensor, b_int8: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, out_dtype_code: int = 0):
        if a_int8.ndim != 2 or b_int8.ndim != 2:
            raise RuntimeError("scaled_int8_mm expects 2D tensors")
        out_dtype = _decode_dtype(out_dtype_code, torch.bfloat16)
        return a_int8.new_empty((a_int8.shape[0], b_int8.shape[0]), dtype=out_dtype)


def _ensure_compile_safe_ops() -> None:
    global _OPS_REGISTERED, _OPS_LIBS
    if _OPS_REGISTERED:
        return

    libs = []
    try:
        lib = torch.library.Library(_OPS_NAMESPACE, "DEF")
        libs.append(lib)
        _register_int8_ops_for_namespace(_OPS_NAMESPACE, lib)
    except Exception:
        # Namespace/op may already exist in long-lived processes.
        op_ns = getattr(torch.ops, _OPS_NAMESPACE, None)
        has_ops = bool(
            op_ns is not None
            and hasattr(op_ns, "fused_quant_scaled_mm")
            and hasattr(op_ns, "scaled_int8_mm")
        )
        if not has_ops:
            raise
    _OPS_LIBS = libs

    _OPS_REGISTERED = True


def _fused_quant_scaled_mm_direct_call(x2d: torch.Tensor, qweight: torch.Tensor, qweight_scale: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    mod = _TRITON_MODULE
    if mod is None:
        raise RuntimeError("Triton backend not initialized")
    if not _TRITON_DIRECT_FUSED_READY:
        return mod.fused_quant_scaled_mm(x2d, qweight, qweight_scale, out_dtype=output_dtype)

    m, k = x2d.shape
    n, k2 = qweight.shape
    if k != k2:
        raise RuntimeError(f"Triton int8 GEMM shape mismatch: x={x2d.shape}, w={qweight.shape}")

    block_m, block_n, block_k, num_warps, num_stages, grid_m, grid_n = _fused_launch_params(m, k, n, x2d.device)
    out = torch.empty((m, n), device=x2d.device, dtype=output_dtype)
    try:
        mod._fused_dynamic_int8_blockscale_gemm_kernel[(grid_m, grid_n)](
            x2d,
            qweight,
            qweight_scale,
            out,
            m,
            n,
            k,
            x2d.stride(0),
            x2d.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            out.stride(0),
            out.stride(1),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    except Exception as exc:
        raise RuntimeError(
            "Triton fused int8 kernel launch failed "
            f"(shape m={m}, k={k}, n={n}; tile=({block_m},{block_n},{block_k}); "
            f"warps={num_warps}, stages={num_stages}). {exc}"
        ) from exc
    return out


def _scaled_int8_mm_direct_call(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    mod = _TRITON_MODULE
    if mod is None:
        raise RuntimeError("Triton backend not initialized")
    if not _TRITON_DIRECT_SCALED_READY:
        return mod.scaled_int8_mm(a_int8, b_int8, a_scale, b_scale, out_dtype=output_dtype)

    m, k = a_int8.shape
    n, k2 = b_int8.shape
    if k != k2:
        raise RuntimeError(f"Triton int8 GEMM shape mismatch: a={a_int8.shape}, w={b_int8.shape}")

    block_m, block_n, block_k, num_warps, num_stages, grid_m, grid_n = _scaled_launch_params(m, k, n, a_int8.device)
    out = torch.empty((m, n), device=a_int8.device, dtype=output_dtype)
    try:
        mod._scaled_int8_gemm_kernel[(grid_m, grid_n)](
            a_int8,
            b_int8,
            a_scale,
            b_scale,
            out,
            m,
            n,
            k,
            a_int8.stride(0),
            a_int8.stride(1),
            b_int8.stride(0),
            b_int8.stride(1),
            out.stride(0),
            out.stride(1),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    except Exception as exc:
        raise RuntimeError(
            "Triton scaled int8 kernel launch failed "
            f"(shape m={m}, k={k}, n={n}; tile=({block_m},{block_n},{block_k}); "
            f"warps={num_warps}, stages={num_stages}). {exc}"
        ) from exc
    return out


def _fused_quant_scaled_mm_call(x2d: torch.Tensor, qweight: torch.Tensor, qweight_scale: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    if _TRITON_MODULE is not None and not _is_compiling_graph() and not (_is_fake_tensor(x2d) or _is_fake_tensor(qweight) or _is_fake_tensor(qweight_scale)):
        return _fused_quant_scaled_mm_direct_call(x2d, qweight, qweight_scale, output_dtype)
    return torch.ops.wan2gp_int8.fused_quant_scaled_mm(x2d, qweight, qweight_scale, _encode_dtype(output_dtype))


def _scaled_int8_mm_call(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if _TRITON_MODULE is not None and not _is_compiling_graph() and not ( _is_fake_tensor(a_int8) or _is_fake_tensor(b_int8) or _is_fake_tensor(a_scale) or _is_fake_tensor(b_scale)):
        return _scaled_int8_mm_direct_call(a_int8, b_int8, a_scale, b_scale, output_dtype)
    return torch.ops.wan2gp_int8.scaled_int8_mm(a_int8, b_int8, a_scale, b_scale, _encode_dtype(output_dtype))


def _use_int8_kernel(input: torch.Tensor, other: torch.Tensor) -> bool:
    if _RUNTIME_DISABLED:
        return False
    if _TRITON_MODULE is None:
        return False
    if not _is_weight_qbytes(other):
        return False
    if other._data.dtype != torch.int8:
        return False
    if not other._data.is_cuda:
        return False

    if _is_qbytes_tensor(input):
        return input._data.dtype == torch.int8 and input._data.is_cuda
    return input.is_cuda and input.dtype in (torch.bfloat16, torch.float16, torch.float32)


def _activation_rows(input_shape: torch.Size) -> int:
    rows = 1
    for dim in input_shape[:-1]:
        rows *= int(dim)
    return rows


def _prefer_native_quanto_path(input: torch.Tensor) -> bool:
    if _NATIVE_FALLBACK_MAX_M < 0:
        return False
    return _activation_rows(input.shape) <= _NATIVE_FALLBACK_MAX_M


def _mark_kernel_used() -> None:
    global _KERNEL_USED_PRINTED
    if _KERNEL_USED_PRINTED:
        return
    _KERNEL_USED_PRINTED = True
    _log("Injected Triton int8 kernels are being used.")


def _int8_linear_forward_triton_dense_fast(ctx, input: torch.Tensor, other: torch.Tensor, bias: Optional[torch.Tensor]):
    ctx.save_for_backward(input, other)
    if _TRITON_MODULE is None:
        raise RuntimeError("Triton backend not initialized")
    _mark_kernel_used()

    input_shape = input.shape
    in_features = int(input_shape[-1])
    out_features = int(other.shape[0])
    a_2d = input.reshape(-1, in_features)
    if not a_2d.is_contiguous():
        a_2d = a_2d.contiguous()
    b_int8 = other._data
    if not b_int8.is_contiguous():
        b_int8 = b_int8.contiguous()
    b_scale = _prepare_weight_scale(other._scale, out_features, b_int8.device)

    if _SHAPE_PROFILE_ON:
        key = (int(a_2d.shape[0]), int(in_features), int(out_features))
        _SHAPE_COUNTS_FUSED[key] = _SHAPE_COUNTS_FUSED.get(key, 0) + 1
    if _TIME_PROFILE_ON and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out_2d = _fused_quant_scaled_mm_call(a_2d, b_int8, b_scale, input.dtype)
        end.record()
        _TIME_PROFILE_EVENTS.append((start, end))
    else:
        out_2d = _fused_quant_scaled_mm_call(a_2d, b_int8, b_scale, input.dtype)

    out = out_2d.reshape(input_shape[:-1] + (out_features,))
    if bias is not None:
        out = out + bias
    return out


def _int8_linear_forward_triton(ctx, input: torch.Tensor, other: torch.Tensor, bias: Optional[torch.Tensor]):
    ctx.save_for_backward(input, other)
    if _TRITON_MODULE is None:
        raise RuntimeError("Triton backend not initialized")
    _mark_kernel_used()

    input_shape = input.shape
    in_features = int(input_shape[-1])
    out_features = int(other.shape[0])
    b_int8 = other._data
    if not b_int8.is_contiguous():
        b_int8 = b_int8.contiguous()
    b_scale = _prepare_weight_scale(other._scale, out_features, b_int8.device)
    output_dtype = _resolve_output_dtype(input, other)
    input_is_qbytes = _is_qbytes_tensor(input)

    if input_is_qbytes:
        a_int8 = input._data.reshape(-1, in_features)
        if a_int8.dtype != torch.int8:
            raise RuntimeError("QBytes input must be int8 for injected path")
        if not a_int8.is_contiguous():
            a_int8 = a_int8.contiguous()
        a_scale = _expand_scale_to_rows(input._scale, a_int8.shape[0], torch.float32, device=a_int8.device)
        if _SHAPE_PROFILE_ON:
            key = (int(a_int8.shape[0]), int(in_features), int(out_features))
            _SHAPE_COUNTS_SCALED[key] = _SHAPE_COUNTS_SCALED.get(key, 0) + 1
        if _TIME_PROFILE_ON and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out_2d = _scaled_int8_mm_call(a_int8, b_int8, a_scale, b_scale, output_dtype)
            end.record()
            _TIME_PROFILE_EVENTS.append((start, end))
        else:
            out_2d = _scaled_int8_mm_call(a_int8, b_int8, a_scale, b_scale, output_dtype)
    else:
        a_2d = input.reshape(-1, in_features)
        if not a_2d.is_contiguous():
            a_2d = a_2d.contiguous()
        if _SHAPE_PROFILE_ON:
            key = (int(a_2d.shape[0]), int(in_features), int(out_features))
            _SHAPE_COUNTS_FUSED[key] = _SHAPE_COUNTS_FUSED.get(key, 0) + 1
        if _TIME_PROFILE_ON and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out_2d = _fused_quant_scaled_mm_call(a_2d, b_int8, b_scale, output_dtype)
            end.record()
            _TIME_PROFILE_EVENTS.append((start, end))
        else:
            out_2d = _fused_quant_scaled_mm_call(a_2d, b_int8, b_scale, output_dtype)

    out = out_2d.reshape(input_shape[:-1] + (out_features,))
    if bias is not None:
        out = out + bias
    return out


def enable_quanto_int8_kernel(triton_mod=None) -> bool:
    global _TRITON_MODULE, _NATIVE_FALLBACK_MAX_M
    if _PATCH_STATE.enabled:
        return True

    try:
        from optimum.quanto.tensor.weights import qbytes as _qbytes
    except Exception as exc:
        _debug(f"cannot import optimum.quanto qbytes ({exc})")
        return False

    if triton_mod is None:
        triton_mod, _ = _probe_triton_backend()
    if triton_mod is None:
        return False
    _TRITON_MODULE = triton_mod
    _refresh_triton_direct_kernel_flags()
    _NATIVE_FALLBACK_MAX_M = _env_int(_ENV_NATIVE_FALLBACK_MAX_M, 0)
    _init_quanto_tensor_types()
    _ensure_compile_safe_ops()

    orig_forward = _qbytes.WeightQBytesLinearFunction.forward

    def forward(ctx, input, other, bias=None):
        dense_hot_path = (
            not _RUNTIME_DISABLED
            and type(input) is torch.Tensor
            and input.is_cuda
            and input.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and _WEIGHT_QBYTES_CLS is not None
            and isinstance(other, _WEIGHT_QBYTES_CLS)
            and other._data.dtype == torch.int8
            and other._data.is_cuda
        )
        if dense_hot_path:
            if _prefer_native_quanto_path(input):
                return orig_forward(ctx, input, other, bias)
            try:
                return _int8_linear_forward_triton_dense_fast(ctx, input, other, bias)
            except Exception as exc:
                short_reason = _summarize_kernel_error(exc)
                if _allow_runtime_fallback():
                    _disable_runtime(short_reason)
                    _debug(f"Full Triton failure detail:\n{_format_exception_detail(exc)}")
                    return orig_forward(ctx, input, other, bias)
                full_detail = _format_exception_detail(exc)
                raise RuntimeError(
                    "Injected Triton int8 kernel failed. "
                    f"Set {_ENV_ALLOW_RUNTIME_FALLBACK}=1 to force fallback to non-injected Quanto path. "
                    f"Reason: {short_reason}\n"
                    f"Full Triton error details:\n{full_detail}"
                ) from exc

        if not _use_int8_kernel(input, other):
            return orig_forward(ctx, input, other, bias)
        if _prefer_native_quanto_path(input):
            return orig_forward(ctx, input, other, bias)
        try:
            return _int8_linear_forward_triton(ctx, input, other, bias)
        except Exception as exc:
            short_reason = _summarize_kernel_error(exc)
            if _allow_runtime_fallback():
                _disable_runtime(short_reason)
                _debug(f"Full Triton failure detail:\n{_format_exception_detail(exc)}")
                return orig_forward(ctx, input, other, bias)
            full_detail = _format_exception_detail(exc)
            raise RuntimeError(
                "Injected Triton int8 kernel failed. "
                f"Set {_ENV_ALLOW_RUNTIME_FALLBACK}=1 to force fallback to non-injected Quanto path. "
                f"Reason: {short_reason}\n"
                f"Full Triton error details:\n{full_detail}"
            ) from exc

    _qbytes.WeightQBytesLinearFunction.forward = staticmethod(forward)
    _PATCH_STATE.enabled = True
    _PATCH_STATE.orig_forward = orig_forward
    return True


def disable_quanto_int8_kernel(notify_disabled = False) -> bool:
    global _FUSED_LAUNCH_CACHE, _FUSED_LAUNCH_CACHE_FIFO, _SCALED_LAUNCH_CACHE, _SCALED_LAUNCH_CACHE_FIFO
    global _TRITON_DIRECT_FUSED_READY, _TRITON_DIRECT_SCALED_READY, _STARTUP_PRINTED
    
    if not _PATCH_STATE.enabled:
        return False
    from optimum.quanto.tensor.weights import qbytes as _qbytes

    _qbytes.WeightQBytesLinearFunction.forward = staticmethod(_PATCH_STATE.orig_forward)
    _PATCH_STATE.enabled = False
    _PATCH_STATE.orig_forward = None
    _FUSED_LAUNCH_CACHE = {}
    _FUSED_LAUNCH_CACHE_FIFO = []
    _SCALED_LAUNCH_CACHE = {}
    _SCALED_LAUNCH_CACHE_FIFO = []
    _TRITON_DIRECT_FUSED_READY = False
    _TRITON_DIRECT_SCALED_READY = False
    _STARTUP_PRINTED = False
    if notify_disabled:
        _startup_status(False, f"disabled by User.")
    return True


def maybe_enable_quanto_int8_kernel(verbose_level: Optional[int] = None) -> bool:
    global _SHAPE_PROFILE_ON, _TIME_PROFILE_ON, _STARTUP_PRINTED 

    _STARTUP_PRINTED = False
    verbose_debug: Optional[bool] = None
    if verbose_level is not None:
        try:
            verbose_debug = int(verbose_level) >= 2
        except Exception:
            verbose_debug = False
    set_kernel_debug(verbose_debug)

    if not _env_flag(_ENV_ENABLE, "1"):
        # _startup_status(False, f"disabled by {_ENV_ENABLE}=0; using non-injected Quanto path.")
        return False

    triton_mod, reason = _probe_triton_backend()
    if triton_mod is None:
        # _startup_status(False, f"{reason}; using non-injected Quanto path.")
        return False
    set_triton_debug = getattr(triton_mod, "set_autotune_debug", None)
    if callable(set_triton_debug):
        set_triton_debug(verbose_debug)

    if not enable_quanto_int8_kernel(triton_mod=triton_mod):
        _startup_status(False, "failed to patch Quanto linear forward; using non-injected Quanto path.")
        return False

    _SHAPE_PROFILE_ON = _env_flag(_ENV_PROFILE_SHAPES, "0")
    _TIME_PROFILE_ON = _env_flag(_ENV_PROFILE_TIME, "0")
    _startup_status(
        True,
        (
            "Triton int8 kernels will be used for Quanto qint8 linear layers "
            "(QBytes int8 activations + fused dynamic int8 activation quantization)."
        ),
    )
    return True



def _print_shape_profile() -> None:
    if not _SHAPE_PROFILE_ON and not _TIME_PROFILE_ON:
        return
    if _SHAPE_PROFILE_ON and _SHAPE_COUNTS_FUSED:
        top_fused = sorted(_SHAPE_COUNTS_FUSED.items(), key=lambda kv: kv[1], reverse=True)[:10]
        _log(f"Fused shape profile (top {len(top_fused)}): {top_fused}")
    if _SHAPE_PROFILE_ON and _SHAPE_COUNTS_SCALED:
        top_scaled = sorted(_SHAPE_COUNTS_SCALED.items(), key=lambda kv: kv[1], reverse=True)[:10]
        _log(f"Scaled shape profile (top {len(top_scaled)}): {top_scaled}")

    if _TIME_PROFILE_ON:
        total_ms = 0.0
        calls = 0
        if _TIME_PROFILE_EVENTS:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for start, end in _TIME_PROFILE_EVENTS:
                total_ms += float(start.elapsed_time(end))
            calls = len(_TIME_PROFILE_EVENTS)
        else:
            total_ms = _TIME_PROFILE_CPU_MS
            calls = _TIME_PROFILE_CALLS
        _log(f"Triton kernel time profile: {total_ms / 1000.0:.3f}s over {calls} calls")


atexit.register(_print_shape_profile)
