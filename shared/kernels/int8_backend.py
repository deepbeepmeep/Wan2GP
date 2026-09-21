"""Optional INT8 backends shared by Quanto and ConvRot linear layers."""
from __future__ import annotations

import importlib
import os

import torch

from shared.kernels import quanto_int8_inject as triton


CHOICES = [("Disabled (PyTorch)", "disabled"), ("Auto", "auto"),
           ("Triton", "triton"), ("Comfy Kitchen Kernels", "kitchen")]
_backend = "pytorch"
revision = 0
_kitchen = None
_kitchen_hip = False
_direct_cutlass = False
_original_forward = None
_ops_registered = False
_fusion_logged = False
_compile_cache_root = None
_compile_cache_backend = None
# Bump when changes to INT8 custom operators invalidate compiled graphs.
_COMPILE_CACHE_VERSION = 1
# Includes row quantization, a possible INT32 GEMM result, and a temporary output.
# Never allocate an activation-sized quantization buffer for a whole video.
_SCRATCH_BYTES = 16 * 1024 * 1024


def prepare_compile_cache(enabled):
    """Isolate compiled graphs by resolved backend; leave eager runs untouched."""
    global _compile_cache_root, _compile_cache_backend
    if not enabled or _compile_cache_backend == _backend:
        return
    from torch._inductor.runtime.runtime_utils import cache_dir

    if _compile_cache_root is None:
        _compile_cache_root = cache_dir()
    path = os.path.join(_compile_cache_root, f"wangp_int8_v{_COMPILE_CACHE_VERSION}", _backend)
    os.makedirs(path, exist_ok=True)
    # Reset in-memory graphs as well when switching backends in the same process.
    torch.compiler.reset()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = path
    _compile_cache_backend = _backend
    print(f"[INT8] Compile Cache: {_backend} (v{_COMPILE_CACHE_VERSION}).")


def _probe_kitchen():
    if not torch.cuda.is_available():
        return None, "requires a CUDA or ROCm GPU"
    hip = torch.version.hip is not None
    if not hip and torch.cuda.get_device_capability() < (7, 5):
        return None, "requires an NVIDIA GPU with compute capability 7.5 or newer"
    try:
        from importlib.metadata import version
        from packaging.version import Version

        if Version(version("comfy-kitchen")) < Version("0.2.35"):
            return None, "requires comfy-kitchen >= 0.2.35"
        module = importlib.import_module("comfy_kitchen.backends.hip" if hip else "comfy_kitchen.backends.cuda")
        if hip and not module.has_wmma():
            return None, "Kitchen HIP INT8 requires a working HIP extension and supported RDNA3/3.5/4 GPU"
        # Exercise both decode and GEMM at selection time, outside graph capture.
        with torch.inference_mode():
            device = torch.device("cuda", torch.cuda.current_device())
            w = torch.zeros((256, 256), dtype=torch.int8, device=device)
            s = torch.ones(256, dtype=torch.float32, device=device)
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for rows in (1, 32):
                    x = torch.zeros((rows, 256), dtype=dtype, device=device)
                    for convrot in (False, True):
                        out = module.int8_linear(x, w, s, out_dtype=dtype, convrot=convrot)
                        if out.shape != x.shape or out.dtype != dtype or not torch.isfinite(out).all():
                            return None, "Kitchen INT8 compatibility probe failed"
            torch.cuda.synchronize(device)
        return module, "available"
    except Exception as exc:
        return None, str(exc)


def resolve_backend(selection):
    """Validate user selection before changing the active runtime or config."""
    if selection not in {value for _, value in CHOICES}:
        raise ValueError(f"Unknown INT8 kernel backend: {selection!r}")
    if selection == "disabled":
        return "pytorch", None
    if selection in ("auto", "kitchen"):
        module, reason = _probe_kitchen()
        if module is not None:
            return "kitchen", module
    module, reason = triton._probe_triton_backend()
    if module is not None:
        return "triton", module
    return "pytorch", None


def kitchen_enabled():
    return _backend == "kitchen"


def can_fuse_linear(module, x):
    """Keep qtype/backend details out of model code; MMGP still owns module calls."""
    from shared.kernels import kernel_policy
    return (not torch.compiler.is_compiling() and kernel_policy.allow_approximate() and kitchen_enabled() and _direct_cutlass
            and torch.is_inference_mode_enabled()
            and not triton._is_fake_tensor(x) and x.is_cuda and x.dtype == torch.bfloat16
            and getattr(module, '_convrot_group_size', 0) == 256
            and 256 <= module.in_features <= 16384 and module.in_features % 256 == 0
            and module.out_features % 8 == 0
            and _kitchen._convrot_fused_shared_memory_fits(x, module.in_features, 256)
            and (not getattr(module, '_mm_lora_old_forward', None)
                 or getattr(module, '_mm_lora_data', None) == {}))


def linear_with_fusion(module, x, *, input_act=None, act_weight=None, act_eps=0.0,
                       residual=None, residual_scale=None, out=None):
    # Quanto's router accepts only x. Keep its normal module/offload hooks while
    # handing the qtype the extra operands for this synchronous call only.
    previous = getattr(module, '_wangp_linear_fusion', None)
    module._wangp_linear_fusion = (input_act, act_weight, act_eps, residual, residual_scale, out)
    try:
        return module(x)
    finally:
        if previous is None:
            del module._wangp_linear_fusion
        else:
            module._wangp_linear_fusion = previous


def linear_multi(modules, x):
    """Reuse one bounded ConvRot input tile across projections, retaining MMGP hooks."""
    outputs = [x.new_empty((x.shape[0], module.out_features)) for module in modules]
    rows = max(32, (_SCRATCH_BYTES // (x.shape[-1] + 4)) // 32 * 32)
    for start in range(0, x.shape[0], rows):
        stop = min(start + rows, x.shape[0])
        tile = x[start:stop]
        q, scales = _kitchen.quantize_int8_rowwise_convrot64(tile, 256)
        for module, output in zip(modules, outputs):
            previous = getattr(module, '_wangp_prequantized_input', None)
            module._wangp_prequantized_input = (q, scales, output[start:stop])
            try:
                module(tile)
            finally:
                if previous is None:
                    del module._wangp_prequantized_input
                else:
                    module._wangp_prequantized_input = previous
        del q, scales
    return outputs


def kitchen_linear_prequantized(weight, bias, q, scales, out):
    scale = triton._prepare_weight_scale(weight._scale, weight.shape[0], out.device)
    bias_arg = (_kitchen._gemm_vector_arg(bias, out.device, out.dtype) if bias is not None
                else _kitchen._empty_cuda_tensor(out.device, out.dtype))
    wrap = _kitchen._wrap_for_dlpack
    used = _kitchen._C.cutlass_int8_dequant(
        wrap(q), wrap(weight._data), wrap(scales), wrap(scale), wrap(bias_arg), wrap(out),
        _kitchen.DTYPE_TO_CODE[out.dtype], torch.cuda.current_stream(out.device).cuda_stream)
    if not used:
        raise RuntimeError("Comfy Kitchen rejected the shared-input INT8 output tile")
    return out


def kitchen_linear_fused(input, weight, bias, input_act, act_weight, act_eps, residual, residual_scale, out=None):
    """Bound temporary quantization/output storage while retaining Kitchen fusions."""
    global _fusion_logged
    if not _fusion_logged:
        print("[INT8] Comfy Kitchen fused normalization/activation and residual linear kernels are being used.")
        _fusion_logged = True
    scale = triton._prepare_weight_scale(weight._scale, weight.shape[0], input.device)
    x = input.reshape(-1, input.shape[-1])
    n, k = weight.shape
    if out is not None and (out.shape != (*input.shape[:-1], n) or out.dtype != input.dtype
                            or out.device != input.device or not out.is_contiguous()):
        raise ValueError("Fused INT8 output must match the output shape, dtype and device and be contiguous")
    out = input.new_empty((x.shape[0], n)) if out is None else out.reshape(-1, n)
    r = residual.reshape(-1, n) if residual is not None else None
    # The raw ConvRot quantizer requires packed rows, including fused SwiGLU input.
    row_bytes = k + 4 + (x.shape[-1] * x.element_size() if not x.is_contiguous() else 0)
    rows = max(32, (_SCRATCH_BYTES // row_bytes) // 32 * 32)
    wrap = _kitchen._wrap_for_dlpack
    stream = torch.cuda.current_stream(input.device).cuda_stream
    bias_arg = (_kitchen._gemm_vector_arg(bias, input.device, input.dtype) if bias is not None
                else _kitchen._empty_cuda_tensor(input.device, input.dtype))
    act_arg = _kitchen._act_weight_arg(input_act, act_weight, input.device, input.dtype)
    residual_arg = (_kitchen._gemm_vector_arg(residual_scale, input.device, input.dtype)
                    if r is not None else None)
    for start in range(0, x.shape[0], rows):
        stop = min(start + rows, x.shape[0])
        tile = x[start:stop].contiguous()
        q = torch.empty((stop-start, k), device=input.device, dtype=torch.int8)
        qs = torch.empty((stop-start, 1), device=input.device, dtype=torch.float32)
        _kitchen._C.quantize_int8_rowwise_convrot64(
            wrap(tile), wrap(q), wrap(qs), 256, False,
            _kitchen._input_act_code(input_act), 0, wrap(act_arg), float(act_eps), stream)
        if r is not None:
            used = _kitchen._C.cutlass_int8_dequant_residual(
                wrap(q), wrap(weight._data), wrap(qs), wrap(scale), wrap(bias_arg),
                wrap(residual_arg), wrap(r[start:stop]), wrap(out[start:stop]),
                _kitchen.DTYPE_TO_CODE[input.dtype], stream)
        else:
            used = _kitchen._C.cutlass_int8_dequant(
                wrap(q), wrap(weight._data), wrap(qs), wrap(scale), wrap(bias_arg),
                wrap(out[start:stop]), _kitchen.DTYPE_TO_CODE[input.dtype], stream)
        if not used:
            raise RuntimeError("Comfy Kitchen rejected the fused SM120 INT8 output tile")
        del q, qs, tile
    return out.reshape(*input.shape[:-1], n)


def _linear_impl(x, weight, scale, bias, convrot):
    m, k = x.shape
    n = weight.shape[0]
    # Bound scratch even on the package's cuBLAS path, which has an INT32 output.
    row_bytes = k + n * (4 + x.element_size()) + 4
    if _kitchen_hip:
        # HIP may copy strided inputs and spill rotated rows plus group maxima.
        row_bytes += 2 * k * x.element_size() + 4 * (k // 256)
    rows = max(1, _SCRATCH_BYTES // row_bytes)
    if rows >= 32:
        rows = rows // 32 * 32
    if m <= rows:
        return _kitchen.int8_linear(x, weight, scale, bias=bias, out_dtype=x.dtype, convrot=convrot)
    if (_direct_cutlass and x.dtype in (torch.bfloat16, torch.float16)
            and x.is_contiguous() and weight.is_contiguous() and n % 8 == 0
            and k % 256 == 0 and 256 <= k <= 16384
            and (not convrot or _kitchen._convrot_fused_shared_memory_fits(x, k, 256))):
        return _cutlass_linear_chunked(x, weight, scale, bias, convrot)
    out = x.new_empty((m, n))
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        out[start:stop].copy_(_kitchen.int8_linear(
            x[start:stop], weight, scale, bias=bias, out_dtype=x.dtype, convrot=convrot))
    return out


def _cutlass_linear_chunked(x, weight, scale, bias, convrot):
    """Write Kitchen's CUTLASS result directly into output slices on validated SM120.

    Only quantized row tiles and their scales occupy scratch space. Avoiding a
    temporary output permits wide GEMMs without tiny, slow row chunks.
    """
    m, k = x.shape
    rows = max(32, (_SCRATCH_BYTES // (k + 4)) // 32 * 32)
    out = x.new_empty((m, weight.shape[0]))
    wrap = _kitchen._wrap_for_dlpack
    bias_arg = (_kitchen._gemm_vector_arg(bias, x.device, x.dtype) if bias is not None
                else _kitchen._empty_cuda_tensor(x.device, x.dtype))
    stream = torch.cuda.current_stream(x.device).cuda_stream
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        if convrot:
            q, scales = _kitchen.quantize_int8_rowwise_convrot64(x[start:stop], 256)
        else:
            q, scales = _kitchen.quantize_int8_rowwise(x[start:stop])
        used = _kitchen._C.cutlass_int8_dequant(
            wrap(q), wrap(weight), wrap(scales), wrap(scale), wrap(bias_arg),
            wrap(out[start:stop]), _kitchen.DTYPE_TO_CODE[x.dtype], stream)
        if not used:
            raise RuntimeError("Comfy Kitchen rejected the SM120 INT8 CUTLASS output tile")
        del q, scales
    return out


def _register_ops():
    global _ops_registered
    if _ops_registered:
        return

    @torch.library.custom_op("wan2gp_kitchen::linear", mutates_args=())
    def linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
               bias: torch.Tensor | None, convrot: bool) -> torch.Tensor:
        return _linear_impl(x, weight, scale, bias, convrot)

    @linear.register_fake
    def fake(x, weight, scale, bias, convrot):
        return x.new_empty((x.shape[0], weight.shape[0]))

    _ops_registered = True


def kitchen_linear(input, weight, bias=None, *, convrot=False):
    scale = triton._prepare_weight_scale(weight._scale, weight.shape[0], input.device)
    x = input.reshape(-1, input.shape[-1])
    if torch.compiler.is_compiling() or triton._is_fake_tensor(input):
        out = torch.ops.wan2gp_kitchen.linear(x, weight._data, scale, bias, convrot)
    else:
        out = _linear_impl(x, weight._data, scale, bias, convrot)
    return out.reshape(*input.shape[:-1], weight.shape[0])


def _quanto_forward(ctx, input, weight, bias=None):
    if (type(input) is torch.Tensor and input.is_cuda
            and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and weight._data.is_cuda and weight._data.dtype == torch.int8
            and (not _kitchen_hip or weight.shape[-1] % 16 == 0)):
        ctx.save_for_backward(input, weight)
        return kitchen_linear(input, weight, bias)
    return _original_forward(ctx, input, weight, bias)


def configure(selection, verbose_level=0, *, resolved=None):
    global _backend, _kitchen, _kitchen_hip, _original_forward, _direct_cutlass, revision
    backend, module = resolve_backend(selection) if resolved is None else resolved
    previous_backend = _backend
    if _original_forward is not None:
        from optimum.quanto.tensor.weights import qbytes
        qbytes.WeightQBytesLinearFunction.forward = staticmethod(_original_forward)
        _original_forward = None
    triton.disable_quanto_int8_kernel()
    os.environ["WAN2GP_QUANTO_INT8_KERNEL"] = "1" if backend == "triton" else "0"
    _backend, _kitchen = "pytorch", None
    _kitchen_hip = False
    _direct_cutlass = False
    if backend == "triton":
        if not triton.maybe_enable_quanto_int8_kernel(verbose_level):
            raise RuntimeError("Failed to enable Triton INT8 kernels")
    elif backend == "kitchen":
        from optimum.quanto.tensor.weights import qbytes
        _register_ops()
        _kitchen = module
        _kitchen_hip = torch.version.hip is not None
        _direct_cutlass = not _kitchen_hip and torch.cuda.get_device_capability() == (12, 0) and not module._DISABLE_CUTLASS_INT8
        _original_forward = qbytes.WeightQBytesLinearFunction.forward
        qbytes.WeightQBytesLinearFunction.forward = staticmethod(_quanto_forward)
    _backend = backend
    if backend != previous_backend:
        revision += 1
    label = {"kitchen": "Comfy Kitchen HIP" if _kitchen_hip else "Comfy Kitchen CUDA", "triton": "Triton", "pytorch": "PyTorch"}[backend]
    print(f"[INT8] Backend: {label} (setting: {selection}).")
    return backend != "pytorch"
