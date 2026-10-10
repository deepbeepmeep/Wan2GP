"""Optional INT8 backends shared by Quanto and ConvRot linear layers."""
from __future__ import annotations

import importlib
import os
from fnmatch import fnmatchcase
from types import MethodType

import torch

from shared.kernels import quanto_int8_inject as triton


CHOICES = [("Disabled (PyTorch)", "disabled"), ("Auto", "auto"),
           ("Triton", "triton"), ("Comfy Kitchen Kernels", "kitchen")]
_backend = "pytorch"
revision = 0
_kitchen = None
_kitchen_hip = False
_direct_cutlass = False
_sm120_cutlass = False
_original_forward = None
_ops_registered = False
_fusion_logged = False
_compile_cache_root = None
_compile_cache_backend = None
_wide_convrot_triton = None
_blockwise_triton = None
_blockwise_required = False
_kitchen_dlpack_export = None  # (Kitchen module, its own DLPack exporter) while the direct exporter replaces it
# Bump when changes to INT8 custom operators invalidate compiled graphs.
_COMPILE_CACHE_VERSION = 3
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


def _export_dlpack(tensor):
    # Kitchen's DLPack export without the per-call checks of Tensor.__dlpack__, which always hold for the inference tensors it gets
    # (strided, on the current device, no stream synchronization): an INT8 linear exports a dozen tensors, so they add up on fast steps
    return torch._C._to_dlpack(tensor.detach() if tensor.requires_grad else tensor)


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


def require_blockwise_int8(model, reason, module_patterns=("*",)):
    """Protect a model or subtree whose plain INT8 linears need finer activation scales.

    Install before MMGP profiling/LoRA hooks. Other qtypes and the user's Disabled
    or Triton selections retain their normal forwards; Kitchen uses Triton for
    these linears, or native Quanto when Triton is unavailable.
    """
    global _blockwise_required, _blockwise_triton
    from optimum.quanto import qint8

    count = 0
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) and getattr(module, "weight_qtype", None) == qint8
                and any(fnmatchcase(name, pattern) for pattern in module_patterns)
                and not hasattr(module, "_int8_precision_forward")):
            module._int8_precision_forward = module.forward
            module.forward = MethodType(_blockwise_module_forward, module)
            count += 1
    if count:
        _blockwise_required = True
        if kitchen_enabled():
            _blockwise_triton, unavailable = triton._probe_triton_backend()
            backend = "Triton" if _blockwise_triton is not None else f"PyTorch ({unavailable})"
            print(f"[INT8] Precision Rule: {backend} for {count} linear layers - {reason}.")


def _blockwise_linear_impl(x, weight, scale, bias):
    if _blockwise_triton is None:
        out = torch.ops.quanto.qbytes_mm(x, weight, scale)
    else:
        out = _blockwise_triton.fused_quant_scaled_mm(x, weight, scale, out_dtype=x.dtype)
    if bias is not None:
        out.add_(bias)
    return out


def _blockwise_module_forward(module, input):
    if (_backend != "kitchen" or type(input) is not torch.Tensor or not input.is_cuda
            or input.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return module._int8_precision_forward(input)
    weight = module.qweight
    x = input.reshape(-1, input.shape[-1])
    if torch.compiler.is_compiling() or triton._is_fake_tensor(input):
        out = torch.ops.wan2gp_kitchen.blockwise_linear(x, weight._data, weight._scale, module.bias)
    else:
        out = _blockwise_linear_impl(x, weight._data, weight._scale, module.bias)
    return out.reshape(*input.shape[:-1], weight.shape[0])


def _weight_scale(weight, n, device):
    # float32 scales of an INT8 weight, made once per weight object: converting them at each call launches a kernel, which costs host
    # bound steps (image models run about a thousand INT8 linears per step)
    if torch.compiler.is_compiling():
        return triton._prepare_weight_scale(weight._scale, n, device)
    scale = weight.__dict__.get("_kitchen_scale", None)
    if scale is None:
        scale = weight._kitchen_scale = triton._prepare_weight_scale(weight._scale, n, device)
    return scale


def _wide_convrot_linear(input, weight, bias=None):
    """Use Triton for ConvRot rows beyond Kitchen's graph-safe CUTLASS width."""
    from shared.qtypes.int8_convrot import _rotate_activation

    rotated = _rotate_activation(input, 256)
    x = rotated.reshape(-1, rotated.shape[-1])
    scale = _weight_scale(weight, weight.shape[0], input.device)
    out = _wide_convrot_triton.fused_quant_scaled_mm(x, weight._data, scale, out_dtype=input.dtype)
    if bias is not None:
        out += bias
    return out.reshape(*input.shape[:-1], weight.shape[0])


def can_fuse_linear(module, x, dtypes=(torch.bfloat16,)):
    """Keep qtype/backend details out of model code; MMGP still owns module calls.
    Callers widen `dtypes` only for activation dtypes validated with their model."""
    from shared.kernels import kernel_policy
    return (not torch.compiler.is_compiling() and kernel_policy.allow_approximate() and kitchen_enabled() and _sm120_cutlass
            and torch.is_inference_mode_enabled()
            and not triton._is_fake_tensor(x) and x.is_cuda and x.dtype in dtypes
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
    scale = _weight_scale(weight, weight._data.shape[0], out.device)
    bias_arg = (_kitchen._gemm_vector_arg(bias, out.device, out.dtype) if bias is not None
                else _kitchen._empty_cuda_tensor(out.device, out.dtype))
    wrap = _kitchen._wrap_for_dlpack
    used = _kitchen._C.cutlass_int8_dequant(
        wrap(q), wrap(weight._data), wrap(scales), wrap(scale), wrap(bias_arg), wrap(out),
        _kitchen.DTYPE_TO_CODE[out.dtype], torch._C._cuda_getCurrentRawStream(out.device.index))
    if not used:
        raise RuntimeError("Comfy Kitchen rejected the shared-input INT8 output tile")
    return out


def kitchen_linear_fused(input, weight, bias, input_act, act_weight, act_eps, residual, residual_scale, out=None):
    """Bound temporary quantization/output storage while retaining Kitchen fusions."""
    global _fusion_logged
    if not _fusion_logged:
        print("[INT8] Comfy Kitchen fused normalization/activation and residual linear kernels are being used.")
        _fusion_logged = True
    scale = _weight_scale(weight, weight._data.shape[0], input.device)
    x = input.reshape(-1, input.shape[-1])
    n, k = weight._data.shape
    if out is not None and (out.shape != (*input.shape[:-1], n) or out.dtype != input.dtype
                            or out.device != input.device or not out.is_contiguous()):
        raise ValueError("Fused INT8 output must match the output shape, dtype and device and be contiguous")
    out = input.new_empty((x.shape[0], n)) if out is None else out.reshape(-1, n)
    r = residual.reshape(-1, n) if residual is not None else None
    # The raw ConvRot quantizer requires packed rows, including fused SwiGLU input.
    row_bytes = k + 4 + (x.shape[-1] * x.element_size() if not x.is_contiguous() else 0)
    rows = max(32, (_SCRATCH_BYTES // row_bytes) // 32 * 32)
    wrap = _kitchen._wrap_for_dlpack
    stream = torch._C._cuda_getCurrentRawStream(input.device.index)
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
    """Write Kitchen's CUTLASS result directly into output slices on supported CUDA GPUs.

    Only quantized row tiles and their scales occupy scratch space. Avoiding a
    temporary output permits wide GEMMs without tiny, slow row chunks.
    """
    m, k = x.shape
    rows = max(32, (_SCRATCH_BYTES // (k + 4)) // 32 * 32)
    out = x.new_empty((m, weight.shape[0]))
    wrap = _kitchen._wrap_for_dlpack
    bias_arg = (_kitchen._gemm_vector_arg(bias, x.device, x.dtype) if bias is not None
                else _kitchen._empty_cuda_tensor(x.device, x.dtype))
    stream = torch._C._cuda_getCurrentRawStream(x.device.index)
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
            raise RuntimeError("Comfy Kitchen rejected the INT8 CUTLASS output tile")
        del q, scales
    return out


@torch.inference_mode()
def _probe_direct_cutlass():
    """Check the actual output-buffer kernel at backend setup, never during capture."""
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        x = torch.arange(65 * 256, device=device, dtype=torch.float32).remainder_(251).sub_(125).div_(128).reshape(65, 256)
        weight = (torch.arange(264 * 256, device=device, dtype=torch.int32) % 255 - 127).to(torch.int8).reshape(264, 256)
        scale = torch.full((264,), 1 / 127, device=device, dtype=torch.float32)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = x.to(dtype)
            for bias in (None, scale.to(dtype)):
                for convrot in (False, True):
                    expected = _kitchen.int8_linear(inputs, weight, scale, bias=bias, out_dtype=dtype, convrot=convrot)
                    actual = _cutlass_linear_chunked(inputs, weight, scale, bias, convrot)
                    if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                        raise RuntimeError("output-buffer kernel did not match Kitchen INT8")
        return True
    except RuntimeError as exc:
        print(f"[INT8] Direct Output Tiles Unavailable: {exc}. Using existing Kitchen batching.")
        return False


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

    @torch.library.custom_op("wan2gp_kitchen::blockwise_linear", mutates_args=())
    def blockwise_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                         bias: torch.Tensor | None) -> torch.Tensor:
        return _blockwise_linear_impl(x, weight, scale, bias)

    @blockwise_linear.register_fake
    def blockwise_fake(x, weight, scale, bias):
        return x.new_empty((x.shape[0], weight.shape[0]))

    _ops_registered = True


def kitchen_linear(input, weight, bias=None, *, convrot=False):
    if convrot and _sm120_cutlass and input.shape[-1] > 16384:
        if (_wide_convrot_triton is not None and not torch.compiler.is_compiling()
                and not triton._is_fake_tensor(input)):
            return _wide_convrot_linear(input, weight, bias)
        # The wide cuBLAS path cannot be captured on validated SM120. Keep a
        # graph-safe Quanto fallback when Triton is unavailable.
        from shared.qtypes.int8_convrot import _rotate_activation
        return torch.nn.functional.linear(_rotate_activation(input, 256), weight, bias)
    n = weight._data.shape[0]  # the plain INT8 data: reading the quantized weight's shape costs a Python dispatch
    scale = _weight_scale(weight, n, input.device)
    x = input.reshape(-1, input.shape[-1])
    if torch.compiler.is_compiling() or triton._is_fake_tensor(input):
        out = torch.ops.wan2gp_kitchen.linear(x, weight._data, scale, bias, convrot)
    else:
        out = _linear_impl(x, weight._data, scale, bias, convrot)
    return out.reshape(*input.shape[:-1], n)


def _quanto_forward(ctx, input, weight, bias=None):
    if (type(input) is torch.Tensor and input.is_cuda
            and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and weight._data.is_cuda and weight._data.dtype == torch.int8
            and (input.shape[-1] <= 16384 or not _sm120_cutlass)
            and weight._data.shape[-1] % (16 if _kitchen_hip else 4) == 0 and weight._data.shape[0] % 4 == 0):  # the kitchen GEMM rejects other widths
        ctx.save_for_backward(input, weight)
        return kitchen_linear(input, weight, bias)
    return _original_forward(ctx, input, weight, bias)


class _KitchenRowProjection:
    """mmgp row projection (offload.linear_rows) of the INT8 ConvRot and Quanto INT8 layers that kitchen_linear computes with
    _cutlass_linear_chunked: the input is quantized once, each range of output rows is the same CUTLASS GEMM on a view of the weight
    rows, so the rows equal those of the layer's own output. The quantized input is as large as half the input it replaces: callers
    release the input (prepare_linear_input takes it)."""

    @staticmethod
    def supports(module, x):
        qtype = getattr(module, "weight_qtype", None)
        if (_kitchen is None or not _direct_cutlass or qtype is None or torch.compiler.is_compiling() or not x.is_cuda
                or x.dtype not in (torch.bfloat16, torch.float16) or not x.is_contiguous()):
            return False
        convrot = qtype.name == "qint8_convrot"
        if convrot and (getattr(module, "_convrot_group_size", 0) != 256 or not kitchen_enabled()):
            return False
        if not convrot and (qtype.name != "qint8" or getattr(module, "_router_forward_impl", None) is not None):
            return False
        weight = module.qweight
        data = weight._data
        (m, k), n = x.shape, data.shape[0]
        rows = max(1, _SCRATCH_BYTES // (k + n * (4 + x.element_size()) + 4))  # _linear_impl's single call threshold
        rows = rows // 32 * 32 if rows >= 32 else rows
        return (m > rows and data.dtype == torch.int8 and data.is_cuda and data.is_contiguous() and weight.dtype == x.dtype
                and n % 8 == 0 and k % 256 == 0 and 256 <= k <= 16384 and (not convrot or _kitchen._convrot_fused_shared_memory_fits(x, k, 256)))

    @staticmethod
    def key(module):
        return "kitchen_int8_convrot" if module.weight_qtype.name == "qint8_convrot" else "kitchen_int8"

    @staticmethod
    def prepare(module, x):
        (m, k), convrot = x.shape, module.weight_qtype.name == "qint8_convrot"
        rows = max(32, (_SCRATCH_BYTES // (k + 4)) // 32 * 32)
        q, scales = torch.empty((m, k), dtype=torch.int8, device=x.device), None
        for start in range(0, m, rows):
            stop = min(start + rows, m)
            tile_q, tile_scales = _kitchen.quantize_int8_rowwise_convrot64(x[start:stop], 256) if convrot else _kitchen.quantize_int8_rowwise(x[start:stop])
            if scales is None:
                scales = torch.empty((m, *tile_scales.shape[1:]), dtype=tile_scales.dtype, device=x.device)
            q[start:stop].copy_(tile_q)
            scales[start:stop].copy_(tile_scales)
            del tile_q, tile_scales
        return q, scales, x.dtype

    @staticmethod
    def rows(module, prepared, start, stop):
        q, scales, dtype = prepared
        weight, bias = module.qweight, module.bias
        (m, k), device = q.shape, q.device
        data = weight._data[start:stop]
        scale = _weight_scale(weight, weight._data.shape[0], device)[start:stop]
        bias_arg = (_kitchen._gemm_vector_arg(bias[start:stop], device, dtype) if bias is not None
                    else _kitchen._empty_cuda_tensor(device, dtype))
        out = torch.empty((m, stop - start), dtype=dtype, device=device)
        wrap = _kitchen._wrap_for_dlpack
        stream = torch._C._cuda_getCurrentRawStream(device.index)
        tile = max(32, (_SCRATCH_BYTES // (k + 4)) // 32 * 32)
        for row in range(0, m, tile):
            end = min(row + tile, m)
            if not _kitchen._C.cutlass_int8_dequant(wrap(q[row:end]), wrap(data), wrap(scales[row:end]), wrap(scale), wrap(bias_arg),
                                                     wrap(out[row:end]), _kitchen.DTYPE_TO_CODE[dtype], stream):
                raise RuntimeError("Comfy Kitchen rejected the INT8 CUTLASS row range")
        return out


_KITCHEN_ROW_PROJECTION = _KitchenRowProjection()


def configure(selection, verbose_level=0, *, resolved=None):
    global _backend, _kitchen, _kitchen_hip, _original_forward, _direct_cutlass, _sm120_cutlass, _wide_convrot_triton, revision
    global _blockwise_triton, _kitchen_dlpack_export
    backend, module = resolve_backend(selection) if resolved is None else resolved
    previous_backend = _backend
    if _kitchen_dlpack_export is not None:
        _kitchen_dlpack_export[0]._wrap_for_dlpack = _kitchen_dlpack_export[1]
        _kitchen_dlpack_export = None
    if _original_forward is not None:
        from optimum.quanto.tensor.weights import qbytes
        qbytes.WeightQBytesLinearFunction.forward = staticmethod(_original_forward)
        _original_forward = None
    triton.disable_quanto_int8_kernel()
    os.environ["WAN2GP_QUANTO_INT8_KERNEL"] = "1" if backend == "triton" else "0"
    _backend, _kitchen = "pytorch", None
    _kitchen_hip = False
    _direct_cutlass = False
    _sm120_cutlass = False
    _wide_convrot_triton = None
    if backend == "triton":
        if not triton.maybe_enable_quanto_int8_kernel(verbose_level):
            raise RuntimeError("Failed to enable Triton INT8 kernels")
    elif backend == "kitchen":
        from optimum.quanto.tensor.weights import qbytes
        _register_ops()
        _kitchen = module
        _kitchen_hip = torch.version.hip is not None
        if not _kitchen_hip:
            _kitchen_dlpack_export = (module, module._wrap_for_dlpack)
            module._wrap_for_dlpack = _export_dlpack
        if not _kitchen_hip and not module._DISABLE_CUTLASS_INT8:
            capability = torch.cuda.get_device_capability()
            _sm120_cutlass = capability == (12, 0)
            _direct_cutlass = _sm120_cutlass or (capability >= (8, 0) and _probe_direct_cutlass())
        if _sm120_cutlass:
            _wide_convrot_triton, _ = triton._probe_triton_backend()
        if _blockwise_required:
            _blockwise_triton, reason = triton._probe_triton_backend()
            label = "Triton" if _blockwise_triton is not None else f"PyTorch ({reason})"
            print(f"[INT8] Precision Rules: {label} for protected linear layers.")
        _original_forward = qbytes.WeightQBytesLinearFunction.forward
        qbytes.WeightQBytesLinearFunction.forward = staticmethod(_quanto_forward)
    _backend = backend
    from mmgp import offload
    (offload.register_row_projection if backend == "kitchen" else offload.unregister_row_projection)(_KITCHEN_ROW_PROJECTION)
    if backend != previous_backend:
        revision += 1
    label = {"kitchen": "Comfy Kitchen HIP" if _kitchen_hip else "Comfy Kitchen CUDA", "triton": "Triton", "pytorch": "PyTorch"}[backend]
    print(f"[INT8] Backend: {label} (setting: {selection}).")
    return backend != "pytorch"
