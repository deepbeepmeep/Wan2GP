"""Per-tensor FP8 activation quantization in two kernels, bit-identical to the PyTorch path of shared.qtypes.scaled_fp8."""
import numpy as np
import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import is_fake

PARTIALS = 1024  # absmax partial results, one per program of the first kernel
BLOCK = 4096
_TL_FP8 = {torch.float8_e4m3fn: tl.float8e4nv, torch.float8_e5m2: tl.float8e5}
_TL_FLOAT = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _absmax_partials_kernel(x_ptr, partial_ptr, n, BLOCK: tl.constexpr, PARTS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(pid * BLOCK, n, PARTS * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc = tl.maximum(acc, tl.abs(tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)))
    tl.store(partial_ptr + pid, tl.max(acc, axis=0))


@triton.jit
def _quantize_kernel(x_ptr, q_ptr, partial_ptr, scale_ptr, n, inv_maxv, maxv, minv, X_DTYPE: tl.constexpr, Q_DTYPE: tl.constexpr, BLOCK: tl.constexpr, PARTS: tl.constexpr):
    pid = tl.program_id(0)
    absmax = tl.max(tl.load(partial_ptr + tl.arange(0, PARTS)), axis=0)
    scale = tl.where(absmax > 0, absmax * inv_maxv, 1.0)  # torch divides a tensor by a cpu scalar through the scalar's float32 reciprocal
    if pid == 0:
        tl.store(scale_ptr, scale)
    scale_x = scale.to(X_DTYPE).to(tl.float32)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.math.div_rn(x, scale_x).to(X_DTYPE).to(tl.float32)  # x / scale computed in float32 and rounded to x's dtype, as torch does
    tl.store(q_ptr + offs, tl.minimum(tl.maximum(y, minv), maxv).to(Q_DTYPE), mask=mask)


def supports(x, fp8_dtype):
    return (x.is_cuda and torch.version.cuda is not None and not torch.compiler.is_compiling() and not is_fake(x)
            and x.dtype in _TL_FLOAT and fp8_dtype in _TL_FP8 and x.is_contiguous() and 0 < x.numel() < 2**31)


def quantize_activation(x, fp8_dtype, minv, maxv):
    n = x.numel()
    partials = torch.empty(PARTIALS, device=x.device, dtype=torch.float32)
    _absmax_partials_kernel[(PARTIALS,)](x, partials, n, BLOCK=BLOCK, PARTS=PARTIALS)
    q = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)
    scale = torch.empty((), device=x.device, dtype=torch.float32)
    inv_maxv = float(np.float32(1.0) / np.float32(maxv))
    _quantize_kernel[(triton.cdiv(n, BLOCK),)](x, q, partials, scale, n, inv_maxv, float(maxv), float(minv), X_DTYPE=_TL_FLOAT[x.dtype], Q_DTYPE=_TL_FP8[fp8_dtype], BLOCK=BLOCK, PARTS=PARTIALS)
    return q, scale


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
