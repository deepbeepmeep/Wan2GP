# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Optional fused RMSNorms of the vLLM engine: one kernel per norm instead of about ten elementwise/reduction launches, which
dominate a decoding step's GPU work at one token. The PyTorch path stays the reference; the row sums use another order, so these
kernels follow kernel_policy.allow_approximate()."""
import functools

import torch
from shared.kernels import kernel_policy

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None


@functools.cache
def _supported(device):
    return triton is not None and torch.cuda.get_device_capability(device) >= (8, 0)


def available(x):
    return x.is_cuda and x.is_contiguous() and kernel_policy.allow_approximate() and _supported(x.device)


if triton is not None:
    @triton.jit(do_not_specialize=["rows"])
    def _zero_centered_rms_norm(X, W, Y, rows, eps, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(X + row * D + cols, mask, other=0.0).to(tl.float32)
        x = x * libdevice.rsqrt(tl.sum(x * x, 0) * (1.0 / D) + eps)
        y = x * (1.0 + tl.load(W + cols, mask, other=0.0).to(tl.float32))
        tl.store(Y + row * D + cols, y.to(Y.dtype.element_ty), mask)

    @triton.jit(do_not_specialize=["rows"])
    def _gated_rms_norm(X, G, W, Y, rows, eps, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(X + row * D + cols, mask, other=0.0).to(tl.float32)
        # Rounding points of GatedRMSNorm: normalized in the input dtype, scaled in the input dtype, gated in float32.
        x = (x * libdevice.rsqrt(tl.sum(x * x, 0) * (1.0 / D) + eps)).to(Y.dtype.element_ty).to(tl.float32)
        x = (tl.load(W + cols, mask, other=0.0).to(tl.float32) * x).to(Y.dtype.element_ty).to(tl.float32)
        g = tl.load(G + row * D + cols, mask, other=0.0).to(tl.float32)
        y = x * tl.div_rn(g, 1.0 + libdevice.exp(-g))
        tl.store(Y + row * D + cols, y.to(Y.dtype.element_ty), mask)

    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()


def zero_centered_rms_norm(x, weight, eps):
    """(x / rms(x)) * (1 + weight), computed in float32, in x's dtype."""
    out = torch.empty_like(x)
    dim = x.shape[-1]
    rows = x.numel() // dim
    _zero_centered_rms_norm[(rows,)](x, weight, out, rows, eps, dim, triton.next_power_of_2(dim), enable_fp_fusion=False)
    return out


def gated_rms_norm(x, gate, weight, eps):
    """weight * rmsnorm(x) * silu(gate), with GatedRMSNorm's rounding points."""
    out = torch.empty_like(x)
    dim = x.shape[-1]
    rows = x.numel() // dim
    _gated_rms_norm[(rows,)](x, gate, weight, out, rows, eps, dim, triton.next_power_of_2(dim), enable_fp_fusion=False)
    return out
