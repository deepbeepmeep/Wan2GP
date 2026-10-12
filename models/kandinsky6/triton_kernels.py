# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fused Kandinsky 6 denoiser elementwise passes: FP32 arithmetic in registers, BF16 hidden states in memory. Each kernel follows
the operation order of the PyTorch path; the row reductions of the norms (mean, variance, mean of squares) are summed in another order:
rounding-level differences.

One binary per kernel and model width: the token and head counts and the strides are not specialized, so a new resolution, video
length or Attention Head Split level compiles nothing. Launch configurations start from a static choice and are tuned once per kernel,
model width and GPU, outside CUDA graph recording and compilation, on rows of a real call (scratch copies of at most 8192 rows); the
winners are kept in a cache file shared by the sessions, so later sessions compile only them."""
import functools

import torch
import triton
import triton.language as tl

from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_kandinsky6_kernels", 1, "Kandinsky 6")
_TUNE_ROWS = 8192  # rows of a call timed per candidate (at least 2048 rows to tune: smaller calls use the static configuration)


def _config(name, dim, rows, device, candidates, scratch):
    return _TUNER.config(name, dim, device, candidates, lambda: scratch(min(rows, _TUNE_ROWS)), rows >= 2048)


@functools.cache
def supported(device):
    """CUDA GPUs with native BF16 (compute capability 8.0+); other devices use the PyTorch path."""
    return device.type == "cuda" and torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0)


@triton.jit(do_not_specialize=["TOKENS"])
def _scale_shift_norm_kernel(X, OUT, SCALE, SHIFT, TOKENS, EPS, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    batch = row // TOKENS
    cols = tl.arange(0, BLOCK)
    valid = cols < D
    x = tl.load(X + row * D + cols, valid, 0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    centered = tl.where(valid, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / D + EPS)
    scale = tl.load(SCALE + batch * D + cols, valid, 0.0).to(tl.float32) + 1.0
    shift = tl.load(SHIFT + batch * D + cols, valid, 0.0).to(tl.float32)
    tl.store(OUT + row * D + cols, (centered * rstd) * scale + shift, valid)


@triton.jit(do_not_specialize=["N", "TOKENS"])
def _gate_add_kernel(X, OUT, GATE, N, TOKENS, D: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    gate = tl.load(GATE + (offsets // (TOKENS * D)) * D + offsets % D, valid, 0.0).to(tl.float32)
    product = tl.load(OUT + offsets, valid, 0.0).to(tl.float32) * gate
    tl.store(X + offsets, product + tl.load(X + offsets, valid, 0.0).to(tl.float32), valid)


@triton.jit(do_not_specialize=["HEADS", "SXB", "SXT", "SXH", "TOKENS"])
def _norm_rope_kernel(X, WEIGHT, COS, SIN, HEADS, SXB, SXT, SXH, TOKENS, EPS, D: tl.constexpr, BLOCK_H: tl.constexpr, HAS_ROPE: tl.constexpr):
    """BLOCK_H heads of one token per program, each head row normalized over its head_dim."""
    row = tl.program_id(0).to(tl.int64)
    token = row % TOKENS
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    dims = tl.arange(0, D)[None, :]
    valid = heads < HEADS
    offsets = (row // TOKENS) * SXB + token * SXT + heads * SXH + dims
    x = tl.load(X + offsets, valid, 0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1)[:, None] / D + EPS)
    # the norm output is rounded to the activation dtype before the rotation, as in the PyTorch path
    y = ((x * rstd) * tl.load(WEIGHT + dims).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    if HAS_ROPE:
        y0, y1 = tl.split(tl.reshape(y, (BLOCK_H, D // 2, 2)))  # interleaved pairs, loaded contiguously
        pairs = tl.arange(0, D // 2)[None, :]
        cos = tl.load(COS + token * (D // 2) + pairs)
        sin = tl.load(SIN + token * (D // 2) + pairs)
        y = tl.reshape(tl.join(y0 * cos - y1 * sin, y0 * sin + y1 * cos), (BLOCK_H, D))
    tl.store(X + offsets, y, valid)


def scale_shift_norm(x, scale, shift):
    """(layer_norm(x) * (scale + 1) + shift) of x (batch, tokens, dim) BF16 with FP32 scale/shift (batch, dim): a new BF16 tensor."""
    batch, tokens, dim = x.shape
    scale, shift = scale.float().contiguous(), shift.float().contiguous()
    block = triton.next_power_of_2(dim)

    def launch(source, out, config):
        _scale_shift_norm_kernel[(source.shape[0] * source.shape[1],)](source, out, scale, shift, source.shape[1], 1e-5, dim, block,
                                                                       num_warps=config[0], enable_fp_fusion=False)

    def scratch(rows):
        source, out = x[:1, :rows], torch.empty_like(x[:1, :rows])
        return lambda config: launch(source, out, config)

    out = torch.empty_like(x)
    launch(x, out, _config("scale_shift_norm", dim, tokens, x.device, ((8,), (4,), (2,), (16,)), scratch))
    return out


def gate_add_(x, out, gate):
    """x += gate * out in place, x and out (batch, tokens, dim) BF16, gate (batch, dim) FP32."""
    batch, tokens, dim = x.shape
    out, gate = out.contiguous(), gate.float().contiguous()

    def launch(target, source, config):
        block, warps = config
        _gate_add_kernel[(triton.cdiv(target.numel(), block),)](target, source, gate, target.numel(), target.shape[1], dim, block,
                                                                num_warps=warps, enable_fp_fusion=False)

    def scratch(rows):
        target, source = x[:1, :rows].clone(), out[:1, :rows]
        return lambda config: launch(target, source, config)

    launch(x, out, _config("gate_add", dim, tokens, x.device, ((2048, 8), (1024, 4), (2048, 4), (4096, 8), (8192, 16)), scratch))
    torch.autograd.graph.increment_version(x)  # an in-place update, as PyTorch records it (MMGP's CUDA graphs copy back mutated arguments)
    return x


def norm_rope_(x, weight, rope, eps):
    """RMSNorm (FP32 weight) then RoPE ((cos, sin) of shape (tokens, 1, head_dim / 2), FP32) of x (batch, tokens, heads, head_dim) BF16 in
    place; x may be a view of a group of heads."""
    batch, tokens, heads, dim = x.shape
    cos, sin = (None, None) if rope is None else rope

    def launch(target, config):
        block_heads, warps = config
        _norm_rope_kernel[(target.shape[0] * target.shape[1], triton.cdiv(heads, block_heads))](
            target, weight, cos, sin, heads, target.stride(0), target.stride(1), target.stride(2), target.shape[1], eps, dim, block_heads,
            rope is not None, num_warps=warps, enable_fp_fusion=False)

    def scratch(rows):
        target = x[:1, :rows].clone()
        return lambda config: launch(target, config)

    launch(x, _config("norm_rope" if rope is not None else "norm", dim, tokens, x.device, ((8, 2), (4, 1), (4, 2), (8, 4), (16, 4), (32, 4)), scratch))
    torch.autograd.graph.increment_version(x)
    return x


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
