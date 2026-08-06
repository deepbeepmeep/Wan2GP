# SPDX-License-Identifier: Apache-2.0
# Bundled from ComfyUI-sol-attn v0.5.2 (commit e2fc225).
"""INT8 quantization for the optional SageAttention-style q/k path.

k is quantized per token after centering by its 64-token block mean, so only
the small per-block residual goes through the int8 dot; the mean term is the
routing score the forward kernel already computes exactly in bf16 and is added
back there. q is quantized per token, fused with the diag routing-threshold
computation so q is read once. Loads use plain pointers with explicit strides
so the kernels also run on pre-Hopper arches.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK_SIZE = 64


@triton.jit
def _quantize_k_kernel(
    k_ptr,
    kc_ptr,
    k8_ptr,
    k_scale,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    NB,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    tile = tl.load(
        k_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(kc_ptr + ((batch * NB + block) * H + head) * D + d).to(tl.float32)
    centered = tl.where(valid[:, None], tile - mean[None, :], 0.0)
    amax = tl.max(tl.abs(centered), axis=1)
    scale = tl.maximum(amax / 127.0, 1e-8)
    k8 = libdevice.rint(centered / scale[:, None]).to(tl.int8)
    tl.store(
        k8_ptr + ((batch * T + rows[:, None]) * H + head) * D + d[None, :],
        k8,
        mask=valid[:, None],
    )
    tl.store(k_scale + (batch * T + rows) * H + head, scale, mask=valid)


@triton.jit
def _q_quant_threshold_kernel(
    q_ptr,
    kc_mean_ptr,
    kc_var_ptr,
    q8_ptr,
    q_scale,
    thr_ptr,
    softmax_scale,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TAU,
):
    """Per-token q quantization fused with the diag routing threshold: one q load."""
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = q_block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    tile = tl.load(
        q_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    amax = tl.max(tl.abs(tile), axis=1)
    scale = tl.maximum(amax / 127.0, 1e-8)
    q8 = libdevice.rint(tile / scale[:, None]).to(tl.int8)
    tl.store(
        q8_ptr + ((batch * T + rows[:, None]) * H + head) * D + d[None, :],
        q8,
        mask=valid[:, None],
    )
    tl.store(q_scale + (batch * T + rows) * H + head, scale, mask=valid)

    centroid = tl.sum(tile, axis=0) / q_len
    mean_kc = tl.load(kc_mean_ptr + batch_head * D + d)
    var_kc = tl.load(kc_var_ptr + batch_head * D + d)
    log2_scale = softmax_scale * 1.4426950408889634
    mean = tl.sum(centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(centroid * centroid * var_kc, axis=0) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)
    tl.store(thr_ptr + (batch * N + q_block) * H + head, mean + TAU * std)


def quantize_k(k, kc):
    """k [B, T, H, D] bf16; kc [B, NB, H, D] bf16 block means.

    Returns k_int8 [B, T, H, D] (per-block-mean residual), k_scale [B, T, H] fp32.
    """
    batch, tokens, heads, head_dim = k.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    k8 = torch.empty(k.shape, device=k.device, dtype=torch.int8)
    k_scale = torch.empty((batch, tokens, heads), device=k.device, dtype=torch.float32)
    _quantize_k_kernel[(blocks, batch * heads)](
        k,
        kc,
        k8,
        k_scale,
        tokens,
        k.stride(0), k.stride(1), k.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        num_warps=4,
    )
    return k8, k_scale


def quantize_q_with_threshold(q, kc_mean, kc_var, *, scale, tau):
    """q [B, T, H, D] bf16; kc stats [B, H, D] fp32 on the smoothed summaries.

    Returns q_int8 [B, T, H, D], q_scale [B, T, H] fp32, threshold [B, NB, H] fp32.
    """
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    q8 = torch.empty(q.shape, device=q.device, dtype=torch.int8)
    q_scale = torch.empty((batch, tokens, heads), device=q.device, dtype=torch.float32)
    threshold = torch.empty((batch, blocks, heads), device=q.device, dtype=torch.float32)
    _q_quant_threshold_kernel[(blocks, batch * heads)](
        q,
        kc_mean,
        kc_var,
        q8,
        q_scale,
        threshold,
        scale,
        tokens,
        q.stride(0), q.stride(1), q.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tau,
        num_warps=4,
    )
    return q8, q_scale, threshold


__all__ = ["quantize_k", "quantize_q_with_threshold"]
