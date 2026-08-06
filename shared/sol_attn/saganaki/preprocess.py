# SPDX-License-Identifier: Apache-2.0
# Bundled from ComfyUI-sol-attn v0.5.2 (commit e2fc225).
"""Block summaries and routing thresholds shared by both CuTe kernels.

Loads use plain pointers with explicit strides: TensorDescriptor emulation on
pre-Hopper arches costs several times more, while on Hopper and newer the
difference is negligible for these small streaming kernels. The compute-heavy
forward kernels keep native TMA descriptors on SM90+.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .quant import quantize_k, quantize_q_with_threshold

BLOCK_SIZE = 64
HEAD_DIM = 128
THRESHOLD_GROUP_SIZE = 64


@triton.jit
def _reduce_kc_kernel(
    k_ptr,
    kc,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    block_len = tl.minimum(BLOCK, T - block * BLOCK)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    values = tl.load(
        k_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + offsets[None, :],
        mask=(rows < T)[:, None] & (offsets < D)[None, :],
        other=0.0,
    )
    summary = tl.sum(values, axis=0) / block_len
    tl.store(
        kc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.jit
def _reduce_vc_kernel(
    v_ptr,
    vc,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    values = tl.load(
        v_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + offsets[None, :],
        mask=(rows < T)[:, None] & (offsets < D)[None, :],
        other=0.0,
    )
    summary = tl.sum(values, axis=0)
    tl.store(
        vc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.jit
def _reduce_kc_stats_kernel(
    kc_ptr,
    kc_mean,
    kc_var_diag,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    TILE_D: tl.constexpr,
    GROUP: tl.constexpr,
):
    d_tile, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    block_offsets = tl.max_contiguous(tl.arange(0, GROUP), GROUP)
    d_offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    total = tl.zeros((TILE_D,), dtype=tl.float32)
    total_sq = tl.zeros((TILE_D,), dtype=tl.float32)
    count = tl.full((), 0.0, dtype=tl.float32)
    for start in range(0, N, GROUP):
        rows = start + block_offsets
        values = tl.load(
            kc_ptr + ((batch * N + rows[:, None]) * H + head) * D + d_offsets[None, :],
            mask=(rows < N)[:, None],
            other=0.0,
        ).to(tl.float32)
        total += tl.sum(values, axis=0)
        total_sq += tl.sum(values * values, axis=0)
        count += tl.sum((rows < N).to(tl.float32), axis=0)
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    valid_d = d_offsets < D
    tl.store(
        kc_mean + batch_head * D + d_offsets,
        mean,
        mask=valid_d,
    )
    tl.store(
        kc_var_diag + batch_head * D + d_offsets,
        variance,
        mask=valid_d,
    )


@triton.jit
def _diag_threshold_kernel(
    q_ptr,
    kc_mean,
    kc_var_diag,
    global_threshold,
    softmax_scale,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
    TAU,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_start = q_block * BLOCK
    q_len = tl.minimum(BLOCK, T - q_start).to(tl.float32)
    d_offsets = tl.arange(0, TILE_D)
    valid_d = d_offsets < D
    rows = q_start + tl.arange(0, BLOCK)
    q_values = tl.load(
        q_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d_offsets[None, :],
        mask=(rows < T)[:, None] & valid_d[None, :],
        other=0.0,
    )
    q_centroid = tl.sum(q_values.to(tl.float32), axis=0) / q_len
    mean_kc = tl.load(
        kc_mean + batch_head * D + d_offsets,
        mask=valid_d,
        other=0.0,
    )
    var_kc = tl.load(
        kc_var_diag + batch_head * D + d_offsets,
        mask=valid_d,
        other=0.0,
    )
    log2_scale = softmax_scale * 1.4426950408889634
    mean = tl.sum(q_centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(
        q_centroid * q_centroid * var_kc, axis=0
    ) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)
    tl.store(
        global_threshold + (batch * N + q_block) * H + head,
        mean + TAU * std,
    )


@triton.jit
def _pool_query_kernel(
    q_ptr,
    q_bar,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_start = q_block * BLOCK
    q_len = tl.minimum(BLOCK, T - q_start).to(tl.float32)
    offsets = tl.arange(0, TILE_D)
    rows = q_start + tl.arange(0, BLOCK)
    values = tl.load(
        q_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + offsets[None, :],
        mask=(rows < T)[:, None] & (offsets < D)[None, :],
        other=0.0,
    )
    centroid = tl.sum(values.to(tl.float32), axis=0) / q_len
    tl.store(
        q_bar + (batch_head * N + q_block) * D + offsets,
        centroid,
        mask=offsets < D,
    )


@triton.jit
def _exact_fused_threshold_kernel(
    q_bar,
    kc_mean,
    kc_second_moment,
    global_threshold,
    softmax_scale,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_D: tl.constexpr,
    TAU,
):
    row_tile, batch_head = tl.program_id(0), tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets = tl.arange(0, TILE_D)
    valid_rows = rows < N
    valid_d = offsets < D

    q_centroid = tl.load(
        q_bar + (batch_head * N + rows[:, None]) * D + offsets[None, :],
        mask=valid_rows[:, None] & valid_d[None, :],
        other=0.0,
    )
    mean_kc = tl.load(
        kc_mean + batch_head * D + offsets,
        mask=valid_d,
        other=0.0,
    )
    second_moment = tl.load(
        kc_second_moment
        + batch_head * D * D
        + offsets[:, None] * D
        + offsets[None, :],
        mask=valid_d[:, None] & valid_d[None, :],
        other=0.0,
    )

    raw_mean = tl.sum(q_centroid.to(tl.float32) * mean_kc[None, :], axis=1)
    projected = tl.dot(
        q_centroid,
        second_moment,
        out_dtype=tl.float32,
    )
    raw_second_moment = tl.sum(
        projected * q_centroid.to(tl.float32),
        axis=1,
    )
    log2_scale = softmax_scale * 1.4426950408889634
    mean = raw_mean * log2_scale
    variance = tl.maximum(
        raw_second_moment - raw_mean * raw_mean,
        0.0,
    ) * (log2_scale * log2_scale)
    threshold = mean + TAU * tl.sqrt(variance + 1.0e-6)
    batch, head = batch_head // H, batch_head % H
    tl.store(
        global_threshold + (batch * N + rows) * H + head,
        threshold,
        mask=valid_rows,
    )


def _reduce_kv(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, head_dim = k.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    kc = torch.empty(
        (batch, blocks, heads, head_dim),
        device=k.device,
        dtype=torch.bfloat16,
    )
    vc = torch.empty_like(kc)
    grid = (triton.cdiv(head_dim, tile_d), blocks, batch * heads)
    _reduce_kc_kernel[grid](
        k,
        kc,
        tokens,
        k.stride(0), k.stride(1), k.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
        num_warps=4,
    )
    _reduce_vc_kernel[grid](
        v,
        vc,
        tokens,
        v.stride(0), v.stride(1), v.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
        num_warps=4,
    )
    return kc, vc


def _compute_diag_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
) -> torch.Tensor:
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    kc_mean = torch.empty(
        (batch, heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    kc_var_diag = torch.empty_like(kc_mean)
    global_threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    _reduce_kc_stats_kernel[(triton.cdiv(head_dim, tile_d), batch * heads)](
        kc,
        kc_mean,
        kc_var_diag,
        heads,
        blocks,
        head_dim,
        tile_d,
        THRESHOLD_GROUP_SIZE,
        num_warps=4,
    )
    _diag_threshold_kernel[(blocks, batch * heads)](
        q,
        kc_mean,
        kc_var_diag,
        global_threshold,
        scale,
        tokens,
        q.stride(0), q.stride(1), q.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
        tau,
        num_warps=4,
    )
    return global_threshold


def _compute_exact_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
) -> torch.Tensor:
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    batch_heads = batch * heads
    kc_bh = kc.permute(0, 2, 1, 3)
    kc_mean = kc_bh.mean(dim=2, dtype=torch.float32)
    kc_second_moment = torch.matmul(
        kc_bh.transpose(-1, -2),
        kc_bh,
    )
    kc_second_moment.div_(blocks)
    q_bar = torch.empty(
        (batch_heads, blocks, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    global_threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    _pool_query_kernel[(blocks, batch_heads)](
        q,
        q_bar,
        tokens,
        q.stride(0), q.stride(1), q.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
        num_warps=4,
    )
    block_m = 64
    _exact_fused_threshold_kernel[(triton.cdiv(blocks, block_m), batch_heads)](
        q_bar,
        kc_mean,
        kc_second_moment,
        global_threshold,
        scale,
        heads,
        blocks,
        head_dim,
        block_m,
        tile_d,
        tau,
        num_warps=4,
    )
    return global_threshold


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    thresh_type: str = "diag",
    int8_qk: bool = False,
) -> tuple:
    kc, vc = _reduce_kv(k, v)
    if not int8_qk:
        if thresh_type == "exact":
            threshold = _compute_exact_threshold(q, kc, tau=tau, scale=scale)
        else:
            threshold = _compute_diag_threshold(q, kc, tau=tau, scale=scale)
        return kc, vc, threshold

    # int8 path: quantize the per-block-mean residual of k per token (the mean
    # term is the routing score the forward computes exactly in bf16), and fuse
    # the q quantization with the diag threshold so q is read once.
    ki, ks = quantize_k(k, kc)
    kv = kc.float().permute(0, 2, 1, 3)  # [B, H, NB, D]
    stat_mean = kv.mean(dim=2).contiguous()
    stat_var = (kv - stat_mean.unsqueeze(2)).pow(2).mean(dim=2).contiguous()
    qi, qs, threshold = quantize_q_with_threshold(q, stat_mean, stat_var, scale=scale, tau=tau)
    if thresh_type == "exact":
        threshold = _compute_exact_threshold(q, kc, tau=tau, scale=scale)
    return kc, vc, threshold, qi, qs, ki, ks


__all__ = ["prepare"]
