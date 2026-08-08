# SPDX-License-Identifier: Apache-2.0
# Bundled from ComfyUI-sol-attn v0.5.2 (commit e2fc225).
"""Triton Sol-Attn reference."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .preprocess import prepare


def _lean_do_bench(fn, quantiles=None, **kwargs):
    """do_bench with an L2-sized flush buffer instead of Triton's flat 256 MB.

    The default flush buffer lands on top of the kernel's own peak during the
    autotune sweep; the device's real L2 is enough to evict between runs.
    Patched only for the duration of the call.
    """
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    size = getattr(props, "L2_cache_size", 0) or (32 << 20)
    buffer = torch.empty(size // 4, dtype=torch.int, device="cuda")
    driver = triton.runtime.driver.active
    original = driver.get_empty_cache_for_benchmark
    driver.get_empty_cache_for_benchmark = lambda: buffer
    try:
        return triton.testing.do_bench(fn, quantiles=quantiles, **kwargs)
    finally:
        driver.get_empty_cache_for_benchmark = original


def _tuned(configs):
    return triton.autotune(
        configs=configs,
        key=["T"],
        cache_results=True,  # persist timings across restarts, not just per process
        do_bench=_lean_do_bench,
    )


_TMA_CONFIGS = [
    # Kept small: every config costs seconds of compile per new T.
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
]

_PTR_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
]


def _tma_compatible(t):
    """What TensorDescriptor accepts: dense last dim, 16-byte alignment elsewhere."""
    if t.data_ptr() % 16 != 0 or t.stride(-1) != 1:
        return False
    return all(stride * t.element_size() % 16 == 0 for stride in t.stride()[:-1])


def _validate(q, k, v, kv_splits, thresh_type):
    """Upstream validation extended to the locally tested SM120 path, plus an
    SM89 pointer fallback.

    The public upstream dispatcher allows SM90/SM100. This standalone Triton
    reference has also been validated on SM120; SM89 runs the pointer kernel
    twins (validated by forced dispatch, not on SM89 hardware). Other
    architectures remain disabled instead of being treated as implicitly
    compatible.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, T, H, 128]")
    if q.shape[1] == 0 or q.shape[3] != 128:
        raise ValueError("Sol-Attn requires T > 0 and head dimension 128")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("q, k, and v must use torch.bfloat16")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not all(_tma_compatible(x) for x in (q, k, v)):
        raise ValueError("q, k, and v must be contiguous or TMA-compatible BTHD tensors")
    if kv_splits not in (1, 2, 4):
        raise ValueError("kv_splits must be 1, 2, or 4")
    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    route_groups = ((q.shape[1] + 63) // 64 + 63) // 64
    if kv_splits > route_groups:
        raise ValueError("each KV split must contain at least one N64 route group")
    arch = torch.cuda.get_device_capability(q.device)
    if arch not in ((9, 0), (10, 0), (12, 0), (8, 9)):
        raise RuntimeError("Sol-Attn supports SM89, SM90, SM100, and SM120")
    return arch


BLOCK = 64
GROUP = 32


@_tuned(_TMA_CONFIGS)
@triton.jit
def _forward(
    q_desc,
    k_desc,
    v_desc,
    kc_desc,
    vc_desc,
    threshold,
    o_desc,
    scale,
    sink_start,
    sink_end,
    sink_q_start,
    sink_q_end,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    q_start = q_block * BLOCK_SIZE
    q = q_desc.load([batch, q_start, head, 0]).reshape([BLOCK_SIZE, D])
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
    route_threshold = tl.load(
        threshold + (batch * NT + q_block) * H + head
    )
    q_in_sink = (q_block >= sink_q_start) & (q_block < sink_q_end)

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = kc_desc.load(
            [batch, group_start, head, 0]
        ).reshape([GROUP_SIZE, D])
        vc = vc_desc.load(
            [batch, group_start, head, v_tile * BV]
        ).reshape([GROUP_SIZE, BV])
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        # Sink: conditioning KV blocks stay exact for every query, and query
        # rows inside the sink range attend everything exactly.
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        routed = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
            | sink_kv
        ) & valid
        exact = tl.where(q_in_sink, valid, routed)

        approximate = valid & ~exact
        approximate_scores = tl.where(
            approximate[None, :], scores, -float("inf")
        )
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        approximate_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(approximate_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            approximate_probability.to(vc.dtype), vc
        )
        lengths = tl.where(
            block_indices == NT - 1, tail_length, BLOCK_SIZE
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            approximate_probability * lengths[None, :], axis=1
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset, GROUP_SIZE, exact_offsets
            )
            kv_start = block * BLOCK_SIZE
            k = k_desc.load(
                [batch, kv_start, head, 0]
            ).reshape([BLOCK_SIZE, D])
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(
                (kv_start + token_offsets)[None, :] < T,
                0.0,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            v = v_desc.load(
                [batch, kv_start, head, v_tile * BV]
            ).reshape([BLOCK_SIZE, BV])
            output = output * alpha[:, None] + tl.dot(
                exact_probability.to(v.dtype), v
            )
            row_max = new_max

    o_desc.store(
        [batch, q_start, head, v_tile * BV],
        (output / row_sum[:, None]).to(tl.bfloat16)[None, :, None, :],
    )


@_tuned(_TMA_CONFIGS)
@triton.jit
def _forward_int8(
    q_desc,
    v_desc,
    kc_desc,
    vc_desc,
    qi_ptr,
    ki_ptr,
    q_scale,
    k_scale,
    threshold,
    o_desc,
    scale,
    sink_start,
    sink_end,
    sink_q_start,
    sink_q_end,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """INT8 q/k exact path. Routing and the approximate path stay bf16; only the
    exact-block residual scores use the int8 dot, with the block-mean term added
    back from the routing scores already computed for the group. int8 tiles load
    through pointers (faster than descriptor emulation for 8-bit tiles)."""
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    d_offsets = tl.arange(0, D)
    q_start = q_block * BLOCK_SIZE
    q = q_desc.load([batch, q_start, head, 0]).reshape([BLOCK_SIZE, D])
    q_rows = q_start + token_offsets
    q_valid = q_rows < T
    q8 = tl.load(
        qi_ptr + ((batch * T + q_rows[:, None]) * H + head) * D + d_offsets[None, :],
        mask=q_valid[:, None],
        other=0,
    )
    qs = tl.load(
        q_scale + (batch * T + q_rows) * H + head,
        mask=q_valid,
        other=1.0,
    )
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
    route_threshold = tl.load(
        threshold + (batch * NT + q_block) * H + head
    )
    q_in_sink = (q_block >= sink_q_start) & (q_block < sink_q_end)

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = kc_desc.load(
            [batch, group_start, head, 0]
        ).reshape([GROUP_SIZE, D])
        vc = vc_desc.load(
            [batch, group_start, head, v_tile * BV]
        ).reshape([GROUP_SIZE, BV])
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        routed = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
            | sink_kv
        ) & valid
        exact = tl.where(q_in_sink, valid, routed)

        approximate = valid & ~exact
        approximate_scores = tl.where(
            approximate[None, :], scores, -float("inf")
        )
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        approximate_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(approximate_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            approximate_probability.to(vc.dtype), vc
        )
        lengths = tl.where(
            block_indices == NT - 1, tail_length, BLOCK_SIZE
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            approximate_probability * lengths[None, :], axis=1
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset, GROUP_SIZE, exact_offsets
            )
            kv_start = block * BLOCK_SIZE
            k_rows = kv_start + token_offsets
            k8 = tl.load(
                ki_ptr + ((batch * T + k_rows[:, None]) * H + head) * D + d_offsets[None, :],
                mask=(k_rows < T)[:, None],
                other=0,
            )
            ks = tl.load(k_scale + (batch * T + k_rows) * H + head, mask=k_rows < T, other=1.0)
            s32 = tl.dot(q8, k8.T)
            approx_col = tl.sum(
                tl.where(group_offsets[None, :] == offset, scores, 0.0), axis=1
            )
            exact_scores = (
                s32.to(tl.float32) * (qs[:, None] * ks[None, :]) * scale_log2
                + approx_col[:, None]
            )
            exact_scores += tl.where(
                (k_rows < T)[None, :],
                0.0,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            v = v_desc.load(
                [batch, kv_start, head, v_tile * BV]
            ).reshape([BLOCK_SIZE, BV])
            output = output * alpha[:, None] + tl.dot(
                exact_probability.to(v.dtype), v
            )
            row_max = new_max

    o_desc.store(
        [batch, q_start, head, v_tile * BV],
        (output / row_sum[:, None]).to(tl.bfloat16)[None, :, None, :],
    )


@_tuned(_PTR_CONFIGS)
@triton.jit
def _forward_ptr(
    q_ptr, k_ptr, v_ptr, kc_ptr, vc_ptr, threshold, o_ptr,
    scale,
    sink_start,
    sink_end,
    sink_q_start,
    sink_q_end,
    T,
    sq_b, sq_t, sq_h,
    sk_b, sk_t, sk_h,
    sv_b, sv_t, sv_h,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Pointer twin of _forward for pre-TMA arches (SM89). Masked loads with
    explicit strides; q/k/v keep their layout, so interleaved qkv views need
    no contiguous() copy."""
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    d_offsets = tl.arange(0, D)
    bv_offsets = v_tile * BV + tl.arange(0, BV)
    q_start = q_block * BLOCK_SIZE
    q_rows = q_start + token_offsets
    q = tl.load(
        q_ptr + batch * sq_b + q_rows[:, None].to(tl.int64) * sq_t + head * sq_h + d_offsets[None, :],
        mask=(q_rows < T)[:, None],
        other=0.0,
    )
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
    route_threshold = tl.load(
        threshold + (batch * NT + q_block) * H + head
    )
    q_in_sink = (q_block >= sink_q_start) & (q_block < sink_q_end)

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = tl.load(
            kc_ptr + ((batch * NT + block_indices[:, None]) * H + head) * D + d_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        vc = tl.load(
            vc_ptr + ((batch * NT + block_indices[:, None]) * H + head) * D + bv_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        routed = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
            | sink_kv
        ) & valid
        exact = tl.where(q_in_sink, valid, routed)

        approximate = valid & ~exact
        approximate_scores = tl.where(
            approximate[None, :], scores, -float("inf")
        )
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        approximate_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(approximate_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            approximate_probability.to(vc.dtype), vc
        )
        lengths = tl.where(
            block_indices == NT - 1, tail_length, BLOCK_SIZE
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            approximate_probability * lengths[None, :], axis=1
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset, GROUP_SIZE, exact_offsets
            )
            kv_start = block * BLOCK_SIZE
            k_rows = kv_start + token_offsets
            k = tl.load(
                k_ptr + batch * sk_b + k_rows[:, None].to(tl.int64) * sk_t + head * sk_h + d_offsets[None, :],
                mask=(k_rows < T)[:, None],
                other=0.0,
            )
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(
                (k_rows < T)[None, :],
                0.0,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            v = tl.load(
                v_ptr + batch * sv_b + k_rows[:, None].to(tl.int64) * sv_t + head * sv_h + bv_offsets[None, :],
                mask=(k_rows < T)[:, None],
                other=0.0,
            )
            output = output * alpha[:, None] + tl.dot(
                exact_probability.to(v.dtype), v
            )
            row_max = new_max

    tl.store(
        o_ptr + ((batch * T + q_rows[:, None]) * H + head) * D + bv_offsets[None, :],
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=(q_rows < T)[:, None],
    )


@_tuned(_PTR_CONFIGS)
@triton.jit
def _forward_int8_ptr(
    q_ptr, v_ptr, kc_ptr, vc_ptr, qi_ptr, ki_ptr, q_scale, k_scale, threshold, o_ptr,
    scale,
    sink_start,
    sink_end,
    sink_q_start,
    sink_q_end,
    T,
    sq_b, sq_t, sq_h,
    sv_b, sv_t, sv_h,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Pointer twin of _forward_int8 for pre-TMA arches (SM89)."""
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    d_offsets = tl.arange(0, D)
    bv_offsets = v_tile * BV + tl.arange(0, BV)
    q_start = q_block * BLOCK_SIZE
    q_rows = q_start + token_offsets
    q_valid = q_rows < T
    q = tl.load(
        q_ptr + batch * sq_b + q_rows[:, None].to(tl.int64) * sq_t + head * sq_h + d_offsets[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )
    qi = tl.load(
        qi_ptr + ((batch * T + q_rows[:, None]) * H + head) * D + d_offsets[None, :],
        mask=q_valid[:, None],
        other=0,
    )
    qs = tl.load(q_scale + (batch * T + q_rows) * H + head, mask=q_valid, other=1.0)
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
    route_threshold = tl.load(threshold + (batch * NT + q_block) * H + head)
    q_in_sink = (q_block >= sink_q_start) & (q_block < sink_q_end)

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = tl.load(
            kc_ptr + ((batch * NT + block_indices[:, None]) * H + head) * D + d_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        vc = tl.load(
            vc_ptr + ((batch * NT + block_indices[:, None]) * H + head) * D + bv_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        routed = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
            | sink_kv
        ) & valid
        exact = tl.where(q_in_sink, valid, routed)

        approximate = valid & ~exact
        approximate_scores = tl.where(
            approximate[None, :], scores, -float("inf")
        )
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        approximate_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(approximate_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            approximate_probability.to(vc.dtype), vc
        )
        lengths = tl.where(
            block_indices == NT - 1, tail_length, BLOCK_SIZE
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            approximate_probability * lengths[None, :], axis=1
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset, GROUP_SIZE, exact_offsets
            )
            kv_start = block * BLOCK_SIZE
            k_rows = kv_start + token_offsets
            k_valid = k_rows < T
            ki = tl.load(
                ki_ptr + ((batch * T + k_rows[:, None]) * H + head) * D + d_offsets[None, :],
                mask=k_valid[:, None],
                other=0,
            )
            ks = tl.load(k_scale + (batch * T + k_rows) * H + head, mask=k_valid, other=1.0)
            s32 = tl.dot(qi, ki.T, out_dtype=tl.int32)
            approx_col = tl.sum(
                tl.where(group_offsets[None, :] == offset, scores, 0.0), axis=1
            )
            exact_scores = (
                s32.to(tl.float32) * (qs[:, None] * ks[None, :]) * scale_log2
                + approx_col[:, None]
            )
            exact_scores += tl.where(k_valid[None, :], 0.0, -float("inf"))
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            v = tl.load(
                v_ptr + batch * sv_b + k_rows[:, None].to(tl.int64) * sv_t + head * sv_h + bv_offsets[None, :],
                mask=k_valid[:, None],
                other=0.0,
            )
            output = output * alpha[:, None] + tl.dot(
                exact_probability.to(v.dtype), v
            )
            row_max = new_max

    tl.store(
        o_ptr + ((batch * T + q_rows[:, None]) * H + head) * D + bv_offsets[None, :],
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=q_valid[:, None],
    )


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    int8_qk: bool = False,
    sink_blocks: tuple = (0, 0),
    sink_q: tuple = (0, 0),
) -> torch.Tensor:
    """Run the readable Triton reference on BTHD inputs.

    ``sink_blocks`` = (start, end) KV block range forced exact for every query
    (e.g. MiniMax H3's packed conditioning rows); ``sink_q`` = query block range
    that attends everything exactly.
    """

    arch = _validate(q, k, v, 1, thresh_type)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    tau = float(tau)
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    sinks = (int(sink_blocks[0]), int(sink_blocks[1]), int(sink_q[0]), int(sink_q[1]))

    if arch[0] < 9:
        # SM89 pointer path: masked loads, q/k/v keep their strides.
        if int8_qk:
            kc, vc, threshold, q8, q_scale, k8, k_scale = prepare(
                q, k, v, scale=scale, tau=tau, thresh_type=thresh_type, int8_qk=True
            )
            _forward_int8_ptr[(1, blocks, batch * heads)](
                q, v, kc, vc, q8, k8, q_scale, k_scale, threshold, output,
                scale,
                *sinks,
                tokens,
                q.stride(0), q.stride(1), q.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                H=heads,
                D=head_dim,
                NT=blocks,
                BV=head_dim,
                BLOCK_SIZE=BLOCK,
                GROUP_SIZE=GROUP,
            )
            return output
        kc, vc, threshold = prepare(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type)
        _forward_ptr[(1, blocks, batch * heads)](
            q, k, v, kc, vc, threshold, output,
            scale,
            *sinks,
            tokens,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            H=heads,
            D=head_dim,
            NT=blocks,
            BV=head_dim,
            BLOCK_SIZE=BLOCK,
            GROUP_SIZE=GROUP,
        )
        return output

    block_shape = [1, BLOCK, 1, head_dim]
    summary_shape = [1, GROUP, 1, head_dim]
    if int8_qk:
        kc, vc, threshold, q8, q_scale, k8, k_scale = prepare(
            q, k, v, scale=scale, tau=tau, thresh_type=thresh_type, int8_qk=True
        )
        _forward_int8[(1, blocks, batch * heads)](
            TensorDescriptor.from_tensor(q, block_shape),
            TensorDescriptor.from_tensor(v, block_shape),
            TensorDescriptor.from_tensor(kc, summary_shape),
            TensorDescriptor.from_tensor(vc, summary_shape),
            q8,
            k8,
            q_scale,
            k_scale,
            threshold,
            TensorDescriptor.from_tensor(output, block_shape),
            scale,
            *sinks,
            tokens,
            heads,
            head_dim,
            blocks,
            head_dim,
            BLOCK,
            GROUP,
        )
        return output
    kc, vc, threshold = prepare(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type)
    _forward[(1, blocks, batch * heads)](
        TensorDescriptor.from_tensor(q, block_shape),
        TensorDescriptor.from_tensor(k, block_shape),
        TensorDescriptor.from_tensor(v, block_shape),
        TensorDescriptor.from_tensor(kc, summary_shape),
        TensorDescriptor.from_tensor(vc, summary_shape),
        threshold,
        TensorDescriptor.from_tensor(output, block_shape),
        scale,
        *sinks,
        tokens,
        heads,
        head_dim,
        blocks,
        head_dim,
        BLOCK,
        GROUP,
    )
    return output


__all__ = ["sol_attn"]
