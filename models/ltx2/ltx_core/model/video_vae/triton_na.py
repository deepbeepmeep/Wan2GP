# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN001, ANN202, PLR0912, PLR0913, PLR0915
"""Triton 3D neighborhood attention using NATTEN ``na3d`` semantics.

Vendored from the official LTX-2.5 implementation, which in turn vendors the
Comfy Kitchen kernel. One program handles a run of queries along W with online
softmax and fp32 accumulation, without materializing the attention scores.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_NEG_INF = tl.constexpr(-3.0e38)


@triton.jit
def _na3d_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    t_size,
    h_size,
    w_size,
    num_heads,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    scale,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    causal_t: tl.constexpr,
    causal_h: tl.constexpr,
    causal_w: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_th = tl.program_id(1)
    pid_bn = tl.program_id(2)

    t_q = pid_th // h_size
    h_q = pid_th % h_size
    base = (pid_bn // num_heads) * s_b + (pid_bn % num_heads) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = q_ptr + base + t_q * s_t + h_q * s_h + w_off[:, None] * s_w + d_off[None, :]
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    if causal_t:
        t_lo = tl.maximum(t_q - kt + 1, 0)
        t_hi = t_q + 1
    else:
        t_lo = tl.minimum(tl.maximum(t_q - kt // 2, 0), t_size - kt)
        t_hi = t_lo + kt
    if causal_h:
        h_lo = tl.maximum(h_q - kh + 1, 0)
        h_hi = h_q + 1
    else:
        h_lo = tl.minimum(tl.maximum(h_q - kh // 2, 0), h_size - kh)
        h_hi = h_lo + kh

    w_q = tl.where(w_valid, w_off, w_size - 1)
    if causal_w:
        w_start = tl.maximum(w_q - kw + 1, 0)
        w_end = w_q + 1
        blk_first = tl.minimum(pid_w * block_q, w_size - 1)
        blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.maximum(blk_first - kw + 1, 0)
        w_hi = blk_last + 1
    else:
        w_start = tl.minimum(tl.maximum(w_q - kw // 2, 0), w_size - kw)
        w_end = w_start + kw
        blk_first = tl.minimum(pid_w * block_q, w_size - 1)
        blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.minimum(tl.maximum(blk_first - kw // 2, 0), w_size - kw)
        w_hi = tl.minimum(tl.maximum(blk_last - kw // 2, 0), w_size - kw) + kw

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    for tk in range(t_lo, t_hi):
        for hk in range(h_lo, h_hi):
            plane = base + tk * s_t + hk * s_h
            for wk0 in range(w_lo, w_hi, block_k):
                wk = wk0 + tl.arange(0, block_k)
                kmask = wk < w_hi
                kv_ptrs = plane + wk[:, None] * s_w + d_off[None, :]
                kv_mask = kmask[:, None] & d_mask[None, :]
                k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                if is_fp32:
                    scores = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * scale
                else:
                    scores = tl.dot(q_blk, tl.trans(k_blk)) * scale
                visible = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None]) & kmask[None, :]
                scores = tl.where(visible, scores, _NEG_INF)
                m_new = tl.maximum(m_i, tl.max(scores, 1))
                alpha = tl.exp(m_i - m_new)
                probabilities = tl.exp(scores - m_new[:, None])
                l_i = l_i * alpha + tl.sum(probabilities, 1)
                v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                if is_fp32:
                    acc = acc * alpha[:, None] + tl.dot(probabilities, v_blk, input_precision="ieee")
                else:
                    acc = acc * alpha[:, None] + tl.dot(probabilities.to(v_blk.dtype), v_blk)
                m_i = m_new

    out = acc / tl.maximum(l_i, 1e-30)[:, None]
    out_ptrs = out_ptr + base + t_q * s_t + h_q * s_h + w_off[:, None] * s_w + d_off[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])


def na3d(qkv_list: list[torch.Tensor], kernel_size: tuple[int, int, int], scale: float = 1.0) -> torch.Tensor:
    """Run neighborhood attention over ``(B, T, H, W, heads, head_dim)`` tensors."""
    q, k, v = qkv_list
    qkv_list.clear()
    batch, t, h, w, num_heads, head_dim = q.shape
    kt, kh, kw = (min(kernel, dim) for kernel, dim in zip(kernel_size, (t, h, w), strict=True))
    out = q

    head_dim_pad = max(16, triton.next_power_of_2(head_dim))
    block_q = 16
    block_k = max(16, min(32, triton.next_power_of_2(min(w, block_q + kw))))
    grid = triton.cdiv(w, block_q), t * h, batch * num_heads
    _na3d_kernel[grid](
        q,
        k,
        v,
        out,
        t,
        h,
        w,
        num_heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        scale,
        kt=kt,
        kh=kh,
        kw=kw,
        causal_t=False,
        causal_h=False,
        causal_w=False,
        hd=head_dim,
        hd_pad=head_dim_pad,
        block_q=block_q,
        block_k=block_k,
        is_fp32=q.dtype == torch.float32,
        num_warps=4,
    )
    return out
