# SPDX-License-Identifier: BSD-3-Clause
# Derived from SageAttention 1.0.6 and Veda-on-ComfyUI; see NOTICE and SAGE_LICENSE.
"""Veda block-sparse INT8 QK / FP16 PV attention, with partial-tile masking."""
import torch
import triton
import triton.language as tl
from shared.kernels import kernel_policy
from shared.kernels.triton_tuner import LaunchTuner

TILE = 128
KEY_BLOCK = 64
LOG2E = 1.4426950408889634
_tuner = LaunchTuner("h3_veda_attention", 1, "H3 Veda")
_launches = ((4, 3), (4, 2), (8, 3))

@triton.jit
def _quantize_kernel(X, Xq, Scale, stride_xn, stride_xh, stride_sh,
                     pre_scale, D: tl.constexpr, BLK: tl.constexpr):
    """One INT8 scale per (block of BLK slots, head), upstream's rounding."""
    blk = tl.program_id(0)
    head = tl.program_id(1)
    offs_n = blk * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    offset = head * stride_xh + offs_n[:, None] * stride_xn + offs_d[None, :]
    x = tl.load(X + offset).to(tl.float32) * pre_scale
    scale = tl.max(tl.abs(x)) / 127.0
    # A block that is all padding has no scale; 1.0 keeps the zeros zero.
    scale = tl.where(scale > 0.0, scale, 1.0)
    quantized = x / scale
    quantized += 0.5 * tl.where(quantized >= 0, 1, -1)
    tl.store(Xq + offset, quantized.to(tl.int8))
    tl.store(Scale + head * stride_sh + blk, scale)


def quantize(x: torch.Tensor, block: int,
             pre_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, H, D] -> (int8 of the same shape, [H, N // block] fp32 scales).

    `pre_scale` is multiplied in before quantising; the caller folds the
    softmax scale into Q that way, exactly as upstream does, so the kernel
    never applies it.
    """
    slots, heads, dim = x.shape
    out = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((heads, slots // block), dtype=torch.float32,
                        device=x.device)
    _quantize_kernel[(slots // block, heads)](
        x, out, scale, x.stride(0), x.stride(1), scale.stride(0),
        pre_scale=pre_scale, D=dim, BLK=block, num_warps=4)
    return out, scale


@triton.jit
def _attention_kernel(Q, K, V, Q_scale, K_scale, Index, Count, Valid, Out,
                      stride_n, stride_h, stride_vn, stride_vh,
                      stride_qs, stride_ks, stride_ih, stride_iq,
                      n_q_tiles, D: tl.constexpr, BLK: tl.constexpr,
                      BLOCK_M: tl.constexpr):
    query_tile = tl.program_id(0)
    head = tl.program_id(1)
    offs_m = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLK)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + head * stride_h + offs_m[:, None] * stride_n
                + offs_d[None, :])
    q_scale = tl.load(Q_scale + head * stride_qs + query_tile)

    # Finite, not -inf: a key block that is entirely padding leaves m_i
    # untouched, and -inf would make (-inf) - (-inf) a NaN on the way in.
    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    count = tl.load(Count + head * n_q_tiles + query_tile) * (BLOCK_M // BLK)
    index_row = Index + head * stride_ih + query_tile * stride_iq
    for i in range(0, count):
        tile = tl.load(index_row + i // (BLOCK_M // BLK))
        sub = i % (BLOCK_M // BLK)
        block = tile * (BLOCK_M // BLK) + sub
        start = block * BLK
        k = tl.load(K + head * stride_h + (start + offs_n)[None, :] * stride_n
                    + offs_d[:, None])
        k_scale = tl.load(K_scale + head * stride_ks + block)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        # Real rows are a prefix of every block, so one comparison masks
        # all the padding. Upstream has no equivalent: it loads keys past
        # the end as zeros, which quietly lets them into the softmax.
        qk = tl.where(offs_n[None, :] < tl.load(Valid + tile) - sub * BLK, qk,
                      -float('inf'))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + head * stride_vh + (start + offs_n)[:, None] * stride_vn
                    + offs_d[None, :])
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
        m_i = m_ij

    # A query tile with nothing kept would divide by zero; selection always
    # keeps the diagonal, so this only guards against a malformed mask.
    l_i = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_i[:, None]
    tl.store(Out + head * stride_h + offs_m[:, None] * stride_n
             + offs_d[None, :], acc.to(Out.type.element_ty))


def attend(q, k, v, index, count, valid_count):
    slots, heads, dim = q.shape
    q_int8, q_scale = quantize(q, TILE, pre_scale=dim ** -0.5 * LOG2E)
    k_int8, k_scale = quantize(k, KEY_BLOCK)
    v16 = v.to(torch.float16)
    out = torch.empty_like(q)
    def launch(destination, config):
        _attention_kernel[(slots // TILE, heads)](
            q_int8, k_int8, v16, q_scale, k_scale, index, count, valid_count, destination,
            q.stride(0), q.stride(1), v16.stride(0), v16.stride(1),
            q_scale.stride(0), k_scale.stride(0), index.stride(0), index.stride(1),
            n_q_tiles=slots // TILE, D=dim, BLK=KEY_BLOCK, BLOCK_M=TILE,
            num_warps=config[0], num_stages=config[1])

    def prepare_scratch():
        scratch = torch.empty_like(out)
        return lambda config: launch(scratch, config)

    config = _launches[0]
    if kernel_policy.allow_approximate():
        config = _tuner.config("attention", (dim, triton.next_power_of_2(slots // TILE), heads), q.device,
                               _launches, prepare_scratch, tunable=True)
    launch(out, config)
    return out
