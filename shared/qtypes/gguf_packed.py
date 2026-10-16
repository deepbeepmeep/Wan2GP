# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fused GGUF multiplication with arbitrary restored GGUF row boundaries.

Packed bytes are indexed by flattened element position. Decoded tiles exist only
inside the kernel; no dense weight tensor or padded activation is allocated.
"""
from typing import List, Optional
import torch
import triton
import triton.language as tl
from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner('wangp_gguf_flat_rows', 2, 'GGUF Packed')
_CONFIGS = ((64, 128, 32, 4, 2), (64, 128, 64, 8, 2),
            (64, 256, 32, 8, 2), (128, 128, 32, 8, 2))


@triton.jit
def _q45_tile(W, positions, valid, Q5: tl.constexpr):
    block = positions // 256
    lane = positions % 256
    group = lane // 32
    size: tl.constexpr = 176 if Q5 else 144
    base = block * size
    d = tl.load((W + base).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
    dm = tl.load((W + base + 2).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
    a = tl.load(W + base + 4 + group % 4, valid, 0).to(tl.int32)
    b = tl.load(W + base + 8 + group % 4, valid, 0).to(tl.int32)
    c = tl.load(W + base + 12 + group % 4, valid, 0).to(tl.int32)
    scale = tl.where(group < 4, a & 63, (c & 15) | ((a >> 2) & 48))
    minimum = tl.where(group < 4, b & 63, (c >> 4) | ((b >> 2) & 48))
    qs_offset: tl.constexpr = 48 if Q5 else 16
    qs = tl.load(W + base + qs_offset + lane // 64 * 32 + lane % 32, valid, 0).to(tl.int32)
    q = (qs >> ((group % 2) * 4)) & 15
    if Q5:
        high = tl.load(W + base + 16 + lane % 32, valid, 0).to(tl.int32)
        q = q | (((high >> group) & 1) << 4)
    # Match the existing FP32 dequantization followed by the compute-dtype cast.
    return (d * scale) * q - dm * minimum


@triton.jit(do_not_specialize=['M', 'stride_m', 'stride_k'])
def _packed_mm(X, W, Bias, Y, M, stride_m, stride_k,
               N: tl.constexpr, K: tl.constexpr, QTYPE: tl.constexpr, BIAS: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        x = tl.load(X + m[:, None] * stride_m + k[None, :] * stride_k,
                    (m[:, None] < M) & (k[None, :] < K), 0).to(Y.dtype.element_ty)
        if K % 16 == 0:
            w = _weight_tile_16(W, n, start * (BK // 16) + tl.arange(0, BK // 16), N, K, QTYPE)
        else:
            w = _weight_tile(W, n[:, None] * K + k[None, :],
                             (n[:, None] < N) & (k[None, :] < K), QTYPE)
        w = w.to(Y.dtype.element_ty)
        acc = tl.dot(x, tl.trans(w), acc, input_precision='ieee')
    if BIAS:
        acc += tl.load(Bias + n, n < N, 0).to(Y.dtype.element_ty).to(tl.float32)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _q45_tile_16(W, rows, chunks, N: tl.constexpr, K: tl.constexpr, Q5: tl.constexpr):
    # Every contiguous group of 16 shares its GGUF scale and minimum, even when
    # a logical row begins halfway through a 32-value quantization sub-block.
    position = rows[:, None] * K + chunks[None, :] * 16
    valid = (rows[:, None] < N) & (chunks[None, :] * 16 < K)
    lane = position % 256
    group = lane // 32
    size: tl.constexpr = 176 if Q5 else 144
    base = position // 256 * size
    d = tl.load((W + base).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
    dm = tl.load((W + base + 2).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
    a = tl.load(W + base + 4 + group % 4, valid, 0).to(tl.int32)
    b = tl.load(W + base + 8 + group % 4, valid, 0).to(tl.int32)
    c = tl.load(W + base + 12 + group % 4, valid, 0).to(tl.int32)
    scale = tl.where(group < 4, a & 63, (c & 15) | ((a >> 2) & 48))
    minimum = tl.where(group < 4, b & 63, (c >> 4) | ((b >> 2) & 48))
    qs_offset: tl.constexpr = 48 if Q5 else 16
    offsets = (base + qs_offset + lane // 64 * 32 + lane % 32)[:, :, None] + tl.arange(0, 16)[None, None, :]
    qs = tl.load(W + offsets, valid[:, :, None], 0).to(tl.int32)
    q = (qs >> ((group % 2) * 4)[:, :, None]) & 15
    if Q5:
        offsets = (base + 16 + lane % 32)[:, :, None] + tl.arange(0, 16)[None, None, :]
        high = tl.load(W + offsets, valid[:, :, None], 0).to(tl.int32)
        q = q | (((high >> group[:, :, None]) & 1) << 4)
    value = (d * scale)[:, :, None] * q - (dm * minimum)[:, :, None]
    return value.reshape(rows.shape[0], chunks.shape[0] * 16)


@triton.jit
def _k236_scale(W, base, lane, valid, QTYPE: tl.constexpr):
    group = lane // 16
    if QTYPE == 'Q2_K':
        d = tl.load((W + base + 80).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
        dm = tl.load((W + base + 82).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
        s = tl.load(W + base + group, valid, 0).to(tl.int32)
        scale, minimum = d * (s & 15), dm * (s >> 4)
    elif QTYPE == 'Q3_K':
        d = tl.load((W + base + 108).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
        low = tl.load(W + base + 96 + group % 8, valid, 0).to(tl.int32)
        high = tl.load(W + base + 104 + group % 4, valid, 0).to(tl.int32)
        s = ((low >> (group // 8 * 4)) & 15) | (((high >> (group // 4 * 2)) & 3) << 4)
        scale, minimum = d * (s - 32), tl.full(base.shape, 0, tl.float32)
    else:
        d = tl.load((W + base + 208).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
        s = tl.load((W + base + 192 + group).to(tl.pointer_type(tl.int8)), valid, 0).to(tl.int32)
        scale, minimum = d * s, tl.full(base.shape, 0, tl.float32)
    return scale, minimum


@triton.jit
def _k236_values(W, base, lane, valid, QTYPE: tl.constexpr):
    if QTYPE == 'Q6_K':
        low = tl.load(W + base + lane // 128 * 64 + lane % 64, valid, 0).to(tl.int32)
        high = tl.load(W + base + 128 + lane // 128 * 32 + lane % 32, valid, 0).to(tl.int32)
        q = ((low >> (lane % 128 // 64 * 4)) & 15) | (((high >> (lane % 128 // 32 * 2)) & 3) << 4)
        return q - 32
    else:
        offset: tl.constexpr = 16 if QTYPE == 'Q2_K' else 32
        low = tl.load(W + base + offset + lane // 128 * 32 + lane % 32, valid, 0).to(tl.int32)
        q = (low >> (lane % 128 // 32 * 2)) & 3
        if QTYPE == 'Q3_K':
            high = tl.load(W + base + lane % 32, valid, 0).to(tl.int32)
            q -= (((high >> (lane // 32)) & 1) ^ 1) * 4
        return q


@triton.jit
def _weight_tile(W, positions, valid, QTYPE: tl.constexpr):
    if QTYPE == 'Q4_K' or QTYPE == 'Q5_K':
        return _q45_tile(W, positions, valid, QTYPE == 'Q5_K')
    elif QTYPE == 'Q2_K' or QTYPE == 'Q3_K' or QTYPE == 'Q6_K':
        size: tl.constexpr = 84 if QTYPE == 'Q2_K' else 110 if QTYPE == 'Q3_K' else 210
        base, lane = positions // 256 * size, positions % 256
        scale, minimum = _k236_scale(W, base, lane, valid, QTYPE)
        return scale * _k236_values(W, base, lane, valid, QTYPE) - minimum
    else:
        size: tl.constexpr = 18 if QTYPE == 'Q4_0' else 20 if QTYPE == 'Q4_1' else 22 if QTYPE == 'Q5_0' else 24 if QTYPE == 'Q5_1' else 34
        base, lane = positions // 32 * size, positions % 32
        d = tl.load((W + base).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
        if QTYPE == 'Q8_0':
            q = tl.load((W + base + 2 + lane).to(tl.pointer_type(tl.int8)), valid, 0).to(tl.int32)
            return d * q
        else:
            offset: tl.constexpr = 2 if QTYPE == 'Q4_0' else 4 if QTYPE == 'Q4_1' else 6 if QTYPE == 'Q5_0' else 8
            low = tl.load(W + base + offset + lane % 16, valid, 0).to(tl.int32)
            q = (low >> (lane // 16 * 4)) & 15
            if QTYPE == 'Q5_0' or QTYPE == 'Q5_1':
                high = tl.load(W + base + offset - 4 + lane // 8, valid, 0).to(tl.int32)
                q |= ((high >> (lane % 8)) & 1) << 4
            if QTYPE == 'Q4_0' or QTYPE == 'Q5_0':
                return d * (q - (8 if QTYPE == 'Q4_0' else 16))
            else:
                minimum = tl.load((W + base + 2).to(tl.pointer_type(tl.float16)), valid, 0).to(tl.float32)
                return d * q + minimum


@triton.jit
def _weight_tile_16(W, rows, chunks, N: tl.constexpr, K: tl.constexpr, QTYPE: tl.constexpr):
    if QTYPE == 'Q4_K' or QTYPE == 'Q5_K':
        return _q45_tile_16(W, rows, chunks, N, K, QTYPE == 'Q5_K')
    elif QTYPE == 'Q2_K' or QTYPE == 'Q3_K' or QTYPE == 'Q6_K':
        position = rows[:, None] * K + chunks[None, :] * 16
        valid = (rows[:, None] < N) & (chunks[None, :] * 16 < K)
        size: tl.constexpr = 84 if QTYPE == 'Q2_K' else 110 if QTYPE == 'Q3_K' else 210
        base, lane = position // 256 * size, position % 256
        scale, minimum = _k236_scale(W, base, lane, valid, QTYPE)
        q = _k236_values(W, base[:, :, None], lane[:, :, None] + tl.arange(0, 16)[None, None, :], valid[:, :, None], QTYPE)
        return (scale[:, :, None] * q - minimum[:, :, None]).reshape(rows.shape[0], chunks.shape[0] * 16)
    else:
        k = (chunks[:, None] * 16 + tl.arange(0, 16)[None, :]).reshape(chunks.shape[0] * 16)
        return _weight_tile(W, rows[:, None] * K + k[None, :], (rows[:, None] < N) & (k[None, :] < K), QTYPE)


def _launch(x, raw, output, n, k, qtype_name, bias, config):
    bm, bn, bk, warps, stages = config
    _packed_mm[(triton.cdiv(x.shape[0], bm), triton.cdiv(n, bn))](
        x, raw, bias, output, x.shape[0], x.stride(0), x.stride(1), n, k,
        qtype_name, bias is not None, bm, bn, bk,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)


def packed_linear(input, raw, shape, qtype_name, bias, dtype, config=None):
    n, k = shape
    x = input.reshape(-1, k)
    output = torch.empty((x.shape[0], n), dtype=dtype, device=x.device)
    if x.shape[0] == 0:
        return output.reshape(*input.shape[:-1], n)
    if bias is not None:
        bias = bias.contiguous()
    if config is None:
        def prepare_scratch():
            scratch_x = x[:min(x.shape[0], 256)].clone()
            scratch_y = output.new_empty((scratch_x.shape[0], n))
            return lambda candidate: _launch(scratch_x, raw, scratch_y, n, k, qtype_name, bias, candidate)
        identity = (n, k, qtype_name, x.dtype, dtype, bias is not None)
        config = _TUNER.config('gguf_flat_mm', identity, x.device, _CONFIGS, prepare_scratch, x.shape[0] >= 16)
    _launch(x, raw, output, n, k, qtype_name, bias, config)
    return output.reshape(*input.shape[:-1], n)


@torch.library.custom_op('wangp_gguf::flat_linear', mutates_args=())
def flat_linear(input: torch.Tensor, raw: torch.Tensor, shape: List[int], qtype_name: str,
                bias: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    return packed_linear(input, raw, shape, qtype_name, bias, dtype)


@flat_linear.register_fake
def _flat_linear_fake(input, raw, shape, qtype_name, bias, dtype):
    return input.new_empty((*input.shape[:-1], shape[0]), dtype=dtype)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
