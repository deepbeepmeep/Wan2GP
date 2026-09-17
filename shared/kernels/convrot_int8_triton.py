"""Tile-local ConvRot for small INT8 decode inputs; no activation workspace."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _regular_step(a, BM: tl.constexpr, STRIDE: tl.constexpr):
    v = a.reshape((BM, 256 // (4 * STRIDE), 4, STRIDE))
    reverse = tl.broadcast_to((3 - tl.arange(0, 4))[None, None, :, None], (BM, 256 // (4 * STRIDE), 4, STRIDE))
    return (tl.sum(v, axis=2)[:, :, None, :] - 2.0 * tl.gather(v, reverse, axis=2)).reshape((BM, 256))


@triton.jit
def _convrot_int8_kernel(x, w, scale, out, bias, M, N, K,
                         SX0: tl.constexpr, SX1: tl.constexpr, SW0: tl.constexpr, SW1: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    rows = (pid % tiles_m) * BM + tl.arange(0, BM)
    cols = (pid // tiles_m) * BN + tl.arange(0, BN)
    kk = tl.arange(0, 256)
    qk = tl.arange(0, 64)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(0, K, 256):
        a = tl.load(x + rows[:, None] * SX0 + (base + kk[None, :]) * SX1, rows[:, None] < M, 0).to(tl.float32)
        a = _regular_step(a, BM, 1)
        a = _regular_step(a, BM, 4)
        a = _regular_step(a, BM, 16)
        a = _regular_step(a, BM, 64)
        # Preserve BF16 rounding before the existing 64-element INT8 quantization.
        a = (a * 0.0625).to(x.dtype.element_ty).to(tl.float32).reshape((BM, 4, 64))
        scales = tl.max(tl.abs(a), axis=2) / 127.0
        scales = tl.where(scales > 0, scales, 1.0)
        quant = libdevice.rint(a / scales[:, :, None])
        quant = tl.maximum(tl.minimum(quant, 127), -128).to(tl.int8).reshape((BM, 256))
        for g in tl.static_range(4):
            qa = tl.gather(quant, tl.broadcast_to((g * 64 + qk)[None, :], (BM, 64)), axis=1)
            s = tl.gather(scales, tl.full((BM, 1), g, tl.int32), axis=1).reshape((BM,))
            b = tl.load(w + cols[None, :] * SW0 + (base + g * 64 + qk[:, None]) * SW1, cols[None, :] < N, 0).to(tl.int8)
            acc += tl.dot(qa, b).to(tl.float32) * s[:, None]
    result = acc * tl.load(scale + cols, cols < N, 0)[None, :].to(tl.float32)
    if HAS_BIAS:
        result = result.to(out.dtype.element_ty).to(tl.float32) + tl.load(bias + cols, cols < N, 0)[None, :].to(tl.float32)
    tl.store(out + rows[:, None] * N + cols[None, :], result, (rows[:, None] < M) & (cols[None, :] < N))


def convrot_int8_mm(x, weight, scale, out, bias, cfg):
    m, k = x.shape
    n = weight.shape[0]
    bm, bn, _, warps, stages = cfg
    _convrot_int8_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](x, weight, scale, out, bias, m, n, k, *x.stride(), *weight.stride(), bm, bn, bias is not None, num_warps=warps, num_stages=stages)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
