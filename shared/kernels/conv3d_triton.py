# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Triton implicit-GEMM 3D / 2D convolution on channels-last 16-bit activations ("fast" kernel precision), shared by the Wan VAEs and the
streamed LTX VAE: FP32 accumulation, the bias and an optional residual added as PyTorch rounds them, optionally the nearest x2 upsampling
of the input read in place (Wan decoder). 1.1-2x faster than cuDNN on RTX 50 GPUs, where it is enabled; only the accumulation order
differs. One binary per channel count, kernel shape, mode and dtype; sizes and paddings are not specialized. Launch configurations start
from a static choice and are tuned once per identity and GPU on scratch outputs of a large call."""
import functools

import torch
import triton
import triton.language as tl

from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_conv3d_triton", 1, "VAE")
_TUNE_ELEMENTS = 1 << 24  # calls writing at least this many elements tune the configurations
_CL = torch.channels_last_3d


@functools.cache
def supported(device):
    """CUDA GPUs with compute capability 8.0+; other devices use the PyTorch path."""
    return device.type == "cuda" and torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0)


@functools.cache
def conv_supported(device):
    """Triton convolutions: validated on compute capability 12.x GPUs (GeForce RTX 50), where they beat cuDNN; others keep cuDNN."""
    return supported(device) and torch.cuda.get_device_capability(device)[0] == 12


_CONV_MIN_PIXELS = 8192  # with fewer output pixels (small tiles at low resolution), cuDNN is as fast or faster (launch cost, occupancy)


def conv_eligible(x, conv, upsample=False):
    """Triton convolution for conv (stride 1, channels multiple of 32) on the 16-bit channels-last input x of a validated GPU."""
    cout, cin = conv.weight.shape[:2]
    if upsample:
        pixels = x.shape[0] * x.shape[2] * x.shape[3] * 4
    else:
        pixels = x.shape[0] * (x.shape[2] - conv.kernel_size[0] + 1) * (x.shape[3] - conv.kernel_size[1] + 1) * (x.shape[4] - conv.kernel_size[2] + 1)
    return x.dtype != torch.float32 and cin % 32 == 0 and cout % 32 == 0 and all(s == 1 for s in conv.stride) and pixels >= _CONV_MIN_PIXELS and conv_supported(x.device)


@triton.jit(do_not_specialize=["M", "T", "H", "W", "TP", "HP", "WP"])
def _conv_kernel(X, WT, BIAS, RES, Y, M, T, H, W, TP, HP, WP, CIN: tl.constexpr, COUT: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
                 KW: tl.constexpr, UPSAMPLE: tl.constexpr, RESIDUAL: tl.constexpr, BLOCK_M: tl.constexpr, BN1: tl.constexpr,
                 BN2: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """Implicit-GEMM convolution with FP32 accumulation of BLOCK_M output pixels x BN1 + BN2 output channels (two products when the
    channels per program are no power of two) of the channels-last output (batch, T, H, W, COUT). Weights WT channels last (COUT, KT, KH,
    KW, CIN): rows of K contiguous per output channel, as cuDNN's channels-last convolution converts them.
    Not UPSAMPLE: X is the padded input (batch, TP, HP, WP, CIN) of a valid convolution; the KW taps of an input row and their channels are
    one contiguous K range. UPSAMPLE: X (batch, HP, WP, CIN), T = TP = 1, a 3x3 convolution with zero padding 1 of its nearest x2
    upsampling (H = 2 HP, W = 2 WP), read in place. As PyTorch: the convolution rounded to the output dtype, plus the bias, rounded, plus
    RES (RESIDUAL), rounded."""
    BN: tl.constexpr = BN1 + BN2
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n: tl.constexpr = COUT // BN
    first_m = (pid // (GROUP_M * num_n)) * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % (GROUP_M * num_n)) % size_m
    pid_n = (pid % (GROUP_M * num_n)) // size_m
    m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = m < M
    w = m % W
    r = m // W
    h = r % H
    r = r // H
    t = r % T
    b = r // T
    n1 = pid_n * BN + tl.arange(0, BN1)
    n2 = pid_n * BN + BN1 + tl.arange(0, BN2 if BN2 > 0 else BN1)
    k = tl.arange(0, BLOCK_K)
    acc1 = tl.zeros((BLOCK_M, BN1), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BN2 if BN2 > 0 else BN1), dtype=tl.float32)
    if UPSAMPLE:
        for kh in tl.static_range(3):
            uh = h + kh - 1
            for kw in tl.static_range(3):
                uw = w + kw - 1
                inside = rows & (uh >= 0) & (uh < H) & (uw >= 0) & (uw < W)
                row = ((b * HP + uh // 2) * WP + uw // 2) * CIN
                for k0 in range(0, CIN, BLOCK_K):
                    a = tl.load(X + row[:, None] + (k0 + k)[None, :], inside[:, None], other=0.0)
                    weights = WT + ((kh * 3 + kw) * CIN + k0 + k)[None, :]
                    acc1 = tl.dot(a, tl.trans(tl.load(weights + (n1 * 9 * CIN)[:, None])), acc1)
                    if BN2 > 0:
                        acc2 = tl.dot(a, tl.trans(tl.load(weights + (n2 * 9 * CIN)[:, None])), acc2)
    else:
        base = (((b * TP + t) * HP + h) * WP + w) * CIN
        for kt in tl.static_range(KT):
            for kh in tl.static_range(KH):
                row = base + (kt * HP + kh) * WP.to(tl.int64) * CIN
                for k0 in range(0, KW * CIN, BLOCK_K):
                    a = tl.load(X + row[:, None] + (k0 + k)[None, :], rows[:, None], other=0.0)
                    weights = WT + ((kt * KH + kh) * KW * CIN + k0 + k)[None, :]
                    acc1 = tl.dot(a, tl.trans(tl.load(weights + (n1 * (KT * KH * KW * CIN))[:, None])), acc1)
                    if BN2 > 0:
                        acc2 = tl.dot(a, tl.trans(tl.load(weights + (n2 * (KT * KH * KW * CIN))[:, None])), acc2)
    dtype = Y.dtype.element_ty
    y = acc1.to(dtype).to(tl.float32) + tl.load(BIAS + n1).to(tl.float32)[None, :]
    if RESIDUAL:
        y = y.to(dtype).to(tl.float32) + tl.load(RES + (m * COUT)[:, None] + n1[None, :], rows[:, None], other=0.0).to(tl.float32)
    tl.store(Y + (m * COUT)[:, None] + n1[None, :], y.to(dtype), rows[:, None])
    if BN2 > 0:
        y = acc2.to(dtype).to(tl.float32) + tl.load(BIAS + n2).to(tl.float32)[None, :]
        if RESIDUAL:
            y = y.to(dtype).to(tl.float32) + tl.load(RES + (m * COUT)[:, None] + n2[None, :], rows[:, None], other=0.0).to(tl.float32)
        tl.store(Y + (m * COUT)[:, None] + n2[None, :], y.to(dtype), rows[:, None])


_WEIGHT_KERNEL_ELEMENTS = 1 << 22


@triton.jit(do_not_specialize=["CIN"])
def _weight_kernel(W, OUT, CIN, TAPS: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
    """BLOCK_C input channels of one output channel of a convolution weight (cout, cin, taps) written channels last (cout, taps, cin)."""
    base = tl.program_id(0).to(tl.int64) * CIN * TAPS
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    t = tl.arange(0, BLOCK_T)
    mask = (c < CIN)[:, None] & (t < TAPS)[None, :]
    tl.store(OUT + base + (t * CIN)[None, :] + c[:, None], tl.load(W + base + (c * TAPS)[:, None] + t[None, :], mask), mask)


def channels_last_weight(weight):
    """weight (cout, cin, *kernel) in the channels-last layout of the channels-last convolutions: large 3D weights by a tiled transpose,
    2.5x faster than PyTorch's strided copy (tiled decodes convert every weight once per tile and chunk); small ones are launch-bound."""
    taps = weight[0, 0].numel()
    layout = _CL if weight.dim() == 5 else torch.channels_last
    if taps == 1 or weight.dim() != 5 or weight.numel() < _WEIGHT_KERNEL_ELEMENTS or weight.is_contiguous(memory_format=layout):
        return weight.contiguous(memory_format=layout)
    out = torch.empty_like(weight, memory_format=layout)
    _weight_kernel[(weight.shape[0], triton.cdiv(weight.shape[1], 64))](weight.contiguous(), out, weight.shape[1], taps, 64, triton.next_power_of_2(taps), num_warps=4)
    return out


def _conv_candidates(cout):
    """Launch configurations (BLOCK_M, BN1, BN2, BLOCK_K, warps, stages), the first being the static choice: output channels by 64
    (96 = 64 + 32: no power-of-two padding), and by 128 when they divide."""
    split = (64, 0) if cout % 64 == 0 else (64, 32) if cout % 96 == 0 else (128, 32) if cout % 160 == 0 else (32, 0)
    candidates = [(128, *split, 32, 4, 4), (128, *split, 32, 4, 3), (64, *split, 32, 4, 4)]
    if cout % 128 == 0:
        candidates.append((128, 128, 0, 32, 4, 3))
    return tuple(candidates)


def conv(x, weight, bias, residual=None, upsample=False):
    """Convolution (stride 1) of the padded channels-last input x (batch, cin, TP, HP, WP) by weight (cout, cin, kt, kh, kw), or with
    upsample of the nearest x2 upsampling of x (frames, cin, h, w) by a 3x3 weight with zero padding 1, plus bias (and residual, the
    output's shape and layout), as the PyTorch path rounds them. Returns the channels-last output."""
    cout, cin = weight.shape[:2]
    x = x.contiguous(memory_format=torch.channels_last if upsample else _CL)  # no copy for the fast path's activations
    residual = None if residual is None else residual.contiguous(memory_format=_CL)
    if upsample:
        batch, _, hp, wp = x.shape
        kt, kh, kw, tp, t, h, w = 1, 3, 3, 1, 1, 2 * hp, 2 * wp
        out = torch.empty((batch, cout, h, w), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    else:
        batch, _, tp, hp, wp = x.shape
        kt, kh, kw = weight.shape[2:]
        t, h, w = tp - kt + 1, hp - kh + 1, wp - kw + 1
        out = torch.empty((batch, cout, t, h, w), dtype=x.dtype, device=x.device, memory_format=_CL)
    pixels = batch * t * h * w
    wt = channels_last_weight(weight)

    def launch(target, count, config, res=residual):
        block_m, bn1, bn2, block_k, warps, stages = config
        _conv_kernel[(triton.cdiv(count, block_m) * (cout // (bn1 + bn2)),)](x, wt, bias, target if res is None else res, target, count, t, h, w,
                                                                             tp, hp, wp, cin, cout, kt, kh, kw, upsample, res is not None,
                                                                             block_m, bn1, bn2, block_k, 8, num_warps=warps, num_stages=stages)

    def scratch():
        count = min(pixels, 1 << 16)
        target = torch.empty(count * cout, dtype=x.dtype, device=x.device)
        return lambda config: launch(target, count, config, residual)  # the residual is only read

    identity = f"{cin}|{cout}|{kt}x{kh}x{kw}|{'upsample' if upsample else 'padded'}|{'residual' if residual is not None else ''}|{x.dtype}"
    candidates = _conv_candidates(cout)
    launch(out, pixels, _TUNER.config("conv", identity, x.device, candidates, scratch, pixels * cout >= _TUNE_ELEMENTS))
    return out


# 2D variant (FlashVSR TCDecoder): a 3x3 convolution with zero padding 1 reading the unpadded channels-last input in place (no padded
# copy), the bias and an optional ReLU in the epilogue (ReLU after the rounding, as PyTorch's separate in-place ReLU). Its own kernel and
# tuning identities: the 3D / upsampling variants above are unchanged.

def conv2d_eligible(x, conv):
    """Triton 2D convolution for conv (3x3, padding 1 with zeros, stride 1, input channels multiple of 32, output channels of 64) on the
    16-bit input x (batch, cin, h, w) of a validated GPU."""
    return (x.dtype != torch.float32 and x.is_cuda and conv.kernel_size == (3, 3) and conv.padding == (1, 1) and conv.stride == (1, 1)
            and conv.dilation == (1, 1) and conv.groups == 1 and conv.padding_mode == "zeros" and conv.in_channels % 32 == 0
            and conv.out_channels % 64 == 0 and x.shape[0] * x.shape[2] * x.shape[3] >= _CONV_MIN_PIXELS and conv_supported(x.device))


@triton.jit(do_not_specialize=["M", "H", "W"])
def _conv2d_kernel(X, WT, BIAS, Y, M, H, W, CIN: tl.constexpr, COUT: tl.constexpr, RELU: tl.constexpr, BLOCK_M: tl.constexpr,
                   BN: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """Implicit-GEMM 3x3 convolution, zero padding 1, FP32 accumulation, of BLOCK_M output pixels x BN output channels: X channels last
    (batch, H, W, CIN), WT (COUT, 3, 3, CIN). As PyTorch: the convolution rounded to the output dtype, plus the bias, rounded, then ReLU."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n: tl.constexpr = COUT // BN
    first_m = (pid // (GROUP_M * num_n)) * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % (GROUP_M * num_n)) % size_m
    pid_n = (pid % (GROUP_M * num_n)) // size_m
    m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = m < M
    w = m % W
    r = m // W
    h = r % H
    b = r // H
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BN), dtype=tl.float32)
    for kh in tl.static_range(3):
        uh = h + kh - 1
        for kw in tl.static_range(3):
            uw = w + kw - 1
            inside = rows & (uh >= 0) & (uh < H) & (uw >= 0) & (uw < W)
            row = ((b * H + uh) * W + uw) * CIN
            for k0 in range(0, CIN, BLOCK_K):
                a = tl.load(X + row[:, None] + (k0 + k)[None, :], inside[:, None], other=0.0)
                acc = tl.dot(a, tl.trans(tl.load(WT + (n * 9 * CIN)[:, None] + ((kh * 3 + kw) * CIN + k0 + k)[None, :])), acc)
    dtype = Y.dtype.element_ty
    y = (acc.to(dtype).to(tl.float32) + tl.load(BIAS + n).to(tl.float32)[None, :]).to(dtype)
    if RELU:
        y = tl.maximum(y, 0.0)
    tl.store(Y + (m * COUT)[:, None] + n[None, :], y, rows[:, None])


def conv2d(x, weight, bias, relu=False):
    """3x3 convolution with zero padding 1 of x (batch, cin, h, w) by weight (cout, cin, 3, 3) plus bias (None: no bias), then ReLU (relu),
    as the PyTorch path rounds them. Returns the channels-last output."""
    batch, cin, h, w = x.shape
    cout = weight.shape[0]
    x = x.contiguous(memory_format=torch.channels_last)
    bias = torch.zeros(cout, dtype=x.dtype, device=x.device) if bias is None else bias
    out = torch.empty((batch, cout, h, w), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    pixels, wt = batch * h * w, weight.contiguous(memory_format=torch.channels_last)

    def launch(target, count, config):
        block_m, bn, block_k, warps, stages = config
        _conv2d_kernel[(triton.cdiv(count, block_m) * (cout // bn),)](x, wt, bias, target, count, h, w, cin, cout, relu, block_m, bn, block_k, 8,
                                                                     num_warps=warps, num_stages=stages)

    def scratch():
        count = min(pixels, 1 << 16)
        target = torch.empty(count * cout, dtype=x.dtype, device=x.device)
        return lambda config: launch(target, count, config)

    candidates = ((128, 64, 32, 4, 3), (128, 64, 64, 4, 3), (64, 64, 32, 4, 4)) + (((128, 128, 32, 8, 3),) if cout % 128 == 0 else ())
    identity = f"{cin}|{cout}|3x3|{'relu' if relu else ''}|{x.dtype}"
    launch(out, pixels, _TUNER.config("conv2d", identity, x.device, candidates, scratch, pixels * cout >= _TUNE_ELEMENTS))
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
