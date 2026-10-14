# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Triton kernels of the Hunyuan3D 2.1 paint stage, bit-identical to the PyTorch paths they replace:
- rope_: the 3D RoPE of the multiview attention (paint_unet._rope_rows) in place, one pass over q or k instead of the FP32 copy, the
  rotated copy and the products of the PyTorch path; the same FP32 operations, each rounded as there (no fused multiply-add).
- copy_channels / scale_add_: the dense blocks of ESRGAN (rrdbnet.py), whose channels_last concatenation buffer takes each convolution
  output in a channel slice and gives each convolution a packed prefix of its channels (cuDNN needs packed inputs): vectorized copies
  instead of PyTorch's strided copy kernel, with the convolution bias and the LeakyReLU applied while copying, and the dense block's
  bias, output scale and residual in one pass. They follow PyTorch's half-precision kernels: FP32 math, rounded to FP16 after each
  operation (the bias is added to the rounded convolution output, as PyTorch does after cuDNN).

Counts and strides are not specialized; the channel counts of the copies are model constants: one binary per kernel and channel count.
Launch configurations start from a static choice and are tuned once per kernel and GPU on scratch outputs."""
import functools

import torch
import triton
import triton.language as tl

from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_hunyuan3d_kernels", 3, "Hunyuan3D")
_TUNE_ROWS = 1 << 16  # rows (tokens x heads) of a scratch tuning run of the RoPE; calls with fewer rows use the static configuration
_TUNE_PIXELS = 1 << 17  # pixels of a scratch tuning run of the ESRGAN kernels


@functools.cache
def supported(device):
    """CUDA GPUs with compute capability 8.0+; other devices use the PyTorch path."""
    return device.type == "cuda" and torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0)


@triton.jit
def _add_rn(a, b):
    """a + b as an opaque scalar add.rn.f32: Triton lowers tensor arithmetic to packed FP32 instructions on Blackwell GPUs, and ptxas fuses
    a packed product and sum into an FMA (sm_120, Triton 3.6), which the PyTorch path rounds separately."""
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit(do_not_specialize=["ROWS", "TOKENS", "HEADS", "SB", "SH", "ST"])
def _rope_kernel(X, COS, SIN, ROWS, TOKENS, HEADS, SB, ST, SH, D: tl.constexpr, BLOCK: tl.constexpr):
    """BLOCK rows (batch, token, head) of x: x * cos + rotate(x) * sin with rotate(x)[2i] = -x[2i + 1], rotate(x)[2i + 1] = x[2i]."""
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < ROWS
    head = rows % HEADS
    token = rows // HEADS % TOKENS
    offsets = (rows // (HEADS * TOKENS) * SB + token * ST + head * SH)[:, None] + tl.arange(0, D)[None, :]
    x = tl.load(X + offsets, valid[:, None], 0.0).to(tl.float32)
    even, odd = tl.split(tl.reshape(x, (BLOCK, D // 2, 2)))  # interleaved pairs, loaded contiguously
    rotated = tl.reshape(tl.join(-odd, even), (BLOCK, D))
    table = token[:, None] * D + tl.arange(0, D)[None, :]
    out = _add_rn(x * tl.load(COS + table, valid[:, None], 0.0), rotated * tl.load(SIN + table, valid[:, None], 0.0))
    tl.store(X + offsets, out.to(X.dtype.element_ty), valid[:, None])


def rope_(x, cos, sin):
    """x (batch, tokens, heads, head_dim) FP16/BF16, possibly a view of a group of heads, rotated in place by the FP32 tables cos and sin
    (1, tokens, 1, head_dim) of interleaved pairs."""
    batch, tokens, heads, dim = x.shape
    cos, sin = cos.reshape(-1, dim), sin.reshape(-1, dim)

    def launch(target, config):
        block, warps = config
        rows = target.shape[0] * target.shape[1] * heads
        _rope_kernel[(triton.cdiv(rows, block),)](target, cos, sin, rows, target.shape[1], heads, target.stride(0), target.stride(1), target.stride(2), dim, block,
                                                  num_warps=warps, enable_fp_fusion=False)

    def scratch():
        target = x[:1, :max(1, _TUNE_ROWS // heads)].clone()
        return lambda config: launch(target, config)

    launch(x, _TUNER.config("rope", dim, x.device, ((32, 4), (16, 2), (64, 4), (64, 8), (128, 8)), scratch, batch * tokens * heads >= _TUNE_ROWS))
    torch.autograd.graph.increment_version(x)
    return x


@triton.jit
def _bias_add(x, BIAS, channels, mask, HAS_BIAS: tl.constexpr):
    """x + bias of FP16/BF16 values (PyTorch adds a convolution's bias to its rounded output, rounded again)."""
    if HAS_BIAS:
        x = (x.to(tl.float32) + tl.load(BIAS + channels, mask).to(tl.float32)).to(x.dtype)
    return x


@triton.jit(do_not_specialize=["PIXELS", "SRC_STRIDE", "DST_STRIDE"])
def _copy_channels_kernel(SRC, DST, BIAS, PIXELS, SRC_STRIDE, DST_STRIDE, CHANNELS: tl.constexpr, C_BLOCK: tl.constexpr, HAS_BIAS: tl.constexpr,
                          LEAKY: tl.constexpr, BLOCK: tl.constexpr):
    """DST[p, :CHANNELS] = SRC[p, :CHANNELS] (+ BIAS, then LeakyReLU 0.2 when LEAKY) for BLOCK pixels p, rows of SRC_STRIDE / DST_STRIDE
    elements."""
    pixels = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    channels = tl.arange(0, C_BLOCK)[None, :]
    valid = (pixels[:, None] < PIXELS) & (channels < CHANNELS)
    x = _bias_add(tl.load(SRC + pixels[:, None] * SRC_STRIDE + channels, valid), BIAS, channels, channels < CHANNELS, HAS_BIAS)
    if LEAKY:
        x = x.to(tl.float32)
        x = tl.where(x > 0, x, x * 0.2).to(DST.dtype.element_ty)
    tl.store(DST + pixels[:, None] * DST_STRIDE + channels, x, valid)


def copy_channels(source, target, bias=None, leaky=False):
    """target = source (+ bias, then LeakyReLU 0.2 when leaky) for NCHW-shaped channels_last tensors (N = 1) whose channels may be a slice
    of a wider buffer: each pixel's channels are contiguous."""
    pixels, channels = source.shape[2] * source.shape[3], source.shape[1]

    def launch(dst, dst_stride, count, config):
        block, warps = config
        _copy_channels_kernel[(triton.cdiv(count, block),)](source, dst, source if bias is None else bias, count, source.stride(3), dst_stride, channels,
                                                            triton.next_power_of_2(channels), bias is not None, leaky, block, num_warps=warps, enable_fp_fusion=False)

    def scratch():
        dst = torch.empty((min(pixels, _TUNE_PIXELS), channels), dtype=target.dtype, device=target.device)
        return lambda config: launch(dst, channels, len(dst), config)

    identity = f"{channels}{'_bias' if bias is not None else ''}{'_leaky' if leaky else ''}"
    launch(target, target.stride(3), pixels, _TUNER.config("copy_channels", identity, source.device, ((64, 4), (32, 4), (128, 4), (128, 8)), scratch, pixels >= _TUNE_PIXELS))
    torch.autograd.graph.increment_version(target)
    return target


@triton.jit(do_not_specialize=["N", "C"])
def _scale_add_kernel(A, X, BIAS, N, C, SCALE, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    a = _bias_add(tl.load(A + offsets, valid), BIAS, offsets % C, valid, HAS_BIAS)
    scaled = (a.to(tl.float32) * SCALE).to(A.dtype.element_ty)
    tl.store(A + offsets, (scaled.to(tl.float32) + tl.load(X + offsets, valid).to(tl.float32)).to(A.dtype.element_ty), valid)


def scale_add_(a, x, scale, bias=None):
    """a = (a + bias) * scale + x in place (a.add_(bias).mul_(scale).add_(x)), for a and x (N, C, H, W) channels_last and dense."""

    def launch(target, count, config):
        block, warps = config
        _scale_add_kernel[(triton.cdiv(count, block),)](target, x, a if bias is None else bias, count, a.shape[1], scale, bias is not None, block, num_warps=warps,
                                                        enable_fp_fusion=False)

    def scratch():
        target = a.permute(0, 2, 3, 1).reshape(-1)[:_TUNE_PIXELS * a.shape[1]].clone()
        return lambda config: launch(target, target.numel(), config)

    launch(a, a.numel(), _TUNER.config("scale_add", f"{a.dtype}{'_bias' if bias is not None else ''}", a.device, ((4096, 8), (2048, 4), (8192, 8), (1024, 4)), scratch,
                                       a.numel() >= _TUNE_PIXELS * a.shape[1]))
    torch.autograd.graph.increment_version(a)
    return a


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
