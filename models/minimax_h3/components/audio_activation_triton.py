"""Alias-free SnakeBeta activation of the H3 audio decoder (BigVGAN) in one Triton kernel: 2x upsampling (replicate padding, 12-tap
transposed Kaiser-sinc filter), SnakeBeta, then 2x downsampling (replicate padding, 12-tap low-pass filter). The unfused module runs a
dozen kernels per activation (the decoder has about a hundred and thirty of them) and writes two intermediates of twice its length."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _snake_beta_aa(X, Y, ALPHA, BETA, UP, DOWN, length, channels, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = n < length
    channel = row % channels
    alpha = tl.exp(tl.load(ALPHA + channel).to(tl.float32))
    inv_beta = 1.0 / (tl.exp(tl.load(BETA + channel).to(tl.float32)) + 1e-9)
    x_row = X + row.to(tl.int64) * length
    out = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(12):
        # sample 2n + k - 5 of the upsampled signal (replicate padding of the downsampler: 5 before, 6 after)
        m = tl.minimum(tl.maximum(2 * n + k - 5, 0), 2 * length - 1)
        # upsampled sample m = 2 * sum over the 6 inputs i of x[i - 5] * up[m + 15 - 2i] (replicate padding of 5, crop of 15)
        last = (m + 15) // 2
        tap = m + 15 - 2 * last
        up = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(6):
            source = tl.minimum(tl.maximum(last - j - 5, 0), length - 1)
            up += tl.load(x_row + source, mask=valid, other=0.0).to(tl.float32) * tl.load(UP + tap + 2 * j).to(tl.float32)
        up = 2.0 * up
        sine = libdevice.sin(alpha * up)
        out += (up + inv_beta * sine * sine) * tl.load(DOWN + k).to(tl.float32)
    tl.store(Y + row.to(tl.int64) * length + n, out.to(Y.dtype.element_ty), mask=valid)


def snake_beta_aa(x, alpha, beta, up_filter, down_filter):
    """x [batch, channels, length], alpha and beta [channels] (log space), 12-tap filters: the activation's output, shaped like x."""
    x = x.contiguous()
    out = torch.empty_like(x)
    batch, channels, length = x.shape
    _snake_beta_aa[(batch * channels, triton.cdiv(length, 512))](x, out, alpha, beta, up_filter.reshape(-1), down_filter.reshape(-1), length,
                                                                 channels, BLOCK=512)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger

install_triton_compilation_logger()
