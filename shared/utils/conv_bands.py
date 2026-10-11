"""Convolutions of large images by bands of rows (stride 1, kernel height 3 with padding 1, or 1 without), optionally after a nearest x2
upsampling.

Output rows s..e-1 read only the (upsampled) input rows s-1..e: each band convolves its rows plus a one-row halo, so every output value has
the same inputs and weights as with the full convolution. Neither the full-size upsampled input nor cuDNN's workspace for the full image is
allocated: at 2048x2048, a VAE decoder convolution takes a workspace of up to twice its output, which no allocator can do without (cuDNN does
not fall back to a smaller one when memory runs out). cuDNN may choose another kernel for a band's shape, so values can differ in rounding only.
"""
import functools

import torch.nn as nn
import torch.nn.functional as F

BAND_BYTES = 128 << 20


def conv_bands(x, conv, upsample=False, mode="nearest", band_bytes=BAND_BYTES):
    """conv(x), or conv(F.interpolate(x, scale_factor=2, mode=mode)) with upsample, for x (batch, channels, height, width); conv (a module or a
    function) runs on each band, whose halo output rows are dropped."""
    n, c, h, w = x.shape
    scale = 2 if upsample else 1
    rows = max(1, band_bytes // (n * c * scale * scale * w * x.element_size()))
    if rows >= h:
        return conv(F.interpolate(x, scale_factor=2.0, mode=mode) if upsample else x)
    out = None
    for start in range(0, h, rows):
        stop = min(h, start + rows)
        lo, hi = max(0, start - 1), min(h, stop + 1)
        band = F.interpolate(x[:, :, lo:hi], scale_factor=2.0, mode=mode) if upsample else x[:, :, lo:hi]
        first = 1 if start > 0 else 0  # the halo row above the band's first output row
        band = band[:, :, scale * (start - lo) - first:scale * (stop - lo) + (stop < h)]
        band = conv(band)[:, :, first:first + scale * (stop - start)]
        if out is None:
            out = band.new_empty((n, band.shape[1], scale * h, band.shape[-1]))
        out[:, :, scale * start:scale * stop].copy_(band)
        del band
    return out


def band_convs(module):
    """Runs every 3x3 convolution of module with stride 1 and padding 1 by bands of rows (when its input exceeds BAND_BYTES)."""
    for conv in module.modules():
        if type(conv) is nn.Conv2d and conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1) and conv.dilation == (1, 1) and conv.padding_mode == "zeros":
            conv.forward = functools.partial(conv_bands, conv=functools.partial(nn.Conv2d.forward, conv))
