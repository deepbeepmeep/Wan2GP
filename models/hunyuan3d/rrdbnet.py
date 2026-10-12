# RRDBNet (ESRGAN, Wang et al. 2018) with the parameter names of BasicSR basicsr/archs/rrdbnet_arch.py, used by Real-ESRGAN x4plus
# (https://github.com/xinntao/Real-ESRGAN), and the Real-ESRGAN RealESRGANer.enhance steps Hunyuan3D 2.1 runs on its texture views
# (hy3dpaint/utils/image_super_utils.py: scale 4, no tiling, no padding, FP16). Reimplemented for WanGP: large convolutions by bands of
# rows (shared.utils.conv_bands, and conv_hr + conv_last together; values can differ from whole-image convolutions by rounding only),
# dense blocks in one channels_last buffer (with Triton copies on CUDA GPUs, triton_kernels.py), output post-processing on the GPU.
#
# Real-ESRGAN: BSD 3-Clause License, Copyright (c) 2021, Xintao Wang. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions
# are met: 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer. 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution. 3. Neither the name of the copyright holder nor
# the names of its contributors may be used to endorse or promote products derived from this software without specific prior written
# permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from shared.utils.conv_bands import BAND_BYTES, band_convs, conv_bands

try:
    from . import triton_kernels as _kernels
except (ImportError, OSError, RuntimeError):
    _kernels = None


def _fused(x):
    return _kernels is not None and x.is_cuda and x.shape[0] == 1 and _kernels.supported(x.device)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

    def forward(self, x):
        """x channels_last. The dense concatenations (x, x1, ..., x4) are one channels_last buffer whose channel slices receive the
        outputs of conv1..conv4: no torch.cat and no NCHW <-> NHWC conversion around cuDNN's NHWC kernels (same values)."""
        n, channels, height, width = x.shape
        grow = self.conv1.out_channels
        dense = torch.empty((n, channels + 4 * grow, height, width), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
        if _fused(x):  # cuDNN reads packed inputs: the prefixes are packed by a vectorized copy instead of PyTorch's strided one
            _kernels.copy_channels(x, dense[:, :channels])
            for i, conv in enumerate((self.conv1, self.conv2, self.conv3, self.conv4)):
                start = channels + i * grow
                source = x if i == 0 else _kernels.copy_channels(dense[:, :start], torch.empty((n, start, height, width), dtype=x.dtype, device=x.device, memory_format=torch.channels_last))
                _kernels.copy_channels(F.conv2d(source, conv.weight, padding=1), dense[:, start:start + grow], conv.bias, leaky=True)
                del source
            return _kernels.scale_add_(F.conv2d(dense, self.conv5.weight, padding=1), x, 0.2, self.conv5.bias)
        dense[:, :channels] = x
        for i, conv in enumerate((self.conv1, self.conv2, self.conv3, self.conv4)):
            start = channels + i * grow
            dense[:, start:start + grow] = F.leaky_relu(F.conv2d(dense[:, :start], conv.weight, conv.bias, padding=1), 0.2, inplace=True)
        return F.conv2d(dense, self.conv5.weight, self.conv5.bias, padding=1).mul_(0.2).add_(x)


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return _kernels.scale_add_(out, x, 0.2) if _fused(x) else out * 0.2 + x


class RRDBNet(nn.Module):
    """x4 RRDBNet (RealESRGAN_x4plus: 23 blocks of 64 features, growth 32)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        band_convs(self)

    def forward(self, x):
        feat = self.conv_first(x).contiguous(memory_format=torch.channels_last)  # channels_last to the end: no layout conversions around cuDNN
        feat = feat + self.conv_body(self.body(feat))
        feat = F.leaky_relu(conv_bands(feat, self.conv_up1, upsample=True), 0.2, inplace=True)
        feat = F.leaky_relu(conv_bands(feat, self.conv_up2, upsample=True), 0.2, inplace=True)
        return self._conv_hr_last(feat)

    def _conv_hr_last(self, feat):
        """conv_last(leaky_relu(conv_hr(feat))) by bands of rows with a two-row halo, so that the full-size output of conv_hr (64 channels
        at the output resolution) is never allocated."""
        n, _, h, w = feat.shape
        rows = max(1, BAND_BYTES // (feat[:, :, :1].numel() * feat.element_size()))
        out = None
        for start in range(0, h, rows):
            stop = min(h, start + rows)
            lo, hi = max(0, start - 2), min(h, stop + 2)
            band = self.conv_last(F.leaky_relu(self.conv_hr(feat[:, :, lo:hi]), 0.2, inplace=True))[:, :, start - lo:stop - lo]
            if out is None:
                out = band.new_empty((n, band.shape[1], h, w))
            out[:, :, start:stop].copy_(band)
            del band
        return out

    def enhance(self, image, device):
        """RealESRGANer.enhance of an RGB uint8 image (H, W, 3) with half=True: the network sees the channels in reversed order (the
        image is taken for BGR), its clamped output is reordered back and rounded to uint8 (4H, 4W, 3)."""
        x = torch.from_numpy(np.ascontiguousarray((image.astype(np.float32) / 255)[..., ::-1].transpose(2, 0, 1))).float()[None]
        out = self(x.to(device).half())[0].float().clamp_(0, 1)
        return (out[[2, 1, 0]].permute(1, 2, 0) * 255.0).round_().to(torch.uint8).cpu().numpy()
