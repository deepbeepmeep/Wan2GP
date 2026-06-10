import torch
import torch.nn as nn
from einops import rearrange
from torchaudio.transforms import Resample
from .blocks import ResidualUnit, WNConv1d


class AutoencoderPretransform(nn.Module):
    def __init__(self, model, scale=1.0, iterate_batch=False, chunked=False):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale = scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.enable_grad = False
        self.iterate_batch = iterate_batch
        self.chunked = chunked

    def encode(self, x, **kwargs):
        return self.model.encode_audio(x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs) / self.scale

    def decode(self, z, chunked=None, **kwargs):
        chunked = self.chunked if chunked is None else chunked
        return self.model.decode_audio(z * self.scale, chunked=chunked, iterate_batch=self.iterate_batch, **kwargs)

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x


class PatchedPretransform(nn.Module):
    def __init__(self, channels, patch_size, oversampling = 1, postfilter_channels = 0, **kwargs):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.oversampling = oversampling

        self.downsampling_ratio = patch_size
        self.io_channels = channels
        self.encoded_channels = channels * patch_size
        self.enable_grad = False

        if self.oversampling > 1:
            self.input_upsampler = Resample(1, self.oversampling)
            self.output_downsampler = Resample(self.oversampling, 1)

        if postfilter_channels > 0:
            self.postfilter = nn.Sequential(
            WNConv1d(in_channels=channels, out_channels=postfilter_channels, kernel_size=7, padding=3, bias=True),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=1, use_snake=True, bias = True),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=3, use_snake=True, bias = True),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=9, use_snake=True, bias = True),
            WNConv1d(in_channels=postfilter_channels, out_channels=channels, kernel_size=7, padding=3, bias=False))

    def _pad(self, x):
        seq_len = x.shape[-1]
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)
        return x
        
    def encode(self, x):
        if self.oversampling > 1:
            x = self.input_upsampler(x)
        x = self._pad(x)
        x = rearrange(x, "b c (l h) -> b (c h) l", h=self.patch_size)
        return x
    def decode(self, x):
        x = rearrange(x, "b (c h) l -> b c (l h)", h=self.patch_size)
        if hasattr(self, 'postfilter'):
            x = self.postfilter(x)
        if self.oversampling > 1:
            x = self.output_downsampler(x)
        return x

