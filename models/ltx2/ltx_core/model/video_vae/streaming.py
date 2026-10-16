# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Exact streamed decoding and encoding of the LTX video VAE: the frames are processed chunk after chunk, each temporal convolution keeping
the last two frames of its input, so that every output frame is the one computed on the whole video, without temporal tiles whose overlaps
would be computed twice and blended. A chunk only holds a few frames per stage; the state carried between chunks is two frames per
temporal convolution and the residual inputs not consumed yet. The stages take their input frames in a list that they empty (handed over),
so that the frames are only held while a stage needs them."""
import math

import torch
from einops import rearrange
from torch import nn

from .ops import patchify, unpatchify
from .resnet import ResnetBlock3D, UNetMidBlock3D
from .sampling import DepthToSpaceUpsample, SpaceToDepthDownsample
from shared.kernels import vae_ops

from ..common.normalization import PixelNorm

def _norm_act(norm, act, x):
    x_list, x = [x], None
    return vae_ops.pixel_norm_silu(x_list, norm, act, channel_norm=isinstance(norm, PixelNorm) and norm.dim == 1)


def _conv(conv, parts):
    return vae_ops.temporal_conv(conv, parts)


def _last_frames(parts, count=2):
    """A copy of the last `count` frames of parts concatenated in time (a view would keep the whole input)."""
    tail = []
    for part in reversed(parts):
        tail.insert(0, part[:, :, -count:])
        count -= tail[0].shape[2]
        if count <= 0:
            break
    return torch.cat(tail, dim=2) if len(tail) > 1 else tail[0].clone()


class _ConvStream:
    """A CausalConv3d of temporal kernel 3 fed by chunks: pads the ends of the video as the convolution of the whole video (first frame
    replicated twice when causal, first and last frames once otherwise)."""

    def __init__(self, module, causal):
        if module.time_kernel_size != 3:
            raise NotImplementedError("LTX VAE streamed decoding supports temporal kernels of 3 frames")
        self.conv, self.causal, self.prev = module.conv, causal, None

    def step(self, frames, last):
        x = frames.pop() if frames else None
        if x is not None:
            parts = [x[:, :, :1].expand(-1, -1, 2 if self.causal else 1, -1, -1) if self.prev is None else self.prev, x]
        elif self.prev is not None:
            parts = [self.prev]
        else:
            return None
        x = None
        if last and not self.causal:
            parts.append(parts[-1][:, :, -1:])
        self.prev = _last_frames(parts)
        return _conv(self.conv, parts) if sum(part.shape[2] for part in parts) >= 3 else None


class _ResnetStream:
    """Non causal: the two frames of history of the first convolution are inputs that the residual still waits for (pending), whose norm
    and activation are computed again rather than kept."""

    def __init__(self, block, causal):
        if block.inject_noise or block.timestep_conditioning:
            raise NotImplementedError("LTX VAE streamed decoding does not support noise injection or timestep conditioning")
        self.block, self.conv1, self.conv2 = block, _ConvStream(block.conv1, causal) if causal else None, _ConvStream(block.conv2, causal)
        self.pending, self.received, self.emitted = None, 0, 0  # inputs not added to an output yet: frames emitted .. received - 1

    def _conv1(self, before, last):
        # outputs first .. stop - 1 of conv1 on the whole video, output o reading inputs o - 1, o, o + 1 (the first and last frames replicated)
        block, received = self.block, self.received
        first, stop = max(before - 1, 0), received if last else max(received - 1, 0)
        if stop <= first:
            return None
        x = _norm_act(block.norm1, block.non_linearity, self.pending[:, :, max(first - 1, 0) - self.emitted : min(stop + 1, received) - self.emitted])
        parts = ([x[:, :, :1]] if first == 0 else []) + [x] + ([x[:, :, -1:]] if last else [])
        x = None
        return _conv(block.conv1.conv, parts)

    def step(self, frames, last):
        block, before, x = self.block, self.received, frames.pop() if frames else None
        if x is not None:  # the inputs wait for the outputs of the convolutions, which lag behind
            self.pending = x if self.pending is None else torch.cat((self.pending, x), dim=2)
            self.received += x.shape[2]
        if self.conv1 is None:
            x = None  # in the pending inputs
            x = self._conv1(before, last)
        else:
            x = self.conv1.step([] if x is None else [_norm_act(block.norm1, block.non_linearity, x)], last)
        if x is not None:
            x = [block.dropout(_norm_act(block.norm2, block.non_linearity, x))]
        x = self.conv2.step(x or [], last)
        if x is None:
            return None
        n = x.shape[2]
        residual, self.pending = self.pending[:, :, :n], self.pending[:, :, n:].clone()
        self.emitted += n
        if not isinstance(block.norm3, nn.Identity):
            residual = block.norm3(residual.movedim(1, -1)).movedim(-1, 1)
        return x.add_(block.conv_shortcut(residual))


class _UpsampleStream:
    def __init__(self, block, causal):
        self.block, self.conv, self.pending, self.first = block, _ConvStream(block.conv, causal), None, True

    def _depth_to_space(self, x):
        stride = self.block.stride
        return rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)", p1=stride[0], p2=stride[1], p3=stride[2])

    def step(self, frames, last):
        block = self.block
        if block.residual and frames:
            self.pending = frames[0] if self.pending is None else torch.cat((self.pending, frames[0]), dim=2)
        x = self.conv.step(frames, last)
        if x is None:
            return None
        n, x = x.shape[2], self._depth_to_space(x)
        if block.residual:
            residual, self.pending = self.pending[:, :, :n], self.pending[:, :, n:].clone()
            residual = self._depth_to_space(residual).repeat(1, math.prod(block.stride) // block.out_channels_reduction_factor, 1, 1, 1)
        if block.stride[0] == 2 and self.first:  # the first frame of the video, as the decoder of the whole video drops it
            x = x[:, :, 1:]
            if block.residual:
                residual = residual[:, :, 1:]
        self.first = False
        return x.add_(residual) if block.residual else x


class _DownsampleStream:
    """SpaceToDepthDownsample (causal): the first frame of the video duplicated once, the frames are grouped by the time stride (a frame left
    without its group waits for the next chunk)."""

    def __init__(self, block):
        self.block, self.conv, self.first, self.carry = block, _ConvStream(block.conv, True), True, None

    def step(self, frames, last):
        block, x = self.block, frames.pop() if frames else None
        if x is None:
            return None
        if block.stride[0] == 2 and self.first:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)
        self.first = False
        if self.carry is not None:
            x, self.carry = torch.cat((self.carry, x), dim=2), None
        p1, p2, p3 = block.stride
        grouped = x.shape[2] // p1 * p1
        if grouped < x.shape[2]:
            x, self.carry = x[:, :, :grouped], x[:, :, grouped:].clone()
        if grouped == 0:
            return None
        x_in = rearrange(x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w", p1=p1, p2=p2, p3=p3)
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=block.group_size).mean(dim=2)
        frames, x = [x], None
        x = rearrange(self.conv.step(frames, last), "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w", p1=p1, p2=p2, p3=p3)
        return x + x_in


class EncoderStream:
    """A VideoEncoder fed by chunks of frames (one spatial tile), the first frame alone: the convolutions are causal and the downsamplers
    keep the frames of an incomplete group, so a latent frame is given as soon as its last frame has been encoded."""

    def __init__(self, encoder):
        self.encoder, self.stages = encoder, [_ConvStream(encoder.conv_in, True)]
        for block in encoder.down_blocks:
            if isinstance(block, UNetMidBlock3D):
                self.stages += [_ResnetStream(resnet, True) for resnet in block.res_blocks]
            elif isinstance(block, ResnetBlock3D):
                self.stages.append(_ResnetStream(block, True))
            elif isinstance(block, SpaceToDepthDownsample):
                self.stages.append(_DownsampleStream(block))
            else:
                raise NotImplementedError(f"LTX VAE streamed encoding does not support {type(block).__name__}")
        self.conv_out = _ConvStream(encoder.conv_out, True)

    def step(self, video):
        """Encodes the next frames ([B, C, F, H, W]): returns the normalized latent means they complete, or None."""
        encoder = self.encoder
        frames = [patchify(video, patch_size_hw=encoder.patch_size, patch_size_t=1)]
        for stage in self.stages:
            x = stage.step(frames, False)
            if x is None:
                return None
            frames = [x]
        return encoder.latent_means(self.conv_out.step([_norm_act(encoder.conv_norm_out, encoder.conv_act, frames.pop())], False))


class DecoderStream:
    """A VideoDecoder fed by chunks of latent frames (one spatial tile): each stage hands what it emits to the next one depth first, in
    pieces that keep the outputs of the stages within `piece_frames` frames (halved before a stage that doubles the frames), and `sink`
    receives the output frames as they are completed."""

    def __init__(self, decoder, piece_frames, sink):
        if decoder.timestep_conditioning:
            raise NotImplementedError("LTX VAE streamed decoding does not support timestep conditioning")
        causal = decoder.causal
        self.decoder, self.sink = decoder, sink
        self.stages = [_ConvStream(decoder.conv_in, causal)]
        for block in decoder.up_blocks:
            if isinstance(block, UNetMidBlock3D):
                self.stages += [_ResnetStream(resnet, causal) for resnet in block.res_blocks]
            elif isinstance(block, ResnetBlock3D):
                self.stages.append(_ResnetStream(block, causal))
            elif isinstance(block, DepthToSpaceUpsample):
                self.stages.append(_UpsampleStream(block, causal))
            else:
                raise NotImplementedError(f"LTX VAE streamed decoding does not support {type(block).__name__}")
        self.conv_out = _ConvStream(decoder.conv_out, causal)
        self.limits = [max(piece_frames // (stage.block.stride[0] if isinstance(stage, _UpsampleStream) else 1), 1) for stage in self.stages] + [piece_frames]

    def _push(self, index, frames, last):
        # frames: [the input of stage `index`] or [], handed over
        if frames and frames[0].shape[2] > self.limits[index]:
            pieces = list(frames.pop().split(self.limits[index], dim=2))
            while pieces:
                self._push(index, [pieces.pop(0)], last and not pieces)
            return
        if index == len(self.stages):
            decoder = self.decoder
            x = self.conv_out.step([_norm_act(decoder.conv_norm_out, decoder.conv_act, frames.pop())] if frames else [], last)
            if x is not None:
                self.sink(unpatchify(x, patch_size_hw=decoder.patch_size, patch_size_t=1))
            return
        out = self.stages[index].step(frames, last)
        out = [] if out is None else [out]
        self._push(index + 1, out, last)

    def step(self, latent, last):
        """Decodes the next latent frames ([B, C, F, h, w], `last` for the end of the video), the output frames completed going to the sink."""
        self._push(0, [self.decoder.per_channel_statistics.un_normalize(latent)], last)
