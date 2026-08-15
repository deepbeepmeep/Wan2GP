"""Decoder-only subset of TAEHV, vendored from madebyollin/taehv.

Source revision: 62f7591f59dfbb4c3c02b7a621d180a9eeaba26c
The implementation keeps the upstream module names and state-dict layout so
strict safetensors loading remains possible, while omitting unused streaming
and encoder runtime helpers.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        return self.conv(x.reshape(-1, self.stride * x.shape[1], x.shape[2], x.shape[3]))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x.reshape(-1, x.shape[1] // self.stride, x.shape[2], x.shape[3])


def _apply(model, x, parallel):
    if parallel:
        n, t, c, h, w = x.shape
        flat = x.reshape(n * t, c, h, w)
        for block in model:
            if isinstance(block, MemBlock):
                current = flat.reshape(n, flat.shape[0] // n, flat.shape[1], flat.shape[2], flat.shape[3])
                memory = F.pad(current, (0, 0, 0, 0, 0, 0, 1, 0))[:, : current.shape[1]].reshape(flat.shape)
                flat = block(flat, memory)
            else:
                flat = block(flat)
        return flat.reshape(n, flat.shape[0] // n, flat.shape[1], flat.shape[2], flat.shape[3])

    queues = [(frame, 0) for frame in x.unbind(1)]
    memory = [None] * len(model)
    output = []
    while queues:
        value, index = queues.pop(0)
        if index == len(model):
            output.append(value.unsqueeze(1))
            continue
        block = model[index]
        if isinstance(block, MemBlock):
            previous = memory[index]
            value_new = block(value, value * 0 if previous is None else previous)
            memory[index] = value
            queues.insert(0, (value_new, index + 1))
        elif isinstance(block, TPool):
            pending = memory[index]
            if pending is None:
                pending = memory[index] = []
            pending.append(value)
            if len(pending) == block.stride:
                value_new = block(torch.cat(pending, 1).view(-1, value.shape[1], value.shape[2], value.shape[3]))
                memory[index] = []
                queues.insert(0, (value_new, index + 1))
        elif isinstance(block, TGrow):
            value_new = block(value)
            chunks = value_new.view(-1, block.stride * value_new.shape[1], value_new.shape[2], value_new.shape[3]).chunk(block.stride, 1)
            for chunk in reversed(chunks):
                queues.insert(0, (chunk, index + 1))
        else:
            queues.insert(0, (block(value), index + 1))
    return torch.cat(output, 1)


class TAEHV(nn.Module):
    def __init__(self, checkpoint_path=None, encoder_time_downscale=(True, True, False), decoder_time_upscale=(False, True, True), decoder_space_upscale=(True, True, True), patch_size=1, latent_channels=16):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size ** 2, 64), nn.ReLU(inplace=True),
            TPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], self.image_channels * self.patch_size ** 2),
        )
        self.t_upscale = 2 ** sum(layer.stride == 2 for layer in self.decoder if isinstance(layer, TGrow))
        self.frames_to_trim = self.t_upscale - 1

    def patch_tgrow_layers(self, state_dict):
        expected = self.state_dict()
        for index, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{index}.conv.weight"
                if state_dict[key].shape[0] > expected[key].shape[0]:
                    state_dict[key] = state_dict[key][-expected[key].shape[0] :]
        return state_dict

    def postprocess_output_frames(self, x):
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x.clamp_(0, 1)

    def decode_video(self, x, parallel=True, show_progress_bar=False):
        decoded = _apply(self.decoder, x, parallel)
        decoded = self.postprocess_output_frames(decoded)
        return decoded[:, self.frames_to_trim :]
