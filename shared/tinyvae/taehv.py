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


SEQUENTIAL_PIECE_ELEMENTS = 2**23  # sequential decoding: frames created by the TGrow layers run the following layers in pieces of at most this many elements


def _apply(model, x, parallel, output_indices=None, abort_check=None, output_transform=None):
    if parallel:
        n, t, c, h, w = x.shape
        flat = x.reshape(n * t, c, h, w)
        for block in model:
            if abort_check is not None and abort_check():
                return None
            if isinstance(block, MemBlock):
                current = flat.reshape(n, flat.shape[0] // n, flat.shape[1], flat.shape[2], flat.shape[3])
                memory = F.pad(current, (0, 0, 0, 0, 0, 0, 1, 0))[:, : current.shape[1]].reshape(flat.shape)
                flat = block(flat, memory)
            else:
                flat = block(flat)
        return flat.reshape(n, flat.shape[0] // n, flat.shape[1], flat.shape[2], flat.shape[3])

    # The frames of the single video in temporal order, one latent frame at a time: the frames the TGrow layers create run the following
    # layers together, in pieces of at most SEQUENTIAL_PIECE_ELEMENTS. Each MemBlock gets every frame's predecessor at its own input, and
    # the layers after the last temporal one, which work frame by frame, only run on the frames to output.
    last_temporal = max((index for index, block in enumerate(model) if isinstance(block, (MemBlock, TGrow))), default=len(model) - 1)
    memory = [None] * len(model)
    output = []
    output_no = 0

    def run(value, start):
        nonlocal output_no
        for index in range(start, len(model)):
            if value.shape[0] > 1 and value.numel() > SEQUENTIAL_PIECE_ELEMENTS:
                for piece in value.split(max(1, SEQUENTIAL_PIECE_ELEMENTS // value[0].numel())):
                    run(piece, index)
                return
            block = model[index]
            if isinstance(block, MemBlock):
                previous = torch.zeros_like(value[:1]) if memory[index] is None else memory[index]
                past = previous if value.shape[0] == 1 else torch.cat([previous, value[:-1]])
                memory[index] = value if value.shape[0] == 1 else value[-1:].clone()  # a view would keep the whole piece alive
                value = block(value, past)
            else:
                value = block(value)
            if index == last_temporal:
                frames = range(output_no, output_no + value.shape[0])
                output_no += value.shape[0]
                if output_indices is not None:
                    value = value[[frame - frames.start for frame in frames if frame in output_indices]]
                    if value.shape[0] == 0:
                        return
        output.extend((frame if output_transform is None else output_transform(frame)).unsqueeze(1) for frame in value.split(1))

    for value in x[0].split(1):
        if abort_check is not None and abort_check():
            return None
        run(value, 0)
    return torch.cat(output, 1)


class TAEHV(nn.Module):
    def __init__(self, checkpoint_path=None, encoder_time_downscale=(True, True, False), decoder_time_upscale=(False, True, True), decoder_space_upscale=(True, True, True), patch_size=1, latent_channels=16):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
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

    def decode_video(self, x, parallel=True, show_progress_bar=False, output_indices=None, abort_check=None, output_transform=None):
        selected = None if output_indices is None else {i + self.frames_to_trim for i in output_indices}
        transform = None if output_transform is None else lambda frame: output_transform(self.postprocess_output_frames(frame))
        decoded = _apply(self.decoder, x, parallel, selected, abort_check, transform)
        if decoded is None:
            return None
        if output_transform is None:
            decoded = self.postprocess_output_frames(decoded)
        return decoded[:, self.frames_to_trim :] if output_indices is None else decoded
