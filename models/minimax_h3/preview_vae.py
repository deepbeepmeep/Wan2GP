"""Tiny decoder used by the optional accurate MiniMax H3 sampling preview.

The decoder weights are published with the MIT-licensed
ComfyUI-MiniMaxH3-PreviewOverride project. This keeps the decoder separate
from the full H3 VAE, so it can be used during sampling without a full-VAE
preview decode.
"""

from __future__ import annotations

import logging
import os
from collections import deque

import torch
from PIL import Image
from safetensors.torch import load_file

from shared.utils import files_locator as fl
from shared.utils.download import download_file


PREVIEW_DECODER_FILENAME = "minimax_h3/taeh3_decoder.safetensors"
PREVIEW_DECODER_URL = (
    "https://github.com/simsim9-stack/ComfyUI-MiniMaxH3-PreviewOverride/"
    "raw/main/minivae/taeh3_decoder.safetensors"
)


def _conv(n_in, n_out, **kwargs):
    return torch.nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(torch.nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class _MemBlock(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = torch.nn.Sequential(
            _conv(n_in * 2, n_out), torch.nn.ReLU(inplace=True),
            _conv(n_out, n_out), torch.nn.ReLU(inplace=True),
            _conv(n_out, n_out),
        )
        self.skip = torch.nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else torch.nn.Identity()
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, x, previous):
        return self.act(self.conv(torch.cat((x, previous), dim=1)) + self.skip(x))


class _TemporalGrow(torch.nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.stride = stride
        self.conv = torch.nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x.reshape(-1, x.shape[1] // self.stride, x.shape[2], x.shape[3])


class H3TinyPreviewDecoder(torch.nn.Module):
    """The decoder-only 24-channel TAEHV architecture used by the published H3 TAE."""

    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            _Clamp(), _conv(24, 256), torch.nn.ReLU(inplace=True),
            _MemBlock(256, 256), _MemBlock(256, 256), _MemBlock(256, 256),
            torch.nn.Upsample(scale_factor=2), _TemporalGrow(256, 1), _conv(256, 128, bias=False),
            _MemBlock(128, 128), _MemBlock(128, 128), _MemBlock(128, 128),
            torch.nn.Upsample(scale_factor=2), _TemporalGrow(128, 2), _conv(128, 64, bias=False),
            _MemBlock(64, 64), _MemBlock(64, 64), _MemBlock(64, 64),
            torch.nn.Upsample(scale_factor=2), _TemporalGrow(64, 2), _conv(64, 64, bias=False),
            torch.nn.ReLU(inplace=True), _conv(64, 3),
        )

    def _decode_sequential(self, x, output_device):
        """Decode one temporal slice at a time, matching the original TAEHV path."""
        batch, frames, _, _, _ = x.shape
        work_queue = deque((frame.squeeze(1), 0) for frame in x.chunk(frames, dim=1))
        memories = [None] * len(self.decoder)
        outputs = []

        while work_queue:
            current, block_index = work_queue.popleft()
            if block_index == len(self.decoder):
                outputs.append(current.to(output_device))
                continue

            block = self.decoder[block_index]
            if isinstance(block, _MemBlock):
                if memories[block_index] is None:
                    next_value = block(current, current * 0)
                else:
                    next_value = block(current, memories[block_index])
                memories[block_index] = current.detach().clone()
                work_queue.appendleft((next_value, block_index + 1))
            elif isinstance(block, _TemporalGrow):
                grown = block(current)
                _, channels, height, width = grown.shape
                for next_value in reversed(grown.view(batch, block.stride * channels, height, width).chunk(block.stride, dim=1)):
                    work_queue.appendleft((next_value, block_index + 1))
            else:
                work_queue.appendleft((block(current), block_index + 1))

        return torch.stack(outputs, dim=1)

    def decode(self, latents, output_device="cpu"):
        """Decode ``[batch, channels, frames, height, width]`` H3 latents to RGB frames."""
        if latents.ndim != 5 or latents.shape[1] != 24:
            raise ValueError(f"Expected [B, 24, T, H, W] H3 latents, got {tuple(latents.shape)}")
        x = self._decode_sequential(latents.movedim(1, 2), output_device)
        # TAEHV produces three warm-up frames before its temporal context settles.
        return x[:, 3:]


def ensure_preview_decoder_file():
    path = fl.locate_file(PREVIEW_DECODER_FILENAME, error_if_none=False)
    if path is not None:
        return path
    path = fl.get_download_location(PREVIEW_DECODER_FILENAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    download_file(PREVIEW_DECODER_URL, path)
    return path


class AccurateH3Preview:
    """Lazy CPU-resident preview decoder with a compact contact-sheet result."""

    def __init__(self):
        self._decoder = None
        self._disabled = False

    def prepare(self):
        if self._decoder is not None or self._disabled:
            return self._decoder is not None
        try:
            decoder = H3TinyPreviewDecoder()
            decoder.load_state_dict(load_file(ensure_preview_decoder_file(), device="cpu"), strict=True)
            self._decoder = decoder.eval().requires_grad_(False)
            return True
        except Exception as exc:
            self._disabled = True
            logging.warning("[MiniMax H3] Tiny VAE preview decoder is unavailable; using the standard preview: %s", exc)
            return False

    @torch.inference_mode()
    def decode(self, latents, preview_frames=4, device="cpu"):
        if not self.prepare() or not torch.is_tensor(latents):
            return None
        if device == "cuda" and latents.device.type == "cuda":
            decode_device = latents.device
            decode_dtype = latents.dtype
        else:
            # H3's full model already occupies most of the GPU.  The TAE
            # decoder's intermediate feature maps can require several GB, so
            # the CPU path avoids evicting the generation model.
            decode_device = torch.device("cpu")
            decode_dtype = torch.float32
        decoder = self._decoder.to(device=decode_device, dtype=decode_dtype)
        latents = latents.detach().to(device=decode_device, dtype=decode_dtype)
        try:
            frames = decoder.decode(latents.unsqueeze(0), output_device="cpu")
            if frames.shape[1] == 0:
                return None
            count = min(int(preview_frames), frames.shape[1])
            indices = torch.linspace(0, frames.shape[1] - 1, count, device=frames.device).round().long()
            frames = (
                frames[0, indices]
                .permute(0, 2, 3, 1)
                .add_(1.0)
                .mul_(127.5)
                .clamp_(0, 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            images = [Image.fromarray(frame) for frame in frames]
            canvas = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)))
            offset = 0
            for image in images:
                canvas.paste(image, (offset, 0))
                offset += image.width
            return canvas
        except Exception as exc:
            logging.warning("[MiniMax H3] Tiny VAE preview decode failed; using the standard preview: %s", exc)
            self._disabled = True
            return None
        finally:
            # The CPU mode keeps the decoder in system RAM between steps. In
            # GPU mode it stays resident so each preview does not repeatedly
            # transfer the decoder weights back and forth.
            if decode_device.type == "cpu":
                self._decoder.to(device="cpu")


__all__ = ["AccurateH3Preview", "PREVIEW_DECODER_FILENAME", "PREVIEW_DECODER_URL"]
