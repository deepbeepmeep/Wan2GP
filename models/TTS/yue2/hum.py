"""Hum preprocessing and conditioning for the Mothersuperior YuE2 adapter.

Oobleck encoder uses the same MIT-licensed building blocks as vae.py.
Hum carrier follows Mothersuperior/YuE2-hum-to-song (CC BY-NC 4.0).
"""
import math
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .vae import ResidualUnit, get_activation, _dependency_interval, _output_length


def open_hum_score(abc):
    lines = abc.strip().splitlines()
    while lines and re.fullmatch(r"(?:V:\s*(?:Vocal|Ins))|(?:Z\d*\|)|", lines[-1].strip()):
        lines.pop()
    if not lines:
        raise ValueError("The hum did not produce a usable melody.")
    return "\n".join(lines) + "\n"


def prosody_carrier(filename, abort_fn, report):
    import librosa
    from scipy.signal import butter, sosfiltfilt

    report("Tracking Hum Pitch")
    audio, _ = librosa.load(filename, sr=48000, mono=True)
    if len(audio) < 1920 or not np.isfinite(audio).all():
        raise ValueError("Supply a finite humming recording of at least 0.04 seconds.")
    if abort_fn():
        raise InterruptedError("Hum preprocessing interrupted")
    f0, _, _ = librosa.pyin(audio, fmin=65, fmax=1000, sr=48000, hop_length=512, frame_length=2048)
    if abort_fn():
        raise InterruptedError("Hum preprocessing interrupted")
    valid = np.flatnonzero(np.isfinite(f0))
    if not len(valid):
        raise ValueError("No voiced pitch was found. Supply a clear human hum without accompaniment.")
    f0[:valid[0]] = f0[valid[0]]
    indices = np.maximum.accumulate(np.where(np.isfinite(f0), np.arange(len(f0)), 0))
    pitch = np.interp(np.arange(len(audio)) / 48000, np.arange(len(f0)) * 512 / 48000, f0[indices])
    report("Preparing Hum Carrier")
    envelope = sosfiltfilt(butter(4, 30, btype="low", fs=48000, output="sos"), np.abs(audio))
    envelope = sosfiltfilt(butter(2, 80, btype="low", fs=48000, output="sos"), envelope).clip(0)
    carrier = envelope * np.sin(2 * np.pi * np.cumsum(pitch) / 48000)
    carrier = (carrier / (np.abs(carrier).max() + 1e-9) * .9).astype(np.float32)
    return torch.from_numpy(carrier)[None, None].expand(1, 2, -1).contiguous()


class HumProjections(nn.Module):
    _convertWeightsFloatTo = None
    _model_dtype = torch.float32
    depths = (0, 7, 14, 21)

    def __init__(self):
        super().__init__()
        self.hum_proj = nn.ModuleList([nn.Linear(64, 2048) for _ in self.depths])
        self.abort_fn = lambda: False

    def forward(self, carrier):
        carrier = F.pad(carrier, (0, 0, 1, 1))[None].to(device=self.hum_proj[0].weight.device, dtype=self.hum_proj[0].weight.dtype)
        features = {}
        for depth, projection in zip(self.depths, self.hum_proj):
            features[depth] = projection(carrier)
            if self.abort_fn():
                raise InterruptedError("Hum conditioning interrupted")
        return features


class HumEncoder(nn.Module):
    _convertWeightsFloatTo = None
    _model_dtype = torch.float32
    _offload_hooks = ["encode"]

    def __init__(self):
        super().__init__()
        widths = [64, 64, 128, 256, 512, 1024, 2048]
        strides = [2, 2, 4, 4, 5, 6]
        layers = [nn.Conv1d(2, 64, 7, padding=3)]
        for a, b, stride in zip(widths, widths[1:], strides):
            # Keep upstream's nested key layout while using shared encoder math.
            block = nn.Module()
            block.layers = nn.Sequential(*(ResidualUnit(a, a, dilation, "snake") for dilation in (1, 3, 9)), get_activation("snake", a), nn.Conv1d(a, b, 2 * stride, stride=stride, padding=math.ceil(stride / 2)))
            layers.append(block)
        layers.extend([get_activation("snake", 2048), nn.Conv1d(2048, 128, 3, padding=1)])
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(layers)

    @torch.inference_mode()
    def encode(self, audio, core_frames, abort_fn, report):
        sequence = [block.layers if hasattr(block, "layers") else block for block in self.encoder.layers]
        length = audio.shape[-1]
        frames = length
        for block in sequence:
            frames = _output_length(block, frames)
        core = core_frames or frames
        result = torch.empty((frames, 64), device="cpu", dtype=torch.float32)
        total = (frames + core - 1) // core
        for tile, start in enumerate(range(0, frames, core)):
            end = min(start + core, frames)
            low, high = start, end - 1
            for block in reversed(sequence):
                low, high = _dependency_interval(block, low, high)
            first = max(0, low // 1920 * 1920)
            last = min(length, (high // 1920 + 1) * 1920)
            x = audio[..., first:last].to(device=self.encoder.layers[0].weight.device, dtype=torch.float32)
            for block in sequence:
                x = block(x)
                if abort_fn():
                    raise InterruptedError("Hum encoding interrupted")
            offset = start - first // 1920
            result[start:end] = x[0, :64, offset:offset + end - start].T.cpu()
            del x
            report(tile + 1, total)
        return result
