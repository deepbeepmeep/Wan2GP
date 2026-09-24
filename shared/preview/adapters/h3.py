from __future__ import annotations

import time
from typing import Any

from PIL import Image

from .ltx2 import preview_sample_count, uniform_frame_indices


def build_h3_decoder(state_dict: dict[str, Any]) -> Any:
    import torch
    import torch.nn as nn

    def conv(n_in: int, n_out: int, **kwargs: Any) -> nn.Conv2d:
        return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

    class Clamp(nn.Module):
        def forward(self, value: Any) -> Any:
            return torch.tanh(value / 3) * 3

    class Block(nn.Module):
        def __init__(self, n_in: int, n_out: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
            self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
            self.fuse = nn.ReLU()

        def forward(self, value: Any) -> Any:
            return self.fuse(self.conv(value) + self.skip(value))

    by_index: dict[int, dict[str, Any]] = {}
    for key, value in state_dict.items():
        head, separator, rest = key.partition(".")
        if not separator or not head.isdigit():
            raise ValueError(f"unexpected taeh3 state key: {key}")
        by_index.setdefault(int(head), {})[rest] = value

    modules: list[nn.Module] = []
    for index in range(max(by_index) + 1):
        entry = by_index.get(index)
        if entry is None:
            modules.append(Clamp() if index == 0 else nn.ReLU() if index == 2 else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            weight = entry["conv.0.weight"]
            modules.append(Block(weight.shape[1], weight.shape[0]))
        elif "weight" in entry:
            weight = entry["weight"]
            modules.append(conv(weight.shape[1], weight.shape[0], bias="bias" in entry))
        else:
            raise ValueError(f"unexpected taeh3 module keys at {index}: {sorted(entry)}")

    decoder = nn.Sequential(*modules)
    decoder.load_state_dict(state_dict, strict=True)

    class H3Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = decoder

        def decode_video(self, value: Any, parallel: bool = True, show_progress_bar: bool = False) -> Any:
            del show_progress_bar
            batch, frames, channels, height, width = value.shape
            flat = value.reshape(batch * frames, channels, height, width)
            if parallel:
                decoded = self.decoder(flat)
            else:
                decoded = torch.cat([self.decoder(frame) for frame in flat.split(1)], dim=0)
            return decoded.unflatten(0, (batch, frames))

    return H3Decoder()


def decode_h3_latent(decoder: Any, latent: Any, *, spec: Any = None, max_edge: int = 512, preview_fps: int = 16, source_fps: float | None = None, duration_seconds: float | None = None, parallel: bool = True) -> tuple[list[Image.Image], float, int]:
    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(latent) or latent.ndim != 4:
        raise ValueError("H3 Tiny VAE preview expects a C,T,H,W tensor")
    if getattr(spec, "adapter_id", None) != "h3" or getattr(spec, "decoder_layout", "NTCHW") != "NTCHW":
        raise ValueError("unsupported H3 Tiny VAE decoder contract")
    if latent.shape[0] != 24 or min(latent.shape[1:]) <= 0:
        raise ValueError(f"unsupported H3 latent shape: {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError("H3 latent contains non-finite values")
    device = next(decoder.parameters()).device
    dtype = next(decoder.parameters()).dtype
    ntchw = latent.detach().unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().to(device=device, dtype=dtype)
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = decoder.decode_video(ntchw, parallel=parallel, show_progress_bar=False)
    if decoded.ndim != 5 or decoded.shape[0] != 1 or decoded.shape[2] != 3 or decoded.shape[1] < 1:
        raise ValueError(f"unexpected taeh3 output shape: {tuple(decoded.shape)}")
    if not torch.isfinite(decoded).all():
        raise ValueError("taeh3 output contains non-finite values")
    expected_frames = latent.shape[1]
    if decoded.shape[1] != expected_frames:
        raise ValueError(f"unexpected taeh3 frame count: got {decoded.shape[1]}, expected {expected_frames}")
    frame_indices = uniform_frame_indices(decoded.shape[1], preview_sample_count(decoded.shape[1], preview_fps, source_fps, duration_seconds))
    selected = decoded[:, frame_indices]
    height, width = selected.shape[-2:]
    scale = min(1.0, max_edge / max(height, width))
    target_size = (max(1, round(height * scale)), max(1, round(width * scale)))
    selected = F.interpolate(selected.reshape(-1, 3, height, width), size=target_size, mode="bilinear", align_corners=False)
    selected = selected.clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    frames = [Image.fromarray(frame.permute(1, 2, 0).numpy()) for frame in selected]
    return frames, (time.perf_counter() - started) * 1000, decoded.shape[1]
