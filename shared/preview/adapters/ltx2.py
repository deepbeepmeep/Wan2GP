from __future__ import annotations

import time
from typing import Any

from PIL import Image


def uniform_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    frame_count = int(frame_count)
    sample_count = max(1, int(sample_count))
    if frame_count <= sample_count:
        return list(range(max(0, frame_count)))
    if sample_count == 1:
        return [0]
    return [round(index * (frame_count - 1) / (sample_count - 1)) for index in range(sample_count)]


def preview_sample_count(frame_count: int, preview_fps: int, source_fps: float | None = None, duration_seconds: float | None = None) -> int:
    frame_count = max(0, int(frame_count))
    if source_fps and source_fps > 0:
        desired = round((frame_count - 1) / source_fps * preview_fps)
    elif duration_seconds and duration_seconds > 0:
        desired = round(duration_seconds * preview_fps)
    else:
        desired = frame_count
    return min(frame_count, 1024, max(2 if frame_count > 1 else 1, desired)) if frame_count else 0


def decode_ltx2_latent(decoder: Any, latent: Any, *, spec: Any = None, max_edge: int = 512, preview_fps: int = 16, source_fps: float | None = None, duration_seconds: float | None = None, parallel: bool = True) -> tuple[list[Image.Image], float, int]:
    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(latent) or latent.ndim != 4:
        raise ValueError("LTX Tiny VAE preview expects a C,T,H,W tensor")
    latent_channels = int(getattr(spec, "latent_channels", 128))
    if getattr(spec, "adapter_id", "ltx2") != "ltx2" or getattr(spec, "decoder_layout", "NTCHW") != "NTCHW":
        raise ValueError("unsupported LTX Tiny VAE decoder contract")
    if latent.shape[0] != latent_channels or min(latent.shape[1:]) <= 0:
        raise ValueError(f"unsupported LTX latent shape: {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError("LTX latent contains non-finite values")
    device = next(decoder.parameters()).device
    dtype = next(decoder.parameters()).dtype
    ntchw = latent.detach().unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().to(device=device, dtype=dtype)
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = decoder.decode_video(ntchw, parallel=parallel, show_progress_bar=False)
    if decoded.ndim != 5 or decoded.shape[0] != 1 or decoded.shape[2] != 3 or decoded.shape[1] < 1:
        raise ValueError(f"unexpected TAEHV output shape: {tuple(decoded.shape)}")
    if not torch.isfinite(decoded).all():
        raise ValueError("TAEHV output contains non-finite values")
    temporal_scale = 2 ** sum(bool(value) for value in getattr(spec, "decoder_time_upscale", (True, True, True)))
    expected_frames = temporal_scale * (latent.shape[1] - 1) + 1
    if decoded.shape[1] != expected_frames:
        raise ValueError(f"unexpected TAEHV frame count: got {decoded.shape[1]}, expected {expected_frames}")
    frame_indices = uniform_frame_indices(decoded.shape[1], preview_sample_count(decoded.shape[1], preview_fps, source_fps, duration_seconds))
    selected = decoded[:, frame_indices]
    height, width = selected.shape[-2:]
    scale = min(1.0, max_edge / max(height, width))
    target_size = (max(1, round(height * scale)), max(1, round(width * scale)))
    selected = F.interpolate(selected.reshape(-1, 3, height, width), size=target_size, mode="bilinear", align_corners=False)
    selected = selected.clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    frames = [Image.fromarray(frame.permute(1, 2, 0).numpy()) for frame in selected]
    return frames, (time.perf_counter() - started) * 1000, decoded.shape[1]
