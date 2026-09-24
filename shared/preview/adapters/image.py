from __future__ import annotations

import time
from typing import Any

from PIL import Image


def _validate_latent(latent: Any, spec: Any, adapter_id: str) -> Any:
    import torch

    if not torch.is_tensor(latent) or latent.ndim != 4:
        raise ValueError("image Tiny VAE preview expects a C,B,H,W tensor")
    decoder_layout = "NTCHW" if adapter_id == "qwen_image" else "NCHW"
    if getattr(spec, "adapter_id", None) != adapter_id or getattr(spec, "decoder_layout", None) != decoder_layout:
        raise ValueError("unsupported image Tiny VAE decoder contract")
    if latent.shape[0] != int(getattr(spec, "latent_channels", 0)) or latent.shape[1] != 1 or min(latent.shape[2:]) <= 0:
        raise ValueError(f"unsupported image latent shape: {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError("image latent contains non-finite values")
    return latent


def _decoder_device_dtype(decoder: Any, latent: Any) -> tuple[Any, Any]:
    parameter = next(iter(decoder.parameters()), None)
    return (parameter.device, parameter.dtype) if parameter is not None else (latent.device, latent.dtype)


def _frame_from_nchw(decoded: Any, max_edge: int) -> Image.Image:
    import torch
    import torch.nn.functional as F

    if decoded.ndim != 4 or tuple(decoded.shape[:2]) != (1, 3) or min(decoded.shape[2:]) <= 0:
        raise ValueError(f"unexpected image Tiny VAE output shape: {tuple(decoded.shape)}")
    if not torch.isfinite(decoded).all():
        raise ValueError("image Tiny VAE output contains non-finite values")
    height, width = decoded.shape[-2:]
    scale = min(1.0, max_edge / max(height, width))
    target_size = (max(1, round(height * scale)), max(1, round(width * scale)))
    image = F.interpolate(decoded, size=target_size, mode="bilinear", align_corners=False)
    image = image.clamp(0, 1).mul(255).round().to(torch.uint8).cpu()[0]
    return Image.fromarray(image.permute(1, 2, 0).numpy())


def decode_taesd_image_latent(decoder: Any, latent: Any, *, spec: Any, max_edge: int = 512, **_: Any) -> tuple[list[Image.Image], float, int]:
    import torch

    latent = _validate_latent(latent, spec, "taesd")
    device, dtype = _decoder_device_dtype(decoder, latent)
    nchw = latent.detach().permute(1, 0, 2, 3).contiguous().to(device=device, dtype=dtype)
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = decoder(nchw)
    return [_frame_from_nchw(decoded, max_edge)], (time.perf_counter() - started) * 1000, 1


def decode_qwen_image_latent(decoder: Any, latent: Any, *, spec: Any, max_edge: int = 512, parallel: bool = True, **_: Any) -> tuple[list[Image.Image], float, int]:
    import torch

    latent = _validate_latent(latent, spec, "qwen_image")
    device, dtype = _decoder_device_dtype(decoder, latent)
    ntchw = latent.detach()[:, 0][None, None].contiguous().to(device=device, dtype=dtype)
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = decoder.decode_video(ntchw, parallel=parallel, show_progress_bar=False)
    if decoded.ndim != 5 or tuple(decoded.shape[:3]) != (1, 1, 3):
        raise ValueError(f"unexpected Qwen Tiny VAE output shape: {tuple(decoded.shape)}")
    return [_frame_from_nchw(decoded[:, 0], max_edge)], (time.perf_counter() - started) * 1000, 1
