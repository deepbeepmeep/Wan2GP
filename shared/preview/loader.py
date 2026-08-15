from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

from .registry import PreviewDecoderSpec


class PreviewDecoderError(RuntimeError):
    pass


_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_weight(path: str | os.PathLike[str], spec: PreviewDecoderSpec) -> tuple[bool, str]:
    target = Path(path)
    if not target.is_file():
        return False, f"Tiny VAE decoder is missing: {spec.relative_path}"
    if target.stat().st_size != spec.size_bytes:
        return False, f"Tiny VAE decoder size mismatch for {target.name}"
    if _sha256(target) != spec.sha256:
        return False, f"Tiny VAE decoder SHA-256 mismatch for {target.name}"
    return True, ""


def load_decoder(path: str | os.PathLike[str], spec: PreviewDecoderSpec, *, device: str = "cpu", dtype: Any = None) -> Any:
    valid, reason = validate_weight(path, spec)
    if not valid:
        raise PreviewDecoderError(reason)
    import torch
    from safetensors.torch import load_file
    target_device = str(device or "cpu")
    target_dtype = dtype or (torch.float16 if target_device.startswith("cuda") else torch.float32)
    cache_key = f"{Path(path).resolve()}::{target_device}::{target_dtype}"
    with _LOCK:
        if cache_key in _CACHE:
            return _CACHE[cache_key]
        state_dict = load_file(str(path), device="cpu")
        if spec.adapter_id == "h3":
            from .adapters.h3 import build_h3_decoder

            model = build_h3_decoder(state_dict)
        else:
            from .vendor.taehv import TAEHV

            model = TAEHV(
                checkpoint_path=None,
                patch_size=spec.patch_size,
                latent_channels=spec.latent_channels,
                encoder_time_downscale=spec.encoder_time_downscale,
                decoder_time_upscale=spec.decoder_time_upscale,
                decoder_space_upscale=(True, True, True),
            )
            model.load_state_dict(model.patch_tgrow_layers(state_dict), strict=True)
        model.eval().requires_grad_(False).to(device=target_device, dtype=target_dtype)
        _CACHE[cache_key] = model
        return model


def unload_decoders() -> None:
    with _LOCK:
        for model in _CACHE.values():
            try:
                model.to("cpu")
            except Exception:
                pass
        _CACHE.clear()


def download_decoder(spec: PreviewDecoderSpec, progress_callback=None) -> str:
    from shared.utils.download import download_file

    local_path = spec.local_path()
    if local_path:
        target = Path(local_path)
    else:
        from shared.utils import files_locator as fl

        target = Path(fl.get_smart_download_location(spec.filename, spec.target_dir))
    target.parent.mkdir(parents=True, exist_ok=True)
    download_file(spec.source_url, str(target), progress_callback=progress_callback)
    valid, reason = validate_weight(target, spec)
    if not valid:
        try:
            target.unlink()
        except OSError:
            pass
        raise PreviewDecoderError(reason)
    return str(target)
