from __future__ import annotations

import hashlib
import inspect
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from .registry import PreviewDecoderSpec


class PreviewDecoderError(RuntimeError):
    pass


_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {}
_LOG = logging.getLogger(__name__)


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
    # Module initialization must not consume the generation's CPU RNG state.
    with _LOCK, torch.random.fork_rng(devices=[]):
        if cache_key in _CACHE:
            return _CACHE[cache_key]
        if spec.adapter_id == "taesd":
            from .vendor.taesd import Decoder

            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
            model = Decoder(spec.latent_channels, use_midblock_gn=(spec.decoder_id == "taef2"))
            model.load_state_dict(state_dict, strict=True)
        else:
            state_dict = load_file(str(path), device="cpu")
        if spec.adapter_id == "h3":
            from .adapters.h3 import build_h3_decoder

            model = build_h3_decoder(state_dict)
        elif spec.adapter_id != "taesd":
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


def download_decoder(spec: PreviewDecoderSpec, progress_callback=None, *, gen: dict[str, Any] | None = None) -> str:
    """Install verified weights without requiring the separate progress PR.

    Native WanGP progress arrives through ``gen``. The optional callback gets
    dicts with both native ``completed`` and the preview UI's ``current`` key.
    A legacy two-argument downloader remains usable, but cannot cancel mid-file.
    """
    from shared.utils.download import download_file

    context = gen if gen is not None else {}

    def check_cancelled() -> None:
        abort_callback = context.get("abort_callback")
        if context.get("abort", False) or (abort_callback is not None and abort_callback()):
            raise PreviewDecoderError("Tiny VAE decoder download cancelled")

    check_cancelled()
    local_path = spec.local_path()
    if local_path:
        target = Path(local_path)
    else:
        from shared.utils import files_locator as fl

        target = Path(fl.get_smart_download_location(spec.filename, spec.target_dir))
    if target.is_file() and validate_weight(target, spec)[0]:
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    previous = context.get("download_progress_callback")
    had_previous = "download_progress_callback" in context
    callback_active = callable(progress_callback)

    def relay(report) -> None:
        nonlocal callback_active
        if callable(previous):
            previous(report)
        if not callback_active or report is None:
            return
        try:
            value = dict(report)
            value["current"] = value.get("completed", value.get("current", 0))
            value["speed_bps"] = value.get("speed", value.get("speed_bps"))
            progress_callback(value)
        except Exception:
            callback_active = False
            _LOG.warning("Tiny VAE install progress callback disabled", exc_info=True)

    # Only the signature is adapted. Hugging Face transfers, cache, retries and
    # cancellation remain entirely owned by WanGP; there is no transfer patch.
    parameters = inspect.signature(download_file).parameters
    native_context = "gen" in parameters
    if callback_active:
        context["download_progress_callback"] = relay
    try:
        with tempfile.TemporaryDirectory(prefix=".preview-install-", dir=target.parent) as staging:
            candidate = Path(staging) / spec.filename
            check_cancelled()
            if native_context:
                download_file(spec.source_url, str(candidate), gen=context)
            else:
                relay({"filename": spec.filename, "completed": 0, "total": None})
                download_file(spec.source_url, str(candidate))
            check_cancelled()
            valid, reason = validate_weight(candidate, spec)
            if not valid:
                raise PreviewDecoderError(reason)
            check_cancelled()
            # Same-filesystem replace publishes only fully validated weights;
            # failed/cancelled repairs leave the prior destination untouched.
            os.replace(candidate, target)
            if not native_context:
                relay({"filename": spec.filename, "completed": spec.size_bytes, "total": spec.size_bytes})
        return str(target)
    finally:
        if callback_active or context.get("download_progress_callback") is relay:
            if had_previous:
                context["download_progress_callback"] = previous
            else:
                context.pop("download_progress_callback", None)
