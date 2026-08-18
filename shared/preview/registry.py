from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreviewDecoderSpec:
    decoder_id: str
    filename: str
    sha256: str
    size_bytes: int
    latent_channels: int
    patch_size: int
    encoder_time_downscale: tuple[bool, ...]
    decoder_time_upscale: tuple[bool, ...]
    compatible_architectures: frozenset[str]
    adapter_id: str
    source_url: str
    target_dir: str = "preview_decoders/taehv"
    decoder_layout: str = "NTCHW"

    @property
    def relative_path(self) -> str:
        return str(Path(self.target_dir) / self.filename)

    def local_path(self) -> str | None:
        try:
            from shared.utils import files_locator as fl

            return fl.locate_file(self.relative_path, error_if_none=False)
        except Exception:
            # Keep capability discovery import-safe in lightweight tooling that
            # does not have the optional GPU runtime installed.
            for root in ("ckpts", "."):
                candidate = Path(root) / self.relative_path
                if candidate.is_file():
                    return str(candidate)
            return None


TAELTX23 = PreviewDecoderSpec(
    decoder_id="taeltx2_3",
    filename="taeltx2_3.safetensors",
    sha256="f0773b4e3e57318e6aa4dd4a35e1d16213a5f160fbc0376163f06888bbcbe246",
    size_bytes=23_531_296,
    latent_channels=128,
    patch_size=4,
    encoder_time_downscale=(True, True, True),
    decoder_time_upscale=(True, True, True),
    compatible_architectures=frozenset({"ltx2_22B"}),
    adapter_id="ltx2",
    source_url="https://raw.githubusercontent.com/madebyollin/taehv/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeltx2_3.safetensors",
)

TAEH3 = PreviewDecoderSpec(
    decoder_id="taeh3",
    filename="taeh3.safetensors",
    sha256="f0f60fa072089997f817402098c2fd90777cb2660dd79cf5df42fc1e3e08e527",
    size_bytes=9_791_388,
    latent_channels=24,
    patch_size=1,
    encoder_time_downscale=(False, False, False),
    decoder_time_upscale=(False, False, False),
    compatible_architectures=frozenset(
        {
            "minimax_h3_fl2va",
            "minimax_h3_fl2va_pruned",
            "minimax_h3_ref2va",
            "minimax_h3_ref2va_pruned",
        }
    ),
    adapter_id="h3",
    source_url="https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/a213ac8bf2f148b4f32372279a7f207846978900/vae_approx/taeh3.safetensors",
    target_dir="preview_decoders/taeh3",
)

DECODERS = {spec.decoder_id: spec for spec in (TAELTX23, TAEH3)}


def get_decoder_for_model(model_type: str, model_def: dict[str, Any] | None = None) -> PreviewDecoderSpec | None:
    architecture = str((model_def or {}).get("architecture") or "").strip()
    capabilities = (model_def or {}).get("capabilities", {})
    live_preview = capabilities.get("live_preview", {}) if isinstance(capabilities, dict) else {}
    if not isinstance(live_preview, dict):
        return None
    for spec in DECODERS.values():
        if (
            architecture in spec.compatible_architectures
            and spec.decoder_id in set(live_preview.get("decoders", ()))
            and "tae" in set(live_preview.get("modes", ()))
        ):
            return spec
    return None


def decoder_capability(model_type: str, model_def: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = get_decoder_for_model(model_type, model_def)
    if spec is None:
        return {"modes": ["off", "rgb"], "decoders": [], "tiny_vae_available": False}
    path = spec.local_path()
    valid = False
    reason = ""
    if path:
        from .loader import validate_weight

        valid, reason = validate_weight(path, spec)
    return {
        "modes": ["off", "rgb", "tae"] if valid else ["off", "rgb"],
        "decoders": [spec.decoder_id],
        "tiny_vae_available": valid,
        "decoder_id": spec.decoder_id,
        "weight_path": path,
        "unavailable_reason": reason,
    }
