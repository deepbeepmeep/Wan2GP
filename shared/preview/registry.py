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
    compatible_model_types: frozenset[str]
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
    compatible_model_types=frozenset(
        {
            "ltx2_22B",
            "ltx2_22B_distilled",
            "ltx2_22B_1_1",
            "ltx2_22B_distilled_1_1",
        }
    ),
    adapter_id="ltx2",
    source_url="https://raw.githubusercontent.com/madebyollin/taehv/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeltx2_3.safetensors",
)

DECODERS = {TAELTX23.decoder_id: TAELTX23}


def get_decoder_for_model(model_type: str, model_def: dict[str, Any] | None = None) -> PreviewDecoderSpec | None:
    model_type = str(model_type or "").strip()
    architecture = str((model_def or {}).get("architecture") or "").strip()
    capabilities = (model_def or {}).get("capabilities", {})
    live_preview = capabilities.get("live_preview", {}) if isinstance(capabilities, dict) else {}
    if not isinstance(live_preview, dict):
        return None
    for spec in DECODERS.values():
        if (
            model_type in spec.compatible_model_types
            and architecture in spec.compatible_architectures
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
