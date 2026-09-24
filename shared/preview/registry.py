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
    compatible_architectures=frozenset({"ltx2_22B", "ltx2_25_22B"}),
    adapter_id="ltx2",
    source_url="https://raw.githubusercontent.com/madebyollin/taehv/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeltx2_3.safetensors",
)

TAELTX2 = PreviewDecoderSpec(
    decoder_id="taeltx_2",
    filename="taeltx_2.safetensors",
    sha256="6e4cc0469134213d0101a46877ea2bce1dc7cf06ff5f5aefb9e4076c03542f7b",
    size_bytes=23_531_296,
    latent_channels=128,
    patch_size=4,
    encoder_time_downscale=(True, True, True),
    decoder_time_upscale=(True, True, True),
    compatible_architectures=frozenset({"ltx2_19B"}),
    adapter_id="ltx2",
    source_url="https://raw.githubusercontent.com/madebyollin/taehv/011dfc2112197741c540e0bdd5b7b67bcc930771/safetensors/taeltx_2.safetensors",
)

TAEF2 = PreviewDecoderSpec(
    decoder_id="taef2",
    filename="taef2_decoder.pth",
    sha256="0a44a31e1ae59eb9dbf9359d24942fd3ef5928162c1800d97a28c61d51be0ab5",
    size_bytes=2_704_756,
    latent_channels=32,
    patch_size=1,
    encoder_time_downscale=(),
    decoder_time_upscale=(),
    compatible_architectures=frozenset({"flux2_klein_4b", "flux2_klein_9b", "ideogram4", "ideogram4_turbotime"}),
    adapter_id="taesd",
    source_url="https://raw.githubusercontent.com/madebyollin/taesd/e87efbcfc5298d84986b5d9280f40d358b6a228d/taef2_decoder.pth",
    target_dir="preview_decoders/taesd",
    decoder_layout="NCHW",
)

TAEF1 = PreviewDecoderSpec(
    decoder_id="taef1",
    filename="taef1_decoder.pth",
    sha256="beae86f2eeaf0cea884dc8ffe639fb297dac8c984441bb53420f07d767785104",
    size_bytes=4_943_336,
    latent_channels=16,
    patch_size=1,
    encoder_time_downscale=(),
    decoder_time_upscale=(),
    compatible_architectures=frozenset({"flux", "z_image"}),
    adapter_id="taesd",
    source_url="https://raw.githubusercontent.com/madebyollin/taesd/e87efbcfc5298d84986b5d9280f40d358b6a228d/taef1_decoder.pth",
    target_dir="preview_decoders/taesd",
    decoder_layout="NCHW",
)

TAEW21_IMAGE = PreviewDecoderSpec(
    decoder_id="taew2_1_image",
    filename="taew2_1.safetensors",
    sha256="04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f",
    size_bytes=22_642_902,
    latent_channels=16,
    patch_size=1,
    encoder_time_downscale=(True, True, False),
    decoder_time_upscale=(False, True, True),
    compatible_architectures=frozenset({"krea2_raw", "krea2_raw_edit", "krea2_turbo", "krea2_turbo_edit", "qwen_image_20B", "qwen_image_edit_20B", "qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"}),
    adapter_id="qwen_image",
    source_url="https://raw.githubusercontent.com/madebyollin/taehv/011dfc2112197741c540e0bdd5b7b67bcc930771/safetensors/taew2_1.safetensors",
    decoder_layout="NTCHW",
)

TAEWAN21 = PreviewDecoderSpec(
    decoder_id="taew2_1",
    filename="taew2_1.safetensors",
    sha256="04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f",
    size_bytes=22_642_902,
    latent_channels=16,
    patch_size=1,
    encoder_time_downscale=(True, True, False),
    decoder_time_upscale=(False, True, True),
    compatible_architectures=frozenset({"t2v", "t2v_1.3B", "i2v", "t2v_2_2", "i2v_2_2"}),
    adapter_id="wan",
    source_url="https://raw.githubusercontent.com/madebyollin/taehv/011dfc2112197741c540e0bdd5b7b67bcc930771/safetensors/taew2_1.safetensors",
)

TAEWAN22 = PreviewDecoderSpec(
    decoder_id="taew2_2",
    filename="taew2_2.safetensors",
    sha256="b84609b2a133d48434bd9636bfcb44bf05168dc436e2d3cecf26256faa1f5325",
    size_bytes=22_848_048,
    latent_channels=48,
    patch_size=2,
    encoder_time_downscale=(True, True, False),
    decoder_time_upscale=(False, True, True),
    compatible_architectures=frozenset({"ti2v_2_2"}),
    adapter_id="wan",
    source_url="https://raw.githubusercontent.com/madebyollin/taehv/011dfc2112197741c540e0bdd5b7b67bcc930771/safetensors/taew2_2.safetensors",
)

TAEHY = PreviewDecoderSpec(
    decoder_id="taehv",
    filename="taehv.safetensors",
    sha256="032ad4ddc689513287ce02fcc099b5278592526ed1eac0324a7fab94bb690ed9",
    size_bytes=22_642_902,
    latent_channels=16,
    patch_size=1,
    encoder_time_downscale=(True, True, False),
    decoder_time_upscale=(False, True, True),
    compatible_architectures=frozenset({"hunyuan", "hunyuan_i2v"}),
    adapter_id="hunyuan",
    source_url="https://raw.githubusercontent.com/madebyollin/taehv/011dfc2112197741c540e0bdd5b7b67bcc930771/safetensors/taehv.safetensors",
)

TAEHY15 = PreviewDecoderSpec(
    decoder_id="taehv1_5",
    filename="taehv1_5.safetensors",
    sha256="b52e245bb86c62e159f50338e2e8f422d4b6f98b467164939c1c031c7d61352e",
    size_bytes=22_755_856,
    latent_channels=32,
    patch_size=2,
    encoder_time_downscale=(True, True, False),
    decoder_time_upscale=(False, True, True),
    compatible_architectures=frozenset({"hunyuan_1_5_t2v", "hunyuan_1_5_i2v"}),
    adapter_id="hunyuan",
    source_url="https://raw.githubusercontent.com/madebyollin/taehv/011dfc2112197741c540e0bdd5b7b67bcc930771/safetensors/taehv1_5.safetensors",
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

DECODERS = {
    spec.decoder_id: spec
    for spec in (TAELTX2, TAELTX23, TAEF2, TAEF1, TAEW21_IMAGE, TAEWAN21, TAEWAN22, TAEHY, TAEHY15, TAEH3)
}


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
