from __future__ import annotations

from typing import Any


PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY = "prompt_enhancer_speculative_decoding"
PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO = 2
PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT = PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO
PROMPT_ENHANCER_SPECULATIVE_DECODING_IDS = frozenset((4, 5))
PROMPT_ENHANCER_SPECULATIVE_DECODING_VRAM_GB = {4: 12, 5: 24}
PROMPT_ENHANCER_SPECULATIVE_DRAFT_COUNTS = (2, 3, 4)
PROMPT_ENHANCER_SPECULATIVE_DECODING_CHOICES = [("Auto", PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO), ("Disabled", 0), *[(f"Enabled with {count} draft tokens", 1 if count == 2 else count) for count in PROMPT_ENHANCER_SPECULATIVE_DRAFT_COUNTS]]


def normalize_prompt_enhancer_speculative_decoding(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO
        if text in {str(count) for count in PROMPT_ENHANCER_SPECULATIVE_DRAFT_COUNTS if count > 2}:
            return int(text)
        return 1 if text in {"1", "true", "yes", "on"} else 0
    try:
        mode = int(value)
        return mode if mode == PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO or mode in PROMPT_ENHANCER_SPECULATIVE_DRAFT_COUNTS else (1 if bool(value) else 0)
    except (TypeError, ValueError):
        return PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT


def prompt_enhancer_supports_speculative_decoding(enhancer_enabled: Any) -> bool:
    try:
        return int(enhancer_enabled or 0) in PROMPT_ENHANCER_SPECULATIVE_DECODING_IDS
    except (TypeError, ValueError):
        return False


def validate_prompt_enhancer_speculative_decoding(enhancer_enabled: Any, value: Any) -> int:
    enabled = normalize_prompt_enhancer_speculative_decoding(value)
    if enabled not in (0, PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO) and not prompt_enhancer_supports_speculative_decoding(enhancer_enabled):
        if int(enhancer_enabled or 0) == 3:
            raise ValueError("Speculative decoding is not available with Qwen3.5-4B.")
        raise ValueError("Speculative decoding requires the Qwen3.5-9B or Qwen3.8-27B prompt enhancer.")
    return enabled


def resolve_prompt_enhancer_speculative_decoding(enhancer_enabled: Any, value: Any, total_vram_gb: float | None = None) -> tuple[int, str]:
    mode = validate_prompt_enhancer_speculative_decoding(enhancer_enabled, value)
    if mode != PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO:
        return mode, ""
    try:
        enhancer_no = int(enhancer_enabled or 0)
    except (TypeError, ValueError):
        enhancer_no = 0
    required_vram_gb = PROMPT_ENHANCER_SPECULATIVE_DECODING_VRAM_GB.get(enhancer_no)
    if required_vram_gb is None:
        return 0, "Speculative Decoding disabled automatically because the selected model does not support it."
    if total_vram_gb is None:
        try:
            import torch

            total_vram_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
        except Exception:
            total_vram_gb = 0
    detected_vram_gb = max(0, int(float(total_vram_gb) + 0.5))
    model_label = "Qwen3.5-9B" if enhancer_no == 4 else "Qwen3.8-27B"
    if detected_vram_gb >= required_vram_gb:
        return 1, f"Speculative Decoding enabled automatically for {model_label}: {detected_vram_gb} GB VRAM detected (minimum {required_vram_gb} GB)."
    return 0, f"Speculative Decoding disabled automatically for {model_label}: {detected_vram_gb} GB VRAM detected (minimum {required_vram_gb} GB)."
