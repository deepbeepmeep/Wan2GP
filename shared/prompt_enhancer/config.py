from __future__ import annotations

from typing import Any


PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY = "prompt_enhancer_speculative_decoding"
PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT = 0
PROMPT_ENHANCER_SPECULATIVE_DECODING_IDS = frozenset((4, 5))
def normalize_prompt_enhancer_speculative_decoding(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "on"} else 0
    return 1 if bool(value) else 0


def prompt_enhancer_supports_speculative_decoding(enhancer_enabled: Any) -> bool:
    try:
        return int(enhancer_enabled or 0) in PROMPT_ENHANCER_SPECULATIVE_DECODING_IDS
    except (TypeError, ValueError):
        return False


def validate_prompt_enhancer_speculative_decoding(enhancer_enabled: Any, value: Any) -> int:
    enabled = normalize_prompt_enhancer_speculative_decoding(value)
    if enabled and not prompt_enhancer_supports_speculative_decoding(enhancer_enabled):
        if int(enhancer_enabled or 0) == 3:
            raise ValueError("Speculative decoding is not available with Qwen3.5-4B.")
        raise ValueError("Speculative decoding requires the Qwen3.5-9B or Qwen3.8-27B prompt enhancer.")
    return enabled
