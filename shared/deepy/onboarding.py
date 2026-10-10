from __future__ import annotations

from typing import Any

from shared.deepy.config import (
    DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS,
    DEEPY_COMPACTION_TYPE_KEY,
    DEEPY_COMPACTION_TYPE_SUMMARIZE,
    DEEPY_CONTEXT_TOKENS_KEY,
    DEEPY_ENABLED_KEY,
    DEEPY_TYPE_KEY,
    DEEPY_TYPE_PRIME,
    deepy_mode_from_config,
)
from shared.remote_llm.config import ENGINE_QWEN38_27B, LLM_CONFIG_KEY, local_enhancer_id, normalize_llm_config


DEEPY_PRIME_Q2_MIN_VRAM_GB = 16
DEEPY_PRIME_Q4_MIN_VRAM_GB = 24


def detected_nvidia_vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available() or not getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None):
            return 0.0
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def deepy_prime_hardware_profile(vram_gb: float | None = None) -> tuple[str, str, int] | None:
    reported_vram_gb = int((detected_nvidia_vram_gb() if vram_gb is None else vram_gb) + 0.5)
    if reported_vram_gb >= DEEPY_PRIME_Q4_MIN_VRAM_GB:
        return "Q4", "gguf", reported_vram_gb
    if reported_vram_gb >= DEEPY_PRIME_Q2_MIN_VRAM_GB:
        return "Q2", "gguf_q2", reported_vram_gb
    return None


def apply_first_launch_deepy_prime_defaults(server_config: dict[str, Any], vram_gb: float | None = None) -> str:
    profile = deepy_prime_hardware_profile(vram_gb)
    if profile is None:
        return ""
    label, quantization, _reported_vram_gb = profile
    llm_config = normalize_llm_config(server_config)
    llm_config["deepy"] = ENGINE_QWEN38_27B
    server_config.update({
        DEEPY_ENABLED_KEY: 1,
        DEEPY_TYPE_KEY: DEEPY_TYPE_PRIME,
        DEEPY_CONTEXT_TOKENS_KEY: DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS,
        DEEPY_COMPACTION_TYPE_KEY: DEEPY_COMPACTION_TYPE_SUMMARIZE,
        "enhancer_enabled": local_enhancer_id(ENGINE_QWEN38_27B),
        "prompt_enhancer_quantization": quantization,
        LLM_CONFIG_KEY: llm_config,
    })
    return label


def deepy_prime_upgrade_message(server_config: dict[str, Any], vram_gb: float | None = None) -> str:
    profile = deepy_prime_hardware_profile(vram_gb)
    if profile is None or deepy_mode_from_config(server_config.get(DEEPY_ENABLED_KEY, 0), server_config.get(DEEPY_TYPE_KEY)) == DEEPY_TYPE_PRIME:
        return ""
    label, _quantization, reported_vram_gb = profile
    return f"✨ **Switch to Deepy Prime {label}** — your NVIDIA GPU ({reported_vram_gb} GB VRAM) is eligible. Gain multi-step planning, image/video/audio tool chaining, and end-to-end handling of long creations. Select Deepy Prime below to switch."
