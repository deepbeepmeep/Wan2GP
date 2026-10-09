# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Optional external LLM engines used by Deepy and Prompt Enhancer."""

from .config import (
    LLM_CONFIG_KEY,
    LLM_ENGINE_CHOICES,
    PROMPT_ENGINE_CHOICES,
    VISUAL_INSPECTOR_CHOICES,
    is_remote_engine,
    normalize_llm_config,
    resolve_role_engine,
    validate_llm_config,
)

__all__ = [
    "LLM_CONFIG_KEY",
    "LLM_ENGINE_CHOICES",
    "PROMPT_ENGINE_CHOICES",
    "VISUAL_INSPECTOR_CHOICES",
    "is_remote_engine",
    "normalize_llm_config",
    "resolve_role_engine",
    "validate_llm_config",
]
