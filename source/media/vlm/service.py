"""Compatibility service layer for VLM prompt-extender initialization."""

from __future__ import annotations

from source.runtime.wgp_ports.vendor_imports import create_qwen_prompt_expander


def initialize_qwen_prompt_extender(*args, **kwargs):
    """Initialize the Qwen prompt extender through the runtime boundary."""
    return create_qwen_prompt_expander(*args, **kwargs)


__all__ = ["initialize_qwen_prompt_extender"]
