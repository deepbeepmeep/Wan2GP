"""Compatibility service layer for VLM prompt-extender initialization."""

from __future__ import annotations

import gc
from pathlib import Path

from source.media.vlm.model import download_qwen_vlm_if_needed
from source.runtime.wgp_bridge import (
    create_qwen_prompt_expander,
    ensure_wan2gp_on_path as _ensure_runtime_bridge_path,
)

DEFAULT_QWEN_VLM_MODEL = "Qwen2.5-VL-7B-Instruct"
ensure_wan2gp_on_path = _ensure_runtime_bridge_path


def initialize_qwen_prompt_extender(
    *,
    device: str = "cuda",
    context: str | None = None,
    expected_items: int | None = None,
):
    """Initialize the shared Qwen VLM prompt extender through the runtime boundary."""
    del context, expected_items

    wan_root = Path(globals()["ensure_wan2gp_on_path"]())
    model_path = wan_root / "ckpts" / DEFAULT_QWEN_VLM_MODEL
    download_qwen_vlm_if_needed(model_path)
    return create_qwen_prompt_expander(
        model_name=str(model_path),
        device=device,
        is_vl=True,
    )


def cleanup_qwen_prompt_extender(extender, *, context: str | None = None) -> None:
    """Best-effort cleanup for VLM prompt extenders."""
    del context
    if extender is None:
        return

    for attr_name in ("model", "processor"):
        if hasattr(extender, attr_name):
            try:
                delattr(extender, attr_name)
            except (AttributeError, TypeError):
                pass

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        pass


__all__ = [
    "DEFAULT_QWEN_VLM_MODEL",
    "cleanup_qwen_prompt_extender",
    "create_qwen_prompt_expander",
    "download_qwen_vlm_if_needed",
    "ensure_wan2gp_on_path",
    "initialize_qwen_prompt_extender",
]
