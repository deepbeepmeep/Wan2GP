"""Compatibility wrappers for direct Qwen task-flow entrypoints."""

from __future__ import annotations

from typing import Any, Dict

from source.models.model_handlers.qwen_handler import QwenHandler


def handle_qwen_image(handler: QwenHandler, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
    """Delegate to the handler instance's canonical image flow."""
    return handler.handle_qwen_image(db_task_params, generation_params)


__all__ = ["handle_qwen_image"]
