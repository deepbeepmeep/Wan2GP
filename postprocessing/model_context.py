"""Loaded-model borrowing contract for postprocessing extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LoadedModelContext:
    """Core-owned model state that a compatible extension may borrow for one call."""

    model: Any
    offloadobj: Any
    model_type: str
    base_model_type: str
    model_family: str
    model_def: Mapping[str, Any]
    profile: int
    config_id: str


def compatible_loaded_model(handler, value, context: LoadedModelContext | None, **kwargs) -> LoadedModelContext | None:
    if context is None or not hasattr(handler, "supports_loaded_model"):
        return None
    return context if handler.supports_loaded_model(value, context, **kwargs) else None


__all__ = ["LoadedModelContext", "compatible_loaded_model"]
