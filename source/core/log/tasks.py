"""Backward-compatible task display-name shims."""

from source.core.log.display_names import model_label

__all__ = [
    "get_display_name",
]


def get_display_name(internal_name: str) -> str:
    """Backward-compatible shim for the shared model label helper."""
    return model_label(internal_name)
