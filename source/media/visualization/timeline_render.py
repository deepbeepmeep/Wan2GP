"""Compatibility wrapper for timeline rendering helpers."""

from __future__ import annotations

from source.media.visualization.timeline import _create_timeline_clip


def render_timeline_frame(**kwargs):
    """Create a timeline clip/frame using the canonical timeline helper."""
    return _create_timeline_clip(**kwargs)


__all__ = ["render_timeline_frame"]
