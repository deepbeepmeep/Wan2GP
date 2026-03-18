"""Thumbnail helpers split from timeline rendering."""

from __future__ import annotations

import numpy as np
from PIL import Image


def _load_timeline_thumbnail(
    image_path: str,
    *,
    max_thumb_width: int,
    max_thumb_height: int,
):
    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((max_thumb_width, max_thumb_height))
        return np.array(image)
    except Exception:
        return np.zeros((max_thumb_height, max_thumb_width, 3), dtype=np.uint8)


__all__ = ["_load_timeline_thumbnail"]
