"""Shared frame-brightness helpers."""

from __future__ import annotations

import numpy as np


def adjust_frame_brightness(frame: np.ndarray, brightness_adjust: float):
    if brightness_adjust == 0:
        return frame
    factor = 1.0 + brightness_adjust
    adjusted = np.clip(frame.astype(np.float32) * factor, 0, 255)
    return adjusted.astype(np.uint8)


__all__ = ["adjust_frame_brightness"]
