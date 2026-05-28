from __future__ import annotations

import time
from typing import Any, Callable


class ThrottledStreamEmitter:
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = max(0.0, float(interval_seconds))
        self._last_emit_at = 0.0

    def emit(self, callback: Callable[..., Any] | None, /, *args, force: bool = False, **kwargs) -> bool:
        if not callable(callback):
            return False
        now = time.monotonic()
        if not force and self.interval_seconds > 0 and self._last_emit_at > 0 and (now - self._last_emit_at) < self.interval_seconds:
            return False
        self._last_emit_at = now
        callback(*args, **kwargs)
        return True
