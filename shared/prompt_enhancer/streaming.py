from __future__ import annotations

import time
from typing import Any, Callable


class ThrottledStreamEmitter:
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = max(0.0, float(interval_seconds))
        self._last_emit_at = 0.0

    def is_due(self) -> bool:
        return self._is_due_at(time.monotonic())

    def _is_due_at(self, now: float) -> bool:
        return self.interval_seconds <= 0 or self._last_emit_at <= 0 or (now - self._last_emit_at) >= self.interval_seconds

    def emit(self, callback: Callable[..., Any] | None, /, *args, force: bool = False, **kwargs) -> bool:
        if not callable(callback):
            return False
        now = time.monotonic()
        if not force and not self._is_due_at(now):
            return False
        self._last_emit_at = now
        callback(*args, **kwargs)
        return True
