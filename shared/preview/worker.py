from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from PIL import Image

from .encoding import encode_preview
from .types import PreviewContext, PreviewMedia, PreviewOptions


@dataclass(frozen=True)
class PreviewJob:
    frames: tuple[Image.Image, ...]
    context: PreviewContext
    options: PreviewOptions
    decode_ms: float
    dropped_count: int = 0


class PreviewWorker:
    """One active job plus one replaceable pending job; submission never blocks."""

    def __init__(self, on_result: Callable[[PreviewMedia], None], on_error: Callable[[Exception], None] | None = None) -> None:
        self._on_result = on_result
        self._on_error = on_error
        self._condition = threading.Condition()
        self._pending: PreviewJob | None = None
        self._active = False
        self._closed = False
        self._invalid_generation_ids: set[str] = set()
        self.dropped_count = 0
        self._thread = threading.Thread(target=self._run, name="wangp-preview-encoder", daemon=True)
        self._thread.start()

    def try_submit(self, job: PreviewJob) -> bool:
        with self._condition:
            if self._closed or job.context.generation_id in self._invalid_generation_ids:
                return False
            if self._pending is not None:
                self.dropped_count += 1
            self._pending = PreviewJob(job.frames, job.context, job.options, job.decode_ms, self.dropped_count)
            self._condition.notify()
            return True

    def invalidate(self, generation_id: str) -> None:
        with self._condition:
            self._invalid_generation_ids.add(generation_id)
            if self._pending is not None and self._pending.context.generation_id == generation_id:
                self._pending = None

    def close(self, *, wait: bool = False) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        if wait:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._pending is None and self._closed:
                    return
                job = self._pending
                self._pending = None
                self._active = True
            try:
                assert job is not None
                media = encode_preview(job.frames, job.context, job.options, decode_ms=job.decode_ms, dropped_count=job.dropped_count)
                with self._condition:
                    invalid = job.context.generation_id in self._invalid_generation_ids or self._closed
                if not invalid:
                    self._on_result(media)
            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(exc)
            finally:
                with self._condition:
                    self._active = False
