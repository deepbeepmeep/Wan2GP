"""Shared polling lifecycle policy primitives."""

from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass
class PollLifecycle:
    poll_interval: float
    post_task_delay: float = 0.0
    stop_event: threading.Event | None = None
    sleep_impl: callable = lambda _seconds: None

    def sleep_idle(self) -> None:
        self.sleep_impl(self.poll_interval)

    def sleep_after_task(self) -> None:
        self.sleep_impl(self.post_task_delay)

    def should_stop(self) -> bool:
        return bool(self.stop_event and self.stop_event.is_set())


@dataclass(frozen=True)
class FairQueuePolicy:
    starvation_seconds: float = 60.0

    def choose_index(self, entries, *, now: float) -> int:
        starved = [
            (index, entry)
            for index, entry in enumerate(entries)
            if (now - entry[1]) >= self.starvation_seconds
        ]
        if not starved:
            return 0
        return min(starved, key=lambda item: item[1][1])[0]


__all__ = ["FairQueuePolicy", "PollLifecycle"]
