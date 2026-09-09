from __future__ import annotations


ADAPTIVE_THRESHOLDS = (0.08, 0.16, 0.28, 0.43, 0.60, 0.80, 1.00)


class CaptureScheduler:
    def __init__(self, total_steps: int, update_rate: str = "adaptive", target_updates: int = 7) -> None:
        self.total_steps = max(1, int(total_steps or 1))
        self.update_rate = update_rate
        self.target_updates = max(2, min(16, int(target_updates or 7)))
        self._thresholds = (
            ADAPTIVE_THRESHOLDS
            if self.target_updates == len(ADAPTIVE_THRESHOLDS)
            else tuple((index / self.target_updates) ** 1.5 for index in range(1, self.target_updates + 1))
        )
        self._context_id = None
        self._captured: set[int] = set()
        self._threshold_index = 0

    def reset(self, context_id: str | None = None) -> None:
        self._context_id = context_id
        self._captured.clear()
        self._threshold_index = 0

    def should_capture(self, step: int, *, context_id: str | None = None, force_refresh: bool = False) -> bool:
        if context_id != self._context_id:
            self.reset(context_id)
        step = int(step)
        if step <= 0 or step > self.total_steps:
            return False
        if step in self._captured:
            return False
        if step == self.total_steps:
            should = True
        elif self.update_rate == "every_step":
            should = True
        elif self.update_rate == "every_2":
            should = step % 2 == 0
        elif self.update_rate == "every_4":
            should = step % 4 == 0
        else:
            position = step / self.total_steps
            should = self._threshold_index == 0 and step <= 2
            if not should:
                should = self._threshold_index < len(self._thresholds) and position >= self._thresholds[self._threshold_index]
        if force_refresh and step > 0:
            should = True
        if not should:
            return False
        self._captured.add(step)
        if self.update_rate == "adaptive":
            if self._threshold_index == 0 and step <= 2:
                self._threshold_index = 1
            position = step / self.total_steps
            while self._threshold_index < len(self._thresholds) and position >= self._thresholds[self._threshold_index]:
                self._threshold_index += 1
        return True

    @property
    def captured_steps(self) -> tuple[int, ...]:
        return tuple(sorted(self._captured))
