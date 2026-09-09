import os
from dataclasses import dataclass, field
from typing import Any

from .service import send_notification_async
from .settings import NOTIFY_GENERATION_KEY, NOTIFY_QUEUE_COMPLETE_KEY, NOTIFY_QUEUE_INTERRUPTED_KEY


_RUN_KEY = "_notification_run"


def _format_duration(value: Any) -> str:
    try:
        seconds = max(0, round(float(value)))
    except (TypeError, ValueError):
        return "unknown"
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _generation_entry(paths: str | list[str], configs: Any) -> dict[str, Any]:
    path_list = paths if isinstance(paths, list) else [paths]
    settings = configs if isinstance(configs, dict) else {}
    return {
        "files": [os.path.basename(str(path)) for path in path_list],
        "generation_time": settings.get("generation_time"),
        "model_type": str(settings.get("model_type", "") or ""),
        "prompt": str(settings.get("prompt", "") or "").replace("\n", " ").strip(),
    }


def _format_generation(entry: dict[str, Any], number: int | None = None, include_prompt: bool = True) -> str:
    prefix = f"{number}. " if number is not None else ""
    files = ", ".join(entry["files"]) or "unnamed output"
    lines = [f"{prefix}{files} — {_format_duration(entry.get('generation_time'))}"]
    if entry.get("model_type"):
        lines.append(f"Model: {entry['model_type']}")
    if include_prompt and entry.get("prompt"):
        prompt = entry["prompt"]
        lines.append(f"Prompt: {prompt[:157] + '...' if len(prompt) > 160 else prompt}")
    return "\n".join(lines)


@dataclass
class NotificationRun:
    total_tasks: int
    completed_tasks: int = 0
    skipped_tasks: int = 0
    failed_tasks: int = 0
    interrupted: bool = False
    aborted: bool = False
    reason: str = ""
    generations: list[dict[str, Any]] = field(default_factory=list)
    _finished: bool = False

    def task_completed(self) -> None:
        self.completed_tasks += 1

    def task_skipped(self) -> None:
        self.skipped_tasks += 1

    def task_failed(self) -> None:
        self.failed_tasks += 1

    def interrupt(self, reason: str, *, aborted: bool = False) -> None:
        if not self.interrupted or aborted:
            self.reason = str(reason or "")
        self.interrupted = True
        self.aborted = self.aborted or aborted

    def fail(self, reason: str) -> None:
        self.interrupted = True
        self.reason = str(reason or "")

    def record(self, entry: dict[str, Any], replace_last: bool) -> None:
        if replace_last and self.generations:
            self.generations[-1] = entry
        else:
            self.generations.append(entry)

    def finish(self) -> dict[str, Any] | None:
        if self._finished:
            return None
        self._finished = True
        reasons = []
        if self.failed_tasks:
            reasons.append(f"{self.failed_tasks} task{'s' if self.failed_tasks != 1 else ''} failed")
        if self.skipped_tasks:
            reasons.append(f"{self.skipped_tasks} task{'s' if self.skipped_tasks != 1 else ''} skipped")
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "interrupted": self.interrupted or self.failed_tasks > 0,
            "aborted": self.aborted,
            "reason": self.reason or ", ".join(reasons),
            "generations": list(self.generations),
        }


def start_queue(gen: dict[str, Any], total_tasks: int) -> NotificationRun:
    run = NotificationRun(max(0, int(total_tasks)))
    gen[_RUN_KEY] = run
    return run


def record_generation(config: dict[str, Any], gen: dict[str, Any], paths: str | list[str], configs: Any, replace_last: bool = False, notify: bool = True) -> None:
    entry = _generation_entry(paths, configs)
    run = gen.get(_RUN_KEY)
    if isinstance(run, NotificationRun):
        run.record(entry, replace_last)
        if notify and config.get(NOTIFY_GENERATION_KEY, False):
            send_notification_async(config, "WanGP generation completed", _format_generation(entry))


def finish_queue(config: dict[str, Any], gen: dict[str, Any], run: NotificationRun, elapsed_seconds: float) -> None:
    outcome = run.finish()
    if outcome is None:
        return
    if gen.get(_RUN_KEY) is run:
        gen.pop(_RUN_KEY)
    interrupted = outcome["interrupted"]
    if not config.get(NOTIFY_QUEUE_INTERRUPTED_KEY if interrupted else NOTIFY_QUEUE_COMPLETE_KEY, False):
        return
    generations = outcome["generations"]
    title = "WanGP queue interrupted" if interrupted else "WanGP queue completed"
    state = "Interrupted" if interrupted else "Completed"
    total_tasks = outcome["total_tasks"]
    lines = [f"{state}: {outcome['completed_tasks']}/{total_tasks} task{'s' if total_tasks != 1 else ''}", f"Total time: {_format_duration(elapsed_seconds)}", f"Generations produced: {len(generations)}"]
    if outcome["reason"]:
        reason = " ".join(outcome["reason"].split())
        lines.append(f"Reason: {reason[:297] + '...' if len(reason) > 300 else reason}")
    if generations:
        lines.extend(["", "Generated media:"])
        lines.extend(_format_generation(entry, number, include_prompt=False) for number, entry in enumerate(generations, 1))
    send_notification_async(config, title, "\n".join(lines))
