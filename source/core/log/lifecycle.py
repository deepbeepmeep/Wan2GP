"""Task lifecycle logging with a single outer anchor per execution context."""

from __future__ import annotations

import contextvars
import os
import reprlib
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from source.core.log.core import debug, error, essential, is_debug_enabled, success
from source.core.log.debug_card import DebugCard
from source.core.log.display_names import friendly_task_id, rel_path, task_type_label

_COMPONENT = "TASK"
_MISSING = object()
_BLOOM_START = "❀"
_BLOOM_DONE = "✿"
_BLOOM_FAIL = "❌"
_SPARK_BARS = "▁▂▃▄▅▆▇█"

__all__ = ["RunSummary", "TaskLifecycleEmitter", "lifecycle", "run_summary"]

_SUMMARY_REPR = reprlib.Repr()
_SUMMARY_REPR.maxstring = 40
_SUMMARY_REPR.maxother = 40
_SUMMARY_REPR.maxlist = 6
_SUMMARY_REPR.maxdict = 6
_SUMMARY_REPR.maxset = 6
_SUMMARY_REPR.maxtuple = 6


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    hi = max(values)
    lo = min(values)
    span = hi - lo or 1.0
    n = len(_SPARK_BARS) - 1
    return "".join(_SPARK_BARS[min(n, int((value - lo) / span * n))] for value in values)


def _format_start_summary(summary: Mapping[str, Any] | None) -> str:
    if not summary:
        return ""

    parts: list[str] = []
    for key, value in summary.items():
        if value in (None, ""):
            continue
        rendered = _SUMMARY_REPR.repr(value)
        if isinstance(value, str) and rendered.startswith("'") and rendered.endswith("'"):
            rendered = rendered[1:-1]
        parts.append(f"{key}={rendered}")
        if len(parts) == 3:
            break

    return f"  •  {', '.join(parts)}" if parts else ""


class _TaskHandle:
    def __init__(self, task_id: str | None, task_type: str, details: dict[str, Any]):
        self.task_id = task_id
        self.task_type = task_type
        self._details = dict(details)

    def set(self, output: Any = _MISSING, error: Any = _MISSING, **extra: Any) -> None:
        if output is not _MISSING:
            self._details["output"] = output
        if error is not _MISSING:
            self._details["error"] = error
        self._details.update(extra)

    def get_details(self) -> dict[str, Any]:
        return dict(self._details)


class _PassthroughHandle:
    def __init__(self, handle: _TaskHandle):
        self._handle = handle

    def set(self, output: Any = _MISSING, error: Any = _MISSING, **extra: Any) -> None:
        self._handle.set(output=output, error=error, **extra)

    def get_details(self) -> dict[str, Any]:
        return self._handle.get_details()


class RunSummary:
    def __init__(self):
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_rendered_count = -1

    def append_task(self, task_type: str, duration: float, status: str, error: str | None = None) -> None:
        with self._lock:
            self._entries.append(
                {
                    "task_type": task_type,
                    "duration": float(duration),
                    "status": status,
                    "error": error,
                }
            )

    def render(self) -> str:
        card = DebugCard("Run Summary", heavy=True)
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return card.row("tasks", "none").render()

        grouped: dict[str, dict[str, Any]] = {}
        overall_durations: list[float] = []
        overall_successes = 0
        overall_failures = 0
        total_duration = 0.0
        for entry in entries:
            duration = float(entry["duration"])
            total_duration += duration
            overall_durations.append(duration)
            stats = grouped.setdefault(
                entry["task_type"],
                {"count": 0, "successes": 0, "failures": 0, "total_duration_s": 0.0, "durations": []},
            )
            stats["count"] += 1
            stats["total_duration_s"] += duration
            stats["durations"].append(duration)
            if entry["status"] == "success":
                stats["successes"] += 1
                overall_successes += 1
            else:
                stats["failures"] += 1
                overall_failures += 1

        card.row(
            "Overall",
            {
                "tasks": len(entries),
                "successes": overall_successes,
                "failures": overall_failures,
                "total_wall_clock_s": round(total_duration, 1),
                "sparkline": _sparkline(overall_durations),
            },
        )

        for task_type in sorted(grouped, key=task_type_label):
            stats = grouped[task_type]
            card.row(
                task_type_label(task_type),
                {
                    "count": stats["count"],
                    "successes": stats["successes"],
                    "failures": stats["failures"],
                    "total_duration_s": round(float(stats["total_duration_s"]), 1),
                    "sparkline": _sparkline(stats["durations"]),
                },
            )

        return card.render()

    def render_to(self, logger: Any | None = None, task_id: str | None = None) -> None:
        with self._lock:
            entry_count = len(self._entries)
            if entry_count == self._last_rendered_count:
                return
            self._last_rendered_count = entry_count
        rendered = self.render()
        if logger is None:
            essential(_COMPONENT, rendered, task_id=task_id)
            return
        logger.essential(rendered, task_id=task_id)


class TaskLifecycleEmitter:
    def __init__(self):
        self._active_anchor: contextvars.ContextVar[_TaskHandle | None] = contextvars.ContextVar(
            "task_lifecycle_active_anchor",
            default=None,
        )
        self.run_summary = RunSummary()

    @contextmanager
    def task(
        self,
        task_id: str | None,
        task_type: str,
        *,
        display_summary: Mapping[str, Any] | None = None,
        **details: Any,
    ) -> Iterator[_TaskHandle | _PassthroughHandle]:
        active = self._active_anchor.get()
        if active is not None:
            if details:
                active.set(**details)
            yield _PassthroughHandle(active)
            return

        handle = _TaskHandle(task_id, task_type, details)
        label = task_type_label(task_type)
        compact_id = friendly_task_id(task_id or "", task_type)
        token = self._active_anchor.set(handle)
        started_at = time.perf_counter()

        # Nested tasks return above, so the breath separator stays outer-task-only.
        # Two blank lines create more separation between task runs.
        print("")
        print("")
        essential(
            _COMPONENT,
            f"{_BLOOM_START} {label} [{compact_id}] started{_format_start_summary(display_summary)}",
            task_id=task_id,
            include_task_prefix=False,
        )
        self._emit_debug_card("Task Start", handle, display_summary=display_summary, label=label, friendly_id=compact_id)

        try:
            yield handle
        except Exception as exc:
            duration = time.perf_counter() - started_at
            message = self._error_message(handle.get_details().get("error"), exc)
            handle.set(error=message)
            error(
                _COMPONENT,
                f"❌ {label} [{compact_id}] failed in {duration:.1f}s: {message}",
                task_id=task_id,
                include_task_prefix=False,
            )
            self.run_summary.append_task(task_type, duration, "failed", error=message)
            # Task Done is terser than Task Start — Task Start already carried label/model/res/frames,
            # so Task Done only adds the new info (status, duration, output/error).
            self._emit_debug_card(
                "Task Done",
                handle,
                status="failed",
                duration_s=round(duration, 1),
                error=message,
            )
            raise
        else:
            duration = time.perf_counter() - started_at
            current_details = handle.get_details()
            message = current_details.get("error")
            if message:
                error(
                    _COMPONENT,
                    f"❌ {label} [{compact_id}] failed in {duration:.1f}s: {message}",
                    task_id=task_id,
                    include_task_prefix=False,
                )
                self.run_summary.append_task(task_type, duration, "failed", error=str(message))
                status = "failed"
            else:
                output = current_details.get("output")
                completion = f"{_BLOOM_DONE} {label} [{compact_id}] done in {duration:.1f}s"
                if output:
                    completion = f"{completion} → {rel_path(output)}"
                success(_COMPONENT, completion, task_id=task_id, include_task_prefix=False)
                self.run_summary.append_task(task_type, duration, "success")
                status = "success"

            # Task Done is terser than Task Start — only the new info.
            self._emit_debug_card(
                "Task Done",
                handle,
                status=status,
                duration_s=round(duration, 1),
            )
        finally:
            self._active_anchor.reset(token)

    def _emit_debug_card(
        self,
        title: str,
        handle: _TaskHandle,
        *,
        display_summary: Mapping[str, Any] | None = None,
        **summary: Any,
    ) -> None:
        if not is_debug_enabled():
            return

        card = DebugCard(title, heavy=True)
        seen: set[str] = set()

        def add(key: str, value: Any) -> None:
            if value in (None, "") or key in seen:
                return
            seen.add(key)
            card.row(key, self._display_value(key, value))

        for key, value in summary.items():
            add(key, value)

        for key, value in (display_summary or {}).items():
            add(key, value)

        for key, value in handle.get_details().items():
            add(key, value)

        debug(_COMPONENT, card.render(), task_id=handle.task_id, include_task_prefix=False)

    @staticmethod
    def _display_value(key: str, value: Any) -> Any:
        if isinstance(value, os.PathLike) or (
            isinstance(value, str) and (key == "output" or key.endswith("_path") or key.endswith("_file"))
        ):
            return rel_path(value)
        return value

    @staticmethod
    def _error_message(explicit_error: Any, exc: BaseException) -> str:
        if explicit_error:
            return str(explicit_error)
        message = str(exc).strip()
        return message or exc.__class__.__name__


lifecycle = TaskLifecycleEmitter()
run_summary = lifecycle.run_summary
