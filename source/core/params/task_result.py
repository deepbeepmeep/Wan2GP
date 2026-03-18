"""Structured result type for task handlers.

Replaces the (bool, str) return convention with a proper dataclass.
Eliminates magic string prefix parsing ("[ORCHESTRATOR_COMPLETE]", etc.)
by encoding the outcome as an enum.

Usage:
    from source.core.params.task_result import TaskResult, TaskOutcome

    # Simple success
    return TaskResult.success(output_path="/path/to/output.mp4")

    # Orchestrator complete
    return TaskResult.orchestrator_complete(output_path="/path/to/final.mp4", thumbnail_url="...")

    # Still orchestrating (children in progress)
    return TaskResult.orchestrating("3/5 segments complete")

    # Failure
    return TaskResult.failed("Model not found")
"""

from dataclasses import dataclass, field
from enum import Enum
import os
import warnings
from typing import Optional


class TaskOutcome(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    ORCHESTRATING = "orchestrating"
    ORCHESTRATOR_COMPLETE = "orchestrator_complete"


@dataclass(frozen=True)
class TaskResult:
    """Structured result from a task handler."""

    outcome: TaskOutcome
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    thumbnail_url: Optional[str] = None
    progress_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def success(cls, output_path: str, **metadata) -> "TaskResult":
        return cls(outcome=TaskOutcome.SUCCESS, output_path=output_path, metadata=metadata)

    @classmethod
    def failed(cls, message: str) -> "TaskResult":
        return cls(outcome=TaskOutcome.FAILED, error_message=message)

    @classmethod
    def orchestrator_complete(
        cls, output_path: str, thumbnail_url: Optional[str] = None
    ) -> "TaskResult":
        return cls(
            outcome=TaskOutcome.ORCHESTRATOR_COMPLETE,
            output_path=output_path,
            thumbnail_url=thumbnail_url,
        )

    @classmethod
    def orchestrating(cls, message: str) -> "TaskResult":
        return cls(outcome=TaskOutcome.ORCHESTRATING, progress_message=message)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_success(self) -> bool:
        """True for SUCCESS and ORCHESTRATOR_COMPLETE outcomes."""
        return self.outcome in (TaskOutcome.SUCCESS, TaskOutcome.ORCHESTRATOR_COMPLETE)

    @property
    def is_terminal(self) -> bool:
        """True for outcomes that indicate the task is done (success or failed)."""
        return self.outcome in (
            TaskOutcome.SUCCESS,
            TaskOutcome.FAILED,
            TaskOutcome.ORCHESTRATOR_COMPLETE,
        )

    # ------------------------------------------------------------------
    # Backward compatibility: tuple unpacking
    # ------------------------------------------------------------------

    def __iter__(self):
        """Allow ``success, output = result`` for backward compat.

        Maps to the old (bool, str|None) convention:
        - SUCCESS → (True, output_path)
        - ORCHESTRATOR_COMPLETE → (True, output_path)
        - ORCHESTRATING → (True, output_path)  # message in output_path
        - FAILED → (False, error_message)
        """
        if self.outcome == TaskOutcome.FAILED:
            yield False
            yield self.error_message
        elif self.outcome == TaskOutcome.ORCHESTRATING:
            yield True
            yield self.progress_message
        else:
            yield True
            yield self.output_path


def normalize_task_result(raw, *, orchestrator_task: bool = False) -> TaskResult:
    """Normalize legacy handler outputs into a TaskResult."""
    if isinstance(raw, TaskResult):
        return raw

    if isinstance(raw, tuple) and len(raw) == 2:
        strict = (os.getenv("HEADLESS_STRICT_TASK_RESULT") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if strict:
            raise TypeError("Legacy tuple task results are disabled in strict mode")

        warnings.warn(
            "Legacy tuple task results are deprecated; return TaskResult instead",
            DeprecationWarning,
            stacklevel=2,
        )
        success, payload = raw
        if success:
            if orchestrator_task and isinstance(payload, str) and not payload.endswith((".mp4", ".png", ".jpg", ".jpeg")):
                return TaskResult.orchestrating(payload)
            if payload is None:
                return TaskResult.success("")
            return TaskResult.success(payload)
        return TaskResult.failed("" if payload is None else str(payload))

    raise TypeError(f"Unsupported task result type: {type(raw).__name__}")
