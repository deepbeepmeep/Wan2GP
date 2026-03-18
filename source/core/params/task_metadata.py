"""Split task metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    run_id: str | None = None
    shot_id: str | None = None
    poll_interval: int | None = None
    poll_timeout: int | None = None

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "RunConfig":
        return cls(
            run_id=params.get("run_id"),
            shot_id=params.get("shot_id"),
            poll_interval=params.get("poll_interval"),
            poll_timeout=params.get("poll_timeout"),
        )

    def to_wgp_format(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("run_id", "shot_id", "poll_interval", "poll_timeout"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class TaskExtensions:
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "TaskExtensions":
        known = {"hires_scale"}
        values = {
            key: value
            for key, value in params.items()
            if key in known or key.startswith("x_") or key.startswith("experimental_")
        }
        return cls(values=values)

    def to_wgp_format(self) -> dict[str, Any]:
        return dict(self.values)


def known_task_keys() -> set[str]:
    return {
        "prompt",
        "run_id",
        "shot_id",
        "poll_interval",
        "poll_timeout",
        "hires_scale",
    }


__all__ = ["RunConfig", "TaskExtensions", "known_task_keys"]
