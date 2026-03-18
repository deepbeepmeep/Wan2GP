"""Typed payloads for travel/edit orchestrator tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def coerce_edit_video_orchestrator_payload(payload, *, context: str, task_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} (task {task_id}): payload must be a mapping")
    normalized = dict(payload)
    portions = normalized.get("portions_to_regenerate")
    if not isinstance(portions, list) or not portions:
        raise ValueError(f"{context} (task {task_id}): portions_to_regenerate is required")
    if not normalized.get("source_video_url"):
        raise ValueError(f"{context} (task {task_id}): source_video_url is required")
    if not normalized.get("run_id"):
        raise ValueError(f"{context} (task {task_id}): run_id is required")
    return normalized


@dataclass(frozen=True)
class EditVideoOrchestratorPlan:
    source_video_url: str
    portions_to_regenerate: list[dict[str, int]]
    run_id: str
    source_video_fps: int | None = None
    replace_mode: bool = False
    enhance_prompt: bool = True
    per_join_settings: list[dict[str, Any]] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EditVideoOrchestratorPlan":
        portions: list[dict[str, int]] = []
        for index, portion in enumerate(payload.get("portions_to_regenerate", [])):
            try:
                start = int(portion["start_frame"])
                end = int(portion["end_frame"])
            except Exception as exc:
                raise ValueError(f"portion[{index}] requires integer start_frame/end_frame") from exc
            portions.append({"start_frame": start, "end_frame": end})

        enhance_prompt = payload.get("enhance_prompt", True)
        if not isinstance(enhance_prompt, bool):
            raise ValueError("enhance_prompt must be a boolean")
        replace_mode = payload.get("replace_mode", False)
        if not isinstance(replace_mode, bool):
            raise ValueError("replace_mode must be a boolean")

        return cls(
            source_video_url=payload["source_video_url"],
            portions_to_regenerate=portions,
            run_id=payload["run_id"],
            source_video_fps=payload.get("source_video_fps"),
            replace_mode=replace_mode,
            enhance_prompt=enhance_prompt,
            per_join_settings=payload.get("per_join_settings"),
        )


__all__ = ["EditVideoOrchestratorPlan", "coerce_edit_video_orchestrator_payload"]
