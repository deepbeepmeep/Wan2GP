"""Compatibility shim for the split travel orchestrator path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from source.core.params.task_result import TaskResult
from source.task_handlers.travel import orchestrator as _canonical

get_orchestrator_child_tasks = _canonical.get_orchestrator_child_tasks
_canonical_resolve_orchestrator_final_output = getattr(
    _canonical, "resolve_orchestrator_final_output", None
)
_CanonicalTravelRunContext = getattr(_canonical, "_TravelRunContext", object)
_CanonicalTravelSegmentRuntimeContext = getattr(
    _canonical, "_TravelSegmentRuntimeContext", object
)


class _TravelRunContext(_CanonicalTravelRunContext):
    """Compatibility alias for the split travel orchestrator runtime context."""


class _TravelSegmentRuntimeContext(_CanonicalTravelSegmentRuntimeContext):
    """Compatibility alias for the split travel segment runtime context."""


def resolve_orchestrator_final_output(*args: Any, **kwargs: Any):
    if _canonical_resolve_orchestrator_final_output is None:
        return None
    return _canonical_resolve_orchestrator_final_output(*args, **kwargs)


def _state_marker(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _segment_sort_key(task: dict[str, Any]) -> int:
    params = task.get("params") or {}
    try:
        return int(params.get("segment_index", -1))
    except (TypeError, ValueError):
        return -1


def _is_complete(task: dict[str, Any]) -> bool:
    return (task.get("status") or "").lower() == "complete"


def _is_terminal_failure(task: dict[str, Any]) -> bool:
    return (task.get("status") or "").lower() in {"failed", "cancelled", "canceled", "error"}


def _travel_handle_existing_children_idempotency(
    *,
    orchestrator_task_id_str: str,
    existing_segments: list[dict[str, Any]],
    existing_stitch: list[dict[str, Any]],
    existing_join_orchestrators: list[dict[str, Any]],
    expected_segments: int,
    required_stitch_count: int,
    required_join_orchestrator_count: int,
) -> TaskResult | None:
    has_required_segments = len(existing_segments) >= expected_segments
    has_required_stitch = len(existing_stitch) >= required_stitch_count
    has_required_join = len(existing_join_orchestrators) >= required_join_orchestrator_count

    marker_payload = {
        "task_id": orchestrator_task_id_str,
        "expected_segments": expected_segments,
        "required_stitch_count": required_stitch_count,
        "required_join_orchestrator_count": required_join_orchestrator_count,
        "segments": [
            {
                "id": task.get("id"),
                "status": task.get("status"),
                "segment_index": (task.get("params") or {}).get("segment_index"),
                "output_location": task.get("output_location"),
            }
            for task in sorted(existing_segments, key=_segment_sort_key)
        ],
        "stitch": [
            {
                "id": task.get("id"),
                "status": task.get("status"),
                "output_location": task.get("output_location"),
            }
            for task in existing_stitch
        ],
        "join_clips_orchestrator": [
            {
                "id": task.get("id"),
                "status": task.get("status"),
                "output_location": task.get("output_location"),
            }
            for task in existing_join_orchestrators
        ],
    }
    marker = _state_marker(marker_payload)

    all_segments_complete = existing_segments and all(_is_complete(task) for task in existing_segments)
    all_stitch_complete = required_stitch_count == 0 or (
        existing_stitch and all(_is_complete(task) for task in existing_stitch)
    )
    all_join_complete = required_join_orchestrator_count == 0 or (
        existing_join_orchestrators and all(_is_complete(task) for task in existing_join_orchestrators)
    )

    if any(_is_terminal_failure(task) for task in existing_segments + existing_stitch + existing_join_orchestrators):
        return TaskResult.failed(
            f"Existing child tasks entered a terminal failure state. marker={marker}"
        )

    if has_required_segments and has_required_stitch and has_required_join:
        if all_segments_complete and all_stitch_complete and all_join_complete:
            final_output = None
            if existing_join_orchestrators:
                final_output = existing_join_orchestrators[0].get("output_location")
            if not final_output and existing_stitch:
                final_output = existing_stitch[0].get("output_location")
            if not final_output and existing_segments:
                final_output = sorted(existing_segments, key=_segment_sort_key)[-1].get(
                    "output_location"
                )
            if not final_output:
                final_output = f"idempotent completion marker={marker}"
            return TaskResult.orchestrator_complete(final_output)

        return TaskResult.orchestrating(
            "Existing child tasks are still running; "
            f"waiting for completion before re-enqueue. marker={marker}"
        )

    if existing_segments or existing_stitch or existing_join_orchestrators:
        return TaskResult.orchestrating(
            "Partial child state detected; waiting for existing children to settle before enqueue. "
            f"marker={marker}"
        )

    return None


def handle_travel_orchestrator_task(
    task_params_from_db: dict[str, Any],
    main_output_dir_base: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
):
    orchestrator_payload = task_params_from_db.get("orchestrator_details") or {}
    if orchestrator_payload.get("num_new_segments_to_generate", 0) <= 0:
        return TaskResult.orchestrator_complete("no-op payload: no new segments to generate")

    original_get_orchestrator_child_tasks = _canonical.get_orchestrator_child_tasks
    try:
        _canonical.get_orchestrator_child_tasks = get_orchestrator_child_tasks
        return _canonical.handle_travel_orchestrator_task(
            task_params_from_db=task_params_from_db,
            main_output_dir_base=main_output_dir_base,
            orchestrator_task_id_str=orchestrator_task_id_str,
            orchestrator_project_id=orchestrator_project_id,
        )
    finally:
        _canonical.get_orchestrator_child_tasks = original_get_orchestrator_child_tasks


__all__ = [
    "_TravelRunContext",
    "_TravelSegmentRuntimeContext",
    "_travel_handle_existing_children_idempotency",
    "get_orchestrator_child_tasks",
    "handle_travel_orchestrator_task",
    "resolve_orchestrator_final_output",
]
