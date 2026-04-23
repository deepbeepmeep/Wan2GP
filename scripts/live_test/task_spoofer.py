"""Fixture-backed task insertion helpers for the live-test harness."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from scripts.live_test import config


DEFAULT_FIXTURE_BY_TASK_TYPE = {
    "qwen_image": "qwen_image_basic",
    "qwen_image_2512": "qwen_image_basic",
    "qwen_image_edit": "qwen_image_edit_basic",
    "qwen_image_style": "qwen_image_style_db_task",
    "z_image_turbo_i2i": "z_image_turbo_i2i_basic",
    "individual_travel_segment": "wan22_i2v_individual_segment",
}


def _coerce_rows(result: Any) -> list[dict[str, Any]]:
    data = getattr(result, "data", None)
    if not data:
        return []
    if isinstance(data, dict):
        return [data]
    return [row for row in data if isinstance(row, dict)]


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _fixture_path(case_name: str) -> Path:
    try:
        return config.FIXTURES[case_name]
    except KeyError as exc:
        raise KeyError(f"Unknown fixture case: {case_name}") from exc


def _default_fixture_case(task_type: str) -> str:
    try:
        return DEFAULT_FIXTURE_BY_TASK_TYPE[task_type]
    except KeyError as exc:
        raise ValueError(
            f"No default fixture mapping exists for task_type={task_type!r}; pass fixture_case or fixture_payload"
        ) from exc


def load_fixture(case_name: str) -> dict[str, Any]:
    """Load a prepared worker-matrix fixture by case name."""
    return json.loads(_fixture_path(case_name).read_text(encoding="utf-8"))


def insert_spoof_task(
    db,
    project_id: str,
    task_type: str,
    params_overrides: dict[str, Any] | None,
    *,
    fixture_case: str | None = None,
    fixture_payload: dict[str, Any] | None = None,
) -> str:
    """Insert a live-test task row based on a fixture payload."""
    source_payload = fixture_payload if fixture_payload is not None else load_fixture(
        fixture_case or _default_fixture_case(task_type)
    )
    payload = copy.deepcopy(source_payload)
    payload.pop("notes", None)
    payload["project_id"] = project_id
    payload["task_type"] = task_type
    payload["status"] = "Queued"

    params = payload.get("params")
    if not isinstance(params, dict):
        params = {}
    payload["params"] = _deep_merge(params, params_overrides or {})
    payload["params"]["live_test"] = True

    result = db.supabase.table("tasks").insert(payload).execute()
    rows = _coerce_rows(result)
    if len(rows) != 1 or not rows[0].get("id"):
        raise RuntimeError("Failed to insert spoof task row")
    return str(rows[0]["id"])


__all__ = ["DEFAULT_FIXTURE_BY_TASK_TYPE", "insert_spoof_task", "load_fixture"]
