"""Task completion polling and generation lookup for live worker tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


TERMINAL_STATUSES = {"Complete", "Failed", "Cancelled"}


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    case_name: str
    task_type: str
    final_status: str
    output_location: str | None
    generation_ids: list[str]
    elapsed_sec: float
    error_summary: str | None


def _coerce_rows(result: Any) -> list[dict[str, Any]]:
    data = getattr(result, "data", None)
    if not data:
        return []
    if isinstance(data, dict):
        return [data]
    return [row for row in data if isinstance(row, dict)]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _linked_generation_ids(rows: list[dict[str, Any]], task_id: str) -> tuple[list[str], str | None]:
    generation_ids: list[str] = []
    seen_generation_ids: set[str] = set()
    first_location: str | None = None
    for row in rows:
        linked = False
        tasks = row.get("tasks")
        if isinstance(tasks, str):
            linked = tasks.strip() == task_id
        elif isinstance(tasks, list):
            linked = any(isinstance(item, str) and item.strip() == task_id for item in tasks)

        params = row.get("params")
        if not linked and isinstance(params, dict):
            linked = str(params.get("source_task_id", "")).strip() == task_id

        if not linked:
            continue

        generation_id = row.get("id")
        if generation_id:
            normalized_id = str(generation_id)
            if normalized_id not in seen_generation_ids:
                generation_ids.append(normalized_id)
                seen_generation_ids.add(normalized_id)
        if first_location is None and row.get("location"):
            first_location = str(row["location"])

    return generation_ids, first_location


def _fetch_task_row(db, task_id: str, project_id: str) -> dict[str, Any] | None:
    rows = _coerce_rows(
        db.supabase.table("tasks")
        .select("id, task_type, status, output_location, error_message, created_at, project_id")
        .eq("id", task_id)
        .eq("project_id", project_id)
        .execute()
    )
    return rows[0] if rows else None


def _fetch_generations_since(
    db,
    *,
    task_id: str,
    project_id: str,
    task_created_at: str | None,
) -> tuple[list[str], str | None]:
    query = (
        db.supabase.table("generations")
        .select("id, tasks, params, location, created_at, project_id")
        .eq("project_id", project_id)
    )
    if task_created_at:
        query = query.gte("created_at", task_created_at)

    rows = _coerce_rows(query.execute())
    if task_created_at:
        created_cutoff = _parse_timestamp(task_created_at)
        filtered_rows = []
        for row in rows:
            row_created_at = _parse_timestamp(row.get("created_at"))
            if created_cutoff is None or row_created_at is None or row_created_at >= created_cutoff:
                filtered_rows.append(row)
        rows = filtered_rows

    return _linked_generation_ids(rows, task_id)


def _failure_summary(task_row: dict[str, Any] | None, final_status: str, generation_ids: list[str]) -> str | None:
    if task_row is None:
        return "Task row disappeared before completion polling finished"

    if final_status in {"Failed", "Cancelled"}:
        return (
            task_row.get("error_message")
            or task_row.get("output_location")
            or f"Task reached terminal status {final_status}"
        )

    if final_status == "Complete" and not generation_ids:
        return "Task completed but no linked generations were found"

    return None


def poll_until_complete(
    db,
    task_id: str,
    project_id: str,
    *,
    timeout_sec: int,
    interval_sec: int = 5,
    case_name: str | None = None,
    task_type: str | None = None,
) -> TaskResult:
    """Poll a task row until terminal and summarize any linked generations."""
    started = time.monotonic()
    deadline = started + timeout_sec

    while time.monotonic() <= deadline:
        task_row = _fetch_task_row(db, task_id, project_id)
        if task_row is not None and task_row.get("status") in TERMINAL_STATUSES:
            final_status = str(task_row["status"])
            generation_ids, generation_location = _fetch_generations_since(
                db,
                task_id=task_id,
                project_id=project_id,
                task_created_at=task_row.get("created_at"),
            )
            output_location = task_row.get("output_location") or generation_location
            return TaskResult(
                task_id=task_id,
                case_name=case_name or task_id,
                task_type=task_type or str(task_row.get("task_type") or ""),
                final_status=final_status,
                output_location=str(output_location) if output_location else None,
                generation_ids=generation_ids,
                elapsed_sec=round(time.monotonic() - started, 3),
                error_summary=_failure_summary(task_row, final_status, generation_ids),
            )

        time.sleep(interval_sec)

    task_row = _fetch_task_row(db, task_id, project_id)
    final_status = str(task_row.get("status")) if task_row else "Timed Out"
    generation_ids, generation_location = _fetch_generations_since(
        db,
        task_id=task_id,
        project_id=project_id,
        task_created_at=task_row.get("created_at") if task_row else None,
    )
    output_location = ((task_row or {}).get("output_location") or generation_location)
    return TaskResult(
        task_id=task_id,
        case_name=case_name or task_id,
        task_type=task_type or str((task_row or {}).get("task_type") or ""),
        final_status=final_status,
        output_location=str(output_location) if output_location else None,
        generation_ids=generation_ids,
        elapsed_sec=round(time.monotonic() - started, 3),
        error_summary=f"Timed out waiting for task {task_id} to reach a terminal state",
    )


__all__ = ["TERMINAL_STATUSES", "TaskResult", "poll_until_complete"]
