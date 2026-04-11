from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any

TERMINAL_STATUSES = {"Complete", "Failed", "Cancelled"}


@dataclass
class _TaskRecord:
    task_id: str
    task_type: str
    params: dict[str, Any]
    project_id: str | None
    attempts: int
    dependant_on: str | list[str] | None
    status: str
    output_location: str | None = None
    thumbnail_url: str | None = None
    created_index: int = 0


def inject_db_config() -> None:
    from source.core.db import config as db_config

    db_config.SUPABASE_URL = "https://preview.supabase.local"
    db_config.SUPABASE_ACCESS_TOKEN = "preview-token"


class SpoofDbRuntime:
    def __init__(self, fixtures: list[dict[str, Any]] | None = None):
        self._records: dict[str, _TaskRecord] = {}
        self._created_order: list[str] = []
        self._counter = 0
        for fixture in fixtures or []:
            self.seed_task(fixture)

    def _log(self, message: str) -> None:
        from source.core.log import headless_logger

        headless_logger.info(message)

    def _next_index(self) -> int:
        self._counter += 1
        return self._counter

    def _normalize_dependant_on(self, dependant_on: str | list[str] | None):
        if dependant_on is None:
            return None
        if isinstance(dependant_on, list):
            return list(dependant_on)
        return dependant_on

    def _orchestrator_ref(self, params: dict[str, Any]) -> str | None:
        if not isinstance(params, dict):
            return None
        direct = params.get("orchestrator_task_id_ref")
        if direct:
            return str(direct)
        nested = params.get("orchestrator_details")
        if isinstance(nested, dict):
            nested_ref = nested.get("orchestrator_task_id_ref") or nested.get("orchestrator_task_id")
            if nested_ref:
                return str(nested_ref)
        return None

    def _task_entry(self, record: _TaskRecord) -> dict[str, Any]:
        params = copy.deepcopy(record.params)
        return {
            "id": record.task_id,
            "task_type": record.task_type,
            "status": record.status,
            "params": params,
            "task_params": copy.deepcopy(params),
            "output_location": record.output_location or "",
            "created_at": f"preview-{record.created_index:04d}",
        }

    def _requeue_parent_orchestrator(self, record: _TaskRecord) -> None:
        parent_id = self._orchestrator_ref(record.params)
        if not parent_id:
            return
        parent = self._records.get(parent_id)
        if parent is None or parent.status in TERMINAL_STATUSES or parent.status == "Queued":
            return
        parent.status = "Queued"
        self._log(
            f"[SPOOF_DB] Requeued parent {parent_id} after child {record.task_id} reached {record.status}"
        )

    def as_task_info(self, task_id: str) -> dict[str, Any]:
        record = self._records[task_id]
        return {
            "task_id": record.task_id,
            "task_type": record.task_type,
            "params": copy.deepcopy(record.params),
            "project_id": record.project_id,
            "attempts": record.attempts,
        }

    def seed_task(self, fixture: dict[str, Any]) -> str:
        task_id = fixture["task_id"]
        record = _TaskRecord(
            task_id=task_id,
            task_type=fixture["task_type"],
            params=copy.deepcopy(fixture["params"]),
            project_id=fixture.get("project_id"),
            attempts=int(fixture.get("attempts", 0)),
            dependant_on=self._normalize_dependant_on(fixture.get("dependant_on")),
            status=fixture.get("status", "Queued"),
            created_index=self._next_index(),
        )
        self._records[task_id] = record
        self._created_order.append(task_id)
        return task_id

    def add_task_to_db(self, task_payload: dict, task_type_str: str, dependant_on: str | list[str] | None = None, db_path: str | None = None) -> str:
        _ = db_path
        task_id = str(uuid.uuid4())
        record = _TaskRecord(
            task_id=task_id,
            task_type=task_type_str,
            params=copy.deepcopy(task_payload),
            project_id=task_payload.get("project_id"),
            attempts=0,
            dependant_on=self._normalize_dependant_on(dependant_on),
            status="Queued",
            created_index=self._next_index(),
        )
        self._records[task_id] = record
        self._created_order.append(task_id)
        self._log(f"[SPOOF_DB] Added {task_type_str} {task_id} dependant_on={record.dependant_on}")
        return task_id

    def get_orchestrator_child_tasks(self, orchestrator_task_id: str) -> dict[str, list[dict[str, Any]]]:
        result = {
            "segments": [],
            "stitch": [],
            "join_clips_segment": [],
            "join_clips_orchestrator": [],
            "join_final_stitch": [],
        }
        for task_id in self._created_order:
            record = self._records[task_id]
            if record.task_id == orchestrator_task_id:
                continue
            if self._orchestrator_ref(record.params) != orchestrator_task_id:
                continue
            task_data = self._task_entry(record)
            if record.task_type == "travel_segment":
                result["segments"].append(task_data)
            elif record.task_type == "travel_stitch":
                result["stitch"].append(task_data)
            elif record.task_type == "join_clips_segment":
                result["join_clips_segment"].append(task_data)
            elif record.task_type == "join_clips_orchestrator":
                result["join_clips_orchestrator"].append(task_data)
            elif record.task_type == "join_final_stitch":
                result["join_final_stitch"].append(task_data)
        return result

    def get_task_dependency(self, task_id: str, max_retries: int = 3, retry_delay: float = 0.5):
        _ = max_retries, retry_delay
        record = self._records.get(task_id)
        return None if record is None else copy.deepcopy(record.dependant_on)

    def get_task_current_status(self, task_id: str) -> str | None:
        record = self._records.get(task_id)
        return None if record is None else record.status

    def cancel_orchestrator_children(self, orchestrator_task_id: str, reason: str = "Orchestrator cancelled") -> int:
        cancelled = 0
        for tasks in self.get_orchestrator_child_tasks(orchestrator_task_id).values():
            for task in tasks:
                record = self._records[task["id"]]
                if record.status in TERMINAL_STATUSES:
                    continue
                record.status = "Cancelled"
                record.output_location = reason
                cancelled += 1
        if cancelled:
            self._log(f"[SPOOF_DB] Cancelled {cancelled} children for {orchestrator_task_id}")
        return cancelled

    def cleanup_duplicate_child_tasks(self, orchestrator_task_id: str, expected_segments: int | None = None) -> dict[str, Any]:
        _ = orchestrator_task_id, expected_segments
        return {
            "duplicate_segments_removed": 0,
            "duplicate_stitch_removed": 0,
            "errors": [],
        }

    def get_task_output_location_from_db(self, task_id: str, runtime_config=None) -> str | None:
        _ = runtime_config
        record = self._records.get(task_id)
        if record is None or record.status != "Complete":
            return None
        return record.output_location

    def update_task_status(self, task_id: str, status: str, output_location: str | None = None):
        record = self._records[task_id]
        record.status = status
        record.output_location = output_location
        self._log(f"[SPOOF_DB] {task_id} -> {status}: {output_location}")
        if status in TERMINAL_STATUSES:
            self._requeue_parent_orchestrator(record)
        return True

    def update_task_status_supabase(self, task_id: str, status: str, output_location: str | None = None, thumbnail_url: str | None = None):
        record = self._records[task_id]
        record.status = status
        record.output_location = output_location
        record.thumbnail_url = thumbnail_url
        self._log(f"[SPOOF_DB] Complete: {task_id} -> {status}: {output_location}")
        if status in TERMINAL_STATUSES:
            self._requeue_parent_orchestrator(record)
        return True

    def reset_generation_started_at(self, task_id: str) -> bool:
        self._log(f"[SPOOF_DB] reset_generation_started_at({task_id})")
        return True

    def requeue_task_for_retry(self, task_id: str, error_message: str, current_attempts: int, error_category: str | None = None) -> bool:
        record = self._records[task_id]
        record.status = "Queued"
        record.output_location = error_message
        record.attempts = int(current_attempts) + 1
        self._log(f"[SPOOF_DB] Requeued {task_id} ({error_category or 'retry'})")
        return True

    # Aliases for run_preview patch wiring.
    add_task = add_task_to_db
    get_children = get_orchestrator_child_tasks
    get_dependency = get_task_dependency
    get_status = get_task_current_status
    cancel_children = cancel_orchestrator_children
    get_output = get_task_output_location_from_db
    update_status = update_task_status
    update_status_complete = update_task_status_supabase
    requeue = requeue_task_for_retry

    def all_terminal(self) -> bool:
        if not self._records:
            return True
        return all(record.status in TERMINAL_STATUSES for record in self._records.values())

    def next_ready_task_id(self) -> str | None:
        for task_id in self._created_order:
            record = self._records[task_id]
            if record.status != "Queued":
                continue
            dependencies = record.dependant_on
            if dependencies is None:
                return task_id
            if isinstance(dependencies, list):
                if all(self.get_task_current_status(dep) == "Complete" for dep in dependencies):
                    return task_id
            elif self.get_task_current_status(dependencies) == "Complete":
                return task_id
        return None


class SpoofTaskFeed:
    def __init__(self, db_runtime: SpoofDbRuntime, idle_polls_between_tasks: int = 1):
        self.db_runtime = db_runtime
        self.idle_polls_between_tasks = max(0, int(idle_polls_between_tasks))
        self._idle_budget = 0

    def poll(self, worker_id: str | None, same_model_only: bool, max_task_wait_minutes: int | None):
        _ = worker_id, same_model_only, max_task_wait_minutes
        from source.core.db.task_claim import ClaimPollOutcome

        if self._idle_budget > 0:
            self._idle_budget -= 1
            return ClaimPollOutcome.EMPTY, None

        task_id = self.db_runtime.next_ready_task_id()
        if task_id is not None:
            self.db_runtime.update_task_status(task_id, "In Progress", self.db_runtime._records[task_id].output_location)
            self._idle_budget = self.idle_polls_between_tasks
            return ClaimPollOutcome.CLAIMED, self.db_runtime.as_task_info(task_id)

        if self.db_runtime.all_terminal():
            return None, None

        return ClaimPollOutcome.EMPTY, None
