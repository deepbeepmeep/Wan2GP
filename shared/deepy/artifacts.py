import copy
import json
import math
import threading
import time
import uuid
from typing import Any


ARTIFACT_INLINE_ITEM_THRESHOLD = 10
ARTIFACT_INLINE_TOKEN_THRESHOLD = 2048
ARTIFACT_LIMITS = {"artifacts": 32, "items_per_artifact": 10000, "bytes_per_artifact": 8 * 1024 * 1024, "query_page_items": 500}
_MAX_ARTIFACTS = ARTIFACT_LIMITS["artifacts"]
_MAX_ITEMS = ARTIFACT_LIMITS["items_per_artifact"]
_MAX_SERIALIZED_BYTES = ARTIFACT_LIMITS["bytes_per_artifact"]
_MAX_PAGE_ITEMS = ARTIFACT_LIMITS["query_page_items"]
_MISSING = object()


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise TypeError("Artifact values must be JSON serializable.") from exc


def _field(value: Any, path: str, default: Any = _MISSING) -> Any:
    current = value
    for part in str(path or "").split("."):
        if not part:
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        if default is _MISSING:
            raise KeyError(f"Unknown artifact field: {path}")
        return default
    return current


def _remove_field(value: dict[str, Any], path: str) -> None:
    parts = [part for part in str(path or "").split(".") if part]
    if not parts:
        raise ValueError("remove path is empty")
    parent: Any = value
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            return
        parent = parent[part]
    if isinstance(parent, dict):
        parent.pop(parts[-1], None)


def _deep_merge(target: dict[str, Any], updates: dict[str, Any], *, preserve_container_types: bool = False, path: str = "") -> None:
    for key, value in updates.items():
        field_path = f"{path}.{key}" if path else str(key)
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value, preserve_container_types=preserve_container_types, path=field_path)
        else:
            existing = target.get(key, _MISSING)
            if preserve_container_types and existing is not _MISSING:
                if isinstance(existing, dict) != isinstance(value, dict) and (isinstance(existing, dict) or isinstance(value, dict)):
                    raise TypeError(f"Ledger field {field_path} cannot change object shape; remove it explicitly before changing its type.")
                if isinstance(existing, list) != isinstance(value, list) and (isinstance(existing, list) or isinstance(value, list)):
                    raise TypeError(f"Ledger field {field_path} cannot change array shape; remove it explicitly before changing its type.")
            target[key] = _json_copy(value)


def _matches(record: Any, filters: list[dict[str, Any]] | None) -> bool:
    for condition in list(filters or []):
        if not isinstance(condition, dict):
            raise TypeError("Artifact filters must be objects.")
        field_name = str(condition.get("field", "") or "").strip()
        if not field_name:
            raise ValueError("Artifact filter field is required.")
        actual = _field(record, field_name, None)
        expected = condition.get("value")
        operator = str(condition.get("op", "eq") or "eq").strip().lower()
        if operator == "eq":
            matched = actual == expected
        elif operator == "ne":
            matched = actual != expected
        elif operator == "contains":
            matched = str(expected).casefold() in str(actual).casefold()
        elif operator == "starts_with":
            matched = str(actual).casefold().startswith(str(expected).casefold())
        elif operator == "ends_with":
            matched = str(actual).casefold().endswith(str(expected).casefold())
        elif operator == "in":
            if not isinstance(expected, list):
                raise TypeError(f"Artifact filter {field_name} requires an array value for in.")
            matched = actual in expected
        elif operator in {"gt", "gte", "lt", "lte"}:
            try:
                left, right = float(actual), float(expected)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Artifact filter {field_name} requires numeric values for {operator}.") from exc
            matched = {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[operator]
        else:
            raise ValueError(f"Unknown artifact filter operator: {operator}")
        if not matched:
            return False
    return True


def _validate_type(value: Any, expected: str) -> bool:
    return {
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "null": value is None,
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "string": isinstance(value, str),
    }.get(expected, True)


def _validate_record(record: Any, schema: dict[str, Any]) -> None:
    if not schema:
        return
    expected_type = str(schema.get("type", "object") or "object")
    if not _validate_type(record, expected_type):
        raise TypeError(f"Artifact item must be {expected_type}.")
    if not isinstance(record, dict):
        return
    missing = [str(name) for name in list(schema.get("required", []) or []) if name not in record]
    if missing:
        raise ValueError(f"Artifact item is missing required field: {missing[0]}")
    for name, definition in dict(schema.get("properties", {}) or {}).items():
        if name not in record or not isinstance(definition, dict) or not definition.get("type"):
            continue
        if not _validate_type(record[name], str(definition["type"])):
            raise TypeError(f"Artifact field {name} must be {definition['type']}.")


class ArtifactWorkspace:
    def __init__(self) -> None:
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._workflows: dict[str, dict[str, Any]] = {}
        self._operation_results: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._artifacts.clear()
            self._workflows.clear()
            self._operation_results.clear()

    def _artifact(self, artifact_id: str) -> dict[str, Any]:
        artifact_id = str(artifact_id or "").strip()
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"Unknown artifact_id: {artifact_id}") from exc

    @staticmethod
    def _operation_signature(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _repeated_operation(self, artifact_id: str, operation_id: str, payload: Any) -> dict[str, Any] | None:
        operation_id = str(operation_id or "").strip()
        if not operation_id:
            return None
        key = (artifact_id, operation_id)
        signature = self._operation_signature(payload)
        previous = self._operation_results.get(key)
        if previous is None:
            return None
        if previous[0] != signature:
            raise ValueError(f"operation_id {operation_id!r} was already used with different arguments.")
        result = _json_copy(previous[1])
        result["replayed"] = True
        return result

    def _remember_operation(self, artifact_id: str, operation_id: str, payload: Any, result: dict[str, Any]) -> None:
        operation_id = str(operation_id or "").strip()
        if operation_id:
            self._operation_results[(artifact_id, operation_id)] = (self._operation_signature(payload), _json_copy(result))

    @staticmethod
    def _check_revision(artifact: dict[str, Any], expected_revision: Any) -> None:
        if expected_revision is not None and int(expected_revision) != int(artifact["revision"]):
            raise ValueError(f"Artifact revision conflict: expected {int(expected_revision)}, current {int(artifact['revision'])}.")

    @staticmethod
    def _serialized_size(artifact: dict[str, Any]) -> int:
        return len(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def _check_size(self, artifact: dict[str, Any]) -> None:
        size = self._serialized_size(artifact)
        if size > _MAX_SERIALIZED_BYTES:
            raise ValueError(f"Artifact exceeds its {_MAX_SERIALIZED_BYTES:,}-byte storage quota ({size:,} bytes).")

    @staticmethod
    def _next_required_action(artifact: dict[str, Any], workflow: dict[str, Any]) -> dict[str, Any] | None:
        artifact_id, cursor, phase = artifact["artifact_id"], workflow["cursor"], workflow["phase"]
        if phase == "prepare_item":
            return {"tool": "wangp_artifact", "action": "prepare", "arguments": {"artifact_id": artifact_id}, "purpose": f"Load the authoritative context packet for item {cursor}."}
        if phase == "commit_item":
            return {"tool": "wangp_artifact", "action": "commit_item", "fixed_arguments": {"artifact_id": artifact_id}, "required_arguments": ["item", "operation_id"], "optional_arguments": ["ledger_values", "ledger_remove", "expected_revision"], "purpose": f"Store item {cursor} and advance the workflow. Include only a compact ledger delta that is ready; ledger_values is optional and missing ledger updates may be applied later with update_ledger."}
        if phase == "finalize":
            return {"tool": "wangp_artifact", "action": "finalize", "arguments": {"artifact_id": artifact_id}, "purpose": "Validate and freeze the completed managed collection."}
        return None

    def _add_workflow_feedback(self, result: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
        workflow = self._workflows.get(artifact["artifact_id"])
        if workflow is None:
            return result
        result.pop("next_required_action", None)
        result["workflow"] = {"managed": True, "phase": workflow["phase"], "cursor": workflow["cursor"], "ledger_id": workflow["ledger_id"]}
        next_action = self._next_required_action(artifact, workflow)
        if next_action is not None:
            result["next_required_action"] = next_action
        return result

    def _normalize_workflow(self, workflow: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
        workflow = _json_copy(dict(workflow or {}))
        allowed = {"ledger_id", "plan_path", "context_fields", "index_field", "end_state_field", "start_index"}
        unknown = sorted(set(workflow) - allowed)
        if unknown:
            raise ValueError(f"Unsupported managed workflow field: {unknown[0]}")
        ledger_id = str(workflow.get("ledger_id", "") or "").strip()
        plan_path = str(workflow.get("plan_path", "") or "").strip()
        context_fields = [str(field).strip() for field in list(workflow.get("context_fields", []) or []) if str(field).strip()]
        if not ledger_id or not plan_path or not context_fields:
            raise ValueError("Managed workflow requires ledger_id, plan_path, and non-empty context_fields.")
        ledger = self._artifact(ledger_id)
        if ledger["kind"] != "ledger":
            raise TypeError("Managed workflow ledger_id must identify a ledger artifact.")
        if artifact["expected_items"] is None:
            raise ValueError("Managed workflow requires expected_items.")
        if len(artifact["items"]) > artifact["expected_items"]:
            raise ValueError("Managed workflow initial items exceed expected_items.")
        index_field = str(workflow.get("index_field", "index") or "index").strip()
        end_state_field = str(workflow.get("end_state_field", "end_state") or "end_state").strip()
        required_fields = set(artifact["schema"].get("required", []) or [])
        missing_schema_field = next((field for field in (index_field, end_state_field) if field not in required_fields), None)
        if missing_schema_field:
            raise ValueError(f"Managed workflow schema must require {missing_schema_field}.")
        if not isinstance(_field(ledger["data"], plan_path), (dict, list)):
            raise TypeError(f"Managed workflow plan_path {plan_path} must identify an object or array.")
        for field in context_fields:
            _field(ledger["data"], field)
        start_index = int(workflow.get("start_index", 1))
        indices = [int(_field(item, index_field)) for item in artifact["items"]]
        if indices != list(range(start_index, start_index + len(indices))):
            raise ValueError(f"Managed workflow initial {index_field} values must be contiguous from {start_index}.")
        cursor = start_index + len(indices)
        return {"ledger_id": ledger_id, "plan_path": plan_path, "context_fields": context_fields, "index_field": index_field, "end_state_field": end_state_field, "start_index": start_index, "cursor": cursor, "phase": "finalize" if len(indices) == artifact["expected_items"] else "prepare_item", "prepared_ledger_revision": None}

    def _workflow_error(self, artifact: dict[str, Any], message: str) -> ValueError:
        workflow = self._workflows[artifact["artifact_id"]]
        return ValueError(f"{message} Next required action: {json.dumps(self._next_required_action(artifact, workflow), ensure_ascii=False, separators=(',', ':'))}")

    def create(self, kind: str = "record_set", title: str = "", schema: dict[str, Any] | None = None, expected_items: int | None = None, initial_items: list[Any] | None = None, initial_data: dict[str, Any] | None = None, workflow: dict[str, Any] | None = None) -> dict[str, Any]:
        kind = str(kind or "record_set").strip().lower()
        if kind not in {"record_set", "ledger"}:
            raise ValueError("Artifact kind must be record_set or ledger.")
        with self._lock:
            if len(self._artifacts) >= _MAX_ARTIFACTS:
                raise ValueError(f"Artifact workspace already contains the maximum of {_MAX_ARTIFACTS} artifacts.")
            artifact_id = f"artifact_{uuid.uuid4().hex[:12]}"
            now = time.time()
            artifact = {
                "artifact_id": artifact_id,
                "kind": kind,
                "title": str(title or "").strip() or ("Project ledger" if kind == "ledger" else "Working collection"),
                "schema": _json_copy(dict(schema or {})),
                "expected_items": None if expected_items is None else max(0, int(expected_items)),
                "revision": 0,
                "finalized": False,
                "created": now,
                "updated": now,
                "items": [],
                "data": {},
            }
            if kind == "record_set":
                items = _json_copy(list(initial_items or []))
                if len(items) > _MAX_ITEMS:
                    raise ValueError(f"Artifact cannot contain more than {_MAX_ITEMS:,} items.")
                for item in items:
                    _validate_record(item, artifact["schema"])
                artifact["items"] = items
            else:
                artifact["data"] = _json_copy(dict(initial_data or {}))
                _validate_record(artifact["data"], artifact["schema"])
            if workflow and kind != "record_set":
                raise ValueError("Managed workflow is available only for record_set artifacts.")
            normalized_workflow = self._normalize_workflow(workflow, artifact) if workflow else None
            self._check_size(artifact)
            self._artifacts[artifact_id] = artifact
            if normalized_workflow is not None:
                self._workflows[artifact_id] = normalized_workflow
            return self.status(artifact_id)

    def status(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact(artifact_id)
            result = {
                "status": "done",
                "artifact_id": artifact["artifact_id"],
                "kind": artifact["kind"],
                "title": artifact["title"],
                "revision": artifact["revision"],
                "finalized": artifact["finalized"],
                "size_bytes": self._serialized_size(artifact),
            }
            if artifact["kind"] == "record_set":
                result.update(item_count=len(artifact["items"]), expected_items=artifact["expected_items"], next_index=len(artifact["items"]) + 1)
            else:
                result["sections"] = list(artifact["data"])
            return self._add_workflow_feedback(result, artifact)

    def list_artifacts(self, offset: int = 0, limit: int = 32) -> dict[str, Any]:
        with self._lock:
            artifacts = sorted(self._artifacts.values(), key=lambda artifact: float(artifact["updated"]), reverse=True)
            offset, limit = max(0, int(offset)), max(1, min(int(limit), _MAX_ARTIFACTS))
            items = [self.status(artifact["artifact_id"]) for artifact in artifacts[offset:offset + limit]]
            return {"status": "done", "items": items, "count": len(items), "matched": len(artifacts), "offset": offset, "has_more": offset + len(items) < len(artifacts), "next_offset": offset + len(items) if offset + len(items) < len(artifacts) else None}

    def prepare(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact(artifact_id)
            workflow = self._workflows.get(artifact_id)
            if workflow is None:
                raise ValueError("prepare is available only for managed record-set workflows.")
            if workflow["phase"] not in {"prepare_item", "commit_item"}:
                raise self._workflow_error(artifact, "The managed workflow is not ready to prepare an item.")
            ledger = self._artifact(workflow["ledger_id"])
            cursor = workflow["cursor"]
            plan = _field(ledger["data"], workflow["plan_path"])
            if isinstance(plan, dict):
                if str(cursor) not in plan:
                    raise KeyError(f"Managed workflow plan has no item {cursor} at {workflow['plan_path']}.")
                current_plan = plan[str(cursor)]
            elif isinstance(plan, list):
                position = cursor - workflow["start_index"]
                if position < 0 or position >= len(plan):
                    raise IndexError(f"Managed workflow plan has no item {cursor} at {workflow['plan_path']}.")
                current_plan = plan[position]
            else:
                raise TypeError(f"Managed workflow plan_path {workflow['plan_path']} must identify an object or array.")
            previous_end_state = None
            if cursor > workflow["start_index"]:
                previous = [item for item in artifact["items"] if int(_field(item, workflow["index_field"])) == cursor - 1]
                if len(previous) != 1:
                    raise ValueError(f"Managed workflow requires exactly one preceding item with {workflow['index_field']}={cursor - 1}.")
                previous_end_state = _json_copy(_field(previous[0], workflow["end_state_field"]))
            context = {field: _json_copy(_field(ledger["data"], field)) for field in workflow["context_fields"]}
            workflow["phase"] = "commit_item"
            workflow["prepared_ledger_revision"] = ledger["revision"]
            exact_queries = {"records": {"action": "query", "arguments": {"artifact_id": artifact_id, "offset": 0, "limit": 50}}, "ledger": {"action": "query", "arguments": {"artifact_id": ledger["artifact_id"], "fields": list(workflow["context_fields"])}}}
            if cursor > workflow["start_index"]:
                exact_queries["previous_record"] = {"action": "query", "arguments": {"artifact_id": artifact_id, "filters": [{"field": workflow["index_field"], "op": "eq", "value": cursor - 1}], "offset": 0, "limit": 1}}
            result = {"status": "done", "artifact_id": artifact_id, "item_index": cursor, "plan": _json_copy(current_plan), "previous_end_state": previous_end_state, "ledger_context": context, "ledger_revision": ledger["revision"], "bounded_context": True, "included_ledger_fields": list(workflow["context_fields"]), "exact_queries": exact_queries, "guidance": "This packet intentionally omits full prior records and unselected ledger sections. Use the supplied exact_queries only when the next item requires exact older material."}
            return self._add_workflow_feedback(result, artifact)

    def append(self, artifact_id: str, items: list[Any], expected_revision: int | None = None, operation_id: str = "") -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact(artifact_id)
            if artifact["kind"] != "record_set":
                raise TypeError("append is available only for record_set artifacts.")
            if artifact["finalized"]:
                raise ValueError("Finalized artifacts are immutable.")
            items = _json_copy(list(items or []))
            if not items:
                raise ValueError("items must be a non-empty array.")
            payload = {"items": items}
            repeated = self._repeated_operation(artifact_id, operation_id, payload)
            if repeated is not None:
                return self._add_workflow_feedback(repeated, artifact)
            workflow = self._workflows.get(artifact_id)
            if workflow is not None:
                raise self._workflow_error(artifact, "Managed workflows use commit_item so record and ledger changes remain atomic.")
            self._check_revision(artifact, expected_revision)
            if len(artifact["items"]) + len(items) > _MAX_ITEMS:
                raise ValueError(f"Artifact cannot contain more than {_MAX_ITEMS:,} items.")
            for item in items:
                _validate_record(item, artifact["schema"])
            previous, previous_updated = list(artifact["items"]), artifact["updated"]
            artifact["items"].extend(items)
            artifact["revision"] += 1
            artifact["updated"] = time.time()
            try:
                self._check_size(artifact)
            except Exception:
                artifact["items"] = previous
                artifact["revision"] -= 1
                artifact["updated"] = previous_updated
                raise
            result = self.status(artifact_id)
            result["accepted"] = len(items)
            self._remember_operation(artifact_id, operation_id, payload, result)
            return result

    def commit_item(self, artifact_id: str, item: Any, ledger_values: dict[str, Any] | None = None, ledger_remove: list[str] | None = None, expected_revision: int | None = None, operation_id: str = "") -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact(artifact_id)
            workflow = self._workflows.get(artifact_id)
            if workflow is None:
                raise ValueError("commit_item is available only for managed record-set workflows.")
            operation_id = str(operation_id or "").strip()
            if not operation_id:
                raise ValueError("Managed commit_item requires operation_id for retry safety.")
            item = _json_copy(item)
            ledger_values = _json_copy(dict(ledger_values or {}))
            ledger_remove = [str(path) for path in list(ledger_remove or [])]
            payload = {"item": item, "ledger_values": ledger_values, "ledger_remove": ledger_remove}
            repeated = self._repeated_operation(artifact_id, operation_id, payload)
            if repeated is not None:
                return self._add_workflow_feedback(repeated, artifact)
            if workflow["phase"] != "commit_item":
                raise self._workflow_error(artifact, "Prepare the authoritative item context before committing it.")
            ledger = self._artifact(workflow["ledger_id"])
            if ledger["revision"] != workflow["prepared_ledger_revision"]:
                workflow["phase"] = "prepare_item"
                workflow["prepared_ledger_revision"] = None
                raise self._workflow_error(artifact, "The ledger changed after preparation; refresh the item context.")
            self._check_revision(artifact, expected_revision)
            if len(artifact["items"]) >= _MAX_ITEMS:
                raise ValueError(f"Artifact cannot contain more than {_MAX_ITEMS:,} items.")
            actual_index = int(_field(item, workflow["index_field"]))
            if actual_index != workflow["cursor"]:
                raise ValueError(f"Managed workflow expected {workflow['index_field']}={workflow['cursor']}; received {actual_index}.")
            _validate_record(item, artifact["schema"])

            next_items = [*artifact["items"], item]
            next_ledger_data = _json_copy(ledger["data"])
            _deep_merge(next_ledger_data, ledger_values, preserve_container_types=True)
            for path in ledger_remove:
                _remove_field(next_ledger_data, path)
            _validate_record(next_ledger_data, ledger["schema"])
            next_artifact = {**artifact, "items": next_items, "revision": artifact["revision"] + 1, "updated": time.time()}
            next_ledger = {**ledger, "data": next_ledger_data}
            if ledger_values or ledger_remove:
                next_ledger["revision"] = ledger["revision"] + 1
                next_ledger["updated"] = time.time()
            self._check_size(next_artifact)
            self._check_size(next_ledger)

            artifact.update(items=next_items, revision=next_artifact["revision"], updated=next_artifact["updated"])
            if ledger_values or ledger_remove:
                ledger.update(data=next_ledger_data, revision=next_ledger["revision"], updated=next_ledger["updated"])
            committed_index = workflow["cursor"]
            workflow["cursor"] += 1
            workflow["prepared_ledger_revision"] = None
            workflow["phase"] = "finalize" if artifact["expected_items"] is not None and len(artifact["items"]) >= artifact["expected_items"] else "prepare_item"
            result = self.status(artifact_id)
            result.update(accepted=1, committed_item_index=committed_index, ledger={"artifact_id": ledger["artifact_id"], "revision": ledger["revision"], "sections": list(ledger["data"])}, exact_record_query={"action": "query", "arguments": {"artifact_id": artifact_id, "filters": [{"field": workflow["index_field"], "op": "eq", "value": committed_index}], "offset": 0, "limit": 1}})
            self._remember_operation(artifact_id, operation_id, payload, result)
            return result

    def query(self, artifact_id: str, filters: list[dict[str, Any]] | None = None, fields: list[str] | None = None, sort_by: str = "", descending: bool = False, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact(artifact_id)
            if artifact["kind"] == "ledger":
                data = artifact["data"]
                selected = _json_copy(data if not fields else {field_name: _field(data, field_name) for field_name in fields})
                return {**self.status(artifact_id), "data": selected, "authoritative": True, "included_fields": list(data) if not fields else list(fields), "guidance": "This is exact ledger data for the requested sections. Query other section names explicitly rather than relying on transcript memory."}
            records = [record for record in artifact["items"] if _matches(record, filters)]
            sort_by = str(sort_by or "").strip()
            if sort_by:
                def sort_key(record: Any) -> tuple[int, Any]:
                    value = _field(record, sort_by, None)
                    if value is None:
                        return 2, ""
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        return 0, float(value)
                    return 1, str(value).casefold()

                records.sort(key=sort_key, reverse=bool(descending))
            offset, limit = max(0, int(offset)), max(1, min(int(limit), _MAX_PAGE_ITEMS))
            page = records[offset:offset + limit]
            if fields:
                page = [{field_name: _field(record, field_name, None) for field_name in fields} for record in page]
            result = {
                **self.status(artifact_id),
                "items": _json_copy(page),
                "count": len(page),
                "matched": len(records),
                "offset": offset,
                "has_more": offset + len(page) < len(records),
                "next_offset": offset + len(page) if offset + len(page) < len(records) else None,
                "authoritative": True,
                "included_fields": list(fields or []),
            }
            result["guidance"] = "This is an exact bounded record page. Use next_offset to continue; do not reconstruct omitted records from memory." if result["has_more"] else "This is the exact requested record page; no matching records remain after this page."
            return result

    def update_records(self, artifact_id: str, values: dict[str, Any], filters: list[dict[str, Any]] | None = None, expected_revision: int | None = None, operation_id: str = "") -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact(artifact_id)
            if artifact["kind"] != "record_set":
                raise TypeError("update_records is available only for record_set artifacts.")
            if artifact["finalized"]:
                raise ValueError("Finalized artifacts are immutable.")
            values = _json_copy(dict(values or {}))
            if not values:
                raise ValueError("values must not be empty.")
            payload = {"values": values, "filters": filters}
            repeated = self._repeated_operation(artifact_id, operation_id, payload)
            if repeated is not None:
                return repeated
            self._check_revision(artifact, expected_revision)
            previous, previous_updated = _json_copy(artifact["items"]), artifact["updated"]
            updated = 0
            for record in artifact["items"]:
                if not isinstance(record, dict) or not _matches(record, filters):
                    continue
                _deep_merge(record, values)
                _validate_record(record, artifact["schema"])
                updated += 1
            if updated == 0:
                raise ValueError("No artifact records matched the update filters.")
            artifact["revision"] += 1
            artifact["updated"] = time.time()
            try:
                self._check_size(artifact)
            except Exception:
                artifact["items"] = previous
                artifact["revision"] -= 1
                artifact["updated"] = previous_updated
                raise
            result = self.status(artifact_id)
            result["updated"] = updated
            self._remember_operation(artifact_id, operation_id, payload, result)
            return result

    def update_ledger(self, artifact_id: str, values: dict[str, Any], remove: list[str] | None = None, expected_revision: int | None = None, operation_id: str = "") -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact(artifact_id)
            if artifact["kind"] != "ledger":
                raise TypeError("update_ledger is available only for ledger artifacts.")
            values, remove = _json_copy(dict(values or {})), [str(path) for path in list(remove or [])]
            if not values and not remove:
                raise ValueError("values or remove is required.")
            payload = {"values": values, "remove": remove}
            repeated = self._repeated_operation(artifact_id, operation_id, payload)
            if repeated is not None:
                return repeated
            self._check_revision(artifact, expected_revision)
            next_data = _json_copy(artifact["data"])
            _deep_merge(next_data, values, preserve_container_types=True)
            for path in remove:
                _remove_field(next_data, path)
            _validate_record(next_data, artifact["schema"])
            next_artifact = {**artifact, "data": next_data, "revision": artifact["revision"] + 1, "updated": time.time()}
            self._check_size(next_artifact)
            artifact.update(data=next_data, revision=next_artifact["revision"], updated=next_artifact["updated"])
            result = self.status(artifact_id)
            self._remember_operation(artifact_id, operation_id, payload, result)
            return result

    def finalize(self, artifact_id: str, expected_revision: int | None = None, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact(artifact_id)
            if artifact["kind"] != "record_set":
                raise TypeError("Only record_set artifacts are finalized; ledgers remain mutable.")
            workflow = self._workflows.get(artifact_id)
            if workflow is not None and workflow["phase"] not in {"finalize", "complete"}:
                raise self._workflow_error(artifact, "The managed workflow cannot be finalized yet.")
            constraints = dict(constraints or {})
            already_finalized = artifact["finalized"]
            if not already_finalized:
                self._check_revision(artifact, expected_revision)
            items = artifact["items"]
            expected_items = constraints.get("expected_items", artifact["expected_items"])
            if expected_items is not None and len(items) != int(expected_items):
                raise ValueError(f"Artifact has {len(items)} items; expected {int(expected_items)}.")
            for field_name in list(constraints.get("required_fields", []) or []):
                for index, item in enumerate(items, 1):
                    value = _field(item, str(field_name), None)
                    if value is None or isinstance(value, str) and not value.strip():
                        raise ValueError(f"Artifact item {index} is missing required field {field_name}.")
            unique_by = str(constraints.get("unique_by", "") or "").strip()
            if unique_by:
                values = [_field(item, unique_by) for item in items]
                if len({json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values}) != len(values):
                    raise ValueError(f"Artifact field {unique_by} contains duplicate values.")
            contiguous_field = str(constraints.get("contiguous_field", "") or "").strip()
            if contiguous_field:
                values = sorted(int(_field(item, contiguous_field)) for item in items)
                start = int(constraints.get("contiguous_start", 1))
                if values != list(range(start, start + len(values))):
                    raise ValueError(f"Artifact field {contiguous_field} must be contiguous from {start}.")
            sum_field = str(constraints.get("sum_field", "") or "").strip()
            if sum_field:
                total = sum(float(_field(item, sum_field)) for item in items)
                expected_sum = float(constraints["expected_sum"])
                tolerance = max(0.0, float(constraints.get("sum_tolerance", 1e-6)))
                if not math.isclose(total, expected_sum, abs_tol=tolerance, rel_tol=0.0):
                    raise ValueError(f"Artifact field {sum_field} totals {total:g}; expected {expected_sum:g}.")
            if not already_finalized:
                artifact["finalized"] = True
                artifact["revision"] += 1
                artifact["updated"] = time.time()
            if workflow is not None:
                workflow["phase"] = "complete"
            result = self.status(artifact_id)
            result["already_finalized"] = already_finalized
            result["validated_constraints"] = _json_copy(constraints)
            return result

    def delete(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact(artifact_id)
            dependent = [managed_id for managed_id, workflow in self._workflows.items() if workflow["ledger_id"] == artifact_id]
            if dependent:
                raise ValueError(f"Ledger is still required by managed workflow {dependent[0]}.")
            del self._artifacts[artifact_id]
            self._workflows.pop(artifact_id, None)
            for key in [key for key in self._operation_results if key[0] == artifact_id]:
                del self._operation_results[key]
            return {"status": "done", "artifact_id": artifact_id, "deleted": True, "title": artifact["title"]}

    def reference_status(self, reference: dict[str, Any]) -> dict[str, Any]:
        artifact_id = str(reference.get("$artifact", "") or "").strip()
        with self._lock:
            artifact = self._artifact(artifact_id)
            result = {"artifact_id": artifact_id, "kind": artifact["kind"], "revision": artifact["revision"], "finalized": artifact["finalized"]}
            if artifact["kind"] == "ledger":
                result["rendered_items"] = 1
                return result
            records = [record for record in artifact["items"] if _matches(record, reference.get("where"))]
            offset = max(0, int(reference.get("offset", 0) or 0))
            limit = reference.get("limit")
            records = records[offset:] if limit is None else records[offset:offset + max(0, int(limit))]
            result.update(source_items=len(artifact["items"]), rendered_items=len(records), expected_items=artifact["expected_items"])
            indices = [record.get("index") for record in records if isinstance(record, dict) and isinstance(record.get("index"), int) and not isinstance(record.get("index"), bool)]
            if len(indices) == len(records) and indices:
                result.update(first_index=indices[0], last_index=indices[-1], contiguous_indices=indices == list(range(indices[0], indices[0] + len(indices))))
            return result

    def resolve(self, reference: dict[str, Any], require_finalized: bool = False) -> Any:
        artifact_id = str(reference.get("$artifact", "") or "").strip()
        with self._lock:
            artifact = self._artifact(artifact_id)
            if require_finalized and artifact["kind"] == "record_set" and not artifact["finalized"]:
                raise ValueError(f"Artifact {artifact_id} must be finalized before it can be consumed.")
            if artifact["kind"] == "ledger":
                value = _json_copy(artifact["data"])
                select = str(reference.get("select", "") or "").strip()
                return _json_copy(_field(value, select)) if select else value
            records = [record for record in artifact["items"] if _matches(record, reference.get("where"))]
            offset = max(0, int(reference.get("offset", 0) or 0))
            limit = reference.get("limit")
            records = records[offset:] if limit is None else records[offset:offset + max(0, int(limit))]
            template = str(reference.get("template", "") or "")
            select = str(reference.get("select", "") or "").strip()
            if template:
                values = []
                for record in records:
                    if not isinstance(record, dict):
                        raise TypeError("Artifact templates require object records.")
                    try:
                        values.append(template.format_map(record))
                    except KeyError as exc:
                        raise KeyError(f"Unknown artifact template field: {exc.args[0]}") from exc
            elif select:
                values = [_json_copy(_field(record, select)) for record in records]
            else:
                values = _json_copy(records)
            if "join" not in reference:
                return values
            prefix, suffix = str(reference.get("prefix", "") or ""), str(reference.get("suffix", "") or "")
            def reference_text(value: Any) -> str:
                return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)

            return str(reference.get("join", "")).join(f"{prefix}{reference_text(value)}{suffix}" for value in values)

    def resolve_references(self, value: Any, require_finalized: bool = False) -> Any:
        if isinstance(value, dict):
            if "$artifact" in value:
                return self.resolve(value, require_finalized=require_finalized)
            return {key: self.resolve_references(child, require_finalized=require_finalized) for key, child in value.items()}
        if isinstance(value, list):
            return [self.resolve_references(child, require_finalized=require_finalized) for child in value]
        return value

    def runtime_context(self, limit: int = 12) -> str:
        with self._lock:
            artifacts = sorted(self._artifacts.values(), key=lambda artifact: float(artifact["updated"]), reverse=True)[:max(1, int(limit))]
            if not artifacts:
                return ""
            lines = ["Active Deepy artifacts are authoritative external working state. Query wangp_artifact rather than reconstructing their contents from memory:"]
            for artifact in artifacts:
                if artifact["kind"] == "record_set":
                    expected = artifact["expected_items"]
                    count = f"{len(artifact['items'])}/{expected}" if expected is not None else str(len(artifact["items"]))
                    details = f"items={count}, revision={artifact['revision']}, finalized={str(bool(artifact['finalized'])).lower()}"
                    workflow = self._workflows.get(artifact["artifact_id"])
                    if workflow is not None:
                        details += f", workflow={workflow['phase']}, cursor={workflow['cursor']}, ledger={workflow['ledger_id']}"
                else:
                    sections = ", ".join(list(artifact["data"])[:8]) or "none"
                    details = f"sections={sections}, revision={artifact['revision']}, mutable=true"
                lines.append(f"- {artifact['artifact_id']}: {artifact['title']} ({artifact['kind']}; {details})")
            return "\n".join(lines)


_MANAGED_WORKFLOW_SCHEMA = {"type": "object", "description": "Optional sequential workflow linked to a ledger. Status and mutations then return a binding next_required_action.", "properties": {"ledger_id": {"type": "string"}, "plan_path": {"type": "string", "description": "Dotted ledger path containing an object keyed by item index or an ordered array."}, "context_fields": {"type": "array", "items": {"type": "string"}, "description": "Bounded ledger fields returned automatically by prepare."}, "index_field": {"type": "string", "default": "index"}, "end_state_field": {"type": "string", "default": "end_state"}, "start_index": {"type": "integer", "default": 1}}, "required": ["ledger_id", "plan_path", "context_fields"], "additionalProperties": False}


ARTIFACT_ACTIONS = {
    "create": {"description": "Create a record collection or mutable project ledger. Set workflow only when sequential context/commit ordering must be enforced.", "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["record_set", "ledger"], "default": "record_set"}, "title": {"type": "string"}, "schema": {"type": "object"}, "expected_items": {"type": "integer"}, "initial_items": {"type": "array"}, "initial_data": {"type": "object"}, "workflow": _MANAGED_WORKFLOW_SCHEMA}}},
    "list": {"description": "List active artifact identifiers and compact status metadata.", "parameters": {"type": "object", "properties": {"offset": {"type": "integer", "default": 0}, "limit": {"type": "integer", "default": 32}}}},
    "prepare": {"description": "Load a bounded authoritative packet for the next managed item. Full prior records and unselected ledger sections stay external and are available through returned exact queries.", "parameters": {"type": "object", "properties": {"artifact_id": {"type": "string"}}, "required": ["artifact_id"]}},
    "commit_item": {"description": "Append one prepared managed item and advance the cursor. ledger_values is optional: save the completed item even when some ledger updates are not ready, then apply those later with update_ledger.", "parameters": {"type": "object", "properties": {"artifact_id": {"type": "string"}, "item": {}, "ledger_values": {"type": "object"}, "ledger_remove": {"type": "array", "items": {"type": "string"}}, "expected_revision": {"type": "integer"}, "operation_id": {"type": "string"}}, "required": ["artifact_id", "item", "operation_id"]}},
    "append": {"description": "Append a bounded batch to a record collection.", "parameters": {"type": "object", "properties": {"artifact_id": {"type": "string"}, "items": {"type": "array"}, "expected_revision": {"type": "integer"}, "operation_id": {"type": "string"}}, "required": ["artifact_id", "items"]}},
    "query": {"description": "Read a bounded record page or selected ledger sections.", "parameters": {"type": "object", "properties": {"artifact_id": {"type": "string"}, "filters": {"type": "array", "items": {"type": "object"}}, "fields": {"type": "array", "items": {"type": "string"}}, "sort_by": {"type": "string"}, "descending": {"type": "boolean"}, "offset": {"type": "integer", "default": 0}, "limit": {"type": "integer", "default": 50}}, "required": ["artifact_id"]}},
    "status": {"description": "Return compact artifact progress and revision metadata.", "parameters": {"type": "object", "properties": {"artifact_id": {"type": "string"}}, "required": ["artifact_id"]}},
    "update_records": {"description": "Merge values into collection records matching generic filters.", "parameters": {"type": "object", "properties": {"artifact_id": {"type": "string"}, "filters": {"type": "array", "items": {"type": "object"}}, "values": {"type": "object"}, "expected_revision": {"type": "integer"}, "operation_id": {"type": "string"}}, "required": ["artifact_id", "values"]}},
    "update_ledger": {"description": "Deep-merge durable project facts into a ledger, optionally removing dotted paths. Existing object/array fields keep their shape unless explicitly removed first.", "parameters": {"type": "object", "properties": {"artifact_id": {"type": "string"}, "values": {"type": "object"}, "remove": {"type": "array", "items": {"type": "string"}}, "expected_revision": {"type": "integer"}, "operation_id": {"type": "string"}}, "required": ["artifact_id"]}},
    "finalize": {"description": "Validate and freeze a record collection before another tool consumes it.", "parameters": {"type": "object", "properties": {"artifact_id": {"type": "string"}, "expected_revision": {"type": "integer"}, "constraints": {"type": "object", "description": "Optional expected_items, required_fields, unique_by, contiguous_field/start, sum_field/expected_sum/tolerance."}}, "required": ["artifact_id"]}},
    "delete": {"description": "Delete one session artifact.", "parameters": {"type": "object", "properties": {"artifact_id": {"type": "string"}}, "required": ["artifact_id"]}},
}


def normalize_artifact_invocation(action: str | None, arguments: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    """Normalize the two common unambiguous forms emitted for the multiplexed artifact tool."""

    action_name = str(action or "").strip()
    if action_name:
        return action_name, arguments
    if arguments is None:
        return "", None
    payload = dict(arguments)
    if not payload:
        return "list", {}
    nested_action = str(payload.get("action", "") or "").strip()
    if nested_action:
        unknown = sorted(set(payload) - {"action", "arguments"})
        if unknown:
            raise ValueError(f"Artifact action must be a top-level tool parameter; unexpected nested field: {unknown[0]}.")
        nested_arguments = payload.get("arguments")
        if not isinstance(nested_arguments, dict):
            raise TypeError("Nested artifact arguments must be an object.")
        return nested_action, dict(nested_arguments)
    raise ValueError("Artifact action is missing. Pass action and arguments as separate top-level wangp_artifact parameters; repeat action after reading its schema.")


def run_artifact_action(workspace: ArtifactWorkspace, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    action = str(action or "").strip()
    if action not in ARTIFACT_ACTIONS:
        raise ValueError(f"Unknown artifact action: {action}")
    arguments = dict(arguments or {})
    parameters = ARTIFACT_ACTIONS[action]["parameters"]
    unknown = sorted(set(arguments) - set(parameters.get("properties", {})))
    if unknown:
        allowed = ", ".join(parameters.get("properties", {})) or "none"
        raise ValueError(f"Unsupported {action} argument: {unknown[0]}. Allowed arguments: {allowed}.")
    required = parameters.get("required", [])
    for name in required:
        if name not in arguments or arguments[name] is None:
            raise ValueError(f"{name} is required.")
    if action == "create":
        return workspace.create(kind=arguments.get("kind", "record_set"), title=arguments.get("title", ""), schema=arguments.get("schema"), expected_items=arguments.get("expected_items"), initial_items=arguments.get("initial_items"), initial_data=arguments.get("initial_data"), workflow=arguments.get("workflow"))
    if action == "list":
        return workspace.list_artifacts(offset=arguments.get("offset", 0), limit=arguments.get("limit", 32))
    if action == "prepare":
        return workspace.prepare(arguments["artifact_id"])
    if action == "commit_item":
        return workspace.commit_item(arguments["artifact_id"], arguments["item"], ledger_values=arguments.get("ledger_values"), ledger_remove=arguments.get("ledger_remove"), expected_revision=arguments.get("expected_revision"), operation_id=arguments["operation_id"])
    if action == "append":
        return workspace.append(arguments["artifact_id"], arguments["items"], expected_revision=arguments.get("expected_revision"), operation_id=arguments.get("operation_id", ""))
    if action == "query":
        return workspace.query(arguments["artifact_id"], filters=arguments.get("filters"), fields=arguments.get("fields"), sort_by=arguments.get("sort_by", ""), descending=arguments.get("descending", False), offset=arguments.get("offset", 0), limit=arguments.get("limit", 50))
    if action == "status":
        return workspace.status(arguments["artifact_id"])
    if action == "update_records":
        return workspace.update_records(arguments["artifact_id"], arguments["values"], filters=arguments.get("filters"), expected_revision=arguments.get("expected_revision"), operation_id=arguments.get("operation_id", ""))
    if action == "update_ledger":
        return workspace.update_ledger(arguments["artifact_id"], arguments.get("values", {}), remove=arguments.get("remove"), expected_revision=arguments.get("expected_revision"), operation_id=arguments.get("operation_id", ""))
    if action == "finalize":
        return workspace.finalize(arguments["artifact_id"], expected_revision=arguments.get("expected_revision"), constraints=arguments.get("constraints"))
    return workspace.delete(arguments["artifact_id"])


__all__ = ["ARTIFACT_ACTIONS", "ARTIFACT_INLINE_ITEM_THRESHOLD", "ARTIFACT_INLINE_TOKEN_THRESHOLD", "ARTIFACT_LIMITS", "ArtifactWorkspace", "normalize_artifact_invocation", "run_artifact_action"]
