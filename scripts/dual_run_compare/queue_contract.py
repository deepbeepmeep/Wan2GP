from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from scripts.dual_run_compare.oracles import (
    BILLING_PATH,
    COMPLETION_HANDLER,
    IDEMPOTENCY_POLICY,
    NON_RAYWORKER_DIR,
    REFUND_STATUS,
    REQUIRED_SHARED_PATH_KEYS,
    active_api_routes,
    load_full_canary_fixture,
    load_registry_snapshot,
)
from scripts.dual_run_compare.status import (
    GREEN,
    RED,
    SECTION_GREEN,
    SECTION_MISSING_EVIDENCE,
    SECTION_RED,
)


WORKER_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = WORKER_ROOT.parent
ORCHESTRATOR_REGISTRY = WORKSPACE_ROOT / "reigh-worker-orchestrator" / "api_orchestrator" / "task_handlers.py"


def parse_task_handlers(path: Path = ORCHESTRATOR_REGISTRY) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    task_handlers: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TASK_HANDLERS" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Name):
                task_handlers[key.value] = value.id
    return task_handlers


def _section(key: str, status: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"key": key, "status": status}
    if reason:
        payload["reason"] = reason
    payload.update(extra)
    return payload


def _expected_handler(route: Mapping[str, Any]) -> str | None:
    executor_handler = route.get("executor_handler")
    if not isinstance(executor_handler, str) or "::" not in executor_handler:
        return None
    return executor_handler.rsplit("::", 1)[1]


def _queue_payload_shape(route_key: str, route: Mapping[str, Any], fixture_dir: Path) -> dict[str, Any]:
    if route.get("canary_depth") == "full_canary":
        fixture = load_full_canary_fixture(route_key, fixture_dir)
        return {
            "source": "full_canary_fixture",
            "keys": sorted(fixture["payload_shape"]),
            "shape": fixture["payload_shape"],
        }
    return {
        "source": "shared_path_policy",
        "keys": sorted(REQUIRED_SHARED_PATH_KEYS),
        "shape": dict(route.get("shared_path_policy", {})),
    }


def build_queue_contract_report(
    snapshot: Mapping[str, Any] | None = None,
    *,
    task_handlers_path: Path = ORCHESTRATOR_REGISTRY,
    fixture_dir: Path = NON_RAYWORKER_DIR,
) -> dict[str, Any]:
    data = snapshot if snapshot is not None else load_registry_snapshot()
    task_handlers = parse_task_handlers(task_handlers_path)
    routes: dict[str, dict[str, Any]] = {}

    for route_key, route in data["routes"].items():
        classification = route.get("route_classification")
        runtime = route.get("runtime")
        expected_handler = _expected_handler(route)
        actual_handler = task_handlers.get(route_key)
        sections: list[dict[str, Any]] = []

        if classification == "worker_pool_fallback":
            routes[route_key] = {
                "route_key": route_key,
                "task_type": route.get("task_type", route_key),
                "status": "fallback",
                "runtime": runtime,
                "runtime_owner": runtime,
                "handler": actual_handler,
                "executor_handler": route.get("executor_handler"),
                "completion_handler": None,
                "billing_path": None,
                "idempotency_policy": None,
                "refund_status": "not_applicable",
                "route_classification": classification,
                "sections": [
                    _section(
                        "worker_pool_fallback",
                        "not_applicable",
                        "worker-pool route is fallback-only for this harness",
                    )
                ],
            }
            continue

        sections.append(
            _section(
                "task_handler_registry",
                SECTION_GREEN if actual_handler == expected_handler else SECTION_RED,
                None if actual_handler == expected_handler else "TASK_HANDLERS mapping drifted from fixture snapshot",
                required=True,
                observed=actual_handler,
                expected=expected_handler,
            )
        )
        shared_policy = route.get("shared_path_policy")
        queue_policy = shared_policy.get("queue_contract") if isinstance(shared_policy, Mapping) else None
        sections.append(
            _section(
                "queue_contract_policy",
                SECTION_GREEN if queue_policy in {"required_full", "required_lightweight"} else SECTION_MISSING_EVIDENCE,
                None if queue_policy in {"required_full", "required_lightweight"} else "queue_contract shared-path policy missing",
                required=True,
                policy=queue_policy,
            )
        )
        payload_shape = _queue_payload_shape(route_key, route, fixture_dir)
        sections.append(
            _section(
                "queue_payload_shape",
                SECTION_GREEN if payload_shape["keys"] else SECTION_MISSING_EVIDENCE,
                None if payload_shape["keys"] else "queue payload shape evidence missing",
                required=True,
                **payload_shape,
            )
        )

        failed = any(section["status"] in {SECTION_RED, SECTION_MISSING_EVIDENCE} for section in sections)
        routes[route_key] = {
            "route_key": route_key,
            "task_type": route.get("task_type", route_key),
            "status": RED if failed else GREEN,
            "runtime": runtime,
            "runtime_owner": runtime,
            "handler": actual_handler,
            "executor_handler": route.get("executor_handler"),
            "completion_handler": COMPLETION_HANDLER,
            "billing_path": BILLING_PATH,
            "idempotency_policy": IDEMPOTENCY_POLICY,
            "refund_status": REFUND_STATUS,
            "route_classification": classification,
            "canary_depth": route.get("canary_depth"),
            "landed_status": route.get("landed_status"),
            "sections": sections,
        }

    failing_active = [
        route_key
        for route_key in active_api_routes(data)
        if routes[route_key]["status"] != GREEN
    ]
    return {
        "status": RED if failing_active else GREEN,
        "task_handler_count": len(task_handlers),
        "active_api_owned_route_count": len(active_api_routes(data)),
        "failing_active_routes": failing_active,
        "routes": routes,
    }
