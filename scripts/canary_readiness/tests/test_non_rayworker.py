from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.canary_readiness.non_rayworker import (
    BILLING_HANDLER,
    COMPLETION_HANDLER,
    REQUIRED_ROUTES,
    build_non_rayworker_gate,
    validate_non_rayworker_observations,
)
from scripts.canary_readiness.schema import EvidenceValidationError
from scripts.dual_run_compare.reporting import DUAL_RUN_DIR


NOW = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
FIXTURE_DIR = DUAL_RUN_DIR / "fixtures" / "non_rayworker"


def _observation(route_key: str, *, observed_at: str = "2026-05-06T11:30:00+00:00") -> dict:
    return {
        "environment": "staging",
        "observed_at": observed_at,
        "task_id": f"task-{route_key}",
        "task_type": route_key,
        "route_key": route_key,
        "runtime": {"backend": "api_orchestrator"},
        "selector_namespace": "canary",
        "selector_version": "2026-05-06",
        "status": "completed",
        "completion_evidence": {"handler": COMPLETION_HANDLER, "status": "completed"},
        "billing_evidence": {"handler": BILLING_HANDLER, "status": "charged"},
        "source_ref": {"kind": "system_logs", "id": f"log-{route_key}"},
        "redaction": {"status": "redacted", "secret_scan": "passed"},
    }


def _observations() -> list[dict]:
    return [_observation(route_key) for route_key in REQUIRED_ROUTES]


def test_non_rayworker_gate_requires_recent_live_observations_for_all_four_routes() -> None:
    gate = build_non_rayworker_gate(_observations(), now=NOW)

    assert gate["status"] == "green"
    assert set(gate["required_routes"]) == set(REQUIRED_ROUTES)
    assert {route["route_key"] for route in gate["routes"]} == set(REQUIRED_ROUTES)
    assert all(route["task_id"] for route in gate["routes"])


def test_non_rayworker_gate_rejects_missing_route_observation() -> None:
    observations = [observation for observation in _observations() if observation["route_key"] != "flux_klein_edit"]

    gate = build_non_rayworker_gate(observations, now=NOW)

    assert gate["status"] == "red"
    route = next(route for route in gate["routes"] if route["route_key"] == "flux_klein_edit")
    assert "missing recent live/staging observation" in route["errors"]


def test_non_rayworker_gate_rejects_fixture_only_and_stale_observations() -> None:
    fixture_only = _observations()
    fixture_only[0]["fixture_only"] = True
    assert build_non_rayworker_gate(fixture_only, now=NOW)["status"] == "red"

    stale = _observations()
    stale[0]["observed_at"] = "2026-05-05T11:59:59+00:00"
    with pytest.raises(EvidenceValidationError, match="stale"):
        validate_non_rayworker_observations(stale, now=NOW)


def test_non_rayworker_gate_requires_shared_completion_and_billing_evidence() -> None:
    observations = _observations()
    observations[0]["completion_evidence"] = {"handler": "other-handler.ts", "status": "completed"}
    observations[1]["billing_evidence"] = {"handler": "other-billing.ts", "status": "charged"}

    gate = build_non_rayworker_gate(observations, now=NOW)

    assert gate["status"] == "red"
    errors = {route["route_key"]: route["errors"] for route in gate["routes"]}
    assert any("generation-handlers" in error for error in errors["video_enhance"])
    assert any("billing.ts" in error for error in errors["image-upscale"])


def test_non_rayworker_gate_prevents_fixture_policy_downgrade(tmp_path: Path) -> None:
    for path in FIXTURE_DIR.glob("*.json"):
        if path.name == "registry_snapshot.json":
            continue
        data = json.loads(path.read_text())
        if data["route_key"] == "video_enhance":
            data["canary_depth"] = "lightweight_shared_path"
            data["report_status_policy"] = "pending_until_shared_oracle_evidence"
        (tmp_path / path.name).write_text(json.dumps(data), encoding="utf-8")

    gate = build_non_rayworker_gate(_observations(), fixture_dir=tmp_path, now=NOW)

    assert gate["status"] == "red"
    video = next(route for route in gate["routes"] if route["route_key"] == "video_enhance")
    assert any("full_canary" in error for error in video["errors"])
    assert any("red_or_green_required" in error for error in video["errors"])

