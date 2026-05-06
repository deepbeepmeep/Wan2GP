from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.canary_readiness.soak import REQUIRED_SOAK_SCENARIOS, build_soak_gate, validate_soak_scenarios


NOW = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)


def _scenario(name: str, *, status: str = "pass") -> dict:
    return {
        "scenario": name,
        "status": status,
        "observed_at": "2026-05-06T11:30:00+00:00",
        "route_labels": {"route_key": "travel_segment", "selector_namespace": "canary"},
        "pool_labels": {"pool": "vibecomfy-canary", "backend": "rayworker"},
        "evidence_refs": [{"kind": "system_logs", "id": f"log-{name}"}],
    }


def _complete_soak() -> list[dict]:
    return [_scenario(name) for name in REQUIRED_SOAK_SCENARIOS]


def test_soak_gate_requires_all_six_sprint10_scenarios_as_structured_pass_evidence() -> None:
    gate = build_soak_gate(_complete_soak(), now=NOW)

    assert gate["status"] == "green"
    assert set(gate["required_scenarios"]) == set(REQUIRED_SOAK_SCENARIOS)
    assert len(gate["scenarios"]) == 6
    for scenario in gate["scenarios"]:
        assert scenario["status"] == "pass"
        assert scenario["observed_at"]
        assert scenario["route_labels"]["route_key"]
        assert scenario["pool_labels"]["pool"]
        assert scenario["evidence_refs"][0]["kind"] == "system_logs"


def test_soak_gate_rejects_missing_required_scenario() -> None:
    scenarios = [scenario for scenario in _complete_soak() if scenario["scenario"] != "disk_near_full"]

    gate = build_soak_gate(scenarios, now=NOW)

    assert gate["status"] == "red"
    assert gate["missing_scenarios"] == ["disk_near_full"]


def test_soak_gate_rejects_free_form_evidence_and_missing_labels() -> None:
    scenario = _scenario("mixed_pools")
    scenario["evidence_refs"] = "checked logs manually"
    scenario["route_labels"] = {}
    scenario["pool_labels"] = {}

    gate = build_soak_gate([scenario], now=NOW)

    assert gate["status"] == "red"
    assert "evidence_refs" in ", ".join(gate["scenarios"][0]["errors"])
    assert "route_labels" in ", ".join(gate["scenarios"][0]["errors"])
    assert "pool_labels" in ", ".join(gate["scenarios"][0]["errors"])


def test_soak_gate_rejects_fail_status_and_stale_timestamp() -> None:
    scenario = _scenario("worker_kill_restart", status="fail")
    scenario["observed_at"] = "2026-05-05T11:59:59+00:00"

    with pytest.raises(ValueError, match="scenario status is fail"):
        validate_soak_scenarios([scenario], now=NOW)

