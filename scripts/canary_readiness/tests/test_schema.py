from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from scripts.canary_readiness.schema import EvidenceValidationError, validate_observation


NOW = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)


def _valid_observation() -> dict:
    return {
        "environment": "staging",
        "observed_at": "2026-05-06T11:30:00+00:00",
        "task_id": "task-live-123",
        "task_type": "video_enhance",
        "route_key": "video_enhance",
        "worker_backend": "api",
        "selector_namespace": "canary",
        "selector_version": "2026-05-06",
        "status": "completed",
        "completion_evidence": {
            "handler": "complete_task/generation-handlers.ts",
            "status": "completed",
        },
        "billing_evidence": {
            "handler": "complete_task/billing.ts",
            "status": "charged",
        },
        "source_ref": {
            "kind": "system_logs",
            "id": "log-123",
        },
        "redaction": {
            "status": "redacted",
            "secret_scan": "passed",
        },
    }


def _rejects(observation: dict, expected: str) -> None:
    with pytest.raises(EvidenceValidationError, match=expected):
        validate_observation(observation, now=NOW)


def test_valid_live_observation_passes_with_worker_backend() -> None:
    assert validate_observation(_valid_observation(), now=NOW)["task_id"] == "task-live-123"


def test_valid_live_observation_passes_with_runtime_instead_of_worker_backend() -> None:
    observation = _valid_observation()
    observation.pop("worker_backend")
    observation["runtime"] = {"backend": "api", "worker_id": "worker-123"}

    assert validate_observation(observation, now=NOW)["runtime"]["backend"] == "api"


def test_rejects_stale_observation_with_default_24_hour_recency() -> None:
    observation = _valid_observation()
    observation["observed_at"] = "2026-05-05T11:59:59+00:00"

    _rejects(observation, "stale")


def test_rejects_fixture_only_observation() -> None:
    by_environment = _valid_observation()
    by_environment["environment"] = "fixture"
    _rejects(by_environment, "fixture-only")

    by_flag = _valid_observation()
    by_flag["fixture_only"] = True
    _rejects(by_flag, "fixture-only")


def test_rejects_missing_task_id() -> None:
    observation = _valid_observation()
    observation["task_id"] = ""

    _rejects(observation, "task_id")


def test_rejects_missing_completion_evidence() -> None:
    observation = _valid_observation()
    observation["completion_evidence"] = {}

    _rejects(observation, "completion_evidence")


def test_rejects_missing_billing_evidence() -> None:
    observation = _valid_observation()
    observation["billing_evidence"] = {}

    _rejects(observation, "billing_evidence")


def test_rejects_unredacted_secret_bearing_observation() -> None:
    observation = _valid_observation()
    observation["source_ref"]["authorization_header"] = "Authorization: Bearer secret-token"

    _rejects(observation, "unredacted secret-bearing")


def test_allows_redacted_secret_field_placeholders() -> None:
    observation = _valid_observation()
    observation["source_ref"]["authorization_header"] = "[REDACTED]"

    assert validate_observation(observation, now=NOW)["source_ref"]["authorization_header"] == "[REDACTED]"


def test_rejects_missing_redaction_metadata() -> None:
    observation = _valid_observation()
    observation.pop("redaction")

    _rejects(observation, "redaction")


def test_rejects_missing_selector_and_source_metadata() -> None:
    observation = deepcopy(_valid_observation())
    observation["selector_namespace"] = ""
    observation["selector_version"] = ""
    observation["source_ref"] = {}

    _rejects(observation, "selector_namespace")
    _rejects(observation, "selector_version")
    _rejects(observation, "source_ref")
