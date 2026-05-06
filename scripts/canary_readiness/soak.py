from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence


REQUIRED_SOAK_SCENARIOS = (
    "mixed_pools",
    "concurrent_claims",
    "selector_flip_in_flight",
    "worker_kill_restart",
    "cold_warm_cache",
    "disk_near_full",
)
PASS_STATUS = "pass"
FAIL_STATUS = "fail"
VALID_STATUSES = frozenset({PASS_STATUS, FAIL_STATUS})
DEFAULT_MAX_AGE = timedelta(hours=24)


def build_soak_gate(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> dict[str, Any]:
    reference_time = now or datetime.now(timezone.utc)
    scenario_reports = [_scenario_report(scenario, now=reference_time, max_age=max_age) for scenario in scenarios]
    by_key = {
        report["scenario"]: report
        for report in scenario_reports
        if report.get("scenario") in REQUIRED_SOAK_SCENARIOS
    }
    missing = [scenario for scenario in REQUIRED_SOAK_SCENARIOS if scenario not in by_key]
    failing = [
        report["scenario"]
        for report in scenario_reports
        if report.get("status") != PASS_STATUS or report.get("errors")
    ]
    status = "green" if not missing and not failing else "red"
    return {
        "key": "soak",
        "title": "Soak",
        "status": status,
        "required_scenarios": list(REQUIRED_SOAK_SCENARIOS),
        "missing_scenarios": missing,
        "failing_scenarios": failing,
        "scenarios": scenario_reports,
    }


def validate_soak_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> list[dict[str, Any]]:
    gate = build_soak_gate(scenarios, now=now, max_age=max_age)
    if gate["status"] != "green":
        reasons = []
        if gate["missing_scenarios"]:
            reasons.append("missing scenarios: " + ", ".join(gate["missing_scenarios"]))
        for scenario in gate["scenarios"]:
            if scenario["errors"]:
                reasons.append(f"{scenario['scenario']}: " + ", ".join(scenario["errors"]))
        raise ValueError("; ".join(reasons))
    return list(gate["scenarios"])


def _scenario_report(
    scenario: Mapping[str, Any],
    *,
    now: datetime,
    max_age: timedelta,
) -> dict[str, Any]:
    key = scenario.get("scenario")
    errors: list[str] = []
    if key not in REQUIRED_SOAK_SCENARIOS:
        errors.append("unknown or missing scenario")

    status = scenario.get("status")
    if status not in VALID_STATUSES:
        errors.append("status must be pass or fail")
    elif status == FAIL_STATUS:
        errors.append("scenario status is fail")

    observed_at = _parse_timestamp(scenario.get("observed_at"))
    if observed_at is None:
        errors.append("observed_at must be an ISO-8601 timestamp")
    elif observed_at < now - max_age:
        errors.append("scenario evidence is stale")

    if not _non_empty_sequence(scenario.get("evidence_refs")):
        errors.append("evidence_refs must be a non-empty structured list")
    elif not all(isinstance(ref, Mapping) and ref for ref in scenario.get("evidence_refs", [])):
        errors.append("evidence_refs entries must be objects")

    if not _label_map(scenario.get("route_labels")):
        errors.append("route_labels must be a non-empty object")
    if not _label_map(scenario.get("pool_labels")):
        errors.append("pool_labels must be a non-empty object")

    return {
        "scenario": key,
        "status": status,
        "observed_at": scenario.get("observed_at"),
        "route_labels": dict(scenario.get("route_labels", {}))
        if isinstance(scenario.get("route_labels"), Mapping)
        else scenario.get("route_labels"),
        "pool_labels": dict(scenario.get("pool_labels", {}))
        if isinstance(scenario.get("pool_labels"), Mapping)
        else scenario.get("pool_labels"),
        "evidence_refs": list(scenario.get("evidence_refs", []))
        if isinstance(scenario.get("evidence_refs"), Sequence)
        and not isinstance(scenario.get("evidence_refs"), (str, bytes, bytearray))
        else scenario.get("evidence_refs"),
        "errors": errors,
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _non_empty_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and bool(value)


def _label_map(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)

