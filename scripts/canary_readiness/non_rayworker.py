from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.canary_readiness.schema import DEFAULT_MAX_AGE, EvidenceValidationError, validate_observation
from scripts.dual_run_compare.reporting import DUAL_RUN_DIR


NON_RAYWORKER_FIXTURE_DIR = DUAL_RUN_DIR / "fixtures" / "non_rayworker"
REQUIRED_ROUTES = (
    "video_enhance",
    "image-upscale",
    "animate_character",
    "flux_klein_edit",
)
COMPLETION_HANDLER = "complete_task/generation-handlers.ts"
BILLING_HANDLER = "complete_task/billing.ts"


def build_non_rayworker_gate(
    observations: Sequence[Mapping[str, Any]],
    *,
    fixture_dir: Path = NON_RAYWORKER_FIXTURE_DIR,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> dict[str, Any]:
    fixtures = _load_fixtures(fixture_dir)
    route_reports = [
        _route_report(
            route_key,
            observations=observations,
            fixture=fixtures.get(route_key),
            now=now,
            max_age=max_age,
        )
        for route_key in REQUIRED_ROUTES
    ]
    status = "green" if all(route["status"] == "green" for route in route_reports) else "red"
    return {
        "key": "non_rayworker_smoke",
        "title": "Active Non-RayWorker Smoke",
        "status": status,
        "required_routes": list(REQUIRED_ROUTES),
        "routes": route_reports,
    }


def validate_non_rayworker_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    fixture_dir: Path = NON_RAYWORKER_FIXTURE_DIR,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> list[dict[str, Any]]:
    gate = build_non_rayworker_gate(observations, fixture_dir=fixture_dir, now=now, max_age=max_age)
    if gate["status"] != "green":
        reasons = []
        for route in gate["routes"]:
            if route["errors"]:
                reasons.append(f"{route['route_key']}: " + ", ".join(route["errors"]))
        raise EvidenceValidationError("; ".join(reasons))
    return list(gate["routes"])


def _route_report(
    route_key: str,
    *,
    observations: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any] | None,
    now: datetime | None,
    max_age: timedelta,
) -> dict[str, Any]:
    errors: list[str] = []
    if not fixture:
        errors.append("missing policy fixture")
    else:
        if fixture.get("canary_depth") != "full_canary":
            errors.append("fixture canary_depth must remain full_canary")
        if fixture.get("report_status_policy") != "red_or_green_required":
            errors.append("fixture report_status_policy must remain red_or_green_required")
        if fixture.get("completion_handler") != COMPLETION_HANDLER:
            errors.append("fixture completion_handler must use shared completion handler")
        if fixture.get("billing_path") != BILLING_HANDLER:
            errors.append("fixture billing_path must use shared billing path")

    route_observations = [observation for observation in observations if observation.get("route_key") == route_key]
    if not route_observations:
        errors.append("missing recent live/staging observation")
        selected: Mapping[str, Any] | None = None
    else:
        selected = route_observations[-1]
        try:
            validate_observation(selected, now=now, max_age=max_age)
        except EvidenceValidationError as exc:
            errors.append(str(exc))
        if selected.get("status") not in {"completed", "green", "succeeded", "pass"}:
            errors.append("observation status must be completed/green/succeeded/pass")
        if _evidence_handler(selected.get("completion_evidence")) != COMPLETION_HANDLER:
            errors.append("completion evidence must reference complete_task/generation-handlers.ts")
        if _evidence_handler(selected.get("billing_evidence")) != BILLING_HANDLER:
            errors.append("billing evidence must reference complete_task/billing.ts")

    return {
        "route_key": route_key,
        "status": "green" if not errors else "red",
        "task_id": selected.get("task_id") if selected else None,
        "observed_at": selected.get("observed_at") if selected else None,
        "errors": errors,
        "fixture_policy": {
            "canary_depth": fixture.get("canary_depth") if fixture else None,
            "report_status_policy": fixture.get("report_status_policy") if fixture else None,
        },
    }


def _load_fixtures(fixture_dir: Path) -> dict[str, Mapping[str, Any]]:
    fixtures: dict[str, Mapping[str, Any]] = {}
    for route_key in REQUIRED_ROUTES:
        path = fixture_dir / f"{route_key}.json"
        if path.is_file():
            fixtures[route_key] = json.loads(path.read_text(encoding="utf-8"))
    return fixtures


def _evidence_handler(evidence: Any) -> str | None:
    if not isinstance(evidence, Mapping):
        return None
    handler = evidence.get("handler") or evidence.get("path")
    return handler if isinstance(handler, str) else None

