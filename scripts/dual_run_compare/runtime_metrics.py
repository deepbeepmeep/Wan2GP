from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from scripts.canary_readiness.schema import DEFAULT_MAX_AGE, EvidenceValidationError, validate_observation

from scripts.dual_run_compare.oracles import (
    BILLING_PATH,
    COMPLETION_HANDLER,
    IDEMPOTENCY_POLICY,
    REFUND_STATUS,
    active_api_routes,
    load_registry_snapshot,
)
from scripts.dual_run_compare.status import (
    FALLBACK,
    GREEN,
    PENDING,
    RED,
    SECTION_GREEN,
    SECTION_MISSING_EVIDENCE,
    SECTION_PENDING,
    SECTION_WGP_ONLY,
    WGP_ONLY,
)


RUNTIME_METRIC_KEYS = ("latency", "vram", "oom", "error_class")
RUNTIME_METADATA_KEYS = (
    "route_key",
    "selector_namespace",
    "selector_version",
    "backend",
    "pool",
    "worker_id",
    "task_id",
    "source_ref",
)


def _section(key: str, status: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"key": key, "status": status}
    if reason:
        payload["reason"] = reason
    payload.update(extra)
    return payload


def _missing_status(route: Mapping[str, Any]) -> str:
    if route.get("route_classification") == "worker_pool_fallback":
        return FALLBACK
    if route.get("report_status_policy") == WGP_ONLY:
        return WGP_ONLY
    if route.get("landed_status") == "landed_full_canary" or route.get("report_status_policy") == "red_or_green_required":
        return RED
    return PENDING


def _runtime_owner(route: Mapping[str, Any]) -> str:
    return str(route.get("runtime") or route.get("route_classification") or "unknown")


def build_runtime_metrics_report(
    snapshot: Mapping[str, Any] | None = None,
    *,
    runtime_observations: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> dict[str, Any]:
    data = snapshot if snapshot is not None else load_registry_snapshot()
    observations_by_route = normalize_runtime_observations(
        runtime_observations,
        now=now,
        max_age=max_age,
    )
    routes: dict[str, dict[str, Any]] = {}

    for route_key, route in data["routes"].items():
        route_observations = observations_by_route.get(route_key, {})
        validation_errors = list(route_observations.get("validation_errors", []))
        missing_status = _missing_status(route)
        sections: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        for metric_key in RUNTIME_METRIC_KEYS:
            if validation_errors:
                metrics[metric_key] = None
                sections.append(
                    _section(
                        metric_key,
                        SECTION_MISSING_EVIDENCE,
                        "runtime observation failed live evidence validation",
                        required=True,
                        validation_errors=validation_errors,
                    )
                )
            elif metric_key in route_observations:
                metrics[metric_key] = route_observations[metric_key]
                sections.append(
                    _section(
                        metric_key,
                        SECTION_GREEN,
                        required=True,
                        observed=route_observations[metric_key],
                    )
                )
            elif missing_status == RED:
                metrics[metric_key] = None
                sections.append(
                    _section(
                        metric_key,
                        SECTION_MISSING_EVIDENCE,
                        "required runtime evidence missing for landed route",
                        required=True,
                    )
                )
            elif missing_status == FALLBACK:
                metrics[metric_key] = None
                sections.append(
                    _section(
                        metric_key,
                        "not_applicable",
                        "worker-pool fallback route has no API runtime evidence requirement",
                        required=False,
                    )
                )
            elif missing_status == WGP_ONLY:
                metrics[metric_key] = None
                sections.append(
                    _section(
                        metric_key,
                        SECTION_WGP_ONLY,
                        "Sprint 2 evidence keeps this route WGP-only; API runtime evidence is not a landed-route requirement",
                        required=False,
                    )
                )
            else:
                metrics[metric_key] = None
                sections.append(
                    _section(
                        metric_key,
                        SECTION_PENDING,
                        "runtime evidence pending for active route",
                        required=True,
                    )
                )

        observed_all = all(section["status"] == SECTION_GREEN for section in sections)
        route_status = GREEN if observed_all and not validation_errors else missing_status
        if validation_errors and missing_status != WGP_ONLY:
            route_status = RED
        routes[route_key] = {
            "route_key": route_key,
            "task_type": route.get("task_type", route_key),
            "status": route_status,
            "route_classification": route.get("route_classification"),
            "runtime_owner": _runtime_owner(route),
            "handler": route.get("executor_handler"),
            "completion_handler": COMPLETION_HANDLER,
            "billing_path": BILLING_PATH,
            "idempotency_policy": IDEMPOTENCY_POLICY,
            "refund_status": REFUND_STATUS,
            "latency": metrics["latency"],
            "vram": metrics["vram"],
            "oom": metrics["oom"],
            "error_class": metrics["error_class"],
            "runtime_observation": {
                key: route_observations.get(key)
                for key in RUNTIME_METADATA_KEYS
                if key in route_observations
            },
            "runtime_observation_validation_errors": validation_errors,
            "sections": sections,
        }

    landed_red_routes = [
        route_key
        for route_key, route in routes.items()
        if (
            data["routes"][route_key].get("landed_status") == "landed_full_canary"
            or data["routes"][route_key].get("report_status_policy") == "red_or_green_required"
        )
        and route["status"] == RED
    ]
    return {
        "status": RED if landed_red_routes else PENDING,
        "runtime_metric_keys": list(RUNTIME_METRIC_KEYS),
        "runtime_metadata_keys": list(RUNTIME_METADATA_KEYS),
        "active_api_owned_route_count": len(active_api_routes(data)),
        "landed_red_routes": landed_red_routes,
        "routes": routes,
    }


def normalize_runtime_observations(
    runtime_observations: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> dict[str, dict[str, Any]]:
    if runtime_observations is None:
        return {}
    if isinstance(runtime_observations, Mapping):
        return {
            route_key: _normalize_legacy_runtime_mapping(route_key, observation)
            for route_key, observation in runtime_observations.items()
        }

    normalized: dict[str, dict[str, Any]] = {}
    for observation in runtime_observations:
        route_key = str(observation.get("route_key") or "")
        if not route_key:
            continue
        try:
            validated = validate_observation(observation, now=now, max_age=max_age)
        except EvidenceValidationError as exc:
            normalized[route_key] = {
                "route_key": route_key,
                "validation_errors": [str(exc)],
            }
            continue
        normalized[route_key] = _normalize_live_runtime_observation(validated)
    return normalized


def _normalize_legacy_runtime_mapping(route_key: str, observation: Mapping[str, Any]) -> dict[str, Any]:
    if "observed_at" in observation:
        enriched = dict(observation)
        enriched.setdefault("route_key", route_key)
        try:
            validated = validate_observation(enriched)
        except EvidenceValidationError as exc:
            return {"route_key": route_key, "validation_errors": [str(exc)]}
        return _normalize_live_runtime_observation(validated)

    normalized = {
        key: observation[key]
        for key in RUNTIME_METRIC_KEYS
        if key in observation
    }
    normalized["route_key"] = route_key
    normalized["validation_errors"] = ["standardized recent live runtime observation is required"]
    for key in RUNTIME_METADATA_KEYS:
        if key in observation:
            normalized[key] = observation[key]
    return normalized


def _normalize_live_runtime_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    runtime = observation.get("runtime") if isinstance(observation.get("runtime"), Mapping) else {}
    runtime_metrics = (
        observation.get("runtime_metrics")
        if isinstance(observation.get("runtime_metrics"), Mapping)
        else runtime.get("metrics")
        if isinstance(runtime.get("metrics"), Mapping)
        else {}
    )
    normalized = {
        "route_key": observation["route_key"],
        "selector_namespace": observation["selector_namespace"],
        "selector_version": observation["selector_version"],
        "backend": observation.get("worker_backend") or runtime.get("backend"),
        "pool": runtime.get("pool") or observation.get("pool"),
        "worker_id": runtime.get("worker_id") or observation.get("worker_id"),
        "task_id": observation["task_id"],
        "source_ref": observation["source_ref"],
        "completion_evidence": observation["completion_evidence"],
        "billing_evidence": observation["billing_evidence"],
    }
    for key in RUNTIME_METRIC_KEYS:
        if key in runtime_metrics:
            normalized[key] = runtime_metrics[key]
        elif key in runtime:
            normalized[key] = runtime[key]
        elif key in observation:
            normalized[key] = observation[key]
    return normalized
