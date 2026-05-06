from __future__ import annotations

from typing import Any, Mapping

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
    runtime_observations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    data = snapshot if snapshot is not None else load_registry_snapshot()
    observations_by_route = runtime_observations or {}
    routes: dict[str, dict[str, Any]] = {}

    for route_key, route in data["routes"].items():
        route_observations = observations_by_route.get(route_key, {})
        missing_status = _missing_status(route)
        sections: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        for metric_key in RUNTIME_METRIC_KEYS:
            if metric_key in route_observations:
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
        route_status = GREEN if observed_all else missing_status
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
        "active_api_owned_route_count": len(active_api_routes(data)),
        "landed_red_routes": landed_red_routes,
        "routes": routes,
    }
