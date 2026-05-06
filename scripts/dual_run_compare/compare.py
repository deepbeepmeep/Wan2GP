from __future__ import annotations

from typing import Any, Mapping, Sequence

from scripts.dual_run_compare.status import (
    GREEN,
    PENDING,
    RED,
    SECTION_GREEN,
    SECTION_MISSING_EVIDENCE,
    SECTION_RED,
    WGP_ONLY,
    map_calibration_status_to_report_status,
)
from scripts.dual_run_compare.thresholds import Thresholds


def compare_metric(metric: Mapping[str, Any], observed: Any, expected: Any | None = None) -> dict[str, Any]:
    comparator = metric["comparator"]
    threshold = metric["threshold"]
    passed: bool
    detail: str

    if comparator == "max":
        passed = float(observed) <= float(threshold)
        detail = f"{observed} <= {threshold}"
    elif comparator == "min":
        passed = float(observed) >= float(threshold)
        detail = f"{observed} >= {threshold}"
    elif comparator == "exact":
        target = expected if expected is not None else threshold
        passed = observed == target
        detail = f"{observed!r} == {target!r}"
    elif comparator == "tolerance":
        if expected is None:
            raise ValueError("tolerance comparator requires an expected value")
        absolute_ms = threshold["absolute_ms"]
        delta = abs(float(observed) - float(expected))
        passed = delta <= float(absolute_ms)
        detail = f"delta {delta} <= {absolute_ms}"
    else:
        raise ValueError(f"unsupported comparator: {comparator}")

    return {
        "metric": metric["metric"],
        "comparator": comparator,
        "observed": observed,
        "expected": expected,
        "threshold": threshold,
        "passed": passed,
        "detail": detail,
    }


def _metric_payload_values(payload: Any) -> tuple[Any, Any | None]:
    if isinstance(payload, Mapping):
        return payload.get("observed"), payload.get("expected")
    return payload, None


def _required_keys(
    route_metric_keys: Sequence[str],
    required_metric_keys: Sequence[str] | None,
) -> tuple[str, ...]:
    if required_metric_keys is None:
        return tuple(route_metric_keys)
    return tuple(dict.fromkeys(required_metric_keys))


def compare_route_observations(
    thresholds: Thresholds,
    route_key: str,
    observations: Mapping[str, Any] | None,
    *,
    required_metric_keys: Sequence[str] | None = None,
    landed: bool = False,
) -> dict[str, Any]:
    route_threshold = thresholds.for_route(route_key)
    initial_report_status = map_calibration_status_to_report_status(route_threshold.calibration_status)
    observed = observations or {}
    required = _required_keys(tuple(route_threshold.metrics), required_metric_keys)

    unknown_required = [metric_key for metric_key in required if metric_key not in route_threshold.metrics]
    if unknown_required:
        raise ValueError(f"unknown required metric for route {route_key}: {', '.join(unknown_required)}")

    unknown_observed = [metric_key for metric_key in observed if metric_key not in route_threshold.metrics]
    if unknown_observed:
        raise ValueError(f"unknown observed metric for route {route_key}: {', '.join(unknown_observed)}")

    sections: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    missing_required = [metric_key for metric_key in required if metric_key not in observed]
    for metric_key in missing_required:
        sections.append(
            {
                "key": metric_key,
                "status": SECTION_MISSING_EVIDENCE,
                "required": True,
                "reason": "required observation missing",
            }
        )

    for metric_key, payload in observed.items():
        observed_value, expected = _metric_payload_values(payload)
        result = compare_metric(route_threshold.metrics[metric_key], observed_value, expected)
        results.append({"key": metric_key, **result})
        sections.append(
            {
                "key": metric_key,
                "status": SECTION_GREEN if result["passed"] else SECTION_RED,
                "required": metric_key in required,
                "result": result,
            }
        )

    failed_required = [result["key"] for result in results if result["key"] in required and not result["passed"]]
    failed_observed = [result["key"] for result in results if not result["passed"]]
    if failed_observed:
        report_status = RED
    elif missing_required:
        report_status = RED if landed else initial_report_status
        if report_status == WGP_ONLY:
            report_status = WGP_ONLY
        elif report_status != RED:
            report_status = PENDING
    elif initial_report_status == WGP_ONLY:
        report_status = WGP_ONLY
    else:
        report_status = GREEN

    return {
        "route_key": route_key,
        "status": report_status,
        "report_status": report_status,
        "initial_report_status": initial_report_status,
        "calibration_status": route_threshold.calibration_status,
        "required_metric_keys": list(required),
        "observed_metric_keys": list(observed),
        "missing_required_observations": missing_required,
        "failed_required_observations": failed_required,
        "sections": sections,
        "metrics": results,
    }
