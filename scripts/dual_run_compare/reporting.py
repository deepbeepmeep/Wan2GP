from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.dual_run_compare.compare import compare_route_observations
from scripts.dual_run_compare.oracles import (
    REQUIRED_FULL_CANARY_SECTIONS,
    load_registry_snapshot,
    validate_billing_idempotency_oracles,
    validate_product_effect_oracles,
    validate_shared_path_oracles,
)
from scripts.dual_run_compare.queue_contract import build_queue_contract_report
from scripts.dual_run_compare.runtime_metrics import build_runtime_metrics_report
from scripts.dual_run_compare.shadow import ARTIFACTS_DIR, create_shadow_envelope, shadow_isolation_report
from scripts.dual_run_compare.status import FALLBACK, GREEN, PENDING, RED, SECTION_RED, WGP_ONLY
from scripts.dual_run_compare.thresholds import DEFAULT_PATH, Thresholds


DUAL_RUN_DIR = DEFAULT_PATH.parent
REPORTS_DIR = DUAL_RUN_DIR / "reports"
KNOWN_ROUTE_CLASSIFICATIONS = frozenset({"active_api_owned", "worker_pool_fallback"})
STATUS_ORDER = {RED: 0, PENDING: 1, GREEN: 2, FALLBACK: 3, WGP_ONLY: 4}
REQUIRED_ORACLE_REPORT_KEYS = (
    "registry_coverage",
    "shared_path_oracles",
    "queue_contract",
    "runtime_metrics",
    "product_effect_oracles",
    "billing_idempotency_oracles",
)


def report_paths(report_id: str, reports_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / f"{report_id}.json", reports_dir / f"{report_id}.md"


def write_reports(report: Mapping[str, Any], report_id: str, reports_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    json_path, markdown_path = report_paths(report_id, reports_dir)
    json_path.write_text(stable_json(report), encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def stable_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def build_oracle_bundle(*, report_id: str, create_shadow_dirs: bool = False) -> dict[str, Any]:
    registry_snapshot = load_registry_snapshot()
    shared_path = validate_shared_path_oracles(registry_snapshot)
    queue_contract = build_queue_contract_report(registry_snapshot)
    runtime_metrics = build_runtime_metrics_report(registry_snapshot)
    product_effect = validate_product_effect_oracles(registry_snapshot)
    billing = validate_billing_idempotency_oracles(registry_snapshot)
    shadow = {
        route_key: shadow_isolation_report(
            create_shadow_envelope(
                report_id,
                route_key,
                artifacts_dir=ARTIFACTS_DIR,
                create=create_shadow_dirs,
            )
        )
        for route_key in registry_snapshot["routes"]
    }
    registry_coverage = build_registry_coverage_report(registry_snapshot)
    return {
        "registry_snapshot": registry_snapshot,
        "registry_coverage": registry_coverage,
        "shared_path_oracles": shared_path,
        "queue_contract": queue_contract,
        "runtime_metrics": runtime_metrics,
        "product_effect_oracles": product_effect,
        "billing_idempotency_oracles": billing,
        "shadow_isolation": shadow,
    }


def build_registry_coverage_report(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    unclassified = [
        route_key
        for route_key, route in snapshot["routes"].items()
        if route.get("route_classification") not in KNOWN_ROUTE_CLASSIFICATIONS
    ]
    active_api = [
        route_key
        for route_key, route in snapshot["routes"].items()
        if route.get("route_classification") == "active_api_owned"
    ]
    worker_pool = [
        route_key
        for route_key, route in snapshot["routes"].items()
        if route.get("route_classification") == "worker_pool_fallback"
    ]
    return {
        "status": RED if unclassified else GREEN,
        "schema_version": snapshot.get("schema_version"),
        "route_count": len(snapshot["routes"]),
        "active_api_owned_route_count": len(active_api),
        "worker_pool_fallback_route_count": len(worker_pool),
        "unclassified_active_api_routes": unclassified,
        "known_route_classifications": sorted(KNOWN_ROUTE_CLASSIFICATIONS),
    }


def enrich_routes(
    *,
    thresholds: Thresholds,
    selected_route_keys: Sequence[str],
    oracle_bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    registry_snapshot = oracle_bundle["registry_snapshot"]
    registry_routes = registry_snapshot["routes"]
    route_keys = sorted(set(selected_route_keys) | set(registry_routes))
    route_reports: list[dict[str, Any]] = []
    for route_key in route_keys:
        registry_route = registry_routes.get(route_key, {})
        landed = _is_landed_required(registry_route)
        comparison = (
            compare_route_observations(thresholds, route_key, None, landed=landed)
            if route_key in thresholds.routes
            else _comparison_stub(route_key)
        )
        route_sections = _route_sections(route_key, comparison, oracle_bundle)
        status = _aggregate_route_status(route_key, comparison, route_sections, oracle_bundle)
        route_reports.append(
            {
                "route_key": route_key,
                "task_type": _task_type(thresholds, route_key, registry_route),
                "cohort": thresholds.routes.get(route_key, {}).get("cohort"),
                "status": status,
                "report_status": status,
                "calibration_status": comparison["calibration_status"],
                "route_classification": registry_route.get("route_classification", "threshold_only"),
                "landed_status": registry_route.get("landed_status"),
                "report_status_policy": registry_route.get("report_status_policy"),
                "canary_depth": registry_route.get("canary_depth"),
                "comparison": comparison,
                "required_sections": _required_sections(route_key, comparison, registry_route),
                "sections": route_sections,
                "metric_results": comparison["metrics"],
                "raw_observation_refs": _raw_observation_refs(route_key, thresholds, registry_route),
                "shadow_isolation": oracle_bundle["shadow_isolation"].get(route_key),
            }
        )
    return route_reports


def attach_reporting_metadata(report: dict[str, Any], oracle_bundle: Mapping[str, Any]) -> dict[str, Any]:
    route_reports = report["routes"]
    report["registry_coverage"] = oracle_bundle["registry_coverage"]
    report["oracle_reports"] = {
        "shared_path_oracles": _without_routes(oracle_bundle["shared_path_oracles"]),
        "queue_contract": _without_routes(oracle_bundle["queue_contract"]),
        "runtime_metrics": _without_routes(oracle_bundle["runtime_metrics"]),
        "product_effect_oracles": _without_routes(oracle_bundle["product_effect_oracles"]),
        "billing_idempotency_oracles": _without_routes(oracle_bundle["billing_idempotency_oracles"]),
    }
    report["route_status_counts"] = _status_counts(route_reports)
    report["exit_policy"] = evaluate_exit_policy(report)
    return report


def evaluate_exit_policy(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[dict[str, Any]] = []
    for run in report.get("live_runs", []):
        if "returncode" in run and run.get("returncode") != 0:
            reasons.append({"kind": "live_run_failed", "returncode": run.get("returncode")})

    for route in report.get("routes", []):
        if _route_is_landed_required(route) and route.get("status") == RED:
            reasons.append({"kind": "landed_red_route", "route_key": route["route_key"]})
        comparison = route.get("comparison", {})
        if (
            route.get("status") == GREEN
            and comparison.get("missing_required_observations")
        ):
            reasons.append(
                {
                    "kind": "sparse_green_attempt",
                    "route_key": route["route_key"],
                    "missing_required_observations": comparison.get("missing_required_observations"),
                }
            )

    registry_coverage = report.get("registry_coverage", {})
    for route_key in registry_coverage.get("unclassified_active_api_routes", []):
        reasons.append({"kind": "unclassified_active_api_route", "route_key": route_key})

    for key in REQUIRED_ORACLE_REPORT_KEYS:
        oracle_report = registry_coverage if key == "registry_coverage" else report.get("oracle_reports", {}).get(key, {})
        if oracle_report.get("status") == RED:
            reasons.append({"kind": "required_oracle_failure", "oracle": key})

    return {
        "exit_code": 1 if reasons else 0,
        "nonzero_reasons": reasons,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Dual-Run Comparison Report: {report['report_id']}",
        "",
        f"- Threshold version: `{report['threshold_version']}`",
        f"- Mode: `{report['mode']}`",
        f"- Created at: `{report['created_at']}`",
        f"- Worker root: `{report['worker_root']}`",
        f"- Route count: {len(report['routes'])}",
        f"- Exit code: `{report.get('exit_policy', {}).get('exit_code', 0)}`",
        "",
        "## Live Commands",
    ]
    if report["live_runs"]:
        for run in report["live_runs"]:
            lines.append(f"- `{' '.join(run['command'])}`")
            if "returncode" in run:
                lines.append(f"  - Return code: `{run['returncode']}`")
            if "not_executed" in run:
                lines.append(f"  - Not executed: `{run['not_executed']}`")
    else:
        lines.append("- No live commands configured.")

    lines.extend(["", "## Registry And Oracles"])
    registry = report.get("registry_coverage", {})
    lines.append(
        f"- Registry coverage: `{registry.get('status')}` "
        f"({registry.get('active_api_owned_route_count', 0)} active API-owned routes)"
    )
    for key, oracle in report.get("oracle_reports", {}).items():
        lines.append(f"- {key}: `{oracle.get('status')}`")

    lines.extend(["", "## Routes"])
    for route in _red_first_routes(report["routes"]):
        lines.append(
            f"### `{route['route_key']}`"
        )
        lines.append(
            f"- Status: `{route['status']}`"
        )
        lines.append(f"- Calibration: `{route['calibration_status']}`")
        lines.append(f"- Classification: `{route['route_classification']}`")
        lines.append(f"- Required sections: {', '.join(f'`{section}`' for section in route['required_sections']) or '`none`'}")
        if route["comparison"].get("missing_required_observations"):
            missing = ", ".join(f"`{key}`" for key in route["comparison"]["missing_required_observations"])
            lines.append(f"- Missing required observations: {missing}")
        lines.append("- Sections:")
        for section in route["sections"]:
            lines.append(f"  - `{section['key']}`: `{section['status']}`")
    return "\n".join(lines) + "\n"


def _without_routes(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "routes"}


def _comparison_stub(route_key: str) -> dict[str, Any]:
    return {
        "route_key": route_key,
        "status": PENDING,
        "report_status": PENDING,
        "initial_report_status": PENDING,
        "calibration_status": "not_in_thresholds",
        "required_metric_keys": [],
        "observed_metric_keys": [],
        "missing_required_observations": [],
        "failed_required_observations": [],
        "sections": [],
        "metrics": [],
    }


def _route_sections(
    route_key: str,
    comparison: Mapping[str, Any],
    oracle_bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    sections.extend(_scoped_sections("comparison", comparison.get("sections", [])))
    for report_key in (
        "shared_path_oracles",
        "queue_contract",
        "runtime_metrics",
        "product_effect_oracles",
        "billing_idempotency_oracles",
    ):
        route_report = oracle_bundle[report_key].get("routes", {}).get(route_key)
        if route_report:
            sections.extend(_scoped_sections(report_key, route_report.get("sections", [])))
    shadow_report = oracle_bundle["shadow_isolation"].get(route_key)
    if shadow_report:
        sections.append(
            {
                "key": "shadow_isolation",
                "status": shadow_report["status"],
                "source": "shadow_isolation",
                "shadow_root": shadow_report["shadow_root"],
                "skipped_side_effects": shadow_report["skipped_side_effects"],
            }
        )
    return sections


def _scoped_sections(source: str, sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(section),
            "key": f"{source}.{section['key']}",
            "source": source,
        }
        for section in sections
    ]


def _aggregate_route_status(
    route_key: str,
    comparison: Mapping[str, Any],
    route_sections: Sequence[Mapping[str, Any]],
    oracle_bundle: Mapping[str, Any],
) -> str:
    registry_route = oracle_bundle["registry_snapshot"]["routes"].get(route_key, {})
    if registry_route.get("route_classification") == "worker_pool_fallback":
        return FALLBACK
    if registry_route.get("report_status_policy") == WGP_ONLY:
        return WGP_ONLY
    if comparison.get("status") == WGP_ONLY:
        return WGP_ONLY

    component_statuses = [comparison.get("status", PENDING)]
    for report_key in (
        "shared_path_oracles",
        "queue_contract",
        "runtime_metrics",
        "product_effect_oracles",
        "billing_idempotency_oracles",
    ):
        route_report = oracle_bundle[report_key].get("routes", {}).get(route_key)
        if route_report:
            component_statuses.append(route_report.get("status", PENDING))
    component_statuses.extend(section["status"] for section in route_sections)

    if RED in component_statuses or SECTION_RED in component_statuses:
        return RED
    if PENDING in component_statuses or "pending" in component_statuses or "missing_evidence" in component_statuses:
        return PENDING
    return GREEN


def _required_sections(
    route_key: str,
    comparison: Mapping[str, Any],
    registry_route: Mapping[str, Any],
) -> list[str]:
    if registry_route.get("canary_depth") == "full_canary":
        return sorted(REQUIRED_FULL_CANARY_SECTIONS)
    required = [f"comparison.{key}" for key in comparison.get("required_metric_keys", [])]
    if registry_route.get("route_classification") == "active_api_owned":
        required.extend(
            [
                "shared_path_oracles.billing",
                "shared_path_oracles.completion",
                "shared_path_oracles.queue_contract",
                "shared_path_oracles.shadow_side_effects",
                "queue_contract.queue_payload_shape",
                "product_effect_oracles.completion_contract",
                "billing_idempotency_oracles.lightweight_billing_policy",
                "billing_idempotency_oracles.spend_ledger_idempotency",
                "billing_idempotency_oracles.refund_path_discovery",
                "shadow_isolation",
            ]
        )
    return sorted(dict.fromkeys(required))


def _raw_observation_refs(
    route_key: str,
    thresholds: Thresholds,
    registry_route: Mapping[str, Any],
) -> list[dict[str, str]]:
    refs = [{"kind": "thresholds", "path": str(thresholds.path), "route_key": route_key}]
    if registry_route:
        refs.append({"kind": "registry_snapshot", "path": str(DEFAULT_PATH.parent / "fixtures" / "non_rayworker" / "registry_snapshot.json"), "route_key": route_key})
        if registry_route.get("canary_depth") == "full_canary":
            refs.append({"kind": "full_canary_fixture", "path": str(DEFAULT_PATH.parent / "fixtures" / "non_rayworker" / f"{route_key}.json"), "route_key": route_key})
    refs.extend(
        [
            {"kind": "shared_path_oracles", "route_key": route_key},
            {"kind": "queue_contract", "route_key": route_key},
            {"kind": "runtime_metrics", "route_key": route_key},
            {"kind": "product_effect_oracles", "route_key": route_key},
            {"kind": "billing_idempotency_oracles", "route_key": route_key},
            {"kind": "shadow_isolation", "route_key": route_key},
        ]
    )
    return refs


def _task_type(thresholds: Thresholds, route_key: str, registry_route: Mapping[str, Any]) -> str:
    if route_key in thresholds.routes:
        return str(thresholds.routes[route_key].get("task_type", route_key))
    return str(registry_route.get("task_type", route_key))


def _is_landed_required(route: Mapping[str, Any]) -> bool:
    return route.get("landed_status") == "landed_full_canary" or route.get("report_status_policy") == "red_or_green_required"


def _route_is_landed_required(route: Mapping[str, Any]) -> bool:
    return route.get("landed_status") == "landed_full_canary" or route.get("report_status_policy") == "red_or_green_required"


def _status_counts(routes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {GREEN: 0, RED: 0, PENDING: 0, FALLBACK: 0, WGP_ONLY: 0}
    for route in routes:
        status = route.get("status")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _red_first_routes(routes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(routes, key=lambda route: (STATUS_ORDER.get(route.get("status"), 99), route["route_key"]))
