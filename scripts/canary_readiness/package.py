from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.dual_run_compare.reporting import REPORTS_DIR, stable_json


CANARY_READINESS_PREFIX = "canary-readiness-"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports"
REQUIRED_NON_RAYWORKER_ROUTES = (
    "video_enhance",
    "image-upscale",
    "animate_character",
    "flux_klein_edit",
)
HARD_GATE_SECTIONS = (
    "prerequisite_evidence",
    "non_rayworker_smoke",
    "soak",
    "dashboards",
    "alerts",
    "rollback_exercise",
    "go_no_go",
)


@dataclass(frozen=True)
class SourceReport:
    path: Path
    report: Mapping[str, Any]


def build_package(
    *,
    package_id: str,
    source_report_dir: Path = REPORTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    created_at: str | None = None,
) -> dict[str, Any]:
    source_reports = load_source_reports(source_report_dir)
    package = {
        "package_id": package_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "source_report_dir": str(source_report_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_reports": [_source_report_summary(source) for source in source_reports],
        "sections": {},
    }
    sections = {
        "prerequisite_evidence": _prerequisite_section(source_reports),
        "non_rayworker_smoke": _non_rayworker_section(source_reports),
        "soak": _placeholder_section(
            "soak",
            "Structured soak evidence is not yet attached to the readiness package.",
        ),
        "dashboards": _placeholder_section(
            "dashboards",
            "Dashboard export evidence is not yet attached to the readiness package.",
        ),
        "alerts": _placeholder_section(
            "alerts",
            "Section 11 alert evidence is not yet attached to the readiness package.",
        ),
        "rollback_exercise": _placeholder_section(
            "rollback_exercise",
            "Rollback exercise evidence is not yet attached to the readiness package.",
        ),
        "go_no_go": _placeholder_section(
            "go_no_go",
            "Go/no-go decision evidence is not yet attached to the readiness package.",
        ),
    }
    package["sections"] = sections
    package["exit_policy"] = evaluate_exit_policy(sections)
    return package


def load_source_reports(source_report_dir: Path) -> list[SourceReport]:
    reports: list[SourceReport] = []
    for path in sorted(source_report_dir.glob("*.json")):
        if _is_generated_readiness_output(path):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict):
            reports.append(SourceReport(path=path, report=report))
    return reports


def write_package(package: Mapping[str, Any], *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    package_id = str(package["package_id"])
    json_path = output_dir / f"{package_id}.json"
    markdown_path = output_dir / f"{package_id}.md"
    json_path.write_text(stable_json(package), encoding="utf-8")
    markdown_path.write_text(markdown_package(package), encoding="utf-8")
    return json_path, markdown_path


def evaluate_exit_policy(sections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reasons: list[dict[str, Any]] = []
    for key in HARD_GATE_SECTIONS:
        section = sections.get(key, {})
        if section.get("status") != "green":
            reasons.append(
                {
                    "kind": "hard_gate_not_green",
                    "section": key,
                    "status": section.get("status", "missing"),
                }
            )
    return {"exit_code": 1 if reasons else 0, "nonzero_reasons": reasons}


def markdown_package(package: Mapping[str, Any]) -> str:
    lines = [
        f"# Canary Readiness Package: {package['package_id']}",
        "",
        f"- Created at: `{package['created_at']}`",
        f"- Source report dir: `{package['source_report_dir']}`",
        f"- Output dir: `{package['output_dir']}`",
        f"- Source reports: {len(package.get('source_reports', []))}",
        f"- Exit code: `{package.get('exit_policy', {}).get('exit_code', 0)}`",
        "",
        "## Source Reports",
    ]
    source_reports = package.get("source_reports", [])
    if source_reports:
        for source in source_reports:
            lines.append(
                f"- `{source.get('report_id')}` from `{source.get('path')}` "
                f"({source.get('route_count', 0)} routes)"
            )
    else:
        lines.append("- No dual-run source reports found.")

    lines.extend(["", "## Sections"])
    for key in HARD_GATE_SECTIONS:
        section = package.get("sections", {}).get(key, {})
        lines.append(f"### {section.get('title', key.replace('_', ' ').title())}")
        lines.append(f"- Status: `{section.get('status', 'missing')}`")
        for reason in section.get("reasons", []):
            lines.append(f"- {reason}")
        evidence_refs = section.get("evidence_refs", [])
        if evidence_refs:
            lines.append("- Evidence refs:")
            for ref in evidence_refs:
                lines.append(f"  - `{ref}`")

    lines.extend(["", "## Go/No-Go"])
    if package.get("exit_policy", {}).get("exit_code"):
        lines.append("- Decision: `no_go`")
    else:
        lines.append("- Decision: `go`")
    return "\n".join(lines) + "\n"


def _is_generated_readiness_output(path: Path) -> bool:
    return path.name.startswith(CANARY_READINESS_PREFIX)


def _source_report_summary(source: SourceReport) -> dict[str, Any]:
    report = source.report
    return {
        "path": str(source.path.resolve()),
        "report_id": report.get("report_id", source.path.stem),
        "created_at": report.get("created_at"),
        "mode": report.get("mode"),
        "route_count": len(report.get("routes", [])),
        "exit_code": report.get("exit_policy", {}).get("exit_code"),
    }


def _prerequisite_section(source_reports: Sequence[SourceReport]) -> dict[str, Any]:
    if not source_reports:
        return _section(
            "prerequisite_evidence",
            "Prerequisite Evidence",
            "red",
            reasons=["No dual-run JSON source reports were found."],
        )
    red_reports = [
        source.path.name
        for source in source_reports
        if source.report.get("exit_policy", {}).get("exit_code") not in (0, None)
    ]
    if red_reports:
        return _section(
            "prerequisite_evidence",
            "Prerequisite Evidence",
            "red",
            reasons=["One or more dual-run reports still have non-zero exit policy."],
            evidence_refs=red_reports,
        )
    return _section(
        "prerequisite_evidence",
        "Prerequisite Evidence",
        "green",
        evidence_refs=[source.path.name for source in source_reports],
    )


def _non_rayworker_section(source_reports: Sequence[SourceReport]) -> dict[str, Any]:
    route_statuses = _latest_route_statuses(source_reports)
    missing = [route for route in REQUIRED_NON_RAYWORKER_ROUTES if route not in route_statuses]
    not_green = [
        f"{route}:{route_statuses[route]}"
        for route in REQUIRED_NON_RAYWORKER_ROUTES
        if route in route_statuses and route_statuses[route] != "green"
    ]
    reasons = []
    if missing:
        reasons.append("Missing active non-RayWorker route evidence: " + ", ".join(missing))
    if not_green:
        reasons.append("Active non-RayWorker route evidence is not green: " + ", ".join(not_green))
    return _section(
        "non_rayworker_smoke",
        "Active Non-RayWorker Smoke",
        "red" if reasons else "green",
        reasons=reasons,
        evidence_refs=[
            f"{route}:{route_statuses[route]}"
            for route in REQUIRED_NON_RAYWORKER_ROUTES
            if route in route_statuses
        ],
    )


def _latest_route_statuses(source_reports: Iterable[SourceReport]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for source in source_reports:
        for route in source.report.get("routes", []):
            route_key = route.get("route_key")
            status = route.get("status") or route.get("report_status")
            if isinstance(route_key, str) and isinstance(status, str):
                statuses[route_key] = status
    return statuses


def _placeholder_section(key: str, reason: str) -> dict[str, Any]:
    return _section(
        key,
        key.replace("_", " ").title(),
        "red",
        reasons=[reason],
    )


def _section(
    key: str,
    title: str,
    status: str,
    *,
    reasons: Sequence[str] | None = None,
    evidence_refs: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "status": status,
        "reasons": list(reasons or []),
        "evidence_refs": list(evidence_refs or []),
    }

