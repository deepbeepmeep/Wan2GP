from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.section3a_matrix import Section3ARow, load_fixture  # noqa: E402
from source.task_handlers.tasks.template_routing import (  # noqa: E402
    derive_route_key,
    resolve_task_route,
    route_snapshot_fields,
    route_support_report_fields,
)


DEFAULT_REPORT = REPO_ROOT / "outputs" / "section3a-smoke.md"
DEFAULT_JSON_REPORT = REPO_ROOT / "outputs" / "section3a-smoke.json"
REPORT_COLUMNS = (
    "row_id",
    "route_key",
    "backend",
    "support_state",
    "guidance_kind",
    "guidance_mode",
    "continuity_case",
    "replay_result",
    "media_contract_result",
    "template_id",
    "wgp_only_or_blocked_reason",
)


class SmokeError(ValueError):
    pass


def run_section3a_smoke(
    *,
    dry_run: bool = True,
    live: bool = False,
    allow_live_enqueue: bool = False,
    report_path: Path | None = DEFAULT_REPORT,
    json_report_path: Path | None = DEFAULT_JSON_REPORT,
) -> dict[str, Any]:
    if live and not allow_live_enqueue:
        raise SmokeError("live Section 3A smoke requires --allow-live-enqueue; dry-run is the default")
    if live:
        raise SmokeError("live Section 3A enqueue/GPU smoke is intentionally not implemented by this dry-run tool")
    if not dry_run:
        raise SmokeError("only dry-run Section 3A smoke is supported")

    rows = [_smoke_row(row) for row in load_fixture()]
    report = {
        "schema_version": "section3a-smoke/v1",
        "dry_run": True,
        "live_requested": live,
        "columns": list(REPORT_COLUMNS),
        "summary": {
            "total": len(rows),
            "passed": sum(1 for row in rows if row["replay_result"] == "pass"),
            "failed": sum(1 for row in rows if row["replay_result"] == "fail"),
            "blocked": sum(1 for row in rows if row["disposition"] == "BLOCKED"),
            "reasoned_rows": sum(1 for row in rows if row["wgp_only_or_blocked_reason"]),
        },
        "rows": rows,
    }
    if json_report_path is not None:
        json_report_path.parent.mkdir(parents=True, exist_ok=True)
        json_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _smoke_row(row: Section3ARow) -> dict[str, Any]:
    params = _params_for_row(row)
    route_key = derive_route_key(row.task_type, params)
    resolved = resolve_task_route(
        task_id=f"section3a-row-{row.row_id}",
        task_type=row.task_type,
        params=params,
        backend=row.expected_backend,
    )
    snapshot = route_snapshot_fields(
        task_type=row.task_type,
        params=params,
        backend=row.expected_backend,
        selector_namespace="section3a-smoke",
        selector_version="dry-run",
    )
    support_fields = route_support_report_fields(route_key)

    errors: list[str] = []
    if route_key != row.route_key_expectation:
        errors.append(f"route_key {route_key!r} does not match fixture {row.route_key_expectation!r}")
    if resolved.backend.value != row.expected_backend:
        errors.append(f"backend {resolved.backend.value!r} does not match fixture {row.expected_backend!r}")
    if resolved.support_state.value != row.support_state_expectation:
        errors.append(
            f"support_state {resolved.support_state.value!r} does not match fixture "
            f"{row.support_state_expectation!r}"
        )
    if snapshot["route_key"] != route_key:
        errors.append("route snapshot route_key does not match current route key")

    reason = row.blocking_reason or support_fields["blocking_reason"] or resolved.fail_closed_reason
    return {
        "row_id": row.row_id,
        "route_key": route_key,
        "backend": resolved.backend.value,
        "support_state": resolved.support_state.value,
        "guidance_kind": row.guidance_kind,
        "guidance_mode": row.guidance_mode,
        "continuity_case": row.continuity_case,
        "replay_result": "pass" if not errors else "fail",
        "media_contract_result": "not_applicable_dry_run",
        "template_id": resolved.template_id,
        "wgp_only_or_blocked_reason": reason,
        "disposition": row.disposition,
        "expected_backend": row.expected_backend,
        "expected_route_key": row.route_key_expectation,
        "expected_support_state": row.support_state_expectation,
        "route_selection_snapshot": snapshot,
        "errors": errors,
    }


def _params_for_row(row: Section3ARow) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model_name": row.worker_model_name,
        "continuity_case": row.continuity_case,
    }
    if row.profile != "default":
        params["profile"] = row.profile
    if row.guidance_kind not in {"none", ""}:
        guidance: dict[str, Any] = {"kind": row.guidance_kind}
        if row.guidance_mode not in {"none", ""}:
            guidance["mode"] = row.guidance_mode
        if row.guidance_kind in {"vace", "ltx_control", "uni3c"}:
            guidance["videos"] = [{"path": f"storage://section3a/row-{row.row_id}-guide.mp4"}]
        params["travel_guidance"] = guidance
    return params


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Section 3A Matrix Smoke",
        "",
        f"- Dry run: `{str(report['dry_run']).lower()}`",
        f"- Rows: `{report['summary']['total']}`",
        f"- Passed: `{report['summary']['passed']}`",
        f"- Failed: `{report['summary']['failed']}`",
        "",
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in REPORT_COLUMNS) + " |",
    ]
    for row in report["rows"]:
        lines.append("| " + " | ".join(_cell(row.get(column)) for column in REPORT_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run Section 3A route matrix smoke report.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="Reserved for explicit live enqueue/GPU smoke.")
    parser.add_argument("--allow-live-enqueue", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        report = run_section3a_smoke(
            dry_run=args.dry_run,
            live=args.live,
            allow_live_enqueue=args.allow_live_enqueue,
            report_path=args.report,
            json_report_path=args.json_report,
        )
    except SmokeError as exc:
        print(f"Section 3A smoke failed: {exc}", file=sys.stderr)
        return 2

    print(f"Section 3A smoke report written: {args.report}")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
