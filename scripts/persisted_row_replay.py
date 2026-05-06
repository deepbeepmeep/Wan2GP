from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.task_handlers.tasks.template_routing import (  # noqa: E402
    normalize_route_snapshot_fields,
    resolve_task_route,
    route_snapshot_fields,
    route_support_report_fields,
)


DEFAULT_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "persisted_rows"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "persisted-row-replay.json"
SNAPSHOT_KEYS = (
    "selector_namespace",
    "route_key",
    "selected_backend",
    "selector_version",
    "support_state",
    "selected_profile",
    "selected_template_id",
    "route_run_id",
    "worker_contract_version",
    "route_selection_snapshot",
)
NESTED_SNAPSHOT_KEYS = (
    "selector_namespace",
    "route_key",
    "selected_backend",
    "selector_version",
    "support_state",
    "template_id",
    "selected_profile",
    "route_run_id",
    "worker_contract_version",
)


class ReplayError(ValueError):
    pass


def load_persisted_row_fixtures(fixtures: Path) -> list[dict[str, Any]]:
    if fixtures.is_file():
        files = [fixtures]
    else:
        files = sorted(fixtures.glob("*.json"))
    if not files:
        raise ReplayError(f"no persisted-row fixture JSON files found under {fixtures}")

    rows: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_rows = payload.get("rows")
        if not isinstance(raw_rows, list):
            raise ReplayError(f"{path} must contain a top-level rows array")
        for index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                raise ReplayError(f"{path} row {index} is not an object")
            rows.append({**row, "_fixture_path": _display_path(path)})
    return rows


def replay_persisted_rows(
    fixtures: Path = DEFAULT_FIXTURES,
    *,
    report_path: Path | None = DEFAULT_REPORT,
    live: bool = False,
    allow_live_enqueue: bool = False,
) -> dict[str, Any]:
    if live and not allow_live_enqueue:
        raise ReplayError("live replay requires --allow-live-enqueue; dry-run is the default")
    if live:
        raise ReplayError("live replay is intentionally not implemented by this dry-run tool")

    rows = [_replay_row(row) for row in load_persisted_row_fixtures(fixtures)]
    report = {
        "schema_version": "persisted-row-replay/v1",
        "dry_run": True,
        "live_requested": live,
        "fixtures": str(fixtures),
        "summary": {
            "total": len(rows),
            "passed": sum(1 for row in rows if row["replay_result"] == "pass"),
            "failed": sum(1 for row in rows if row["replay_result"] == "fail"),
        },
        "rows": rows,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _replay_row(row: Mapping[str, Any]) -> dict[str, Any]:
    params = _mapping(row.get("params"), "params")
    task_type = _required_str(row, "task_type")
    fixture_id = _required_str(row, "fixture_id")
    backend = str(row.get("backend") or row.get("expected_backend") or "wgp")
    expected_route_key = str(row.get("expected_route_key") or "")
    expected_backend = str(row.get("expected_backend") or backend)
    selector_namespace = str(row.get("selector_namespace") or "production")
    selector_version = row.get("selector_version")
    route_run_id = row.get("route_run_id")

    current_snapshot = route_snapshot_fields(
        task_type=task_type,
        params=params,
        backend=backend,
        selector_namespace=selector_namespace,
        selector_version=selector_version,
        run_id=route_run_id,
    )
    normalized_snapshot = normalize_route_snapshot_fields(
        _mapping(row.get("route_contract"), "route_contract", allow_empty=True),
        task_type=task_type,
        params=params,
        backend=backend,
    )
    resolved = resolve_task_route(
        task_id=fixture_id,
        task_type=task_type,
        params=params,
        backend=backend,
    )
    support_fields = route_support_report_fields(current_snapshot["route_key"])

    errors = _snapshot_errors(
        normalized_snapshot=normalized_snapshot,
        current_snapshot=current_snapshot,
        expected_route_key=expected_route_key,
        expected_backend=expected_backend,
    )
    guidance = _guidance_fields(params)
    result = "pass" if not errors else "fail"
    reason = "; ".join(errors) or support_fields["blocking_reason"] or resolved.fail_closed_reason

    return {
        "fixture_id": fixture_id,
        "fixture_path": row.get("_fixture_path"),
        "row_id": row.get("row_id"),
        "task_type": task_type,
        "route_key": current_snapshot["route_key"],
        "backend": current_snapshot["selected_backend"],
        "support_state": current_snapshot["support_state"],
        "guidance_kind": guidance["guidance_kind"],
        "guidance_mode": guidance["guidance_mode"],
        "continuity_case": _continuity_case(params),
        "replay_result": result,
        "media_contract_result": "not_applicable_dry_run",
        "template_id": current_snapshot["selected_template_id"],
        "wgp_only_or_blocked_reason": reason,
        "snapshot_shape_result": "pass" if _has_full_snapshot_shape(normalized_snapshot) else "fail",
        "normalized_snapshot": normalized_snapshot,
        "current_snapshot": current_snapshot,
    }


def _snapshot_errors(
    *,
    normalized_snapshot: Mapping[str, Any],
    current_snapshot: Mapping[str, Any],
    expected_route_key: str,
    expected_backend: str,
) -> list[str]:
    errors: list[str] = []
    if not _has_full_snapshot_shape(normalized_snapshot):
        errors.append("normalized snapshot is missing required full-contract fields")
    if expected_route_key and current_snapshot["route_key"] != expected_route_key:
        errors.append(
            f"current route_key {current_snapshot['route_key']!r} does not match expected {expected_route_key!r}"
        )
    if expected_backend and current_snapshot["selected_backend"] != expected_backend:
        errors.append(
            f"current backend {current_snapshot['selected_backend']!r} does not match expected {expected_backend!r}"
        )
    if normalized_snapshot["route_key"] != current_snapshot["route_key"]:
        errors.append(
            f"normalized route_key {normalized_snapshot['route_key']!r} does not match current "
            f"{current_snapshot['route_key']!r}"
        )
    if normalized_snapshot["selected_backend"] != current_snapshot["selected_backend"]:
        errors.append(
            f"normalized backend {normalized_snapshot['selected_backend']!r} does not match current "
            f"{current_snapshot['selected_backend']!r}"
        )
    return errors


def _has_full_snapshot_shape(snapshot: Mapping[str, Any]) -> bool:
    if any(key not in snapshot for key in SNAPSHOT_KEYS):
        return False
    nested = snapshot.get("route_selection_snapshot")
    if not isinstance(nested, Mapping):
        return False
    return all(key in nested for key in NESTED_SNAPSHOT_KEYS)


def _guidance_fields(params: Mapping[str, Any]) -> dict[str, str]:
    guidance = params.get("travel_guidance")
    if isinstance(guidance, Mapping):
        kind = str(guidance.get("kind") or "none")
        mode = str(guidance.get("mode") or ("uni3c" if kind == "uni3c" else "none"))
        return {"guidance_kind": kind, "guidance_mode": mode}
    return {
        "guidance_kind": str(params.get("guidance_kind") or "none"),
        "guidance_mode": str(params.get("guidance_mode") or "none"),
    }


def _continuity_case(params: Mapping[str, Any]) -> str:
    if params.get("continuity_case"):
        return str(params["continuity_case"])
    if params.get("video_source"):
        return "video_source"
    return "first_last"


def _mapping(value: Any, field: str, *, allow_empty: bool = False) -> dict[str, Any]:
    if value is None and allow_empty:
        return {}
    if not isinstance(value, Mapping):
        raise ReplayError(f"{field} must be an object")
    return dict(value)


def _required_str(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ReplayError(f"persisted-row fixture is missing required string field {field!r}")
    return value


def _display_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved_path)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run replay persisted route rows through current routing.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--live", action="store_true", help="Reserved for explicit live enqueue/GPU replay.")
    parser.add_argument("--allow-live-enqueue", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        report = replay_persisted_rows(
            args.fixtures,
            report_path=args.report,
            live=args.live,
            allow_live_enqueue=args.allow_live_enqueue,
        )
    except ReplayError as exc:
        print(f"persisted-row replay failed: {exc}", file=sys.stderr)
        return 2

    print(f"Persisted-row replay report written: {args.report}")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
