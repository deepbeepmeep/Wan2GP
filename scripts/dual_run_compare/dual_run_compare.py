from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.dual_run_compare.reporting import (
    attach_reporting_metadata,
    build_oracle_bundle,
    enrich_routes,
    write_reports,
)
from scripts.dual_run_compare.thresholds import DEFAULT_PATH, Thresholds
from scripts.dual_run_compare.wgp_self_repeat import (
    LiveRunConfig,
    _run_command,
    _selected_route_keys,
    build_live_test_command,
)


DUAL_RUN_DIR = DEFAULT_PATH.parent
WORKER_ROOT = DUAL_RUN_DIR.parents[1]
VALID_VARIANTS = ("fresh", "update")


def _build_live_commands(args: argparse.Namespace) -> list[list[str]]:
    return [
        build_live_test_command(
            LiveRunConfig(
                variant=variant,
                wgp_profile=args.wgp_profile,
                timeout_image=args.timeout_image,
                timeout_travel_segment=args.timeout_travel_segment,
                timeout_travel_orchestrator=args.timeout_travel_orchestrator,
                pod_id=args.pod_id,
                spawn_takeover=args.spawn_takeover,
                no_terminate=args.no_terminate,
                ref=args.ref,
                anchor_image_a=args.anchor_image_a,
                anchor_image_b=args.anchor_image_b,
                dry_run=args.dry_run_live,
            ),
            python_executable=args.python_executable,
        )
        for variant in args.variant
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = Thresholds.load(args.thresholds, strict=True)
    route_keys = _selected_route_keys(thresholds, args.route_key)
    commands = _build_live_commands(args)
    oracle_bundle = build_oracle_bundle(report_id=args.report_id, create_shadow_dirs=False)

    if args.execute_live:
        mode = "live"
        live_runs = [_run_command(command, cwd=args.repo_root) for command in commands]
    else:
        mode = "dry_run"
        live_runs = [{"command": command, "not_executed": "dry_run"} for command in commands]

    route_reports = enrich_routes(
        thresholds=thresholds,
        selected_route_keys=route_keys,
        oracle_bundle=oracle_bundle,
    )
    report = {
        "report_id": args.report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold_version": thresholds.raw["version"],
        "threshold_path": str(Path(args.thresholds).resolve()),
        "mode": mode,
        "worker_root": str(args.repo_root.resolve()),
        "live_runs": live_runs,
        "routes": route_reports,
    }
    return attach_reporting_metadata(report, oracle_bundle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sprint 3 dual-run comparison reports.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Render fixture-backed reports without live side effects.")
    mode.add_argument("--execute-live", action="store_true", help="Execute paired live-test commands.")
    parser.add_argument("--thresholds", default=str(DEFAULT_PATH))
    parser.add_argument("--route-key", action="append", default=[])
    parser.add_argument("--variant", action="append", choices=VALID_VARIANTS, default=None)
    parser.add_argument("--wgp-profile", type=int, default=3)
    parser.add_argument("--timeout-image", type=int, default=3600)
    parser.add_argument("--timeout-travel-segment", type=int, default=1800)
    parser.add_argument("--timeout-travel-orchestrator", type=int, default=3600)
    parser.add_argument("--pod-id")
    parser.add_argument("--spawn-takeover", action="store_true")
    termination = parser.add_mutually_exclusive_group()
    termination.add_argument("--no-terminate", dest="no_terminate", action="store_true")
    termination.add_argument("--terminate", dest="no_terminate", action="store_false")
    parser.set_defaults(no_terminate=None)
    parser.add_argument("--ref")
    parser.add_argument("--anchor-image-a")
    parser.add_argument("--anchor-image-b")
    parser.add_argument("--dry-run-live", action="store_true")
    parser.add_argument("--report-id", default=None)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--repo-root", type=Path, default=WORKER_ROOT)
    return parser


def _finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.variant is None:
        args.variant = ["fresh", "fresh"]
    if len(args.variant) != 2:
        raise SystemExit("--variant must be supplied exactly twice for paired dual-run comparison")
    if args.report_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.report_id = f"dual-run-{timestamp}"
    if args.dry_run and args.execute_live:
        raise SystemExit("--dry-run and --execute-live are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = _finalize_args(parser.parse_args(argv))
    report = build_report(args)
    json_path, markdown_path = write_reports(report, args.report_id)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return int(report["exit_policy"]["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
