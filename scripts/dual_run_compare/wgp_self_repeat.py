from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.dual_run_compare.thresholds import DEFAULT_PATH, Thresholds


DUAL_RUN_DIR = DEFAULT_PATH.parent
REPORTS_DIR = DUAL_RUN_DIR / "reports"
LIVE_TEST_SCRIPT = Path("scripts/live_test/main.py")
VALID_VARIANTS = ("fresh", "update")
INVALID_LIVE_TEST_ARGS = frozenset({"--backend", "--semantic-validation", "--timeout"})


@dataclass(frozen=True)
class LiveRunConfig:
    variant: str
    wgp_profile: int
    timeout_image: int
    timeout_travel_segment: int
    timeout_travel_orchestrator: int
    pod_id: str | None = None
    spawn_takeover: bool = False
    no_terminate: bool | None = None
    ref: str | None = None
    anchor_image_a: str | None = None
    anchor_image_b: str | None = None
    dry_run: bool = False


def build_live_test_command(config: LiveRunConfig, *, python_executable: str = "python") -> list[str]:
    if config.variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(VALID_VARIANTS)}")
    if config.variant == "fresh" and (config.pod_id or config.spawn_takeover):
        raise ValueError("fresh variant cannot include update takeover options")
    if config.variant == "update" and bool(config.pod_id) == bool(config.spawn_takeover):
        raise ValueError("update variant requires exactly one of pod_id or spawn_takeover")

    command = [
        python_executable,
        str(LIVE_TEST_SCRIPT),
        "--variant",
        config.variant,
        "--wgp-profile",
        str(config.wgp_profile),
        "--timeout-image",
        str(config.timeout_image),
        "--timeout-travel-segment",
        str(config.timeout_travel_segment),
        "--timeout-travel-orchestrator",
        str(config.timeout_travel_orchestrator),
    ]
    if config.variant == "fresh" and config.ref:
        command.extend(["--ref", config.ref])
    if config.variant == "update":
        if config.pod_id:
            command.extend(["--pod-id", config.pod_id])
        if config.spawn_takeover:
            command.append("--spawn-takeover")
    if config.no_terminate is True:
        command.append("--no-terminate")
    elif config.no_terminate is False:
        command.append("--terminate")
    if config.anchor_image_a:
        command.extend(["--anchor-image-a", config.anchor_image_a])
    if config.anchor_image_b:
        command.extend(["--anchor-image-b", config.anchor_image_b])
    if config.dry_run:
        command.append("--dry-run")
    _assert_valid_live_test_command(command)
    return command


def _assert_valid_live_test_command(command: Sequence[str]) -> None:
    invalid = sorted(INVALID_LIVE_TEST_ARGS.intersection(command))
    if invalid:
        raise ValueError(f"invalid live-test arguments present: {', '.join(invalid)}")
    for variant in VALID_VARIANTS:
        for index, token in enumerate(command):
            if token == variant and (index == 0 or command[index - 1] != "--variant"):
                raise ValueError(f"variant {variant!r} must be passed via --variant, not positionally")


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


def compare_route_observations(
    thresholds: Thresholds,
    route_key: str,
    observations: Mapping[str, Any] | None,
) -> dict[str, Any]:
    route_threshold = thresholds.for_route(route_key)
    if not observations:
        return {
            "route_key": route_key,
            "status": "not_evaluated_no_metric_observations",
            "calibration_status": route_threshold.calibration_status,
            "metric_keys": list(route_threshold.metrics),
        }

    results = []
    for metric_key, payload in observations.items():
        if metric_key not in route_threshold.metrics:
            raise ValueError(f"unknown observed metric for route {route_key}: {metric_key}")
        if isinstance(payload, Mapping):
            observed = payload.get("observed")
            expected = payload.get("expected")
        else:
            observed = payload
            expected = None
        results.append(compare_metric(route_threshold.metrics[metric_key], observed, expected))
    return {
        "route_key": route_key,
        "status": "green" if all(result["passed"] for result in results) else "red",
        "calibration_status": route_threshold.calibration_status,
        "metrics": results,
    }


def _selected_route_keys(thresholds: Thresholds, requested: Sequence[str]) -> list[str]:
    if not requested:
        return list(thresholds.routes)
    unknown = sorted(set(requested) - set(thresholds.routes))
    if unknown:
        raise KeyError(f"unknown route keys: {', '.join(unknown)}")
    return list(requested)


def _run_command(command: Sequence[str], *, cwd: Path) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": list(command),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _report_paths(report_id: str) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return (
        REPORTS_DIR / f"{report_id}.json",
        REPORTS_DIR / f"{report_id}.md",
    )


def _write_reports(report: Mapping[str, Any], report_id: str) -> tuple[Path, Path]:
    json_path, markdown_path = _report_paths(report_id)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown_report(report))
    return json_path, markdown_path


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# WGP Self Repeatability Report: {report['report_id']}",
        "",
        f"- Threshold version: `{report['threshold_version']}`",
        f"- Mode: `{report['mode']}`",
        f"- Created at: `{report['created_at']}`",
        f"- Route count: {len(report['routes'])}",
        "",
        "## Commands",
    ]
    for run in report["live_runs"]:
        lines.append(f"- `{' '.join(run['command'])}`")
        if "returncode" in run:
            lines.append(f"  - Return code: `{run['returncode']}`")
    if not report["live_runs"]:
        lines.append("- No live commands executed.")

    lines.extend(["", "## Routes"])
    for route in report["routes"]:
        lines.append(
            f"- `{route['route_key']}`: {route['comparison']['status']} "
            f"(calibration: `{route['comparison']['calibration_status']}`)"
        )
    if report.get("deferral"):
        lines.extend(
            [
                "",
                "## Deferral",
                f"- Blocker: {report['deferral']['blocker']}",
                f"- Next action: {report['deferral']['next_action']}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = Thresholds.load(args.thresholds, strict=True)
    route_keys = _selected_route_keys(thresholds, args.route_key)
    commands = [
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

    live_runs: list[dict[str, Any]]
    mode: str
    deferral: dict[str, str] | None = None
    if args.record_deferral:
        mode = "deferral"
        live_runs = [{"command": command, "not_executed": "record_deferral"} for command in commands]
        deferral = {"blocker": args.blocker, "next_action": args.next_action}
    else:
        mode = "live"
        live_runs = [_run_command(command, cwd=args.repo_root) for command in commands]

    route_reports = [
        {
            "route_key": route_key,
            "task_type": thresholds.routes[route_key]["task_type"],
            "cohort": thresholds.routes[route_key]["cohort"],
            "comparison": compare_route_observations(thresholds, route_key, None),
        }
        for route_key in route_keys
    ]
    return {
        "report_id": args.report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold_version": thresholds.raw["version"],
        "threshold_path": str(Path(args.thresholds).resolve()),
        "mode": mode,
        "deferral": deferral,
        "live_runs": live_runs,
        "routes": route_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or defer Sprint 0B WGP self-repeatability checks.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute-live", action="store_true", help="Execute paired live-test commands.")
    mode.add_argument("--record-deferral", action="store_true", help="Record a route-keyed deferral report.")
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
    parser.add_argument("--blocker", default="GPU execution unavailable for Sprint 0B calibration.")
    parser.add_argument("--next-action", default="Run paired live WGP self-repeatability once GPU execution is available.")
    parser.add_argument("--report-id", default=None)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def _finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.variant is None:
        args.variant = ["fresh", "fresh"]
    if len(args.variant) != 2:
        raise SystemExit("--variant must be supplied exactly twice for paired self-repeatability")
    if args.report_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.report_id = f"wgp-self-repeat-{timestamp}"
    if args.record_deferral and args.execute_live:
        raise SystemExit("--record-deferral and --execute-live are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = _finalize_args(parser.parse_args(argv))
    report = build_report(args)
    json_path, markdown_path = _write_reports(report, args.report_id)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    if args.execute_live:
        return 0 if all(run.get("returncode") == 0 for run in report["live_runs"]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
