"""No-GPU validation path for the live-test harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.live_test import config
from scripts.live_test.launch_command import build_direct_worker_command, build_run_worker_command
from scripts.live_test.matrix import build_matrix, render_case_payload
from scripts.live_test.preflight import assert_user_queue_clean, get_or_create_live_test_project
from scripts.live_test.safety_gate import assert_safe_to_take_over
from scripts.live_test.token_resolver import resolve_token_to_user_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the live worker harness without touching RunPod.")
    parser.add_argument("--pod-id", help="Optional pod ID to run the PAT-aware takeover safety gate against.")
    parser.add_argument("--skip-db", action="store_true", help="Skip token resolution and Supabase preflight checks.")
    parser.add_argument(
        "--require-db",
        action="store_true",
        help="Fail instead of degrading to static checks when Supabase preflight cannot run.",
    )
    parser.add_argument("--wgp-profile", type=int, default=3)
    parser.add_argument("--idle-release-minutes", type=int, default=0)
    parser.add_argument("--worker-id", default="smoke-worker")
    parser.add_argument("--anchor-image-a", default=config.ANCHOR_IMAGE_A_URL)
    parser.add_argument("--anchor-image-b", default=config.ANCHOR_IMAGE_B_URL)
    parser.add_argument("--timeout-image", type=int, default=config.TIMEOUT_IMAGE_SEC)
    parser.add_argument(
        "--timeout-travel-segment",
        type=int,
        default=config.TIMEOUT_INDIVIDUAL_TRAVEL_SEGMENT_SEC,
    )
    parser.add_argument(
        "--timeout-travel-orchestrator",
        type=int,
        default=config.TIMEOUT_TRAVEL_ORCHESTRATOR_SEC,
    )
    return parser


def _print_case_summary(cases) -> None:
    print("Validated matrix cases:")
    for case in cases:
        print(f"- {case.name} ({case.task_type}, timeout={case.timeout_sec}s)")


def _static_validate_cases(cases) -> None:
    for index, case in enumerate(cases, start=1):
        payload = render_case_payload(
            case,
            project_id="smoke-project",
            unique_suffix=f"smoke-{index}",
        )
        json.dumps(payload)
        params = payload.get("params", {})
        if params.get("live_test") is not True:
            raise RuntimeError(f"{case.name} did not set params.live_test=True")
        if payload.get("status") != "Queued":
            raise RuntimeError(f"{case.name} did not render with status='Queued'")


def _run_db_checks(pod_id: str | None) -> None:
    token = config.require_env("REIGH_LIVE_TEST_TOKEN")
    db = config.DatabaseClient()
    user_id = resolve_token_to_user_id(db, token)
    assert_user_queue_clean(db, user_id)
    project_id = get_or_create_live_test_project(db, user_id)
    print(f"Resolved REIGH_LIVE_TEST_TOKEN to user_id={user_id}")
    print(f"Using live-test project_id={project_id}")
    if pod_id:
        assert_safe_to_take_over(db, pod_id, user_id, allow_fresh_heartbeat=False)
        print(f"Safety gate passed for pod_id={pod_id}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    token = config.require_env("REIGH_LIVE_TEST_TOKEN")
    supabase_url = config.get_env("SUPABASE_URL", "https://example.supabase.co")

    cases = build_matrix(
        anchor_image_a=args.anchor_image_a,
        anchor_image_b=args.anchor_image_b,
        timeout_image_sec=args.timeout_image,
        timeout_travel_segment_sec=args.timeout_travel_segment,
        timeout_travel_orchestrator_sec=args.timeout_travel_orchestrator,
    )
    _static_validate_cases(cases)
    _print_case_summary(cases)

    print("")
    print("Run-worker launch command:")
    print(
        build_run_worker_command(
            "/workspace/Reigh-Worker-LiveTest",
            reigh_token=token,
            supabase_url=supabase_url,
            worker_id=args.worker_id,
            wgp_profile=args.wgp_profile,
            idle_release_minutes=args.idle_release_minutes,
        )
    )
    print("")
    print("Direct-worker restore command:")
    print(
        build_direct_worker_command(
            "/workspace/Reigh-Worker",
            cli_args=["python", "worker.py", "--task-id", "smoke-task", "--gpu-id", "0"],
        )
    )

    if args.skip_db:
        return 0

    if not (config.ENV.supabase_url and config.ENV.supabase_service_role_key):
        if args.require_db or args.pod_id:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for DB-backed smoke checks")
        print("")
        print("Skipping DB-backed smoke checks because Supabase credentials are not set.")
        return 0

    try:
        print("")
        _run_db_checks(args.pod_id)
    except Exception as exc:
        if args.require_db or args.pod_id:
            raise
        print("")
        print(f"Skipping DB-backed smoke checks after failure: {exc}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Smoke check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
