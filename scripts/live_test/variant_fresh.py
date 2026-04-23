"""Variant A driver: launch a fresh pod and run the live-test matrix."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.live_test import config
from scripts.live_test.heartbeat_waiter import wait_until_ready
from scripts.live_test.launch_command import build_run_worker_command
from scripts.live_test.logger import get_logger
from scripts.live_test.matrix import build_matrix, render_case_payload, run_matrix
from scripts.live_test.preflight import assert_user_queue_clean, get_or_create_live_test_project
from scripts.live_test.report import write_report
from scripts.live_test.ssh_bootstrap import (
    clone_repo_into,
    export_env,
    fetch_worker_logs,
    launch_worker_detached,
    open_session,
    run_install,
)
from scripts.live_test.terminate_guard import guarded_terminate
from scripts.live_test.token_resolver import resolve_token_to_user_id


FRESH_VARIANT = "fresh"
FRESH_WORKDIR = "/workspace/Reigh-Worker-LiveTest"
FRESH_REPO_URL = "https://github.com/banodoco/Reigh-Worker.git"

log = get_logger(__name__)


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_matrix_cases(args) -> list:
    return build_matrix(
        anchor_image_a=args.anchor_image_a,
        anchor_image_b=args.anchor_image_b,
        timeout_image_sec=args.timeout_image,
        timeout_travel_segment_sec=args.timeout_travel_segment,
        timeout_travel_orchestrator_sec=args.timeout_travel_orchestrator,
    )


def _build_worker_env(token: str, supabase_url: str, service_role_key: str) -> dict[str, str]:
    return {
        "REIGH_ACCESS_TOKEN": token,
        "SUPABASE_SERVICE_ROLE_KEY": service_role_key,
        "SUPABASE_URL": supabase_url,
        "WORKER_DB_CLIENT_AUTH_MODE": "worker",
    }


def _runs_root() -> Path:
    return config.WORKER_ROOT / "scripts" / "live_test" / "runs"


def _prepare_context(args) -> dict[str, Any]:
    token = config.require_env("REIGH_LIVE_TEST_TOKEN")
    db = config.DatabaseClient()
    user_id = resolve_token_to_user_id(db, token)
    assert_user_queue_clean(db, user_id)
    project_id = get_or_create_live_test_project(db, user_id)
    cases = _build_matrix_cases(args)
    return {
        "db": db,
        "token": token,
        "user_id": user_id,
        "project_id": project_id,
        "cases": cases,
    }


def _validate_cases(cases: list, project_id: str) -> None:
    for index, case in enumerate(cases, start=1):
        render_case_payload(case, project_id=project_id, unique_suffix=f"fresh-{index}")


def _print_dry_run_plan(*, token: str, project_id: str, cases: list, args) -> None:
    supabase_url = config.require_env("SUPABASE_URL")
    launch_command = build_run_worker_command(
        FRESH_WORKDIR,
        reigh_token=token,
        supabase_url=supabase_url,
        worker_id="<runpod-pod-id>",
        wgp_profile=args.wgp_profile,
        idle_release_minutes=0,
    )

    print("Variant: fresh")
    print(f"Project ID: {project_id}")
    print(f"Clone target: {FRESH_WORKDIR}")
    print(f"Terminate after run: {not args.no_terminate}")
    print("Injected env vars:")
    for key in ("REIGH_ACCESS_TOKEN", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_URL", "WORKER_DB_CLIENT_AUTH_MODE"):
        print(f"- {key}")
    print("Planned launch command:")
    print(launch_command)
    print("Planned tasks:")
    for case in cases:
        print(f"- {case.name} ({case.task_type}, timeout={case.timeout_sec}s)")


def run(args) -> int:
    context = _prepare_context(args)
    token = context["token"]
    db = context["db"]
    project_id = context["project_id"]
    cases = context["cases"]

    _validate_cases(cases, project_id)

    if args.dry_run:
        _print_dry_run_plan(token=token, project_id=project_id, cases=cases, args=args)
        return 0

    api_key: str | None = None
    api_key = config.require_env("RUNPOD_API_KEY")
    supabase_url = config.require_env("SUPABASE_URL")
    service_role_key = config.require_env("SUPABASE_SERVICE_ROLE_KEY")
    worker_env = _build_worker_env(token, supabase_url, service_role_key)
    out_dir = _runs_root() / _timestamp_label()

    pod_id: str | None = None
    ssh = None
    try:
        from runpod_lifecycle.api import create_pod as create_pod_and_wait
        from runpod_lifecycle.api import get_network_volumes

        network_volume_id: str | None = None
        selected_volume_name: str | None = None
        try:
            available = get_network_volumes(api_key)
            by_name = {v.get("name"): v.get("id") for v in available if v.get("name")}
            for candidate in config.RUNPOD_STORAGE_VOLUMES:
                if by_name.get(candidate):
                    network_volume_id = by_name[candidate]
                    selected_volume_name = candidate
                    break
        except Exception as exc:
            log.warning("could not list network volumes (%s); continuing without one", exc)

        if network_volume_id:
            log.info("attaching network volume %s (%s) at %s", selected_volume_name, network_volume_id, config.RUNPOD_VOLUME_MOUNT_PATH)
        else:
            log.warning("no network volume matched %s; pod will only have ephemeral container disk", list(config.RUNPOD_STORAGE_VOLUMES))

        pod = create_pod_and_wait(
            api_key=api_key,
            gpu_type_id=config.RUNPOD_GPU_TYPE,
            image_name=config.RUNPOD_WORKER_IMAGE,
            name=f"reigh-live-test-fresh-{_timestamp_label().lower()}",
            network_volume_id=network_volume_id,
            volume_mount_path=config.RUNPOD_VOLUME_MOUNT_PATH,
            disk_in_gb=config.LIVE_TEST_DISK_SIZE_GB,
            container_disk_in_gb=config.LIVE_TEST_CONTAINER_DISK_GB,
            min_vcpu_count=config.RUNPOD_MIN_VCPU_COUNT,
            min_memory_in_gb=config.RUNPOD_MIN_MEMORY_GB,
            template_id=config.RUNPOD_TEMPLATE_ID,
            env_vars=worker_env,
        )
        if not pod or not pod.get("id"):
            raise RuntimeError("create_pod_and_wait did not return a pod id")

        pod_id = str(pod["id"])
        ssh = open_session(pod_id, api_key)
        clone_repo_into(ssh, FRESH_WORKDIR, FRESH_REPO_URL, branch=args.ref or "main")
        run_install(ssh, FRESH_WORKDIR)

        command = build_run_worker_command(
            FRESH_WORKDIR,
            reigh_token=token,
            supabase_url=supabase_url,
            worker_id=pod_id,
            wgp_profile=args.wgp_profile,
            idle_release_minutes=0,
        )
        launch_worker_detached(ssh, export_env(worker_env) + " && " + command)
        wait_until_ready(db, worker_id=pod_id, timeout_sec=900)

        results = run_matrix(db, project_id, cases)
        write_report(results, FRESH_VARIANT, pod_id, out_dir)
        return 0
    finally:
        if ssh is not None:
            try:
                logs = fetch_worker_logs(ssh, FRESH_WORKDIR)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "worker_logs.txt").write_text(logs, encoding="utf-8")
            except Exception as exc:
                log.warning("failed to fetch fresh variant worker logs: %s", exc)
            finally:
                disconnect = getattr(ssh, "disconnect", None)
                if callable(disconnect):
                    disconnect()
        guarded_terminate(pod_id, api_key if not args.dry_run else None, no_terminate=args.no_terminate)


__all__ = [
    "FRESH_REPO_URL",
    "FRESH_VARIANT",
    "FRESH_WORKDIR",
    "run",
]
