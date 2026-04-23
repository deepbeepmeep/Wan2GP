"""Variant B driver: update/take over an installed pod and run the live-test matrix."""

from __future__ import annotations

import asyncio
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.live_test import config
from scripts.live_test.git_ops import (
    cleanup_temp_branch,
    push_working_copy_to_temp_branch,
    restore_local_state,
    snapshot_local_state,
)
from scripts.live_test.heartbeat_waiter import wait_until_ready
from scripts.live_test.launch_command import build_direct_worker_command, build_run_worker_command
from scripts.live_test.logger import get_logger
from scripts.live_test.matrix import build_matrix, render_case_payload, run_matrix
from scripts.live_test.preflight import assert_user_queue_clean, get_or_create_live_test_project
from scripts.live_test.report import write_report
from scripts.live_test.safety_gate import assert_safe_to_take_over
from scripts.live_test.ssh_bootstrap import (
    WorkerProcessInfo,
    capture_current_worker_cmdline,
    export_env,
    fetch_worker_logs,
    kill_supervisor_and_worker,
    launch_worker_detached,
    open_session,
)
from scripts.live_test.terminate_guard import guarded_terminate
from scripts.live_test.token_resolver import resolve_token_to_user_id


UPDATE_VARIANT = "update"
UPDATE_WORKDIR = "/workspace/Reigh-Worker"

log = get_logger(__name__)


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _runs_root() -> Path:
    return config.WORKER_ROOT / "scripts" / "live_test" / "runs"


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
        render_case_payload(case, project_id=project_id, unique_suffix=f"update-{index}")


def _ssh_execute(ssh, command: str, *, timeout: int = 1800, check: bool = True) -> tuple[str, str]:
    exit_code, stdout, stderr = ssh.execute_command(command, timeout=timeout)
    if check and exit_code != 0:
        raise RuntimeError(
            f"Remote command failed with exit {exit_code}: {command}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stdout, stderr


def _quote(value: str) -> str:
    return shlex.quote(str(value))


def _read_remote_branch(ssh) -> str:
    stdout, _ = _ssh_execute(
        ssh,
        f"bash -lc {_quote(f'cd {UPDATE_WORKDIR} && git symbolic-ref --short HEAD || echo DETACHED')}",
        timeout=60,
    )
    return stdout.strip() or "DETACHED"


def _read_remote_sha(ssh) -> str:
    stdout, _ = _ssh_execute(
        ssh,
        f"bash -lc {_quote(f'cd {UPDATE_WORKDIR} && git rev-parse HEAD')}",
        timeout=60,
    )
    return stdout.strip()


def _extract_worker_id_from_cmdline(cmdline: list[str]) -> str | None:
    for index, item in enumerate(cmdline):
        if item == "--worker" and index + 1 < len(cmdline):
            return cmdline[index + 1]
    return None


def _resolve_existing_worker_id(db, pod_id: str, prev_proc: WorkerProcessInfo | None) -> str:
    if prev_proc:
        worker_id = _extract_worker_id_from_cmdline(prev_proc.cmdline)
        if worker_id:
            return worker_id

    result = db.supabase.table("workers").select("id, metadata, created_at, last_heartbeat").execute()
    rows = list(getattr(result, "data", None) or [])
    matching = []
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and str(metadata.get("runpod_id")) == pod_id:
            matching.append(row)

    if not matching:
        raise RuntimeError(f"Could not resolve existing worker_id for pod {pod_id}")

    matching.sort(
        key=lambda row: (
            str(row.get("last_heartbeat") or ""),
            str(row.get("created_at") or ""),
        ),
        reverse=True,
    )
    return str(matching[0]["id"])


def _build_supervisor_restore_command(workdir: str, cli_args: list[str]) -> str:
    if not cli_args:
        raise ValueError("Expected captured supervisor cmdline")
    return f"cd {_quote(workdir)} && nohup {' '.join(_quote(arg) for arg in cli_args)} > logs/startup.log 2>&1 &"


def _should_skip_restore(branch_name: str) -> bool:
    return branch_name == "DETACHED" or branch_name.startswith("live-test/")


def _remote_checkout_and_sync(ssh, branch: str) -> None:
    command = (
        f"cd {shlex.quote(UPDATE_WORKDIR)} && "
        "git fetch origin && "
        f"git checkout {shlex.quote(branch)} && "
        f"git pull --ff-only origin {shlex.quote(branch)} && "
        "uv sync --locked --extra cuda124"
    )
    _ssh_execute(ssh, f"bash -lc {_quote(command)}", timeout=3600)


def _restore_remote_state(
    ssh,
    *,
    prev_remote_branch: str,
    prev_remote_sha: str,
    prev_proc: WorkerProcessInfo | None,
) -> None:
    if _should_skip_restore(prev_remote_branch):
        kill_supervisor_and_worker(ssh)
        log.info(
            "skipping pod restore because previous branch was scratch or detached: %s",
            prev_remote_branch,
        )
        return

    kill_supervisor_and_worker(ssh)
    restore_command = (
        f"cd {shlex.quote(UPDATE_WORKDIR)} && "
        f"git checkout {shlex.quote(prev_remote_branch)} && "
        f"git reset --hard {shlex.quote(prev_remote_sha)} && "
        "uv sync --locked --extra cuda124"
    )
    _ssh_execute(ssh, f"bash -lc {_quote(restore_command)}", timeout=3600)

    if prev_proc is None:
        log.info("previous pod had no worker process; leaving restored checkout stopped")
        return

    if prev_proc.family == "supervisor":
        launch_worker_detached(ssh, _build_supervisor_restore_command(UPDATE_WORKDIR, prev_proc.cmdline))
        return

    launch_worker_detached(ssh, build_direct_worker_command(UPDATE_WORKDIR, cli_args=prev_proc.cmdline))


def _spawn_takeover_pod(db, api_key: str) -> tuple[str, str]:
    from gpu_orchestrator.worker_spawner import create_worker_spawner

    spawner = create_worker_spawner(config=None, db=db)
    worker_id = spawner.generate_worker_id()
    created = asyncio.run(db.create_worker_record(worker_id, spawner.gpu_type))
    if not created:
        raise RuntimeError(f"Failed to create worker record for {worker_id}")

    spawn_result = asyncio.run(spawner.spawn_worker(worker_id))
    if not spawn_result or not spawn_result.get("runpod_id"):
        raise RuntimeError(f"spawn_worker did not return a runpod_id for {worker_id}")

    pod_id = str(spawn_result["runpod_id"])
    metadata = {
        "runpod_id": pod_id,
        "pod_details": spawn_result.get("pod_details"),
        "ram_tier": spawn_result.get("ram_tier"),
        "storage_volume": spawn_result.get("storage_volume"),
    }
    asyncio.run(db.update_worker_status(worker_id, "spawning", metadata))

    started = asyncio.run(spawner.start_worker_process(pod_id, worker_id, has_pending_tasks=False))
    if not started:
        raise RuntimeError(f"start_worker_process failed for worker {worker_id} on pod {pod_id}")

    return worker_id, pod_id


def _print_dry_run_plan(*, cases: list, token: str, args) -> None:
    supabase_url = config.require_env("SUPABASE_URL")
    mode = "spawn-takeover" if args.spawn_takeover else "existing"
    pod_hint = args.pod_id or "<spawned-runpod-id>"
    worker_hint = "<new-worker-id>" if args.spawn_takeover else "<existing-worker-id>"
    print("Variant: update")
    print(f"Mode: {mode}")
    print(f"Target pod: {pod_hint}")
    print("Injected env vars:")
    for key in ("REIGH_ACCESS_TOKEN", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_URL", "WORKER_DB_CLIENT_AUTH_MODE"):
        print(f"- {key}")
    print("Planned launch command:")
    print(
        build_run_worker_command(
            UPDATE_WORKDIR,
            reigh_token=token,
            supabase_url=supabase_url,
            worker_id=worker_hint,
            wgp_profile=args.wgp_profile,
            idle_release_minutes=0,
        )
    )
    print("Planned tasks:")
    for case in cases:
        print(f"- {case.name} ({case.task_type}, timeout={case.timeout_sec}s)")


def run(args) -> int:
    context = _prepare_context(args)
    token = context["token"]
    db = context["db"]
    user_id = context["user_id"]
    project_id = context["project_id"]
    cases = context["cases"]

    _validate_cases(cases, project_id)

    if args.dry_run:
        _print_dry_run_plan(cases=cases, token=token, args=args)
        return 0

    api_key: str | None = None
    api_key = config.require_env("RUNPOD_API_KEY")
    supabase_url = config.require_env("SUPABASE_URL")
    service_role_key = config.require_env("SUPABASE_SERVICE_ROLE_KEY")
    worker_env = _build_worker_env(token, supabase_url, service_role_key)
    mode = "spawn-takeover" if args.spawn_takeover else "existing"
    out_dir = _runs_root() / _timestamp_label()

    ssh = None
    pod_id: str | None = None
    worker_id: str | None = None
    snapshot = None
    branch: str | None = None
    preserve_branch = True
    prev_remote_branch = "DETACHED"
    prev_remote_sha = ""
    prev_proc: WorkerProcessInfo | None = None

    try:
        if args.spawn_takeover:
            worker_id, pod_id = _spawn_takeover_pod(db, api_key)
        else:
            pod_id = args.pod_id

        assert pod_id
        assert_safe_to_take_over(
            db,
            pod_id,
            user_id,
            allow_fresh_heartbeat=args.spawn_takeover,
        )

        snapshot = snapshot_local_state("reigh-worker")
        branch, _sha = push_working_copy_to_temp_branch("reigh-worker", snapshot)

        ssh = open_session(pod_id, api_key)
        prev_remote_branch = _read_remote_branch(ssh)
        prev_remote_sha = _read_remote_sha(ssh)
        prev_proc = capture_current_worker_cmdline(ssh)

        if not worker_id:
            worker_id = _resolve_existing_worker_id(db, pod_id, prev_proc)

        _remote_checkout_and_sync(ssh, branch)
        kill_supervisor_and_worker(ssh)
        launch_worker_detached(
            ssh,
            export_env(worker_env)
            + " && "
            + build_run_worker_command(
                UPDATE_WORKDIR,
                reigh_token=token,
                supabase_url=supabase_url,
                worker_id=worker_id,
                wgp_profile=args.wgp_profile,
                idle_release_minutes=0,
            ),
        )
        wait_until_ready(db, worker_id=worker_id, timeout_sec=900)

        results = run_matrix(db, project_id, cases)
        write_report(results, f"{UPDATE_VARIANT}-{mode}", pod_id, out_dir)
        _restore_remote_state(
            ssh,
            prev_remote_branch=prev_remote_branch,
            prev_remote_sha=prev_remote_sha,
            prev_proc=prev_proc,
        )
        preserve_branch = False
        return 0
    finally:
        if branch:
            kept_branch = cleanup_temp_branch(branch, preserve=preserve_branch, submodule_path="reigh-worker")
            if preserve_branch:
                print(f"Preserved temp branch for inspection: {kept_branch}")
        if snapshot is not None:
            restore_local_state("reigh-worker", snapshot)
        if ssh is not None:
            try:
                logs = fetch_worker_logs(ssh, UPDATE_WORKDIR)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "worker_logs.txt").write_text(logs, encoding="utf-8")
            except Exception as exc:
                log.warning("failed to fetch update variant worker logs: %s", exc)
            finally:
                disconnect = getattr(ssh, "disconnect", None)
                if callable(disconnect):
                    disconnect()
        guarded_terminate(pod_id, api_key if not args.dry_run else None, no_terminate=args.no_terminate)


__all__ = [
    "UPDATE_VARIANT",
    "UPDATE_WORKDIR",
    "run",
]
