"""Subprocess adapter for Sprint 2 direct VibeComfy routes.

The worker must not import VibeComfy directly.  This module crosses the Python
3.11 VibeComfy boundary with ``subprocess`` only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from source.core.log import headless_logger
from source.runtime.vibecomfy_profile import (
    PROCESS_DEFAULT_PROFILE,
    build_memory_profile_cli_args,
)
from source.task_handlers.tasks.template_routing import (
    ResolvedTask,
    RouteSupportState,
    WorkerBackend,
)


_MAX_CAPTURE_CHARS = 4000
_OUTPUT_EXTENSIONS = {
    ".apng",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".png",
    ".webm",
    ".webp",
}


def handle_vibecomfy_resolved_task(
    resolved: ResolvedTask,
    main_output_dir_base: str | Path,
) -> tuple[bool, str | None]:
    """Run a supported direct VibeComfy route and return the discovered output."""

    validation_error = _validate_supported_resolved_task(resolved)
    if validation_error:
        return False, validation_error

    run_workspace = _prepare_run_workspace(main_output_dir_base, resolved.task_id)
    command = _build_vibecomfy_command(resolved)
    env = _build_subprocess_env(run_workspace)

    headless_logger.debug_block(
        "VIBECOMFY_ROUTE",
        {
            "task_id": resolved.task_id,
            "task_type": resolved.task_type,
            "route_key": resolved.route_key,
            "backend": resolved.backend.value,
            "template_id": resolved.template_id,
            "support_state": resolved.support_state.value,
            "memory_profile": _memory_profile_for_log(resolved),
        },
        task_id=resolved.task_id,
    )

    try:
        completed = subprocess.run(
            command,
            cwd=run_workspace,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        message = _failure_message(
            resolved=resolved,
            exit_code=None,
            stderr=str(exc),
            stdout="",
        )
        _log_failure(
            resolved=resolved,
            exit_code=None,
            stderr=str(exc),
            stdout="",
        )
        headless_logger.error(message, task_id=resolved.task_id)
        return False, message

    stdout = _bounded(completed.stdout)
    stderr = _bounded(completed.stderr)
    if completed.returncode != 0:
        message = _failure_message(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr,
            stdout=stdout,
        )
        _log_failure(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr,
            stdout=stdout,
        )
        headless_logger.error(message, task_id=resolved.task_id)
        return False, message

    output_path = _discover_output_path(stdout=stdout, run_workspace=run_workspace)
    if output_path is None:
        message = _failure_message(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr or "no output path discovered",
            stdout=stdout,
        )
        _log_failure(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr or "no output path discovered",
            stdout=stdout,
        )
        headless_logger.error(message, task_id=resolved.task_id)
        return False, message

    headless_logger.debug_block(
        "VIBECOMFY_COMPLETE",
        {
            "task_id": resolved.task_id,
            "route_key": resolved.route_key,
            "backend": resolved.backend.value,
            "template_id": resolved.template_id,
            "memory_profile": _memory_profile_for_log(resolved),
            "exit_code": completed.returncode,
            "output_path": output_path,
        },
        task_id=resolved.task_id,
    )
    return True, output_path


def _validate_supported_resolved_task(resolved: ResolvedTask) -> str | None:
    if resolved.backend != WorkerBackend.VIBECOMFY:
        return f"VibeComfy adapter received non-VibeComfy backend {resolved.backend.value}"
    if resolved.fail_closed_reason:
        return (
            f"VibeComfy backend fail-closed for task {resolved.task_id} "
            f"({resolved.route_key}): {resolved.fail_closed_reason}"
        )
    if resolved.support_state != RouteSupportState.VIBECOMFY_SUPPORTED:
        return (
            f"VibeComfy route {resolved.route_key!r} is "
            f"{resolved.support_state.value}; adapter will not execute it"
        )
    if not resolved.template_id:
        return f"VibeComfy route {resolved.route_key!r} has no template_id"
    return None


def _build_vibecomfy_command(resolved: ResolvedTask) -> list[str]:
    command = [
        _vibecomfy_python(),
        "-m",
        "vibecomfy.cli",
        "run",
        str(resolved.template_id),
        "--ready",
        "--runtime",
        "embedded",
    ]

    prompt = resolved.params.get("prompt")
    if prompt is not None:
        command.extend(["--prompt", str(prompt)])

    seed = resolved.params.get("seed")
    if seed is not None:
        command.extend(["--seed", str(int(seed))])

    steps = resolved.params.get("steps", resolved.params.get("num_inference_steps"))
    if steps is not None:
        command.extend(["--steps", str(int(steps))])

    command.extend(
        build_memory_profile_cli_args(
            process_default=_process_default_memory_profile(),
            override_profile=_override_memory_profile(resolved),
        )
    )
    return command


def _prepare_run_workspace(main_output_dir_base: str | Path, task_id: str) -> Path:
    safe_task_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)
    run_workspace = Path(main_output_dir_base) / "vibecomfy_runs" / safe_task_id
    run_workspace.mkdir(parents=True, exist_ok=True)
    return run_workspace


def _build_subprocess_env(run_workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["VIBECOMFY_WORKER_RUN_DIR"] = str(run_workspace)

    vibecomfy_cwd = os.environ.get("VIBECOMFY_CWD")
    if vibecomfy_cwd:
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{vibecomfy_cwd}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else vibecomfy_cwd
        )

    return env


def _vibecomfy_python() -> str:
    return os.environ.get("VIBECOMFY_PYTHON") or "python3.11"


def _process_default_memory_profile() -> int | None:
    raw_profile = os.environ.get("VIBECOMFY_MEMORY_PROFILE")
    if raw_profile is None or raw_profile.strip() == "":
        return None
    return int(raw_profile)


def _override_memory_profile(resolved: ResolvedTask) -> int | None:
    raw_profile = resolved.params.get("override_profile")
    if raw_profile is None:
        return PROCESS_DEFAULT_PROFILE
    return int(raw_profile)


def _memory_profile_for_log(resolved: ResolvedTask) -> int | None:
    try:
        args = build_memory_profile_cli_args(
            process_default=_process_default_memory_profile(),
            override_profile=_override_memory_profile(resolved),
        )
    except ValueError:
        return None
    if "--memory-profile" not in args:
        return None
    return int(args[args.index("--memory-profile") + 1])


def _discover_output_path(*, stdout: str, run_workspace: Path) -> str | None:
    for line in stdout.splitlines():
        if line.startswith("output: "):
            return _resolve_output_path(line.removeprefix("output: ").strip(), run_workspace)

    metadata_path = _metadata_path_from_stdout(stdout, run_workspace)
    if metadata_path and metadata_path.exists():
        output = _output_from_metadata(metadata_path, run_workspace)
        if output:
            return output

    for output_path in _candidate_output_files(run_workspace):
        return str(output_path)

    return None


def _metadata_path_from_stdout(stdout: str, run_workspace: Path) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("metadata: "):
            raw_path = line.removeprefix("metadata: ").strip()
            return Path(_resolve_output_path(raw_path, run_workspace))
    return None


def _output_from_metadata(metadata_path: Path, run_workspace: Path) -> str | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    for output in _flatten_outputs(metadata.get("outputs")):
        return _resolve_output_path(str(output), run_workspace)
    return None


def _flatten_outputs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        outputs: list[str] = []
        for item in value.values():
            outputs.extend(_flatten_outputs(item))
        return outputs
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        outputs = []
        for item in value:
            outputs.extend(_flatten_outputs(item))
        return outputs
    return []


def _resolve_output_path(value: str, run_workspace: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(run_workspace / path)


def _candidate_output_files(run_workspace: Path) -> list[Path]:
    candidates = [
        path
        for path in run_workspace.rglob("*")
        if path.is_file() and path.suffix.lower() in _OUTPUT_EXTENSIONS
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _failure_message(
    *,
    resolved: ResolvedTask,
    exit_code: int | None,
    stderr: str,
    stdout: str,
) -> str:
    return (
        "VibeComfy task failed "
        f"task_id={resolved.task_id} "
        f"backend={resolved.backend.value} "
        f"template={resolved.template_id} "
        f"profile={_memory_profile_for_log(resolved)} "
        f"exit_code={exit_code} "
        f"stderr={_bounded(stderr)!r} "
        f"stdout={_bounded(stdout)!r}"
    )


def _log_failure(
    *,
    resolved: ResolvedTask,
    exit_code: int | None,
    stderr: str,
    stdout: str,
) -> None:
    headless_logger.debug_block(
        "VIBECOMFY_FAILURE",
        {
            "task_id": resolved.task_id,
            "task_type": resolved.task_type,
            "route_key": resolved.route_key,
            "backend": resolved.backend.value,
            "template_id": resolved.template_id,
            "memory_profile": _memory_profile_for_log(resolved),
            "exit_code": exit_code,
            "stderr": _bounded(stderr),
            "stdout": _bounded(stdout),
        },
        task_id=resolved.task_id,
    )


def _bounded(value: str | None, limit: int = _MAX_CAPTURE_CHARS) -> str:
    text = value or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


__all__ = ["handle_vibecomfy_resolved_task"]
