"""Launch-command builders for live worker tests."""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath


def _quote(value: str) -> str:
    return shlex.quote(value)


def _normalize_cli_args(cli_args: list[str]) -> list[str]:
    normalized = [str(arg) for arg in cli_args if str(arg).strip()]
    while normalized and normalized[0] == "nohup":
        normalized = normalized[1:]
    while normalized and normalized[-1] == "&":
        normalized = normalized[:-1]
    return normalized


def build_run_worker_command(
    workdir: str,
    *,
    reigh_token: str,
    supabase_url: str,
    worker_id: str,
    wgp_profile: int,
    idle_release_minutes: int,
) -> str:
    workdir_q = _quote(workdir)
    prefix = [
        f"cd {workdir_q}",
        "mkdir -p logs",
        'export PATH="$HOME/.local/bin:$PATH"',
    ]
    worker_parts = [
        "nohup",
        "uv",
        "run",
        "--python",
        "3.10",
        "--extra",
        "cuda124",
        "python",
        "run_worker.py",
        "--supabase-url",
        _quote(supabase_url),
        "--reigh-access-token",
        _quote(reigh_token),
        "--worker",
        _quote(worker_id),
        "--wgp-profile",
        str(wgp_profile),
        "--idle-release-minutes",
        str(idle_release_minutes),
        "--save-logging",
        "logs/worker.log",
        "</dev/null",
        ">",
        "logs/startup.log",
        "2>&1",
        "&",
    ]
    return " && ".join(prefix) + " && " + " ".join(worker_parts) + " disown"


def build_direct_worker_command(workdir: str, *, cli_args: list[str]) -> str:
    normalized = _normalize_cli_args(cli_args)
    if not normalized:
        raise ValueError("cli_args is empty")
    if not any(arg.endswith("worker.py") or arg == "worker.py" for arg in normalized):
        raise ValueError("cli_args does not contain worker.py")

    workdir_q = _quote(workdir)
    rendered_args = " ".join(_quote(arg) for arg in normalized)
    startup_log = PurePosixPath("logs") / "startup.log"
    return f"cd {workdir_q} && nohup {rendered_args} > {startup_log} 2>&1 &"


__all__ = ["build_direct_worker_command", "build_run_worker_command"]
