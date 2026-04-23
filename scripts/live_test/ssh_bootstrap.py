"""SSH-side worker bootstrap helpers shared by the live-test variants."""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass
from typing import Literal

import scripts.live_test as live_test_pkg


APT_INSTALL_PACKAGES = (
    "python3.10-venv",
    "python3.10-dev",
    "build-essential",
    "ffmpeg",
    "git",
    "curl",
    "wget",
)
PROCESS_SCAN_COMMAND = (
    r"ps -eo pid=,args= | grep -E '(run_worker\.py|worker\.py|source\.runtime\.worker)' | grep -v grep"
)
KILL_COMMAND = (
    "pkill -f run_worker.py; "
    "pkill -f 'python worker.py'; "
    "pkill -f 'source.runtime.worker'; "
    "sleep 2"
)


@dataclass(frozen=True)
class WorkerProcessInfo:
    family: Literal["supervisor", "direct"]
    cmdline: list[str]
    pid: int


def _quote(value: str) -> str:
    return shlex.quote(str(value))


def _execute(ssh, command: str, *, timeout: int = 600, check: bool = True) -> tuple[str, str]:
    exit_code, stdout, stderr = ssh.execute_command(command, timeout=timeout)
    if check and exit_code != 0:
        raise RuntimeError(
            f"Remote command failed with exit {exit_code}: {command}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stdout, stderr


def open_session(pod_id: str, api_key: str, *, ssh_wait_timeout: int = 300, poll_interval: int = 5):
    deadline = time.monotonic() + ssh_wait_timeout
    ssh_details = None
    while time.monotonic() < deadline:
        ssh_details = live_test_pkg.get_pod_ssh_details(pod_id, api_key)
        if ssh_details and ssh_details.get("ip") and ssh_details.get("port"):
            break
        time.sleep(poll_interval)
    if not ssh_details or not ssh_details.get("ip") or not ssh_details.get("port"):
        raise RuntimeError(
            f"Could not resolve SSH details for pod {pod_id} within {ssh_wait_timeout}s"
        )

    import os as _os
    private_key_path = _os.environ.get("REIGH_LIVE_TEST_SSH_KEY") or "~/.ssh/id_ed25519"
    ssh = live_test_pkg.SSHClient(
        hostname=str(ssh_details["ip"]),
        port=int(ssh_details["port"]),
        username="root",
        password=ssh_details.get("password"),
        private_key_path=private_key_path,
    )
    connect_deadline = time.monotonic() + ssh_wait_timeout
    last_err: Exception | None = None
    while time.monotonic() < connect_deadline:
        try:
            ssh.connect()
            return ssh
        except Exception as exc:
            last_err = exc
            time.sleep(poll_interval)
    raise RuntimeError(f"Could not SSH into pod {pod_id} within {ssh_wait_timeout}s: {last_err}")


def clone_repo_into(ssh, workdir: str, repo_url: str, branch: str) -> None:
    parent = workdir.rsplit("/", 1)[0] or "/"
    command = (
        f"mkdir -p {_quote(parent)} && "
        f"rm -rf {_quote(workdir)} && "
        f"git clone --branch {_quote(branch)} --single-branch --recurse-submodules {_quote(repo_url)} {_quote(workdir)}"
    )
    _execute(ssh, command, timeout=1800)


def run_install(ssh, workdir: str) -> None:
    package_list = " ".join(APT_INSTALL_PACKAGES)
    command = (
        "bash -lc "
        + _quote(
            "set -euo pipefail\n"
            "apt-get update\n"
            f"apt-get install -y {package_list}\n"
            "if ! command -v uv >/dev/null 2>&1; then\n"
            "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "  export PATH=\"$HOME/.local/bin:$PATH\"\n"
            "fi\n"
            f"cd {shlex.quote(workdir)}\n"
            "export PATH=\"$HOME/.local/bin:$PATH\"\n"
            "uv sync --locked --extra cuda124\n"
        )
    )
    _execute(ssh, command, timeout=3600)


def export_env(env: dict[str, str]) -> str:
    exports = dict(env)
    required = {
        "REIGH_ACCESS_TOKEN",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_URL",
        "WORKER_DB_CLIENT_AUTH_MODE",
    }
    missing = sorted(name for name in required if not exports.get(name))
    if missing:
        raise ValueError(f"Missing required environment values for export_env: {', '.join(missing)}")
    if exports["WORKER_DB_CLIENT_AUTH_MODE"] != "worker":
        raise ValueError("WORKER_DB_CLIENT_AUTH_MODE must be 'worker' for PAT live tests")
    return " && ".join(f"export {key}={_quote(value)}" for key, value in sorted(exports.items()))


def capture_current_worker_cmdline(ssh) -> WorkerProcessInfo | None:
    stdout, _ = _execute(ssh, PROCESS_SCAN_COMMAND, check=False)
    rows: list[tuple[int, list[str]]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_str, args = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        rows.append((pid, shlex.split(args)))

    supervisor_rows = [row for row in rows if any("run_worker.py" in arg for arg in row[1])]
    if supervisor_rows:
        pid, cmdline = supervisor_rows[0]
        return WorkerProcessInfo(family="supervisor", cmdline=cmdline, pid=pid)

    direct_rows = [
        row
        for row in rows
        if any(arg == "worker.py" or arg.endswith("/worker.py") or "source.runtime.worker" in arg for arg in row[1])
    ]
    if direct_rows:
        pid, cmdline = direct_rows[0]
        return WorkerProcessInfo(family="direct", cmdline=cmdline, pid=pid)

    return None


def kill_supervisor_and_worker(ssh) -> None:
    _execute(ssh, KILL_COMMAND, check=False, timeout=30)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        stdout, _ = _execute(
            ssh,
            "pgrep -af run_worker.py || true; "
            "pgrep -af 'python worker.py' || true; "
            "pgrep -af source.runtime.worker || true",
            check=False,
            timeout=30,
        )
        if not stdout.strip():
            return
        time.sleep(1)
    raise RuntimeError(f"Worker processes are still running after kill attempt:\n{stdout}")


def launch_worker_detached(ssh, command_line: str) -> None:
    # Wrap so bash exits immediately after backgrounding; avoid paramiko hanging
    # on stdout EOF when a nohup child inherits the ssh channel streams.
    wrapped = f"bash -c {shlex.quote(command_line + ' ; exit 0')} </dev/null >/dev/null 2>&1"
    client = ssh.client
    transport = client.get_transport()
    channel = transport.open_session()
    try:
        channel.set_combine_stderr(True)
        channel.exec_command(wrapped)
        deadline = time.monotonic() + 30
        while not channel.exit_status_ready() and time.monotonic() < deadline:
            time.sleep(0.2)
        if not channel.exit_status_ready():
            return  # detached; don't wait further
        exit_code = channel.recv_exit_status()
        if exit_code != 0:
            raise RuntimeError(f"Detached launch command exited with {exit_code}")
    finally:
        channel.close()


def fetch_worker_logs(ssh, workdir: str, lines: int = 300) -> str:
    startup_script = _quote(
        f"cd {workdir} && "
        f'{{ echo "=== startup.log ==="; tail -n {int(lines)} logs/startup.log 2>/dev/null || true; }}'
    )
    startup_stdout, _ = _execute(
        ssh,
        f"bash -lc {startup_script}",
        check=False,
        timeout=60,
    )
    worker_script = _quote(
        f"cd {workdir} && "
        f'{{ echo "=== worker.log ==="; tail -n {int(lines)} logs/worker.log 2>/dev/null || true; }}'
    )
    worker_stdout, _ = _execute(
        ssh,
        f"bash -lc {worker_script}",
        check=False,
        timeout=60,
    )
    return "\n".join(part.rstrip() for part in (startup_stdout, worker_stdout) if part.strip())


__all__ = [
    "APT_INSTALL_PACKAGES",
    "KILL_COMMAND",
    "PROCESS_SCAN_COMMAND",
    "WorkerProcessInfo",
    "capture_current_worker_cmdline",
    "clone_repo_into",
    "export_env",
    "fetch_worker_logs",
    "kill_supervisor_and_worker",
    "launch_worker_detached",
    "open_session",
    "run_install",
]
