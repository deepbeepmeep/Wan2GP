"""Guardian helpers used by runtime tests and compatibility shims."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_worker_fragment(worker_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in worker_id)


def _guardian_log_path(worker_id: str, suffix: str) -> Path:
    return Path("/tmp") / f"guardian_{_safe_worker_fragment(worker_id)}{suffix}"


def collect_logs_from_queue(log_queue, max_count: int = 100) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    while len(logs) < max_count:
        try:
            logs.append(log_queue.get_nowait())
        except Exception:
            break
    return logs


def check_process_alive(pid: int, start_time: float | None = None) -> bool:
    try:
        os.kill(pid, 0)
        if start_time is not None:
            try:
                import psutil

                proc = psutil.Process(pid)
                if abs(proc.create_time() - start_time) > 0.1:
                    return False
            except Exception:
                return False
        return True
    except (OSError, ProcessLookupError):
        return False


def get_vram_info() -> tuple[int, int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            total, used = result.stdout.strip().split(",")
            return int(float(total)), int(float(used))
    except Exception:
        pass
    return 0, 0


def send_heartbeat_with_logs(
    worker_id: str,
    vram_total: int,
    vram_used: int,
    logs: list[dict[str, Any]],
    config: dict[str, str],
    status: str = "active",
) -> bool:
    try:
        payload = json.dumps(
            {
                "worker_id_param": worker_id,
                "vram_total_mb_param": vram_total,
                "vram_used_mb_param": vram_used,
                "logs_param": logs,
                "status_param": status,
            }
        )
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                "-m",
                "10",
                f'{config["db_url"]}/rest/v1/rpc/func_worker_heartbeat_with_logs',
                "-H",
                f'apikey: {config["api_key"]}',
                "-H",
                "Content-Type: application/json",
                "-H",
                "Prefer: return=representation",
                "-d",
                payload,
            ],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def guardian_main(worker_id: str, worker_pid: int, log_queue, config: dict[str, str]):
    worker_start_time = None
    try:
        import psutil

        worker_start_time = psutil.Process(worker_pid).create_time()
    except Exception:
        worker_start_time = None

    while True:
        if not check_process_alive(worker_pid, worker_start_time):
            send_heartbeat_with_logs(worker_id, 0, 0, [], config, status="crashed")
            break

        total, used = get_vram_info()
        logs = collect_logs_from_queue(log_queue, max_count=100)
        send_heartbeat_with_logs(worker_id, total, used, logs, config, status="active")
        time.sleep(20)
