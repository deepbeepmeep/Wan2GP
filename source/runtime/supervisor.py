from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections import deque

from source.runtime.worker_protocol import IDLE_RELEASE_EXIT_CODE

WORKER_SHIM = str(Path(__file__).resolve().parents[2] / "worker.py")
if not Path(WORKER_SHIM).is_file():
    raise FileNotFoundError(f"worker shim not found: {WORKER_SHIM}")


def _build_child_cmd(argv_tail):
    return [sys.executable, "-u", WORKER_SHIM, *argv_tail]


def _normalize_exit_code(returncode: int) -> int:
    if returncode < 0:
        return 128 + (-returncode)
    return returncode


def main(argv: list[str] | None = None) -> int:
    argv_tail = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    crashes = deque()
    shutdown_state = {"requested": False}
    child_state = {"proc": None}

    def _forward_signal(sig, _frame):
        shutdown_state["requested"] = True
        child = child_state["proc"]
        if child is None or child.poll() is not None:
            return
        if os.name == "nt":
            child.terminate()
            return
        child.send_signal(sig)

    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)

    while True:
        child = subprocess.Popen(
            _build_child_cmd(argv_tail),
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )
        child_state["proc"] = child
        returncode = child.wait()
        child_state["proc"] = None
        normalized_rc = _normalize_exit_code(returncode)

        if shutdown_state["requested"]:
            return normalized_rc
        if returncode == IDLE_RELEASE_EXIT_CODE:
            logger.info("[SUPERVISOR] idle-release restart")
            continue
        if returncode == 0:
            return 0
        if returncode < 0:
            return normalized_rc

        now = time.monotonic()
        crashes.append(now)
        while crashes and (now - crashes[0]) > 300:
            crashes.popleft()
        if len(crashes) >= 5:
            logger.error("[SUPERVISOR] circuit open after 5 crashes in 5 min — giving up")
            return returncode
        time.sleep(min(2 ** (len(crashes) - 1), 30))
