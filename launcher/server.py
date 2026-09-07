"""Run wgp.py and know when its web UI is ready to be shown."""

import logging
import os
import socket
import threading
import time

from . import config, environment, process

LOGGER = logging.getLogger(__name__)

# run.bat restarts WanGP when it exits with this code (the in-app "restart"
# button); the launcher has to honour the same contract.
RESTART_EXIT_CODE = 42


def _port_is_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host, preferred):
    if preferred and _port_is_free(host, preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        chosen = sock.getsockname()[1]
    LOGGER.info("Port %s unavailable, using %s instead", preferred, chosen)
    return chosen


def _port_accepts_connections(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class WanGPServer:
    """Owns the wgp.py process: start, watch, restart on code 42, stop."""

    def __init__(self, root, python_exe, cfg, on_line=None):
        self.root = root
        self.python_exe = python_exe
        self.cfg = cfg
        self.on_line = on_line or (lambda _line: None)

        server_cfg = cfg.get("server", {})
        self.host = server_cfg.get("host", "127.0.0.1")
        self.port = pick_port(self.host, server_cfg.get("preferred_port", 7860))
        self.timeout = server_cfg.get("startup_timeout_seconds", 900)

        self._proc = None
        self._reader = None
        self._stopping = threading.Event()
        self._exited = threading.Event()
        self._log_path = os.path.join(config.logs_dir(), "wangp-server.log")
        self._log_handle = None
        self._restart_callback = None

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    @property
    def log_path(self):
        return self._log_path

    def _argv(self):
        argv = [
            self.python_exe,
            environment.MAIN_SCRIPT,
            "--server-name",
            self.host,
            "--server-port",
            str(self.port),
        ]
        argv.extend(environment.read_extra_args(self.root))
        return argv

    def _emit(self, line):
        self.on_line(line)
        if self._log_handle is not None:
            try:
                self._log_handle.write(line + "\n")
                self._log_handle.flush()
            except OSError:
                pass

    def _pump(self):
        """Forward the child's output, then restart it if it asked for one."""
        assert self._proc is not None and self._proc.stdout is not None
        for raw in self._proc.stdout:
            self._emit(raw.rstrip("\r\n"))
        self._proc.wait()
        code = self._proc.returncode
        LOGGER.info("wgp.py exited with code %s", code)

        if code == RESTART_EXIT_CODE and not self._stopping.is_set():
            self._emit("[*] WanGP asked for a restart; relaunching ...")
            try:
                self._spawn()
            except OSError as exc:
                self._emit(f"[!] Restart failed: {exc}")
                self._exited.set()
            return

        if not self._stopping.is_set():
            self._emit(f"[!] WanGP stopped unexpectedly (exit code {code}).")
        self._exited.set()

    def _spawn(self):
        if self._log_handle is None:
            self._log_handle = open(self._log_path, "a", encoding="utf-8", errors="replace")
        self._proc = process.popen(self._argv(), cwd=self.root, env=process.child_env())
        self._reader = threading.Thread(target=self._pump, name="wangp-server-output", daemon=True)
        self._reader.start()

    def start(self):
        self._emit(f"[*] Starting WanGP on {self.url}")
        self._spawn()

    def wait_until_ready(self, cancelled=None):
        """Block until the web UI answers, the process dies, or we time out."""
        deadline = time.monotonic() + self.timeout
        announced = False
        while time.monotonic() < deadline:
            if cancelled is not None and cancelled():
                return False
            if _port_accepts_connections(self.host, self.port):
                self._emit("[*] Web interface is up.")
                return True
            if self._exited.is_set():
                return False
            if not announced and time.monotonic() > deadline - self.timeout + 20:
                announced = True
                self._emit("[*] Still starting: WanGP is loading its model list ...")
            time.sleep(0.5)
        self._emit(f"[!] WanGP did not answer within {self.timeout} seconds.")
        return False

    def is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        self._stopping.set()
        if self._proc is not None:
            self._emit("[*] Stopping WanGP ...")
            process.terminate(self._proc)
            self._proc = None
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=5)
        self._reader = None
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None
