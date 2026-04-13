"""Shared subprocess helpers with normalized binary resolution and tracking."""

from __future__ import annotations

import os
import shutil
import subprocess

CalledProcessError = subprocess.CalledProcessError
TimeoutExpired = subprocess.TimeoutExpired


def _normalize_command(command):
    if not command:
        raise ValueError("empty command")
    normalized = list(command)
    resolved = shutil.which(normalized[0])
    if resolved:
        normalized[0] = resolved
    return normalized


def _get_subprocess_tracker():
    """Return (register, deregister) callables if the shutdown diagnostic
    tracker has been installed by server.main(), or (None, None) if not yet."""
    # Use sys.modules to avoid circular imports — server.py imports modules that
    # import subprocess_utils, so we can't do a normal import.
    import sys as _sys
    _srv_mod = _sys.modules.get("source.runtime.worker.server")
    if _srv_mod is None:
        return None, None
    # The functions are stashed in the module globals() dict by server.main()
    g = vars(_srv_mod)
    reg = g.get("_register_subprocess")
    dereg = g.get("_deregister_subprocess")
    return reg, dereg


def _default_creation_flags():
    """On Windows, return CREATE_NEW_PROCESS_GROUP so child processes do NOT
    share our console Ctrl-C group.  This prevents a child crash from sending
    CTRL_C_EVENT to the worker."""
    if os.name != "nt":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}


def run_subprocess(command, timeout=None, **kwargs):
    # Inject Windows creation flags unless the caller already specified them
    if "creationflags" not in kwargs:
        kwargs.update(_default_creation_flags())
    normalized = _normalize_command(command)
    cmd_str = " ".join(str(c) for c in normalized[:6])
    reg, dereg = _get_subprocess_tracker()
    # For run(), we don't get the PID until after it finishes, so track
    # as a "transient" entry.
    result = subprocess.run(normalized, timeout=timeout, **kwargs)
    # Log the exit for any completed subprocess (pid not available from run())
    return result


def popen_subprocess(command, **kwargs):
    if "creationflags" not in kwargs:
        kwargs.update(_default_creation_flags())
    normalized = _normalize_command(command)
    cmd_str = " ".join(str(c) for c in normalized[:6])
    proc = subprocess.Popen(normalized, **kwargs)
    reg, _dereg = _get_subprocess_tracker()
    if reg:
        reg(proc.pid, cmd_str)
    return _TrackedPopen(proc, cmd_str)


class _TrackedPopen:
    """Thin wrapper around Popen that deregisters from the subprocess tracker
    when the process exits (via wait/poll/communicate)."""

    def __init__(self, proc: subprocess.Popen, cmd_str: str):
        self._proc = proc
        self._cmd_str = cmd_str
        self._deregistered = False

    def _maybe_deregister(self):
        if self._deregistered:
            return
        rc = self._proc.returncode
        if rc is not None:
            self._deregistered = True
            _, dereg = _get_subprocess_tracker()
            if dereg:
                dereg(self._proc.pid, rc)

    # Delegate everything to the inner Popen
    def __getattr__(self, name):
        return getattr(self._proc, name)

    def wait(self, *args, **kwargs):
        result = self._proc.wait(*args, **kwargs)
        self._maybe_deregister()
        return result

    def poll(self):
        result = self._proc.poll()
        self._maybe_deregister()
        return result

    def communicate(self, *args, **kwargs):
        result = self._proc.communicate(*args, **kwargs)
        self._maybe_deregister()
        return result

    def kill(self):
        self._proc.kill()

    def terminate(self):
        self._proc.terminate()

    @property
    def pid(self):
        return self._proc.pid

    @property
    def returncode(self):
        return self._proc.returncode

    @property
    def stdin(self):
        return self._proc.stdin

    @property
    def stdout(self):
        return self._proc.stdout

    @property
    def stderr(self):
        return self._proc.stderr


__all__ = [
    "CalledProcessError",
    "TimeoutExpired",
    "_normalize_command",
    "popen_subprocess",
    "run_subprocess",
]
