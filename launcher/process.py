"""Subprocess helpers: run child processes windowless and stream their output."""

import logging
import os
import subprocess
import sys

LOGGER = logging.getLogger(__name__)

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010

# setup.py prompts with input() when several environments exist. We hand every
# child a stdin full of newlines so those prompts resolve to their default
# instead of raising EOFError.
_AUTO_ANSWER = "\n" * 32


def _creation_flags(visible_console):
    if os.name != "nt":
        return 0
    return CREATE_NEW_CONSOLE if visible_console else CREATE_NO_WINDOW


def popen(argv, cwd=None, env=None, visible_console=False, capture=True):
    kwargs = {
        "cwd": cwd,
        "env": env,
        "creationflags": _creation_flags(visible_console),
    }
    if capture:
        kwargs.update(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    LOGGER.info("Running: %s (cwd=%s)", subprocess.list2cmdline(argv), cwd)
    return subprocess.Popen(argv, **kwargs)


def stream(argv, on_line, cwd=None, env=None, cancelled=None):
    """Run ``argv``, forwarding every output line to ``on_line``.

    Returns the process exit code, or -1 if ``cancelled()`` asked us to stop.
    """
    proc = popen(argv, cwd=cwd, env=env)
    try:
        if proc.stdin is not None:
            try:
                proc.stdin.write(_AUTO_ANSWER)
                proc.stdin.flush()
            except OSError:
                pass
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if line:
                on_line(line)
            if cancelled is not None and cancelled():
                terminate(proc)
                return -1
        proc.wait()
        return proc.returncode
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass


def run_quiet(argv, cwd=None, timeout=30):
    """Run a short command and return (returncode, combined output)."""
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_creation_flags(False),
        )
        return completed.returncode, (completed.stdout or "").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)


def terminate(proc):
    """Stop a process and the console child processes it spawned."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        # setup.py and wgp.py both spawn grandchildren (pip, ffmpeg, workers);
        # taskkill /T is the only reliable way to take the whole tree down.
        run_quiet(["taskkill", "/F", "/T", "/PID", str(proc.pid)], timeout=20)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def open_in_explorer(path):
    if not os.path.exists(path):
        return
    if os.name == "nt":
        os.startfile(path)  # noqa: S606 - documented Windows shell open
    else:
        subprocess.Popen(["xdg-open", path])


def open_console(argv, cwd=None):
    """Launch an interactive command in its own visible console window."""
    return popen(argv, cwd=cwd, visible_console=True, capture=False)


def child_env(extra=None):
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # PyInstaller leaks its bootstrap paths into children; strip them so the
    # WanGP environment's own interpreter resolves its packages normally.
    for key in ("PYTHONHOME", "PYTHONPATH", "_MEIPASS2"):
        env.pop(key, None)
    if extra:
        env.update(extra)
    return env


def bundled_python():
    """The interpreter to use for plain-stdlib chores such as setup.py."""
    if getattr(sys, "frozen", False):
        return None
    return sys.executable
