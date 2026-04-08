from __future__ import annotations

import atexit
import faulthandler
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

_LOCK = threading.RLock()
_STATE = {
    "installed": False,
    "log_file": None,
    "log_path": None,
    "stdout": None,
    "stderr": None,
    "prev_excepthook": None,
    "prev_threading_excepthook": None,
    "prev_unraisablehook": None,
}


class _TeeStream:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        if not data:
            return 0
        text = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
        with _LOCK:
            written = _safe_write(self._stream, text)
            _safe_write(self._log_file, text)
            if "\n" in text or "\r" in text:
                _safe_flush(self._stream)
                _safe_flush(self._log_file)
        return written if written is not None else len(text)

    def flush(self):
        with _LOCK:
            _safe_flush(self._stream)
            _safe_flush(self._log_file)

    def isatty(self):
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self):
        return getattr(self._stream, "fileno")()

    def writable(self):
        return bool(getattr(self._stream, "writable", lambda: True)())

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._stream, "errors", "strict")

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _safe_write(stream, text):
    if stream is None:
        return None
    try:
        return stream.write(text)
    except Exception:
        return None


def _safe_flush(stream):
    if stream is None:
        return
    try:
        stream.flush()
    except Exception:
        return


def _flush_log(sync=False):
    log_file = _STATE.get("log_file")
    if log_file is None:
        return
    _safe_flush(log_file)
    if sync:
        try:
            os.fsync(log_file.fileno())
        except Exception:
            return


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log_line(message, sync=False):
    log_file = _STATE.get("log_file")
    if log_file is None:
        return
    with _LOCK:
        _safe_write(log_file, f"[crash-diagnostics {_timestamp()}] {message}\n")
        _flush_log(sync=sync)


def _log_trace(prefix, exc_type, exc_value, exc_traceback):
    _log_line(prefix, sync=True)
    formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)).rstrip()
    if formatted:
        with _LOCK:
            _safe_write(_STATE.get("log_file"), f"{formatted}\n")
            _flush_log(sync=True)


def _install_stream_tees(log_file):
    if _STATE["stdout"] is None:
        _STATE["stdout"] = sys.stdout
    if _STATE["stderr"] is None:
        _STATE["stderr"] = sys.stderr
    if sys.stdout is _STATE["stdout"]:
        sys.stdout = _TeeStream(sys.stdout, log_file)
    if sys.stderr is _STATE["stderr"]:
        sys.stderr = _TeeStream(sys.stderr, log_file)


def _install_exception_hooks():
    prev_excepthook = sys.excepthook
    _STATE["prev_excepthook"] = prev_excepthook

    def _excepthook(exc_type, exc_value, exc_traceback):
        _log_trace(f"Unhandled exception in main thread '{threading.current_thread().name}'", exc_type, exc_value, exc_traceback)
        if prev_excepthook not in (None, _excepthook):
            try:
                prev_excepthook(exc_type, exc_value, exc_traceback)
                return
            except Exception:
                pass
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _excepthook

    if hasattr(threading, "excepthook"):
        prev_threading_excepthook = threading.excepthook
        _STATE["prev_threading_excepthook"] = prev_threading_excepthook

        def _threading_excepthook(args):
            thread_name = getattr(args.thread, "name", "unknown")
            _log_trace(f"Unhandled exception in thread '{thread_name}'", args.exc_type, args.exc_value, args.exc_traceback)
            if prev_threading_excepthook not in (None, _threading_excepthook):
                try:
                    prev_threading_excepthook(args)
                except Exception:
                    return

        threading.excepthook = _threading_excepthook

    if hasattr(sys, "unraisablehook"):
        prev_unraisablehook = sys.unraisablehook
        _STATE["prev_unraisablehook"] = prev_unraisablehook

        def _unraisablehook(unraisable):
            obj_name = type(unraisable.object).__name__ if getattr(unraisable, "object", None) is not None else "None"
            err_msg = getattr(unraisable, "err_msg", None) or "Unraisable exception"
            _log_line(f"{err_msg} on object type '{obj_name}'", sync=True)
            _log_trace("Unraisable exception traceback", unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)
            if prev_unraisablehook not in (None, _unraisablehook):
                try:
                    prev_unraisablehook(unraisable)
                except Exception:
                    return

        sys.unraisablehook = _unraisablehook


def _install_faulthandler(log_file):
    try:
        faulthandler.enable(file=log_file, all_threads=True)
        _log_line("faulthandler enabled for fatal native crashes", sync=True)
    except Exception as exc:
        _log_line(f"Failed to enable faulthandler: {exc}", sync=True)


def _log_startup_context(anchor_path):
    _log_line(f"Crash diagnostics active. Log file: {_STATE['log_path']}", sync=True)
    _log_line(f"PID={os.getpid()} PPID={os.getppid() if hasattr(os, 'getppid') else 'n/a'}", sync=True)
    _log_line(f"Python={sys.version.splitlines()[0]}", sync=True)
    _log_line(f"Platform={platform.platform()}", sync=True)
    _log_line(f"CWD={os.getcwd()}", sync=True)
    _log_line(f"Entry={anchor_path}", sync=True)
    _log_line(f"ARGV={sys.argv}", sync=True)


def _on_exit():
    _log_line("Process exit reached", sync=True)


def install_wgp_crash_diagnostics(anchor_file):
    try:
        with _LOCK:
            if _STATE["installed"]:
                return str(_STATE["log_path"] or "")

            anchor_path = Path(anchor_file).resolve()
            log_dir = anchor_path.parent / "crash"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"wgp_crash_{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}.log"
            log_file = log_path.open("a", encoding="utf-8", buffering=1)
            _STATE["log_file"] = log_file
            _STATE["log_path"] = log_path

            _install_stream_tees(log_file)
            _install_exception_hooks()
            _install_faulthandler(log_file)
            atexit.register(_on_exit)
            _STATE["installed"] = True

        _log_startup_context(anchor_path)
        print(f"[crash-diagnostics] Logging to {log_path}")
        return str(log_path)
    except Exception:
        return ""
