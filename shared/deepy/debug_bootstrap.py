from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


_DEBUG_ARG = "--debug-deepy"
DEBUG_DEEPY_ENABLED = False
DEBUG_DEEPY_LOG_PATH: Path | None = None
_BOOTSTRAPPED = False
_STREAMS_WRAPPED = False
_DEBUG_TARGET_DIR: Path | None = None
_LOG_STREAM: TextIO | None = None
_LOG_LOCK = threading.RLock()
_THREAD_STATE = threading.local()
_EXTERNAL_CAPTURE_DEPTH = 0
_START_NOTICE_EMITTED = False


class _SelectiveTeeTextStream:
    def __init__(self, wrapped: TextIO):
        self._wrapped = wrapped
        self.encoding = getattr(wrapped, "encoding", None)
        self.errors = getattr(wrapped, "errors", None)

    def write(self, data):
        written = self._wrapped.write(data)
        _maybe_write_log(data)
        return written

    def writelines(self, lines):
        for line in list(lines or []):
            self.write(line)

    def flush(self):
        self._wrapped.flush()
        log_stream = _LOG_STREAM
        if log_stream is not None:
            log_stream.flush()

    def isatty(self):
        return bool(getattr(self._wrapped, "isatty", lambda: False)())

    def fileno(self):
        return self._wrapped.fileno()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _find_debug_arg(argv: list[str]) -> str | None:
    debug_dir = None
    i = 1
    while i < len(argv):
        arg = str(argv[i])
        if arg == _DEBUG_ARG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{_DEBUG_ARG} requires a folder path.")
            debug_dir = str(argv[i + 1])
            i += 2
            continue
        if arg.startswith(f"{_DEBUG_ARG}="):
            debug_dir = arg.split("=", 1)[1]
            if not debug_dir:
                raise SystemExit(f"{_DEBUG_ARG} requires a folder path.")
        i += 1
    return debug_dir


def _force_verbose_level(argv: list[str], level: str = "2") -> list[str]:
    rewritten = [argv[0]]
    verbose_seen = False
    i = 1
    while i < len(argv):
        arg = str(argv[i])
        if arg == "--verbose":
            verbose_seen = True
            rewritten.extend(["--verbose", level])
            has_value = i + 1 < len(argv) and not str(argv[i + 1]).startswith("-")
            i += 2 if has_value else 1
            continue
        if arg.startswith("--verbose="):
            verbose_seen = True
            rewritten.append(f"--verbose={level}")
            i += 1
            continue
        rewritten.append(arg)
        i += 1
    if not verbose_seen:
        rewritten.extend(["--verbose", level])
    return rewritten


def _resolve_debug_dir(raw_dir: str) -> Path:
    path = Path.cwd() if raw_dir == "." else Path(raw_dir).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve(strict=False)
    if path.exists() and not path.is_dir():
        raise SystemExit(f"{_DEBUG_ARG} path must be a folder: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unwrap_stream(stream: TextIO) -> TextIO:
    return getattr(stream, "_wrapped", stream)


def _install_stream_tee() -> None:
    global _STREAMS_WRAPPED
    if _STREAMS_WRAPPED:
        return
    sys.stdout = _SelectiveTeeTextStream(sys.stdout)
    sys.stderr = _SelectiveTeeTextStream(sys.stderr)
    _STREAMS_WRAPPED = True


def _is_deepy_capture_active() -> bool:
    return int(getattr(_THREAD_STATE, "deepy_log_depth", 0) or 0) > 0


def _should_log_current_write() -> bool:
    return _LOG_STREAM is not None and (_EXTERNAL_CAPTURE_DEPTH > 0 or _is_deepy_capture_active())


def _maybe_write_log(data: str | None) -> None:
    if not data or not _should_log_current_write():
        return
    with _LOG_LOCK:
        if _LOG_STREAM is None:
            return
        _LOG_STREAM.write(data)


def _emit_start_notice() -> None:
    global _START_NOTICE_EMITTED
    if _START_NOTICE_EMITTED or _LOG_STREAM is None or DEBUG_DEEPY_LOG_PATH is None:
        return
    notice = f"[DeepyDebug] Verbose level forced to 2. Logging to {DEBUG_DEEPY_LOG_PATH}\n"
    with _LOG_LOCK:
        _unwrap_stream(sys.stdout).write(notice)
        _unwrap_stream(sys.stdout).flush()
        if _LOG_STREAM is not None:
            _LOG_STREAM.write(notice)
            _LOG_STREAM.flush()
    _START_NOTICE_EMITTED = True


def bootstrap_deepy_debug() -> None:
    global _BOOTSTRAPPED, _DEBUG_TARGET_DIR
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True
    debug_dir = _find_debug_arg(list(sys.argv))
    if debug_dir is None:
        return
    sys.argv = _force_verbose_level(list(sys.argv))
    _DEBUG_TARGET_DIR = _resolve_debug_dir(debug_dir)
    _install_stream_tee()


def ensure_deepy_debug_started() -> bool:
    global DEBUG_DEEPY_ENABLED, DEBUG_DEEPY_LOG_PATH, _LOG_STREAM
    if _DEBUG_TARGET_DIR is None:
        return False
    if _LOG_STREAM is not None:
        return True
    with _LOG_LOCK:
        if _LOG_STREAM is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            DEBUG_DEEPY_LOG_PATH = _DEBUG_TARGET_DIR / f"debug_deepy_{stamp}.log"
            _LOG_STREAM = DEBUG_DEEPY_LOG_PATH.open("a", encoding="utf-8", buffering=1)
            DEBUG_DEEPY_ENABLED = True
    _emit_start_notice()
    return True


@contextmanager
def deepy_log_scope(*, start_if_needed: bool = False) -> Iterator[None]:
    enabled = ensure_deepy_debug_started() if start_if_needed else _LOG_STREAM is not None
    if not enabled:
        yield
        return
    depth = int(getattr(_THREAD_STATE, "deepy_log_depth", 0) or 0)
    _THREAD_STATE.deepy_log_depth = depth + 1
    try:
        yield
    finally:
        if depth <= 0:
            if hasattr(_THREAD_STATE, "deepy_log_depth"):
                delattr(_THREAD_STATE, "deepy_log_depth")
        else:
            _THREAD_STATE.deepy_log_depth = depth


def deepy_print(*args, **kwargs) -> None:
    with deepy_log_scope():
        print(*args, **kwargs)


@contextmanager
def capture_external_logs() -> Iterator[None]:
    global _EXTERNAL_CAPTURE_DEPTH
    if not ensure_deepy_debug_started():
        yield
        return
    _EXTERNAL_CAPTURE_DEPTH += 1
    try:
        yield
    finally:
        _EXTERNAL_CAPTURE_DEPTH = max(0, _EXTERNAL_CAPTURE_DEPTH - 1)
