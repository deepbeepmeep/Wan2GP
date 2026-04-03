from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


_DEBUG_ARG = "--debug-deepy"
DEBUG_DEEPY_ENABLED = False
DEBUG_DEEPY_LOG_PATH: Path | None = None
_BOOTSTRAPPED = False


class _TeeTextStream:
    def __init__(self, wrapped: TextIO, log_stream: TextIO):
        self._wrapped = wrapped
        self._log_stream = log_stream
        self.encoding = getattr(wrapped, "encoding", None)
        self.errors = getattr(wrapped, "errors", None)

    def write(self, data):
        written = self._wrapped.write(data)
        self._log_stream.write(data)
        return written

    def writelines(self, lines):
        self._wrapped.writelines(lines)
        self._log_stream.writelines(lines)

    def flush(self):
        self._wrapped.flush()
        self._log_stream.flush()

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


def _install_stream_tee(log_path: Path) -> None:
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeTextStream(sys.stdout, log_stream)
    sys.stderr = _TeeTextStream(sys.stderr, log_stream)


def bootstrap_deepy_debug() -> None:
    global _BOOTSTRAPPED, DEBUG_DEEPY_ENABLED, DEBUG_DEEPY_LOG_PATH
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True
    debug_dir = _find_debug_arg(list(sys.argv))
    if debug_dir is None:
        return
    sys.argv = _force_verbose_level(list(sys.argv))
    target_dir = _resolve_debug_dir(debug_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DEBUG_DEEPY_LOG_PATH = target_dir / f"debug_deepy_{stamp}.log"
    _install_stream_tee(DEBUG_DEEPY_LOG_PATH)
    DEBUG_DEEPY_ENABLED = True
    print(f"[DeepyDebug] Verbose level forced to 2. Logging to {DEBUG_DEEPY_LOG_PATH}")
