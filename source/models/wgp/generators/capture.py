"""Output capture infrastructure for WGP generation calls.

Provides stdout/stderr/logging capture so that WGP output can be inspected
for error extraction when generation silently fails.
"""

import sys
import logging as py_logging
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple


class TailBuffer:
    """Ring-buffer that keeps only the last *max_chars* characters."""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self._buf = ""

    def write(self, text: str):
        if not text:
            return
        try:
            self._buf += str(text)
            if len(self._buf) > self.max_chars:
                self._buf = self._buf[-self.max_chars:]
        except (TypeError, ValueError, MemoryError):
            # Never let logging capture break generation
            pass

    def getvalue(self) -> str:
        return self._buf


class LoggingTailBuffer(TailBuffer):
    """Ring-buffer that ALSO logs output to the logging system in real-time (only in debug mode)."""

    def __init__(self, max_chars: int, task_id: Optional[str] = None, log_func: Optional[Callable] = None):
        super().__init__(max_chars)
        self.task_id = task_id
        self.log_func = log_func  # Function to call to log (e.g., generation_logger.debug)
        self._line_buffer = ""  # Buffer lines until newline to avoid log spam
        self._debug_mode = False

        # Check if debug mode is enabled
        try:
            from source.core.log.api import is_debug_enabled
            self._debug_mode = is_debug_enabled()
        except (ImportError, AttributeError):
            pass

    def write(self, text: str):
        if not text:
            return
        try:
            # Write to memory buffer (always)
            super().write(text)

            # Stream to logger only if debug mode is enabled (to avoid log spam in production)
            if self._debug_mode and self.task_id and self.log_func:
                self._line_buffer += str(text)
                # Log complete lines to avoid cluttering logs with partial output
                while '\n' in self._line_buffer:
                    line, self._line_buffer = self._line_buffer.split('\n', 1)
                    if line.strip():  # Only log non-empty lines
                        try:
                            self.log_func(f"[WGP_STREAM] {line}", task_id=self.task_id)
                        except (TypeError, OSError, RuntimeError, AttributeError):
                            # If logging fails, silently continue - don't break generation
                            pass
        except (TypeError, ValueError, MemoryError):
            # Never let logging capture break generation
            pass


class TeeWriter:
    """Tee stdout/stderr: capture while still printing to console.

    Proxies common file-like attributes (encoding/isatty/fileno/etc.)
    to avoid breaking libraries that inspect the stream object.
    """

    def __init__(self, original, capture):
        self._original = original
        self._capture = capture

    def write(self, text):
        try:
            self._original.write(text)
        except (OSError, ValueError):
            pass
        try:
            self._capture.write(text)
        except (OSError, ValueError):
            pass

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        try:
            self._original.flush()
        except (OSError, ValueError):
            pass

    def isatty(self):
        try:
            return self._original.isatty()
        except (OSError, ValueError):
            return False

    def fileno(self):
        return self._original.fileno()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", None)

    def __getattr__(self, name):
        # Proxy everything else to the underlying stream
        return getattr(self._original, name)


class CaptureHandler(py_logging.Handler):
    """Logging handler that stores recent records in a deque for later inspection."""

    def __init__(self, log_deque: Deque):
        super().__init__(level=py_logging.DEBUG)
        self._log_deque = log_deque
        self._dedupe: Deque = deque(maxlen=200)  # avoid spamming duplicates

    def emit(self, record):
        try:
            msg = self.format(record)
            key = (record.levelname, record.name, msg)
            if key in self._dedupe:
                return
            self._dedupe.append(key)
            self._log_deque.append({
                "level": record.levelname,
                "name": record.name,
                "message": msg,
            })
        except (ValueError, TypeError, RuntimeError):
            pass


# ---------------------------------------------------------------------------
# Default capture sizes
# ---------------------------------------------------------------------------
_CAPTURE_STDOUT_CHARS = 20_000
_CAPTURE_STDERR_CHARS = 20_000
_CAPTURE_PYLOG_RECORDS = 1000  # keep last N log records (all levels)


def run_with_capture(
    fn: Callable[..., Any],
    task_id: Optional[str] = None,
    log_func: Optional[Callable] = None,
    **kwargs,
) -> Tuple[Optional[Any], TailBuffer, TailBuffer, Deque[Dict[str, str]]]:
    """Execute *fn* while capturing stdout, stderr, and Python logging.

    NOTE: This monkeypatches sys.stdout/sys.stderr for the duration of the
    call.  That is process-global and can capture output from other threads.
    In this repo, generation is effectively single-task-at-a-time per worker
    process, so this is acceptable.

    If *fn* raises an exception, it is re-raised after capture cleanup,
    but the captured buffers are still available via the ``captured_stdout``,
    ``captured_stderr``, and ``captured_logs`` attributes on the exception
    object (attached as ``__captured_stdout__`` etc.).

    Args:
        fn: The function to execute
        task_id: Optional task ID for logging context
        log_func: Optional logging function (e.g., generation_logger.debug) to stream output
        **kwargs: Arguments to pass to fn

    Returns:
        (return_value, captured_stdout, captured_stderr, captured_logs)
        *return_value* is whatever *fn* returns.

    Raises:
        Any exception raised by *fn*, with captured output attached.
    """
    # Use logging-aware buffers if a log function is provided
    if log_func:
        captured_stdout = LoggingTailBuffer(_CAPTURE_STDOUT_CHARS, task_id=task_id, log_func=log_func)
        captured_stderr = LoggingTailBuffer(_CAPTURE_STDERR_CHARS, task_id=task_id, log_func=log_func)
    else:
        captured_stdout = TailBuffer(_CAPTURE_STDOUT_CHARS)
        captured_stderr = TailBuffer(_CAPTURE_STDERR_CHARS)
    captured_logs: Deque[Dict[str, str]] = deque(maxlen=_CAPTURE_PYLOG_RECORDS)

    # Set up capture handler on root logger + non-propagating library loggers
    capture_handler = CaptureHandler(captured_logs)
    capture_handler.setFormatter(py_logging.Formatter("%(levelname)s:%(name)s: %(message)s"))

    loggers_to_capture = [py_logging.getLogger()]

    # Preserve library logger thresholds set by suppress_library_logging().
    for logger in loggers_to_capture:
        logger.addHandler(capture_handler)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    caught_exc: Optional[BaseException] = None
    result = None

    try:
        sys.stdout = TeeWriter(original_stdout, captured_stdout)  # type: ignore[assignment]
        sys.stderr = TeeWriter(original_stderr, captured_stderr)  # type: ignore[assignment]

        result = fn(**kwargs)
    except BaseException as e:
        caught_exc = e
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Remove capture handler without overriding configured logger levels.
        for logger in loggers_to_capture:
            logger.removeHandler(capture_handler)

    if caught_exc is not None:
        # Attach captured output to the exception so callers can inspect it
        caught_exc.__captured_stdout__ = captured_stdout  # type: ignore[attr-defined]
        caught_exc.__captured_stderr__ = captured_stderr  # type: ignore[attr-defined]
        caught_exc.__captured_logs__ = captured_logs  # type: ignore[attr-defined]
        raise caught_exc

    return result, captured_stdout, captured_stderr, captured_logs
