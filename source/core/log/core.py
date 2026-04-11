"""
Centralized logging utility for Headless-Wan2GP

Provides structured logging with debug vs essential log levels.
Essential logs are always shown, debug logs only appear when debug mode is enabled.
"""

import datetime
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Mapping, Optional

from source.core.log.debug_card import DebugCard, format_value

__all__ = [
    "set_log_file",
    "enable_debug_mode",
    "disable_debug_mode",
    "init_from_env",
    "is_debug_enabled",
    "suppress_library_logging",
    "essential",
    "success",
    "warning",
    "error",
    "critical",
    "debug",
    "debug_block",
    "progress",
    "status",
    "ComponentLogger",
    "headless_logger",
    "queue_logger",
    "orchestrator_logger",
    "travel_logger",
    "generation_logger",
    "model_logger",
    "task_logger",
    "set_log_interceptor",
    "set_current_task_context",
    "flush_log_buffer",
]

# Global debug mode flag - set by the main application
_debug_mode = False
# Global log file handle
_log_file = None
_log_file_lock = None

def set_log_file(path: str):
    """Set a file path to mirror all logs to."""
    global _log_file, _log_file_lock
    import threading
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(path, 'a', encoding='utf-8')
        _log_file_lock = threading.Lock()
        essential("LOGGING", f"Logging to file enabled: {path}")
    except OSError as e:
        error("LOGGING", f"Failed to set log file {path}: {e}")

def _write_to_log_file(formatted_message: str):
    """Write message to log file if enabled."""
    global _log_file, _log_file_lock
    if _log_file and _log_file_lock:
        try:
            with _log_file_lock:
                _log_file.write(formatted_message + "\n")
                _log_file.flush()
        except OSError as e:
            # Log file write failed (disk full, file closed, etc.)
            # Print to stderr as a last resort since our own logging infrastructure is broken
            print(f"[core.log] Failed to write to log file: {e}", file=sys.stderr)

def enable_debug_mode():
    """Enable debug logging globally."""
    global _debug_mode
    _debug_mode = True

def _is_env_debug() -> bool:
    """Return True when REIGH_DEBUG requests debug logging."""
    return os.environ.get("REIGH_DEBUG", "").strip().lower() in ("1", "true", "yes")

def init_from_env():
    """Enable debug mode when requested by environment configuration."""
    if _is_env_debug():
        enable_debug_mode()

def suppress_library_logging():
    """Reduce noisy third-party Python logging for non-debug runs."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    for logger_name in ("diffusers", "transformers", "torch", "PIL", "mmgp", "triton", "httpx"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def disable_debug_mode():
    """Disable debug logging globally."""
    global _debug_mode
    _debug_mode = False

def is_debug_enabled() -> bool:
    """Check if debug mode is currently enabled."""
    return _debug_mode

def _get_timestamp() -> str:
    """Get formatted timestamp for logs."""
    return datetime.datetime.now().strftime("%H:%M:%S")

def _format_message(
    level: str,
    component: str,
    message: str,
    task_id: Optional[str] = None,
    *,
    include_task_prefix: bool = True,
) -> str:
    """Format a log message with consistent structure."""
    timestamp = _get_timestamp()

    if task_id and include_task_prefix:
        return f"[{timestamp}] {level} {component} [Task {task_id}] {message}"
    return f"[{timestamp}] {level} {component} {message}"

def _append_exc_info(formatted: str, exc_info: bool) -> str:
    """Append traceback to formatted message if exc_info=True and an exception is active."""
    if exc_info:
        exc_text = traceback.format_exc()
        if exc_text and exc_text.strip() != "NoneType: None":
            formatted = f"{formatted}\n{exc_text.rstrip()}"
    return formatted


def _format_debug_block_message(stage: str, items: Mapping[str, object]) -> str:
    """Format a structured debug block using aligned DebugCard rendering."""
    if not items:
        return stage

    if len(items) == 1:
        key, value = next(iter(items.items()))
        return f"{stage} {key}={format_value(value)}"

    card = DebugCard(stage)
    for key, value in items.items():
        card.row(key, value)
    return card.render()

def essential(
    component: str,
    message: str,
    task_id: Optional[str] = None,
    exc_info: bool = False,
    *,
    include_task_prefix: bool = True,
):
    """Log an essential message that should always be shown."""
    formatted = _append_exc_info(
        _format_message("INFO", component, message, task_id, include_task_prefix=include_task_prefix),
        exc_info,
    )
    print(formatted)
    _write_to_log_file(formatted)

def success(
    component: str,
    message: str,
    task_id: Optional[str] = None,
    exc_info: bool = False,
    *,
    include_task_prefix: bool = True,
):
    """Log a success message that should always be shown.

    The level column uses plain `INFO` for visual consistency with other log lines.
    If you want a visible success marker (like `✓`), put it in the message content —
    the task lifecycle emitter does this for the `✓ Task done` anchor line.
    """
    formatted = _append_exc_info(
        _format_message("INFO", component, message, task_id, include_task_prefix=include_task_prefix),
        exc_info,
    )
    print(formatted)
    _write_to_log_file(formatted)

def warning(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a warning message that should always be shown.

    Uses plain `WARN` in the level column for visual consistency.
    """
    formatted = _append_exc_info(_format_message("WARN", component, message, task_id), exc_info)
    print(formatted)
    _write_to_log_file(formatted)

def error(
    component: str,
    message: str,
    task_id: Optional[str] = None,
    exc_info: bool = False,
    *,
    include_task_prefix: bool = True,
):
    """Log an error message that should always be shown."""
    formatted = _append_exc_info(
        _format_message("ERROR", component, message, task_id, include_task_prefix=include_task_prefix),
        exc_info,
    )
    print(formatted, file=sys.stderr)
    _write_to_log_file(formatted)

def critical(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a critical/fatal error message that should always be shown."""
    formatted = _append_exc_info(_format_message("FATAL", component, message, task_id), exc_info)
    print(formatted, file=sys.stderr)
    _write_to_log_file(formatted)

def debug(
    component: str,
    message: str,
    task_id: Optional[str] = None,
    exc_info: bool = False,
    *,
    include_task_prefix: bool = True,
):
    """Log a debug message that only appears when debug mode is enabled."""
    if _debug_mode:
        formatted = _append_exc_info(
            _format_message("DEBUG", component, message, task_id, include_task_prefix=include_task_prefix),
            exc_info,
        )
        print(formatted)
        _write_to_log_file(formatted)


def debug_block(component: str, stage: str, items: Mapping[str, object], task_id: Optional[str] = None):
    """Log a structured debug block for a chokepoint when debug mode is enabled.

    The card's header line does NOT carry the `[Task X]` prefix — the prefix would
    dominate the ──── STAGE ──── title visually, and card bodies already carry
    enough context (task_id is still sent to the DB interceptor below).
    """
    if _debug_mode:
        message = _format_debug_block_message(stage, items)
        formatted = _format_message("DEBUG", component, message, task_id, include_task_prefix=False)
        print(formatted)
        _write_to_log_file(formatted)
        _intercept_log("DEBUG", f"{component}: {message}", task_id)


def debug_anomaly(component: str, stage: str, message: str, task_id: Optional[str] = None):
    """Log an anomaly-focused debug message when debug mode is enabled."""
    if _debug_mode:
        anomaly_message = f"  \u26a1 {stage}: {message}"
        formatted = _format_message("DEBUG", component, anomaly_message, task_id)
        print(formatted)
        _write_to_log_file(formatted)
        _intercept_log("DEBUG", f"{component}: {anomaly_message}", task_id)

def progress(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a progress message that should always be shown."""
    formatted = _append_exc_info(_format_message("\u23f3", component, message, task_id), exc_info)
    print(formatted)
    _write_to_log_file(formatted)

def status(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a status message that should always be shown."""
    formatted = _append_exc_info(_format_message("\U0001f4ca", component, message, task_id), exc_info)
    print(formatted)
    _write_to_log_file(formatted)

# Component-specific loggers for better organization
class ComponentLogger:
    """Logger for a specific component with consistent naming."""

    def __init__(self, component_name: str):
        self.component = component_name

    def essential(
        self,
        message: str,
        task_id: Optional[str] = None,
        exc_info: bool = False,
        *,
        include_task_prefix: bool = True,
    ):
        essential(self.component, message, task_id, exc_info=exc_info, include_task_prefix=include_task_prefix)

    def success(
        self,
        message: str,
        task_id: Optional[str] = None,
        exc_info: bool = False,
        *,
        include_task_prefix: bool = True,
    ):
        success(self.component, message, task_id, exc_info=exc_info, include_task_prefix=include_task_prefix)

    def warning(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        warning(self.component, message, task_id, exc_info=exc_info)

    def error(
        self,
        message: str,
        task_id: Optional[str] = None,
        exc_info: bool = False,
        *,
        include_task_prefix: bool = True,
    ):
        error(self.component, message, task_id, exc_info=exc_info, include_task_prefix=include_task_prefix)

    def critical(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        critical(self.component, message, task_id, exc_info=exc_info)

    def debug(
        self,
        message: str,
        task_id: Optional[str] = None,
        exc_info: bool = False,
        *,
        include_task_prefix: bool = True,
    ):
        debug(self.component, message, task_id, exc_info=exc_info, include_task_prefix=include_task_prefix)

    def debug_block(self, stage: str, items: Mapping[str, object], task_id: Optional[str] = None):
        debug_block(self.component, stage, items, task_id)

    def debug_anomaly(self, stage: str, message: str, task_id: Optional[str] = None):
        debug_anomaly(self.component, stage, message, task_id)

    def progress(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        progress(self.component, message, task_id, exc_info=exc_info)

    def status(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        status(self.component, message, task_id, exc_info=exc_info)

    def info(
        self,
        message: str,
        task_id: Optional[str] = None,
        exc_info: bool = False,
        *,
        include_task_prefix: bool = True,
    ):
        """Alias for essential() to maintain compatibility with standard logging."""
        essential(self.component, message, task_id, exc_info=exc_info, include_task_prefix=include_task_prefix)

# Pre-configured loggers for main components
headless_logger = ComponentLogger("HEADLESS")
queue_logger = ComponentLogger("QUEUE")
orchestrator_logger = ComponentLogger("ORCHESTRATOR")
travel_logger = ComponentLogger("TRAVEL")
generation_logger = ComponentLogger("GENERATION")
model_logger = ComponentLogger("MODEL")
task_logger = ComponentLogger("TASK")

# -----------------------------------------------------------------------------
# Interceptor globals and redefinitions
# -----------------------------------------------------------------------------

from source.core.log.database import CustomLogInterceptor

# Global log interceptor instance (set in worker.py)
_log_interceptor: Optional[CustomLogInterceptor] = None


def set_log_interceptor(interceptor: Optional[CustomLogInterceptor]):
    """Set the global log interceptor for database logging."""
    global _log_interceptor
    _log_interceptor = interceptor


def flush_log_buffer():
    """Flush buffered logs to ensure they reach the database.

    Call this in exception handlers before re-raising, so crash
    diagnostics aren't lost in the buffer.
    """
    if _log_interceptor:
        _log_interceptor.log_buffer.flush()


def set_current_task_context(task_id: Optional[str]):
    """
    Set/clear the task context used for associating intercepted logs with a task_id.

    This is intentionally a thin wrapper around the active interceptor instance so that
    worker threads (e.g. the headless generation queue) can correctly tag logs even if
    the main polling thread is busy or if the queue is multi-threaded.
    """
    if _log_interceptor:
        try:
            _log_interceptor.set_current_task(task_id)
        except (ValueError, TypeError, OSError) as e:
            # Interceptor failed to set task context - log to stderr to avoid recursion
            print(f"[core.log] Failed to set task context on log interceptor: {e}", file=sys.stderr)


# Update logging functions to use interceptor
def _intercept_log(level: str, message: str, task_id: Optional[str] = None):
    """Send log to interceptor if enabled."""
    if _log_interceptor:
        _log_interceptor.capture_log(level, message, task_id)


# Modify existing logging functions to intercept
_original_essential = essential
def essential(
    component: str,
    message: str,
    task_id: Optional[str] = None,
    exc_info: bool = False,
    *,
    include_task_prefix: bool = True,
):
    """Log an essential message that should always be shown."""
    _original_essential(
        component,
        message,
        task_id,
        exc_info=exc_info,
        include_task_prefix=include_task_prefix,
    )
    _intercept_log("INFO", f"{component}: {message}", task_id)


_original_success = success
def success(
    component: str,
    message: str,
    task_id: Optional[str] = None,
    exc_info: bool = False,
    *,
    include_task_prefix: bool = True,
):
    """Log a success message that should always be shown."""
    _original_success(
        component,
        message,
        task_id,
        exc_info=exc_info,
        include_task_prefix=include_task_prefix,
    )
    _intercept_log("INFO", f"{component}: {message}", task_id)


_original_warning = warning
def warning(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a warning message that should always be shown."""
    _original_warning(component, message, task_id, exc_info=exc_info)
    _intercept_log("WARNING", f"{component}: {message}", task_id)


_original_error = error
def error(
    component: str,
    message: str,
    task_id: Optional[str] = None,
    exc_info: bool = False,
    *,
    include_task_prefix: bool = True,
):
    """Log an error message that should always be shown."""
    _original_error(
        component,
        message,
        task_id,
        exc_info=exc_info,
        include_task_prefix=include_task_prefix,
    )
    _intercept_log("ERROR", f"{component}: {message}", task_id)


_original_critical = critical
def critical(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a critical/fatal error message that should always be shown."""
    _original_critical(component, message, task_id, exc_info=exc_info)
    _intercept_log("ERROR", f"{component}: {message}", task_id)


_original_debug = debug
def debug(
    component: str,
    message: str,
    task_id: Optional[str] = None,
    exc_info: bool = False,
    *,
    include_task_prefix: bool = True,
):
    """Log a debug message that only appears when debug mode is enabled."""
    _original_debug(component, message, task_id, exc_info=exc_info, include_task_prefix=include_task_prefix)
    if _debug_mode:
        _intercept_log("DEBUG", f"{component}: {message}", task_id)


_original_progress = progress
def progress(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a progress message that should always be shown."""
    _original_progress(component, message, task_id, exc_info=exc_info)
    _intercept_log("INFO", f"{component}: {message}", task_id)


_original_status = status
def status(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a status message that should always be shown."""
    _original_status(component, message, task_id, exc_info=exc_info)
    _intercept_log("INFO", f"{component}: {message}", task_id)
