"""Database log buffering and interceptor classes."""

import logging
import threading
import datetime
from datetime import timezone
from typing import Optional, Any, List, Dict

__all__ = [
    "LogBuffer",
    "WorkerDatabaseLogHandler",
    "CustomLogInterceptor",
]


class LogBuffer:
    """
    Thread-safe buffer for collecting logs.

    Logs are stored in memory and flushed periodically with heartbeat updates.
    This prevents excessive database calls while maintaining log history.

    Can optionally send logs to a guardian process via multiprocessing.Queue
    for bulletproof heartbeat delivery that cannot be blocked by GIL or I/O.
    """

    def __init__(self, max_size: int = 100, shared_queue=None):
        """
        Initialize log buffer.

        Args:
            max_size: Maximum logs to buffer before auto-flush (default: 100)
            shared_queue: Optional multiprocessing.Queue to send logs to guardian process
        """
        self.logs: List[Dict[str, Any]] = []
        self.max_size = max_size
        self.lock = threading.RLock()  # Use RLock for reentrancy (flush() called from add())
        self.total_logs = 0
        self.total_flushes = 0
        self.shared_queue = shared_queue  # Queue to guardian process

    def add(
        self,
        level: str,
        message: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Add a log entry to buffer.

        Args:
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            message: Log message
            task_id: Optional task ID for context
            metadata: Optional additional metadata

        Returns:
            List of logs if buffer is full and auto-flushed, otherwise []
        """
        log_entry = {
            'timestamp': datetime.datetime.now(timezone.utc).isoformat(),
            'level': level,
            'message': message,
            'task_id': task_id,
            'metadata': metadata or {}
        }

        # Send to guardian process if available (non-blocking)
        if self.shared_queue:
            try:
                self.shared_queue.put_nowait(log_entry)
            except (OSError, ValueError):
                # Queue full or not available - not critical, guardian will catch up
                pass

        with self.lock:
            self.logs.append(log_entry)
            self.total_logs += 1

            # Auto-flush if buffer is full
            if len(self.logs) >= self.max_size:
                return self.flush()

        return []

    def flush(self) -> List[Dict[str, Any]]:
        """
        Get and clear all buffered logs.

        Returns:
            List of log entries
        """
        with self.lock:
            logs = self.logs.copy()
            self.logs = []
            if logs:
                self.total_flushes += 1
            return logs

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        with self.lock:
            return {
                'current_buffer_size': len(self.logs),
                'total_logs_buffered': self.total_logs,
                'total_flushes': self.total_flushes
            }


class WorkerDatabaseLogHandler(logging.Handler):
    """
    Custom logging handler that buffers logs for database storage.

    Usage:
        log_buffer = LogBuffer()
        handler = WorkerDatabaseLogHandler('gpu-worker-123', log_buffer)
        logging.getLogger().addHandler(handler)

        # Set current task when processing
        handler.set_current_task('task-id-456')
    """

    def __init__(
        self,
        worker_id: str,
        log_buffer: LogBuffer,
        min_level: int = logging.INFO
    ):
        """
        Initialize handler.

        Args:
            worker_id: Worker's unique ID
            log_buffer: LogBuffer instance to collect logs
            min_level: Minimum log level to buffer (default: INFO)
        """
        super().__init__()
        self.worker_id = worker_id
        self.log_buffer = log_buffer
        self.current_task_id: Optional[str] = None
        self.setLevel(min_level)

    def set_current_task(self, task_id: Optional[str]):
        """Set current task ID for context."""
        self.current_task_id = task_id

    def emit(self, record: logging.LogRecord):
        """
        Capture log record to buffer.

        Called automatically by logging framework.
        """
        try:
            # Extract metadata from record
            metadata = {
                'module': record.module,
                'funcName': record.funcName,
                'lineno': record.lineno,
            }

            # Add exception info if present
            if record.exc_info:
                metadata['exception'] = self.format(record)

            # Add to buffer
            self.log_buffer.add(
                level=record.levelname,
                message=record.getMessage(),
                task_id=self.current_task_id,
                metadata=metadata
            )
        except (ValueError, KeyError, TypeError, OSError):
            self.handleError(record)


# Intercept logging calls from our custom logging functions
class CustomLogInterceptor:
    """
    Intercepts calls from our custom logging functions (essential, error, etc.)
    and adds them to the log buffer for database storage.
    """

    def __init__(self, log_buffer: LogBuffer):
        """
        Initialize interceptor.

        Args:
            log_buffer: LogBuffer instance to collect logs
        """
        self.log_buffer = log_buffer
        self._task_context = threading.local()
        self.original_print = None

    def set_current_task(self, task_id: Optional[str]):
        """Set current task ID for context."""
        self._task_context.current_task_id = task_id

    def capture_log(self, level: str, message: str, task_id: Optional[str] = None):
        """
        Capture a log message to the buffer.

        Args:
            level: Log level
            message: Log message
            task_id: Task ID (uses current_task_id if not provided)
        """
        self.log_buffer.add(
            level=level,
            message=message,
            task_id=task_id if task_id is not None else getattr(self._task_context, "current_task_id", None),
            metadata={}
        )
