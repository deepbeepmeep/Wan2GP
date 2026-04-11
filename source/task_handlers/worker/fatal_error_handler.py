"""Fatal Error Handler for Worker Processes

This module defines fatal errors that indicate the worker is in an unrecoverable
state and should be terminated. These errors typically indicate hardware/driver
issues that cannot be resolved without intervention.

Fatal error categories:
1. CUDA/GPU driver failures (initialization, driver lost, etc.)
2. Critical hardware errors (GPU hung, PCIe errors)
3. Model corruption/critical loading failures
4. Irrecoverable memory errors
5. Critical VRAM/OOM errors that persist
"""

import re
import os
import sys

from source.core.log import headless_logger


class FatalWorkerError(Exception):
    """
    Exception raised when a worker encounters a fatal error that requires termination.
    
    These errors indicate the worker is in an unrecoverable state (GPU driver issues,
    hardware failures, etc.) and should be killed rather than continuing to process tasks.
    """
    
    def __init__(self, message: str, original_error: Exception = None, error_category: str = None):
        self.original_error = original_error
        self.error_category = error_category or "unknown"
        super().__init__(message)


# Fatal error patterns that indicate the worker should be killed
# These patterns are CONSERVATIVE - only truly unrecoverable errors
# 
# Each category has a threshold - number of consecutive errors before killing worker
FATAL_ERROR_PATTERNS = {
    "cuda_driver": {
        "patterns": [
            # CUDA driver completely failed to initialize - this is unrecoverable
            r"CUDA driver initialization failed.*might not have a CUDA gpu",
            r"no CUDA-capable device is detected",
            r"Found no NVIDIA driver",
            r"CUDA driver version is insufficient for CUDA runtime version",
            # NOTE: We removed "device already in use" - that's usually transient
        ],
        "threshold": 2,  # CUDA driver errors are usually persistent, allow 2 attempts
    },
    "cuda_hardware": {
        "patterns": [
            # Hardware has physically failed or become inaccessible
            r"GPU has fallen off the bus",
            r"catastrophic driver failure",
            r"CUDA error: the launch timed out and was terminated.*watchdog",
            # NOTE: We removed general "illegal memory access" - those can sometimes be model bugs
        ],
        "threshold": 1,  # Hardware failures are immediate - GPU off bus won't recover
    },
    "nvml": {
        "patterns": [
            # NVML library completely unavailable (driver installation issue)
            r"NVML Shared Library Not Found",
            r"Failed to initialize NVML.*libnvidia",
        ],
        "threshold": 2,  # Library missing is persistent but allow retry in case of transient
    },
    # NOTE: We removed "model_corruption" category entirely - those errors could be:
    #   - Config issues (wrong model for task type)
    #   - Partial downloads (can be fixed by re-downloading)
    #   - Version mismatches (can be fixed by updating)
    # These should fail the task but not kill the worker
    
    # NOTE: We removed "critical_oom" category entirely - OOM errors are almost always
    # recoverable by retrying with smaller batch sizes or letting memory clear
    
    "system_critical": {
        "patterns": [
            # System-level fatal errors that won't resolve
            r"Bus error.*signal",
            r"Segmentation fault.*core dumped",
            r"Fatal Python error.*core dumped",
        ],
        "threshold": 1,  # Segfaults/bus errors are immediate failures
    },
}

# =============================================================================
# RETRYABLE ERROR PATTERNS
# =============================================================================
# These errors indicate transient failures that may succeed on retry.
# Tasks matching these patterns should be requeued (up to max_attempts) rather
# than permanently marked as Failed.

RETRYABLE_ERROR_PATTERNS = {
    # NOTE: OOM is intentionally NOT retryable. Retrying on the same worker with the
    # same parameters will always fail again. If you have workers with different VRAM
    # capacities, manually requeue the task or implement worker exclusion logic.
    "generation_no_output": {
        "patterns": [
            r"No output generated",
            r"Generation produced no output",
        ],
        "max_attempts": 2,
        "description": "Generation completed but produced no output - often transient",
    },
    "edge_function_transient": {
        "patterns": [
            r"\[EDGE_FAIL:.*:HTTP_500\]",
            r"\[EDGE_FAIL:.*:5XX_TRANSIENT\]",
            r"\[EDGE_FAIL:.*:TIMEOUT\]",
            r"\[EDGE_FAIL:.*:NETWORK\]",
            r"WORKER_ERROR.*Function exited due to an error",
        ],
        "max_attempts": 3,
        "description": "Edge function failures - often CDN issues or cold starts",
    },
    "network_transient": {
        "patterns": [
            r"ConnectionError",
            r"Connection reset by peer",
            r"Connection refused",
            r"Network is unreachable",
            r"timed out",
        ],
        "max_attempts": 3,
        "description": "Network connectivity issues - usually resolve on retry",
    },
}

# Default max attempts if category doesn't specify
DEFAULT_MAX_ATTEMPTS = 2


def is_retryable_error(error_message: str) -> tuple[bool, str | None, int]:
    """
    Determine if an error is retryable and the task should be requeued.
    
    Args:
        error_message: String representation of the error
    
    Returns:
        Tuple of (is_retryable: bool, category: str | None, max_attempts: int)
    """
    if not error_message:
        return False, None, 0
    
    error_str = str(error_message)
    
    # Check against all retryable error patterns
    for category, config in RETRYABLE_ERROR_PATTERNS.items():
        patterns = config["patterns"]
        max_attempts = config.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        
        for pattern in patterns:
            if re.search(pattern, error_str, re.IGNORECASE):
                return True, category, max_attempts
    
    return False, None, 0


# =============================================================================
# FATAL ERROR TRACKING
# =============================================================================

# Global counter for tracking consecutive fatal errors (reset on successful task)
_consecutive_fatal_errors = 0
_last_fatal_category = None


def reset_fatal_error_counter():
    """Reset the consecutive fatal error counter (call after successful task completion)."""
    global _consecutive_fatal_errors, _last_fatal_category
    _consecutive_fatal_errors = 0
    _last_fatal_category = None


def is_fatal_error(error_message: str, exception: Exception = None) -> tuple[bool, str | None, int | None]:
    """
    Determine if an error is fatal and requires worker termination.
    
    Args:
        error_message: String representation of the error
        exception: Original exception object (optional, for type checking)
    
    Returns:
        Tuple of (is_fatal: bool, category: str | None, threshold: int | None)
    """
    if not error_message:
        return False, None, None
    
    error_str = str(error_message).lower()
    
    # Check against all fatal error patterns
    for category, config in FATAL_ERROR_PATTERNS.items():
        patterns = config["patterns"]
        threshold = config["threshold"]
        
        for pattern in patterns:
            if re.search(pattern, error_str, re.IGNORECASE):
                return True, category, threshold
    
    # Check exception type for specific critical exceptions
    if exception:
        # CUDA errors from torch
        if "torch.cuda" in str(type(exception)):
            # Some CUDA errors are recoverable (OOM can be), but driver errors are not
            if "driver" in error_str or "initialization" in error_str:
                return True, "cuda_driver", FATAL_ERROR_PATTERNS["cuda_driver"]["threshold"]
        
        # System-level errors
        if isinstance(exception, (MemoryError, OSError)):
            if "cannot allocate memory" in error_str:
                return True, "system_critical", FATAL_ERROR_PATTERNS["system_critical"]["threshold"]
    
    return False, None, None


def _mark_worker_for_termination(
    worker_id: str,
    task_id: str,
    error_reason: str,
    logger=None
) -> bool:
    """
    Mark worker for termination in Supabase when it encounters a fatal error.
    
    This function:
    1. Marks the worker as 'error' in Supabase workers table
    2. Resets the task back to 'Queued' status
    3. Orchestrator will detect error status and terminate the RunPod pod
    
    Args:
        worker_id: The worker's unique ID
        task_id: The current task ID that failed
        error_reason: Description of the fatal error
        logger: Logger instance (optional)
    
    Returns:
        bool: True if marking was successful, False otherwise
    """
    try:
        # Only works with Supabase
        import os
        from datetime import datetime, timezone
        
        from source.core.db import config as _db_config
        from source.core.db.edge import request as edge_request

        runtime = _db_config.get_db_runtime_config()
        supabase_url = (
            os.getenv("SUPABASE_URL")
            or getattr(runtime, "supabase_url", None)
            or _db_config.SUPABASE_URL
        )
        supabase_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
            or edge_request.resolve_edge_auth_token(
                scope="service",
                runtime_config=runtime,
            )
        )

        if not supabase_url or not supabase_key:
            if logger:
                logger.warning("Supabase auth configuration not available - cannot mark worker for termination")
            return False
        
        from supabase import create_client
        supabase = create_client(supabase_url, supabase_key)
        
        if logger:
            logger.info(f"Marking worker {worker_id} for termination")
        
        # 1. Get worker info to retrieve RunPod ID
        worker_response = supabase.table('workers').select('id, metadata').eq('id', worker_id).single().execute()
        
        if not worker_response.data:
            if logger:
                logger.error(f"Worker {worker_id} not found in database")
            return False
        
        worker = worker_response.data
        runpod_id = worker.get('metadata', {}).get('runpod_id')
        
        # 2. Mark worker as error
        error_time = datetime.now(timezone.utc).isoformat()
        
        update_response = supabase.table('workers').update({
            'status': 'error',
            'metadata': {
                **worker.get('metadata', {}),
                'error_reason': error_reason,
                'error_time': error_time,
                'self_terminated': True,
            }
        }).eq('id', worker_id).execute()
        
        if not update_response.data:
            if logger:
                logger.error(f"Failed to mark worker as error")
            return False
        
        # 3. Reset task to Queued
        task_response = supabase.table('tasks').update({
            'status': 'Queued',
            'worker_id': None,
            'generation_started_at': None,
            'error_details': error_reason,
        }).eq('id', task_id).eq('worker_id', worker_id).eq('status', 'In Progress').execute()
        
        if logger:
            if task_response.data:
                logger.info(f"Task {task_id} reset to Queued")
            else:
                logger.warning(f"Task may not have been reset (already completed or not assigned to this worker)")
            
            logger.info(f"Worker marked for termination. Orchestrator will terminate RunPod pod {runpod_id if runpod_id else 'UNKNOWN'}")
        
        return True
        
    except (OSError, ValueError, RuntimeError) as e:
        if logger:
            logger.error(f"Failed to mark worker for termination: {e}")
        return False


def handle_fatal_error_in_worker(
    error_message: str, 
    exception: Exception = None,
    logger = None,
    worker_id: str = None,
    task_id: str = None
):
    """
    Handle a fatal error in a worker context.
    
    This function should be called when a fatal error is detected. It will:
    1. Check if error matches fatal patterns
    2. Track consecutive fatal errors (requires FATAL_ERROR_THRESHOLD to trigger)
    3. Log the fatal error with context
    4. Attempt to mark the current task as failed (if task_id provided)
    5. Signal the worker to shut down and exit
    
    Args:
        error_message: The error message
        exception: The original exception (optional)
        logger: Logger instance (optional)
        worker_id: Worker identifier (optional, for logging)
        task_id: Current task ID (optional, for marking failed)
    """
    global _consecutive_fatal_errors, _last_fatal_category
    
    is_fatal, category, threshold = is_fatal_error(error_message, exception)
    
    if not is_fatal:
        # Not a fatal error, just return
        return
    
    # Reset counter if category changed (different type of error)
    if _last_fatal_category and _last_fatal_category != category:
        if logger:
            logger.info(f"Error category changed from {_last_fatal_category} to {category}, resetting counter")
        _consecutive_fatal_errors = 0
    
    # Increment consecutive fatal error counter
    _consecutive_fatal_errors += 1
    _last_fatal_category = category
    
    if logger:
        logger.warning(
            f"Fatal-pattern error detected ({_consecutive_fatal_errors}/{threshold}): "
            f"{category} - {error_message[:200]}"
        )
    
    # Only actually kill the worker if we've hit the category-specific threshold
    if _consecutive_fatal_errors < threshold:
        if logger:
            logger.warning(
                f"Not terminating yet - need {threshold - _consecutive_fatal_errors} more "
                f"consecutive '{category}' errors before killing worker"
            )
        return
    
    if logger:
        logger.critical(
            f"🚨 FATAL ERROR DETECTED - Worker will terminate 🚨\n"
            f"Category: {category}\n"
            f"Worker: {worker_id or 'unknown'}\n"
            f"Task: {task_id or 'none'}\n"
            f"Error: {error_message}\n"
            f"This error indicates the worker is in an unrecoverable state.\n"
            f"The worker will shut down immediately."
        )
    else:
        headless_logger.critical(
            f"FATAL WORKER ERROR - TERMINATING | "
            f"Category: {category} | "
            f"Worker: {worker_id or 'unknown'} | "
            f"Task: {task_id or 'none'} | "
            f"Error: {error_message}"
        )
    
    # Attempt to mark worker for termination via Supabase (if worker_id provided)
    if worker_id and task_id:
        try:
            success = _mark_worker_for_termination(
                worker_id=worker_id,
                task_id=task_id,
                error_reason=f"FATAL {category}: {error_message[:500]}",
                logger=logger
            )
            if success and logger:
                logger.info("Worker successfully marked for termination in database")
        except (OSError, ValueError, RuntimeError) as e:
            if logger:
                logger.error(f"Failed to mark worker for termination: {e}")
    
    # Signal shutdown
    if logger:
        logger.critical("Initiating worker shutdown due to fatal error...")
    
    # Raise the fatal error to propagate up to main loop
    raise FatalWorkerError(
        f"Fatal {category} error: {error_message}",
        original_error=exception,
        error_category=category
    )


def is_running_as_worker() -> bool:
    """
    Determine if the current process is running as a worker.
    
    This checks for worker-specific environment variables or command-line arguments
    to avoid killing non-worker processes (like interactive sessions).
    
    Returns:
        True if running as a worker, False otherwise
    """
    # Check for worker-specific environment variable
    if os.getenv("WAN2GP_WORKER_MODE") == "true":
        return True
    
    # Check command-line arguments for worker flags
    if "--worker" in sys.argv or "-w" in sys.argv:
        return True
    
    # Check if running as worker.py
    if "worker.py" in sys.argv[0]:
        return True
    
    return False


def check_and_handle_fatal_error(
    error_message: str,
    exception: Exception = None,
    logger = None,
    worker_id: str = None,
    task_id: str = None
):
    """
    Check if an error is fatal and handle it if running as a worker.
    
    This is the main entry point for fatal error handling. It:
    1. Checks if running as a worker (safe - won't kill interactive sessions)
    2. Checks if error matches fatal patterns
    3. Tracks consecutive errors with category-specific thresholds
    4. Raises FatalWorkerError when threshold is reached
    
    Args:
        error_message: The error message
        exception: The original exception (optional)
        logger: Logger instance (optional)
        worker_id: Worker identifier (optional)
        task_id: Current task ID (optional)
    """
    # Safety check: only handle fatal errors in worker context
    if not is_running_as_worker():
        return
    
    # Delegate to actual handler
    handle_fatal_error_in_worker(
        error_message=error_message,
        exception=exception,
        logger=logger,
        worker_id=worker_id,
        task_id=task_id
    )


# Backward compatibility alias
safe_handle_fatal_error = check_and_handle_fatal_error
