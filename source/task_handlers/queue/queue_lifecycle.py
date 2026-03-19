"""
Queue lifecycle management extracted from HeadlessTaskQueue.

Contains start, stop, and submit_task logic.  Every public function takes the
``HeadlessTaskQueue`` instance (aliased *queue*) as its first argument.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any

from source.task_handlers.queue.task_processor import worker_loop, _monitor_loop


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

def start_queue(queue: Any, preload_model=None):
    """
    Start the task queue processing service.

    Args:
        queue: The HeadlessTaskQueue instance
        preload_model: Optional model to pre-load before processing tasks.
                      If specified, the model will be loaded immediately after
                      workers start, making the first task much faster.
                      Example: "wan_2_2_vace_lightning_baseline_2_2_2"
    """
    if queue.running:
        queue.logger.warning("Queue already running")
        return

    queue.running = True
    queue.shutdown_event.clear()

    # Start worker threads
    for i in range(queue.max_workers):
        worker = threading.Thread(
            target=worker_loop,
            args=(queue,),
            name=f"GenerationWorker-{i}",
            daemon=True
        )
        worker.start()
        queue.worker_threads.append(worker)

    # Start monitoring thread
    monitor = threading.Thread(
        target=_monitor_loop,
        args=(queue,),
        name="QueueMonitor",
        daemon=True
    )
    monitor.start()
    queue.worker_threads.append(monitor)

    queue.logger.info(f"Task queue started with {queue.max_workers} workers")

    # Pre-load model if specified
    if preload_model:
        queue.logger.info(f"Pre-loading model: {preload_model}")
        try:
            # Initialize orchestrator and load model in background
            queue._ensure_orchestrator()
            if queue.orchestrator is None:
                raise RuntimeError("Orchestrator initialization failed during preload")
            queue.orchestrator.load_model(preload_model)
            queue.current_model = preload_model
            queue.logger.info(f"Model {preload_model} pre-loaded successfully")
        except (RuntimeError, ValueError, OSError) as e:
            # Log the full error with traceback
            queue.logger.error(f"FATAL: Failed to pre-load model {preload_model}: {e}\n{traceback.format_exc()}")

            # If orchestrator failed to initialize, this is fatal - worker cannot function
            if queue.orchestrator is None:
                queue.logger.error("Orchestrator failed to initialize - worker cannot process tasks. Exiting.")
                raise RuntimeError(f"Orchestrator initialization failed during preload: {e}") from e
            else:
                # Orchestrator is OK but model load failed - this is recoverable
                queue.logger.warning(f"Model {preload_model} failed to load, but orchestrator is ready. Worker will continue.")


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

def stop_queue(queue: Any, timeout: float = 30.0):
    """Stop the task queue processing service."""
    if not queue.running:
        return

    queue.logger.info("Shutting down task queue...")
    queue.running = False
    queue.shutdown_event.set()

    # Wait for workers to finish
    for worker in queue.worker_threads:
        worker.join(timeout=timeout)

    # Save queue state (integrate with wgp.py's save system)
    if hasattr(queue, '_save_queue_state'):
        queue._save_queue_state()

    # Optionally unload model to free VRAM
    # queue._cleanup_models()

    queue.logger.info("Task queue shutdown complete")


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def submit_task_impl(queue: Any, task: Any) -> str:
    """
    Submit a new generation task to the queue.

    Args:
        queue: The HeadlessTaskQueue instance
        task: Generation task to process

    Returns:
        Task ID for tracking
    """
    with queue.queue_lock:
        # Pre-validate task conversion (actual conversion happens in _process_task)
        queue._convert_to_wgp_task(task)

        # Add to queue with priority (higher priority = processed first)
        queue.task_queue.put((-task.priority, time.time(), task))
        queue.task_history[task.id] = task
        queue.stats["tasks_submitted"] += 1

        queue.logger.info(f"Task submitted: {task.id} (model: {task.model}, priority: {task.priority})")
        return task.id
