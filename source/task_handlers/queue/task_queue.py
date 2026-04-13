#!/usr/bin/env python3
"""
WanGP Headless Task Queue Manager

A persistent service that maintains model state and processes generation tasks
via a queue system. This script keeps models loaded in memory and delegates
actual generation to headless_wgp.py while managing the queue, persistence,
and task scheduling.

Key Features:
- Persistent model state (models stay loaded until switched)
- Task queue with priority support
- Auto model switching and memory management
- Status monitoring and progress tracking
- Uses wgp.py's native queue and state management
- Hot-swappable task processing

Usage:
    # Start the headless service
    python headless_model_management.py --wan-dir /path/to/WanGP --port 8080
    
    # Submit tasks via API or queue files
    curl -X POST http://localhost:8080/generate \
         -H "Content-Type: application/json" \
         -d '{"model": "vace_14B", "prompt": "mystical forest", "video_guide": "input.mp4"}'
"""

import os
import sys
import time
import heapq

import threading
import queue
import argparse
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from source.core.log import queue_logger
from source.runtime.wgp_bridge import get_model_def
from source.runtime.process_globals import get_runtime_context
from source.task_handlers.queue.polling_policy import FairQueuePolicy
from source.task_handlers.queue.download_ops import switch_model_impl, convert_to_wgp_task_impl
from source.task_handlers.queue.task_processor import (
    worker_loop,
    process_task_impl,
    execute_generation_impl,
    cleanup_memory_after_task,
)
from source.task_handlers.queue.queue_lifecycle import start_queue, stop_queue, submit_task_impl
from source.task_handlers.queue.wgp_init import WgpInitMixin


# Add WanGP to path for imports
def setup_wgp_path(wan_dir: str):
    """Setup WanGP path and imports."""
    wan_dir = os.path.abspath(wan_dir)
    runtime_context = get_runtime_context(
        "queue.task_queue_path",
        wan_root=wan_dir,
        default_argv=["headless_wgp.py"],
        require_cwd=False,
    )
    runtime_context.prepare(require_cwd=False)
    return wan_dir

# Task definitions
@dataclass
class GenerationTask:
    """Represents a single generation task."""
    id: str
    model: str
    prompt: str
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: str = None
    status: str = "pending"  # pending, processing, completed, failed
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass 
class QueueStatus:
    """Current queue status information."""
    pending_tasks: int
    processing_task: Optional[str]
    completed_tasks: int
    failed_tasks: int
    current_model: Optional[str]
    uptime: float
    memory_usage: Dict[str, Any]


class HeadlessTaskQueue:
    """
    Main task queue manager that integrates with wgp.py's existing queue system.

    This class leverages wgp.py's built-in task management and state persistence
    while providing a clean API for headless operation.

    Runtime helpers:
        WgpInitMixin: CUDA warmup and lazy WanOrchestrator initialization
    """
    
    def __init__(self, wan_dir: str, max_workers: int = 1, debug_mode: bool = False, main_output_dir: Optional[str] = None):
        """
        Initialize the headless task queue.

        Args:
            wan_dir: Path to WanGP directory
            max_workers: Number of concurrent generation workers (recommend 1 for GPU)
            debug_mode: Enable verbose debug logging (should match worker's --debug flag)
            main_output_dir: Optional path for output directory. If not provided, defaults to
                           'outputs' directory next to wan_dir (preserves backwards compatibility)
        """
        self.wan_dir = setup_wgp_path(wan_dir)
        self.max_workers = max_workers
        self.main_output_dir = main_output_dir
        self.running = False
        self.start_time = time.time()
        self.debug_mode = debug_mode  # Now controlled by caller
        
        # Headless stubs to avoid optional UI deps (tkinter/matanyone) during import
        try:
            import types
            # Stub tkinter if not available
            if 'tkinter' not in sys.modules:
                sys.modules['tkinter'] = types.ModuleType('tkinter')
            # Stub preprocessing.matanyone.app with minimal interface.
            # IMPORTANT: dummy packages MUST have __path__ set so Python's
            # import machinery can still traverse subpackages (e.g.
            # preprocessing.matanyone.utils) when wgp.py is imported later.
            _preprocessing_dir = os.path.join(self.wan_dir, 'preprocessing')
            _matanyone_dir = os.path.join(_preprocessing_dir, 'matanyone')
            dummy_pkg = types.ModuleType('preprocessing')
            dummy_pkg.__path__ = [_preprocessing_dir]
            dummy_matanyone = types.ModuleType('preprocessing.matanyone')
            dummy_matanyone.__path__ = [_matanyone_dir]
            dummy_app = types.ModuleType('preprocessing.matanyone.app')
            def _noop_handler():
                class _Dummy:
                    def __getattr__(self, _):
                        return None
                return _Dummy()
            dummy_app.get_vmc_event_handler = _noop_handler  # type: ignore
            sys.modules['preprocessing'] = dummy_pkg
            sys.modules['preprocessing.matanyone'] = dummy_matanyone
            sys.modules['preprocessing.matanyone.app'] = dummy_app
        except (TypeError, AttributeError, ImportError) as e:
            logging.getLogger('HeadlessQueue').debug(f"Failed to set up headless UI stubs: {e}")
        # Don't import wgp during initialization to avoid CUDA/argparse conflicts
        # wgp will be imported lazily when needed (e.g., in _apply_sampler_cfg_preset)
        # This allows the queue to initialize even if CUDA isn't ready yet
        self.wgp = None
        
        # Defer orchestrator initialization to avoid CUDA init during queue setup
        # Orchestrator imports wgp, which triggers deep imports that call torch.cuda
        # We'll initialize it lazily when first needed
        self.orchestrator = None
        self._orchestrator_init_attempted = False
        logging.getLogger('HeadlessQueue').debug("HeadlessTaskQueue created (orchestrator will initialize on first use)")
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.fair_queue_policy = FairQueuePolicy()
        self.task_history: Dict[str, GenerationTask] = {}
        self.current_task: Optional[GenerationTask] = None
        self.current_model: Optional[str] = None
        
        # Threading
        self.worker_threads: List[threading.Thread] = []
        self.queue_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "model_switches": 0,
            "total_generation_time": 0.0
        }
        
        # Setup logging
        self._setup_logging()
        
        # Initialize wgp state (reuse existing state management)
        self._init_wgp_integration()
        
        self.logger.debug(f"HeadlessTaskQueue initialized with WanGP at {wan_dir}")

    def ensure_orchestrator_runtime(self):
        """Run the extracted orchestrator bootstrap flow on this queue host."""
        return WgpInitMixin._ensure_orchestrator(self)

    def _ensure_orchestrator(self):
        return self.ensure_orchestrator_runtime()

    def import_and_create_orchestrator_runtime(self):
        """Create the WanOrchestrator using the extracted runtime bootstrap helper."""
        return WgpInitMixin._import_and_create_orchestrator(self)

    def _import_and_create_orchestrator(self):
        return self.import_and_create_orchestrator_runtime()

    def init_wgp_integration_runtime(self):
        """Populate queue-side WGP integration state using the extracted runtime helper."""
        return WgpInitMixin._init_wgp_integration(self)

    def _init_wgp_integration(self):
        return self.init_wgp_integration_runtime()

    def _setup_logging(self):
        """Setup structured logging that goes to Supabase via the log interceptor."""
        # Keep Python's basic logging for local file backup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('headless.log')
            ]
        )
        self._file_logger = logging.getLogger('HeadlessQueue')
        
        # Use queue_logger (ComponentLogger) as main logger - this goes to Supabase
        # ComponentLogger has compatible interface: .info(), .error(), .warning(), .debug()
        self.logger = queue_logger

    def dequeue_task(self, timeout: float = 1.0):
        """Pull the next task using the queue fairness policy."""
        deadline = time.monotonic() + max(timeout, 0.0)

        while self.running and not self.shutdown_event.is_set():
            with self.task_queue.mutex:
                entries = self.task_queue.queue
                if entries:
                    index = self.fair_queue_policy.choose_index(entries, now=time.time())
                    priority, timestamp, task = entries.pop(index)
                    heapq.heapify(entries)
                    self.task_queue.not_full.notify()
                    return priority, timestamp, task

            if time.monotonic() >= deadline:
                raise queue.Empty
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        raise queue.Empty

    def get_task_status(self, task_id: str) -> Optional[GenerationTask]:
        """Get status of a specific task."""
        return self.task_history.get(task_id)

    def wait_for_completion(self, task_id: str, timeout: float = 300.0) -> Dict[str, Any]:
        """
        Wait for a task to complete and return the result.

        Args:
            task_id: ID of the task to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Dictionary with 'success', 'output_path', and optional 'error' keys
        """
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            task_status = self.get_task_status(task_id)

            if task_status is None:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found in queue"
                }

            if task_status.status == "completed":
                return {
                    "success": True,
                    "output_path": task_status.result_path
                }
            elif task_status.status == "failed":
                return {
                    "success": False,
                    "error": task_status.error_message or "Task failed with unknown error"
                }

            # Task is still pending or processing, wait a bit
            time.sleep(1.0)

        # Timeout reached
        return {
            "success": False,
            "error": f"Task {task_id} did not complete within {timeout} seconds"
        }

    def has_active_work(self) -> bool:
        """Return True if a worker thread is processing a task or tasks are queued.

        Also checks that at least one worker thread is still alive — if all
        threads have died (e.g. native segfault), ``current_task`` may be stale.
        """
        with self.queue_lock:
            if self.current_task is None and self.task_queue.empty():
                return False
            # Guard against stale current_task from a dead worker thread.
            if self.current_task is not None and not any(t.is_alive() for t in self.worker_threads):
                self.logger.warning(
                    f"[QUEUE] current_task {self.current_task.id} set but all worker threads dead — clearing stale state"
                )
                self.current_task = None
                return False
            return True

    def get_queue_status(self) -> QueueStatus:
        """Get current queue status."""
        with self.queue_lock:
            return QueueStatus(
                pending_tasks=self.task_queue.qsize(),
                processing_task=self.current_task.id if self.current_task else None,
                completed_tasks=self.stats["tasks_completed"],
                failed_tasks=self.stats["tasks_failed"],
                current_model=self.current_model,
                uptime=time.time() - self.start_time,
                memory_usage=self._get_memory_usage()
            )

    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics. Returns stub data for now."""
        return {
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "system_memory_used": 0,
            "model_memory_usage": 0
        }

    def _cleanup_memory_after_task(self, task_id: str):
        """Clean up memory after task completion WITHOUT unloading models."""
        return cleanup_memory_after_task(self, task_id)
    
    def start(self, preload_model: Optional[str] = None):
        """Start the task queue processing service."""
        return start_queue(self, preload_model=preload_model)
    
    def stop(self, timeout: float = 30.0):
        """Stop the task queue processing service."""
        return stop_queue(self, timeout=timeout)
    
    def submit_task(self, task: GenerationTask) -> str:
        """Submit a new generation task to the queue."""
        return submit_task_impl(self, task)
    
    def _worker_loop(self):
        """Main worker loop for processing tasks."""
        return worker_loop(self)
    
    def _process_task(self, task: GenerationTask, worker_name: str):
        """Process a single generation task."""
        return process_task_impl(self, task, worker_name)
    
    def _switch_model(self, model_key: str, worker_name: str) -> bool:
        """Ensure the correct model is loaded using wgp.py's model management."""
        return switch_model_impl(self, model_key, worker_name)
    
    def _execute_generation(self, task: GenerationTask, worker_name: str) -> str:
        """Execute the actual generation using headless_wgp.py."""
        return execute_generation_impl(self, task, worker_name)

    def _model_supports_vace(self, model_key: str) -> bool:
        """
        Check if a model supports VACE features (video guides, masks, etc.).
        """
        # Ensure orchestrator is initialized before checking model support
        self._ensure_orchestrator()
        
        try:
            # Use orchestrator's VACE detection with model key
            if hasattr(self.orchestrator, 'is_model_vace'):
                return self.orchestrator.is_model_vace(model_key)
            elif hasattr(self.orchestrator, '_is_vace'):
                # Fallback: load model and check (less efficient)
                current_model = self.current_model
                if current_model != model_key:
                    # Would need to load model to check - use name-based detection as fallback
                    return "vace" in model_key.lower()
                return self.orchestrator._is_vace()
            else:
                # Ultimate fallback: name-based detection
                return "vace" in model_key.lower()
        except (AttributeError, ValueError, TypeError) as e:
            self.logger.warning(f"Could not determine VACE support for model '{model_key}': {e}")
            return "vace" in model_key.lower()
    
    def _is_single_image_task(self, task: GenerationTask) -> bool:
        """
        Check if this is a single image task that should be converted from video to PNG.
        """
        # Check if video_length is 1 (single frame) and this looks like an image task
        video_length = task.parameters.get("video_length", 0)
        return video_length == 1
    
    def _convert_single_frame_video_to_png(self, task: GenerationTask, video_path: str, worker_name: str) -> str:
        """
        Convert a single-frame video to PNG format for single image tasks.
        
        This restores the functionality that was in the original single_image.py handler
        where single-frame videos were converted to PNG files.
        """
        try:
            import cv2
            
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                self.logger.error(f"Video file does not exist for PNG conversion: {video_path}")
                return video_path  # Return original path if conversion fails
            
            # Create PNG output path with sanitized filename to prevent upload issues
            original_filename = video_path_obj.stem
            
            # Sanitize the filename for storage compatibility
            try:
                from source.utils.output_paths import sanitize_filename_for_storage

                sanitized_filename = sanitize_filename_for_storage(original_filename)
                if not sanitized_filename:
                    sanitized_filename = "generated_image"

            except ImportError:
                # Fallback sanitization if import fails
                import re
                sanitized_filename = re.sub(r'[§®©™@·º½¾¿¡~\x00-\x1F\x7F-\x9F<>:"/\\|?*,]', '', original_filename)
                sanitized_filename = re.sub(r'\s+', '_', sanitized_filename.strip())
                if not sanitized_filename:
                    sanitized_filename = "generated_image"
            
            # Create PNG path with sanitized filename
            png_path = video_path_obj.parent / f"{sanitized_filename}.png"
            last_error = "cap-not-opened"
            
            # Log sanitization if filename changed
            if sanitized_filename != original_filename:
                self.logger.debug_anomaly("PNG_CONVERSION", f"Task {task.id}: Sanitized filename '{original_filename}' -> '{sanitized_filename}'")
            
            self.logger.debug_anomaly("PNG_CONVERSION", f"Task {task.id}: Converting {video_path_obj.name} to {png_path.name}")
            
            for attempt in range(1, 4):
                cap = cv2.VideoCapture(str(video_path_obj))
                try:
                    if not cap.isOpened():
                        last_error = "cap-not-opened"
                    else:
                        ret, frame = cap.read()
                        success = cv2.imwrite(str(png_path), frame) if ret else False
                        if ret and success and png_path.exists():
                            self.logger.debug_anomaly("PNG_CONVERSION", f"Task {task.id}: Successfully saved PNG to {png_path}")

                            # Release the video handle BEFORE deleting the file.
                            # On Windows, unlink() fails with WinError 32 if cv2
                            # still holds the file open.
                            cap.release()

                            # Clean up the original video file
                            try:
                                video_path_obj.unlink()
                                self.logger.debug_anomaly("PNG_CONVERSION", f"Task {task.id}: Removed original video file")
                            except OSError as e_cleanup:
                                self.logger.warning(f"[PNG_CONVERSION] Task {task.id}: Could not remove original video: {e_cleanup}")

                            return str(png_path)
                        last_error = "frame-read-or-write"
                finally:
                    cap.release()
                if attempt < 3:
                    time.sleep(0.25)
            self.logger.error(f"[PNG_CONVERSION] Task {task.id}: conversion failed after 3 attempts ({last_error})")
                
        except ImportError:
            self.logger.warning(f"[PNG_CONVERSION] Task {task.id}: OpenCV not available, keeping video format")
        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error(f"[PNG_CONVERSION] Task {task.id}: Error during conversion: {e}")
        
        # Return original video path if conversion failed
        return video_path
    
    def _monitor_loop(self):
        """Background monitoring and maintenance loop."""
        from source.task_handlers.queue.task_processor import _monitor_loop as _monitor_loop_impl
        return _monitor_loop_impl(self)
    
    def _convert_to_wgp_task(self, task: GenerationTask) -> Dict[str, Any]:
        """Convert task to WGP parameters using typed TaskConfig."""
        return convert_to_wgp_task_impl(self, task)
    
    def _apply_sampler_cfg_preset(self, model_key: str, sample_solver: str, wgp_params: Dict[str, Any]):
        """Apply sampler-specific CFG and flow_shift settings from model configuration."""
        try:
            # Import WGP to get model definition
            model_def = get_model_def(model_key)
            
            # Check if model has sampler-specific presets
            sampler_presets = model_def.get("sampler_cfg_presets", {})
            if sample_solver in sampler_presets:
                preset = sampler_presets[sample_solver]
                
                # Apply preset settings, but allow task parameters to override
                applied_params = {}
                for param, value in preset.items():
                    if param not in wgp_params:  # Only apply if not explicitly set in task
                        wgp_params[param] = value
                        applied_params[param] = value
                        
                self.logger.debug(f"Applied sampler '{sample_solver}' CFG preset: {applied_params}")
            else:
                self.logger.debug(f"No CFG preset found for sampler '{sample_solver}' in model '{model_key}'")
                
        except (ValueError, KeyError, AttributeError, TypeError, ImportError) as e:
            self.logger.warning(f"Failed to apply sampler CFG preset: {e}")
    

def create_sample_task(task_id: str, model: str, prompt: str, **params) -> GenerationTask:
    """Helper to create sample tasks for testing."""
    return GenerationTask(
        id=task_id,
        model=model,
        prompt=prompt,
        parameters=params
    )


def main():
    """Main entry point for the headless service."""
    from source.core.log.core import _is_env_debug, disable_debug_mode, enable_debug_mode, suppress_library_logging
    from source.core.log.lifecycle import lifecycle

    parser = argparse.ArgumentParser(description="WanGP Headless Task Queue")
    parser.add_argument("--wan-dir", required=True, help="Path to WanGP directory")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    debug_mode = args.debug or _is_env_debug()

    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        enable_debug_mode()
    else:
        disable_debug_mode()
        suppress_library_logging()

    # Initialize queue
    task_queue = HeadlessTaskQueue(args.wan_dir, max_workers=args.workers, debug_mode=debug_mode)

    # Setup signal handlers
    def signal_handler(signum, frame):
        queue_logger.essential("Received shutdown signal, stopping...")
        lifecycle.run_summary.render_to(queue_logger)
        task_queue.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start services
        task_queue.start()

        queue_logger.essential(f"Headless queue started")
        queue_logger.essential(f"WanGP directory: {args.wan_dir}")
        queue_logger.essential(f"Workers: {args.workers}")
        queue_logger.essential("Press Ctrl+C to stop...")

        # Example: Submit some test tasks
        if debug_mode:
            queue_logger.debug("Submitting test tasks...")

            # Test T2V task
            t2v_task = create_sample_task(
                "test-t2v-1",
                "t2v",
                "a mystical forest with glowing trees",
                resolution="1280x720",
                video_length=49,
                seed=42
            )
            task_queue.submit_task(t2v_task)

        # Keep running until shutdown
        while task_queue.running:
            time.sleep(1.0)

            # Print periodic status
            if debug_mode:
                status = task_queue.get_queue_status()
                queue_logger.debug(f"Queue: {status.pending_tasks} pending, "
                      f"{status.completed_tasks} completed, "
                      f"{status.failed_tasks} failed")

    except KeyboardInterrupt:
        queue_logger.essential("Shutdown requested by user")
        lifecycle.run_summary.render_to(queue_logger)
    except Exception as e:
        queue_logger.error(f"Fatal error: {e}")
        raise
    finally:
        lifecycle.run_summary.render_to(queue_logger)
        task_queue.stop()


if __name__ == "__main__":
    main()
