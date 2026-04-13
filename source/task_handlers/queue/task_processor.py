"""
Task processing, generation execution, and worker/monitor loops.

Contains the core logic for processing a single generation task
(``process_task_impl``), delegating the actual generation to
headless_wgp.py via the orchestrator (``execute_generation_impl``),
and the ``worker_loop`` / ``_monitor_loop`` entry-points used by
``queue_lifecycle``.

Every public function takes the ``HeadlessTaskQueue`` instance
(aliased *queue*) as its first argument.
"""

from __future__ import annotations

import logging
import os
import queue as queue_mod
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# Re-export so callers that previously reached cleanup_memory_after_task
# via worker_thread (and now via task_processor) continue to work.
from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task  # noqa: F401

from source.core.log.display_names import model_label, rel_path
from source.core.log.lifecycle import lifecycle
from source.core.log.queue_runtime import queue_logger
from source.runtime.wgp_bridge import get_wgp_runtime_module_mutable


# ---------------------------------------------------------------------------
# Thread safety for WGP global monkey-patching
# ---------------------------------------------------------------------------
_wgp_patch_lock = threading.Lock()


def _build_task_display_summary(task: Any) -> dict[str, Any]:
    params = getattr(task, "parameters", None) or {}
    summary: dict[str, Any] = {}

    task_model = getattr(task, "model", None)
    if task_model:
        summary["model"] = model_label(task_model)

    resolution = params.get("resolution")
    if isinstance(resolution, str) and resolution:
        summary["resolution"] = resolution
    elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        summary["resolution"] = f"{resolution[0]}x{resolution[1]}"
    elif resolution not in (None, ""):
        summary["resolution"] = resolution
    else:
        parsed_resolution = params.get("parsed_resolution_wh")
        if isinstance(parsed_resolution, (list, tuple)) and len(parsed_resolution) == 2:
            summary["resolution"] = f"{parsed_resolution[0]}x{parsed_resolution[1]}"

    frames = params.get("video_length") or params.get("num_frames")
    if frames not in (None, ""):
        summary["frames"] = frames

    return summary


def _check_fatal_error_and_raise(*, queue: Any, error_message_str: str, exception: BaseException, task_id: str):
    try:
        from source.task_handlers.worker.fatal_error_handler import check_and_handle_fatal_error, FatalWorkerError

        check_and_handle_fatal_error(
            error_message=error_message_str,
            exception=exception,
            logger=queue.logger,
            worker_id=os.getenv("WORKER_ID"),
            task_id=task_id,
        )
    except FatalWorkerError:
        raise
    except (RuntimeError, ValueError, OSError, ImportError) as fatal_check_error:
        queue.logger.error(f"[TASK_ERROR] Error checking for fatal errors: {fatal_check_error}")


# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------

def process_task_impl(queue: Any, task: Any, worker_name: str):
    """
    Process a single generation task.

    This is where we delegate to headless_wgp.py while managing
    model persistence and state.
    """
    try:
        pass
    except Exception as e:
        _check_fatal_error_and_raise(
            queue=queue,
            error_message_str=str(e),
            exception=e,
            task_id=getattr(task, "id", "unknown"),
        )
        raise
    except BaseException as e:
        _check_fatal_error_and_raise(
            queue=queue,
            error_message_str=str(e),
            exception=e,
            task_id=getattr(task, "id", "unknown"),
        )
        raise

    # Ensure logs emitted during this generation are attributed to this task.
    # This runs inside the GenerationWorker thread, which is where wgp/headless_wgp runs.
    try:
        from source.core.log import set_current_task_context  # local import to avoid cycles
        set_current_task_context(task.id)
    except (ImportError, AttributeError, TypeError):
        # Swallowing is intentional: logging context is optional and failures are debug-only
        pass

    with queue.queue_lock:
        queue.current_task = task
        task.status = "processing"

    start_time = time.time()
    billing_reset_ok = False
    task_type = task.parameters.get("_source_task_type") or task.model
    summary = _build_task_display_summary(task)

    try:
        with lifecycle.task(
            task.id,
            task_type,
            model=task.model,
            display_summary=summary,
        ) as anchor:
            try:
                # 1. Ensure correct model is loaded (orchestrator checks WGP's ground truth)
                switch_start = time.time()
                queue._switch_model(task.model, worker_name)
                switch_elapsed = time.time() - switch_start
                # 2. Reset billing start time now that model is loaded
                # This ensures users aren't charged for model loading time
                try:
                    from source.core.db.task_status import reset_generation_started_at
                    reset_generation_started_at(task.id)
                    billing_reset_ok = True
                except (OSError, ValueError, RuntimeError) as e_billing:
                    # Don't fail the task if billing reset fails - just log it
                    queue.logger.debug_anomaly("BILLING", f"Failed to reset generation_started_at for task {task.id}: {e_billing}")

                # 3. Delegate actual generation to orchestrator
                # The orchestrator handles the heavy lifting while we manage the queue
                gen_start = time.time()
                result_path = queue._execute_generation(task, worker_name)
                gen_elapsed = time.time() - gen_start
                # 3. Validate output and update task status
                processing_time = time.time() - start_time
                is_success = bool(result_path)
                try:
                    if is_success:
                        # If a path was returned, check existence where possible
                        rp = Path(result_path)
                        is_success = rp.exists()

                        # Some environments (e.g. networked volumes) can briefly report a freshly-written file as missing.
                        # Do a short retry before failing the task.
                        if not is_success:
                            try:
                                # Note: os is imported at module level - don't re-import here as it causes
                                # "local variable 'os' referenced before assignment" due to Python scoping
                                import time as _time

                                retry_s = 2.0
                                interval_s = 0.2
                                attempts = max(1, int(retry_s / interval_s))

                                for _ in range(attempts):
                                    _time.sleep(interval_s)
                                    if rp.exists():
                                        is_success = True
                                        break

                                if not is_success:
                                    # Tagged diagnostics to debug "phantom output path" failures
                                    # Keep output bounded to avoid log spam.
                                    tag = "[TravelNoOutputGenerated]"
                                    cwd = os.getcwd()
                                    parent = rp.parent
                                    queue.logger.error(f"{tag} Output path missing after generation: {result_path}")
                                    queue.logger.error(f"{tag} CWD: {cwd}")
                                    try:
                                        queue.logger.error(f"{tag} Parent exists: {parent} -> {parent.exists()}")
                                    except OSError as _e:
                                        queue.logger.error(f"{tag} Parent exists check failed: {type(_e).__name__}: {_e}")

                                    try:
                                        if parent.exists():
                                            # Show a small sample of directory contents to spot mismatched output dirs.
                                            entries = sorted([p.name for p in parent.iterdir()])[:50]
                                            queue.logger.error(f"{tag} Parent dir sample (first {len(entries)}): {entries}")
                                    except OSError as _e:
                                        queue.logger.error(f"{tag} Parent list failed: {type(_e).__name__}: {_e}")

                                    # Common alternative location when running from Wan2GP/ with relative outputs
                                    try:
                                        alt_parent = Path(cwd) / "outputs"
                                        if alt_parent != parent and alt_parent.exists():
                                            alt_entries = sorted([p.name for p in alt_parent.iterdir()])[:50]
                                            queue.logger.error(f"{tag} Alt outputs dir: {alt_parent} sample (first {len(alt_entries)}): {alt_entries}")
                                    except OSError as _e:
                                        queue.logger.error(f"{tag} Alt outputs list failed: {type(_e).__name__}: {_e}")
                            except OSError:
                                # Never let diagnostics break the worker loop.
                                pass
                except OSError:
                    # If any exception while checking, keep prior truthiness
                    pass

                with queue.queue_lock:
                    task.processing_time = processing_time
                    if is_success:
                        task.status = "completed"
                        task.result_path = result_path
                        queue.stats["tasks_completed"] += 1
                        queue.stats["total_generation_time"] += processing_time
                    else:
                        task.status = "failed"
                        task.error_message = "No output generated"
                        queue.stats["tasks_failed"] += 1
                        anchor.set(error=task.error_message)
                        queue.logger.debug(f"Task {task.id} failed after {processing_time:.1f}s: No output generated")

                # Memory cleanup after each task (does NOT unload models)
                # This clears PyTorch's internal caches and Python garbage to prevent fragmentation
                queue._cleanup_memory_after_task(task.id)

                if is_success:
                    anchor.set(output=result_path)

            except (RuntimeError, ValueError, OSError, TypeError, AttributeError) as e:
                # Handle task failure
                processing_time = time.time() - start_time
                error_message_str = str(e)

                with queue.queue_lock:
                    task.status = "failed"
                    task.error_message = error_message_str
                    task.processing_time = processing_time
                    queue.stats["tasks_failed"] += 1

                anchor.set(error=error_message_str)
                queue.logger.error(f"[TASK_ERROR] Task {task.id} failed after {processing_time:.1f}s: {type(e).__name__}: {e}")
                queue.logger.error(f"[TASK_ERROR] Full traceback:\n{traceback.format_exc()}")

                # DEEP DIAGNOSTIC: Log exception details for NoneType errors
                if isinstance(e, (TypeError, AttributeError)):
                    queue.logger.error(f"[TASK_ERROR_DEEP] Exception is {type(e).__name__} - likely attribute access on None")
                    queue.logger.error(f"[TASK_ERROR_DEEP] Exception module: {type(e).__module__}")
                    queue.logger.error(f"[TASK_ERROR_DEEP] Exception args: {e.args}")
                    if hasattr(e, '__traceback__'):
                        import sys
                        tb = e.__traceback__
                        queue.logger.error(f"[TASK_ERROR_DEEP] Traceback frames:")
                        frame_num = 0
                        while tb is not None:
                            frame = tb.tb_frame
                            queue.logger.error(f"  Frame {frame_num}: {frame.f_code.co_filename}:{tb.tb_lineno} in {frame.f_code.co_name}")
                            queue.logger.error(f"    Local vars: {list(frame.f_locals.keys())}")
                            tb = tb.tb_next
                            frame_num += 1

                try:
                    from source.core.log import flush_log_buffer
                    flush_log_buffer()
                except (ImportError, AttributeError, OSError):
                    pass  # Don't let flush errors mask the original exception

                _check_fatal_error_and_raise(
                    queue=queue,
                    error_message_str=error_message_str,
                    exception=e,
                    task_id=task.id,
                )
            except BaseException as e:
                # Catch any unexpected exceptions to prevent task from silently dying
                processing_time = time.time() - start_time
                error_message_str = str(e)

                with queue.queue_lock:
                    task.status = "failed"
                    task.error_message = error_message_str
                    task.processing_time = processing_time
                    queue.stats["tasks_failed"] += 1

                anchor.set(error=error_message_str)
                queue.logger.critical(f"[TASK_ERROR] Task {task.id} hit UNEXPECTED exception after {processing_time:.1f}s: {type(e).__name__}: {e}")
                queue.logger.critical(f"[TASK_ERROR] Full traceback:\n{traceback.format_exc()}")

                try:
                    from source.core.log import flush_log_buffer
                    flush_log_buffer()
                except (ImportError, AttributeError, OSError):
                    pass  # Don't let flush errors mask the original exception

                _check_fatal_error_and_raise(
                    queue=queue,
                    error_message_str=error_message_str,
                    exception=e,
                    task_id=task.id,
                )

    finally:
        with queue.queue_lock:
            queue.current_task = None
        try:
            from source.core.log import set_current_task_context  # local import to avoid cycles
            set_current_task_context(None)
        except (ImportError, AttributeError, TypeError):
            # Swallowing is intentional: logging context is optional and failures are debug-only
            pass


# ---------------------------------------------------------------------------
# Generation execution
# ---------------------------------------------------------------------------

def execute_generation_impl(queue: Any, task: Any, worker_name: str) -> str:
    """
    Execute the actual generation using headless_wgp.py.

    This delegates to the orchestrator while providing progress tracking
    and integration with our queue system. Enhanced to support video guides,
    masks, image references, and other advanced features.
    """
    # Ensure orchestrator is initialized before generation
    queue._ensure_orchestrator()

    # Convert task parameters to WanOrchestrator format
    wgp_params = queue._convert_to_wgp_task(task)

    # Remove model and prompt from params since they're passed separately to avoid duplication
    generation_params = {k: v for k, v in wgp_params.items() if k not in ("model", "prompt")}

    # CRITICAL: Apply phase_config patches NOW in the worker thread where wgp is imported
    # Acquire lock to prevent concurrent tasks from corrupting shared wgp globals
    lock_acquire_start = time.time()
    with _wgp_patch_lock:
        lock_acquired_elapsed = time.time() - lock_acquire_start
        if lock_acquired_elapsed > 0.1:
            queue.logger.warning(f"[LOCK_ACQUIRE] Task {task.id} waited {lock_acquired_elapsed:.3f}s for _wgp_patch_lock (possible contention)")
        try:
            return _execute_generation_with_patches(queue, task, worker_name, generation_params)
        finally:
            pass


def _execute_generation_with_patches(
    queue: Any,
    task: Any,
    worker_name: str,
    generation_params: dict,
):
    """Run generation with wgp global patching, under _wgp_patch_lock."""
    # Store patch info for cleanup in finally block
    _patch_applied = False
    _parsed_phase_config_for_restore = None
    _model_name_for_restore = None

    if "_parsed_phase_config" in generation_params and "_phase_config_model_name" in generation_params:
        parsed_phase_config = generation_params.pop("_parsed_phase_config")
        model_name = generation_params.pop("_phase_config_model_name")

        # Save for restoration
        _parsed_phase_config_for_restore = parsed_phase_config
        _model_name_for_restore = model_name

        from source.core.params.phase_config import apply_phase_config_patch
        apply_phase_config_patch(parsed_phase_config, model_name, task.id)
        _patch_applied = True

    # Handle svi2pro: This is a model_def property, not a generate() parameter
    # We need to patch it into BOTH wgp.models_def AND wan_model.model_def
    # because the model captures model_def at load time, not at generation time
    _svi2pro_original = None
    _svi2pro_patched = False
    _wan_model_patched = False
    if generation_params.get("svi2pro"):
        try:
            wgp = get_wgp_runtime_module_mutable()
            model_key = task.model

            # Patch 1: wgp.models_def (for any new model loads)
            if model_key in wgp.models_def:
                _svi2pro_original = wgp.models_def[model_key].get("svi2pro")
                wgp.models_def[model_key]["svi2pro"] = True
                _svi2pro_patched = True
                queue.logger.debug_anomaly("SVI2PRO", f"Patched wgp.models_def['{model_key}']['svi2pro'] = True (was: {_svi2pro_original})", task_id=task.id)

            # Patch 2: wan_model.model_def DIRECTLY (the actual object used during generation)
            # This is critical because the model was loaded BEFORE we patched models_def
            if hasattr(wgp, 'wan_model') and wgp.wan_model is not None:
                if hasattr(wgp.wan_model, 'model_def') and wgp.wan_model.model_def is not None:
                    # Patch svi2pro
                    wgp.wan_model.model_def["svi2pro"] = True

                    # CRITICAL: Also patch sliding_window=True - required for video continuation
                    # Without this, reuse_frames=0 and video_source context is ignored
                    _sliding_window_original = wgp.wan_model.model_def.get("sliding_window")
                    wgp.wan_model.model_def["sliding_window"] = True

                    # CRITICAL: Patch sliding_window_defaults to bypass WGP's latent alignment formula
                    # Without this, sliding_window_overlap=4 becomes 1 via: (4-1)//4*4+1 = 1
                    # The original SVI model has overlap_default=4, which makes the formula skip
                    wgp.wan_model.model_def["sliding_window_defaults"] = {"overlap_default": 4}

                    # CRITICAL: In the kijai-style SVI+end-frame pixel concatenation path, the middle frames
                    # are initialized as zeros before VAE encode (matching kijai and original Wan2GP).
                    # Use zeros for empty frames (standard approach, matching kijai)
                    wgp.wan_model.model_def["svi_empty_frames_mode"] = "zeros"

                    # Also patch wgp.models_def for consistency (this is what test_any_sliding_window reads!)
                    if model_key in wgp.models_def:
                        _sw_before = wgp.models_def[model_key].get("sliding_window", "NOT_SET")
                        wgp.models_def[model_key]["sliding_window"] = True
                        wgp.models_def[model_key]["sliding_window_defaults"] = {"overlap_default": 4}
                        wgp.models_def[model_key]["svi_empty_frames_mode"] = "zeros"
                        queue.logger.debug(
                            f"[SVI2PRO] Patched wgp.models_def['{model_key}']['sliding_window'] = True (was: {_sw_before})",
                            task_id=task.id,
                        )

                    _wan_model_patched = True

                    queue.logger.debug(
                        f"[SVI2PRO] Task {task.id}: model={model_key}, patched_models_def={_svi2pro_patched}, "
                        f"patched_wan_model={True}, previous_sliding_window={_sliding_window_original}",
                        task_id=task.id,
                    )
                else:
                    queue.logger.warning(f"[SVI2PRO] wan_model exists but has no model_def", task_id=task.id)
            else:
                queue.logger.warning(f"[SVI2PRO] wgp.wan_model not found - model may not be loaded yet", task_id=task.id)

        except (RuntimeError, AttributeError, ImportError, KeyError) as e:
            queue.logger.warning(f"[SVI2PRO] Failed to patch svi2pro: {e}", task_id=task.id)
        # Remove from generation_params since it's not a generate() parameter
        generation_params.pop("svi2pro", None)

    # Determine generation type and delegate - wrap in try/finally for patch restoration
    import signal
    generation_start_time = time.time()
    GENERATION_TIMEOUT_SECONDS = 1200  # 20 minutes - longer than task timeout but prevents indefinite hangs

    def timeout_handler(signum, frame):
        """Handler for SIGALRM when generation takes too long"""
        elapsed = time.time() - generation_start_time
        queue.logger.error(
            f"[GENERATION_TIMEOUT] Task {task.id} exceeded {GENERATION_TIMEOUT_SECONDS}s timeout after {elapsed:.1f}s. "
            f"Generation is likely stuck in WGP/CUDA. This usually indicates a GPU synchronization deadlock.",
            task_id=task.id
        )
        raise RuntimeError(f"Generation timeout after {elapsed:.1f}s")

    _sigalrm_supported = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
    try:
        # Set timeout alarm (will raise SIGALRM) — POSIX only; Windows has no SIGALRM/signal.alarm
        if _sigalrm_supported:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(GENERATION_TIMEOUT_SECONDS)
            except (ValueError, OSError) as e:
                queue.logger.debug_anomaly("GENERATION", f"timeout not available on this platform: {e}", task_id=task.id)
        else:
            queue.logger.debug_anomaly("GENERATION", "signal.SIGALRM unavailable (Windows); generation timeout disabled", task_id=task.id)

        # Check if model supports VACE features
        model_supports_vace = queue._model_supports_vace(task.model)
        is_flux = queue.orchestrator._is_flux()
        is_z_image = queue.orchestrator._is_z_image()
        if model_supports_vace:
            generation_path = "vace"
        elif is_flux:
            generation_path = "flux"
        elif is_z_image:
            generation_path = "image"
        else:
            generation_path = "t2v"
        if model_supports_vace:
            # CRITICAL: VACE models require a video_guide parameter
            if "video_guide" in generation_params and generation_params["video_guide"]:
                pass
            else:
                error_msg = f"VACE model '{task.model}' requires a video_guide parameter but none was provided. VACE models cannot perform pure text-to-video generation."
                queue.logger.error(f"[GENERATION] Task {task.id}: {error_msg}")
                raise ValueError(error_msg)

            gen_call_start = time.time()
            result = queue.orchestrator.generate_vace(
                prompt=task.prompt,
                model_type=task.model,  # Pass model type for parameter resolution
                **generation_params
            )
            gen_call_elapsed = time.time() - gen_call_start
        elif is_flux:
            # For Flux, map video_length to num_images
            if "video_length" in generation_params:
                generation_params["num_images"] = generation_params.pop("video_length")

            gen_call_start = time.time()
            result = queue.orchestrator.generate_flux(
                prompt=task.prompt,
                model_type=task.model,  # Pass model type for parameter resolution
                **generation_params
            )
            gen_call_elapsed = time.time() - gen_call_start
        else:
            # T2V, Z Image, or other models - pass model_type for proper parameter resolution
            # Z Image models are handled as image models by preflight (image_mode=1,
            # video_length=1) so they generate a single frame directly via generate_t2v.
            # Note: WGP stdout is captured to svi_debug.txt file instead of logger
            # to avoid recursion issues
            gen_call_start = time.time()
            result = queue.orchestrator.generate_t2v(
                prompt=task.prompt,
                model_type=task.model,  # CRITICAL: Pass model type for parameter resolution
                **generation_params
            )
            gen_call_elapsed = time.time() - gen_call_start
        queue.logger.debug_block(
            "GENERATE_DONE",
            {
                "strategy": generation_path,
                "duration_s": round(gen_call_elapsed, 2),
                "output": rel_path(result) if result else None,
            },
            task_id=task.id,
        )

        # Post-process single frame videos to PNG for single_image tasks
        # BUT: Skip PNG conversion for travel segments (they must remain as videos for stitching)
        is_travel_segment = task.parameters.get("_source_task_type") == "travel_segment"
        if queue._is_single_image_task(task) and not is_travel_segment:
            png_result = queue._convert_single_frame_video_to_png(task, result, worker_name)
            if png_result:
                return png_result

        return result

    except (RuntimeError, ValueError, OSError, TypeError, AttributeError) as e:
        queue.logger.error(f"[GENERATION] Task {task.id} generation FAILED: {type(e).__name__}: {e}")
        queue.logger.error(f"[GENERATION] Traceback:\n{traceback.format_exc()}")
        raise
    except BaseException as e:
        queue.logger.critical(f"[GENERATION] Task {task.id} generation hit UNEXPECTED exception: {type(e).__name__}: {e}")
        queue.logger.critical(f"[GENERATION] Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # CRITICAL: Restore model patches to prevent contamination across tasks
        if _patch_applied and _parsed_phase_config_for_restore and _model_name_for_restore:
            try:
                from source.core.params.phase_config import restore_model_patches
                restore_model_patches(
                    _parsed_phase_config_for_restore,
                    _model_name_for_restore,
                    task.id
                )
            except (RuntimeError, ImportError, OSError) as restore_error:
                queue.logger.warning(f"[PHASE_CONFIG] Failed to restore model patches for task {task.id}: {restore_error}")

        # Restore svi2pro and sliding_window if we patched them
        if _svi2pro_patched or _wan_model_patched:
            try:
                wgp = get_wgp_runtime_module_mutable()
                model_key = task.model

                # Restore wgp.models_def
                if _svi2pro_patched and model_key in wgp.models_def:
                    if _svi2pro_original is None:
                        wgp.models_def[model_key].pop("svi2pro", None)
                    else:
                        wgp.models_def[model_key]["svi2pro"] = _svi2pro_original
                    # Also restore sliding_window
                    wgp.models_def[model_key].pop("sliding_window", None)
                    queue.logger.debug_anomaly("SVI2PRO", f"Restored wgp.models_def['{model_key}']['svi2pro'] to {_svi2pro_original}", task_id=task.id)

                # Restore wan_model.model_def
                if _wan_model_patched and hasattr(wgp, 'wan_model') and wgp.wan_model is not None:
                    if hasattr(wgp.wan_model, 'model_def') and wgp.wan_model.model_def is not None:
                        if _svi2pro_original is None:
                            wgp.wan_model.model_def.pop("svi2pro", None)
                        else:
                            wgp.wan_model.model_def["svi2pro"] = _svi2pro_original
                        # Also restore sliding_window
                        wgp.wan_model.model_def.pop("sliding_window", None)
                        queue.logger.debug_anomaly("SVI2PRO", f"Restored wan_model.model_def: svi2pro={_svi2pro_original}, sliding_window=removed", task_id=task.id)

            except (RuntimeError, ImportError, AttributeError, KeyError) as restore_error:
                queue.logger.warning(f"[SVI2PRO] Failed to restore svi2pro for task {task.id}: {restore_error}")

        # Cancel the generation timeout alarm (if set and still active) — POSIX only
        if _sigalrm_supported:
            try:
                signal.alarm(0)
            except (ValueError, NameError) as e:
                # Signal not available or not set, silently continue
                queue.logger.debug_anomaly("GENERATION", f"Task {task.id}: Could not cancel timeout alarm: {e}")


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def worker_loop(queue: Any):
    """Main worker loop for processing tasks."""
    import threading
    worker_name = threading.current_thread().name
    queue.logger.debug_anomaly("WORKER_LOOP", f"{worker_name} started")

    iteration_count = 0
    while queue.running and not queue.shutdown_event.is_set():
        iteration_count += 1
        try:
            # Log queue status (every 10 iterations to avoid log spam)
            if iteration_count % 10 == 0:
                queue_size = queue.task_queue.qsize()
                queue.logger.debug_anomaly("WORKER_LOOP", f"{worker_name} iteration {iteration_count}: queue_size={queue_size}")

            # Get next task (blocks with timeout)
            try:
                get_start = time.time()
                priority, timestamp, task = queue.dequeue_task(timeout=1.0)
                get_elapsed = time.time() - get_start
                queue.logger.debug(
                    f"[WORKER_LOOP] {worker_name} retrieved task {task.id} from queue "
                    f"(dequeue_task took {get_elapsed:.3f}s)"
                )
            except queue_mod.Empty:
                continue

            # Process the task
            process_task_impl(queue, task, worker_name)

        except (RuntimeError, ValueError, OSError) as e:
            queue.logger.error(f"[WORKER_LOOP] {worker_name} error in iteration {iteration_count}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            time.sleep(1.0)
        except BaseException as e:
            # Catch any unexpected exceptions to prevent worker from dying silently
            queue.logger.critical(f"[WORKER_LOOP] {worker_name} UNEXPECTED ERROR in iteration {iteration_count}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            time.sleep(1.0)

    queue.logger.debug_anomaly("WORKER_LOOP", f"{worker_name} stopped after {iteration_count} iterations")


# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------

def _monitor_loop(queue: Any):
    """Background monitoring and maintenance loop."""
    queue.logger.debug("Queue monitor started")

    while queue.running and not queue.shutdown_event.is_set():
        try:
            # Monitor loop placeholder - future home for memory/queue/timeout monitoring
            time.sleep(10.0)  # Monitor every 10 seconds

        except (RuntimeError, ValueError, OSError) as e:
            queue.logger.error(f"Monitor error: {e}\n{traceback.format_exc()}")
            time.sleep(5.0)

    queue.logger.debug("Queue monitor stopped")
