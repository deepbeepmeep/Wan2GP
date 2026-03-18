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

from source.core.log import queue_logger


# ---------------------------------------------------------------------------
# Thread safety for WGP global monkey-patching
# ---------------------------------------------------------------------------
_wgp_patch_lock = threading.Lock()


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
    except (ImportError, AttributeError, TypeError) as e:
        # Swallowing is intentional: logging context is optional and failures are debug-only
        logging.getLogger('HeadlessQueue').debug(f"Failed to set task logging context for {task.id}: {e}")

    with queue.queue_lock:
        queue.current_task = task
        task.status = "processing"

    queue.logger.info(f"[TASK_PROCESSING] {worker_name} processing task {task.id} (model: {task.model})")
    start_time = time.time()

    try:
        # 1. Ensure correct model is loaded (orchestrator checks WGP's ground truth)
        queue.logger.debug(f"[TASK_PROCESSING] Task {task.id}: Phase 1 - Switching to model {task.model}")
        switch_start = time.time()
        queue._switch_model(task.model, worker_name)
        switch_elapsed = time.time() - switch_start
        queue.logger.debug(f"[TASK_PROCESSING] Task {task.id}: Phase 1 complete - Model switch took {switch_elapsed:.2f}s")

        # 2. Reset billing start time now that model is loaded
        # This ensures users aren't charged for model loading time
        queue.logger.debug(f"[TASK_PROCESSING] Task {task.id}: Phase 2 - Resetting billing")
        try:
            from source.core.db.task_status import reset_generation_started_at
            reset_generation_started_at(task.id)
            queue.logger.debug(f"[TASK_PROCESSING] Task {task.id}: Phase 2 complete - Billing reset succeeded")
        except (OSError, ValueError, RuntimeError) as e_billing:
            # Don't fail the task if billing reset fails - just log it
            queue.logger.warning(f"[BILLING] Failed to reset generation_started_at for task {task.id}: {e_billing}")

        # 3. Delegate actual generation to orchestrator
        # The orchestrator handles the heavy lifting while we manage the queue
        queue.logger.debug(f"[TASK_PROCESSING] Task {task.id}: Phase 3 - Starting generation")
        gen_start = time.time()
        result_path = queue._execute_generation(task, worker_name)
        gen_elapsed = time.time() - gen_start
        queue.logger.debug(f"[TASK_PROCESSING] Task {task.id}: Phase 3 complete - Generation took {gen_elapsed:.2f}s, result: {result_path}")

        # Verify we're still in Wan2GP directory after generation
        current_dir = os.getcwd()
        if "Wan2GP" not in current_dir:
            queue.logger.warning(
                f"[PATH_CHECK] After generation: Current directory changed!\n"
                f"  Current: {current_dir}\n"
                f"  Expected: Should contain 'Wan2GP'\n"
                f"  This may cause issues for subsequent tasks!"
            )
        else:
            queue.logger.debug(f"[PATH_CHECK] After generation: Still in Wan2GP")

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
                queue.logger.info(f"Task {task.id} completed in {processing_time:.1f}s: {result_path}")
            else:
                task.status = "failed"
                task.error_message = "No output generated"
                queue.stats["tasks_failed"] += 1
                queue.logger.error(f"Task {task.id} failed after {processing_time:.1f}s: No output generated")

        # Memory cleanup after each task (does NOT unload models)
        # This clears PyTorch's internal caches and Python garbage to prevent fragmentation
        queue._cleanup_memory_after_task(task.id)

    except (RuntimeError, ValueError, OSError, TypeError, AttributeError) as e:
        # Handle task failure
        processing_time = time.time() - start_time
        error_message_str = str(e)

        with queue.queue_lock:
            task.status = "failed"
            task.error_message = error_message_str
            task.processing_time = processing_time
            queue.stats["tasks_failed"] += 1

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
        except (ImportError, AttributeError, TypeError) as e:
            # Swallowing is intentional: logging context is optional and failures are debug-only
            logging.getLogger('HeadlessQueue').debug(f"Failed to clear task logging context: {e}")


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

    queue.logger.info(f"{worker_name} executing generation for task {task.id} (model: {task.model})")

    # Convert task parameters to WanOrchestrator format
    wgp_params = queue._convert_to_wgp_task(task)

    # Remove model and prompt from params since they're passed separately to avoid duplication
    generation_params = {k: v for k, v in wgp_params.items() if k not in ("model", "prompt")}

    # DEBUG: Log all parameter keys to verify _parsed_phase_config is present
    queue.logger.info(f"[PHASE_CONFIG_DEBUG] Task {task.id}: generation_params keys: {list(generation_params.keys())}")

    # CRITICAL: Apply phase_config patches NOW in the worker thread where wgp is imported
    # Acquire lock to prevent concurrent tasks from corrupting shared wgp globals
    queue.logger.info(f"[LOCK_ACQUIRE] Task {task.id} attempting to acquire _wgp_patch_lock...")
    lock_acquire_start = time.time()
    with _wgp_patch_lock:
        lock_acquired_elapsed = time.time() - lock_acquire_start
        if lock_acquired_elapsed > 0.1:
            queue.logger.warning(f"[LOCK_ACQUIRE] Task {task.id} waited {lock_acquired_elapsed:.3f}s for _wgp_patch_lock (possible contention)")
        else:
            queue.logger.debug(f"[LOCK_ACQUIRE] Task {task.id} acquired _wgp_patch_lock in {lock_acquired_elapsed:.3f}s")
        try:
            return _execute_generation_with_patches(queue, task, worker_name, generation_params)
        finally:
            queue.logger.debug(f"[LOCK_RELEASE] Task {task.id} releasing _wgp_patch_lock")


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

        queue.logger.info(f"[PHASE_CONFIG] Applying model patch in GenerationWorker for '{model_name}'")

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
            import wgp
            model_key = task.model

            # Patch 1: wgp.models_def (for any new model loads)
            if model_key in wgp.models_def:
                _svi2pro_original = wgp.models_def[model_key].get("svi2pro")
                wgp.models_def[model_key]["svi2pro"] = True
                _svi2pro_patched = True
                queue.logger.info(f"[SVI2PRO] Patched wgp.models_def['{model_key}']['svi2pro'] = True (was: {_svi2pro_original})", task_id=task.id)

            # Patch 2: wan_model.model_def DIRECTLY (the actual object used during generation)
            # This is critical because the model was loaded BEFORE we patched models_def
            if hasattr(wgp, 'wan_model') and wgp.wan_model is not None:
                if hasattr(wgp.wan_model, 'model_def') and wgp.wan_model.model_def is not None:
                    # Diagnostic: check if they're the same object
                    models_def_obj = wgp.models_def.get(model_key)
                    wan_model_def_obj = wgp.wan_model.model_def
                    same_object = models_def_obj is wan_model_def_obj
                    queue.logger.info(f"[SVI2PRO_DIAG] models_def['{model_key}'] id={id(models_def_obj)}, wan_model.model_def id={id(wan_model_def_obj)}, same_object={same_object}", task_id=task.id)

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
                        queue.logger.info(f"[SVI2PRO] Patched wgp.models_def['{model_key}']['sliding_window'] = True (was: {_sw_before})", task_id=task.id)

                    _wan_model_patched = True

                    # Verify the patches took effect
                    verify_svi2pro = wgp.wan_model.model_def.get("svi2pro")
                    verify_sliding = wgp.wan_model.model_def.get("sliding_window")
                    verify_defaults = wgp.wan_model.model_def.get("sliding_window_defaults")
                    queue.logger.info(f"[SVI2PRO] Patched wan_model.model_def: svi2pro={verify_svi2pro}, sliding_window={verify_sliding}, sliding_window_defaults={verify_defaults} (was: {_sliding_window_original})", task_id=task.id)
                else:
                    queue.logger.warning(f"[SVI2PRO] wan_model exists but has no model_def", task_id=task.id)
            else:
                queue.logger.warning(f"[SVI2PRO] wgp.wan_model not found - model may not be loaded yet", task_id=task.id)

        except (RuntimeError, AttributeError, ImportError, KeyError) as e:
            queue.logger.warning(f"[SVI2PRO] Failed to patch svi2pro: {e}", task_id=task.id)
        # Remove from generation_params since it's not a generate() parameter
        generation_params.pop("svi2pro", None)

    # Log generation parameters for debugging
    queue_logger.debug(f"[GENERATION_DEBUG] Task {task.id}: Generation parameters:", task_id=task.id)
    for key, value in generation_params.items():
        if key in ["video_guide", "video_mask", "image_refs"]:
            queue_logger.debug(f"[GENERATION_DEBUG]   {key}: {value}", task_id=task.id)
        elif key in ["video_length", "resolution", "num_inference_steps"]:
            queue_logger.debug(f"[GENERATION_DEBUG]   {key}: {value}", task_id=task.id)

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

    try:
        # Set timeout alarm (will raise SIGALRM)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(GENERATION_TIMEOUT_SECONDS)
            queue.logger.debug(f"[GENERATION] Task {task.id}: Set {GENERATION_TIMEOUT_SECONDS}s timeout alarm")
        except (ValueError, OSError) as e:
            # Signal not available on all platforms (e.g., Windows), silently continue
            queue.logger.debug(f"[GENERATION] Task {task.id}: Timeout not available on this platform: {e}")

        # Check if model supports VACE features
        queue.logger.debug(f"[GENERATION] Task {task.id}: Checking if model supports VACE...")
        model_supports_vace = queue._model_supports_vace(task.model)
        queue.logger.debug(f"[GENERATION] Task {task.id}: model_supports_vace={model_supports_vace}")

        if model_supports_vace:
            queue.logger.info(f"[GENERATION] Task {task.id}: Using VACE generation path for model {task.model}")

            # CRITICAL: VACE models require a video_guide parameter
            if "video_guide" in generation_params and generation_params["video_guide"]:
                queue.logger.debug(f"[GENERATION] Task {task.id}: Video guide provided: {generation_params['video_guide']}")
            else:
                error_msg = f"VACE model '{task.model}' requires a video_guide parameter but none was provided. VACE models cannot perform pure text-to-video generation."
                queue.logger.error(f"[GENERATION] Task {task.id}: {error_msg}")
                raise ValueError(error_msg)

            queue.logger.debug(f"[GENERATION] Task {task.id}: Calling orchestrator.generate_vace()")
            gen_call_start = time.time()
            result = queue.orchestrator.generate_vace(
                prompt=task.prompt,
                model_type=task.model,  # Pass model type for parameter resolution
                **generation_params
            )
            gen_call_elapsed = time.time() - gen_call_start
            queue.logger.debug(f"[GENERATION] Task {task.id}: generate_vace() returned after {gen_call_elapsed:.2f}s: {result}")
        elif queue.orchestrator._is_flux():
            queue.logger.info(f"[GENERATION] Task {task.id}: Using Flux generation path for model {task.model}")

            # For Flux, map video_length to num_images
            if "video_length" in generation_params:
                generation_params["num_images"] = generation_params.pop("video_length")
                queue.logger.debug(f"[GENERATION] Task {task.id}: Mapped video_length to num_images")

            queue.logger.debug(f"[GENERATION] Task {task.id}: Calling orchestrator.generate_flux()")
            gen_call_start = time.time()
            result = queue.orchestrator.generate_flux(
                prompt=task.prompt,
                model_type=task.model,  # Pass model type for parameter resolution
                **generation_params
            )
            gen_call_elapsed = time.time() - gen_call_start
            queue.logger.debug(f"[GENERATION] Task {task.id}: generate_flux() returned after {gen_call_elapsed:.2f}s: {result}")
        else:
            queue.logger.info(f"[GENERATION] Task {task.id}: Using T2V generation path for model {task.model}")

            # T2V or other models - pass model_type for proper parameter resolution
            # Note: WGP stdout is captured to svi_debug.txt file instead of logger
            # to avoid recursion issues
            queue.logger.debug(f"[GENERATION] Task {task.id}: Calling orchestrator.generate_t2v()")
            gen_call_start = time.time()
            result = queue.orchestrator.generate_t2v(
                prompt=task.prompt,
                model_type=task.model,  # CRITICAL: Pass model type for parameter resolution
                **generation_params
            )
            gen_call_elapsed = time.time() - gen_call_start
            queue.logger.debug(f"[GENERATION] Task {task.id}: generate_t2v() returned after {gen_call_elapsed:.2f}s: {result}")

        queue.logger.info(f"{worker_name} generation completed for task {task.id}: {result}")

        # Post-process single frame videos to PNG for single_image tasks
        # BUT: Skip PNG conversion for travel segments (they must remain as videos for stitching)
        is_travel_segment = task.parameters.get("_source_task_type") == "travel_segment"
        if queue._is_single_image_task(task) and not is_travel_segment:
            png_result = queue._convert_single_frame_video_to_png(task, result, worker_name)
            if png_result:
                queue.logger.info(f"{worker_name} converted single frame video to PNG: {png_result}")
                return png_result

        queue.logger.debug(f"[GENERATION] Task {task.id}: Total generation execution completed successfully")
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
        queue.logger.debug(f"[GENERATION] Task {task.id}: Entering finally block for cleanup")

        # CRITICAL: Restore model patches to prevent contamination across tasks
        if _patch_applied and _parsed_phase_config_for_restore and _model_name_for_restore:
            queue.logger.debug(f"[GENERATION] Task {task.id}: Restoring phase config patches for model '{_model_name_for_restore}'")
            try:
                from source.core.params.phase_config import restore_model_patches
                restore_start = time.time()
                restore_model_patches(
                    _parsed_phase_config_for_restore,
                    _model_name_for_restore,
                    task.id
                )
                restore_elapsed = time.time() - restore_start
                queue.logger.info(f"[PHASE_CONFIG] Restored original model definition for '{_model_name_for_restore}' after task {task.id} (took {restore_elapsed:.3f}s)")
            except (RuntimeError, ImportError, OSError) as restore_error:
                queue.logger.warning(f"[PHASE_CONFIG] Failed to restore model patches for task {task.id}: {restore_error}")

        # Restore svi2pro and sliding_window if we patched them
        if _svi2pro_patched or _wan_model_patched:
            queue.logger.debug(f"[GENERATION] Task {task.id}: Restoring svi2pro/sliding_window patches")
            try:
                import wgp
                model_key = task.model

                # Restore wgp.models_def
                if _svi2pro_patched and model_key in wgp.models_def:
                    if _svi2pro_original is None:
                        wgp.models_def[model_key].pop("svi2pro", None)
                    else:
                        wgp.models_def[model_key]["svi2pro"] = _svi2pro_original
                    # Also restore sliding_window
                    wgp.models_def[model_key].pop("sliding_window", None)
                    queue.logger.info(f"[SVI2PRO] Restored wgp.models_def['{model_key}']['svi2pro'] to {_svi2pro_original}", task_id=task.id)

                # Restore wan_model.model_def
                if _wan_model_patched and hasattr(wgp, 'wan_model') and wgp.wan_model is not None:
                    if hasattr(wgp.wan_model, 'model_def') and wgp.wan_model.model_def is not None:
                        if _svi2pro_original is None:
                            wgp.wan_model.model_def.pop("svi2pro", None)
                        else:
                            wgp.wan_model.model_def["svi2pro"] = _svi2pro_original
                        # Also restore sliding_window
                        wgp.wan_model.model_def.pop("sliding_window", None)
                        queue.logger.info(f"[SVI2PRO] Restored wan_model.model_def: svi2pro={_svi2pro_original}, sliding_window=removed", task_id=task.id)

            except (RuntimeError, ImportError, AttributeError, KeyError) as restore_error:
                queue.logger.warning(f"[SVI2PRO] Failed to restore svi2pro for task {task.id}: {restore_error}")

        # Cancel the generation timeout alarm (if set and still active)
        queue.logger.debug(f"[GENERATION] Task {task.id}: Canceling timeout alarm")
        try:
            signal.alarm(0)
            queue.logger.debug(f"[GENERATION] Task {task.id}: Timeout alarm canceled")
        except (ValueError, NameError) as e:
            # Signal not available or not set, silently continue
            queue.logger.debug(f"[GENERATION] Task {task.id}: Could not cancel timeout alarm: {e}")

        queue.logger.debug(f"[GENERATION] Task {task.id}: Finally block cleanup complete")


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def worker_loop(queue: Any):
    """Main worker loop for processing tasks."""
    import threading
    worker_name = threading.current_thread().name
    queue.logger.info(f"{worker_name} started")

    iteration_count = 0
    while queue.running and not queue.shutdown_event.is_set():
        iteration_count += 1
        try:
            # Log queue status (every 10 iterations to avoid log spam)
            if iteration_count % 10 == 0:
                queue_size = queue.task_queue.qsize()
                queue.logger.debug(f"[WORKER_LOOP] {worker_name} iteration {iteration_count}: queue_size={queue_size}")

            # Get next task (blocks with timeout)
            try:
                queue.logger.debug(f"[WORKER_LOOP] {worker_name} attempting to get task from queue...")
                get_start = time.time()
                priority, timestamp, task = queue.task_queue.get(timeout=1.0)
                get_elapsed = time.time() - get_start
                queue.logger.info(f"[WORKER_LOOP] {worker_name} retrieved task {task.id} from queue (queue.get took {get_elapsed:.3f}s)")
            except queue_mod.Empty:
                queue.logger.debug(f"[WORKER_LOOP] {worker_name} queue empty, continuing...")
                continue

            # Process the task
            queue.logger.info(f"[WORKER_LOOP] {worker_name} starting process_task_impl for {task.id}")
            process_task_impl(queue, task, worker_name)
            queue.logger.info(f"[WORKER_LOOP] {worker_name} completed process_task_impl for {task.id}")

        except (RuntimeError, ValueError, OSError) as e:
            queue.logger.error(f"[WORKER_LOOP] {worker_name} error in iteration {iteration_count}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            time.sleep(1.0)
        except BaseException as e:
            # Catch any unexpected exceptions to prevent worker from dying silently
            queue.logger.critical(f"[WORKER_LOOP] {worker_name} UNEXPECTED ERROR in iteration {iteration_count}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            time.sleep(1.0)

    queue.logger.info(f"[WORKER_LOOP] {worker_name} stopped after {iteration_count} iterations")


# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------

def _monitor_loop(queue: Any):
    """Background monitoring and maintenance loop."""
    queue.logger.info("Queue monitor started")

    while queue.running and not queue.shutdown_event.is_set():
        try:
            # Monitor loop placeholder - future home for memory/queue/timeout monitoring
            time.sleep(10.0)  # Monitor every 10 seconds

        except (RuntimeError, ValueError, OSError) as e:
            queue.logger.error(f"Monitor error: {e}\n{traceback.format_exc()}")
            time.sleep(5.0)

    queue.logger.info("Queue monitor stopped")
