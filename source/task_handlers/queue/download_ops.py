"""
Model-switching and task-conversion logic extracted from HeadlessTaskQueue.

Every public function in this module takes the ``HeadlessTaskQueue`` instance
(aliased *queue*) as its first argument so that it can access ``queue.logger``,
``queue.orchestrator``, ``queue.current_model``, etc.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

from source.core.log import is_debug_enabled
from source.runtime.process_globals import temporary_process_globals


# ---------------------------------------------------------------------------
# Model switching
# ---------------------------------------------------------------------------

def switch_model_impl(queue: "HeadlessTaskQueue", model_key: str, worker_name: str) -> bool:
    """
    Ensure the correct model is loaded using wgp.py's model management.

    This leverages the orchestrator's model loading while tracking
    the change in our queue system. The orchestrator checks WGP's ground truth
    (wgp.transformer_type) to determine if a switch is actually needed.

    Returns:
        bool: True if a model switch actually occurred, False if already loaded
    """
    # Ensure orchestrator is initialized before switching models
    queue._ensure_orchestrator()

    switch_start = time.time()

    try:
        # Use orchestrator's model loading - it checks WGP's ground truth
        # and returns whether a switch actually occurred
        switched = queue.orchestrator.load_model(model_key)

        if switched:
            # Only do switch-specific actions if a switch actually occurred
            queue.logger.debug(f"{worker_name} switched model: {queue.current_model} → {model_key}")

            queue.stats["model_switches"] += 1
            switch_time = time.time() - switch_start
            queue.logger.debug(f"Model switch completed in {switch_time:.1f}s")

        # Always sync our tracking with orchestrator's state
        queue.current_model = model_key
        return switched

    except (RuntimeError, ValueError, OSError) as e:
        queue.logger.error(f"Model switch failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Task conversion
# ---------------------------------------------------------------------------

def convert_to_wgp_task_impl(queue: "HeadlessTaskQueue", task: "GenerationTask") -> Dict[str, Any]:
    """
    Convert task to WGP parameters using typed TaskConfig.

    This:
    1. Parses all params into TaskConfig at the boundary
    2. Handles LoRA downloads via LoRAConfig
    3. Converts to WGP format only at the end
    """
    from source.core.params import TaskConfig

    queue.logger.debug_block(
        "CONVERT_TASK",
        {
            "task_id": task.id,
            "model": task.model,
            "param_keys": list(task.parameters.keys()),
        },
    )

    # Parse into typed config
    try:
        config = TaskConfig.from_db_task(
            task.parameters,
            task_id=task.id,
            task_type=task.parameters.get('_source_task_type', ''),
            model=task.model,
            debug_mode=is_debug_enabled()
        )
    except Exception as e:
        queue.logger.error(f"[CONVERT_DEBUG] Failed to parse TaskConfig: {type(e).__name__}: {e}")
        import traceback
        queue.logger.error(f"[CONVERT_DEBUG] Traceback:\n{traceback.format_exc()}")
        raise

    # Add prompt and model
    config.generation.prompt = task.prompt
    config.model = task.model

    # Handle LoRA downloads if any are pending
    if config.lora.has_pending_downloads():
        queue.logger.debug_anomaly("LORA_PROCESS", f"Task {task.id}: {len(config.lora.get_pending_downloads())} LoRAs need downloading")

        with temporary_process_globals(cwd=queue.wan_dir):
            from source.models.lora.lora_utils import download_lora_from_url

            for url, mult in list(config.lora.get_pending_downloads().items()):
                try:
                    local_path = download_lora_from_url(url, task.id, model_type=task.model)
                    if local_path:
                        config.lora.mark_downloaded(url, local_path)
                        queue.logger.debug_anomaly("LORA_DOWNLOAD", f"Task {task.id}: Downloaded {os.path.basename(local_path)}")
                    else:
                        queue.logger.warning(f"[LORA_DOWNLOAD] Task {task.id}: Failed to download {url}")
                except (OSError, ValueError, RuntimeError) as e:
                    queue.logger.warning(f"[LORA_DOWNLOAD] Task {task.id}: Error downloading {url}: {e}")

    # Validate before conversion
    errors = config.validate()
    if errors:
        queue.logger.warning(f"[TASK_CONFIG] Task {task.id}: Validation warnings: {errors}")

    # Convert to WGP format (single conversion point)
    wgp_params = config.to_wgp_format()

    # Ensure prompt and model are set
    wgp_params["prompt"] = task.prompt
    wgp_params["model"] = task.model

    # Filter out infrastructure params
    for param in ["supabase_url", "supabase_anon_key", "supabase_access_token"]:
        wgp_params.pop(param, None)

    return wgp_params
