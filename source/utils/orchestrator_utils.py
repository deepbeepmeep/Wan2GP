"""Orchestrator parameter extraction and failure reporting utilities."""

from source.core.db.config import STATUS_FAILED
from source.core.db.task_status import update_task_status
from source.core.log import headless_logger

__all__ = [
    "extract_orchestrator_parameters",
    "report_orchestrator_failure",
]

def extract_orchestrator_parameters(db_task_params: dict, task_id: str = "unknown") -> dict:
    """
    Centralized extraction of parameters from orchestrator_details.

    This handles the common pattern where task parameters contain nested orchestrator_details
    that need to be extracted and flattened into the main parameter space.

    Callers and field usage:
    - source.task_handlers.magic_edit._handle_magic_edit_task:
      image_url, prompt, resolution, seed, in_scene
    - source.task_handlers.tasks.task_conversion.db_task_to_generation_task:
      phase_config, additional_loras
    - source.task_handlers.travel.orchestrator.handle_travel_orchestrator_task:
      carries prompt/negative_prompt, resolution, seed/model aliases,
      guidance + phase fields, LoRA metadata, amount_of_motion, and phase_config
      onto child segment payloads for downstream compatibility

    Args:
        db_task_params: Raw task parameters from database
        task_id: Task ID for logging

    Returns:
        Dict with extracted parameters added at the top level
    """
    extracted_params = db_task_params.copy()

    orchestrator_details = db_task_params.get("orchestrator_details", {})
    if orchestrator_details:
        # Extract specific parameters that should be available at top level
        extraction_map = {
            "additional_loras": "additional_loras",
            "prompt": "prompt",
            "negative_prompt": "negative_prompt",
            "resolution": "resolution",
            "video_length": "video_length",
            "seed": "seed",
            "model": "model",
            "num_inference_steps": "num_inference_steps",
            "guidance_scale": "guidance_scale",
            "guidance2_scale": "guidance2_scale",
            "guidance3_scale": "guidance3_scale",
            "guidance_phases": "guidance_phases",
            "flow_shift": "flow_shift",
            "switch_threshold": "switch_threshold",
            "switch_threshold2": "switch_threshold2",
            "model_switch_phase": "model_switch_phase",
            "sample_solver": "sample_solver",
            # LoRA parameters from phase_config
            "lora_names": "lora_names",
            "lora_multipliers": "lora_multipliers",
            "activated_loras": "activated_loras",
            # Magic edit parameters
            "image_url": "image_url",
            "in_scene": "in_scene",
            "amount_of_motion": "amount_of_motion",
            "phase_config": "phase_config",
        }

        extracted_count = 0
        for orchestrator_key, param_key in extraction_map.items():
            if orchestrator_key in orchestrator_details:
                # Only extract if not already present at top level (top level takes precedence)
                if param_key not in extracted_params:
                    value = orchestrator_details[orchestrator_key]
                    # For additional_loras, only extract if it has actual entries (not empty dict)
                    if orchestrator_key == "additional_loras" and not value:
                        continue
                    extracted_params[param_key] = value
                    extracted_count += 1

        # Note: orchestrator_details is already in db_task_params, no need to duplicate

        if extracted_count > 0:
            headless_logger.debug(f"Task {task_id}: extracted {extracted_count} params from orchestrator_details", task_id=task_id)

    return extracted_params

def report_orchestrator_failure(task_params_dict: dict, error_msg: str) -> None:
    """Update the parent orchestrator task to FAILED when a sub-task encounters a fatal error.

    Args:
        task_params_dict: The params payload of the *current* sub-task.
            It is expected to contain a reference to the orchestrator via one
            of the standard keys (e.g. ``orchestrator_task_id_ref``).
        error_msg: Human-readable message describing the failure.
    """
    orchestrator_id = None
    # Common payload keys that may reference the orchestrator task
    for key in (
        "orchestrator_task_id_ref",
        "orchestrator_task_id",
        "parent_orchestrator_task_id",
        "orchestrator_id"):
        orchestrator_id = task_params_dict.get(key)
        if orchestrator_id:
            break

    if not orchestrator_id:
        headless_logger.warning(
            f"[report_orchestrator_failure] No orchestrator reference found in payload. Message: {error_msg}"
        )
        return

    # Truncate very long messages to avoid DB column overflow
    truncated_msg = error_msg[:500]

    try:
        update_task_status(
            orchestrator_id,
            STATUS_FAILED,
            truncated_msg)
        headless_logger.debug(
            f"[report_orchestrator_failure] Marked orchestrator task {orchestrator_id} as FAILED with message: {truncated_msg}"
        )
    except (OSError, ValueError, RuntimeError) as e_update:  # pragma: no cover
        headless_logger.error(
            f"[report_orchestrator_failure] Failed to update orchestrator status for {orchestrator_id}: {e_update}"
        )
