"""
WGP Parameter Resolution

Resolves generation parameters with explicit precedence:
1. Task explicit parameters (highest priority)
2. Model JSON configuration (medium priority)
3. System defaults (lowest priority)
"""

from source.core.log import (
    generation_logger,
    safe_dict_repr,
    safe_log_change,
)
from source.runtime.wgp_bridge import get_default_settings


def resolve_parameters(orchestrator, model_type: str, task_params: dict) -> dict:
    """
    Resolve generation parameters with explicit precedence:
    1. Task explicit parameters (highest priority)
    2. Model JSON configuration (medium priority)
    3. System defaults (lowest priority)

    Args:
        orchestrator: WanOrchestrator instance (used for logger access; currently
                      unused beyond that, but kept for future flexibility)
        model_type: Model identifier (e.g., "optimised-t2i")
        task_params: Parameters explicitly provided by the task/user

    Returns:
        Resolved parameter dictionary with proper precedence
    """
    # 1. Start with system defaults (lowest priority) - migrated from worker.py
    resolved_params = {
        "resolution": "1280x720",
        "video_length": 49,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "guidance2_scale": 7.5,
        "flow_shift": 7.0,
        "sample_solver": "euler",
        "switch_threshold": 500,
        "seed": -1,  # Random seed (matches worker.py behavior)
        "negative_prompt": "",
        "activated_loras": [],
        "loras_multipliers": "",
    }

    # 2. Apply model JSON configuration (medium priority)
    try:
        model_defaults = get_default_settings(model_type)
        generation_logger.debug(
            f"Model defaults for '{model_type}': "
            f"{safe_dict_repr(model_defaults) if model_defaults else 'None'}"
        )

        if model_defaults:
            model_items = list(model_defaults.items())
            applied_model_defaults = 0
            skipped_model_defaults = 0

            for idx, (param, value) in enumerate(model_items):
                # JSON passthrough mode: Allow activated_loras and loras_multipliers to pass directly
                if param not in ["prompt"]:
                    try:
                        old_value = resolved_params.get(param, "NOT_SET")
                    except Exception as e:
                        # Catch the exact error and log full context
                        generation_logger.critical(f"[CRASH_POINT] LOOP [{idx+1}] failed at resolved_params.get('{param}')")
                        generation_logger.critical(f"[CRASH_POINT] Error type: {type(e).__name__}: {e}")
                        generation_logger.critical(f"[CRASH_POINT] resolved_params is: {resolved_params}")
                        generation_logger.critical(f"[CRASH_POINT] resolved_params type: {type(resolved_params)}")
                        import traceback
                        generation_logger.critical(f"[CRASH_POINT] Full traceback:\n{traceback.format_exc()}")
                        raise

                    # DEFENSIVE: Skip None values from model defaults (they shouldn't override)
                    if value is None:
                        skipped_model_defaults += 1
                        continue

                    resolved_params[param] = value
                    applied_model_defaults += 1
                    # Safe logging: Use safe_log_change to prevent hanging on large values
                    try:
                        change_log = safe_log_change(param, old_value, value)
                        generation_logger.debug(change_log)
                    except Exception as log_e:
                        # Catch logging errors and log them separately
                        generation_logger.critical(f"[CRASH_POINT] LOOP [{idx+1}] failed at safe_log_change('{param}')")
                        generation_logger.critical(f"[CRASH_POINT] Error type: {type(log_e).__name__}: {log_e}")
                        generation_logger.critical(f"[CRASH_POINT] old_value: {old_value}, type: {type(old_value)}")
                        generation_logger.critical(f"[CRASH_POINT] value: {value}, type: {type(value)}")
                        import traceback
                        generation_logger.critical(f"[CRASH_POINT] Full traceback:\n{traceback.format_exc()}")
                        raise
                else:
                    skipped_model_defaults += 1

            generation_logger.debug_block(
                "MODEL_DEFAULTS",
                {
                    "model": model_type,
                    "applied": applied_model_defaults,
                    "skipped": skipped_model_defaults,
                    "total": len(model_items),
                }
            )
        else:
            generation_logger.warning(f"No model configuration found for '{model_type}'")

    except (ValueError, KeyError, TypeError) as e:
        generation_logger.warning(f"Could not load model configuration for '{model_type}': {e}")

    # 3. Apply task explicit parameters (highest priority)
    generation_logger.debug(
        f"Task explicit parameters for '{model_type}': {safe_dict_repr(task_params)}"
    )

    for param, value in task_params.items():
        if value is not None:  # Don't override with None values
            old_value = resolved_params.get(param, "NOT_SET")
            resolved_params[param] = value
            generation_logger.debug(safe_log_change(param, old_value, value))

    generation_logger.debug(
        f"Parameter resolution for '{model_type}': "
        f"task_overrides={sum(1 for value in task_params.values() if value is not None)}, "
        f"resolved_keys={list(resolved_params.keys())}"
    )
    return resolved_params


def query_resolve_parameters(orchestrator, model_type: str, task_params: dict) -> dict:
    """Compatibility alias for resolve_parameters."""
    return resolve_parameters(orchestrator=orchestrator, model_type=model_type, task_params=task_params)
