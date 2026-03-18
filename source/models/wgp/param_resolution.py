"""
WGP Parameter Resolution

Resolves generation parameters with explicit precedence:
1. Task explicit parameters (highest priority)
2. Model JSON configuration (medium priority)
3. System defaults (lowest priority)
"""

from source.core.log import (
    generation_logger,
    safe_dict_repr, safe_log_params, safe_log_change
)


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
        import wgp
        model_defaults = wgp.get_default_settings(model_type)
        # Safe logging: Use safe_dict_repr to prevent hanging
        generation_logger.debug(f"get_default_settings('{model_type}') returned: {safe_dict_repr(model_defaults) if model_defaults else 'None'}")
        generation_logger.debug(f"Type: {type(model_defaults)}")

        if model_defaults:
            # DIAGNOSTIC: Log model_defaults structure before iteration
            generation_logger.info(f"🔍 DIAGNOSTIC: model_defaults has {len(model_defaults)} parameters")
            generation_logger.info(f"🔍 DIAGNOSTIC: model_defaults keys: {list(model_defaults.keys())}")
            generation_logger.info(f"🔍 DIAGNOSTIC: model_defaults is type: {type(model_defaults)}")
            generation_logger.info(f"🔍 DIAGNOSTIC: model_defaults id: {id(model_defaults)}")

            # AGGRESSIVE: Dump full model_defaults values
            generation_logger.info(f"🔍 AGGRESSIVE_DUMP: model_defaults full contents:")
            for key in list(model_defaults.keys())[:10]:  # First 10 for safety
                val = model_defaults.get(key)
                generation_logger.info(f"  {key}: {type(val).__name__} = {repr(val)[:200]}")
            generation_logger.info(f"  ... and {max(0, len(model_defaults) - 10)} more")

            # Safe logging: Only show keys before applying model config
            generation_logger.debug(f"Before applying model config - resolved_params keys: {list(resolved_params.keys())}")

            # DEFENSIVE: Create a snapshot of items to prevent iterator invalidation
            model_items = list(model_defaults.items())
            generation_logger.info(f"🔍 DIAGNOSTIC: Created snapshot of {len(model_items)} items for iteration")

            for idx, (param, value) in enumerate(model_items):
                # DIAGNOSTIC: Log progress every item to pinpoint exact freeze location
                generation_logger.info(f"🔍 LOOP [{idx+1}/{len(model_items)}]: Processing param='{param}', value_type={type(value).__name__}")

                # JSON passthrough mode: Allow activated_loras and loras_multipliers to pass directly
                if param not in ["prompt"]:
                    generation_logger.debug(f"🔍 LOOP [{idx+1}]: Getting old value for '{param}'")

                    # DEEP DIAGNOSTIC: Log resolved_params state before .get() call
                    try:
                        generation_logger.debug(f"[DEEP_DIAG] resolved_params type: {type(resolved_params).__name__}")
                        generation_logger.debug(f"[DEEP_DIAG] resolved_params is None: {resolved_params is None}")
                        if resolved_params is not None:
                            generation_logger.debug(f"[DEEP_DIAG] resolved_params keys count: {len(resolved_params)}")
                            generation_logger.debug(f"[DEEP_DIAG] param '{param}' in resolved_params: {param in resolved_params}")
                    except Exception as diag_e:
                        generation_logger.error(f"[DEEP_DIAG] Failed to log resolved_params state: {diag_e}")

                    # AGGRESSIVE: Dump current resolved_params dict around critical parameters
                    if param in ['guidance_phases', 'guidance_scale', 'guidance2_scale', 'guidance3_scale']:
                        generation_logger.info(f"[AGGRESSIVE] LOOP [{idx+1}] PRE-PROCESSING resolved_params snapshot:")
                        generation_logger.info(f"  Type: {type(resolved_params).__name__}")
                        if isinstance(resolved_params, dict):
                            guidance_keys = [k for k in resolved_params.keys() if 'guidance' in k or 'switch' in k or 'phase' in k]
                            for k in guidance_keys:
                                v = resolved_params.get(k)
                                generation_logger.info(f"  {k}: {type(v).__name__} = {repr(v)[:100]}")
                        else:
                            generation_logger.info(f"  WARNING: resolved_params is {type(resolved_params)}, not dict!")

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
                        generation_logger.debug(f"⏭️  LOOP [{idx+1}]: Skipped '{param}' - model default is None (invalid)")
                        continue

                    generation_logger.debug(f"🔍 LOOP [{idx+1}]: Assigning new value for '{param}'")
                    resolved_params[param] = value

                    # AGGRESSIVE: Log after assignment
                    if param in ['guidance_phases', 'guidance_scale', 'guidance2_scale', 'guidance3_scale']:
                        generation_logger.info(f"[AGGRESSIVE] LOOP [{idx+1}] POST-ASSIGNMENT:")
                        generation_logger.info(f"  resolved_params['{param}'] = {repr(resolved_params.get(param))[:100]}")
                        generation_logger.info(f"  old_value was: {repr(old_value)[:100]}")
                        generation_logger.info(f"  value param was: {repr(value)[:100]}")

                    generation_logger.debug(f"🔍 LOOP [{idx+1}]: Logging change for '{param}'")
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

                    generation_logger.debug(f"✅ LOOP [{idx+1}]: Completed '{param}'")
                else:
                    generation_logger.info(f"⏭️  LOOP [{idx+1}]: Skipped '{param}' (excluded)")

            generation_logger.info(f"✅ DIAGNOSTIC: Loop completed successfully, processed {len(model_items)} items")

            # Safe logging: Only show keys after applying model config
            generation_logger.debug(f"After applying model config - resolved_params keys: {list(resolved_params.keys())}")
            generation_logger.debug(f"Applied model config for '{model_type}': {len(model_defaults)} parameters")
        else:
            generation_logger.warning(f"No model configuration found for '{model_type}'")

    except (ValueError, KeyError, TypeError) as e:
        generation_logger.warning(f"Could not load model configuration for '{model_type}': {e}")
        generation_logger.debug(f"Exception details: {str(e)}")
        import traceback
        generation_logger.debug(f"Traceback: {traceback.format_exc()}")

    # 3. Apply task explicit parameters (highest priority)
    # Safe logging: Use safe_dict_repr to prevent hanging on large nested structures
    generation_logger.debug(safe_log_params(task_params, "Task explicit parameters"))
    generation_logger.debug(f"Before applying task params - resolved_params keys: {list(resolved_params.keys())}")

    for param, value in task_params.items():
        if value is not None:  # Don't override with None values
            old_value = resolved_params.get(param, "NOT_SET")
            resolved_params[param] = value
            # Avoid logging large nested dicts that can hang
            if param in ["orchestrator_payload", "orchestrator_details", "full_orchestrator_payload"]:
                generation_logger.debug(f"Task override {param}: <large dict with {len(value) if isinstance(value, dict) else '?'} keys>")
            else:
                # Limit string representation to prevent hanging on large values
                try:
                    old_str = str(old_value)
                    value_str = str(value)
                    max_chars = 500
                    if len(old_str) > max_chars:
                        old_str = old_str[:max_chars] + "..."
                    if len(value_str) > max_chars:
                        value_str = value_str[:max_chars] + "..."
                    generation_logger.debug(f"Task override {param}: {old_str} → {value_str}")
                except (ValueError, KeyError, TypeError) as e:
                    generation_logger.debug(f"Task override {param}: <repr failed: {e}>")

    # Log keys only to avoid hanging on large nested structures
    generation_logger.debug(f"FINAL resolved_params keys: {list(resolved_params.keys())}")
    generation_logger.debug(f"Parameter resolution for '{model_type}': {len(task_params)} task overrides applied")
    return resolved_params


def query_resolve_parameters(orchestrator, model_type: str, task_params: dict) -> dict:
    """Compatibility alias for resolve_parameters."""
    return resolve_parameters(orchestrator=orchestrator, model_type=model_type, task_params=task_params)
