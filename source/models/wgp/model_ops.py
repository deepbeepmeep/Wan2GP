"""
WGP Model Operations

Handles model loading, unloading, and dynamic model definition loading.
Also provides cached Uni3C ControlNet loading.
"""

import os
import gc

from source.core.log.model_runtime import generation_logger, model_logger
from source.runtime.wgp_bridge import (
    clear_wgp_loaded_model_state,
    get_model_name,
    get_model_recursive_prop,
    get_wgp_runtime_module,
    get_wgp_runtime_module_mutable,
    load_uni3c_controlnet,
    set_wgp_loaded_model_state,
    set_wgp_model_def,
    set_wgp_reload_needed,
    upsert_wgp_model_definition,
)


def load_missing_model_definition(orchestrator, model_key: str, json_path: str):
    """
    Dynamically load a missing model definition from JSON file.
    Replicates WGP's model loading logic for individual models.

    Args:
        orchestrator: WanOrchestrator instance (provides wan_root)
        model_key: Model identifier string
        json_path: Path to the model definition JSON file
    """
    import json
    wgp = get_wgp_runtime_module_mutable()

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            json_def = json.load(f)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Error while parsing Model Definition File '{json_path}': {str(e)}") from e

    model_def = json_def["model"]
    model_def["path"] = json_path
    del json_def["model"]
    settings = json_def

    existing_model_def = wgp.models_def.get(model_key, None)
    if existing_model_def is not None:
        existing_settings = existing_model_def.get("settings", None)
        if existing_settings is not None:
            existing_settings.update(settings)
        existing_model_def.update(model_def)
    else:
        set_wgp_model_def(model_key, model_def)
        model_def = upsert_wgp_model_definition(model_key, model_def, initialize=True)
        model_def["settings"] = settings


def load_model_impl(orchestrator, model_key: str) -> bool:
    """Load and validate a model type using WGP's exact generation-time pattern.

    Args:
        orchestrator: WanOrchestrator instance
        model_key: Model identifier (e.g., "t2v", "vace_14B", "flux")

    Returns:
        bool: True if a model switch actually occurred, False if already loaded

    This replicates the exact model loading logic from WGP's generate_video function
    (lines 4249-4258) rather than the UI preloading function.
    """
    from source.models.wgp.generation_helpers import notify_worker_model_switch

    if orchestrator.smoke_mode:
        # In smoke mode, skip heavy WGP model loading
        switched = model_key != orchestrator.current_model
        orchestrator.current_model = model_key
        orchestrator.state["model_type"] = model_key
        model_logger.info(f"[SMOKE] Pretending to load model: {model_key}")
        return switched

    if orchestrator._get_base_model_type(model_key) is None:
        raise ValueError(f"Unknown model: {model_key}")

    wgp = get_wgp_runtime_module_mutable()

    # Debug: Check if model definition is missing and diagnose why
    model_def = wgp.get_model_def(model_key)
    if model_def is None:
        available_models = list(wgp.models_def.keys())
        current_dir = os.getcwd()
        model_logger.warning(f"Model definition for '{model_key}' not found!")
        model_logger.debug(f"Current working directory: {current_dir}")
        model_logger.debug(f"Available models: {available_models}")
        model_logger.debug(f"Looking for model file: {orchestrator.wan_root}/defaults/{model_key}.json")

        # Check if the JSON file exists and try to load it dynamically
        json_path = os.path.join(orchestrator.wan_root, "defaults", f"{model_key}.json")
        if os.path.exists(json_path):
            model_logger.warning(f"Model JSON file exists at {json_path} but wasn't loaded into models_def - attempting dynamic load")
            try:
                load_missing_model_definition(orchestrator, model_key, json_path)
                model_def = wgp.get_model_def(model_key)  # Try again after loading
                if model_def:
                    model_logger.success(f"Successfully loaded missing model definition for {model_key}")
                else:
                    model_logger.error(f"Failed to load model definition for {model_key} even after dynamic loading")
            except (RuntimeError, OSError, ValueError) as e:
                model_logger.error(f"Failed to dynamically load model definition: {e}")
        else:
            model_logger.error(f"Model JSON file missing at {json_path}")

    architecture = model_def.get('architecture') if model_def else 'unknown'
    modules = get_model_recursive_prop(model_key, "modules", return_list=True)
    model_logger.debug(f"Model Info: {model_key} | Architecture: {architecture} | Modules: {modules}")

    # Use WGP's EXACT model loading pattern from generate_video (lines 4249-4258)
    # This is the SINGLE SOURCE OF TRUTH for whether a switch is needed
    current_model_info = f"(current: {wgp.transformer_type})" if wgp.transformer_type else "(no model loaded)"
    switched = False

    if model_key != wgp.transformer_type or wgp.reload_needed:
        model_logger.info(f"🔄 MODEL SWITCH: Using WGP's generate_video pattern - switching from {current_model_info} to {model_key}")

        # Notify edge function BEFORE switch starts (worker will be busy loading)
        notify_worker_model_switch(old_model=wgp.transformer_type, new_model=model_key)

        # Cleanup legacy collision-prone LoRAs BEFORE loading the new model.
        # This matches the original intent upstream of wgp.load_models() and avoids
        # accidental reuse of wrong/old LoRA artifacts across switches.
        try:
            from source.models.lora.lora_utils import cleanup_legacy_lora_collisions
            cleanup_legacy_lora_collisions()
        except (RuntimeError, OSError, ValueError) as e:
            model_logger.warning(f"LoRA cleanup failed during model switch: {e}")

        # Replicate WGP's exact unloading pattern (lines 4250-4254)
        current_offloadobj = getattr(wgp, "offloadobj", None)
        if current_offloadobj is not None:
            current_offloadobj.release()
        clear_wgp_loaded_model_state()
        gc.collect()

        # CRITICAL: Clear CUDA cache after unloading to free reserved VRAM before loading new model
        # Without this, old model's reserved memory persists and new model OOMs during loading
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model_logger.debug("Cleared CUDA cache after model unload")

        # Replicate WGP's exact loading pattern (lines 4255-4258)
        model_logger.debug(f"Loading model {get_model_name(model_key)}...")
        wan_model, offloadobj = wgp.load_models(model_key)
        set_wgp_loaded_model_state(wan_model, offloadobj)
        model_logger.debug("Model loaded")
        set_wgp_reload_needed(False)
        switched = True

        # Note: transformer_type is set automatically by load_models() at line 2929

        model_logger.info(f"✅ MODEL: Loaded using WGP's exact generate_video pattern")
    else:
        model_logger.debug(f"📋 MODEL: Model {model_key} already loaded, no switch needed")

    # Update our tracking to match WGP's state
    orchestrator.current_model = model_key
    orchestrator.state["model_type"] = model_key
    orchestrator.offloadobj = get_wgp_runtime_module().offloadobj  # Keep reference to WGP's offload object

    family = orchestrator._get_model_family(model_key, for_ui=True)
    if switched:
        model_logger.success(f"✅ MODEL Loaded model: {model_key} ({family}) using WGP's exact generate_video pattern")

    return switched


def unload_model_impl(orchestrator):
    """Unload the current model using WGP's native unload function.

    Args:
        orchestrator: WanOrchestrator instance
    """
    if orchestrator.smoke_mode:
        model_logger.info(f"[SMOKE] Unload model: {orchestrator.current_model}")
        orchestrator.current_model = None
        orchestrator.offloadobj = None
        orchestrator.state["model_type"] = None
        orchestrator._cached_uni3c_controlnet = None
        return
    wgp = get_wgp_runtime_module_mutable()

    if orchestrator.current_model and wgp.wan_model is not None:
        model_logger.info(f"🔄 MODEL UNLOAD: Unloading {orchestrator.current_model} using WGP's unload_model_if_needed")

        # Create a state object that WGP functions expect
        temp_state = {"model_type": orchestrator.current_model}

        # Use WGP's native unload function
        try:
            wgp.unload_model_if_needed(temp_state)
            model_logger.info(f"✅ MODEL: WGP unload_model_if_needed completed")

            # Clear our tracking
            orchestrator.current_model = None
            orchestrator.offloadobj = None
            orchestrator.state["model_type"] = None
            orchestrator._cached_uni3c_controlnet = None  # Clear Uni3C cache on model unload

        except (RuntimeError, OSError, ValueError) as e:
            model_logger.error(f"WGP unload_model_if_needed failed: {e}")
            raise
    else:
        model_logger.debug(f"📋 MODEL: No model to unload")


def get_or_load_uni3c_controlnet(orchestrator):
    """Get cached Uni3C controlnet or load it from disk.

    This caches the controlnet across generations to avoid the ~2 minute
    disk load time for the 1.9GB checkpoint on each generation.

    Args:
        orchestrator: WanOrchestrator instance (provides wan_root and cache)

    Returns:
        WanControlNet instance or None if loading fails
    """
    if orchestrator._cached_uni3c_controlnet is not None:
        generation_logger.info("[UNI3C_CACHE] Using cached Uni3C controlnet (skipping disk load)")
        return orchestrator._cached_uni3c_controlnet

    try:
        import torch

        ckpts_dir = os.path.join(orchestrator.wan_root, "ckpts")
        generation_logger.info(f"[UNI3C_CACHE] Loading Uni3C controlnet from disk (first use)...")

        # load_uni3c_controlnet handles:
        # - base_dtype attribute setting
        # - per-layer dtype control (patch embeddings stay float32)
        # - proper model initialization
        controlnet = load_uni3c_controlnet(
            ckpts_dir=ckpts_dir,
            device="cuda",
            dtype=torch.float16,
            use_cache=False  # We manage our own cache
        )
        controlnet.eval()  # Ensure inference mode

        # Cache for future generations
        orchestrator._cached_uni3c_controlnet = controlnet
        generation_logger.info(f"[UNI3C_CACHE] Uni3C controlnet loaded and cached for future generations")

        return controlnet

    except (RuntimeError, OSError, ValueError) as e:
        generation_logger.warning(f"[UNI3C_CACHE] Failed to pre-load Uni3C controlnet: {e}")
        generation_logger.warning("[UNI3C_CACHE] Falling back to on-demand loading in any2video")
        return None
