"""
WGP LoRA Setup

Handles LoRA discovery and initialization for a model type.
Matches WGP's exact setup_loras call pattern from generate_video_tab.
"""

import os

from source.core.log import model_logger
from source.runtime.wgp_bridge import get_lora_dir, setup_loras


def setup_loras_for_model(orchestrator, model_type: str):
    """Initialize LoRA discovery for a model type.

    This matches WGP's exact setup_loras call pattern from generate_video_tab.
    Scans the LoRA directory and populates state with available LoRAs.
    The actual loading/activation happens during generation.

    Args:
        orchestrator: WanOrchestrator instance (provides state dict)
        model_type: Model identifier (e.g., "t2v", "vace_14B")
    """
    try:
        # Use exact same call pattern as WGP's generate_video_tab (line 6941)
        # setup_loras(model_type, transformer, lora_dir, lora_preselected_preset, split_linear_modules_map)
        preset_to_load = ""  # No preset in headless mode (equivalent to lora_preselected_preset)

        loras, loras_names, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset = setup_loras(
            model_type,           # Same as WGP
            None,                 # transformer=None for discovery phase (same as WGP)
            get_lora_dir(model_type),  # lora_dir (same as WGP)
            preset_to_load,       # lora_preselected_preset="" (same as WGP)
            None                  # split_linear_modules_map=None (same as WGP)
        )

        # Update state with discovered LoRAs - exact same pattern as WGP (lines 6943-6945)
        orchestrator.state["loras"] = loras
        orchestrator.state["loras_presets"] = loras_presets
        orchestrator.state["loras_names"] = loras_names

        if loras:
            model_logger.debug(f"Discovered {len(loras)} LoRAs for {model_type}: {[os.path.basename(l) for l in loras[:3]]}{'...' if len(loras) > 3 else ''}")
        else:
            model_logger.debug(f"No LoRAs found for {model_type}")

    except (RuntimeError, OSError, ValueError) as e:
        model_logger.warning(f"LoRA discovery failed for {model_type}: {e}")
        # Keep empty defaults to prevent crashes
        orchestrator.state["loras"] = []
        orchestrator.state["loras_names"] = []
        orchestrator.state["loras_presets"] = {}
