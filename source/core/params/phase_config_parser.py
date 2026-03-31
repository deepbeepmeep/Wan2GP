"""
Phase config parser: converts phase_config dicts into timestep thresholds and LoRA schedules.

Moved here from source.task_handlers.tasks.task_conversion to fix a layering violation
(core/params/phase.py was importing from the higher-layer task_handlers package).
"""

import copy
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from source.core.log import headless_logger

# Diffusion timestep scaling constant: maps normalized [0,1] sigmas to diffusion timestep range
DIFFUSION_TIMESTEP_SCALE = 1000


def _timestep_transform(t: float, shift: float = 5.0, num_timesteps: int = DIFFUSION_TIMESTEP_SCALE) -> float:
    t = t / num_timesteps
    new_t = shift * t / (1 + (shift - 1) * t)
    return new_t * num_timesteps


def _validate_phase_config(phase_config: dict, num_inference_steps: int, task_id: str) -> Tuple[int, list, float, list]:
    """Validate and auto-correct phase_config fields. Returns (num_phases, steps_per_phase, flow_shift, phases_config)."""
    num_phases = phase_config.get("num_phases", 3)
    steps_per_phase = phase_config.get("steps_per_phase", [2, 2, 2])
    flow_shift = phase_config.get("flow_shift", 5.0)
    phases_config = phase_config.get("phases", [])

    if len(steps_per_phase) != num_phases or len(phases_config) != num_phases:
        inferred_phases = len(steps_per_phase)
        if len(phases_config) != inferred_phases:
            inferred_phases = len(phases_config)
            headless_logger.warning(
                f"phase_config mismatch: num_phases={num_phases}, steps_per_phase has {len(steps_per_phase)} values, "
                f"phases array has {len(phases_config)} entries. Using phases array length: {inferred_phases}",
                task_id=task_id
            )
        else:
            headless_logger.warning(
                f"phase_config mismatch: num_phases={num_phases} but steps_per_phase has {len(steps_per_phase)} values. "
                f"Auto-correcting num_phases to {inferred_phases}",
                task_id=task_id
            )
        num_phases = inferred_phases

    if num_phases not in [2, 3]:
        raise ValueError(f"Task {task_id}: num_phases must be 2 or 3, got {num_phases}")

    total_steps = sum(steps_per_phase)
    if total_steps != num_inference_steps:
        raise ValueError(f"Task {task_id}: steps_per_phase {steps_per_phase} sum to {total_steps}, but num_inference_steps is {num_inference_steps}")

    phases_config = phase_config.get("phases", [])
    if len(phases_config) != num_phases:
        raise ValueError(f"Task {task_id}: Expected {num_phases} phase configs, got {len(phases_config)}")

    return num_phases, steps_per_phase, flow_shift, phases_config


def _generate_timesteps(sample_solver: str, flow_shift: float, num_inference_steps: int, task_id: str) -> List[float]:
    """Generate diffusion timesteps based on solver type and flow shift."""
    if sample_solver == "causvid":
        headless_logger.warning("phase_config with causvid solver is not recommended - causvid uses hardcoded timesteps", task_id=task_id)

    if sample_solver in ("unipc", ""):
        sigma_max = 1.0
        sigma_min = 0.001
        sigmas = list(np.linspace(sigma_max, sigma_min, num_inference_steps + 1, dtype=np.float32))[:-1]
        sigmas = [flow_shift * s / (1 + (flow_shift - 1) * s) for s in sigmas]
        return [s * DIFFUSION_TIMESTEP_SCALE for s in sigmas]

    if sample_solver in ("dpm++", "dpm++_sde"):
        sigmas = list(np.linspace(1, 0, num_inference_steps + 1, dtype=np.float32))[:num_inference_steps]
        sigmas = [flow_shift * s / (1 + (flow_shift - 1) * s) for s in sigmas]
        return [s * DIFFUSION_TIMESTEP_SCALE for s in sigmas]

    if sample_solver == "euler":
        timesteps = list(np.linspace(DIFFUSION_TIMESTEP_SCALE, 1, num_inference_steps, dtype=np.float32))
        timesteps.append(0.)
        return [_timestep_transform(t, shift=flow_shift, num_timesteps=DIFFUSION_TIMESTEP_SCALE) for t in timesteps][:-1]

    return list(np.linspace(DIFFUSION_TIMESTEP_SCALE, 1, num_inference_steps, dtype=np.float32))


def _calculate_switch_thresholds(
    timesteps: List[float], steps_per_phase: list, num_phases: int, num_inference_steps: int
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate timestep thresholds where guidance scale switches between phases."""
    switch_threshold = None
    switch_threshold2 = None

    switch_step_1 = steps_per_phase[0]
    if switch_step_1 < num_inference_steps:
        last_phase1_t = timesteps[switch_step_1 - 1]
        first_phase2_t = timesteps[switch_step_1]
        switch_threshold = float((last_phase1_t + first_phase2_t) / 2)

    if num_phases >= 3:
        switch_step_2 = steps_per_phase[0] + steps_per_phase[1]
        if switch_step_2 < num_inference_steps:
            last_phase2_t = timesteps[switch_step_2 - 1]
            first_phase3_t = timesteps[switch_step_2]
            switch_threshold2 = float((last_phase2_t + first_phase3_t) / 2)

    return switch_threshold, switch_threshold2


def _process_lora_schedules(
    phases_config: list, steps_per_phase: list, num_phases: int, task_id: str
) -> Tuple[List[str], List[str], dict]:
    """Collect unique LoRA URLs and build per-phase multiplier strings. Returns (lora_urls, multiplier_strings, additional_loras)."""
    all_lora_urls = []
    seen_urls: set = set()
    for phase_cfg in phases_config:
        for lora in phase_cfg.get("loras", []):
            url = lora["url"]
            if not url or not url.strip():
                continue
            if url not in seen_urls:
                all_lora_urls.append(url)
                seen_urls.add(url)

    lora_multipliers = []
    additional_loras = {}

    for lora_url in all_lora_urls:
        phase_mults = []
        for phase_idx, phase_cfg in enumerate(phases_config):
            lora_in_phase = None
            for lora in phase_cfg.get("loras", []):
                if lora["url"] == lora_url:
                    lora_in_phase = lora
                    break

            if lora_in_phase:
                multiplier_str = str(lora_in_phase["multiplier"])
                if "," in multiplier_str:
                    values = multiplier_str.split(",")
                    expected_count = steps_per_phase[phase_idx]
                    if len(values) != expected_count:
                        raise ValueError(f"Task {task_id}: Phase {phase_idx+1} LoRA multiplier has {len(values)} values, but phase has {expected_count} steps")
                phase_mults.append(str(lora_in_phase["multiplier"]) if lora_in_phase else "0")
            else:
                phase_mults.append("0")

        lora_multipliers.append(";".join(phase_mults))

        try:
            first_val = float(phase_mults[0].split(",")[0])
            additional_loras[lora_url] = first_val
        except (ValueError, IndexError):
            additional_loras[lora_url] = 1.0

    return all_lora_urls, lora_multipliers, additional_loras


def _prepare_patch_config(
    phase_config: dict, phases_config: list, num_phases: int,
    all_lora_urls: List[str], model_name: str, task_id: str
) -> Optional[dict]:
    """Load WGP defaults and build a patched config dict, or None if unavailable."""
    wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"
    original_config_path = wan_dir / "defaults" / f"{model_name}.json"
    if not original_config_path.exists():
        return None

    try:
        with open(original_config_path, 'r') as f:
            original_config = json.load(f)

        temp_config = copy.deepcopy(original_config)
        temp_config["guidance_phases"] = num_phases

        guidance_scales = [p.get("guidance_scale", 1.0) for p in phases_config]
        if len(guidance_scales) >= 1: temp_config["guidance_scale"] = guidance_scales[0]
        if len(guidance_scales) >= 2: temp_config["guidance2_scale"] = guidance_scales[1]
        if len(guidance_scales) >= 3: temp_config["guidance3_scale"] = guidance_scales[2]

        temp_config["flow_shift"] = phase_config.get("flow_shift", temp_config.get("flow_shift", 5.0))
        temp_config["sample_solver"] = phase_config.get("sample_solver", temp_config.get("sample_solver", "euler"))

        if "model" in temp_config:
            temp_config["model"]["loras"] = []
            temp_config["model"]["loras_multipliers"] = []
            svi_loras_present = any(
                "SVI" in lora_url or "svi" in lora_url.lower()
                for lora_url in all_lora_urls
            )
            if svi_loras_present or phase_config.get("svi2pro", False):
                temp_config["model"]["svi2pro"] = True
                headless_logger.debug("[PATCH_CONFIG] Added svi2pro=True to model definition (SVI LoRAs detected)", task_id=task_id)

        return temp_config
    except (OSError, ValueError, KeyError) as e:
        headless_logger.warning(f"Could not prepare phase_config patch: {e}", task_id=task_id)
        return None


def parse_phase_config(phase_config: dict, num_inference_steps: int, task_id: str = "unknown", model_name: str = None, debug_mode: bool = False) -> dict:
    """Parse phase_config override block and return computed parameters."""
    num_phases, steps_per_phase, flow_shift, phases_config = _validate_phase_config(
        phase_config, num_inference_steps, task_id
    )

    sample_solver = phase_config.get("sample_solver", "euler")
    timesteps = _generate_timesteps(sample_solver, flow_shift, num_inference_steps, task_id)
    headless_logger.debug(f"Generated timesteps for phase_config: {[f'{t:.1f}' for t in timesteps]}", task_id=task_id)

    switch_threshold, switch_threshold2 = _calculate_switch_thresholds(
        timesteps, steps_per_phase, num_phases, num_inference_steps
    )

    result = {
        "guidance_phases": num_phases,
        "switch_threshold": switch_threshold,
        "switch_threshold2": switch_threshold2,
        "flow_shift": flow_shift,
        "sample_solver": sample_solver,
        "model_switch_phase": phase_config.get("model_switch_phase", 2),
    }

    if num_phases >= 1:
        result["guidance_scale"] = phases_config[0].get("guidance_scale", 3.0)
    if num_phases >= 2:
        result["guidance2_scale"] = phases_config[1].get("guidance_scale", 1.0)
    if num_phases >= 3:
        result["guidance3_scale"] = phases_config[2].get("guidance_scale", 1.0)

    all_lora_urls, lora_multipliers, additional_loras = _process_lora_schedules(
        phases_config, steps_per_phase, num_phases, task_id
    )

    # Filter out LoRAs incompatible with the target model architecture.
    # Wan 2.2 LoRAs (wan2.2_*) are incompatible with LTX-2 models and vice versa.
    if model_name and all_lora_urls:
        is_ltx2 = "ltx2" in model_name.lower()
        filtered_indices = []
        for i, url in enumerate(all_lora_urls):
            url_lower = url.lower()
            wan_lora = "wan2.2" in url_lower or "wan_2.2" in url_lower
            if is_ltx2 and wan_lora:
                headless_logger.warning(
                    f"Skipping Wan 2.2 LoRA incompatible with LTX-2 model '{model_name}': {url}",
                    task_id=task_id,
                )
                continue
            filtered_indices.append(i)
        if len(filtered_indices) < len(all_lora_urls):
            all_lora_urls = [all_lora_urls[i] for i in filtered_indices]
            lora_multipliers = [lora_multipliers[i] for i in filtered_indices]

    if model_name:
        patch = _prepare_patch_config(phase_config, phases_config, num_phases, all_lora_urls, model_name, task_id)
        if patch:
            result["_patch_config"] = patch
            result["_patch_model_name"] = model_name

    # CRITICAL: lora_names must contain the URLs so LoRAConfig can:
    # 1. Detect URLs and mark them as PENDING for download
    # 2. Associate each URL with its corresponding phase-format multiplier
    # 3. Download via _download_lora_from_url() in the queue
    # Without this, activated_loras ends up empty and the multipliers aren't associated
    result["lora_names"] = all_lora_urls
    result["lora_multipliers"] = lora_multipliers
    result["additional_loras"] = additional_loras

    return result
