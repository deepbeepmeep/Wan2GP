"""SVI (Stable Video Infinity) Configuration for End Frame Chaining."""

from typing import Any, Dict, Optional

from source.models.lora.sanitizer import sanitize_lora_values


# =============================================================================
# SVI (Stable Video Infinity) Configuration for End Frame Chaining
# =============================================================================
# SVI LoRAs with phase multipliers (format: "phase1_mult;phase2_mult")
# These LoRAs enable SVI encoding which uses anchor images for consistent long-form generation
SVI_LORAS = {
    # SVI Pro LoRAs (high noise for phase 1, low noise for phase 2)
    "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors": "1;0",
    "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors": "0;1",
    # LightX2V acceleration LoRAs
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": "1.2;0",
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": "0;1",
}

# SVI-specific generation parameters
SVI_DEFAULT_PARAMS = {
    "guidance_phases": 2,           # 2-phase setup for SVI LoRAs
    "num_inference_steps": 6,       # Lightning steps
    "guidance_scale": 1,            # Low guidance for Lightning
    "guidance2_scale": 1,
    "flow_shift": 5,
    "switch_threshold": 883,        # Phase switch threshold
    "model_switch_phase": 1,        # Switch model at phase 1
    "sample_solver": "euler",
}

# SVI stitch overlap - smaller than VACE since end/start frames should match
SVI_STITCH_OVERLAP = 4  # frames



def get_svi_lora_arrays(
    existing_urls: list = None,
    existing_multipliers: list = None,
    svi_strength: float = None,
    svi_strength_1: float = None,
    svi_strength_2: float = None
) -> tuple[list, list]:
    """
    Get SVI LoRAs as (urls, multipliers) arrays for direct use with activated_loras/loras_multipliers.

    This replaces the dict-based get_svi_additional_loras() approach, providing arrays that can be
    directly appended to activated_loras and loras_multipliers without conversion.

    Args:
        existing_urls: Existing LoRA URLs list to merge with
        existing_multipliers: Existing multiplier strings list (phase format like "1.2;0")
        svi_strength: Optional global multiplier to scale ALL SVI LoRA strengths (0.0-1.0+)
        svi_strength_1: Optional multiplier for high-noise LoRAs (phase 1 active: "X;0" pattern)
        svi_strength_2: Optional multiplier for low-noise LoRAs (phase 2 active: "0;X" pattern)

    Returns:
        Tuple of (urls_list, multipliers_list) - ready for activated_loras and loras_multipliers
    """
    result_urls = list(existing_urls) if existing_urls else []
    result_mults = list(existing_multipliers) if existing_multipliers else []

    # Determine if any scaling is needed
    has_scaling = (
        (svi_strength is not None and svi_strength != 1.0) or
        svi_strength_1 is not None or
        svi_strength_2 is not None
    )

    for url, phase_mult in SVI_LORAS.items():
        # Skip if already in list (avoid duplicates)
        if url in result_urls:
            continue

        if has_scaling:
            # Parse phase multiplier string (e.g., "1;0" or "1.2;0" or "0;1")
            parts = str(phase_mult).split(";")

            # Determine if this is a high-noise (phase 1) or low-noise (phase 2) LoRA
            is_high_noise = len(parts) >= 2 and float(parts[0]) > 0 and float(parts[1]) == 0
            is_low_noise = len(parts) >= 2 and float(parts[0]) == 0 and float(parts[1]) > 0

            # Determine which strength multiplier to use
            if is_high_noise and svi_strength_1 is not None:
                strength = svi_strength_1
            elif is_low_noise and svi_strength_2 is not None:
                strength = svi_strength_2
            elif svi_strength is not None:
                strength = svi_strength
            else:
                strength = 1.0

            # Apply scaling
            scaled_parts = []
            for p in parts:
                try:
                    val = float(p)
                    scaled_val = val * strength
                    if scaled_val == int(scaled_val):
                        scaled_parts.append(str(int(scaled_val)))
                    else:
                        scaled_parts.append(f"{scaled_val:.2g}")
                except ValueError:
                    scaled_parts.append(p)
            scaled_mult = ";".join(scaled_parts)
        else:
            scaled_mult = phase_mult

        result_urls.append(url)
        result_mults.append(scaled_mult)

    return result_urls, result_mults


def merge_svi_into_generation_params(
    generation_params: Dict[str, Any],
    svi_strength: Optional[float] = None,
    svi_strength_1: Optional[float] = None,
    svi_strength_2: Optional[float] = None,
) -> None:
    """
    Merge SVI LoRAs into generation_params in-place.

    Parses existing activated_loras/loras_multipliers from generation_params,
    appends SVI LoRA URLs with scaled multipliers, and writes back.

    Args:
        generation_params: Dict to update with merged LoRA arrays.
        svi_strength: Global multiplier for all SVI LoRAs.
        svi_strength_1: Multiplier for high-noise (phase 1) LoRAs.
        svi_strength_2: Multiplier for low-noise (phase 2) LoRAs.
    """
    existing_urls = generation_params.get("activated_loras", [])
    existing_mults_raw = generation_params.get("loras_multipliers", "")
    if isinstance(existing_mults_raw, str):
        existing_mults = existing_mults_raw.split() if existing_mults_raw else []
    else:
        existing_mults = list(existing_mults_raw) if existing_mults_raw else []

    merged_urls, merged_mults = get_svi_lora_arrays(
        existing_urls=existing_urls,
        existing_multipliers=existing_mults,
        svi_strength=svi_strength,
        svi_strength_1=svi_strength_1,
        svi_strength_2=svi_strength_2,
    )
    architecture = generation_params.get("model") or generation_params.get("model_name")
    if architecture:
        sanitized = sanitize_lora_values(
            merged_urls,
            merged_mults,
            architecture=architecture,
        )
        merged_urls = sanitized.loras
        merged_mults = sanitized.multipliers

    generation_params["activated_loras"] = merged_urls
    generation_params["loras_multipliers"] = " ".join(merged_mults)
