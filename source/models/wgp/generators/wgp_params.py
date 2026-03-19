"""WGP parameter dictionary builders.

Constructs the keyword-argument dictionaries passed to wgp.generate_video()
for both passthrough mode and normal mode.
"""

from typing import Any, Callable, Dict, Optional

from source.core.log import generation_logger


def make_send_cmd() -> Callable:
    """Create a send_cmd callback for WGP progress reporting."""

    def send_cmd(cmd: str, data=None):
        if cmd == "status":
            generation_logger.debug(f"Status: {data}")
        elif cmd == "progress":
            if isinstance(data, list) and len(data) >= 2:
                progress, status = data[0], data[1]
                generation_logger.debug(f"Progress: {progress}% - {status}")
            else:
                generation_logger.debug(f"Progress: {data}")
        elif cmd == "output":
            generation_logger.essential("Output generated")
        elif cmd == "exit":
            generation_logger.essential("Generation completed")
        elif cmd == "error":
            generation_logger.error(f"Error: {data}")
        elif cmd == "info":
            generation_logger.debug(f"Info: {data}")
        elif cmd == "preview":
            generation_logger.debug("Preview updated")

    return send_cmd


def build_passthrough_params(
    *,
    state: dict,
    current_model: str,
    image_mode: int,
    resolved_params: dict,
    video_guide: Optional[Any],
    video_mask: Optional[Any],
    video_prompt_type: Optional[str],
    control_net_weight: Optional[float],
    control_net_weight2: Optional[float],
) -> Dict[str, Any]:
    """Build the WGP parameter dict for JSON passthrough mode.

    In passthrough mode, nearly all parameters come directly from the
    resolved JSON with sensible defaults for anything missing.
    """
    task = {"id": 1, "params": {}, "repeats": 1}
    send_cmd = make_send_cmd()
    custom_settings = resolved_params.get("custom_settings")
    if not isinstance(custom_settings, dict):
        custom_settings = {}
    if "pace" in resolved_params and "pace" not in custom_settings:
        custom_settings["pace"] = resolved_params["pace"]
    if "exaggeration" in resolved_params and "exaggeration" not in custom_settings:
        custom_settings["exaggeration"] = resolved_params["exaggeration"]

    wgp_params: Dict[str, Any] = {
        # Core parameters (fixed, not overridable)
        'task': task,
        'send_cmd': send_cmd,
        'state': state,
        'model_type': current_model,
        'image_mode': image_mode,

        # Required parameters with defaults
        'prompt': resolved_params.get('prompt', ''),
        'alt_prompt': resolved_params.get('alt_prompt', ''),
        'negative_prompt': resolved_params.get('negative_prompt', ''),
        'resolution': resolved_params.get('resolution', '1280x720'),
        'video_length': resolved_params.get('video_length', 81),
        'duration_seconds': resolved_params.get('duration_seconds', 0),
        'pause_seconds': resolved_params.get('pause_seconds', 0),
        'batch_size': resolved_params.get('batch_size', 1),
        'seed': resolved_params.get('seed', 42),
        'force_fps': 'auto',

        # VACE control parameters
        'video_guide': video_guide,
        'video_mask': video_mask,
        'video_guide2': None,
        'video_mask2': None,
        'video_prompt_type': video_prompt_type or 'VM',
        'control_net_weight': control_net_weight or 1.0,
        'control_net_weight2': control_net_weight2 or 1.0,
        'denoising_strength': 1.0,

        # LoRA parameters
        'activated_loras': [],
        'loras_multipliers': '',

        # Audio parameters
        'audio_guidance_scale': 1.0,
        'audio_scale': resolved_params.get('audio_scale', 1.0),
        'embedded_guidance_scale': resolved_params.get('embedded_guidance_scale', 0.0),
        'repeat_generation': 1,
        'multi_prompts_gen_type': 0,
        'multi_images_gen_type': 0,
        'skip_steps_cache_type': '',
        'skip_steps_multiplier': 1.0,
        'skip_steps_start_step_perc': 0.0,

        # Image parameters
        'image_prompt_type': 'disabled',
        'image_start': None,
        'image_end': None,
        'model_mode': 0,
        'video_source': None,
        'keep_frames_video_source': '',
        'input_video_strength': resolved_params.get('input_video_strength', 1.0),
        'image_refs': None,
        'image_refs_strengths': resolved_params.get('image_refs_strengths'),
        'frames_positions': '',
        'image_guide': None,
        'keep_frames_video_guide': '',
        'video_guide_outpainting': '0 0 0 0',
        'image_mask': None,
        'mask_expand': 0,

        # Audio parameters
        'audio_guide': None,
        'audio_guide2': None,
        'audio_source': None,
        'audio_prompt_type': '',
        'speakers_locations': '',

        # Sliding window parameters
        'sliding_window_size': 129,
        'sliding_window_overlap': 0,
        'sliding_window_color_correction_strength': 0.0,
        'sliding_window_overlap_noise': 0.1,
        'sliding_window_discard_last_frames': 0,
        'image_refs_relative_size': 50,

        # Post-processing parameters
        'remove_background_images_ref': 0,
        'temporal_upsampling': '',
        'spatial_upsampling': '',
        'film_grain_intensity': 0.0,
        'film_grain_saturation': 0.0,
        'MMAudio_setting': 0,
        'MMAudio_prompt': '',
        'MMAudio_neg_prompt': '',

        # Advanced parameters
        'RIFLEx_setting': 0,
        'NAG_scale': 0.0,
        'NAG_tau': 1.0,
        'NAG_alpha': 0.0,
        'perturbation_switch': resolved_params.get('perturbation_switch', 0),
        'perturbation_layers': resolved_params.get('perturbation_layers', [9]),
        'perturbation_start_perc': resolved_params.get('perturbation_start_perc', 10),
        'perturbation_end_perc': resolved_params.get('perturbation_end_perc', 90),
        'apg_switch': 0,
        'cfg_star_switch': 0,
        'cfg_zero_step': 0,
        'prompt_enhancer': 0,
        'min_frames_if_references': 9,
        'override_profile': -1,
        'override_attention': resolved_params.get('override_attention', ''),
        'custom_settings': custom_settings or None,
        'top_p': resolved_params.get('top_p', 0.9),
        'top_k': resolved_params.get('top_k', 50),
        'self_refiner_setting': resolved_params.get('self_refiner_setting', 0),
        'self_refiner_plan': resolved_params.get('self_refiner_plan', ''),
        'self_refiner_f_uncertainty': resolved_params.get('self_refiner_f_uncertainty', 0.0),
        'self_refiner_certain_percentage': resolved_params.get('self_refiner_certain_percentage', 0.999),

        # New v9.1 required parameters
        'alt_guidance_scale': 0.0,
        'alt_scale': resolved_params.get('alt_scale', 0.0),
        'masking_strength': 1.0,
        'control_net_weight_alt': 1.0,
        'motion_amplitude': 1.0,
        'custom_guide': None,
        'temperature': 1.0,
        'output_filename': '',

        # Mode and filename
        'mode': 'generate',
        'model_filename': '',
    }

    # Override with ALL parameters from JSON (this preserves exact JSON values)
    for param_key, param_value in resolved_params.items():
        if param_key not in ('task', 'send_cmd', 'state', 'model_type'):
            wgp_params[param_key] = param_value
            if param_key == 'guidance2_scale':
                generation_logger.info(f"[PASSTHROUGH_DEBUG] Setting {param_key} = {param_value} from JSON")

    # Debug: Check final guidance2_scale value before WGP call
    generation_logger.info(f"[PASSTHROUGH_DEBUG] Final wgp_params guidance2_scale = {wgp_params.get('guidance2_scale', 'NOT_SET')}")

    return wgp_params


def build_normal_params(
    *,
    state: dict,
    current_model: str,
    image_mode: int,
    resolved_params: dict,
    prompt: str,
    actual_video_length: int,
    actual_batch_size: int,
    actual_guidance: float,
    final_embedded_guidance: float,
    is_flux: bool,
    video_guide: Optional[Any],
    video_mask: Optional[Any],
    video_prompt_type: Optional[str],
    control_net_weight: Optional[float],
    control_net_weight2: Optional[float],
    activated_loras: list,
    loras_multipliers_str: str,
) -> Dict[str, Any]:
    """Build the WGP parameter dict for normal (non-passthrough) mode."""
    task = {"id": 1, "params": {}, "repeats": 1}
    send_cmd = make_send_cmd()

    # Supply defaults for required WGP args that may be unused depending on phases/model
    guidance3_scale_value = resolved_params.get(
        "guidance3_scale",
        resolved_params.get("guidance2_scale", actual_guidance),
    )
    switch_threshold2_value = resolved_params.get("switch_threshold2", 0)
    guidance_phases_value = resolved_params.get("guidance_phases", 1)
    model_switch_phase_value = resolved_params.get("model_switch_phase", 1)
    image_refs_relative_size_value = resolved_params.get("image_refs_relative_size", 50)
    override_profile_value = resolved_params.get("override_profile", -1)
    custom_settings_value = resolved_params.get("custom_settings")
    if not isinstance(custom_settings_value, dict):
        custom_settings_value = {}
    if "pace" in resolved_params and "pace" not in custom_settings_value:
        custom_settings_value["pace"] = resolved_params["pace"]
    if "exaggeration" in resolved_params and "exaggeration" not in custom_settings_value:
        custom_settings_value["exaggeration"] = resolved_params["exaggeration"]

    wgp_params: Dict[str, Any] = {
        # Core parameters (fixed, not overridable)
        'task': task,
        'send_cmd': send_cmd,
        'state': state,
        'model_type': current_model,
        'prompt': resolved_params.get("prompt", prompt),
        'alt_prompt': resolved_params.get("alt_prompt", ""),
        'negative_prompt': resolved_params.get("negative_prompt", ""),
        'resolution': resolved_params.get("resolution", "1280x720"),
        'video_length': actual_video_length,
        'duration_seconds': resolved_params.get("duration_seconds", 0),
        'pause_seconds': resolved_params.get("pause_seconds", 0),
        'batch_size': actual_batch_size,
        'seed': resolved_params.get("seed", 42),
        'force_fps': "auto",
        'image_mode': image_mode,

        # VACE control parameters (only pass supported fields upstream)
        'video_guide': video_guide,
        'video_mask': video_mask,
        'video_guide2': None,
        'video_mask2': None,
        'video_prompt_type': video_prompt_type,
        'control_net_weight': control_net_weight,
        'control_net_weight2': control_net_weight2,
        'denoising_strength': 1.0,

        # LoRA parameters in normal mode
        'activated_loras': activated_loras,
        'loras_multipliers': loras_multipliers_str,

        # Overridable parameters from resolved configuration
        'num_inference_steps': resolved_params.get("num_inference_steps", 25),
        'guidance_scale': actual_guidance,
        'guidance2_scale': resolved_params.get("guidance2_scale", actual_guidance),
        'guidance3_scale': guidance3_scale_value,
        'switch_threshold': resolved_params.get("switch_threshold", 500),
        'switch_threshold2': switch_threshold2_value,
        'guidance_phases': guidance_phases_value,
        'model_switch_phase': model_switch_phase_value,
        'embedded_guidance_scale': final_embedded_guidance if is_flux else 0.0,
        'flow_shift': resolved_params.get("flow_shift", 7.0),
        'sample_solver': resolved_params.get("sample_solver", "euler"),

        # New v9.1 required parameters
        'alt_guidance_scale': resolved_params.get("alt_guidance_scale", 0.0),
        'alt_scale': resolved_params.get("alt_scale", 0.0),
        'audio_scale': resolved_params.get("audio_scale", 1.0),
        'masking_strength': resolved_params.get("masking_strength", 1.0),
        'control_net_weight_alt': resolved_params.get("control_net_weight_alt", 1.0),
        'motion_amplitude': resolved_params.get("motion_amplitude", 1.0),
        'custom_guide': resolved_params.get("custom_guide", None),
        'temperature': resolved_params.get("temperature", 1.0),
        'output_filename': resolved_params.get("output_filename", ""),
    }

    # Standard defaults for other parameters
    wgp_params.update({
        'audio_guidance_scale': 1.0,
        'repeat_generation': 1,
        'multi_prompts_gen_type': 0,
        'multi_images_gen_type': 0,
        'skip_steps_cache_type': "",
        'skip_steps_multiplier': 1.0,
        'skip_steps_start_step_perc': 0.0,

        # Image parameters
        'image_prompt_type': resolved_params.get("image_prompt_type", "disabled"),
        'image_start': resolved_params.get("image_start"),
        'image_end': resolved_params.get("image_end"),
        'image_refs': resolved_params.get("image_refs"),
        'image_refs_strengths': resolved_params.get("image_refs_strengths"),
        'frames_positions': resolved_params.get("frames_positions", ""),
        'image_guide': resolved_params.get("image_guide"),
        'image_mask': resolved_params.get("image_mask"),

        # Video parameters
        'model_mode': 0,
        'video_source': None,
        'keep_frames_video_source': "",
        'input_video_strength': resolved_params.get("input_video_strength", 1.0),
        'keep_frames_video_guide': "",
        'video_guide_outpainting': "0 0 0 0",
        'mask_expand': 0,

        # Audio parameters
        'audio_guide': None,
        'audio_guide2': None,
        'audio_source': None,
        'audio_prompt_type': "",
        'speakers_locations': "",

        # Sliding window
        'sliding_window_size': 129,
        'sliding_window_overlap': 0,
        'sliding_window_color_correction_strength': 0.0,
        'sliding_window_overlap_noise': 0.1,
        'sliding_window_discard_last_frames': 0,
        'latent_noise_mask_strength': 0.0,
        'vid2vid_init_video': None,
        'vid2vid_init_strength': 0.7,
        'image_refs_relative_size': image_refs_relative_size_value,

        # Post-processing
        'remove_background_images_ref': 0,
        'temporal_upsampling': "",
        'spatial_upsampling': "",
        'film_grain_intensity': 0.0,
        'film_grain_saturation': 0.0,
        'MMAudio_setting': 0,
        'MMAudio_prompt': "",
        'MMAudio_neg_prompt': "",

        # Advanced parameters
        'RIFLEx_setting': 0,
        'NAG_scale': 0.0,
        'NAG_tau': 1.0,
        'NAG_alpha': 0.0,
        'perturbation_switch': resolved_params.get("perturbation_switch", 0),
        'perturbation_layers': resolved_params.get("perturbation_layers", [9]),
        'perturbation_start_perc': resolved_params.get("perturbation_start_perc", 10),
        'perturbation_end_perc': resolved_params.get("perturbation_end_perc", 90),
        'apg_switch': 0,
        'cfg_star_switch': 0,
        'cfg_zero_step': 0,
        'prompt_enhancer': 0,
        'min_frames_if_references': 9,
        'override_profile': override_profile_value,
        'override_attention': resolved_params.get("override_attention", ""),
        'custom_settings': custom_settings_value or None,
        'top_p': resolved_params.get("top_p", 0.9),
        'top_k': resolved_params.get("top_k", 50),
        'self_refiner_setting': resolved_params.get("self_refiner_setting", 0),
        'self_refiner_plan': resolved_params.get("self_refiner_plan", ""),
        'self_refiner_f_uncertainty': resolved_params.get("self_refiner_f_uncertainty", 0.0),
        'self_refiner_certain_percentage': resolved_params.get("self_refiner_certain_percentage", 0.999),

        # Mode and filename
        'mode': "generate",
        'model_filename': "",
    })

    return wgp_params


def _build_allowed_override_keys() -> set[str]:
    passthrough_keys = set(
        build_passthrough_params(
            state={},
            current_model="wan",
            image_mode=0,
            resolved_params={},
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        ).keys()
    )
    normal_keys = set(
        build_normal_params(
            state={},
            current_model="wan",
            image_mode=0,
            resolved_params={},
            prompt="",
            actual_video_length=81,
            actual_batch_size=1,
            actual_guidance=5.0,
            final_embedded_guidance=0.0,
            is_flux=False,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
            activated_loras=[],
            loras_multipliers_str="",
        ).keys()
    )
    return passthrough_keys | normal_keys | {"phase_config"}


_IMMUTABLE_KWARG_OVERRIDE_KEYS = {"task", "send_cmd", "state", "model_type"}
_ALLOWED_KWARG_OVERRIDE_KEYS = _build_allowed_override_keys()


def apply_kwargs_overrides(wgp_params: Dict[str, Any], kwargs: dict) -> None:
    """Override any WGP parameter with values from kwargs (in-place)."""
    for key, value in kwargs.items():
        if key in _IMMUTABLE_KWARG_OVERRIDE_KEYS:
            generation_logger.debug(f"Ignoring immutable kwargs override: {key}")
            continue
        if key not in _ALLOWED_KWARG_OVERRIDE_KEYS:
            generation_logger.debug(f"Ignoring unknown/internal kwargs override: {key}")
            continue
        if key in wgp_params:
            generation_logger.debug(f"Overriding parameter from kwargs: {key}={wgp_params[key]} -> {value}")
        else:
            generation_logger.debug(f"Adding allowed parameter from kwargs: {key}={value}")
        wgp_params[key] = value
