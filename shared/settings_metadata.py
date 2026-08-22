from __future__ import annotations

from typing import Any

from shared import extra_settings


_SLIDING = {
    "sliding_window_size", "sliding_window_overlap", "sub_parallel_window_size", "sub_parallel_window_overlap",
    "sliding_window_color_correction_strength", "sliding_window_overlap_noise", "sliding_window_discard_last_frames", "sliding_window_trim_first_frames",
}
_VISUAL_POSTPROCESSING = {"temporal_upsampling", "spatial_upsampling", "film_grain_intensity", "film_grain_saturation"}
_VIDEO_MEDIA_OPTIONS = {
    "force_fps", "keep_frames_video_source", "keep_frames_video_guide", "video_guide_outpainting", "video_guide_outpainting_ratio",
    "mask_expand", "frames_positions", "image_refs_relative_size", "remove_background_images_ref", "input_video_strength",
}
_NEUTRAL = {
    "client_id": {None, ""}, "mode": {None, ""}, "output_filename": {None, ""}, "config": {None, ""},
    "override_profile": {None, -1, "-1"}, "override_attention": {None, ""},
}


def _drop(settings: dict[str, Any], keys) -> None:
    for key in keys:
        settings.pop(key, None)


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_metadata_settings(settings: dict[str, Any], model_def: dict[str, Any], *, attention_mode: str = "", prompt_enhancer_visible: bool = False) -> dict[str, Any]:
    model_type = str(settings.get("model_type", "") or "")
    base_model_type = str(model_def.get("architecture", model_type) or model_type)
    audio_only = bool(model_def.get("audio_only", False))
    context = {"model_type": model_type, "guidance_phases": settings.get("guidance_phases", 1)}
    declared = extra_settings.iter_defs(model_def, **context)
    visible = extra_settings.iter_defs(model_def, only_visible=True, **context)
    _drop(settings, declared.keys() - visible.keys())
    capabilities = (
        (audio_only and model_def.get("temperature", True), {"temperature"}),
        (model_def.get("duration_slider") is not None, {"duration_seconds"}),
        (model_def.get("pause_between_sentences", False), {"pause_seconds"}),
        (model_def.get("inference_steps", True), {"num_inference_steps"}),
        (model_def.get("sample_solvers") is not None, {"sample_solver"}),
        (model_def.get("self_refiner", False), {"self_refiner_setting", "self_refiner_plan", "self_refiner_f_uncertainty", "self_refiner_certain_percentage"}),
        (model_def.get("audio_guidance", False), {"speakers_locations"}),
        (any(model_def.get(key, False) for key in ("tea_cache", "mag_cache", "spectrum_cache", "first_block_cache")), {"skip_steps_cache_type", "skip_steps_multiplier", "skip_steps_start_step_perc"}),
        (model_def.get("perturbation", False), {"perturbation_switch", "perturbation_layers", "perturbation_start_perc", "perturbation_end_perc"}),
        (model_def.get("cfg_zero", False), {"cfg_zero_step"}),
        (model_def.get("cfg_star", False), {"cfg_star_switch"}),
        (model_def.get("adaptive_projected_guidance", False), {"apg_switch"}),
        (model_def.get("NAG", False), {"NAG_scale", "NAG_tau", "NAG_alpha"}),
        (model_def.get("riflex", False), {"RIFLEx_setting"}),
    )
    for supported, keys in capabilities:
        if not supported:
            _drop(settings, keys)
    guidance_max_phases = int(model_def.get("guidance_max_phases", 0) or 0)
    guidance_phases = int(settings.get("guidance_phases", 1) or 0)
    if guidance_max_phases < 1:
        _drop(settings, {"guidance_scale", "guidance_phases"})
    if guidance_max_phases < 2 or guidance_phases < 2:
        _drop(settings, {"guidance2_scale", "switch_threshold"})
    if guidance_max_phases < 3 or guidance_phases < 3:
        _drop(settings, {"guidance3_scale", "switch_threshold2", "model_switch_phase"})

    for key, neutral_values in _NEUTRAL.items():
        if settings.get(key) in neutral_values:
            settings.pop(key, None)
    for key in ("image_prompt_type", "video_prompt_type", "audio_prompt_type"):
        if not str(settings.get(key, "") or ""):
            settings.pop(key, None)
    if not str(settings.get("alt_prompt", "") or "") and model_def.get("alt_prompt") is None:
        settings.pop("alt_prompt", None)
    if not str(settings.get("negative_prompt", "") or ""):
        settings.pop("negative_prompt", None)
    if not audio_only and not model_def.get("any_audio_prompt", False) and not model_def.get("returns_audio", False):
        settings.pop("audio_prompt_type", None)
    if not settings.get("custom_settings"):
        settings.pop("custom_settings", None)
    if not str(settings.get("force_fps", "") or ""):
        settings.pop("force_fps", None)
    if not prompt_enhancer_visible:
        settings.pop("prompt_enhancer", None)
    if model_def.get("model_modes") is None:
        settings.pop("model_mode", None)
    settings.pop("keep_intermediate_sliding_windows", None)
    if str(attention_mode or "").strip().lower() != "sol" or not model_def.get("sol_attention", False):
        settings.pop("attention_sparsity", None)

    image_output = not audio_only and int(settings.get("image_mode", 0) or 0) > 0
    video_output = not audio_only and not image_output
    if audio_only:
        _drop(settings, {"resolution", "batch_size", "video_length", "image_mode", "image_prompt_type", "video_prompt_type", "RIFLEx_setting", *_SLIDING, *_VISUAL_POSTPROCESSING, *_VIDEO_MEDIA_OPTIONS})
    elif image_output:
        _drop(settings, {"video_length", "force_fps", "keep_frames_video_source", "keep_frames_video_guide", "video_guide_outpainting", "video_guide_outpainting_ratio", "RIFLEx_setting", "temporal_upsampling", *_SLIDING})
    else:
        settings.pop("batch_size", None)

    image_prompt_type = str(settings.get("image_prompt_type", "") or "")
    video_prompt_type = str(settings.get("video_prompt_type", "") or "")
    guide_choices = model_def.get("guide_custom_choices_image") if image_output and model_def.get("guide_custom_choices_image") is not None else model_def.get("guide_custom_choices")
    if guide_choices is None and model_def.get("guide_preprocessing") is None:
        _drop(settings, {"keep_frames_video_guide", "mask_expand"})
    if "G" not in video_prompt_type:
        settings.pop("denoising_strength", None)
        if not model_def.get("mask_strength_always_enabled", False):
            settings.pop("masking_strength", None)
    input_video_strength = model_def.get("input_video_strength", {})
    if not str(input_video_strength.get("label", "") or "").strip() or not (any(flag in image_prompt_type for flag in "SVLE") or "F" in video_prompt_type):
        settings.pop("input_video_strength", None)
    if not any(flag in image_prompt_type for flag in "SVL"):
        settings.pop("keep_frames_video_source", None)
    elif base_model_type not in {"sky_df_1.3B", "sky_df_14B", "ltxv_13B"} and not model_def.get("vace_class", False):
        settings.pop("keep_frames_video_source", None)
    if "V" not in video_prompt_type:
        settings.pop("keep_frames_video_guide", None)
    if "A" not in video_prompt_type:
        settings.pop("mask_expand", None)
    if "I" not in video_prompt_type:
        _drop(settings, {"image_refs_relative_size", "remove_background_images_ref"})
    elif not model_def.get("any_image_refs_relative_size", False):
        settings.pop("image_refs_relative_size", None)
    if "F" not in video_prompt_type:
        settings.pop("frames_positions", None)
    if model_def.get("video_guide_outpainting") is None:
        _drop(settings, {"video_guide_outpainting", "video_guide_outpainting_ratio"})
    if str(settings.get("video_guide_outpainting", "") or "") in {"", "#"}:
        _drop(settings, {"video_guide_outpainting", "video_guide_outpainting_ratio"})
    if not (model_def.get("vace_class", False) or model_def.get("t2v_class", False)):
        settings.pop("min_frames_if_references", None)
    if not model_def.get("multiple_images_as_text_prompts", False):
        settings.pop("multi_images_gen_type", None)
    if model_def.get("no_negative_prompt", False):
        settings.pop("negative_prompt", None)
    if model_def.get("audio_scale_name") is None or not image_output:
        settings.pop("audio_scale", None)

    if not str(settings.get("temporal_upsampling", "") or ""):
        settings.pop("temporal_upsampling", None)
    if not str(settings.get("spatial_upsampling", "") or ""):
        settings.pop("spatial_upsampling", None)
    if _number(settings.get("film_grain_intensity"), 0) <= 0:
        _drop(settings, {"film_grain_intensity", "film_grain_saturation"})
    if not str(settings.get("postprocess_audio", "") or ""):
        _drop(settings, {"postprocess_audio", "postprocess_audio_prompt", "postprocess_audio_neg_prompt"})
    else:
        from postprocessing import audio_processors

        postprocess_audio_meta = audio_processors.method_metadata(settings["postprocess_audio"])
        if not (postprocess_audio_meta["needs_prompt"] or postprocess_audio_meta["needs_negative_prompt"]) or model_def.get("returns_audio", False) or model_def.get("any_audio_prompt", False):
            _drop(settings, {"postprocess_audio_prompt", "postprocess_audio_neg_prompt"})
    if not str(settings.get("replace_voice_method", "") or ""):
        settings.pop("replace_voice_method", None)

    activated_loras = settings.get("activated_loras", []) or []
    if model_def.get("no_lora", False) or not activated_loras:
        _drop(settings, {"activated_loras", "loras_multipliers"})
    if not str(settings.get("skip_steps_cache_type", "") or ""):
        _drop(settings, {"skip_steps_cache_type", "skip_steps_multiplier", "skip_steps_start_step_perc"})
    if _number(settings.get("perturbation_switch"), 0) <= 0:
        _drop(settings, {"perturbation_switch", "perturbation_layers", "perturbation_start_perc", "perturbation_end_perc"})
    if _number(settings.get("self_refiner_setting"), 0) <= 0:
        _drop(settings, {"self_refiner_setting", "self_refiner_plan", "self_refiner_f_uncertainty", "self_refiner_certain_percentage"})
    if _number(settings.get("NAG_scale"), 1) <= 1:
        _drop(settings, {"NAG_scale", "NAG_tau", "NAG_alpha"})
    if not settings.get("apg_switch"):
        settings.pop("apg_switch", None)
    if not settings.get("cfg_star_switch"):
        settings.pop("cfg_star_switch", None)
    if _number(settings.get("cfg_zero_step"), -1) < 0:
        settings.pop("cfg_zero_step", None)

    if not video_output or not model_def.get("sliding_window", False):
        _drop(settings, _SLIDING)
    elif not model_def.get("sub_parallel_windows", False):
        _drop(settings, {"sub_parallel_window_size", "sub_parallel_window_overlap"})
    _drop(settings, [key for key, value in settings.items() if value is None])
    return settings


__all__ = ["clean_metadata_settings"]
