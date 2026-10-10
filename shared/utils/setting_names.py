"""Setting-name validation shared by every generation entry point.

Accepted names are the shared catalog in models/_settings.json, the non-setting keys that travel with
settings (identity, versioning, UI transport and the metadata WanGP writes into saved media), and retired
names that older saved settings, queues and defaults may still carry; retired names are ignored rather than rejected.
Other media metadata (written by tools and plugins) is not a setting: split_settings separates it before reuse.
"""

import difflib
import json
from functools import cache
from pathlib import Path

SETTINGS_CATALOG = Path(__file__).resolve().parents[2] / "models" / "_settings.json"

NON_SETTING_KEYS = frozenset({
    "model_type", "base_model_type", "model_filename", "type", "settings_version", "mode", "state", "spatial_upsampler_parameters",
    "priority", "frame_scheduler", "send_cmd", "task", "target", "image_mask_guide", "loras_choices", "plugin_data",
    "profile_priority", "help", "creation_date", "creation_timestamp", "generation_time", "is_image", "window_no", "custom_guide_used", "_settings_bundle_task_count",
    "image_quality", "video_quality", "hdr", "hdr_video_crf", "modules", "transformer_loras_filenames", "transformer_loras_multipliers",
    "gallery_media_ids", "deepy_media_id", "deepy_media_fingerprint", "deepy_session_id",
})

DEPRECATED_SETTING_KEYS = frozenset({
    "prompts", "lset_name", "slg_switch", "slg_layers", "slg_start_perc", "slg_end_perc", "tea_cache_setting", "tea_cache_start_step_perc", "spatial_upsampler_face_count",
    "scheduler_type", "shift", "adaptive_switch", "MMAudio_setting", "MMAudio_prompt", "MMAudio_neg_prompt", "seedvc_voice_sample", "seedvc_voice_sample2",
    "fps", "sampling_steps", "sampler_solver", "sample_solvers", "enhanced_prompt", "enhanced_alt_prompt", "audio_postprocess", "audio_cfg_scale", "pace", "exaggeration",
})

# Names commonly guessed for an existing setting, and what to use instead.
SETTING_HINTS = {
    "steps": "num_inference_steps", "num_steps": "num_inference_steps", "inference_steps": "num_inference_steps",
    "width": 'resolution ("WIDTHxHEIGHT")', "height": 'resolution ("WIDTHxHEIGHT")', "size": 'resolution ("WIDTHxHEIGHT")', "aspect_ratio": 'resolution ("WIDTHxHEIGHT")',
    "cfg": "guidance_scale", "cfg_scale": "guidance_scale", "guidance": "guidance_scale",
    "sampler": "sample_solver", "scheduler": "sample_solver", "solver": "sample_solver",
    "frames": "video_length", "num_frames": "video_length", "duration": "duration_seconds", "negative": "negative_prompt",
    "lora": "activated_loras", "loras": "activated_loras",
    **dict.fromkeys(("accelerator", "accelerator_profile", "profile"), "an accelerator profile's settings"),
}


@cache
def catalog_setting_names() -> frozenset:
    return frozenset(json.loads(SETTINGS_CATALOG.read_text(encoding="utf-8")))


def accepted_setting_names() -> frozenset:
    return catalog_setting_names() | NON_SETTING_KEYS | DEPRECATED_SETTING_KEYS


def unknown_setting_names(settings) -> list:
    accepted = accepted_setting_names()
    return [name for name in settings if name not in accepted]


def split_settings(settings) -> tuple:
    """Separate reusable settings from other metadata, such as saved-media details written by tools."""
    accepted = accepted_setting_names()
    return {name: value for name, value in settings.items() if name in accepted}, {name: value for name, value in settings.items() if name not in accepted}


def unknown_settings_error(settings) -> str:
    unknown = unknown_setting_names(settings)
    if not unknown:
        return ""
    described = []
    for name in unknown:
        hint = SETTING_HINTS.get(name.lower()) or next(iter(difflib.get_close_matches(name, catalog_setting_names(), n=1, cutoff=0.75)), "")
        described.append(f"{name} (use {hint})" if hint else name)
    return f"Unknown setting{'s' if len(unknown) > 1 else ''}: {', '.join(described)}. Model-specific options belong in custom_settings."
