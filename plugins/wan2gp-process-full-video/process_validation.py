from __future__ import annotations

import gradio as gr

from . import common
from . import frame_planning as frames


def validate_user_process_definition(process_definition: dict | None, get_model_def) -> list[str]:
    settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
    if not isinstance(settings, dict):
        return ["The selected settings file could not be read."]
    model_type = str(settings.get("model_type") or "").strip()
    if len(model_type) == 0:
        return ["The selected settings file does not define a model."]
    try:
        model_def = frames.require_model_def(model_type, get_model_def)
    except gr.Error as exc:
        return [common.get_error_message(exc) or "The selected model is not available."]
    problems: list[str] = []
    image_mode = common.coerce_int(settings.get("image_mode"), 0)
    if image_mode != 0 or bool(model_def.get("audio_only", False)) or bool(model_def.get("image_outputs", False)) or bool(settings.get("image_outputs", False)):
        problems.append("The selected settings must generate a video, not images or audio only.")
    video_prompt_type = str(settings.get("video_prompt_type") or "")
    if "V" not in video_prompt_type:
        problems.append("Control Video must be enabled.")
    if "I" in video_prompt_type:
        problems.append("Reference Images must be disabled.")
    image_prompt_types_allowed = str(model_def.get("image_prompt_types_allowed", "") or "")
    if "V" not in image_prompt_types_allowed:
        problems.append("The selected model must support using a source video for continuation.")
    audio_prompt_type = str(settings.get("audio_prompt_type") or "")
    audio_features = []
    if "A" in audio_prompt_type:
        audio_features.append("Audio Source")
    if "B" in audio_prompt_type:
        audio_features.append("Audio Source #2")
    if "X" in audio_prompt_type:
        audio_features.append("two-speaker auto separation")
    if len(audio_features) > 0:
        problems.append("Disable these audio features: " + ", ".join(audio_features) + ".")
    return problems


def format_user_process_validation_error(process_definition: dict | None, problems: list[str]) -> str:
    name = str((process_definition or {}).get("name") or "selected settings").strip()
    if len(problems) == 0:
        return ""
    return f'Cannot add "{name}" as a full-video process:\n- ' + "\n- ".join(problems)
