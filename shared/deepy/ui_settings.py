from __future__ import annotations

from typing import Any

from shared.deepy.config import (
    DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT,
    DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY,
    DEEPY_DEFAULT_EDIT_IMAGE,
    DEEPY_DEFAULT_GEN_IMAGE,
    DEEPY_DEFAULT_GEN_VIDEO,
    DEEPY_TOOL_EDIT_IMAGE_KEY,
    DEEPY_TOOL_GEN_IMAGE_KEY,
    DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY,
    DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY,
    DEEPY_TOOL_GEN_VIDEO_KEY,
    DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY,
    get_deepy_config_value,
    normalize_deepy_auto_cancel_queue_tasks,
)
from shared.deepy import tool_settings as deepy_tool_settings


ASSISTANT_OVERRIDE_DIMENSION_MIN = 256
ASSISTANT_OVERRIDE_DIMENSION_MAX = 3840
ASSISTANT_OVERRIDE_DIMENSION_STEP = 16
ASSISTANT_OVERRIDE_WIDTH_DEFAULT = 1280
ASSISTANT_OVERRIDE_HEIGHT_DEFAULT = 720
ASSISTANT_OVERRIDE_FRAMES_MIN = 5
ASSISTANT_OVERRIDE_FRAMES_MAX = 768
ASSISTANT_OVERRIDE_FRAMES_DEFAULT = 81
ASSISTANT_OVERRIDE_SEED_DEFAULT = -1
ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY = "deepy_use_template_properties"
ASSISTANT_OVERRIDE_WIDTH_KEY = "deepy_width"
ASSISTANT_OVERRIDE_HEIGHT_KEY = "deepy_height"
ASSISTANT_OVERRIDE_NUM_FRAMES_KEY = "deepy_num_frames"
ASSISTANT_OVERRIDE_SEED_KEY = "deepy_seed"


def _clamp_int(value: Any, default: int, minimum: int, maximum: int, step: int = 1) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = int(default)
    number = max(minimum, min(maximum, number))
    if step > 1:
        number = minimum + int(round((number - minimum) / step)) * step
        number = max(minimum, min(maximum, number))
    return int(number)


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "off", "no"}:
            return False
        if text in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def normalize_assistant_use_template_properties(value: Any) -> bool:
    return _normalize_bool(value)


def normalize_assistant_override_width(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_WIDTH_DEFAULT, ASSISTANT_OVERRIDE_DIMENSION_MIN, ASSISTANT_OVERRIDE_DIMENSION_MAX, ASSISTANT_OVERRIDE_DIMENSION_STEP)


def normalize_assistant_override_height(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT, ASSISTANT_OVERRIDE_DIMENSION_MIN, ASSISTANT_OVERRIDE_DIMENSION_MAX, ASSISTANT_OVERRIDE_DIMENSION_STEP)


def normalize_assistant_override_num_frames(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_FRAMES_DEFAULT, ASSISTANT_OVERRIDE_FRAMES_MIN, ASSISTANT_OVERRIDE_FRAMES_MAX, 1)


def normalize_assistant_override_seed(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_SEED_DEFAULT, -1, 999999999, 1)


def get_persisted_assistant_tool_ui_settings(server_config: dict[str, Any] | None = None) -> dict[str, Any]:
    source = server_config if isinstance(server_config, dict) else {}
    return normalize_assistant_tool_ui_settings(
        auto_cancel_queue_tasks=source.get(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, get_deepy_config_value(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT)),
        use_template_properties=source.get(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, get_deepy_config_value(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, True)),
        width=source.get(ASSISTANT_OVERRIDE_WIDTH_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_WIDTH_KEY, ASSISTANT_OVERRIDE_WIDTH_DEFAULT)),
        height=source.get(ASSISTANT_OVERRIDE_HEIGHT_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_HEIGHT_KEY, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT)),
        num_frames=source.get(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, ASSISTANT_OVERRIDE_FRAMES_DEFAULT)),
        seed=source.get(ASSISTANT_OVERRIDE_SEED_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_SEED_KEY, ASSISTANT_OVERRIDE_SEED_DEFAULT)),
        video_with_speech_variant=source.get(DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY, deepy_tool_settings.get_default_video_with_speech_variant())),
        image_generator_variant=source.get(DEEPY_TOOL_GEN_IMAGE_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_IMAGE_KEY, DEEPY_DEFAULT_GEN_IMAGE)),
        image_editor_variant=source.get(DEEPY_TOOL_EDIT_IMAGE_KEY, get_deepy_config_value(DEEPY_TOOL_EDIT_IMAGE_KEY, DEEPY_DEFAULT_EDIT_IMAGE)),
        video_generator_variant=source.get(DEEPY_TOOL_GEN_VIDEO_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_KEY, DEEPY_DEFAULT_GEN_VIDEO)),
        speech_from_description_variant=source.get(DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY, deepy_tool_settings.get_default_speech_from_description_variant())),
        speech_from_sample_variant=source.get(DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY, deepy_tool_settings.get_default_speech_from_sample_variant())),
    )


def store_assistant_tool_ui_settings(server_config: dict[str, Any] | None, settings: dict[str, Any] | None) -> bool:
    if not isinstance(server_config, dict):
        return False
    normalized = normalize_assistant_tool_ui_settings(**dict(settings or {}))
    changed = False
    updates = {
        DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY: normalized["auto_cancel_queue_tasks"],
        ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY: normalized["use_template_properties"],
        ASSISTANT_OVERRIDE_WIDTH_KEY: normalized["width"],
        ASSISTANT_OVERRIDE_HEIGHT_KEY: normalized["height"],
        ASSISTANT_OVERRIDE_NUM_FRAMES_KEY: normalized["num_frames"],
        ASSISTANT_OVERRIDE_SEED_KEY: normalized["seed"],
        DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY: normalized["video_with_speech_variant"],
        DEEPY_TOOL_GEN_IMAGE_KEY: normalized["image_generator_variant"],
        DEEPY_TOOL_EDIT_IMAGE_KEY: normalized["image_editor_variant"],
        DEEPY_TOOL_GEN_VIDEO_KEY: normalized["video_generator_variant"],
        DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY: normalized["speech_from_description_variant"],
        DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY: normalized["speech_from_sample_variant"],
    }
    for key, value in updates.items():
        if server_config.get(key) == value:
            continue
        server_config[key] = value
        changed = True
    return changed


def get_template_selector_state() -> dict[str, Any]:
    persisted = get_persisted_assistant_tool_ui_settings()
    return {
        "image_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_image", current_variant=persisted["image_generator_variant"]),
        "selected_image_generator": persisted["image_generator_variant"],
        "image_editor_choices": deepy_tool_settings.list_tool_variant_choices("edit_image", current_variant=persisted["image_editor_variant"]),
        "selected_image_editor": persisted["image_editor_variant"],
        "video_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_video", current_variant=persisted["video_generator_variant"]),
        "selected_video_generator": persisted["video_generator_variant"],
        "video_with_speech_choices": deepy_tool_settings.list_tool_variant_choices("gen_video_with_speech", current_variant=persisted["video_with_speech_variant"]),
        "selected_video_with_speech": persisted["video_with_speech_variant"],
        "speech_from_description_choices": deepy_tool_settings.list_tool_variant_choices("gen_speech_from_description", current_variant=persisted["speech_from_description_variant"]),
        "selected_speech_from_description": persisted["speech_from_description_variant"],
        "speech_from_sample_choices": deepy_tool_settings.list_tool_variant_choices("gen_speech_from_sample", current_variant=persisted["speech_from_sample_variant"]),
        "selected_speech_from_sample": persisted["speech_from_sample_variant"],
    }


def refresh_template_selector_state(current_image_generator: Any, current_image_editor: Any, current_video_generator: Any, current_video_with_speech: Any, current_speech_from_description: Any, current_speech_from_sample: Any) -> dict[str, Any]:
    deepy_tool_settings.refresh_tool_presets()
    return {
        "image_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_image", current_variant=current_image_generator),
        "selected_image_generator": deepy_tool_settings.find_tool_variant("gen_image", current_image_generator),
        "image_editor_choices": deepy_tool_settings.list_tool_variant_choices("edit_image", current_variant=current_image_editor),
        "selected_image_editor": deepy_tool_settings.find_tool_variant("edit_image", current_image_editor),
        "video_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_video", current_variant=current_video_generator),
        "selected_video_generator": deepy_tool_settings.find_tool_variant("gen_video", current_video_generator),
        "video_with_speech_choices": deepy_tool_settings.list_tool_variant_choices("gen_video_with_speech", current_variant=current_video_with_speech),
        "selected_video_with_speech": deepy_tool_settings.find_tool_variant("gen_video_with_speech", current_video_with_speech),
        "speech_from_description_choices": deepy_tool_settings.list_tool_variant_choices("gen_speech_from_description", current_variant=current_speech_from_description),
        "selected_speech_from_description": deepy_tool_settings.find_tool_variant("gen_speech_from_description", current_speech_from_description),
        "speech_from_sample_choices": deepy_tool_settings.list_tool_variant_choices("gen_speech_from_sample", current_variant=current_speech_from_sample),
        "selected_speech_from_sample": deepy_tool_settings.find_tool_variant("gen_speech_from_sample", current_speech_from_sample),
    }


def normalize_assistant_tool_ui_settings(
    *,
    auto_cancel_queue_tasks: Any = None,
    use_template_properties: Any = None,
    priority: Any = None,
    width: Any = None,
    height: Any = None,
    num_frames: Any = None,
    seed: Any = None,
    video_with_speech_variant: Any = None,
    image_generator_variant: Any = None,
    image_editor_variant: Any = None,
    video_generator_variant: Any = None,
    speech_from_description_variant: Any = None,
    speech_from_sample_variant: Any = None,
) -> dict[str, Any]:
    return {
        "auto_cancel_queue_tasks": normalize_deepy_auto_cancel_queue_tasks(get_deepy_config_value(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT) if auto_cancel_queue_tasks is None else auto_cancel_queue_tasks),
        "use_template_properties": normalize_assistant_use_template_properties(get_deepy_config_value(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, True) if use_template_properties is None else use_template_properties),
        "width": normalize_assistant_override_width(get_deepy_config_value(ASSISTANT_OVERRIDE_WIDTH_KEY, ASSISTANT_OVERRIDE_WIDTH_DEFAULT) if width is None else width),
        "height": normalize_assistant_override_height(get_deepy_config_value(ASSISTANT_OVERRIDE_HEIGHT_KEY, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT) if height is None else height),
        "num_frames": normalize_assistant_override_num_frames(get_deepy_config_value(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, ASSISTANT_OVERRIDE_FRAMES_DEFAULT) if num_frames is None else num_frames),
        "seed": normalize_assistant_override_seed(get_deepy_config_value(ASSISTANT_OVERRIDE_SEED_KEY, ASSISTANT_OVERRIDE_SEED_DEFAULT) if seed is None else seed),
        "video_with_speech_variant": deepy_tool_settings.resolve_tool_variant("gen_video_with_speech", get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY, deepy_tool_settings.get_default_video_with_speech_variant()) if video_with_speech_variant is None else video_with_speech_variant, default_variant=deepy_tool_settings.get_default_video_with_speech_variant()),
        "image_generator_variant": deepy_tool_settings.resolve_tool_variant("gen_image", get_deepy_config_value(DEEPY_TOOL_GEN_IMAGE_KEY, DEEPY_DEFAULT_GEN_IMAGE) if image_generator_variant is None else image_generator_variant, default_variant=DEEPY_DEFAULT_GEN_IMAGE),
        "image_editor_variant": deepy_tool_settings.resolve_tool_variant("edit_image", get_deepy_config_value(DEEPY_TOOL_EDIT_IMAGE_KEY, DEEPY_DEFAULT_EDIT_IMAGE) if image_editor_variant is None else image_editor_variant, default_variant=DEEPY_DEFAULT_EDIT_IMAGE),
        "video_generator_variant": deepy_tool_settings.resolve_tool_variant("gen_video", get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_KEY, DEEPY_DEFAULT_GEN_VIDEO) if video_generator_variant is None else video_generator_variant, default_variant=DEEPY_DEFAULT_GEN_VIDEO),
        "speech_from_description_variant": deepy_tool_settings.resolve_tool_variant("gen_speech_from_description", get_deepy_config_value(DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY, deepy_tool_settings.get_default_speech_from_description_variant()) if speech_from_description_variant is None else speech_from_description_variant, default_variant=deepy_tool_settings.get_default_speech_from_description_variant()),
        "speech_from_sample_variant": deepy_tool_settings.resolve_tool_variant("gen_speech_from_sample", get_deepy_config_value(DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY, deepy_tool_settings.get_default_speech_from_sample_variant()) if speech_from_sample_variant is None else speech_from_sample_variant, default_variant=deepy_tool_settings.get_default_speech_from_sample_variant()),
    }


__all__ = [
    "ASSISTANT_OVERRIDE_DIMENSION_MAX",
    "ASSISTANT_OVERRIDE_DIMENSION_MIN",
    "ASSISTANT_OVERRIDE_DIMENSION_STEP",
    "ASSISTANT_OVERRIDE_FRAMES_DEFAULT",
    "ASSISTANT_OVERRIDE_FRAMES_MAX",
    "ASSISTANT_OVERRIDE_FRAMES_MIN",
    "ASSISTANT_OVERRIDE_HEIGHT_DEFAULT",
    "ASSISTANT_OVERRIDE_HEIGHT_KEY",
    "ASSISTANT_OVERRIDE_NUM_FRAMES_KEY",
    "ASSISTANT_OVERRIDE_SEED_DEFAULT",
    "ASSISTANT_OVERRIDE_SEED_KEY",
    "ASSISTANT_OVERRIDE_WIDTH_DEFAULT",
    "ASSISTANT_OVERRIDE_WIDTH_KEY",
    "ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY",
    "get_persisted_assistant_tool_ui_settings",
    "store_assistant_tool_ui_settings",
    "get_template_selector_state",
    "normalize_assistant_override_height",
    "normalize_assistant_override_num_frames",
    "normalize_assistant_override_seed",
    "normalize_assistant_override_width",
    "normalize_assistant_tool_ui_settings",
    "normalize_assistant_use_template_properties",
    "refresh_template_selector_state",
]
