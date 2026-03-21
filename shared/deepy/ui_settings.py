from __future__ import annotations

from typing import Any

from shared.deepy.config import (
    DEEPY_DEFAULT_IMAGE_EDITOR,
    DEEPY_DEFAULT_IMAGE_EDITOR_KEY,
    DEEPY_DEFAULT_IMAGE_GENERATOR,
    DEEPY_DEFAULT_IMAGE_GENERATOR_KEY,
    DEEPY_DEFAULT_VIDEO_GENERATOR,
    DEEPY_DEFAULT_VIDEO_GENERATOR_KEY,
    get_deepy_config_value,
)
from shared.deepy import tool_settings as deepy_tool_settings


ASSISTANT_OVERRIDE_DIMENSION_MIN = 256
ASSISTANT_OVERRIDE_DIMENSION_MAX = 1920
ASSISTANT_OVERRIDE_DIMENSION_STEP = 16
ASSISTANT_OVERRIDE_WIDTH_DEFAULT = 1280
ASSISTANT_OVERRIDE_HEIGHT_DEFAULT = 720
ASSISTANT_OVERRIDE_FRAMES_MIN = 5
ASSISTANT_OVERRIDE_FRAMES_MAX = 768
ASSISTANT_OVERRIDE_FRAMES_DEFAULT = 81
ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY = "deepy_use_template_properties"
ASSISTANT_OVERRIDE_WIDTH_KEY = "deepy_width"
ASSISTANT_OVERRIDE_HEIGHT_KEY = "deepy_height"
ASSISTANT_OVERRIDE_NUM_FRAMES_KEY = "deepy_num_frames"


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


def normalize_assistant_priority(value: Any) -> bool:
    return _normalize_bool(value)


def normalize_assistant_override_width(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_WIDTH_DEFAULT, ASSISTANT_OVERRIDE_DIMENSION_MIN, ASSISTANT_OVERRIDE_DIMENSION_MAX, ASSISTANT_OVERRIDE_DIMENSION_STEP)


def normalize_assistant_override_height(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT, ASSISTANT_OVERRIDE_DIMENSION_MIN, ASSISTANT_OVERRIDE_DIMENSION_MAX, ASSISTANT_OVERRIDE_DIMENSION_STEP)


def normalize_assistant_override_num_frames(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_FRAMES_DEFAULT, ASSISTANT_OVERRIDE_FRAMES_MIN, ASSISTANT_OVERRIDE_FRAMES_MAX, 1)


def get_persisted_assistant_tool_ui_settings(server_config: dict[str, Any] | None = None, *, priority: Any = False) -> dict[str, Any]:
    source = server_config if isinstance(server_config, dict) else {}
    return normalize_assistant_tool_ui_settings(
        use_template_properties=source.get(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, get_deepy_config_value(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, True)),
        priority=priority,
        width=source.get(ASSISTANT_OVERRIDE_WIDTH_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_WIDTH_KEY, ASSISTANT_OVERRIDE_WIDTH_DEFAULT)),
        height=source.get(ASSISTANT_OVERRIDE_HEIGHT_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_HEIGHT_KEY, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT)),
        num_frames=source.get(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, ASSISTANT_OVERRIDE_FRAMES_DEFAULT)),
        image_generator_variant=source.get(DEEPY_DEFAULT_IMAGE_GENERATOR_KEY, get_deepy_config_value(DEEPY_DEFAULT_IMAGE_GENERATOR_KEY, DEEPY_DEFAULT_IMAGE_GENERATOR)),
        image_editor_variant=source.get(DEEPY_DEFAULT_IMAGE_EDITOR_KEY, get_deepy_config_value(DEEPY_DEFAULT_IMAGE_EDITOR_KEY, DEEPY_DEFAULT_IMAGE_EDITOR)),
        video_generator_variant=source.get(DEEPY_DEFAULT_VIDEO_GENERATOR_KEY, get_deepy_config_value(DEEPY_DEFAULT_VIDEO_GENERATOR_KEY, DEEPY_DEFAULT_VIDEO_GENERATOR)),
    )


def store_assistant_tool_ui_settings(server_config: dict[str, Any] | None, settings: dict[str, Any] | None) -> bool:
    if not isinstance(server_config, dict):
        return False
    normalized = normalize_assistant_tool_ui_settings(**dict(settings or {}))
    changed = False
    updates = {
        ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY: normalized["use_template_properties"],
        ASSISTANT_OVERRIDE_WIDTH_KEY: normalized["width"],
        ASSISTANT_OVERRIDE_HEIGHT_KEY: normalized["height"],
        ASSISTANT_OVERRIDE_NUM_FRAMES_KEY: normalized["num_frames"],
        DEEPY_DEFAULT_IMAGE_GENERATOR_KEY: normalized["image_generator_variant"],
        DEEPY_DEFAULT_IMAGE_EDITOR_KEY: normalized["image_editor_variant"],
        DEEPY_DEFAULT_VIDEO_GENERATOR_KEY: normalized["video_generator_variant"],
    }
    for key, value in updates.items():
        if server_config.get(key) == value:
            continue
        server_config[key] = value
        changed = True
    return changed


def get_template_selector_state() -> dict[str, Any]:
    return {
        "image_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_image"),
        "selected_image_generator": deepy_tool_settings.get_default_image_generator_variant(),
        "image_editor_choices": deepy_tool_settings.list_tool_variant_choices("edit_image"),
        "selected_image_editor": deepy_tool_settings.get_default_image_editor_variant(),
        "video_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_video"),
        "selected_video_generator": deepy_tool_settings.get_default_video_generator_variant(),
    }


def refresh_template_selector_state(current_image_generator: Any, current_image_editor: Any, current_video_generator: Any) -> dict[str, Any]:
    deepy_tool_settings.refresh_tool_presets()
    return {
        "image_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_image"),
        "selected_image_generator": deepy_tool_settings.find_tool_variant("gen_image", current_image_generator),
        "image_editor_choices": deepy_tool_settings.list_tool_variant_choices("edit_image"),
        "selected_image_editor": deepy_tool_settings.find_tool_variant("edit_image", current_image_editor),
        "video_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_video"),
        "selected_video_generator": deepy_tool_settings.find_tool_variant("gen_video", current_video_generator),
    }


def normalize_assistant_tool_ui_settings(
    *,
    use_template_properties: Any = None,
    priority: Any = False,
    width: Any = None,
    height: Any = None,
    num_frames: Any = None,
    image_generator_variant: Any = None,
    image_editor_variant: Any = None,
    video_generator_variant: Any = None,
) -> dict[str, Any]:
    return {
        "use_template_properties": normalize_assistant_use_template_properties(get_deepy_config_value(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, True) if use_template_properties is None else use_template_properties),
        "priority": normalize_assistant_priority(priority),
        "width": normalize_assistant_override_width(get_deepy_config_value(ASSISTANT_OVERRIDE_WIDTH_KEY, ASSISTANT_OVERRIDE_WIDTH_DEFAULT) if width is None else width),
        "height": normalize_assistant_override_height(get_deepy_config_value(ASSISTANT_OVERRIDE_HEIGHT_KEY, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT) if height is None else height),
        "num_frames": normalize_assistant_override_num_frames(get_deepy_config_value(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, ASSISTANT_OVERRIDE_FRAMES_DEFAULT) if num_frames is None else num_frames),
        "image_generator_variant": deepy_tool_settings.resolve_tool_variant("gen_image", get_deepy_config_value(DEEPY_DEFAULT_IMAGE_GENERATOR_KEY, DEEPY_DEFAULT_IMAGE_GENERATOR) if image_generator_variant is None else image_generator_variant, default_variant=DEEPY_DEFAULT_IMAGE_GENERATOR),
        "image_editor_variant": deepy_tool_settings.resolve_tool_variant("edit_image", get_deepy_config_value(DEEPY_DEFAULT_IMAGE_EDITOR_KEY, DEEPY_DEFAULT_IMAGE_EDITOR) if image_editor_variant is None else image_editor_variant, default_variant=DEEPY_DEFAULT_IMAGE_EDITOR),
        "video_generator_variant": deepy_tool_settings.resolve_tool_variant("gen_video", get_deepy_config_value(DEEPY_DEFAULT_VIDEO_GENERATOR_KEY, DEEPY_DEFAULT_VIDEO_GENERATOR) if video_generator_variant is None else video_generator_variant, default_variant=DEEPY_DEFAULT_VIDEO_GENERATOR),
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
    "ASSISTANT_OVERRIDE_WIDTH_DEFAULT",
    "ASSISTANT_OVERRIDE_WIDTH_KEY",
    "ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY",
    "get_persisted_assistant_tool_ui_settings",
    "store_assistant_tool_ui_settings",
    "get_template_selector_state",
    "normalize_assistant_override_height",
    "normalize_assistant_override_num_frames",
    "normalize_assistant_override_width",
    "normalize_assistant_priority",
    "normalize_assistant_tool_ui_settings",
    "normalize_assistant_use_template_properties",
    "refresh_template_selector_state",
]
