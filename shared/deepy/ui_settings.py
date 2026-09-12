from __future__ import annotations

from typing import Any

from shared.deepy.config import (
    DEEPY_MODEL_SPEED_KEY,
    DEEPY_MODEL_SIZE_KEY,
    DEEPY_TYPE_KEY,
    DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT,
    DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY,
    DEEPY_DEFAULT_EDIT_IMAGE,
    DEEPY_DEFAULT_GEN_IMAGE,
    DEEPY_DEFAULT_GEN_SONG,
    DEEPY_DEFAULT_GEN_VIDEO_WITH_REFS,
    DEEPY_DEFAULT_GEN_VIDEO,
    DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT,
    DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY,
    DEEPY_SESSION_GALLERY_MEDIA_MODE_DEFAULT,
    DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY,
    DEEPY_SESSION_RESET_MODE_DEFAULT,
    DEEPY_SESSION_RESET_MODE_KEY,
    DEEPY_MULTI_SESSION_DEFAULT,
    DEEPY_MULTI_SESSION_KEY,
    DEEPY_TOOL_EDIT_IMAGE_KEY,
    DEEPY_TOOL_GEN_IMAGE_KEY,
    DEEPY_TOOL_GEN_SONG_KEY,
    DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY,
    DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY,
    DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY,
    DEEPY_TOOL_GEN_VIDEO_KEY,
    DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY,
    get_deepy_config_value,
    normalize_deepy_auto_cancel_queue_tasks,
    normalize_deepy_separate_requests_with_empty_line,
    normalize_deepy_session_gallery_media_mode,
    normalize_deepy_session_reset_mode,
    normalize_deepy_session_mode,
)
from shared.deepy import tool_settings as deepy_tool_settings


ASSISTANT_OVERRIDE_DIMENSION_MIN = 256
ASSISTANT_OVERRIDE_DIMENSION_MAX = 3840
ASSISTANT_OVERRIDE_DIMENSION_STEP = 16
ASSISTANT_OVERRIDE_WIDTH_DEFAULT = 1280
ASSISTANT_OVERRIDE_HEIGHT_DEFAULT = 720
ASSISTANT_OVERRIDE_FRAMES_MIN = 5
ASSISTANT_OVERRIDE_FRAMES_MAX = 3000
ASSISTANT_OVERRIDE_FRAMES_DEFAULT = 81
ASSISTANT_OVERRIDE_AUDIO_DURATION_MIN = 1
ASSISTANT_OVERRIDE_AUDIO_DURATION_MAX = 600
ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT = 10
ASSISTANT_OVERRIDE_SEED_DEFAULT = -1
ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY = "deepy_use_template_properties"
ASSISTANT_OVERRIDE_WIDTH_KEY = "deepy_width"
ASSISTANT_OVERRIDE_HEIGHT_KEY = "deepy_height"
ASSISTANT_OVERRIDE_NUM_FRAMES_KEY = "deepy_num_frames"
ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY = "deepy_audio_duration"
ASSISTANT_OVERRIDE_SEED_KEY = "deepy_seed"


TEMPLATE_TOOL_LAYOUT = (
    ("gen_image", "edit_image"),
    ("gen_video", "gen_video_with_speech"),
    ("gen_video_with_refs", "gen_song"),
    ("gen_speech_from_description", "gen_speech_from_sample"),
)
TEMPLATE_TOOL_UI_KEY = {
    "gen_video": "video_generator_variant",
    "gen_video_with_speech": "video_with_speech_variant",
    "gen_image": "image_generator_variant",
    "gen_song": "song_variant",
    "gen_video_with_refs": "with_refs_variant",
    "edit_image": "image_editor_variant",
    "gen_speech_from_description": "speech_from_description_variant",
    "gen_speech_from_sample": "speech_from_sample_variant",
}

MODEL_SELECTION_FIELDS = [
    {"key": "model_speed", "label": "Speed", "choices": [("Fast", "fast"), ("Standard", "standard"), ("No preference", "any")]},
    {"key": "model_size", "label": "Model Size", "choices": [("Smaller", "smaller"), ("Larger", "larger"), ("No preference", "any")]},
]


def normalize_model_preference(value, key):
    field = next(field for field in MODEL_SELECTION_FIELDS if field["key"] == key)
    if value not in [choice[1] for choice in field["choices"]]:
        raise ValueError(f"Invalid {field['label']} preference: {value}")
    return value


PROPERTY_MODE_CHOICES = [
    ("Use template defaults", True),
    ("Use the values below", False),
]
GENERATION_PROPERTY_FIELDS = [
    {"key": "width", "label": "Default Width", "minimum": ASSISTANT_OVERRIDE_DIMENSION_MIN, "maximum": ASSISTANT_OVERRIDE_DIMENSION_MAX, "step": ASSISTANT_OVERRIDE_DIMENSION_STEP},
    {"key": "height", "label": "Default Height", "minimum": ASSISTANT_OVERRIDE_DIMENSION_MIN, "maximum": ASSISTANT_OVERRIDE_DIMENSION_MAX, "step": ASSISTANT_OVERRIDE_DIMENSION_STEP},
    {"key": "num_frames", "label": "Default Number of Frames", "minimum": ASSISTANT_OVERRIDE_FRAMES_MIN, "maximum": ASSISTANT_OVERRIDE_FRAMES_MAX, "step": 1},
    {"key": "audio_duration", "label": "Default Audio Duration (seconds)", "minimum": ASSISTANT_OVERRIDE_AUDIO_DURATION_MIN, "maximum": ASSISTANT_OVERRIDE_AUDIO_DURATION_MAX, "step": 1},
    {"key": "seed", "label": "Seed (-1 for random)", "minimum": -1, "maximum": 999999999, "step": 1},
]


def get_simplified_settings_form(settings, *, prime=None):
    if prime is None:
        prime = get_deepy_config_value(DEEPY_TYPE_KEY, "zero") == "prime"
    preferences = MODEL_SELECTION_FIELDS if prime else []
    templates = [{"key": TEMPLATE_TOOL_UI_KEY[tool], "label": deepy_tool_settings.TOOL_DISPLAY_NAMES[tool], "choices": deepy_tool_settings.list_tool_variant_choices(tool, current_variant=settings[TEMPLATE_TOOL_UI_KEY[tool]])} for row in TEMPLATE_TOOL_LAYOUT for tool in row]
    keys = [*(field["key"] for field in preferences), "use_template_properties", *(field["key"] for field in GENERATION_PROPERTY_FIELDS), *(field["key"] for field in templates)]
    return {"values": {key: settings[key] for key in keys}, "preferences": preferences, "property_modes": PROPERTY_MODE_CHOICES, "properties": GENERATION_PROPERTY_FIELDS, "templates": templates}


def validate_simplified_settings(values, form):
    if not isinstance(values, dict) or values.keys() - form["values"].keys():
        raise ValueError("Only the displayed Deepy settings can be changed here.")
    fields = {field["key"]: field for field in form["properties"]}
    choices = {field["key"]: [value for label, value in field["choices"]] for field in [*form["templates"], *form["preferences"]]}
    for key, value in values.items():
        if key == "use_template_properties":
            if type(value) is not bool:
                raise ValueError("The property mode must be a boolean.")
        elif key in fields:
            field = fields[key]
            if type(value) is not int or not field["minimum"] <= value <= field["maximum"] or (value - field["minimum"]) % field["step"]:
                raise ValueError(f'{field["label"]}: expected {field["minimum"]}–{field["maximum"]}, step {field["step"]}.')
        elif not isinstance(value, str) or value not in choices[key]:
            raise ValueError("Select one of the displayed choices.")


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


def normalize_assistant_override_audio_duration(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT, ASSISTANT_OVERRIDE_AUDIO_DURATION_MIN, ASSISTANT_OVERRIDE_AUDIO_DURATION_MAX, 1)


def normalize_assistant_override_seed(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_SEED_DEFAULT, -1, 999999999, 1)


def normalize_assistant_separate_requests_with_empty_line(value: Any) -> bool:
    return normalize_deepy_separate_requests_with_empty_line(value)


def get_persisted_assistant_tool_ui_settings(server_config: dict[str, Any] | None = None) -> dict[str, Any]:
    source = server_config if isinstance(server_config, dict) else {}
    return normalize_assistant_tool_ui_settings(
        auto_cancel_queue_tasks=source.get(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, get_deepy_config_value(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT)),
        separate_requests_with_empty_line=source.get(DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY, get_deepy_config_value(DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY, DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT)),
        model_speed=source.get(DEEPY_MODEL_SPEED_KEY, get_deepy_config_value(DEEPY_MODEL_SPEED_KEY, "fast")),
        model_size=source.get(DEEPY_MODEL_SIZE_KEY, get_deepy_config_value(DEEPY_MODEL_SIZE_KEY, "smaller")),
        use_template_properties=source.get(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, get_deepy_config_value(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, True)),
        width=source.get(ASSISTANT_OVERRIDE_WIDTH_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_WIDTH_KEY, ASSISTANT_OVERRIDE_WIDTH_DEFAULT)),
        height=source.get(ASSISTANT_OVERRIDE_HEIGHT_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_HEIGHT_KEY, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT)),
        num_frames=source.get(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, ASSISTANT_OVERRIDE_FRAMES_DEFAULT)),
        audio_duration=source.get(ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY, ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT)),
        seed=source.get(ASSISTANT_OVERRIDE_SEED_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_SEED_KEY, ASSISTANT_OVERRIDE_SEED_DEFAULT)),
        video_with_speech_variant=source.get(DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY, deepy_tool_settings.get_default_video_with_speech_variant())),
        image_generator_variant=source.get(DEEPY_TOOL_GEN_IMAGE_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_IMAGE_KEY, DEEPY_DEFAULT_GEN_IMAGE)),
        song_variant=source.get(DEEPY_TOOL_GEN_SONG_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_SONG_KEY, DEEPY_DEFAULT_GEN_SONG)),
        with_refs_variant=source.get(DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY, DEEPY_DEFAULT_GEN_VIDEO_WITH_REFS)),
        image_editor_variant=source.get(DEEPY_TOOL_EDIT_IMAGE_KEY, get_deepy_config_value(DEEPY_TOOL_EDIT_IMAGE_KEY, DEEPY_DEFAULT_EDIT_IMAGE)),
        video_generator_variant=source.get(DEEPY_TOOL_GEN_VIDEO_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_KEY, DEEPY_DEFAULT_GEN_VIDEO)),
        speech_from_description_variant=source.get(DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY, deepy_tool_settings.get_default_speech_from_description_variant())),
        speech_from_sample_variant=source.get(DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY, get_deepy_config_value(DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY, deepy_tool_settings.get_default_speech_from_sample_variant())),
    )


def store_assistant_tool_ui_settings(server_config: dict[str, Any] | None, settings: dict[str, Any] | None) -> bool:
    if not isinstance(server_config, dict):
        return False
    normalized = normalize_assistant_tool_ui_settings(**dict(settings or {}))
    updates = {
        DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY: normalized["auto_cancel_queue_tasks"],
        DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY: normalized["separate_requests_with_empty_line"],
        DEEPY_MODEL_SPEED_KEY: normalized["model_speed"],
        DEEPY_MODEL_SIZE_KEY: normalized["model_size"],
        ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY: normalized["use_template_properties"],
        ASSISTANT_OVERRIDE_WIDTH_KEY: normalized["width"],
        ASSISTANT_OVERRIDE_HEIGHT_KEY: normalized["height"],
        ASSISTANT_OVERRIDE_NUM_FRAMES_KEY: normalized["num_frames"],
        ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY: normalized["audio_duration"],
        ASSISTANT_OVERRIDE_SEED_KEY: normalized["seed"],
        DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY: normalized["video_with_speech_variant"],
        DEEPY_TOOL_GEN_IMAGE_KEY: normalized["image_generator_variant"],
        DEEPY_TOOL_GEN_SONG_KEY: normalized["song_variant"],
        DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY: normalized["with_refs_variant"],
        DEEPY_TOOL_EDIT_IMAGE_KEY: normalized["image_editor_variant"],
        DEEPY_TOOL_GEN_VIDEO_KEY: normalized["video_generator_variant"],
        DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY: normalized["speech_from_description_variant"],
        DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY: normalized["speech_from_sample_variant"],
    }
    server_config.update(updates)
    return True


def get_persisted_assistant_session_ui_settings(server_config: dict[str, Any] | None = None) -> dict[str, Any]:
    source = server_config if isinstance(server_config, dict) else {}
    return {
        "multi_session": normalize_deepy_session_mode(source.get(DEEPY_MULTI_SESSION_KEY, get_deepy_config_value(DEEPY_MULTI_SESSION_KEY, DEEPY_MULTI_SESSION_DEFAULT))),
        "reset_mode": normalize_deepy_session_reset_mode(source.get(DEEPY_SESSION_RESET_MODE_KEY, get_deepy_config_value(DEEPY_SESSION_RESET_MODE_KEY, DEEPY_SESSION_RESET_MODE_DEFAULT))),
        "gallery_media_mode": normalize_deepy_session_gallery_media_mode(source.get(DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY, get_deepy_config_value(DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY, DEEPY_SESSION_GALLERY_MEDIA_MODE_DEFAULT))),
    }


def store_assistant_session_ui_settings(server_config: dict[str, Any] | None, *, multi_session: Any, reset_mode: Any, gallery_media_mode: Any) -> bool:
    if not isinstance(server_config, dict):
        return False
    server_config.update({
        DEEPY_MULTI_SESSION_KEY: normalize_deepy_session_mode(multi_session),
        DEEPY_SESSION_RESET_MODE_KEY: normalize_deepy_session_reset_mode(reset_mode),
        DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY: normalize_deepy_session_gallery_media_mode(gallery_media_mode),
    })
    return True


def get_template_selector_state() -> dict[str, Any]:
    persisted = get_persisted_assistant_tool_ui_settings()
    return {
        "image_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_image", current_variant=persisted["image_generator_variant"]),
        "selected_image_generator": persisted["image_generator_variant"],
        "song_choices": deepy_tool_settings.list_tool_variant_choices("gen_song", current_variant=persisted["song_variant"]),
        "with_refs_choices": deepy_tool_settings.list_tool_variant_choices("gen_video_with_refs", current_variant=persisted["with_refs_variant"]),
        "selected_song": persisted["song_variant"],
        "selected_with_refs": persisted["with_refs_variant"],
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


def refresh_template_selector_state(current_image_generator: Any, current_image_editor: Any, current_video_generator: Any, current_video_with_speech: Any, current_song: Any, current_with_refs: Any, current_speech_from_description: Any, current_speech_from_sample: Any) -> dict[str, Any]:
    deepy_tool_settings.refresh_tool_presets()
    return {
        "image_generator_choices": deepy_tool_settings.list_tool_variant_choices("gen_image", current_variant=current_image_generator),
        "selected_image_generator": deepy_tool_settings.find_tool_variant("gen_image", current_image_generator),
        "song_choices": deepy_tool_settings.list_tool_variant_choices("gen_song", current_variant=current_song),
        "with_refs_choices": deepy_tool_settings.list_tool_variant_choices("gen_video_with_refs", current_variant=current_with_refs),
        "selected_song": deepy_tool_settings.find_tool_variant("gen_song", current_song),
        "selected_with_refs": deepy_tool_settings.find_tool_variant("gen_video_with_refs", current_with_refs),
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
    separate_requests_with_empty_line: Any = None,
    use_template_properties: Any = None,
    priority: Any = None,
    model_speed: Any = None,
    model_size: Any = None,
    width: Any = None,
    height: Any = None,
    num_frames: Any = None,
    audio_duration: Any = None,
    seed: Any = None,
    video_with_speech_variant: Any = None,
    image_generator_variant: Any = None,
    song_variant: Any = None,
    with_refs_variant: Any = None,
    image_editor_variant: Any = None,
    video_generator_variant: Any = None,
    speech_from_description_variant: Any = None,
    speech_from_sample_variant: Any = None,
) -> dict[str, Any]:
    return {
        "auto_cancel_queue_tasks": normalize_deepy_auto_cancel_queue_tasks(get_deepy_config_value(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT) if auto_cancel_queue_tasks is None else auto_cancel_queue_tasks),
        "separate_requests_with_empty_line": normalize_assistant_separate_requests_with_empty_line(get_deepy_config_value(DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY, DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT) if separate_requests_with_empty_line is None else separate_requests_with_empty_line),
        "model_speed": normalize_model_preference(get_deepy_config_value(DEEPY_MODEL_SPEED_KEY, "fast") if model_speed is None else model_speed, "model_speed"),
        "model_size": normalize_model_preference(get_deepy_config_value(DEEPY_MODEL_SIZE_KEY, "smaller") if model_size is None else model_size, "model_size"),
        "use_template_properties": normalize_assistant_use_template_properties(get_deepy_config_value(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, True) if use_template_properties is None else use_template_properties),
        "width": normalize_assistant_override_width(get_deepy_config_value(ASSISTANT_OVERRIDE_WIDTH_KEY, ASSISTANT_OVERRIDE_WIDTH_DEFAULT) if width is None else width),
        "height": normalize_assistant_override_height(get_deepy_config_value(ASSISTANT_OVERRIDE_HEIGHT_KEY, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT) if height is None else height),
        "num_frames": normalize_assistant_override_num_frames(get_deepy_config_value(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, ASSISTANT_OVERRIDE_FRAMES_DEFAULT) if num_frames is None else num_frames),
        "audio_duration": normalize_assistant_override_audio_duration(get_deepy_config_value(ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY, ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT) if audio_duration is None else audio_duration),
        "seed": normalize_assistant_override_seed(get_deepy_config_value(ASSISTANT_OVERRIDE_SEED_KEY, ASSISTANT_OVERRIDE_SEED_DEFAULT) if seed is None else seed),
        "video_with_speech_variant": deepy_tool_settings.resolve_tool_variant("gen_video_with_speech", get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY, deepy_tool_settings.get_default_video_with_speech_variant()) if video_with_speech_variant is None else video_with_speech_variant, default_variant=deepy_tool_settings.get_default_video_with_speech_variant()),
        "image_generator_variant": deepy_tool_settings.resolve_tool_variant("gen_image", get_deepy_config_value(DEEPY_TOOL_GEN_IMAGE_KEY, DEEPY_DEFAULT_GEN_IMAGE) if image_generator_variant is None else image_generator_variant, default_variant=DEEPY_DEFAULT_GEN_IMAGE),
        "song_variant": deepy_tool_settings.resolve_tool_variant("gen_song", get_deepy_config_value(DEEPY_TOOL_GEN_SONG_KEY, DEEPY_DEFAULT_GEN_SONG) if song_variant is None else song_variant, default_variant=DEEPY_DEFAULT_GEN_SONG),
        "with_refs_variant": deepy_tool_settings.resolve_tool_variant("gen_video_with_refs", get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY, DEEPY_DEFAULT_GEN_VIDEO_WITH_REFS) if with_refs_variant is None else with_refs_variant, default_variant=DEEPY_DEFAULT_GEN_VIDEO_WITH_REFS),
        "image_editor_variant": deepy_tool_settings.resolve_tool_variant("edit_image", get_deepy_config_value(DEEPY_TOOL_EDIT_IMAGE_KEY, DEEPY_DEFAULT_EDIT_IMAGE) if image_editor_variant is None else image_editor_variant, default_variant=DEEPY_DEFAULT_EDIT_IMAGE),
        "video_generator_variant": deepy_tool_settings.resolve_tool_variant("gen_video", get_deepy_config_value(DEEPY_TOOL_GEN_VIDEO_KEY, DEEPY_DEFAULT_GEN_VIDEO) if video_generator_variant is None else video_generator_variant, default_variant=DEEPY_DEFAULT_GEN_VIDEO),
        "speech_from_description_variant": deepy_tool_settings.resolve_tool_variant("gen_speech_from_description", get_deepy_config_value(DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY, deepy_tool_settings.get_default_speech_from_description_variant()) if speech_from_description_variant is None else speech_from_description_variant, default_variant=deepy_tool_settings.get_default_speech_from_description_variant()),
        "speech_from_sample_variant": deepy_tool_settings.resolve_tool_variant("gen_speech_from_sample", get_deepy_config_value(DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY, deepy_tool_settings.get_default_speech_from_sample_variant()) if speech_from_sample_variant is None else speech_from_sample_variant, default_variant=deepy_tool_settings.get_default_speech_from_sample_variant()),
    }


__all__ = [
    "GENERATION_PROPERTY_FIELDS",
    "PROPERTY_MODE_CHOICES",
    "TEMPLATE_TOOL_LAYOUT",
    "TEMPLATE_TOOL_UI_KEY",
    "get_simplified_settings_form",
    "validate_simplified_settings",
    "ASSISTANT_OVERRIDE_DIMENSION_MAX",
    "ASSISTANT_OVERRIDE_DIMENSION_MIN",
    "ASSISTANT_OVERRIDE_DIMENSION_STEP",
    "ASSISTANT_OVERRIDE_FRAMES_DEFAULT",
    "ASSISTANT_OVERRIDE_FRAMES_MAX",
    "ASSISTANT_OVERRIDE_FRAMES_MIN",
    "ASSISTANT_OVERRIDE_HEIGHT_DEFAULT",
    "ASSISTANT_OVERRIDE_HEIGHT_KEY",
    "ASSISTANT_OVERRIDE_NUM_FRAMES_KEY",
    "ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT",
    "ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY",
    "ASSISTANT_OVERRIDE_AUDIO_DURATION_MAX",
    "ASSISTANT_OVERRIDE_AUDIO_DURATION_MIN",
    "ASSISTANT_OVERRIDE_SEED_DEFAULT",
    "ASSISTANT_OVERRIDE_SEED_KEY",
    "ASSISTANT_OVERRIDE_WIDTH_DEFAULT",
    "ASSISTANT_OVERRIDE_WIDTH_KEY",
    "ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY",
    "get_persisted_assistant_tool_ui_settings",
    "get_persisted_assistant_session_ui_settings",
    "store_assistant_tool_ui_settings",
    "store_assistant_session_ui_settings",
    "get_template_selector_state",
    "normalize_assistant_override_height",
    "normalize_assistant_override_num_frames",
    "normalize_assistant_override_audio_duration",
    "normalize_assistant_override_seed",
    "normalize_assistant_override_width",
    "normalize_assistant_separate_requests_with_empty_line",
    "normalize_assistant_tool_ui_settings",
    "normalize_assistant_use_template_properties",
    "refresh_template_selector_state",
]
