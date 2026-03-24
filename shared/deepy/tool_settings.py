from __future__ import annotations

import copy
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from shared.deepy.config import (
    DEEPY_DEFAULT_EDIT_IMAGE,
    DEEPY_DEFAULT_GEN_IMAGE,
    DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION,
    DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE,
    DEEPY_DEFAULT_GEN_VIDEO,
    DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH,
    DEEPY_TOOL_EDIT_IMAGE_KEY,
    DEEPY_TOOL_GEN_IMAGE_KEY,
    DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY,
    DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY,
    DEEPY_TOOL_GEN_VIDEO_KEY,
    DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY,
    get_deepy_config_value,
    normalize_deepy_tool_edit_image,
    normalize_deepy_tool_gen_image,
    normalize_deepy_tool_gen_speech_from_description,
    normalize_deepy_tool_gen_speech_from_sample,
    normalize_deepy_tool_gen_video,
    normalize_deepy_tool_gen_video_with_speech,
)


_DEEPY_DIR = Path(__file__).resolve().parent
SETTINGS_DIR = _DEEPY_DIR / "settings"
DEFAULT_IMAGE_EDITOR_VARIANT = DEEPY_DEFAULT_EDIT_IMAGE
DEFAULT_VIDEO_WITH_SPEECH_VARIANT = DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH
DEFAULT_SPEECH_FROM_DESCRIPTION_VARIANT = DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION
DEFAULT_SPEECH_FROM_SAMPLE_VARIANT = DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE
TOOL_DISPLAY_NAMES = {
    "gen_image": "Image Generator",
    "edit_image": "Image Editor",
    "gen_video": "Video Generator",
    "gen_video_with_speech": "Video With Speech",
    "gen_speech_from_description": "Speech From Description",
    "gen_speech_from_sample": "Speech From Sample",
}
_TOOL_TEMPLATE_VALIDATION_ERRORS = {
    "gen_video": "The settings should generate a video",
    "gen_video_with_speech": "The settings should generate a Video and accept an Audio Prompt",
    "gen_image": "The settings of the model must generate an Image",
    "edit_image": "The settings of the model must generate an Image and accept an Image Ref",
    "gen_speech_from_description": "The model should generate only an audio output",
    "gen_speech_from_sample": "The model should generate only an audio output and a sample audio ois expected",
}
_TOOL_CONFIG_SPECS = {
    "gen_image": {"key": DEEPY_TOOL_GEN_IMAGE_KEY, "default": DEEPY_DEFAULT_GEN_IMAGE, "normalize": normalize_deepy_tool_gen_image},
    "edit_image": {"key": DEEPY_TOOL_EDIT_IMAGE_KEY, "default": DEEPY_DEFAULT_EDIT_IMAGE, "normalize": normalize_deepy_tool_edit_image},
    "gen_video": {"key": DEEPY_TOOL_GEN_VIDEO_KEY, "default": DEEPY_DEFAULT_GEN_VIDEO, "normalize": normalize_deepy_tool_gen_video},
    "gen_video_with_speech": {
        "key": DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY,
        "default": DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH,
        "normalize": normalize_deepy_tool_gen_video_with_speech,
    },
    "gen_speech_from_description": {
        "key": DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY,
        "default": DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION,
        "normalize": normalize_deepy_tool_gen_speech_from_description,
    },
    "gen_speech_from_sample": {
        "key": DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY,
        "default": DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE,
        "normalize": normalize_deepy_tool_gen_speech_from_sample,
    },
}
_PRESET_SOURCE_BUILTIN = "builtin"
_PRESET_SOURCE_LINKED = "linked"
_PRESET_SOURCE_PRIORITY = {
    _PRESET_SOURCE_BUILTIN: 0,
    _PRESET_SOURCE_LINKED: 1,
}
_LEGACY_VARIANT_ALIASES = {
    "edit_image": {"Qwen_Edit": DEEPY_DEFAULT_EDIT_IMAGE},
    "gen_image": {"Z_Image_Turbo": DEEPY_DEFAULT_GEN_IMAGE},
    "gen_video": {"ltx2_22B_distilled": DEEPY_DEFAULT_GEN_VIDEO},
}
_LIVE_FILE_PRESET_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _canonical_variant(tool_name: str, variant: Any) -> str:
    text = str(variant or "").strip()
    if len(text) == 0:
        return ""
    return _LEGACY_VARIANT_ALIASES.get(str(tool_name or "").strip(), {}).get(text, text)


def _tool_config_spec(tool_name: str) -> dict[str, Any]:
    return _TOOL_CONFIG_SPECS.get(str(tool_name or "").strip(), {})


def _get_configured_tool_variant(tool_name: str) -> str:
    spec = _tool_config_spec(tool_name)
    if len(spec) == 0:
        return ""
    raw_value = get_deepy_config_value(spec["key"], spec["default"])
    return str(spec["normalize"](raw_value) or "").strip()


def _looks_like_linked_variant(value: Any) -> bool:
    text = str(value or "").strip().strip('"').replace("\\", "/")
    if len(text) == 0 or not text.lower().endswith(".json"):
        return False
    if text.startswith("/") or text.startswith("./") or text.startswith("../"):
        return False
    if len(text) >= 2 and text[1] == ":":
        return False
    parts = [part.strip() for part in text.split("/")]
    return len(parts) == 2 and all(parts)


def _normalize_linked_variant(value: Any) -> str | None:
    if not _looks_like_linked_variant(value):
        return None
    base_model_type, filename = [part.strip() for part in str(value or "").strip().strip('"').replace("\\", "/").split("/", 1)]
    return f"{base_model_type}/{Path(filename).name}"


def _parse_linked_variant(value: Any) -> tuple[str, str] | None:
    normalized = _normalize_linked_variant(value)
    if normalized is None:
        return None
    return tuple(normalized.split("/", 1))  # type: ignore[return-value]


def _get_main_callable(name: str) -> Any:
    main_module = sys.modules.get("__main__")
    return None if main_module is None else getattr(main_module, str(name or "").strip(), None)


def _get_base_model_type_name(model_type: Any) -> str:
    text = str(model_type or "").strip()
    if len(text) == 0:
        return ""
    get_base_model_type = _get_main_callable("get_base_model_type")
    if callable(get_base_model_type):
        try:
            resolved = str(get_base_model_type(text) or "").strip()
        except Exception:
            resolved = ""
        if len(resolved) > 0:
            return resolved
    return text


def _resolve_linked_variant_path(value: Any) -> Path | None:
    parsed = _parse_linked_variant(value)
    if parsed is None:
        return None
    base_model_type, filename = parsed
    get_lora_dir = _get_main_callable("get_lora_dir")
    if not callable(get_lora_dir):
        return None
    try:
        lora_dir = Path(get_lora_dir(base_model_type))
    except Exception:
        return None
    candidate = (lora_dir / filename).resolve()
    if candidate.is_file() and candidate.suffix.lower() == ".json":
        return candidate
    return None


def _iter_builtin_settings_dirs() -> tuple[Path, ...]:
    if not SETTINGS_DIR.is_dir():
        return ()
    return tuple(sorted(path for path in SETTINGS_DIR.iterdir() if path.is_dir()))


def _build_preset_entry(path: Path, *, variant: str | None = None, label: str | None = None, source: str | None = None) -> dict[str, Any] | None:
    variant_value = str(variant or path.stem or "").strip()
    if len(variant_value) == 0:
        return None
    return {
        "variant": variant_value,
        "label": str(label or variant_value).strip() or variant_value,
        "path": path,
        "source": str(source or _PRESET_SOURCE_BUILTIN).strip() or _PRESET_SOURCE_BUILTIN,
    }


def _build_linked_variant_entry(variant: Any) -> dict[str, Any] | None:
    normalized = _normalize_linked_variant(variant)
    if normalized is None:
        return None
    preset_path = _resolve_linked_variant_path(normalized)
    if preset_path is None:
        return None
    _, filename = normalized.split("/", 1)
    label = Path(filename).stem
    return _build_preset_entry(preset_path, variant=normalized, label=label, source=_PRESET_SOURCE_LINKED)


def _add_or_replace_preset_entry(tool_entries: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    variant = str(entry.get("variant", "")).strip()
    if len(variant) == 0:
        return
    for index, existing in enumerate(tool_entries):
        if str(existing.get("variant", "")).strip() != variant:
            continue
        current_priority = _PRESET_SOURCE_PRIORITY.get(str(existing.get("source", _PRESET_SOURCE_BUILTIN)).strip(), 0)
        next_priority = _PRESET_SOURCE_PRIORITY.get(str(entry.get("source", _PRESET_SOURCE_BUILTIN)).strip(), 0)
        if next_priority >= current_priority:
            tool_entries[index] = entry
        return
    tool_entries.append(entry)


@lru_cache(maxsize=1)
def _preset_index() -> dict[str, tuple[dict[str, Any], ...]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for tool_dir in _iter_builtin_settings_dirs():
        tool_name = str(tool_dir.name or "").strip()
        if len(tool_name) == 0:
            continue
        tool_entries = index.setdefault(tool_name, [])
        for path in sorted(tool_dir.glob("*.json")):
            entry = _build_preset_entry(path)
            if entry is not None:
                _add_or_replace_preset_entry(tool_entries, entry)
    return {tool_name: tuple(entries) for tool_name, entries in index.items()}


def _tool_entries(tool_name: str, current_variant: Any = None) -> list[dict[str, Any]]:
    entries = list(_preset_index().get(str(tool_name or "").strip(), ()))
    seen = {str(entry.get("variant", "")).strip() for entry in entries}
    for candidate in (_get_configured_tool_variant(tool_name), current_variant):
        linked_entry = _build_linked_variant_entry(candidate)
        if linked_entry is not None:
            variant = str(linked_entry.get("variant", "")).strip()
            if variant not in seen:
                entries.append(linked_entry)
                seen.add(variant)
    return entries


def list_tool_variants(tool_name: str, current_variant: Any = None) -> list[str]:
    return [str(entry.get("variant", "")).strip() for entry in _tool_entries(tool_name, current_variant=current_variant) if len(str(entry.get("variant", "")).strip()) > 0]


def list_tool_variant_choices(tool_name: str, current_variant: Any = None) -> list[tuple[str, str]]:
    return [
        (str(entry.get("label", "")).strip() or str(entry.get("variant", "")).strip(), str(entry.get("variant", "")).strip())
        for entry in _tool_entries(tool_name, current_variant=current_variant)
        if len(str(entry.get("variant", "")).strip()) > 0
    ]


def find_tool_variant(tool_name: str, requested_variant: Any, current_variant: Any = None) -> str | None:
    tool_name = str(tool_name or "").strip()
    requested = _canonical_variant(tool_name, requested_variant)
    if len(requested) == 0:
        return None
    linked_variant = _normalize_linked_variant(requested)
    if linked_variant is not None:
        return linked_variant if _resolve_linked_variant_path(linked_variant) is not None else None
    variants = list_tool_variants(tool_name, current_variant=current_variant)
    if requested in variants:
        return requested
    requested_cf = requested.casefold()
    for variant in variants:
        if variant.casefold() == requested_cf:
            return variant
    return None


def get_tool_variant_path(tool_name: str, requested_variant: Any, current_variant: Any = None) -> Path | None:
    linked_path = _resolve_linked_variant_path(requested_variant)
    if linked_path is not None:
        return linked_path
    resolved_variant = find_tool_variant(tool_name, requested_variant, current_variant=current_variant)
    if resolved_variant is None:
        return None
    for entry in _tool_entries(tool_name, current_variant=current_variant):
        if str(entry.get("variant", "")).strip() == resolved_variant:
            return Path(entry["path"])
    return None


def resolve_wangp_settings_file(state: Any, selected_value: Any) -> Path | None:
    value = str(selected_value or "").strip()
    if len(value) == 0 or "/" in value or "\\" in value or not value.lower().endswith(".json"):
        return None
    get_state_model_type = _get_main_callable("get_state_model_type")
    get_lora_dir = _get_main_callable("get_lora_dir")
    if not callable(get_state_model_type) or not callable(get_lora_dir):
        return None
    try:
        model_type = get_state_model_type(state)
        lora_dir = Path(get_lora_dir(model_type))
    except Exception:
        return None
    source_path = (lora_dir / Path(value).name).resolve()
    if source_path.is_file() and source_path.suffix.lower() == ".json":
        return source_path
    return None


def _load_wangp_settings_payload(source_path: Path) -> dict[str, Any]:
    with Path(source_path).resolve().open("r", encoding="utf-8") as reader:
        payload = json.load(reader)
    if not isinstance(payload, dict):
        raise TypeError(f"WanGP settings file '{Path(source_path).name}' must contain a JSON object.")
    return payload


def _get_model_def_from_settings_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    model_type = str(payload.get("model_type", "")).strip()
    if len(model_type) == 0:
        return None
    get_model_def = _get_main_callable("get_model_def")
    if not callable(get_model_def):
        return None
    try:
        model_def = get_model_def(model_type)
    except Exception:
        return None
    return model_def if isinstance(model_def, dict) else None


def _int_setting(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key, None)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _sequence_setting_has_value(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key, None)
    if isinstance(value, list):
        return any(len(str(item).strip()) > 0 for item in value if item is not None)
    return len(str(value or "").strip()) > 0


def validate_wangp_settings_payload_for_tool(tool_name: str, payload: dict[str, Any]) -> str | None:
    lookup_name = str(tool_name or "").strip()
    if lookup_name not in _TOOL_TEMPLATE_VALIDATION_ERRORS:
        return None
    image_mode = _int_setting(payload, "image_mode")
    accepts_audio_prompt = "A" in str(payload.get("audio_prompt_type", "") or "")
    has_image_refs = _sequence_setting_has_value(payload, "image_refs")
    model_def = _get_model_def_from_settings_payload(payload)
    audio_only = bool(model_def.get("audio_only", False)) if isinstance(model_def, dict) else False
    checks = {
        "gen_video": image_mode == 0,
        "gen_video_with_speech": image_mode == 0 and accepts_audio_prompt,
        "gen_image": image_mode == 1,
        "edit_image": image_mode == 1 and has_image_refs,
        "gen_speech_from_description": audio_only,
        "gen_speech_from_sample": audio_only and accepts_audio_prompt,
    }
    return None if checks.get(lookup_name, True) else _TOOL_TEMPLATE_VALIDATION_ERRORS[lookup_name]


def validate_wangp_settings_for_tool(tool_name: str, source_path: Path) -> str | None:
    return validate_wangp_settings_payload_for_tool(tool_name, _load_wangp_settings_payload(source_path))


def build_linked_tool_variant(state: Any, source_path: Path) -> str:
    source_file = Path(source_path).resolve()
    if not source_file.is_file() or source_file.suffix.lower() != ".json":
        raise FileNotFoundError(f"Deepy source settings file not found: {source_file}")
    payload = _load_wangp_settings_payload(source_file)
    model_type = str(payload.get("model_type", "")).strip()
    if len(model_type) == 0:
        get_state_model_type = _get_main_callable("get_state_model_type")
        if callable(get_state_model_type):
            try:
                model_type = str(get_state_model_type(state) or "").strip()
            except Exception:
                model_type = ""
    base_model_type = _get_base_model_type_name(model_type)
    if len(base_model_type) == 0:
        raise ValueError(f"Unable to resolve base model type for {source_file.name}.")
    return f"{base_model_type}/{source_file.name}"


def is_linked_tool_variant(requested_variant: Any) -> bool:
    return _parse_linked_variant(requested_variant) is not None


def resolve_tool_variant(tool_name: str, requested_variant: Any, default_variant: str | None = None) -> str:
    tool_name = str(tool_name or "").strip()
    static_variants = [str(entry.get("variant", "")).strip() for entry in _preset_index().get(tool_name, ()) if len(str(entry.get("variant", "")).strip()) > 0]
    if len(static_variants) == 0:
        raise FileNotFoundError(f"No Deepy presets found for tool '{tool_name}' in {SETTINGS_DIR}.")
    requested = _canonical_variant(tool_name, requested_variant)
    if len(requested) > 0:
        linked_variant = _normalize_linked_variant(requested)
        if linked_variant is not None:
            if _resolve_linked_variant_path(linked_variant) is not None:
                return linked_variant
        else:
            resolved_variant = find_tool_variant(tool_name, requested)
            if resolved_variant is not None:
                return resolved_variant
    fallback = _canonical_variant(tool_name, default_variant)
    if len(fallback) > 0:
        linked_variant = _normalize_linked_variant(fallback)
        if linked_variant is not None:
            if _resolve_linked_variant_path(linked_variant) is not None:
                return linked_variant
        else:
            resolved_variant = find_tool_variant(tool_name, fallback)
            if resolved_variant is not None:
                return resolved_variant
    return static_variants[0]


def get_default_image_generator_variant() -> str:
    configured = _get_configured_tool_variant("gen_image")
    return resolve_tool_variant("gen_image", configured, default_variant=DEEPY_DEFAULT_GEN_IMAGE)


def get_default_video_generator_variant() -> str:
    configured = _get_configured_tool_variant("gen_video")
    return resolve_tool_variant("gen_video", configured, default_variant=DEEPY_DEFAULT_GEN_VIDEO)


def get_default_image_editor_variant() -> str:
    configured = _get_configured_tool_variant("edit_image")
    return resolve_tool_variant("edit_image", configured, default_variant=DEEPY_DEFAULT_EDIT_IMAGE)


def get_default_video_with_speech_variant() -> str:
    configured = _get_configured_tool_variant("gen_video_with_speech")
    return resolve_tool_variant("gen_video_with_speech", configured, default_variant=DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH)


def get_default_speech_from_description_variant() -> str:
    configured = _get_configured_tool_variant("gen_speech_from_description")
    return resolve_tool_variant("gen_speech_from_description", configured, default_variant=DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION)


def get_default_speech_from_sample_variant() -> str:
    configured = _get_configured_tool_variant("gen_speech_from_sample")
    return resolve_tool_variant("gen_speech_from_sample", configured, default_variant=DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE)


def _format_ineligible_tool_settings_error(tool_name: str, error_text: str) -> str:
    return f"settings no eligible for tool {str(tool_name or '').strip()}: {str(error_text or '').strip()}"


@lru_cache(maxsize=None)
def _load_static_tool_preset(tool_name: str, variant: str) -> dict[str, Any]:
    preset_path = None
    for entry in _preset_index().get(str(tool_name or "").strip(), ()):
        if str(entry.get("variant", "")).strip() == str(variant or "").strip():
            preset_path = Path(entry["path"])
            break
    if preset_path is None or not preset_path.is_file():
        raise FileNotFoundError(f"Deepy preset file not found for tool '{tool_name}' variant '{variant}'.")
    with preset_path.open("r", encoding="utf-8") as reader:
        payload = json.load(reader)
    if not isinstance(payload, dict):
        raise TypeError(f"Deepy preset '{preset_path.name}' must contain a JSON object.")
    return payload


def _load_live_tool_preset(tool_name: str, variant: str, preset_path: Path) -> dict[str, Any]:
    resolved_path = Path(preset_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Deepy preset file not found for tool '{tool_name}' variant '{variant}'.")
    cache_key = (str(tool_name or "").strip(), str(variant or "").strip())
    stat = resolved_path.stat()
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    cached = _LIVE_FILE_PRESET_CACHE.get(cache_key)
    if isinstance(cached, dict) and cached.get("path") == str(resolved_path) and int(cached.get("mtime_ns", -1)) == mtime_ns:
        cached_error = str(cached.get("eligibility_error", "") or "").strip()
        if len(cached_error) > 0:
            raise ValueError(_format_ineligible_tool_settings_error(tool_name, cached_error))
        cached_payload = cached.get("payload", None)
        if isinstance(cached_payload, dict):
            return cached_payload
    payload = _load_wangp_settings_payload(resolved_path)
    eligibility_error = str(validate_wangp_settings_payload_for_tool(tool_name, payload) or "").strip()
    _LIVE_FILE_PRESET_CACHE[cache_key] = {
        "path": str(resolved_path),
        "mtime_ns": mtime_ns,
        "payload": copy.deepcopy(payload) if len(eligibility_error) == 0 else None,
        "eligibility_error": eligibility_error,
    }
    if len(eligibility_error) > 0:
        raise ValueError(_format_ineligible_tool_settings_error(tool_name, eligibility_error))
    return payload


def load_tool_preset(tool_name: str, variant: str) -> dict[str, Any]:
    lookup_name = str(tool_name or "").strip()
    resolved_variant = resolve_tool_variant(lookup_name, variant)
    linked_path = _resolve_linked_variant_path(resolved_variant)
    if linked_path is not None:
        return _load_live_tool_preset(lookup_name, resolved_variant, linked_path)
    return _load_static_tool_preset(lookup_name, resolved_variant)


def clone_tool_preset(tool_name: str, variant: str) -> dict[str, Any]:
    return copy.deepcopy(load_tool_preset(tool_name, variant))


def refresh_tool_presets() -> None:
    _preset_index.cache_clear()
    _load_static_tool_preset.cache_clear()
    _LIVE_FILE_PRESET_CACHE.clear()


def build_generation_task(
    tool_name: str,
    variant: str,
    *,
    prompt: str,
    client_id: str,
    alt_prompt: str | None = None,
    audio_guide: str | None = None,
    image_start_target: str = "image_start",
    image_start: str | None = None,
    image_end: str | None = None,
    image_refs: list[str] | None = None,
) -> dict[str, Any]:
    task = clone_tool_preset(tool_name, variant)
    task["prompt"] = str(prompt or "").strip()
    task["client_id"] = str(client_id or "").strip()
    if alt_prompt is not None:
        alt_prompt = str(alt_prompt).strip()
        if len(alt_prompt) > 0:
            task["alt_prompt"] = alt_prompt
    if audio_guide is not None:
        audio_guide = str(audio_guide).strip()
        if len(audio_guide) > 0:
            task["audio_guide"] = audio_guide
    if image_start is not None:
        image_start = str(image_start).strip()
        if len(image_start) > 0:
            if str(image_start_target or "image_start").strip() == "image_refs":
                existing_image_refs = task.get("image_refs", None)
                image_refs_list = [] if not isinstance(existing_image_refs, list) else [str(path).strip() for path in existing_image_refs if len(str(path).strip()) > 0]
                image_refs_list.insert(0, image_start)
                task["image_refs"] = image_refs_list
                task.pop("image_start", None)
            else:
                task["image_start"] = image_start
    if image_end is not None:
        image_end = str(image_end).strip()
        if len(image_end) > 0:
            task["image_end"] = image_end
    if image_refs is not None:
        image_refs_list = [str(path).strip() for path in image_refs if len(str(path).strip()) > 0]
        existing_image_refs = task.get("image_refs", None)
        if isinstance(existing_image_refs, list) and len(existing_image_refs) > 0:
            merged_image_refs = [str(path).strip() for path in existing_image_refs if len(str(path).strip()) > 0]
            merged_image_refs.extend(path for path in image_refs_list if path not in merged_image_refs)
            task["image_refs"] = merged_image_refs
        else:
            task["image_refs"] = image_refs_list
    return task


__all__ = [
    "DEFAULT_IMAGE_EDITOR_VARIANT",
    "TOOL_DISPLAY_NAMES",
    "SETTINGS_DIR",
    "build_generation_task",
    "build_linked_tool_variant",
    "clone_tool_preset",
    "find_tool_variant",
    "get_default_image_editor_variant",
    "get_default_image_generator_variant",
    "get_default_speech_from_description_variant",
    "get_default_speech_from_sample_variant",
    "get_default_video_generator_variant",
    "get_default_video_with_speech_variant",
    "get_tool_variant_path",
    "is_linked_tool_variant",
    "list_tool_variant_choices",
    "list_tool_variants",
    "load_tool_preset",
    "refresh_tool_presets",
    "resolve_tool_variant",
    "resolve_wangp_settings_file",
    "validate_wangp_settings_for_tool",
    "validate_wangp_settings_payload_for_tool",
]
