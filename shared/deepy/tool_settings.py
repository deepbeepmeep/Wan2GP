from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from shared.deepy.config import (
    DEEPY_DEFAULT_IMAGE_EDITOR,
    DEEPY_DEFAULT_IMAGE_EDITOR_KEY,
    DEEPY_DEFAULT_IMAGE_GENERATOR,
    DEEPY_DEFAULT_IMAGE_GENERATOR_KEY,
    DEEPY_DEFAULT_VIDEO_GENERATOR,
    DEEPY_DEFAULT_VIDEO_GENERATOR_KEY,
    get_deepy_config_value,
    normalize_deepy_default_image_editor,
    normalize_deepy_default_image_generator,
    normalize_deepy_default_video_generator,
)


_DEEPY_DIR = Path(__file__).resolve().parent
SETTINGS_DIR = _DEEPY_DIR / "settings"
CUSTOM_SETTINGS_DIR = _DEEPY_DIR / "custom_settings"
SHARED_CUSTOM_SETTINGS_DIR = _DEEPY_DIR.parent / "custom_settings"
DEFAULT_IMAGE_EDITOR_VARIANT = DEEPY_DEFAULT_IMAGE_EDITOR
DEFAULT_VIDEO_WITH_SPEECH_VARIANT = "Infinitalk"
DEFAULT_SPEECH_FROM_DESCRIPTION_VARIANT = "Qwen3 1.7B"
DEFAULT_SPEECH_FROM_SAMPLE_VARIANT = "Index TTS 2"
_LEGACY_VARIANT_ALIASES = {
    "edit_image": {"Qwen_Edit": DEEPY_DEFAULT_IMAGE_EDITOR},
    "gen_image": {"Z_Image_Turbo": DEEPY_DEFAULT_IMAGE_GENERATOR},
    "gen_video": {"ltx2_22B_distilled": DEEPY_DEFAULT_VIDEO_GENERATOR},
}


def _canonical_variant(tool_name: str, variant: Any) -> str:
    variant = str(variant or "").strip()
    if len(variant) == 0:
        return ""
    return _LEGACY_VARIANT_ALIASES.get(str(tool_name or "").strip(), {}).get(variant, variant)


def _looks_like_preset_path(value: Any) -> bool:
    text = str(value or "").strip().strip('"')
    if len(text) == 0:
        return False
    return text.endswith(".json") or any(sep in text for sep in (os.sep, "/", "\\"))


def _resolve_direct_preset_path(value: Any) -> Path | None:
    text = str(value or "").strip().strip('"')
    if len(text) == 0:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.is_file() and candidate.suffix.lower() == ".json":
        return candidate
    return None


def _iter_builtin_settings_dirs() -> tuple[Path, ...]:
    if not SETTINGS_DIR.is_dir():
        return ()
    return tuple(sorted(path for path in SETTINGS_DIR.iterdir() if path.is_dir()))


def _iter_custom_settings_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for root in (CUSTOM_SETTINGS_DIR, SHARED_CUSTOM_SETTINGS_DIR):
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


@lru_cache(maxsize=1)
def _preset_index() -> dict[str, tuple[dict[str, Any], ...]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for tool_dir in _iter_builtin_settings_dirs():
        tool_name = str(tool_dir.name or "").strip()
        if len(tool_name) == 0:
            continue
        tool_entries = index.setdefault(tool_name, [])
        seen_variants = {str(entry.get("variant", "")).strip() for entry in tool_entries}
        for path in sorted(tool_dir.glob("*.json")):
            variant = str(path.stem or "").strip()
            if len(variant) == 0 or variant in seen_variants:
                continue
            tool_entries.append({"variant": variant, "label": variant, "path": path})
            seen_variants.add(variant)
    for root in _iter_custom_settings_roots():
        if not root.is_dir():
            continue
        for tool_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            tool_name = str(tool_dir.name or "").strip()
            if len(tool_name) == 0:
                continue
            tool_entries = index.setdefault(tool_name, [])
            seen_variants = {str(entry.get("variant", "")).strip() for entry in tool_entries}
            for path in sorted(tool_dir.glob("*.json")):
                variant = str(path.stem or "").strip()
                if len(variant) == 0 or variant in seen_variants:
                    continue
                tool_entries.append({"variant": variant, "label": variant, "path": path})
                seen_variants.add(variant)
    return {tool_name: tuple(entries) for tool_name, entries in index.items()}


def list_tool_variants(tool_name: str) -> list[str]:
    return [str(entry.get("variant", "")).strip() for entry in _preset_index().get(str(tool_name or "").strip(), ()) if len(str(entry.get("variant", "")).strip()) > 0]


def list_tool_variant_choices(tool_name: str) -> list[tuple[str, str]]:
    choices = []
    for entry in _preset_index().get(str(tool_name or "").strip(), ()):
        variant = str(entry.get("variant", "")).strip()
        if len(variant) == 0:
            continue
        choices.append((str(entry.get("label", "")).strip() or variant, variant))
    return choices


def find_tool_variant(tool_name: str, requested_variant: Any) -> str | None:
    tool_name = str(tool_name or "").strip()
    direct_path = _resolve_direct_preset_path(requested_variant)
    if direct_path is not None:
        return str(direct_path)
    requested = _canonical_variant(tool_name, requested_variant)
    if len(requested) == 0:
        return None
    variants = list_tool_variants(tool_name)
    if requested in variants:
        return requested
    requested_cf = requested.casefold()
    for variant in variants:
        if variant.casefold() == requested_cf:
            return variant
    return None


def resolve_tool_variant(tool_name: str, requested_variant: Any, default_variant: str | None = None) -> str:
    tool_name = str(tool_name or "").strip()
    variants = list_tool_variants(tool_name)
    if len(variants) == 0:
        raise FileNotFoundError(f"No Deepy presets found for tool '{tool_name}' in {SETTINGS_DIR}.")
    if _looks_like_preset_path(requested_variant):
        direct_path = _resolve_direct_preset_path(requested_variant)
        if direct_path is None:
            raise FileNotFoundError(f"Deepy preset file not found: {str(requested_variant or '').strip()}")
        return str(direct_path)
    requested = find_tool_variant(tool_name, requested_variant)
    if requested is not None:
        return requested
    fallback = find_tool_variant(tool_name, default_variant)
    if fallback is not None:
        return fallback
    return variants[0]


def get_default_image_generator_variant() -> str:
    configured = normalize_deepy_default_image_generator(get_deepy_config_value(DEEPY_DEFAULT_IMAGE_GENERATOR_KEY, DEEPY_DEFAULT_IMAGE_GENERATOR))
    return resolve_tool_variant("gen_image", configured, default_variant=DEEPY_DEFAULT_IMAGE_GENERATOR)


def get_default_video_generator_variant() -> str:
    configured = normalize_deepy_default_video_generator(get_deepy_config_value(DEEPY_DEFAULT_VIDEO_GENERATOR_KEY, DEEPY_DEFAULT_VIDEO_GENERATOR))
    return resolve_tool_variant("gen_video", configured, default_variant=DEEPY_DEFAULT_VIDEO_GENERATOR)


def get_default_image_editor_variant() -> str:
    configured = normalize_deepy_default_image_editor(get_deepy_config_value(DEEPY_DEFAULT_IMAGE_EDITOR_KEY, DEEPY_DEFAULT_IMAGE_EDITOR))
    return resolve_tool_variant("edit_image", configured, default_variant=DEEPY_DEFAULT_IMAGE_EDITOR)


def get_default_video_with_speech_variant() -> str:
    return resolve_tool_variant("gen_video_with_speech", "", default_variant=DEFAULT_VIDEO_WITH_SPEECH_VARIANT)


def get_default_speech_from_description_variant() -> str:
    return resolve_tool_variant("gen_speech_from_description", "", default_variant=DEFAULT_SPEECH_FROM_DESCRIPTION_VARIANT)


def get_default_speech_from_sample_variant() -> str:
    return resolve_tool_variant("gen_speech_from_sample", "", default_variant=DEFAULT_SPEECH_FROM_SAMPLE_VARIANT)


@lru_cache(maxsize=None)
def load_tool_preset(tool_name: str, variant: str) -> dict[str, Any]:
    tool_name = str(tool_name or "").strip()
    variant = resolve_tool_variant(tool_name, variant)
    preset_path = _resolve_direct_preset_path(variant)
    if preset_path is None:
        for entry in _preset_index().get(tool_name, ()):
            if str(entry.get("variant", "")).strip() == variant:
                preset_path = Path(entry["path"])
                break
    if preset_path is None or not preset_path.is_file():
        raise FileNotFoundError(f"Deepy preset file not found for tool '{tool_name}' variant '{variant}'.")
    with preset_path.open("r", encoding="utf-8") as reader:
        payload = json.load(reader)
    if not isinstance(payload, dict):
        raise TypeError(f"Deepy preset '{preset_path.name}' must contain a JSON object.")
    return payload


def clone_tool_preset(tool_name: str, variant: str) -> dict[str, Any]:
    return copy.deepcopy(load_tool_preset(tool_name, variant))


def refresh_tool_presets() -> None:
    _preset_index.cache_clear()
    load_tool_preset.cache_clear()


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
    "SETTINGS_DIR",
    "build_generation_task",
    "clone_tool_preset",
    "find_tool_variant",
    "get_default_image_editor_variant",
    "get_default_image_generator_variant",
    "get_default_video_generator_variant",
    "list_tool_variant_choices",
    "list_tool_variants",
    "load_tool_preset",
    "refresh_tool_presets",
    "resolve_tool_variant",
]
