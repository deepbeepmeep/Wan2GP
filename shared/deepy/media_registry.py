from __future__ import annotations

import os
import re
import time
from typing import Any

from shared.utils.audio_video import read_image_metadata


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
_VIDEO_EXTENSIONS = {".mp4", ".mkv"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac"}
_MEDIA_TYPES = {"image", "video", "audio", "any", "all"}
_TYPE_HINTS = {
    "image": ("image", "images", "picture", "photo", "photos", "pic", "pics"),
    "video": ("video", "videos", "clip", "movie", "footage"),
    "audio": ("audio", "sound", "song", "music", "track", "voice"),
}
_ALIAS_PREVIOUS_RE = re.compile(r"\b(previous|prior|before\s+last|second\s+last|penultimate)\b", flags=re.IGNORECASE)
_ALIAS_LAST_RE = re.compile(r"\b(last|latest|most\s+recent)\b", flags=re.IGNORECASE)
_REFERENCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "audio",
    "clip",
    "file",
    "generated",
    "image",
    "latest",
    "last",
    "media",
    "most",
    "my",
    "of",
    "on",
    "photo",
    "picture",
    "please",
    "prior",
    "recent",
    "show",
    "sound",
    "that",
    "the",
    "this",
    "track",
    "use",
    "video",
}
_SEARCH_LIMIT = 5


def normalize_media_type(media_type: str | None, reference: str | None = None) -> str:
    normalized = str(media_type or "").strip().lower()
    if normalized not in _MEDIA_TYPES:
        normalized = "any"
    elif normalized == "all":
        normalized = "any"
    if normalized != "any":
        return normalized
    reference_text = _normalize_text(reference)
    for candidate_type, hints in _TYPE_HINTS.items():
        if any(hint in reference_text for hint in hints):
            return candidate_type
    return "any"


def get_media_record(session, media_id: str) -> dict[str, Any] | None:
    lookup_id = str(media_id or "").strip()
    if len(lookup_id) == 0:
        return None
    normalized_lookup_id = _normalize_media_id_lookup(lookup_id)
    for record in session.media_registry:
        record_id = str(record.get("media_id", "")).strip()
        if record_id == lookup_id or record_id == normalized_lookup_id:
            return record
    return None


def register_media(
    session,
    path: str,
    settings: dict[str, Any] | None = None,
    *,
    source: str = "wangp",
    client_id: str = "",
    label: str | None = None,
    media_type: str | None = None,
) -> dict[str, Any] | None:
    path = str(path or "").strip()
    if len(path) == 0:
        return None
    detected_type = normalize_media_type(media_type or _detect_media_type(path))
    if detected_type == "any":
        return None
    resolved_settings = _resolve_settings(path, settings)
    prompt = str((resolved_settings or {}).get("prompt", "") or "").strip()
    prompt_summary = _summarize_prompt(prompt, detected_type)
    path_key = _normalize_path_key(path)
    existing = None
    for record in session.media_registry:
        if record.get("path_key") == path_key:
            existing = record
            break
    if existing is None:
        session.media_registry_counter += 1
        existing = {
            "media_id": f"{detected_type}_{session.media_registry_counter}",
            "path_key": path_key,
        }
        session.media_registry.insert(0, existing)
    else:
        session.media_registry.remove(existing)
        session.media_registry.insert(0, existing)
    existing.update(
        {
            "media_type": detected_type,
            "path": path,
            "source": str(source or "wangp").strip() or "wangp",
            "client_id": str(client_id or "").strip(),
            "settings": dict(resolved_settings or {}),
            "label": str(label or prompt_summary or _default_label(path, detected_type)).strip() or _default_label(path, detected_type),
            "prompt_summary": prompt_summary,
            "prompt": prompt,
            "filename": os.path.basename(path),
            "updated_at": float(time.time()),
        }
    )
    return existing


def collapse_gallery_media(file_list: list[Any], file_settings_list: list[Any]) -> tuple[list[Any], list[Any]]:
    gallery_files = list(file_list or [])
    gallery_settings = list(file_settings_list or [])
    collapsed_pairs: list[tuple[Any, Any]] = []
    seen_client_keys: set[tuple[str, str]] = set()
    for index in range(len(gallery_files) - 1, -1, -1):
        path = gallery_files[index]
        settings = gallery_settings[index] if index < len(gallery_settings) else None
        client_key = _gallery_client_media_key(path, settings)
        if client_key is not None:
            if client_key in seen_client_keys:
                continue
            seen_client_keys.add(client_key)
        collapsed_pairs.append((path, settings))
    collapsed_pairs.reverse()
    collapsed_files = []
    collapsed_settings = []
    for path, settings in collapsed_pairs:
        if path is None and settings is None:
            continue
        collapsed_files.append(path)
        collapsed_settings.append(settings)
    return collapsed_files, collapsed_settings


def find_last_gallery_media_by_client(file_list: list[Any], file_settings_list: list[Any], client_id: str, *, media_type: str | None = None) -> tuple[str | None, dict[str, Any] | None]:
    lookup_client_id = str(client_id or "").strip()
    if len(lookup_client_id) == 0:
        return None, None
    gallery_files = list(file_list or [])
    gallery_settings = list(file_settings_list or [])
    expected_media_type = normalize_media_type(media_type)
    for index in range(len(gallery_files) - 1, -1, -1):
        settings = gallery_settings[index] if index < len(gallery_settings) else None
        if not isinstance(settings, dict):
            continue
        if str(settings.get("client_id", "") or "").strip() != lookup_client_id:
            continue
        path = str(gallery_files[index] or "").strip()
        if len(path) == 0:
            continue
        detected_media_type = _detect_media_type(path)
        if expected_media_type != "any" and detected_media_type != expected_media_type:
            continue
        return path, settings
    return None, None


def sync_recent_generated_media(session, file_list: list[Any], file_settings_list: list[Any], max_items: int = _SEARCH_LIMIT) -> list[dict[str, Any]]:
    collapsed_file_list, collapsed_settings_list = collapse_gallery_media(file_list, file_settings_list)
    recent_items = list(zip(collapsed_file_list, collapsed_settings_list))
    if max_items > 0:
        recent_items = recent_items[-max_items:]
    synced = []
    for path, settings in recent_items:
        record = register_media(
            session,
            str(path or ""),
            settings=settings if isinstance(settings, dict) else None,
            source=_resolve_source(settings),
            client_id=str((settings or {}).get("client_id", "") or "").strip(),
            label=_label_from_settings(settings, str(path or "")),
        )
        if record is not None:
            synced.append(record)
    return synced


def resolve_media_reference(session, reference: str, media_type: str = "any", limit: int = _SEARCH_LIMIT) -> dict[str, Any]:
    resolved_type = normalize_media_type(media_type, reference=reference)
    filtered = _filter_records(session.media_registry, resolved_type)
    reference_text = str(reference or "").strip()
    if len(filtered) == 0:
        return {"status": "not_found", "media_type": resolved_type, "reference": reference_text, "matches": []}
    alias_record = _resolve_alias(filtered, reference_text)
    if alias_record is not None:
        return {"status": "resolved", "media_type": resolved_type, "reference": reference_text, "media": _compact_media(alias_record, why="matched recent alias")}
    direct_record = get_media_record(session, reference_text)
    if direct_record is not None and (resolved_type == "any" or direct_record.get("media_type") == resolved_type):
        return {"status": "resolved", "media_type": resolved_type, "reference": reference_text, "media": _compact_media(direct_record, why="matched media id")}
    ranked = _rank_records(filtered, reference_text)
    if len(ranked) == 0:
        return {
            "status": "not_found",
            "media_type": resolved_type,
            "reference": reference_text,
            "matches": [_compact_media(record, why="recent") for record in filtered[: max(1, limit)]],
        }
    if len(ranked) == 1:
        return {"status": "resolved", "media_type": resolved_type, "reference": reference_text, "media": _compact_media(ranked[0][0], why=ranked[0][1])}
    return {
        "status": "candidates",
        "media_type": resolved_type,
        "reference": reference_text,
        "matches": [_compact_media(record, why=why) for record, why in ranked[: max(2, limit)]],
    }


def _resolve_settings(path: str, settings: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(settings, dict) and len(settings) > 0:
        return settings
    if _detect_media_type(path) == "image" and os.path.isfile(path):
        metadata = read_image_metadata(path)
        if isinstance(metadata, dict):
            return metadata
    return {}


def _resolve_alias(records: list[dict[str, Any]], reference: str) -> dict[str, Any] | None:
    normalized = _normalize_text(reference)
    if len(normalized) == 0:
        return records[0]
    if _ALIAS_PREVIOUS_RE.search(normalized):
        return records[1] if len(records) > 1 else None
    if _ALIAS_LAST_RE.search(normalized):
        return records[0]
    return None


def _rank_records(records: list[dict[str, Any]], reference: str) -> list[tuple[dict[str, Any], str]]:
    normalized_reference = _normalize_text(reference)
    reference_tokens = [token for token in _tokenize(reference) if token not in _REFERENCE_STOPWORDS]
    ranked = []
    for index, record in enumerate(records):
        search_blob = _normalize_text(" ".join(filter(None, [record.get("label", ""), record.get("prompt_summary", ""), record.get("prompt", ""), record.get("filename", "")])))
        if len(search_blob) == 0:
            continue
        score = 0
        why = ""
        if len(normalized_reference) > 0 and normalized_reference in search_blob:
            score += 100
            why = "exact text match"
        search_tokens = set(_tokenize(search_blob))
        token_hits = [token for token in reference_tokens if token in search_tokens]
        if token_hits:
            score += 15 * len(token_hits)
            why = f"matched {', '.join(token_hits[:3])}"
        score += max(0, 10 - index)
        if score <= 0:
            continue
        ranked.append((score, record, why or "recent semantic match"))
    ranked.sort(key=lambda item: (-item[0], item[1].get("updated_at", 0.0)), reverse=False)
    return [(record, why) for _score, record, why in ranked]


def _compact_media(record: dict[str, Any], why: str = "") -> dict[str, Any]:
    payload = {
        "media_id": record.get("media_id", ""),
        "media_type": record.get("media_type", ""),
        "label": record.get("label", ""),
        "source": record.get("source", ""),
        "filename": record.get("filename", ""),
    }
    prompt_summary = str(record.get("prompt_summary", "") or "").strip()
    if len(prompt_summary) > 0:
        payload["prompt_summary"] = prompt_summary
    if len(str(why or "").strip()) > 0:
        payload["why"] = str(why).strip()
    return payload


def _filter_records(records: list[dict[str, Any]], media_type: str) -> list[dict[str, Any]]:
    if media_type == "any":
        return list(records)
    return [record for record in records if record.get("media_type") == media_type]


def _normalize_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path or "").strip()))


def _normalize_media_id_lookup(media_id: str) -> str:
    normalized = re.sub(r"\s+", "_", str(media_id or "").strip().lower())
    return normalized


def _normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").strip().lower())).strip()


def _tokenize(text: str | None) -> list[str]:
    return [token for token in _normalize_text(text).split(" ") if len(token) > 0]


def _detect_media_type(path: str) -> str:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    if ext in _AUDIO_EXTENSIONS:
        return "audio"
    return "any"


def _default_label(path: str, media_type: str) -> str:
    base_name = os.path.splitext(os.path.basename(str(path or "").strip()))[0].replace("_", " ").strip()
    return base_name or f"Generated {media_type}"


def _summarize_prompt(prompt: str, media_type: str) -> str:
    prompt = str(prompt or "").strip()
    if len(prompt) == 0:
        return f"Generated {media_type}"
    first_sentence = re.split(r"[\n.;]", prompt, maxsplit=1)[0].strip()
    first_clause = re.split(r"\s*,\s*", first_sentence, maxsplit=1)[0].strip()
    summary = first_clause or first_sentence or prompt
    words = summary.split()
    if len(words) > 12:
        summary = " ".join(words[:12])
    return summary.strip() or f"Generated {media_type}"


def _resolve_source(settings: dict[str, Any] | None) -> str:
    client_id = str((settings or {}).get("client_id", "") or "").strip()
    if client_id.startswith("ai"):
        return "deepy"
    return "wangp"


def _label_from_settings(settings: dict[str, Any] | None, path: str) -> str | None:
    if not isinstance(settings, dict):
        return None
    prompt = str(settings.get("prompt", "") or "").strip()
    media_type = _detect_media_type(path)
    if len(prompt) == 0:
        return None
    return _summarize_prompt(prompt, media_type)


def _gallery_client_media_key(path: Any, settings: Any) -> tuple[str, str] | None:
    if not isinstance(settings, dict):
        return None
    client_id = str(settings.get("client_id", "") or "").strip()
    if len(client_id) == 0:
        return None
    media_type = _detect_media_type(str(path or "").strip())
    if media_type not in {"video", "audio"}:
        return None
    return media_type, client_id


__all__ = [
    "collapse_gallery_media",
    "find_last_gallery_media_by_client",
    "get_media_record",
    "normalize_media_type",
    "register_media",
    "resolve_media_reference",
    "sync_recent_generated_media",
]
