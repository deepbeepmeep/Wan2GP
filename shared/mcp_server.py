"""MCP server adapter for WanGP's in-process API."""

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import io
import mimetypes
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.api import SessionJob


_MAX_STORED_EVENTS = 500
_MEDIA_TRANSFER_TTL_SECONDS = 600
_MAX_UPLOAD_BYTES = 8 * 1024**3
_TRANSPORT_ALIASES = {
    "stdio": "stdio",
    "sse": "sse",
    "streamable-http": "streamable-http",
    "streamable_http": "streamable-http",
}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_AGENT_GUIDE_PATH = Path(__file__).resolve().parents[1] / "wangp-agent" / "SKILL.md"
_DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
_DEEPY_VISUAL_TOOL_IDS = {"gen_image", "edit_image", "gen_video", "gen_video_with_speech"}
_DEEPY_VIDEO_TOOL_IDS = {"gen_video", "gen_video_with_speech"}
_DEEPY_AUDIO_TOOL_IDS = {"gen_song", "gen_speech_from_description", "gen_speech_from_sample"}
_TOOLBOX_ACTIONS = {
    "create_color_frame",
    "inspect_media",
    "inspect_video",
    "extract_image",
    "extract_video",
    "extract_audio",
    "transcribe_media",
    "mute_video",
    "replace_audio",
    "resize_crop",
    "merge_videos",
    "search_doc",
    "load_doc_section",
    "get_media_details",
}
_TOOLBOX_MEDIA_PARAMETERS = {
    "inspect_media": ("media_id", "media_ids"),
    "inspect_video": ("media_id",),
    "extract_image": ("media_id",),
    "extract_video": ("media_id",),
    "extract_audio": ("media_id",),
    "transcribe_media": ("media_id",),
    "mute_video": ("media_id",),
    "replace_audio": ("video_id", "audio_id"),
    "resize_crop": ("media_id",),
    "merge_videos": ("video_first", "video_second"),
    "get_media_details": ("media_id",),
}
_POSTPROCESS_PATH_PARAMETERS = {
    "audio_media_id": "audio_path",
    "voice_sample_media_id": "voice_sample_path",
    "voice_sample2_media_id": "voice_sample2_path",
}
_MCP_MEDIA_SETTING_KEYS = {
    "image_start", "image_end", "image_refs", "image_guide", "image_mask",
    "video_guide", "video_guide2", "video_mask", "video_source",
    "audio_guide", "audio_guide2", "audio_source",
    "replace_voice_sample", "replace_voice_sample2", "custom_guide",
}
_GALLERY_LOCK = threading.RLock()


def _read_markdown_section(path: Path, start_heading: str, end_heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(start_heading)
    end = text.find(end_heading, start + len(start_heading))
    return text[start:] if end < 0 else text[start:end].rstrip()


def _register_documentation_resources(mcp) -> None:
    def document_reader(document_path: Path):
        def read_document() -> str:
            return document_path.read_text(encoding="utf-8")
        return read_document

    for path in sorted(_DOCS_DIR.glob("*.md")):
        resource_uri = f"wangp://docs/{path.stem.casefold()}"
        read_document = document_reader(path)
        read_document.__name__ = f"read_{path.stem.casefold()}_documentation"
        description = "WanGP generation settings: model selection, prompts, output dimensions, sampling and guidance, media inputs, acceleration and caching, post-processing, sliding windows, LoRAs, flags, and model API metadata." if path.stem.casefold() == "settings" else f"WanGP {path.stem} documentation."
        mcp.resource(resource_uri, name=path.stem.casefold(), title=path.stem.replace("_", " ").title(), description=description, mime_type="text/markdown")(read_document)

    @mcp.resource("wangp://docs/settings/prompt-flags", name="prompt_flags", title="WanGP Prompt-Type Flags", description="Exact image_prompt_type, video_prompt_type, and audio_prompt_type flag definitions.", mime_type="text/markdown")
    def read_prompt_flags() -> str:
        return _read_markdown_section(_DOCS_DIR / "SETTINGS.md", "### `image_prompt_type`", "### `prompt_enhancer`")


def _normalize_transport(value: str | None) -> str:
    transport = str(value or "stdio").strip().lower()
    try:
        return _TRANSPORT_ALIASES[transport]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported MCP transport: {value}. Use stdio, sse, or streamable-http.") from exc


@contextlib.contextmanager
def _stdio_safe_startup_output(transport: str):
    if transport != "stdio":
        yield
        return
    target = sys.stderr if sys.stderr is not None else io.StringIO()
    with contextlib.redirect_stdout(target):
        yield


def _artifact_to_dict(artifact: Any) -> dict[str, Any]:
    return {
        "path": artifact.path,
        "media_type": artifact.media_type,
        "client_id": artifact.client_id,
        "hdr": artifact.hdr,
        "audio_sampling_rate": artifact.audio_sampling_rate,
        "fps": artifact.fps,
        "has_video_tensor_uint8": artifact.video_tensor_uint8 is not None,
        "has_video_tensor_hdr": artifact.video_tensor_hdr is not None,
        "has_audio_tensor": artifact.audio_tensor is not None,
        "has_flashvsr_continue_cache": artifact.flashvsr_continue_cache is not None,
    }


def _error_to_dict(error: Any) -> dict[str, Any]:
    return {
        "message": error.message,
        "task_index": error.task_index,
        "task_id": error.task_id,
        "stage": error.stage,
        "cancelled": error.cancelled,
    }


def _result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "success": result.success,
        "cancelled": result.cancelled,
        "generated_files": list(result.generated_files),
        "errors": [_error_to_dict(error) for error in result.errors],
        "total_tasks": result.total_tasks,
        "successful_tasks": result.successful_tasks,
        "failed_tasks": result.failed_tasks,
        "artifacts": [_artifact_to_dict(artifact) for artifact in result.artifacts],
    }


def _event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "timestamp": event.timestamp,
        "data": _json_safe(event.data),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    class_name = type(value).__name__
    if class_name == "GeneratedArtifact":
        return _artifact_to_dict(value)
    if class_name == "GenerationError":
        return _error_to_dict(value)
    if class_name == "GenerationResult":
        return _result_to_dict(value)
    if class_name == "SessionEvent":
        return _event_to_dict(value)
    if class_name == "StreamMessage":
        return {"stream": value.stream, "text": value.text}
    if class_name == "PreviewUpdate":
        return {
            "has_image_preview": value.image is not None,
            "phase": value.phase,
            "status": value.status,
            "progress": value.progress,
            "current_step": value.current_step,
            "total_steps": value.total_steps,
        }
    if dataclasses.is_dataclass(value):
        return {field.name: _json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=str)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        return {"type": type(value).__name__, "shape": list(shape) if shape is not None else None, "dtype": str(dtype) if dtype is not None else None}
    return str(value)


def _gallery_media_type(path: str, gallery: str) -> str:
    if gallery == "audio":
        return "audio"
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _AUDIO_EXTENSIONS:
        return "audio"
    return "video"


def _gallery_history(session) -> dict[str, dict[str, Any]]:
    history = getattr(session, "_mcp_gallery_history", None)
    if history is None:
        history = {}
        session._mcp_gallery_history = history
    return history


def _gallery_path_exists(session, path: str) -> bool:
    candidate = Path(str(path or ""))
    if not candidate.is_absolute():
        candidate = Path(getattr(session, "_root", Path.cwd())) / candidate
    return candidate.is_file()


def _gallery_records(session, media_type: str = "all", limit: int = 50) -> list[dict[str, Any]]:
    requested_type = str(media_type or "all").strip().lower()
    if requested_type not in {"all", "image", "video", "audio"}:
        raise ValueError("media_type must be all, image, video, or audio")
    limit = max(1, min(int(limit), 500))
    gen = session._state["gen"]
    current_records = []
    gallery_defs = (
        ("visual", list(gen.get("file_list", []) or []), list(gen.get("file_settings_list", []) or []), int(gen.get("selected", -1))),
        ("audio", list(gen.get("audio_file_list", []) or []), list(gen.get("audio_file_settings_list", []) or []), int(gen.get("audio_selected", -1))),
    )
    for gallery, paths, settings_list, selected_index in gallery_defs:
        for index, path in enumerate(paths):
            resolved_path = str(path or "").strip()
            item_type = _gallery_media_type(resolved_path, gallery)
            settings = settings_list[index] if index < len(settings_list) and isinstance(settings_list[index], dict) else {}
            gallery_key = hashlib.sha1(resolved_path.replace("\\", "/").casefold().encode("utf-8")).hexdigest()[:12]
            record = {
                "media_id": f"{gallery}:{gallery_key}",
                "gallery": gallery,
                "index": index,
                "path": resolved_path,
                "media_type": item_type,
                "selected": index == selected_index,
                "in_gallery": True,
                "settings": _json_safe(settings),
            }
            if record["selected"] and gallery == "visual" and item_type == "video":
                record["current_time_seconds"] = gen.get("selected_video_time")
            current_records.append(record)
    with _GALLERY_LOCK:
        history = _gallery_history(session)
        for media_id, record in list(history.items()):
            if not _gallery_path_exists(session, record.get("path", "")):
                history.pop(media_id, None)
                continue
            record.update({"index": None, "selected": False, "in_gallery": False})
            record.pop("current_time_seconds", None)
        for record in current_records:
            history.pop(record["media_id"], None)
            history[record["media_id"]] = copy.deepcopy(record)
        records = [copy.deepcopy(record) for record in history.values() if requested_type == "all" or record["media_type"] == requested_type]
    return records[-limit:]


def _compact_gallery_stats(settings: dict[str, Any], media_type: str, path: str) -> dict[str, Any]:
    def positive_number(*keys: str) -> float | None:
        for key in keys:
            value = settings.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                return number
        return None

    if media_type == "video":
        from shared.utils.video_decode import probe_video_stream_metadata

        metadata = probe_video_stream_metadata(path)
        if metadata is None:
            return {}
        width, height = int(metadata.get("display_width", 0) or 0), int(metadata.get("display_height", 0) or 0)
        frame_count = int(metadata.get("frame_count", 0) or 0)
        fps = float(metadata.get("fps_float", 0) or 0)
        duration = float(metadata.get("duration", 0) or 0)
        if duration <= 0 and frame_count > 0 and fps > 0:
            duration = frame_count / fps
        stats = {}
        if width > 0 and height > 0:
            stats["resolution"] = f"{width}x{height}"
        if frame_count > 0:
            stats["frame_count"] = frame_count
        if fps > 0:
            stats["fps"] = int(fps) if fps.is_integer() else round(fps, 3)
        if duration > 0:
            stats["duration_seconds"] = int(duration) if duration.is_integer() else round(duration, 3)
        return stats

    stats = {}
    if media_type == "image":
        resolution = str(settings.get("resolution", "") or "").strip()
        if not resolution:
            width, height = positive_number("width"), positive_number("height")
            if width is not None and height is not None:
                resolution = f"{int(width)}x{int(height)}"
        if resolution:
            stats["resolution"] = resolution
    elif media_type == "audio":
        duration = positive_number("duration_seconds", "audio_duration")
        if duration is not None:
            stats["duration_seconds"] = int(duration) if duration.is_integer() else round(duration, 3)
    return stats


def _compact_gallery_record(record: dict[str, Any]) -> dict[str, Any]:
    from shared.deepy.media_registry import summarize_prompt

    media_id = str(record.get("media_id", "") or "").strip()
    media_type = str(record.get("media_type", "") or "").strip()
    settings = record.get("settings") if isinstance(record.get("settings"), dict) else {}
    prompt = str(settings.get("prompt", "") or "").strip()
    result = {
        "media_id": media_id,
        "gallery": record.get("gallery", ""),
        "media_type": media_type,
        "filename": Path(str(record.get("path", "") or "")).name,
        "description": summarize_prompt(prompt, media_type),
        "source": "Deepy" if str(settings.get("client_id", "") or "").strip().lower().startswith("ai") else "WanGP",
        "selected": bool(record.get("selected", False)),
        "in_gallery": bool(record.get("in_gallery", False)),
    }
    if record.get("current_time_seconds") is not None:
        result["current_time_seconds"] = record["current_time_seconds"]
    result.update(_compact_gallery_stats(settings, media_type, str(record.get("path", "") or "")))
    return result


def _gallery_item(session, media_id: str) -> dict[str, Any]:
    lookup = str(media_id or "").strip().lower()
    _gallery_records(session, limit=500)
    with _GALLERY_LOCK:
        record = _gallery_history(session).get(lookup)
        if record is not None and _gallery_path_exists(session, record.get("path", "")):
            return copy.deepcopy(record)
    raise KeyError(f"Unknown WanGP media_id: {media_id}")


def _extract_media_settings(session, path: str) -> dict[str, Any]:
    runtime = session._ensure_runtime()
    with contextlib.chdir(runtime.root):
        settings, _any_visual, _any_audio = runtime.module.get_settings_from_file(session._state, path, False, False, False)
    return settings if isinstance(settings, dict) else {}


def _media_settings(session, *, media_id: str | None = None, path: str | None = None, allow_read_file_system: bool = False) -> dict[str, Any]:
    media_id = str(media_id or "").strip()
    path = str(path or "").strip()
    if bool(media_id) == bool(path):
        raise ValueError("Provide exactly one of media_id or path.")
    if media_id:
        if not _is_media_id(media_id):
            raise ValueError("media_id must be returned by wangp_list_gallery.")
        record = _gallery_item(session, media_id)
        resolved_path = _resolve_existing_path(record["path"], "Gallery media")
        settings = record.get("settings") if isinstance(record.get("settings"), dict) else {}
        if not settings:
            settings = _extract_media_settings(session, resolved_path)
        return {"status": "done", "source": "gallery", "media_id": media_id, "media_type": record["media_type"], "filename": Path(resolved_path).name, "settings": _json_safe(settings)}
    if not allow_read_file_system:
        raise PermissionError("Direct filesystem paths are disabled for this MCP server. Use a media_id returned by wangp_list_gallery.")
    if _is_media_id(path):
        raise ValueError("Use media_id, not path, for Gallery media.")
    resolved_path = _resolve_existing_path(path, "media path")
    media_type = _mcp_media_type(resolved_path)
    if not media_type:
        raise ValueError(f"Unsupported media file extension: {Path(resolved_path).suffix}")
    return {"status": "done", "source": "filesystem", "media_id": "", "path": resolved_path, "media_type": media_type, "filename": Path(resolved_path).name, "settings": _json_safe(_extract_media_settings(session, resolved_path))}


def _compact_model_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for record in records:
        item = copy.deepcopy(record)
        item.pop("setting_values", None)
        compact.append(item)
    return compact


def _mcp_model_definition(model_def: dict[str, Any] | None) -> dict[str, Any] | None:
    if model_def is None:
        return None
    result = copy.deepcopy(model_def)
    result.pop("settings", None)
    return result


def _deepy_ui_settings(session) -> dict[str, Any]:
    from shared.deepy import ui_settings as deepy_ui_settings

    assistant_session = session._state.get("assistant_session")
    live_settings = getattr(assistant_session, "tool_ui_settings", None)
    return deepy_ui_settings.normalize_assistant_tool_ui_settings(**live_settings) if isinstance(live_settings, dict) and live_settings else deepy_ui_settings.get_persisted_assistant_tool_ui_settings()


def _deepy_template_defaults(session) -> dict[str, str]:
    settings = _deepy_ui_settings(session)
    return {
        "gen_image": settings["image_generator_variant"],
        "edit_image": settings["image_editor_variant"],
        "gen_video": settings["video_generator_variant"],
        "gen_video_with_speech": settings["video_with_speech_variant"],
        "gen_song": settings["song_variant"],
        "gen_speech_from_description": settings["speech_from_description_variant"],
        "gen_speech_from_sample": settings["speech_from_sample_variant"],
    }


def _deepy_template_catalog(session, tool_id: str | None = None) -> list[dict[str, Any]]:
    from shared.deepy import tool_settings as deepy_tool_settings

    requested_tool = str(tool_id or "").strip()
    if requested_tool and requested_tool not in deepy_tool_settings.GENERATION_TOOL_IDS:
        raise ValueError(f"tool_id must be one of: {', '.join(deepy_tool_settings.GENERATION_TOOL_IDS)}")
    defaults = _deepy_template_defaults(session)
    tool_ids = [requested_tool] if requested_tool else list(deepy_tool_settings.GENERATION_TOOL_IDS)
    catalog = []
    for one_tool_id in tool_ids:
        default_template = defaults[one_tool_id]
        templates = [
            {"template": variant, "label": label, "default": variant == default_template}
            for label, variant in deepy_tool_settings.list_tool_variant_choices(one_tool_id, current_variant=default_template)
        ]
        catalog.append({"tool_id": one_tool_id, "label": deepy_tool_settings.TOOL_DISPLAY_NAMES[one_tool_id], "default_template": default_template, "templates": templates})
    return catalog


def _deepy_general_properties(session) -> dict[str, Any]:
    settings = _deepy_ui_settings(session)
    return {key: settings[key] for key in ("use_template_properties", "width", "height", "num_frames", "audio_duration", "seed")}


def _strip_deepy_general_property_settings(tool_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    stripped = copy.deepcopy(settings)
    conflicting_keys = {"seed"}
    if tool_id in _DEEPY_VISUAL_TOOL_IDS:
        conflicting_keys.update(("resolution", "width", "height"))
        if tool_id in _DEEPY_VIDEO_TOOL_IDS:
            conflicting_keys.update(("video_length", "num_frames"))
    elif tool_id in _DEEPY_AUDIO_TOOL_IDS:
        conflicting_keys.update(("duration_seconds", "audio_duration"))
    for key in conflicting_keys:
        stripped.pop(key, None)
    return stripped


def _strip_deepy_settings_metadata(settings: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(settings)
    stripped.pop("settings_version", None)
    stripped.pop("type", None)
    return stripped


def _relevant_deepy_general_properties(tool_id: str, general_properties: dict[str, Any]) -> dict[str, Any]:
    if tool_id in _DEEPY_VIDEO_TOOL_IDS:
        keys = ("width", "height", "num_frames", "seed")
    elif tool_id in _DEEPY_VISUAL_TOOL_IDS:
        keys = ("width", "height", "seed")
    elif tool_id in _DEEPY_AUDIO_TOOL_IDS:
        keys = ("audio_duration", "seed")
    else:
        keys = ()
    return {key: general_properties[key] for key in keys}


def _deepy_template_settings(session, tool_id: str, template: str) -> dict[str, Any]:
    from shared.deepy import tool_settings as deepy_tool_settings

    catalog = _deepy_template_catalog(session, tool_id=tool_id)[0]
    requested_template = str(template or "").strip()
    resolved_template = catalog["default_template"] if requested_template.casefold() == "default" else deepy_tool_settings.find_tool_variant(tool_id, requested_template, current_variant=catalog["default_template"])
    if resolved_template is None:
        raise ValueError(f"Unknown Deepy template '{template}' for tool '{tool_id}'.")
    template_settings = session.merge_settings_with_defaults(deepy_tool_settings.clone_tool_preset(tool_id, resolved_template))
    general_properties = _deepy_general_properties(session)
    effective_settings = session.prepare_settings_for_export(template_settings)
    if not general_properties["use_template_properties"]:
        effective_settings = _strip_deepy_general_property_settings(tool_id, effective_settings)
    effective_settings = _strip_deepy_settings_metadata(effective_settings)
    result = {
        "tool_id": tool_id,
        "template": resolved_template,
        "default": resolved_template == catalog["default_template"],
        "default_template": catalog["default_template"],
        "general_properties_active": not general_properties["use_template_properties"] and tool_id in _DEEPY_VISUAL_TOOL_IDS | _DEEPY_AUDIO_TOOL_IDS,
        "settings": _json_safe(effective_settings),
    }
    if not general_properties["use_template_properties"]:
        result["general_properties"] = _json_safe(_relevant_deepy_general_properties(tool_id, general_properties))
    return result


def _resolve_existing_path(value: Any, label: str) -> str:
    path = Path(str(value or "").strip()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return str(path)


def _mcp_media_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    if suffix in _AUDIO_EXTENSIONS:
        return "audio"
    return ""


def _is_media_id(value: Any) -> bool:
    return str(value or "").strip().lower().startswith(("visual:", "audio:"))


def _resolve_mcp_media_reference(session, value: Any, label: str, allow_read_file_system: bool) -> str:
    reference = str(value or "").strip()
    if not reference:
        raise ValueError(f"{label} is empty")
    if _is_media_id(reference):
        path = _gallery_item(session, reference)["path"]
        return _resolve_existing_path(path, label)
    if not allow_read_file_system:
        raise PermissionError(f"Direct filesystem paths are disabled for this MCP server. Use a media_id returned by wangp_list_gallery for {label}, or restart with filesystem reads enabled.")
    return _resolve_existing_path(reference, label)


def _resolve_mcp_media_input(session, media_id: str | None, path: str | None, allow_read_file_system: bool) -> str:
    media_id, path = str(media_id or "").strip(), str(path or "").strip()
    if bool(media_id) == bool(path):
        raise ValueError("Provide exactly one of media_id or path.")
    if media_id and not _is_media_id(media_id):
        raise ValueError("media_id must be returned by wangp_list_gallery.")
    if path and _is_media_id(path):
        raise ValueError("Use media_id, not path, for Gallery media.")
    return _resolve_mcp_media_reference(session, media_id or path, "media_id" if media_id else "path", allow_read_file_system)


def _resolve_mcp_media_value(session, value: Any, label: str, allow_read_file_system: bool) -> Any:
    if value is None or isinstance(value, str) and not value.strip():
        return value
    if isinstance(value, (list, tuple)):
        return [_resolve_mcp_media_reference(session, item, label, allow_read_file_system) for item in value]
    return _resolve_mcp_media_reference(session, value, label, allow_read_file_system)


def _resolve_generation_media(session, source: dict[str, Any] | list[dict[str, Any]], allow_read_file_system: bool):
    resolved = copy.deepcopy(source)

    def resolve_task(task: dict[str, Any]) -> None:
        if isinstance(task.get("tasks"), list):
            for child in task["tasks"]:
                if not isinstance(child, dict):
                    raise TypeError("Manifest tasks must be dictionaries")
                resolve_task(child)
            return
        settings = task.get("params") if isinstance(task.get("params"), dict) else task.get("settings") if isinstance(task.get("settings"), dict) else task
        for key in _MCP_MEDIA_SETTING_KEYS & settings.keys():
            settings[key] = _resolve_mcp_media_value(session, settings[key], key, allow_read_file_system)
        if not allow_read_file_system:
            for lora in settings.get("activated_loras", []) or []:
                lora_text = str(lora or "").strip()
                lora_path = Path(lora_text)
                if lora_path.is_absolute() or ".." in lora_text.replace("\\", "/").split("/") or lora_text.lower().startswith("file:"):
                    raise PermissionError("Direct filesystem paths are disabled for activated_loras. Use identifiers returned by wangp_list_loras.")

    if isinstance(resolved, list):
        for task in resolved:
            if not isinstance(task, dict):
                raise TypeError("Generation tasks must be dictionaries")
            resolve_task(task)
    else:
        resolve_task(resolved)
    return resolved


def _register_gallery_media(session, path: str) -> dict[str, Any]:
    resolved_path = _resolve_existing_path(path, "uploaded media")
    media_type = _mcp_media_type(resolved_path)
    if not media_type:
        raise ValueError(f"Unsupported media file extension: {Path(resolved_path).suffix}")
    if media_type == "image":
        from PIL import Image

        with Image.open(resolved_path) as image:
            image.verify()
    elif media_type == "video":
        from shared.utils.video_decode import probe_video_stream_metadata

        if probe_video_stream_metadata(resolved_path) is None:
            raise ValueError(f"Unable to read uploaded video: {Path(resolved_path).name}")
    else:
        from shared.utils.audio_video import get_audio_file_sample_rate

        get_audio_file_sample_rate(resolved_path)
    runtime = session._ensure_runtime()
    with contextlib.chdir(runtime.root):
        settings, _any_visual, _any_audio = runtime.module.get_settings_from_file(session._state, resolved_path, False, False, False)
    settings = settings if isinstance(settings, dict) else {}
    gen = session._state["gen"]
    with _GALLERY_LOCK:
        if media_type == "audio":
            paths = gen["audio_file_list"]
            settings_list = gen["audio_file_settings_list"]
            selected_key = "audio_selected"
        else:
            paths = gen["file_list"]
            settings_list = gen["file_settings_list"]
            selected_key = "selected"
        if resolved_path in paths:
            index = paths.index(resolved_path)
            if index < len(settings_list) and settings and not settings_list[index]:
                settings_list[index] = settings
        else:
            paths.append(resolved_path)
            settings_list.append(settings)
            index = len(paths) - 1
        gen[selected_key] = index
    return next(record for record in _gallery_records(session, limit=500) if record["path"] == resolved_path)


class _MediaTransferStore:
    def __init__(self, session) -> None:
        self.session = session
        self._uploads: dict[str, dict[str, Any]] = {}
        self._downloads: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _cleanup(self) -> None:
        cutoff = time.time() - _MEDIA_TRANSFER_TTL_SECONDS
        for store in (self._uploads, self._downloads):
            for token, record in list(store.items()):
                if record["created"] < cutoff:
                    store.pop(token, None)

    def create_upload(self, filename: str) -> dict[str, Any]:
        safe_name = Path(str(filename or "").strip()).name
        if safe_name in {"", ".", ".."} or not _mcp_media_type(safe_name):
            raise ValueError("filename must have a supported image, video, or audio extension")
        token = uuid.uuid4().hex
        with self._lock:
            self._cleanup()
            self._uploads[token] = {"created": time.time(), "filename": safe_name}
        return {"upload_url": f"/wangp_api/gallery/upload/{token}", "method": "PUT", "filename": safe_name, "expires_in_seconds": _MEDIA_TRANSFER_TTL_SECONDS, "max_bytes": _MAX_UPLOAD_BYTES}

    def consume_upload(self, token: str) -> dict[str, Any] | None:
        with self._lock:
            self._cleanup()
            return self._uploads.pop(str(token or ""), None)

    def create_download(self, media_id: str) -> dict[str, Any]:
        record = _gallery_item(self.session, media_id)
        path = _resolve_existing_path(record["path"], "Gallery media")
        token = uuid.uuid4().hex
        with self._lock:
            self._cleanup()
            self._downloads[token] = {"created": time.time(), "media_id": record["media_id"]}
        return {"media_id": record["media_id"], "download_url": f"/wangp_api/gallery/download/{token}", "filename": Path(path).name, "media_type": record["media_type"], "size": Path(path).stat().st_size, "expires_in_seconds": _MEDIA_TRANSFER_TTL_SECONDS}

    def consume_download(self, token: str) -> dict[str, Any] | None:
        with self._lock:
            self._cleanup()
            transfer = self._downloads.pop(str(token or ""), None)
        return None if transfer is None else _gallery_item(self.session, transfer["media_id"])

    def upload_path(self, filename: str) -> Path:
        root = Path(self.session._output_dir) if self.session._output_dir is not None else self.session._root / "outputs"
        upload_root = (root / "mcp_uploads" / uuid.uuid4().hex).resolve()
        upload_root.mkdir(parents=True, exist_ok=True)
        return upload_root / Path(filename).name


def _mcp_postprocessing_processes(media_type: str, allow_read_file_system: bool = False, processes: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    from postprocessing import catalog as postprocessing_catalog

    processes = postprocessing_catalog.call_processes(postprocessing_catalog.query_processes(media_type) if processes is None else processes)
    for process in processes:
        for parameter in process.get("parameters", ()):
            original_name = str(parameter.get("name", "") or "")
            replacement = _POSTPROCESS_PATH_PARAMETERS.get(original_name)
            if replacement is None:
                if parameter.get("media_type") == "image":
                    parameter["description"] = str(parameter.get("description", "") or "").rstrip() + (" Values may be Gallery media IDs or existing server image paths." if allow_read_file_system else " Values must be Gallery media IDs returned by wangp_list_gallery.")
                continue
            parameter["name"] = replacement
            parameter["description"] = "Media ID for the required audio file" + (" or an existing server filesystem path." if allow_read_file_system else ". Direct filesystem paths are disabled for this server.")
    return processes


def _build_mcp_postprocessing_task(source_path: str, process: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    from postprocessing import catalog as postprocessing_catalog

    process_id = str(process["id"])
    process_type = str(process["type"])
    task = {
        "prompt": str(process["label"]),
        "resolution": "",
        "image_mode": 1 if _mcp_media_type(source_path) == "image" else 0,
        "repeat_generation": 1,
        "seed": int(parameters.get("seed", -1)),
        "temporal_upsampling": "",
        "spatial_upsampling": "",
        "film_grain_intensity": 0,
        "film_grain_saturation": 0.5,
        "postprocess_audio": "",
        "postprocess_audio_prompt": str(parameters.get("prompt", "")),
        "postprocess_audio_neg_prompt": str(parameters.get("negative_prompt", "")),
        "replace_voice_method": "",
        "replace_voice_sample": None,
        "replace_voice_sample2": None,
    }
    if process_type == postprocessing_catalog.PROCESS_TYPE_SPATIAL_UPSAMPLING:
        process_value = postprocessing_catalog.build_process_value(process, parameters)
        if process_value is None:
            raise ValueError("The spatial upsampling parameters could not be converted to a valid process value.")
        task.update({"mode": "edit_postprocessing", "video_source": source_path, "spatial_upsampling": process_value})
    elif process_type == postprocessing_catalog.PROCESS_TYPE_TEMPORAL_UPSAMPLING:
        process_value = postprocessing_catalog.build_process_value(process, parameters)
        if process_value is None:
            raise ValueError("The temporal upsampling parameters could not be converted to a valid process value.")
        task.update({"mode": "edit_postprocessing", "video_source": source_path, "temporal_upsampling": process_value})
    elif process_type in {postprocessing_catalog.PROCESS_TYPE_SOUNDTRACK, postprocessing_catalog.PROCESS_TYPE_VOICE_REPLACEMENT}:
        task.update({"mode": "edit_remux", "video_source": source_path, "postprocess_audio": process_id})
    elif process_type == postprocessing_catalog.PROCESS_TYPE_AUDIO_EDIT:
        task.update({"mode": "edit_audio", "audio_source": source_path, "postprocess_audio": process_id})
    parameter_defs = {str(parameter["name"]): parameter for parameter in process.get("parameters", ())}
    spatial_parameter_names = set()
    if process_type == postprocessing_catalog.PROCESS_TYPE_SPATIAL_UPSAMPLING:
        spatial_parameters = {name: value for name, value in parameters.items() if name != "multiplier"}
        spatial_parameter_names.update(spatial_parameters)
        task.update(spatial_parameters)
    path_task_keys = {"audio_path": "audio_source", "voice_sample_path": "replace_voice_sample", "voice_sample2_path": "replace_voice_sample2"}
    for parameter_name, task_key in path_task_keys.items():
        if parameter_name in parameters:
            task[task_key] = _resolve_existing_path(parameters[parameter_name], parameter_name)
    standard_parameters = {"multiplier", "seed", "prompt", "negative_prompt", *path_task_keys, *spatial_parameter_names}
    for name, value in parameters.items():
        if name not in standard_parameters:
            task[str(parameter_defs[name].get("setting", name))] = value
    return task


def _build_default_toolbox(session):
    from shared.deepy.engine import AssistantSessionState, DeepyZeroTools

    runtime = session._ensure_runtime()
    assistant_session = session._state.get("assistant_session")
    if assistant_session is None:
        assistant_session = AssistantSessionState()
        session._state["assistant_session"] = assistant_session
    return DeepyZeroTools(session._state["gen"], runtime.module.get_processed_queue, lambda *_args, **_kwargs: None, session=assistant_session, get_output_filepath=runtime.module.get_output_filepath, record_file_metadata=runtime.module.record_file_metadata, get_server_config=lambda: runtime.module.server_config)


def _toolbox_discovery(toolbox, allow_read_file_system: bool = False) -> list[dict[str, Any]]:
    actions = []
    for schema in toolbox.get_tool_schemas():
        function = copy.deepcopy(schema.get("function", {}))
        name = str(function.get("name", "") or "")
        if name not in _TOOLBOX_ACTIONS:
            continue
        for parameter_name in _TOOLBOX_MEDIA_PARAMETERS.get(name, ()):
            parameter = function.get("parameters", {}).get("properties", {}).get(parameter_name)
            if isinstance(parameter, dict):
                plural = parameter.get("type") == "array"
                maximum = int(parameter.get("maxItems", 0) or 0)
                plural_description = f"One to {maximum} media IDs returned by wangp_list_gallery" if maximum > 0 else "Media IDs returned by wangp_list_gallery"
                parameter["description"] = f"{plural_description if plural else 'A media ID returned by wangp_list_gallery'}" + (" or existing server filesystem paths." if allow_read_file_system and plural else " or an existing server filesystem path." if allow_read_file_system else ". Direct filesystem paths are disabled for this server.")
        if name == "inspect_media":
            nested_media_id = function.get("parameters", {}).get("properties", {}).get("media_inputs", {}).get("items", {}).get("properties", {}).get("media_id")
            if isinstance(nested_media_id, dict):
                nested_media_id["description"] = "A media ID returned by wangp_list_gallery or an existing server filesystem path." if allow_read_file_system else "A media ID returned by wangp_list_gallery. Direct filesystem paths are disabled for this server."
        actions.append(function)
    return sorted(actions, key=lambda action: str(action.get("name", "")))


def _resolve_toolbox_arguments(session, toolbox, action: str, arguments: dict[str, Any], allow_read_file_system: bool = False) -> dict[str, Any]:
    from shared.deepy import media_registry

    resolved = dict(arguments or {})

    def resolve_media_id(value: Any, parameter_name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{parameter_name} must be a string.")
        value = value.strip()
        if not value or media_registry.get_media_record(toolbox.session, value) is not None:
            return value
        media_path = _resolve_mcp_media_reference(session, value, parameter_name, allow_read_file_system)
        record = media_registry.register_media(toolbox.session, media_path, source="gallery" if _is_media_id(value) else "filesystem")
        if record is None:
            raise ValueError(f"Unsupported media file: {media_path}")
        return record["media_id"]

    for parameter_name in _TOOLBOX_MEDIA_PARAMETERS.get(action, ()):
        raw_value = resolved.get(parameter_name, None)
        if raw_value is None:
            continue
        if parameter_name == "media_ids" and not isinstance(raw_value, list):
            raise ValueError("media_ids must be an array.")
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        resolved_values = []
        for index, raw_item in enumerate(values):
            item_parameter_name = f"{parameter_name}[{index}]" if isinstance(raw_value, list) else parameter_name
            resolved_values.append(resolve_media_id(raw_item, item_parameter_name))
        resolved[parameter_name] = resolved_values if isinstance(raw_value, list) else resolved_values[0]
    if action == "inspect_media" and resolved.get("media_inputs") is not None:
        raw_inputs = resolved["media_inputs"]
        if not isinstance(raw_inputs, list):
            raise ValueError("media_inputs must be an array.")
        resolved_inputs = []
        for index, raw_input in enumerate(raw_inputs):
            if not isinstance(raw_input, dict):
                raise ValueError(f"media_inputs[{index}] must be an object.")
            resolved_input = dict(raw_input)
            resolved_input["media_id"] = resolve_media_id(resolved_input.get("media_id"), f"media_inputs[{index}].media_id")
            resolved_inputs.append(resolved_input)
        resolved["media_inputs"] = resolved_inputs
    return resolved


def _register_toolbox_result_media(session, result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    output_paths = []
    if isinstance(result.get("output_file"), str) and result["output_file"].strip():
        output_paths.append(result["output_file"])
    if isinstance(result.get("output_files"), list):
        output_paths.extend(path for path in result["output_files"] if isinstance(path, str) and path.strip())
    gallery_items = [_register_gallery_media(session, path) for path in output_paths if Path(path).is_file() and _mcp_media_type(path)]
    if gallery_items:
        result = dict(result)
        result["gallery_items"] = gallery_items
    return result


class _JobRecord:
    def __init__(self, job_id: str, job: "SessionJob", session) -> None:
        self.job_id = job_id
        self.job = job
        self.session = session
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.events: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._watcher = threading.Thread(target=self._watch, daemon=True, name=f"wangp-mcp-job-{job_id}")

    def start(self) -> None:
        self._watcher.start()

    def _watch(self) -> None:
        for event in self.job.events.iter(timeout=0.2):
            event_dict = _event_to_dict(event)
            with self._lock:
                self.events.append(event_dict)
                if len(self.events) > _MAX_STORED_EVENTS:
                    del self.events[: len(self.events) - _MAX_STORED_EVENTS]
                self.updated_at = time.time()
                if event.kind == "completed" and type(event.data).__name__ == "GenerationResult":
                    self.result = _result_to_dict(event.data)
        self._capture_result_if_done()

    def _capture_result_if_done(self) -> None:
        if not self.job.done:
            return
        try:
            result = _result_to_dict(self.job.result(timeout=0))
        except Exception:
            return
        with self._lock:
            self.result = result
            self.updated_at = time.time()

    def snapshot(self, event_limit: int = 20) -> dict[str, Any]:
        self._capture_result_if_done()
        event_limit = 20 if event_limit is None else event_limit
        event_limit = max(0, min(int(event_limit), _MAX_STORED_EVENTS))
        with self._lock:
            events = copy.deepcopy(self.events[-event_limit:] if event_limit else [])
            result = copy.deepcopy(self.result)
            updated_at = self.updated_at
        if isinstance(result, dict):
            generated_paths = {str(path).replace("\\", "/").casefold() for path in result.get("generated_files", [])}
            result["gallery_items"] = [_compact_gallery_record(record) for record in _gallery_records(self.session, limit=500) if record["path"].replace("\\", "/").casefold() in generated_paths]
        return {
            "job_id": self.job_id,
            "done": self.job.done,
            "cancel_requested": self.job.cancel_requested,
            "webui_submission_ready": self.job.webui_submission_ready,
            "webui_load_queue_token": self.job.webui_load_queue_token if self.job.webui_submission_ready else "",
            "created_at": self.created_at,
            "updated_at": updated_at,
            "events": events,
            "result": result,
        }


class _JobStore:
    def __init__(self, session) -> None:
        self._session = session
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.Lock()

    def submit(self, source: dict[str, Any] | list[dict[str, Any]]) -> _JobRecord:
        job = self._session.submit(source)
        record = _JobRecord(uuid.uuid4().hex, job, self._session)
        with self._lock:
            self._jobs[record.job_id] = record
        record.start()
        return record

    def get(self, job_id: str) -> _JobRecord:
        with self._lock:
            record = self._jobs.get(str(job_id or "").strip())
        if record is None:
            raise KeyError(f"Unknown WanGP job_id: {job_id}")
        return record


def _config_file_from_arg(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        return str(path / "wgp_config.json")
    return str(path)


def build_server_for_session(session, settings: dict[str, Any] | None = None, toolbox=None, default_job_event_limit: int = 20, allow_read_file_system: bool = False, http_media_transfer: bool = False):
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError("WanGP MCP server requires the 'mcp' Python package. Install project requirements or run `pip install mcp`.") from exc

    jobs = _JobStore(session)
    mcp = FastMCP("WanGP", **dict(settings or {}))
    _register_documentation_resources(mcp)
    default_job_event_limit = max(0, min(int(default_job_event_limit), _MAX_STORED_EVENTS))
    allow_read_file_system = bool(allow_read_file_system)
    transfer_store = _MediaTransferStore(session) if http_media_transfer else None
    toolbox_instance = toolbox

    if transfer_store is not None:
        @mcp.custom_route("/wangp_api/gallery/upload/{token}", methods=["PUT"], include_in_schema=False)
        async def upload_gallery_media(request):
            from starlette.responses import JSONResponse

            transfer = transfer_store.consume_upload(request.path_params["token"])
            if transfer is None:
                return JSONResponse({"error": "Upload expired or not found."}, status_code=404)
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    return JSONResponse({"error": "Invalid Content-Length header."}, status_code=400)
                if declared_length > _MAX_UPLOAD_BYTES:
                    return JSONResponse({"error": f"Upload exceeds the {_MAX_UPLOAD_BYTES}-byte limit."}, status_code=413)
            target = transfer_store.upload_path(transfer["filename"])
            size = 0
            try:
                with target.open("xb") as writer:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > _MAX_UPLOAD_BYTES:
                            raise ValueError(f"Upload exceeds the {_MAX_UPLOAD_BYTES}-byte limit.")
                        writer.write(chunk)
                record = _register_gallery_media(session, str(target))
            except Exception as exc:
                target.unlink(missing_ok=True)
                try:
                    target.parent.rmdir()
                except OSError:
                    pass
                return JSONResponse({"error": str(exc)}, status_code=413 if size > _MAX_UPLOAD_BYTES else 400)
            return JSONResponse({"status": "uploaded", "size": size, **_compact_gallery_record(record)})

        @mcp.custom_route("/wangp_api/gallery/download/{token}", methods=["GET"], include_in_schema=False)
        async def download_gallery_media(request):
            from starlette.responses import FileResponse, JSONResponse

            record = transfer_store.consume_download(request.path_params["token"])
            if record is None:
                return JSONResponse({"error": "Download expired or not found."}, status_code=404)
            path = _resolve_existing_path(record["path"], "Gallery media")
            return FileResponse(path, filename=Path(path).name, media_type=mimetypes.guess_type(path)[0] or "application/octet-stream")

    def get_toolbox():
        nonlocal toolbox_instance
        if toolbox_instance is None:
            toolbox_instance = _build_default_toolbox(session)
        return toolbox_instance

    @mcp.prompt(name="wangp_agent", title="WanGP Agent Guide", description="Instructions for discovering WanGP models, building settings, running jobs, and handling media.")
    def wangp_agent_prompt() -> str:
        return _AGENT_GUIDE_PATH.read_text(encoding="utf-8")

    @mcp.tool()
    def wangp_list_models(family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, model_type: str | None = None, main_output: str | None = None, inputs: str | None = None, name: str | None = None, query: str | None = None, limit: int = 10, offset: int = 0, include_availability: bool = False) -> list[dict[str, Any]]:
        """List at most 10 compact model records. All string filters accept case-insensitive * and ? globs; query searches names, ids, families, and descriptions. For generation without a user-specified model, use the corresponding default template instead of browsing."""

        return _compact_model_metadata(session.list_model_metadata(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs, name=name, query=query, limit=max(1, min(int(limit), 10)), offset=max(0, int(offset)), include_availability=include_availability))

    @mcp.tool()
    def wangp_search_models(query: str, family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, main_output: str | None = None, inputs: str | None = None, limit: int = 10, offset: int = 0, include_availability: bool = False) -> dict[str, Any]:
        """Search models by user-facing name, model id, architecture, family, or description and return a bounded page with totals. Other string filters accept * and ? globs."""

        if not str(query or "").strip():
            raise ValueError("query is required")
        matches = session.list_model_metadata(family=family, base_model_type=base_model_type, finetune=finetune, main_output=main_output, inputs=inputs, query=query, include_availability=include_availability)
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 50))
        page = _compact_model_metadata(matches[offset:offset + limit])
        return {"matches": page, "total_matches": len(matches), "returned": len(page), "offset": offset, "has_more": offset + len(page) < len(matches)}

    @mcp.tool()
    def wangp_list_model_defs(family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, model_type: str | None = None, main_output: str | None = None, inputs: str | None = None, name: str | None = None, query: str | None = None, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        """List a bounded page of full WanGP model definitions. String filters accept * and ? globs. Prefer search, then fetch one selected model."""

        return session.list_model_defs(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs, name=name, query=query, limit=max(1, min(int(limit), 20)), offset=max(0, int(offset)))

    @mcp.tool()
    def wangp_get_model(model_type: str) -> dict[str, Any] | None:
        """Return one WanGP model definition with parameter declarations and capabilities. Embedded default values are omitted. Do not call this after a template query unless a required parameter remains absent from both template settings and the compact schema."""

        return _mcp_model_definition(session.get_model_def(model_type))

    @mcp.tool()
    def wangp_get_model_metadata(model_type: str, include_availability: bool = False) -> dict[str, Any] | None:
        """Return one compact model metadata record."""

        return session.get_model_metadata(model_type, include_availability=include_availability)

    @mcp.tool()
    def wangp_get_model_availability(model_type: str) -> dict[str, Any]:
        """Return local file availability for one model using the same status as the UI model selector."""

        return session.get_model_availability(model_type)

    @mcp.tool()
    def wangp_list_model_availability(family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, model_type: str | None = None, main_output: str | None = None, inputs: str | None = None, name: str | None = None, query: str | None = None, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """List a bounded page of local model-file availability records using the model discovery filters."""

        return session.list_model_availability(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs, name=name, query=query, limit=max(1, min(int(limit), 50)), offset=max(0, int(offset)))

    @mcp.tool()
    def wangp_get_default_settings(model_type: str) -> dict[str, Any]:
        """Return pristine model defaults generated from WanGP and the model handler, filtered to relevant fields and without fixed metadata such as type or settings version. User-saved UI defaults are not included. Do not call this after a template query because template settings already include these model defaults."""

        return _strip_deepy_settings_metadata(session.get_exported_default_settings(model_type))

    @mcp.tool()
    def wangp_list_loras(model_type: str, name: str | None = None) -> dict[str, Any]:
        """Recursively list locally available LoRAs for a model. Optional name accepts a case-insensitive * and ? glob over each subfolder-relative identifier. Returned identifiers can be copied directly into activated_loras and paired with loras_multipliers."""

        return session.list_loras(model_type, name=name)

    @mcp.tool()
    def wangp_get_model_schema(model_type: str) -> dict[str, Any] | None:
        """Return a compact capability and usage summary for one model. After a template query, call this only to verify a required capability or limit that the template does not establish. Use wangp_get_default_settings only for raw non-template generation requests."""

        return session.get_model_schema(model_type)

    @mcp.tool()
    def wangp_list_gallery(media_type: str = "all", limit: int = 50, selected_only: bool = False) -> list[dict[str, Any]]:
        """List compact summaries of current and remembered session Gallery media, including actual video resolution, frame count, FPS, and duration from WanGP's cached media probe. Set selected_only to return only live UI selections. Use media_id with wangp_get_media_settings only when full generation settings are needed."""

        records = _gallery_records(session, media_type=media_type, limit=500 if selected_only else limit)
        if selected_only:
            records = [record for record in records if record["selected"]]
        return [_compact_gallery_record(record) for record in records[-max(1, min(int(limit), 500)):]]

    @mcp.tool()
    def wangp_get_media_settings(media_id: str | None = None, path: str | None = None) -> dict[str, Any]:
        """Return generation settings for one media file. Provide exactly one input: media_id for Gallery media, or path for a server file when filesystem reads are enabled."""

        return _media_settings(session, media_id=media_id, path=path, allow_read_file_system=allow_read_file_system)

    if allow_read_file_system:
        @mcp.tool()
        def wangp_list_files(path: str, extensions: list[str] | str | None = None) -> dict[str, Any]:
            """List files directly inside a server directory, optionally filtered by extensions. Returns names, extensions, paths, and byte sizes."""

            from shared.deepy import filesystem

            return filesystem.list_files(path, extensions)

        @mcp.tool()
        def wangp_query_file(media_id: str | None = None, path: str | None = None) -> dict[str, Any]:
            """Inspect exactly one media source: media_id for Gallery media, or path for a server file. Returns useful media metadata, or UTF-8 text up to 16,000 characters."""

            from shared.deepy import filesystem

            return filesystem.query_file(_resolve_mcp_media_input(session, media_id, path, True))

    if transfer_store is not None:
        @mcp.tool()
        def wangp_create_gallery_upload(filename: str) -> dict[str, Any]:
            """Create a short-lived HTTP PUT URL for uploading image, video, or audio media. A successful upload registers and selects the media in the appropriate WanGP Gallery."""

            return transfer_store.create_upload(filename)

        @mcp.tool()
        def wangp_create_gallery_download(media_id: str) -> dict[str, Any]:
            """Create a short-lived HTTP GET URL for downloading one Gallery item. Only registered Gallery media can be downloaded."""

            return transfer_store.create_download(media_id)

    @mcp.tool()
    def wangp_list_deepy_templates(tool_id: str | None = None) -> list[dict[str, Any]]:
        """List settings templates available in Deepy's Template Settings section. The current default for every tool is marked explicitly."""

        return _deepy_template_catalog(session, tool_id=tool_id)

    @mcp.tool()
    def wangp_get_deepy_template_settings(tool_id: str, template: str) -> dict[str, Any]:
        """Return merged, migrated, model-filtered, non-conflicting settings for a Deepy template. For any generation task or derived subtask without a user-specified model, call this with template='default' before model discovery and use the result directly. If general_properties is returned, apply those values after settings."""

        return _deepy_template_settings(session, tool_id, template)

    @mcp.tool()
    def wangp_postprocess(media_id: str | None = None, path: str | None = None, process: str | None = None, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Discover or run compatible post-processing. Provide exactly one input: media_id for Gallery media, or path for a server file when filesystem reads are enabled. Omit process for discovery."""

        from postprocessing import catalog as postprocessing_catalog

        source_path = _resolve_mcp_media_input(session, media_id, path, allow_read_file_system)
        media_type = _mcp_media_type(source_path)
        if not media_type:
            raise ValueError(f"Unsupported media file extension: {Path(source_path).suffix}")
        processes = postprocessing_catalog.query_processes(media_type)
        discovered_processes = _mcp_postprocessing_processes(media_type, allow_read_file_system, processes)
        process_id = str(process or "").strip()
        if not process_id:
            return {"status": "discovery", "path": source_path, "media_type": media_type, "processes": discovered_processes, "count": len(discovered_processes)}
        matches = [candidate for candidate in processes if candidate["id"] == process_id]
        if len(matches) != 1:
            raise ValueError(f"Post-processing process '{process_id}' is {'not available' if not matches else 'ambiguous'} for {media_type}.")
        process_def = matches[0]
        normalized_parameters, error = postprocessing_catalog.normalize_parameters(process_def, parameters)
        if error:
            raise ValueError(error)
        for parameter_name in set(_POSTPROCESS_PATH_PARAMETERS.values()) & normalized_parameters.keys():
            normalized_parameters[parameter_name] = _resolve_mcp_media_reference(session, normalized_parameters[parameter_name], parameter_name, allow_read_file_system)
        for parameter_def in process_def.get("parameters", ()):
            parameter_name = str(parameter_def.get("name", "") or "")
            if parameter_def.get("media_type") == "image" and parameter_name in normalized_parameters:
                normalized_parameters[parameter_name] = _resolve_mcp_media_value(session, normalized_parameters[parameter_name], parameter_name, allow_read_file_system)
        task = _build_mcp_postprocessing_task(source_path, process_def, normalized_parameters)
        record = jobs.submit(task)
        snapshot = record.snapshot(event_limit=20)
        snapshot.update({"source_path": source_path, "media_type": media_type, "process": process_id, "process_label": process_def["label"], "parameters": normalized_parameters})
        return snapshot

    @mcp.tool()
    def wangp_toolbox(action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Discover or run WanGP media utilities. Omit action for a compact action list; pass action without arguments for its schema; then pass arguments to execute. Media arguments accept media IDs returned by wangp_list_gallery; direct server paths require startup permission."""

        sandbox_toolbox = get_toolbox()
        action_name = str(action or "").strip()
        actions = _toolbox_discovery(sandbox_toolbox, allow_read_file_system)
        if not action_name:
            compact_actions = [{"name": candidate["name"], "description": candidate.get("description", "")} for candidate in actions]
            return {"status": "discovery", "actions": compact_actions, "count": len(compact_actions), "next": "Call again with one action and omit arguments to get its exact schema."}
        if action_name not in _TOOLBOX_ACTIONS:
            raise ValueError(f"Unknown toolbox action '{action_name}'. Call without action to discover available actions.")
        action_defs = [candidate for candidate in actions if candidate["name"] == action_name]
        if not action_defs:
            raise ValueError(f"Toolbox action '{action_name}' is unavailable in this WanGP runtime.")
        if arguments is None:
            return {"status": "schema", "action": action_defs[0]}
        resolved_arguments = _resolve_toolbox_arguments(session, sandbox_toolbox, action_name, dict(arguments or {}), allow_read_file_system)
        validation_error = sandbox_toolbox.validate_tool_call(action_name, resolved_arguments)
        if validation_error:
            raise ValueError(validation_error)
        return _register_toolbox_result_media(session, sandbox_toolbox.call(action_name, resolved_arguments))

    @mcp.tool()
    def wangp_generate(source: dict[str, Any] | list[dict[str, Any]], wait: bool = False, timeout_s: float | None = None, event_limit: int | None = None) -> dict[str, Any]:
        """Start a WanGP generation from a settings dict, task dict, or task list."""

        if not isinstance(source, (dict, list)):
            raise TypeError("source must be a settings dict, task dict, manifest dict, or task list")
        _gallery_records(session, limit=500)
        record = jobs.submit(_resolve_generation_media(session, source, allow_read_file_system))
        if wait:
            record.job.result(timeout=timeout_s)
        return record.snapshot(event_limit=default_job_event_limit if event_limit is None else event_limit)

    @mcp.tool()
    def wangp_get_job(job_id: str, event_limit: int | None = None) -> dict[str, Any]:
        """Poll a WanGP generation job."""

        return jobs.get(job_id).snapshot(event_limit=default_job_event_limit if event_limit is None else event_limit)

    @mcp.tool()
    def wangp_cancel_job(job_id: str, event_limit: int | None = None) -> dict[str, Any]:
        """Request cancellation of a WanGP generation job."""

        record = jobs.get(job_id)
        record.job.cancel()
        return record.snapshot(event_limit=default_job_event_limit if event_limit is None else event_limit)

    from shared.utils.plugins import get_deepy_prime_plugin_tools

    for definition in get_deepy_prime_plugin_tools():
        if definition.requires_file_system and not allow_read_file_system:
            continue
        if mcp._tool_manager.get_tool(definition.name) is not None:
            raise RuntimeError(f"Deepy Prime plugin '{definition.plugin_id}' cannot replace built-in MCP tool '{definition.name}'.")
        mcp.tool(name=definition.name, title=definition.display_name, description=definition.description)(definition.function)

    return mcp


def build_inprocess_server(session, toolbox=None, default_job_event_limit: int = 20, allow_read_file_system: bool = False):
    return build_server_for_session(session, toolbox=toolbox, default_job_event_limit=default_job_event_limit, allow_read_file_system=allow_read_file_system)


def build_server(args: argparse.Namespace):
    from shared.api import init

    args.transport = _normalize_transport(getattr(args, "transport", "stdio"))
    console_output = bool(args.console_output) and args.transport != "stdio"
    with _stdio_safe_startup_output(args.transport):
        session = init(
            root=args.root,
            config_path=_config_file_from_arg(args.config),
            output_dir=args.output_dir,
            cli_args=tuple(args.cli_arg or ()),
            console_output=console_output,
            console_isatty=False,
        )
    settings: dict[str, Any] = {}
    if args.host is not None:
        settings["host"] = args.host
    if args.port is not None:
        settings["port"] = args.port
    if args.transport == "streamable-http":
        settings["json_response"] = True
        settings["stateless_http"] = True
    return build_server_for_session(session, settings, default_job_event_limit=getattr(args, "job_event_limit", 20), allow_read_file_system=getattr(args, "allow_read_file_system", False), http_media_transfer=args.transport != "stdio")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WanGP as an MCP server.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="WanGP repository root.")
    parser.add_argument("--config", default=None, help="Path to wgp_config.json or a directory containing it.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated media.")
    parser.add_argument("--cli-arg", action="append", default=[], help="Extra argument passed to wgp.py during runtime initialization. Repeat for multiple args.")
    parser.add_argument("--console-output", action="store_true", help="Mirror WanGP stdout/stderr to the MCP server console.")
    parser.add_argument("--transport", default="stdio", help="MCP transport: stdio, sse, or streamable-http.")
    parser.add_argument("--host", default=None, help="Optional host for non-stdio transports.")
    parser.add_argument("--port", type=int, default=None, help="Optional port for non-stdio transports.")
    parser.add_argument("--job-event-limit", type=int, default=20, help="Default number of recent progress events included in job snapshots; use 0 for terminal state/results only.")
    parser.add_argument("--allow-read-file-system", action="store_true", help="Allow MCP tool arguments to reference arbitrary server filesystem paths. Disabled by default; media IDs returned by wangp_list_gallery remain available.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        server = build_server(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return run_server(server, args)


def run_server(server, args: argparse.Namespace) -> int:
    transport = _normalize_transport(getattr(args, "transport", "stdio"))
    if transport == "stdio":
        server.run()
    else:
        server.run(transport=transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
