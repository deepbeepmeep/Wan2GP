from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from shared.utils.audio_video import extract_audio_tracks
from shared.utils.utils import get_video_frame, get_video_info
from shared.deepy.config import (
    DEEPY_CONTEXT_TOKENS_DEFAULT,
    DEEPY_CONTEXT_TOKENS_KEY,
    DEEPY_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_VRAM_ALWAYS,
    DEEPY_VRAM_UNLOAD,
    DEEPY_VRAM_UNLOAD_ON_REQUEST,
    get_deepy_config_value,
    normalize_deepy_context_tokens,
    normalize_deepy_custom_system_prompt,
    normalize_deepy_vram_mode,
)
from shared.deepy import DEFAULT_SYSTEM_PROMPT as ASSISTANT_SYSTEM_PROMPT
from shared.deepy import media_registry, tool_settings as deepy_tool_settings, ui_settings as deepy_ui_settings, video_tools as deepy_video_tools, vision as deepy_vision
from shared.gradio import assistant_chat
from shared.prompt_enhancer import qwen35_text
from shared.prompt_enhancer.qwen35_assistant_runtime import (
    Qwen35AssistantRuntime,
    extract_tool_calls,
    render_assistant_messages,
    render_text_user_turn_suffix,
    render_tool_turn_suffix,
    strip_inline_tool_call_text,
    strip_tool_blocks,
    strip_trailing_stop_markup,
)


ASSISTANT_DEBUG = False

_TOOL_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}
_AI_GEN_NO = 0
_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
_DEEPY_DOCS = {
    "finetunes": {"title": "Finetunes", "path": _DOCS_DIR / "FINETUNES.md"},
    "getting_started": {"title": "Getting Started", "path": _DOCS_DIR / "GETTING_STARTED.md"},
    "loras": {"title": "Loras", "path": _DOCS_DIR / "LORAS.md"},
    "overview": {"title": "Overview", "path": _DOCS_DIR / "OVERVIEW.md"},
    "prompts": {"title": "Prompts", "path": _DOCS_DIR / "PROMPTS.md"},
    "vace": {"title": "VACE", "path": _DOCS_DIR / "VACE.md"},
}
_SELECTED_REFERENCE_RE = re.compile(r"\b(selected|current(?:ly)?\s+selected|current\s+(?:item|media))\b", flags=re.IGNORECASE)
_POST_TRIM_WINDOW_FRACTION = 0.25


def set_assistant_debug(enabled: bool) -> None:
    global ASSISTANT_DEBUG
    ASSISTANT_DEBUG = bool(enabled)


def _json_type_from_annotation(annotation) -> str:
    annotation_name = getattr(annotation, "__name__", str(annotation))
    return _TOOL_TYPE_MAP.get(annotation_name, "string")


def assistant_tool(
    name: str | None = None,
    description: str = "",
    parameters: dict[str, dict[str, Any]] | None = None,
    display_name: str | None = None,
    pause_runtime: bool = True,
    pause_reason: str = "tool",
):
    def decorator(func):
        func._assistant_tool = {
            "name": str(name or func.__name__).strip(),
            "display_name": str(display_name or name or func.__name__).strip(),
            "description": str(description or "").strip(),
            "parameters": dict(parameters or {}),
            "pause_runtime": bool(pause_runtime),
            "pause_reason": str(pause_reason or "tool").strip() or "tool",
        }
        return func

    return decorator


@dataclass(slots=True)
class AssistantSessionState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    rendered_token_ids: list[int] = field(default_factory=list)
    rendered_messages_len: int = 0
    runtime_snapshot: dict[str, Any] | None = None
    media_registry: list[dict[str, Any]] = field(default_factory=list)
    media_registry_counter: int = 0
    chat_html: str = ""
    chat_transcript: list[dict[str, Any]] = field(default_factory=list)
    chat_transcript_counter: int = 0
    interrupt_requested: bool = False
    drop_state_requested: bool = False
    worker_active: bool = False
    control_queue: Any | None = None
    queued_job_count: int = 0
    chat_epoch: int = 0
    release_vram_callback: Callable[[], None] | None = None
    force_loading_status_once: bool = False
    current_turn: dict[str, Any] | None = None
    interruption_notice: str = ""
    runtime_status_note: str = ""
    runtime_status_signature: str = ""
    rendered_system_prompt_signature: str = ""
    rendered_context_window_tokens: int = 0
    pending_replay_reason: str = ""
    tool_ui_settings: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssistantRuntimeHooks:
    acquire_gpu: Callable[[], None]
    release_gpu: Callable[..., None]
    register_gpu_resident: Callable[[Callable[[], None] | None, bool], None]
    clear_gpu_resident: Callable[[], None]
    ensure_loaded: Callable[[], tuple[Any, Any]]
    unload_runtime: Callable[[], None]
    unload_weights: Callable[[], None]
    ensure_vision_loaded: Callable[[], tuple[Any, Any]] | None = None


def get_or_create_assistant_session(state) -> AssistantSessionState:
    session = state.get("assistant_session", None)
    if isinstance(session, AssistantSessionState):
        return session
    session = AssistantSessionState()
    state["assistant_session"] = session
    return session


def clear_assistant_session(session: AssistantSessionState) -> None:
    session.messages.clear()
    session.rendered_token_ids.clear()
    session.rendered_messages_len = 0
    session.runtime_snapshot = None
    session.media_registry.clear()
    session.media_registry_counter = 0
    session.chat_html = ""
    session.queued_job_count = 0
    session.release_vram_callback = None
    session.force_loading_status_once = False
    session.current_turn = None
    session.interruption_notice = ""
    session.runtime_status_note = ""
    session.runtime_status_signature = ""
    session.rendered_system_prompt_signature = ""
    session.rendered_context_window_tokens = 0
    session.pending_replay_reason = ""
    session.tool_ui_settings = {}
    assistant_chat.reset_session_chat(session)


def begin_assistant_turn(session: AssistantSessionState, user_message_id: str, user_text: str) -> None:
    session.current_turn = {
        "user_message_id": str(user_message_id or "").strip(),
        "user_text": str(user_text or "").strip(),
        "messages_len": len(session.messages),
        "rendered_token_ids": list(session.rendered_token_ids),
        "rendered_messages_len": int(session.rendered_messages_len or 0),
        "runtime_snapshot": session.runtime_snapshot,
        "rendered_system_prompt_signature": session.rendered_system_prompt_signature,
        "rendered_context_window_tokens": session.rendered_context_window_tokens,
        "assistant_message_id": "",
        "interrupt_recorded": False,
    }


def mark_assistant_turn_message(session: AssistantSessionState, message_id: str) -> None:
    checkpoint = session.current_turn
    if not isinstance(checkpoint, dict):
        return
    checkpoint["assistant_message_id"] = str(message_id or "").strip()


def _build_interruption_notice(user_text: str) -> str:
    collapsed = re.sub(r"\s+", " ", str(user_text or "").strip())
    if len(collapsed) > 280:
        collapsed = collapsed[:277].rstrip() + "..."
    if len(collapsed) == 0:
        return "The previous user request was interrupted by the user before completion. Do not continue that cancelled turn unless the user explicitly asks to resume it."
    return f"The previous user request was interrupted by the user before completion. Do not continue that cancelled turn unless the user explicitly asks to resume it. Cancelled request: {collapsed}"


def _describe_prefix_mismatch(current_token_ids: list[int], target_tokens: list[int]) -> str:
    current_len = len(current_token_ids)
    target_len = len(target_tokens)
    shared = min(current_len, target_len)
    mismatch_index = next((idx for idx, (current_token, target_token) in enumerate(zip(current_token_ids, target_tokens)) if int(current_token) != int(target_token)), shared)
    if mismatch_index >= shared:
        if current_len == target_len:
            return f"live sequence and canonicalized prompt had the same length ({current_len} tokens) but different token identity at the end"
        if current_len < target_len:
            return f"canonicalized prompt diverged right after the live prefix at token {mismatch_index} (live={current_len}, canonical={target_len})"
        return f"live runtime contained {current_len - target_len} extra trailing tokens beyond the canonicalized prompt (live={current_len}, canonical={target_len})"
    return f"live sequence diverged from canonicalized prompt at token {mismatch_index} (live={current_len}, canonical={target_len})"


def rollback_assistant_turn(session: AssistantSessionState, interrupted_badge: str = "Interrupted") -> bool:
    checkpoint = session.current_turn
    if not isinstance(checkpoint, dict):
        return False
    target_len = int(checkpoint.get("messages_len", len(session.messages)))
    if len(session.messages) > target_len:
        del session.messages[target_len:]
    session.rendered_token_ids = [int(token_id) for token_id in checkpoint.get("rendered_token_ids", []) or []]
    try:
        session.rendered_messages_len = int(checkpoint.get("rendered_messages_len", 0) or 0)
    except Exception:
        session.rendered_messages_len = 0
    session.runtime_snapshot = checkpoint.get("runtime_snapshot", None)
    session.rendered_system_prompt_signature = str(checkpoint.get("rendered_system_prompt_signature", "") or "")
    try:
        session.rendered_context_window_tokens = int(checkpoint.get("rendered_context_window_tokens", 0) or 0)
    except Exception:
        session.rendered_context_window_tokens = 0
    assistant_message_id = str(checkpoint.get("assistant_message_id", "") or "").strip()
    if len(assistant_message_id) > 0:
        assistant_chat.remove_message(session, assistant_message_id)
    user_message_id = str(checkpoint.get("user_message_id", "") or "").strip()
    if len(user_message_id) > 0:
        assistant_chat.set_message_badge(session, user_message_id, interrupted_badge)
    if not bool(checkpoint.get("interrupt_recorded", False)):
        session.interruption_notice = _build_interruption_notice(str(checkpoint.get("user_text", "") or ""))
        if ASSISTANT_DEBUG:
            print("[Assistant] Interruption notice computed:")
            print(session.interruption_notice)
        checkpoint["interrupt_recorded"] = True
    return True


def finish_assistant_turn(session: AssistantSessionState) -> None:
    session.current_turn = None


def request_assistant_interrupt(session: AssistantSessionState) -> None:
    session.interrupt_requested = True


def request_assistant_reset(session: AssistantSessionState) -> None:
    request_assistant_interrupt(session)
    session.drop_state_requested = True
    session.chat_epoch += 1
    session.queued_job_count = 0


def set_assistant_tool_ui_settings(session: AssistantSessionState, **kwargs) -> dict[str, Any]:
    normalized = deepy_ui_settings.normalize_assistant_tool_ui_settings(**kwargs)
    session.tool_ui_settings = dict(normalized)
    return session.tool_ui_settings


def _next_ai_client_id() -> str:
    global _AI_GEN_NO
    _AI_GEN_NO += 1
    return f"ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_AI_GEN_NO}"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _strip_partial_tool_markup(text: str) -> str:
    stripped = strip_trailing_stop_markup(str(text or ""))
    lowered = stripped.lower()
    cut_points = []
    for marker in ("<tool_call>", "<function=", "<function ", '{"name"', "{'name'"):
        idx = lowered.find(marker)
        if idx >= 0:
            cut_points.append(idx)
    if cut_points:
        stripped = stripped[: min(cut_points)]
    return stripped.rstrip()


class tools:
    def __init__(self, gen, get_processed_queue, send_cmd, session: AssistantSessionState | None = None, get_output_filepath: Callable[[str, bool, bool], str] | None = None, record_file_metadata: Callable[..., None] | None = None, get_server_config: Callable[[], dict[str, Any]] | None = None):
        self.gen = gen
        self.get_processed_queue = get_processed_queue
        self.send_cmd = send_cmd
        self.session = session
        self.get_output_filepath = get_output_filepath
        self.record_file_metadata = record_file_metadata
        self.get_server_config = get_server_config
        self._vision_query_callback: Callable[[dict[str, Any], str], dict[str, Any]] | None = None
        self._tool_progress_callback: Callable[..., None] | None = None

    def _log(self, message: str) -> None:
        if ASSISTANT_DEBUG:
            print(f"[AssistantTool] {message}")

    def _is_interrupted(self) -> bool:
        return self.session is not None and self.session.interrupt_requested

    def _interrupted_result(self, client_id: str, task: dict[str, Any]) -> dict[str, Any]:
        self._log(f"Generation interrupted for {client_id}")
        result = {
            "status": "interrupted",
            "client_id": client_id,
            "output_file": "",
            "prompt": task["prompt"],
            "resolution": task["resolution"],
            "error": "Interrupted by reset request.",
        }
        self._update_tool_progress("error", "Interrupted", result)
        return result

    def _set_status(self, text: str | None, kind: str = "working") -> None:
        self.send_cmd("chat_output", assistant_chat.build_status_event(text, kind=kind, visible=text is not None and len(str(text).strip()) > 0))

    def bind_runtime_tools(self, vision_query_callback: Callable[[dict[str, Any], str], dict[str, Any]] | None = None, tool_progress_callback: Callable[..., None] | None = None) -> None:
        self._vision_query_callback = vision_query_callback
        self._tool_progress_callback = tool_progress_callback

    def _update_tool_progress(self, status: str | None = None, status_text: str | None = None, result: dict[str, Any] | None = None) -> None:
        if callable(self._tool_progress_callback):
            self._tool_progress_callback(status=status, status_text=status_text, result=result)

    def _get_tool_ui_settings(self) -> dict[str, Any]:
        if self.session is not None and isinstance(self.session.tool_ui_settings, dict) and len(self.session.tool_ui_settings) > 0:
            return deepy_ui_settings.normalize_assistant_tool_ui_settings(**self.session.tool_ui_settings)
        return deepy_ui_settings.normalize_assistant_tool_ui_settings()

    def get_tool_template_filename(self, tool_name: str) -> str:
        setting_key = {
            "gen_image": "image_generator_variant",
            "edit_image": "image_editor_variant",
            "gen_video": "video_generator_variant",
        }.get(str(tool_name or "").strip(), "")
        if len(setting_key) == 0:
            return ""
        variant = str(self._get_tool_ui_settings().get(setting_key, "") or "").strip()
        if len(variant) == 0:
            return ""
        template_name = Path(variant).name
        if len(template_name) == 0:
            return ""
        if template_name.lower().endswith(".json"):
            return template_name
        return f"{template_name}.json"

    def get_tool_transcript_label(self, tool_name: str) -> str:
        label = self.get_tool_display_name(tool_name)
        if str(tool_name or "").strip() not in {"gen_image", "edit_image", "gen_video"}:
            return label
        template_label = Path(self.get_tool_template_filename(tool_name)).stem.strip()
        return label if len(template_label) == 0 else f"{label} [{template_label}]"

    def _apply_generation_overrides(self, task: dict[str, Any], *, include_num_frames: bool) -> dict[str, Any]:
        ui_settings = self._get_tool_ui_settings()
        if ui_settings["use_template_properties"]:
            return task
        task["resolution"] = f"{ui_settings['width']}x{ui_settings['height']}"
        if include_num_frames:
            task["video_length"] = int(ui_settings["num_frames"])
        return task

    def _sync_recent_media(self, max_items: int = 5) -> None:
        if self.session is None:
            return
        file_list, file_settings_list, audio_file_list, audio_file_settings_list = self.get_processed_queue(self.gen)
        media_registry.sync_recent_generated_media(self.session, file_list, file_settings_list, max_items=max_items)
        media_registry.sync_recent_generated_media(self.session, audio_file_list, audio_file_settings_list, max_items=max_items)

    def _compact_media_payload(self, record: dict[str, Any], why: str = "") -> dict[str, Any]:
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

    def _get_selected_runtime_snapshot(self) -> dict[str, Any] | None:
        media_record, error_result = self._get_selected_media_record("any")
        if error_result is not None or media_record is None:
            return None
        snapshot = {
            "selected_media_id": str(media_record.get("media_id", "") or "").strip(),
            "selected_media_type": str(media_record.get("media_type", "") or "").strip(),
        }
        label = str(media_record.get("label", "") or "").strip()
        if len(label) > 0:
            snapshot["selected_media_label"] = label
        if snapshot["selected_media_type"] == "video":
            snapshot.update(self._get_selected_video_position(media_record))
        return snapshot

    def _is_selected_reference(self, reference: str) -> bool:
        return _SELECTED_REFERENCE_RE.search(str(reference or "").strip()) is not None

    def _get_selected_media_record(self, requested_media_type: str = "any") -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if self.session is None:
            return None, {"status": "error", "media_type": str(requested_media_type or "any").strip() or "any", "error": "Assistant session is not available."}
        file_list, file_settings_list, audio_file_list, audio_file_settings_list = self.get_processed_queue(self.gen)
        source = str((self.gen or {}).get("current_gallery_source", "video") or "video").strip().lower()
        if source == "audio":
            raw_choice = (self.gen or {}).get("audio_selected", -1)
            file_list, file_settings_list = list(audio_file_list or []), list(audio_file_settings_list or [])
        else:
            raw_choice = (self.gen or {}).get("selected", -1)
            file_list, file_settings_list = list(file_list or []), list(file_settings_list or [])
        try:
            choice = int(raw_choice if raw_choice is not None else -1)
        except Exception:
            choice = -1
        if choice < 0 or choice >= len(file_list):
            return None, {"status": "error", "media_type": str(requested_media_type or "any").strip() or "any", "error": "No media is currently selected in WanGP galleries."}
        selected_path = str(file_list[choice] or "").strip()
        selected_settings = file_settings_list[choice] if choice < len(file_settings_list) and isinstance(file_settings_list[choice], dict) else None
        media_record = media_registry.register_media(
            self.session,
            selected_path,
            settings=selected_settings,
            source="deepy" if str((selected_settings or {}).get("client_id", "") or "").strip().startswith("ai_") else "wangp",
            client_id=str((selected_settings or {}).get("client_id", "") or "").strip(),
        )
        if media_record is None:
            return None, {"status": "error", "media_type": str(requested_media_type or "any").strip() or "any", "error": "The currently selected gallery item is not a supported media file."}
        actual_media_type = str(media_record.get("media_type", "") or "").strip() or "unknown media type"
        resolved_media_type = media_registry.normalize_media_type(requested_media_type)
        if resolved_media_type != "any" and actual_media_type != resolved_media_type:
            return None, {
                "status": "error",
                "media_type": resolved_media_type,
                "selected_media_type": actual_media_type,
                "actual_media_type": actual_media_type,
                "error": f"The currently selected media is a {actual_media_type}, not a {resolved_media_type}.",
            }
        return media_record, None

    def _get_selected_video_position(self, media_record: dict[str, Any]) -> dict[str, Any]:
        if str(media_record.get("media_type", "") or "").strip() != "video":
            return {}
        try:
            current_time = float((self.gen or {}).get("selected_video_time", 0.0) or 0.0)
        except Exception:
            current_time = 0.0
        current_time = max(0.0, current_time)
        try:
            media_path = str(media_record.get("path", "")).strip()
            _fps, _width, _height, _frame_count = get_video_info(media_path)
        except Exception:
            media_path = ""
        try:
            frame_no = deepy_video_tools.resolve_video_frame_no(media_path, time_seconds=current_time) if len(media_path) > 0 else 0
        except Exception:
            frame_no = 0
        return {"current_time_seconds": round(current_time, 3), "current_frame_no": frame_no}

    def _register_tool_media(self, path: str, settings: dict[str, Any], label: str | None = None) -> dict[str, Any] | None:
        if self.session is None:
            return None
        return media_registry.register_media(
            self.session,
            path,
            settings=settings,
            source="deepy",
            client_id=str(settings.get("client_id", "") or "").strip(),
            label=label,
        )

    def _resolve_direct_output_path(self, file_path: str, is_image: bool, audio_only: bool) -> str:
        file_path = str(file_path or "").strip()
        if len(file_path) == 0:
            raise RuntimeError("Output file path is empty.")
        if callable(self.get_output_filepath):
            resolved = str(self.get_output_filepath(file_path, is_image, audio_only) or "").strip()
            if len(resolved) > 0:
                return resolved
        return os.path.abspath(os.path.normpath(file_path))

    def _record_direct_media(self, output_path: str, settings: dict[str, Any], *, is_image: bool, audio_only: bool, label: str | None = None) -> dict[str, Any] | None:
        if not os.path.isfile(output_path):
            raise RuntimeError(f"Output file was not created: {output_path}")
        if not callable(self.record_file_metadata):
            raise RuntimeError("WanGP direct media recording is not available.")
        self.record_file_metadata(output_path, settings, is_image, audio_only, self.gen)
        self.send_cmd("refresh_gallery", {"path": output_path})
        return self._register_tool_media(output_path, settings, label=label)

    def _server_config(self) -> dict[str, Any]:
        if callable(self.get_server_config):
            return dict(self.get_server_config() or {})
        return {}

    def _get_video_output_settings(self) -> tuple[str, str]:
        server_config = self._server_config()
        return str(server_config.get("video_output_codec", "libx264_8") or "libx264_8"), str(server_config.get("video_container", "mp4") or "mp4")

    def _get_standalone_audio_output_codec(self) -> str:
        server_config = self._server_config()
        return str(server_config.get("audio_stand_alone_output_codec", "wav") or "wav")

    def _get_video_audio_output_codec(self) -> str:
        server_config = self._server_config()
        return str(server_config.get("audio_output_codec", "aac_128") or "aac_128")

    def _resolve_image_media(self, media_id: str, parameter_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        media_id = str(media_id or "").strip()
        if len(media_id) == 0:
            return None, None
        if self.session is None:
            return None, {"status": "error", parameter_name: media_id, "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return None, {"status": "error", parameter_name: media_id, "error": f"Unknown media id for {parameter_name}."}
        if media_record.get("media_type") != "image":
            actual_media_type = str(media_record.get("media_type", "") or "").strip() or "unknown media type"
            return None, {
                "status": "error",
                parameter_name: media_record.get("media_id", ""),
                "actual_media_type": actual_media_type,
                "media_type": actual_media_type,
                "error": f"{parameter_name} must reference an image, not a {actual_media_type}.",
            }
        return media_record, None

    def _resolve_video_media(self, media_id: str, parameter_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        media_id = str(media_id or "").strip()
        if len(media_id) == 0:
            return None, {"status": "error", parameter_name: media_id, "error": f"{parameter_name} is required."}
        if self.session is None:
            return None, {"status": "error", parameter_name: media_id, "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return None, {"status": "error", parameter_name: media_id, "error": f"Unknown media id for {parameter_name}."}
        if media_record.get("media_type") != "video":
            actual_media_type = str(media_record.get("media_type", "") or "").strip() or "unknown media type"
            return None, {
                "status": "error",
                parameter_name: media_record.get("media_id", ""),
                "actual_media_type": actual_media_type,
                "media_type": actual_media_type,
                "error": f"{parameter_name} must reference a video, not a {actual_media_type}.",
            }
        return media_record, None

    def _resolve_audio_media(self, media_id: str, parameter_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        media_id = str(media_id or "").strip()
        if len(media_id) == 0:
            return None, {"status": "error", parameter_name: media_id, "error": f"{parameter_name} is required."}
        if self.session is None:
            return None, {"status": "error", parameter_name: media_id, "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return None, {"status": "error", parameter_name: media_id, "error": f"Unknown media id for {parameter_name}."}
        if media_record.get("media_type") != "audio":
            actual_media_type = str(media_record.get("media_type", "") or "").strip() or "unknown media type"
            return None, {
                "status": "error",
                parameter_name: media_record.get("media_id", ""),
                "actual_media_type": actual_media_type,
                "media_type": actual_media_type,
                "error": f"{parameter_name} must reference an audio file, not a {actual_media_type}.",
            }
        return media_record, None

    def _parse_time_value(self, value: Any, parameter_name: str, *, required: bool = False) -> tuple[float | None, dict[str, Any] | None]:
        if value is None or str(value).strip() == "":
            return (None, {"status": "error", "error": f"{parameter_name} is required."}) if required else (None, None)
        try:
            resolved = float(value)
        except Exception:
            return None, {"status": "error", "error": f"{parameter_name} must be a number."}
        if resolved < 0:
            return None, {"status": "error", "error": f"{parameter_name} must be >= 0."}
        return resolved, None

    def _build_direct_media_settings(self, source_media: dict[str, Any], comments: str, **updates: Any) -> dict[str, Any]:
        settings = dict(source_media.get("settings", {}) or {})
        settings["client_id"] = _next_ai_client_id()
        settings["comments"] = str(comments or "").strip()
        end_time = time.time()
        settings["creation_date"] = datetime.fromtimestamp(end_time).isoformat(timespec="seconds")
        settings["creation_timestamp"] = int(end_time)
        for key, value in updates.items():
            if value is not None:
                settings[key] = value
        return settings

    def _update_video_metadata_fields(self, output_path: str, settings: dict[str, Any]) -> None:
        try:
            fps, width, height, frames_count = get_video_info(output_path)
            settings["resolution"] = f"{width}x{height}"
            settings["video_length"] = int(frames_count)
            if fps > 0:
                settings["duration_seconds"] = round(frames_count / fps, 3)
        except Exception:
            pass

    def _update_audio_metadata_fields(self, output_path: str, settings: dict[str, Any]) -> None:
        duration = deepy_video_tools.get_media_duration(output_path)
        if duration is not None:
            settings["duration_seconds"] = round(duration, 3)

    def _queue_generation_task(self, task: dict[str, Any], *, activity_label: str, output_label: str | None = None) -> dict[str, Any]:
        if not isinstance(self.gen, dict):
            raise RuntimeError("WanGP generation queue is not available.")
        client_id = str(task.get("client_id", "") or "").strip()
        prompt = str(task.get("prompt", "") or "").strip()
        resolution = str(task.get("resolution", "") or "").strip()
        gen = self.gen
        self.get_processed_queue(gen)
        self._set_status(f"Queueing {activity_label}...", kind="tool")
        self._update_tool_progress("running", "Queued", {"status": "queued", "client_id": client_id, "prompt": prompt, "resolution": resolution})
        if self._get_tool_ui_settings().get("priority", False):
            task["priority"] = True
        else:
            task.pop("priority", None)
        gen["inline_queue"] = task
        self.send_cmd("load_queue_trigger", {"client_id": client_id})
        self._log(f"Queued {activity_label} for {client_id}")

        queue_detect_deadline = time.time() + 5
        while time.time() < queue_detect_deadline:
            if self._is_interrupted():
                return self._interrupted_result(client_id, task)
            queue_errors = gen.get("queue_errors", None) or {}
            if client_id in queue_errors:
                error_text = str(queue_errors[client_id][0])
                self._log(f"Queue error detected for {client_id}: {error_text}")
                self._set_status(f"{activity_label.capitalize()} failed: {error_text}", kind="error")
                result = {
                    "status": "error",
                    "client_id": client_id,
                    "output_file": "",
                    "prompt": prompt,
                    "resolution": resolution,
                    "error": error_text,
                }
                self._update_tool_progress("error", "Error", result)
                return result
            queue = gen.get("queue", []) or []
            if any(isinstance(item, dict) and isinstance(item.get("params"), dict) and item["params"].get("client_id") == client_id for item in queue):
                self._set_status(f"{activity_label.capitalize()} started...", kind="tool")
                self._update_tool_progress("running", "Running", {"status": "running", "client_id": client_id, "prompt": prompt, "resolution": resolution})
                break
            time.sleep(0.25)

        while True:
            if self._is_interrupted():
                return self._interrupted_result(client_id, task)
            queue_errors = gen.get("queue_errors", None) or {}
            if client_id in queue_errors:
                error_text = str(queue_errors[client_id][0])
                self._log(f"Generation error detected for {client_id}: {error_text}")
                self._set_status(f"{activity_label.capitalize()} failed: {error_text}", kind="error")
                result = {
                    "status": "error",
                    "client_id": client_id,
                    "output_file": "",
                    "prompt": prompt,
                    "resolution": resolution,
                    "error": error_text,
                }
                self._update_tool_progress("error", "Error", result)
                return result
            file_list, file_settings_list, _audio_file_list, _audio_file_settings_list = self.get_processed_queue(gen)
            for file_path, file_settings in zip(file_list, file_settings_list):
                if not isinstance(file_settings, dict):
                    continue
                if file_settings.get("client_id", "") != client_id:
                    continue
                media_record = self._register_tool_media(str(file_path), file_settings, label=output_label)
                result = {
                    "status": "done",
                    "client_id": client_id,
                    "output_file": str(file_path),
                    "media_id": "" if media_record is None else media_record.get("media_id", ""),
                    "prompt": prompt,
                    "resolution": resolution,
                    "error": "",
                }
                self._log(f"{activity_label.capitalize()} completed for {client_id}: {file_path}")
                self._set_status(f"{activity_label.capitalize()} finished.", kind="tool")
                self.send_cmd("refresh_gallery", {"path": str(file_path)})
                self._update_tool_progress("done", "Done", result)
                return result
            time.sleep(0.5)

    @assistant_tool(
        display_name="Generate Image",
        description="Queue and generate an image from a text prompt inside WanGP, then wait until the output image is available.",
        parameters={
            "prompt": {
                "type": "string",
                "description": "The image generation prompt to send to WanGP.",
            }
        },
    )
    def gen_image(self, prompt: str) -> dict[str, Any]:
        client_id = _next_ai_client_id()
        generator_variant = self._get_tool_ui_settings()["image_generator_variant"]
        template_file = self.get_tool_template_filename("gen_image")
        task = deepy_tool_settings.build_generation_task("gen_image", generator_variant, prompt=prompt, client_id=client_id)
        task = self._apply_generation_overrides(task, include_num_frames=False)
        if len(task["prompt"]) == 0:
            self._set_status("Image generation failed: prompt is empty.", kind="error")
            return {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": "",
                "resolution": task["resolution"],
                "error": "Prompt is empty.",
            }
        result = self._queue_generation_task(task, activity_label="image generation", output_label="Generated image")
        result["generator_variant"] = generator_variant
        if len(template_file) > 0:
            result["template_file"] = template_file
        return result

    @assistant_tool(
        display_name="Generate Video",
        description="Queue and generate a video from a text prompt inside WanGP, optionally using a start image and an end image, then wait until the output video is available.",
        parameters={
            "prompt": {
                "type": "string",
                "description": "The video generation prompt to send to WanGP.",
            },
            "image_start": {
                "type": "string",
                "description": "Optional media id of the start image returned by Resolve Media.",
                "required": False,
            },
            "image_end": {
                "type": "string",
                "description": "Optional media id of the end image returned by Resolve Media.",
                "required": False,
            },
        },
    )
    def gen_video(self, prompt: str, image_start: str | None = None, image_end: str | None = None) -> dict[str, Any]:
        self._sync_recent_media()
        start_media, error_result = self._resolve_image_media(image_start or "", "image_start")
        if error_result is not None:
            error_result.update({"prompt": str(prompt or "").strip(), "output_file": ""})
            return error_result
        end_media, error_result = self._resolve_image_media(image_end or "", "image_end")
        if error_result is not None:
            error_result.update({"prompt": str(prompt or "").strip(), "output_file": ""})
            return error_result
        client_id = _next_ai_client_id()
        generator_variant = self._get_tool_ui_settings()["video_generator_variant"]
        template_file = self.get_tool_template_filename("gen_video")
        task = deepy_tool_settings.build_generation_task(
            "gen_video",
            generator_variant,
            prompt=prompt,
            client_id=client_id,
            image_start=None if start_media is None else str(start_media.get("path", "")).strip(),
            image_end=None if end_media is None else str(end_media.get("path", "")).strip(),
        )
        task = self._apply_generation_overrides(task, include_num_frames=True)
        if len(task["prompt"]) == 0:
            self._set_status("Video generation failed: prompt is empty.", kind="error")
            return {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": "",
                "resolution": task.get("resolution", ""),
                "error": "Prompt is empty.",
            }
        result = self._queue_generation_task(task, activity_label="video generation", output_label="Generated video")
        result["generator_variant"] = generator_variant
        if len(template_file) > 0:
            result["template_file"] = template_file
        if start_media is not None:
            result["source_start_media_id"] = start_media.get("media_id", "")
        if end_media is not None:
            result["source_end_media_id"] = end_media.get("media_id", "")
        return result

    @assistant_tool(
        display_name="Edit Image",
        description="Edit a previously resolved image using an instruction prompt inside WanGP and wait until the edited image is available.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id returned by Resolve Media.",
            },
            "prompt": {
                "type": "string",
                "description": "The instruction prompt describing how to modify the image.",
            },
        },
    )
    def edit_image(self, media_id: str, prompt: str) -> dict[str, Any]:
        self._sync_recent_media()
        if self.session is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "prompt": str(prompt or "").strip(), "output_file": "", "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "prompt": str(prompt or "").strip(), "output_file": "", "error": "Unknown media id."}
        if media_record.get("media_type") != "image":
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "media_type": media_record.get("media_type", ""),
                "prompt": str(prompt or "").strip(),
                "output_file": "",
                "error": "Edit Image currently supports images only.",
            }
        editor_variant = self._get_tool_ui_settings()["image_editor_variant"]
        template_file = self.get_tool_template_filename("edit_image")
        client_id = _next_ai_client_id()
        task = deepy_tool_settings.build_generation_task(
            "edit_image",
            editor_variant,
            prompt=prompt,
            client_id=client_id,
            image_refs=[str(media_record.get("path", "")).strip()],
        )
        task = self._apply_generation_overrides(task, include_num_frames=False)
        if len(task["prompt"]) == 0:
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "prompt": "",
                "output_file": "",
                "error": "Prompt is empty.",
            }
        result = self._queue_generation_task(task, activity_label="image editing", output_label="Edited image")
        result["editor_variant"] = editor_variant
        if len(template_file) > 0:
            result["template_file"] = template_file
        result["source_media_id"] = media_record.get("media_id", "")
        return result

    @assistant_tool(
        display_name="Extract Image",
        description="Extract one image from a previously resolved video at a specific frame number or exact playback time and add it to WanGP galleries.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id for the source video returned by Resolve Media.",
            },
            "frame_no": {
                "type": "integer",
                "description": "Optional frame number to extract from the source video.",
                "required": False,
            },
            "time_seconds": {
                "type": "number",
                "description": "Optional exact playback time in seconds. Prefer this for the currently selected video frame because it matches the player position more accurately.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def extract_image(self, media_id: str, frame_no: int | None = None, time_seconds: float | None = None) -> dict[str, Any]:
        self._sync_recent_media()
        source_media, error_result = self._resolve_video_media(media_id, "media_id")
        if error_result is not None:
            return error_result
        try:
            frame_no = None if frame_no is None or str(frame_no).strip() == "" else int(frame_no)
        except Exception:
            return {"status": "error", "media_id": str(media_id or "").strip(), "frame_no": frame_no, "output_file": "", "error": "frame_no must be an integer."}
        time_seconds, error_result = self._parse_time_value(time_seconds, "time_seconds", required=False)
        if error_result is not None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "frame_no": frame_no, "time_seconds": time_seconds, "output_file": "", "error": error_result["error"]}
        if frame_no is None and time_seconds is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "frame_no": None, "time_seconds": None, "output_file": "", "error": "frame_no or time_seconds is required."}
        self._set_status("Extracting image...", kind="tool")
        self._update_tool_progress("running", "Extracting", {"status": "running", "media_id": source_media.get("media_id", ""), "frame_no": frame_no, "time_seconds": time_seconds})
        source_path = str(source_media.get("path", "")).strip()
        try:
            resolved_frame_no = deepy_video_tools.resolve_video_frame_no(source_path, frame_no=frame_no, time_seconds=time_seconds)
        except Exception as exc:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "frame_no": frame_no, "time_seconds": time_seconds, "output_file": "", "error": str(exc)}
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        output_suffix = f"frame{resolved_frame_no}" if time_seconds is None else f"frame{resolved_frame_no}_t{int(round(float(time_seconds or 0.0) * 1000.0))}ms"
        output_name = f"{source_name}_{output_suffix}.png"
        output_path = self._resolve_direct_output_path(output_name, True, False)
        try:
            deepy_video_tools.extract_video_frame(source_path, output_path, frame_no=frame_no, time_seconds=time_seconds)
        except Exception as exc:
            result = {
                "status": "error",
                "media_id": source_media.get("media_id", ""),
                "frame_no": resolved_frame_no,
                "time_seconds": time_seconds,
                "output_file": "",
                "error": str(exc),
            }
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Image extraction failed: {exc}", kind="error")
            return result
        comments = f'Extracted frame {resolved_frame_no} from "{os.path.basename(source_path)}"' if time_seconds is None else f'Extracted frame {resolved_frame_no} at {time_seconds:.3f}s from "{os.path.basename(source_path)}"'
        extracted_settings = self._build_direct_media_settings(source_media, comments)
        media_record = self._record_direct_media(output_path, extracted_settings, is_image=True, audio_only=False, label="Extracted image")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "source_media_id": source_media.get("media_id", ""),
            "frame_no": resolved_frame_no,
            "time_seconds": time_seconds,
            "output_file": output_path,
            "error": "",
        }
        self._update_tool_progress("done", "Done", result)
        self._set_status("Image extracted.", kind="tool")
        return result

    @assistant_tool(
        display_name="Extract Video",
        description="Extract a video segment from a previously resolved video using start_time and either end_time or duration.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id for the source video returned by Resolve Media.",
            },
            "start_time": {
                "type": "number",
                "description": "Start time in seconds.",
            },
            "end_time": {
                "type": "number",
                "description": "Optional end time in seconds.",
                "required": False,
            },
            "duration": {
                "type": "number",
                "description": "Optional segment duration in seconds.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def extract_video(self, media_id: str, start_time: float, end_time: float | None = None, duration: float | None = None) -> dict[str, Any]:
        self._sync_recent_media()
        source_media, error_result = self._resolve_video_media(media_id, "media_id")
        if error_result is not None:
            return error_result
        start_time, error_result = self._parse_time_value(start_time, "start_time", required=True)
        if error_result is not None:
            error_result.update({"media_id": source_media.get("media_id", ""), "output_file": ""})
            return error_result
        end_time, error_result = self._parse_time_value(end_time, "end_time")
        if error_result is not None:
            error_result.update({"media_id": source_media.get("media_id", ""), "output_file": ""})
            return error_result
        duration, error_result = self._parse_time_value(duration, "duration")
        if error_result is not None:
            error_result.update({"media_id": source_media.get("media_id", ""), "output_file": ""})
            return error_result
        if end_time is not None and duration is not None:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": "Specify either end_time or duration, not both."}
        self._set_status("Extracting video...", kind="tool")
        self._update_tool_progress("running", "Extracting", {"status": "running", "media_id": source_media.get("media_id", ""), "start_time": start_time, "end_time": end_time, "duration": duration})
        source_path = str(source_media.get("path", "")).strip()
        video_codec, video_container = self._get_video_output_settings()
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        output_path = self._resolve_direct_output_path(f"{base_name}_clip{deepy_video_tools.get_video_container_extension(video_container)}", False, False)
        try:
            output_path = deepy_video_tools.extract_video(source_path, output_path, start_time=start_time, end_time=end_time, duration=duration, video_codec=video_codec, video_container=video_container, audio_codec=self._get_video_audio_output_codec())
        except Exception as exc:
            result = {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Video extraction failed: {exc}", kind="error")
            return result
        comments = f'Extracted video segment from "{os.path.basename(source_path)}" starting at {start_time}s'
        if end_time is not None:
            comments += f" ending at {end_time}s"
        elif duration is not None:
            comments += f" with duration {duration}s"
        extracted_settings = self._build_direct_media_settings(source_media, comments)
        self._update_video_metadata_fields(output_path, extracted_settings)
        media_record = self._record_direct_media(output_path, extracted_settings, is_image=False, audio_only=False, label="Extracted video")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "source_media_id": source_media.get("media_id", ""),
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "output_file": output_path,
            "error": "",
        }
        self._update_tool_progress("done", "Done", result)
        self._set_status("Video extracted.", kind="tool")
        return result

    @assistant_tool(
        display_name="Extract Audio",
        description="Extract audio from a previously resolved video or clip an existing audio file using optional start_time, end_time, or duration.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id for the source video or audio returned by Resolve Media.",
            },
            "start_time": {
                "type": "number",
                "description": "Optional start time in seconds. Defaults to the beginning.",
                "required": False,
            },
            "end_time": {
                "type": "number",
                "description": "Optional end time in seconds.",
                "required": False,
            },
            "duration": {
                "type": "number",
                "description": "Optional segment duration in seconds.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def extract_audio(self, media_id: str, start_time: float | None = None, end_time: float | None = None, duration: float | None = None) -> dict[str, Any]:
        self._sync_recent_media()
        if self.session is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "output_file": "", "error": "Assistant session is not available."}
        source_media = media_registry.get_media_record(self.session, media_id)
        if source_media is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "output_file": "", "error": "Unknown media id."}
        if source_media.get("media_type") not in {"audio", "video"}:
            actual_media_type = str(source_media.get("media_type", "") or "").strip() or "unknown media type"
            return {"status": "error", "media_id": source_media.get("media_id", ""), "actual_media_type": actual_media_type, "media_type": actual_media_type, "output_file": "", "error": f"media_id must reference audio or video, not a {actual_media_type}."}
        start_time, error_result = self._parse_time_value(start_time, "start_time")
        if error_result is not None:
            error_result.update({"media_id": source_media.get("media_id", ""), "output_file": ""})
            return error_result
        end_time, error_result = self._parse_time_value(end_time, "end_time")
        if error_result is not None:
            error_result.update({"media_id": source_media.get("media_id", ""), "output_file": ""})
            return error_result
        duration, error_result = self._parse_time_value(duration, "duration")
        if error_result is not None:
            error_result.update({"media_id": source_media.get("media_id", ""), "output_file": ""})
            return error_result
        if end_time is not None and duration is not None:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": "Specify either end_time or duration, not both."}
        self._set_status("Extracting audio...", kind="tool")
        self._update_tool_progress("running", "Extracting", {"status": "running", "media_id": source_media.get("media_id", ""), "start_time": start_time, "end_time": end_time, "duration": duration})
        source_path = str(source_media.get("path", "")).strip()
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        audio_codec = self._get_standalone_audio_output_codec()
        output_path = self._resolve_direct_output_path(f"{base_name}_audio{deepy_video_tools.get_audio_standalone_extension(audio_codec)}", False, True)
        try:
            output_path = deepy_video_tools.extract_audio(source_path, output_path, start_time=start_time, end_time=end_time, duration=duration, audio_codec=audio_codec)
        except Exception as exc:
            result = {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Audio extraction failed: {exc}", kind="error")
            return result
        comments = f'Extracted audio from "{os.path.basename(source_path)}"'
        if start_time is not None:
            comments += f" starting at {start_time}s"
        if end_time is not None:
            comments += f" ending at {end_time}s"
        elif duration is not None:
            comments += f" with duration {duration}s"
        extracted_settings = self._build_direct_media_settings(source_media, comments)
        self._update_audio_metadata_fields(output_path, extracted_settings)
        media_record = self._record_direct_media(output_path, extracted_settings, is_image=False, audio_only=True, label="Extracted audio")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "source_media_id": source_media.get("media_id", ""),
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "output_file": output_path,
            "error": "",
        }
        self._update_tool_progress("done", "Done", result)
        self._set_status("Audio extracted.", kind="tool")
        return result

    @assistant_tool(
        display_name="Mute Video",
        description="Create a copy of a previously resolved video with all audio removed.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id for the source video returned by Resolve Media.",
            },
        },
        pause_runtime=False,
    )
    def mute_video(self, media_id: str) -> dict[str, Any]:
        self._sync_recent_media()
        source_media, error_result = self._resolve_video_media(media_id, "media_id")
        if error_result is not None:
            return error_result
        self._set_status("Muting video...", kind="tool")
        self._update_tool_progress("running", "Muting", {"status": "running", "media_id": source_media.get("media_id", "")})
        source_path = str(source_media.get("path", "")).strip()
        _video_codec, video_container = self._get_video_output_settings()
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        output_path = self._resolve_direct_output_path(f"{base_name}_muted{deepy_video_tools.get_video_container_extension(video_container)}", False, False)
        try:
            output_path = deepy_video_tools.mute_video(source_path, output_path)
        except Exception as exc:
            result = {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Video muting failed: {exc}", kind="error")
            return result
        muted_settings = self._build_direct_media_settings(source_media, f'Removed audio from "{os.path.basename(source_path)}"')
        self._update_video_metadata_fields(output_path, muted_settings)
        media_record = self._record_direct_media(output_path, muted_settings, is_image=False, audio_only=False, label="Muted video")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "source_media_id": source_media.get("media_id", ""),
            "output_file": output_path,
            "error": "",
        }
        self._update_tool_progress("done", "Done", result)
        self._set_status("Video muted.", kind="tool")
        return result

    @assistant_tool(
        display_name="Replace Audio",
        description="Replace the soundtrack of a previously resolved video with a previously resolved audio file.",
        parameters={
            "video_id": {
                "type": "string",
                "description": "The media id for the source video returned by Resolve Media.",
            },
            "audio_id": {
                "type": "string",
                "description": "The media id for the replacement audio returned by Resolve Media.",
            },
        },
        pause_runtime=False,
    )
    def replace_audio(self, video_id: str, audio_id: str) -> dict[str, Any]:
        self._sync_recent_media()
        video_media, error_result = self._resolve_video_media(video_id, "video_id")
        if error_result is not None:
            return error_result
        audio_media, error_result = self._resolve_audio_media(audio_id, "audio_id")
        if error_result is not None:
            return error_result
        self._set_status("Replacing video audio...", kind="tool")
        self._update_tool_progress("running", "Replacing", {"status": "running", "video_id": video_media.get("media_id", ""), "audio_id": audio_media.get("media_id", "")})
        video_path = str(video_media.get("path", "")).strip()
        audio_path = str(audio_media.get("path", "")).strip()
        _video_codec, video_container = self._get_video_output_settings()
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = self._resolve_direct_output_path(f"{base_name}_audio_replaced{deepy_video_tools.get_video_container_extension(video_container)}", False, False)
        try:
            output_path = deepy_video_tools.replace_audio(video_path, audio_path, output_path, audio_codec=self._get_video_audio_output_codec())
        except Exception as exc:
            result = {"status": "error", "video_id": video_media.get("media_id", ""), "audio_id": audio_media.get("media_id", ""), "output_file": "", "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Audio replacement failed: {exc}", kind="error")
            return result
        replaced_settings = self._build_direct_media_settings(video_media, f'Replaced audio of "{os.path.basename(video_path)}" with "{os.path.basename(audio_path)}"')
        self._update_video_metadata_fields(output_path, replaced_settings)
        media_record = self._record_direct_media(output_path, replaced_settings, is_image=False, audio_only=False, label="Video with replaced audio")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "source_video_id": video_media.get("media_id", ""),
            "source_audio_id": audio_media.get("media_id", ""),
            "output_file": output_path,
            "error": "",
        }
        self._update_tool_progress("done", "Done", result)
        self._set_status("Video audio replaced.", kind="tool")
        return result

    @assistant_tool(
        display_name="Resize Crop Video",
        description="Resize and crop a previously resolved video in one step. Crop values can be expressed in pixels or percent.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id for the source video returned by Resolve Media.",
            },
            "width": {
                "type": "integer",
                "description": "Optional output width in pixels after cropping.",
                "required": False,
            },
            "height": {
                "type": "integer",
                "description": "Optional output height in pixels after cropping.",
                "required": False,
            },
            "crop_left": {
                "type": "number",
                "description": "Optional amount to crop from the left side.",
                "required": False,
            },
            "crop_top": {
                "type": "number",
                "description": "Optional amount to crop from the top side.",
                "required": False,
            },
            "crop_right": {
                "type": "number",
                "description": "Optional amount to crop from the right side.",
                "required": False,
            },
            "crop_bottom": {
                "type": "number",
                "description": "Optional amount to crop from the bottom side.",
                "required": False,
            },
            "crop_unit": {
                "type": "string",
                "description": "Crop unit: pixels or percent.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def resize_crop_video(self, media_id: str, width: int | None = None, height: int | None = None, crop_left: float | None = None, crop_top: float | None = None, crop_right: float | None = None, crop_bottom: float | None = None, crop_unit: str | None = None) -> dict[str, Any]:
        self._sync_recent_media()
        source_media, error_result = self._resolve_video_media(media_id, "media_id")
        if error_result is not None:
            return error_result
        try:
            width = None if width is None or str(width).strip() == "" else int(width)
            height = None if height is None or str(height).strip() == "" else int(height)
            crop_left = 0 if crop_left is None or str(crop_left).strip() == "" else float(crop_left)
            crop_top = 0 if crop_top is None or str(crop_top).strip() == "" else float(crop_top)
            crop_right = 0 if crop_right is None or str(crop_right).strip() == "" else float(crop_right)
            crop_bottom = 0 if crop_bottom is None or str(crop_bottom).strip() == "" else float(crop_bottom)
        except Exception:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": "width and height must be integers, crop values must be numbers."}
        crop_unit = str(crop_unit or "pixels").strip().lower() or "pixels"
        if crop_unit not in {"pixels", "percent"}:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": "crop_unit must be 'pixels' or 'percent'."}
        self._set_status("Resizing and cropping video...", kind="tool")
        self._update_tool_progress("running", "Processing", {"status": "running", "media_id": source_media.get("media_id", ""), "width": width, "height": height, "crop_left": crop_left, "crop_top": crop_top, "crop_right": crop_right, "crop_bottom": crop_bottom, "crop_unit": crop_unit})
        source_path = str(source_media.get("path", "")).strip()
        video_codec, video_container = self._get_video_output_settings()
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        output_path = self._resolve_direct_output_path(f"{base_name}_resized{deepy_video_tools.get_video_container_extension(video_container)}", False, False)
        try:
            output_path = deepy_video_tools.resize_crop_video(source_path, output_path, width=width, height=height, crop_left=crop_left, crop_top=crop_top, crop_right=crop_right, crop_bottom=crop_bottom, crop_unit=crop_unit, video_codec=video_codec, video_container=video_container, audio_codec=self._get_video_audio_output_codec())
        except Exception as exc:
            result = {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Resize/crop failed: {exc}", kind="error")
            return result
        comments = f'Resized/cropped "{os.path.basename(source_path)}"'
        if width is not None or height is not None:
            comments += f" to {width if width is not None else 'auto'}x{height if height is not None else 'auto'}"
        if any(value > 0 for value in (crop_left, crop_top, crop_right, crop_bottom)):
            comments += f" with crop {crop_left}/{crop_top}/{crop_right}/{crop_bottom} {crop_unit}"
        resized_settings = self._build_direct_media_settings(source_media, comments)
        self._update_video_metadata_fields(output_path, resized_settings)
        media_record = self._record_direct_media(output_path, resized_settings, is_image=False, audio_only=False, label="Resized/cropped video")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "source_media_id": source_media.get("media_id", ""),
            "output_file": output_path,
            "error": "",
        }
        self._update_tool_progress("done", "Done", result)
        self._set_status("Video resize/crop finished.", kind="tool")
        return result

    @assistant_tool(
        display_name="Merge Videos",
        description="Merge two previously resolved videos into one clip, resizing the second video when needed so it matches the first video dimensions.",
        parameters={
            "video_first": {
                "type": "string",
                "description": "The media id for the first video returned by Resolve Media.",
            },
            "video_second": {
                "type": "string",
                "description": "The media id for the second video returned by Resolve Media.",
            },
        },
        pause_runtime=False,
    )
    def merge_videos(self, video_first: str, video_second: str) -> dict[str, Any]:
        self._sync_recent_media()
        first_media, error_result = self._resolve_video_media(video_first, "video_first")
        if error_result is not None:
            return error_result
        second_media, error_result = self._resolve_video_media(video_second, "video_second")
        if error_result is not None:
            return error_result
        self._set_status("Merging videos...", kind="tool")
        self._update_tool_progress("running", "Merging", {"status": "running", "video_first": first_media.get("media_id", ""), "video_second": second_media.get("media_id", "")})
        first_path = str(first_media.get("path", "")).strip()
        second_path = str(second_media.get("path", "")).strip()
        first_name = os.path.basename(first_path)
        second_name = os.path.basename(second_path)
        video_codec, video_container = self._get_video_output_settings()
        output_name = f"merged_{first_media.get('media_id', 'video')}_{second_media.get('media_id', 'video')}{deepy_video_tools.get_video_container_extension(video_container)}"
        output_path = self._resolve_direct_output_path(output_name, False, False)
        output_path = deepy_video_tools.merge_videos(first_path, second_path, output_path=output_path, video_codec=video_codec, video_container=video_container, audio_codec=self._get_video_audio_output_codec())
        merged_settings = dict(second_media.get("settings", {}) or {})
        merged_settings["client_id"] = _next_ai_client_id()
        merged_settings["comments"] = f'Merged from "{first_name} & {second_name}"'
        end_time = time.time()
        merged_settings["creation_date"] = datetime.fromtimestamp(end_time).isoformat(timespec="seconds")
        merged_settings["creation_timestamp"] = int(end_time)
        try:
            fps, width, height, frames_count = get_video_info(output_path)
            merged_settings["resolution"] = f"{width}x{height}"
            merged_settings["video_length"] = int(frames_count)
            if fps > 0:
                merged_settings["duration_seconds"] = round(frames_count / fps, 3)
        except Exception:
            pass
        media_record = self._record_direct_media(output_path, merged_settings, is_image=False, audio_only=False, label="Merged video")
        result = {
            "status": "done",
            "output_file": output_path,
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "video_first": first_media.get("media_id", ""),
            "video_second": second_media.get("media_id", ""),
            "error": "",
        }
        self._update_tool_progress("done", "Done", result)
        self._set_status("Video merge finished.", kind="tool")
        return result

    @assistant_tool(
        display_name="Load Doc",
        description="Load one WanGP documentation page into Deepy's context by its doc id.",
        parameters={
            "doc_id": {
                "type": "string",
                "description": "Documentation id to load: finetunes, getting_started, loras, overview, prompts, or vace.",
            },
        },
        pause_runtime=False,
    )
    def load_doc(self, doc_id: str) -> dict[str, Any]:
        lookup_id = str(doc_id or "").strip().lower()
        doc_entry = _DEEPY_DOCS.get(lookup_id, None)
        if doc_entry is None:
            return {
                "status": "error",
                "doc_id": lookup_id,
                "available_doc_ids": sorted(_DEEPY_DOCS.keys()),
                "error": "Unknown documentation id.",
            }
        doc_path = doc_entry["path"]
        self._set_status(f"Loading {doc_entry['title']} documentation...", kind="tool")
        self._update_tool_progress("running", "Loading", {"status": "running", "doc_id": lookup_id})
        try:
            content = doc_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            result = {
                "status": "error",
                "doc_id": lookup_id,
                "title": doc_entry["title"],
                "path": str(doc_path),
                "error": str(exc),
            }
            self._update_tool_progress("error", "Error", result)
            return result
        result = {
            "status": "done",
            "doc_id": lookup_id,
            "title": doc_entry["title"],
            "path": str(doc_path.relative_to(_DOCS_DIR.parent)).replace("\\", "/"),
            "content": content,
            "error": "",
        }
        self._update_tool_progress("done", "Loaded", {"status": "done", "doc_id": lookup_id, "title": doc_entry["title"], "path": result["path"], "error": ""})
        self._set_status(f"{doc_entry['title']} documentation loaded.", kind="tool")
        return result

    @assistant_tool(
        display_name="Get Selected Media",
        description="Return the media id for the currently selected WanGP gallery item. If the selected item is a video, also report the current player time and frame number.",
        parameters={
            "media_type": {
                "type": "string",
                "description": "Optional desired media type: image, video, audio, or any.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def get_selected_media(self, media_type: str = "any") -> dict[str, Any]:
        self._sync_recent_media()
        media_record, error_result = self._get_selected_media_record(media_type)
        if error_result is not None:
            return error_result
        result = {
            "status": "done",
            **self._compact_media_payload(media_record),
            "path": str(media_record.get("path", "")).strip(),
            "error": "",
        }
        result.update(self._get_selected_video_position(media_record))
        return result

    @assistant_tool(
        display_name="Get Media Details",
        description="Return detailed local metadata for a previously resolved image or video.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id returned by Resolve Media.",
            },
        },
        pause_runtime=False,
    )
    def get_media_details(self, media_id: str) -> dict[str, Any]:
        self._sync_recent_media()
        if self.session is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "error": "Unknown media id."}
        media_path = str(media_record.get("path", "")).strip()
        media_type = str(media_record.get("media_type", "")).strip().lower()
        if media_type not in {"image", "video"}:
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "media_type": media_type,
                "error": "Detailed media info currently supports images and videos.",
            }
        self._set_status("Reading media details...", kind="tool")
        self._update_tool_progress("running", "Reading", {"status": "running", "media_id": media_record.get("media_id", ""), "media_type": media_type})
        try:
            if media_type == "image":
                with Image.open(media_path) as image_handle:
                    width, height = image_handle.size
                result = {
                    "status": "done",
                    "media_id": media_record.get("media_id", ""),
                    "label": media_record.get("label", ""),
                    "media_type": "image",
                    "path": media_path,
                    "filename": os.path.basename(media_path),
                    "width": int(width),
                    "height": int(height),
                    "resolution": f"{int(width)}x{int(height)}",
                    "frame_count": 1,
                    "fps": None,
                    "duration_seconds": None,
                    "has_audio": False,
                    "audio_track_count": 0,
                    "error": "",
                }
            else:
                fps, width, height, frame_count = get_video_info(media_path)
                audio_track_count = int(extract_audio_tracks(media_path, query_only=True))
                result = {
                    "status": "done",
                    "media_id": media_record.get("media_id", ""),
                    "label": media_record.get("label", ""),
                    "media_type": "video",
                    "path": media_path,
                    "filename": os.path.basename(media_path),
                    "width": int(width),
                    "height": int(height),
                    "resolution": f"{int(width)}x{int(height)}",
                    "frame_count": int(frame_count),
                    "fps": int(fps),
                    "duration_seconds": (float(frame_count) / float(fps)) if fps > 0 else None,
                    "has_audio": audio_track_count > 0,
                    "audio_track_count": audio_track_count,
                    "error": "",
                }
        except Exception as exc:
            result = {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "media_type": media_type,
                "path": media_path,
                "error": str(exc),
            }
            self._update_tool_progress("error", "Error", result)
            return result
        self._update_tool_progress("done", "Done", result)
        self._set_status("Media details loaded.", kind="tool")
        return result

    @assistant_tool(
        display_name="Resolve Media",
        description="Look up previously generated WanGP media by natural reference such as last image, previous image, or a short description.",
        parameters={
            "reference": {
                "type": "string",
                "description": "The user's natural-language reference to previously generated media, such as 'last image' or 'robot on the moon'.",
            },
            "media_type": {
                "type": "string",
                "description": "The desired media type: image, video, audio, or any.",
            },
        },
        pause_runtime=False,
    )
    def resolve_media_reference(self, reference: str, media_type: str) -> dict[str, Any]:
        self._sync_recent_media()
        if self.session is None:
            return {"status": "error", "reference": str(reference or "").strip(), "media_type": str(media_type or "any").strip() or "any", "matches": [], "error": "Assistant session is not available."}
        if self._is_selected_reference(reference):
            media_record, error_result = self._get_selected_media_record(media_type)
            if error_result is not None:
                error_result.setdefault("reference", str(reference or "").strip())
                return error_result
            selected_media = self._compact_media_payload(media_record, why="matched selected media")
            selected_media.update(self._get_selected_video_position(media_record))
            return {"status": "resolved", "media_type": media_registry.normalize_media_type(media_type, reference=reference), "reference": str(reference or "").strip(), "media": selected_media, "error": ""}
        result = media_registry.resolve_media_reference(self.session, reference, media_type)
        result.setdefault("error", "")
        return result

    @assistant_tool(
        display_name="Inspect Media",
        description="Ask Deepy to inspect a previously resolved image or a frame from a previously resolved video and answer a visual question about it.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id returned by Resolve Media.",
            },
            "question": {
                "type": "string",
                "description": "The visual question to answer about that media item.",
            },
            "frame_no": {
                "type": "integer",
                "description": "Optional frame number to inspect when media_id refers to a video. If omitted, the first frame is used.",
                "required": False,
            },
        },
        pause_runtime=True,
        pause_reason="vision",
    )
    def inspect_media(self, media_id: str, question: str, frame_no: int | None = None) -> dict[str, Any]:
        self._sync_recent_media()
        try:
            frame_no = None if frame_no is None or str(frame_no).strip() == "" else int(frame_no)
        except Exception:
            return {"status": "error", "media_id": str(media_id or "").strip(), "question": str(question or "").strip(), "answer": "", "error": "frame_no must be an integer."}
        self._update_tool_progress("running", "Inspecting", {"status": "running", "media_id": str(media_id or "").strip(), "question": str(question or "").strip(), "frame_no": frame_no})
        if self.session is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "question": str(question or "").strip(), "answer": "", "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "question": str(question or "").strip(), "answer": "", "error": "Unknown media id."}
        if media_record.get("media_type") not in {"image", "video"}:
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "media_type": media_record.get("media_type", ""),
                "question": str(question or "").strip(),
                "answer": "",
                "error": "Visual inspection currently supports images and videos.",
            }
        if self._vision_query_callback is None:
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "media_type": media_record.get("media_type", ""),
                "question": str(question or "").strip(),
                "answer": "",
                "error": "Deepy vision inspection is not available.",
            }
        return self._vision_query_callback(media_record, question, frame_no)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name)
            metadata = getattr(method, "_assistant_tool", None)
            if metadata is None:
                continue
            properties = {}
            required = []
            annotations = getattr(method, "__annotations__", {})
            for param_name, param_meta in metadata["parameters"].items():
                properties[param_name] = {
                    "type": param_meta.get("type") or _json_type_from_annotation(annotations.get(param_name, str)),
                    "description": str(param_meta.get("description", "")).strip(),
                }
                if bool(param_meta.get("required", True)):
                    required.append(param_name)
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": metadata["name"],
                        "description": metadata["description"],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return schemas

    def get_tool_display_name(self, tool_name: str) -> str:
        lookup_name = str(tool_name or "").strip()
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name)
            metadata = getattr(method, "_assistant_tool", None)
            if metadata is None or metadata["name"] != lookup_name:
                continue
            return str(metadata.get("display_name", lookup_name)).strip() or lookup_name
        return lookup_name.replace("_", " ").replace("-", " ").strip().title() or "Tool"

    def get_tool_policy(self, tool_name: str) -> dict[str, Any]:
        lookup_name = str(tool_name or "").strip()
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name)
            metadata = getattr(method, "_assistant_tool", None)
            if metadata is None or metadata["name"] != lookup_name:
                continue
            return {
                "pause_runtime": bool(metadata.get("pause_runtime", True)),
                "pause_reason": str(metadata.get("pause_reason", "tool") or "tool"),
            }
        return {"pause_runtime": True, "pause_reason": "tool"}

    def validate_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        lookup_name = str(tool_name or "").strip()
        call_args = dict(arguments or {})
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name)
            metadata = getattr(method, "_assistant_tool", None)
            if metadata is None or metadata["name"] != lookup_name:
                continue
            for param_name, param_meta in metadata["parameters"].items():
                if not bool(param_meta.get("required", True)):
                    continue
                value = call_args.get(param_name, None)
                if value is None:
                    return f"{param_name} is required."
                if str(param_meta.get("type", "")).strip().lower() == "string" and len(str(value or "").strip()) == 0:
                    return f"{param_name} is empty."
            return ""
        return ""

    def infer_tool_calls(self, raw_text: str) -> list[dict[str, Any]]:
        candidate_texts = []
        thinking_text, answer_text = qwen35_text._split_generated_text(raw_text)
        for candidate in (raw_text, answer_text, thinking_text):
            candidate = str(candidate or "").strip()
            if len(candidate) > 0:
                candidate_texts.append(candidate)

        by_name = {}
        sole_tool_name = None
        sole_tool_params = set()
        for schema in self.get_tool_schemas():
            function_spec = schema.get("function", {})
            tool_name = str(function_spec.get("name", "")).strip()
            if len(tool_name) == 0:
                continue
            by_name[tool_name] = set(function_spec.get("parameters", {}).get("properties", {}).keys())
        if len(by_name) == 1:
            sole_tool_name = next(iter(by_name))
            sole_tool_params = by_name[sole_tool_name]

        for candidate in candidate_texts:
            pseudo_match = re.search(r"Tool call:\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", candidate, flags=re.DOTALL)
            if pseudo_match is not None:
                tool_name = pseudo_match.group(1).strip()
                raw_args = pseudo_match.group(2).strip()
                arguments = {}
                for arg_name, quoted_value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"', raw_args):
                    arguments[arg_name] = quoted_value
                for arg_name, quoted_value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'([^']*)'", raw_args):
                    arguments[arg_name] = quoted_value
                if tool_name in by_name:
                    return [{"name": tool_name, "arguments": arguments}]

            fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
            json_candidate = fenced_match.group(1).strip() if fenced_match is not None else strip_trailing_stop_markup(candidate).strip()
            try:
                parsed = json.loads(json_candidate)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            if "name" in parsed and "arguments" in parsed:
                tool_name = str(parsed.get("name", "")).strip()
                arguments = parsed.get("arguments", {})
                if isinstance(arguments, dict) and tool_name in by_name:
                    return [{"name": tool_name, "arguments": arguments}]
            if sole_tool_name is not None and set(parsed.keys()).issubset(sole_tool_params):
                return [{"name": sole_tool_name, "arguments": parsed}]
        return []

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name)
            metadata = getattr(method, "_assistant_tool", None)
            if metadata is None:
                continue
            if metadata["name"] != tool_name:
                continue
            return method(**dict(arguments or {}))
        raise KeyError(f"Unknown assistant tool: {tool_name}")


class AssistantEngine:
    def __init__(self, session: AssistantSessionState, runtime_hooks: AssistantRuntimeHooks, tool_box: tools, send_cmd, debug_enabled: bool | None = None, thinking_enabled: bool = True, vram_mode: str = DEEPY_VRAM_UNLOAD):
        self.session = session
        self.runtime_hooks = runtime_hooks
        self.tool_box = tool_box
        self.send_cmd = send_cmd
        self.debug_enabled = ASSISTANT_DEBUG if debug_enabled is None else bool(debug_enabled)
        self.thinking_enabled = bool(thinking_enabled)
        self.vram_mode = normalize_deepy_vram_mode(vram_mode)
        self.runtime: Qwen35AssistantRuntime | None = None
        self._gpu_acquired = False
        self._skip_pause_snapshot = False
        self._active_turn_id = ""
        self._active_tool_context: tuple[str, str] | None = None
        self._stream_answer_text = ""
        self._stream_reasoning_text = ""
        self._stream_reasoning_block_id = ""
        self._stream_thinking_unknown = False
        self._stream_thinking_open = False
        bind_runtime_tools = getattr(self.tool_box, "bind_runtime_tools", None)
        if callable(bind_runtime_tools):
            bind_runtime_tools(vision_query_callback=self._run_visual_query, tool_progress_callback=self._handle_tool_progress)

    def _log(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[Assistant] {message}")

    def _emit_chat_event(self, payload: str | None) -> None:
        if payload is None or len(str(payload).strip()) == 0:
            return
        self.send_cmd("chat_output", payload)

    def _set_status(self, text: str | None, kind: str = "thinking") -> None:
        self._emit_chat_event(assistant_chat.build_status_event(text, kind=kind, visible=text is not None and len(str(text).strip()) > 0))

    def _hide_status(self) -> None:
        self._emit_chat_event(assistant_chat.build_status_event(None, visible=False))

    def _get_context_window_tokens(self) -> int:
        return normalize_deepy_context_tokens(get_deepy_config_value(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT))

    def _get_custom_system_prompt(self) -> str:
        return normalize_deepy_custom_system_prompt(get_deepy_config_value(DEEPY_CUSTOM_SYSTEM_PROMPT_KEY, ""))

    def _build_system_prompt(self, *, log_injections: bool = False) -> str:
        system_prompt = ASSISTANT_SYSTEM_PROMPT.rstrip()
        custom_system_prompt = self._get_custom_system_prompt()
        if len(custom_system_prompt) > 0:
            system_prompt = f"{system_prompt}\n\n{custom_system_prompt}"
        if len(self.session.interruption_notice.strip()) > 0:
            if log_injections:
                self._log(f"Injecting interruption notice into system prompt: {self.session.interruption_notice.strip()}")
            system_prompt = f"{system_prompt.rstrip()}\n\n{self.session.interruption_notice.strip()}"
        return system_prompt

    def _current_system_prompt_signature(self) -> str:
        return self._build_system_prompt()

    def _remember_render_state(self) -> None:
        self.session.rendered_system_prompt_signature = self._current_system_prompt_signature()
        self.session.rendered_context_window_tokens = self._get_context_window_tokens()
        self.session.rendered_messages_len = len(self.session.messages)

    def _message_render_content(self, message: dict[str, Any]) -> str:
        model_content = message.get("model_content", None)
        if isinstance(model_content, str) and len(model_content) > 0:
            return model_content
        return str(message.get("content", "") or "")

    def _get_pending_render_messages(self) -> list[dict[str, Any]]:
        try:
            start_idx = int(self.session.rendered_messages_len or 0)
        except Exception:
            start_idx = 0
        start_idx = max(0, min(start_idx, len(self.session.messages)))
        return list(self.session.messages[start_idx:])

    def _can_append_pending_user_suffix(self) -> bool:
        if self.session.rendered_system_prompt_signature != self._current_system_prompt_signature():
            return False
        if int(self.session.rendered_context_window_tokens or 0) != self._get_context_window_tokens():
            return False
        pending_messages = self._get_pending_render_messages()
        return len(pending_messages) == 1 and str(pending_messages[0].get("role", "")).strip().lower() == "user"

    def _pending_user_render_content(self) -> str:
        pending_messages = self._get_pending_render_messages()
        if len(pending_messages) != 1:
            return ""
        if str(pending_messages[0].get("role", "")).strip().lower() != "user":
            return ""
        return self._message_render_content(pending_messages[0]).strip()

    def _can_append_pending_tool_suffix(self) -> bool:
        if self.session.rendered_system_prompt_signature != self._current_system_prompt_signature():
            return False
        if int(self.session.rendered_context_window_tokens or 0) != self._get_context_window_tokens():
            return False
        pending_messages = self._get_pending_render_messages()
        return len(pending_messages) > 0 and all(str(message.get("role", "")).strip().lower() == "tool" for message in pending_messages)

    def _pending_tool_render_contents(self) -> list[str]:
        return [self._message_render_content(message).strip() for message in self._get_pending_render_messages() if str(message.get("role", "")).strip().lower() == "tool" and len(self._message_render_content(message).strip()) > 0]

    def _refresh_runtime_status_note(self) -> None:
        snapshot = self.tool_box._get_selected_runtime_snapshot()
        if snapshot is None:
            if len(str(self.session.runtime_status_signature or "").strip()) == 0:
                self.session.runtime_status_note = ""
                return
            normalized_snapshot = {
                "selected_media_id": None,
                "selected_media_type": None,
                "selected_media_label": None,
                "current_time_seconds": None,
                "current_frame_no": None,
            }
        else:
            normalized_snapshot = {
                "selected_media_id": str(snapshot.get("selected_media_id", "") or "").strip() or None,
                "selected_media_type": str(snapshot.get("selected_media_type", "") or "").strip() or None,
                "selected_media_label": str(snapshot.get("selected_media_label", "") or "").strip() or None,
                "current_time_seconds": snapshot.get("current_time_seconds", None),
                "current_frame_no": snapshot.get("current_frame_no", None),
            }
        signature = _json_dumps(normalized_snapshot)
        if signature == self.session.runtime_status_signature:
            self.session.runtime_status_note = ""
            return
        lines = [
            "<wangp_runtime_update>",
            "Hidden WanGP runtime state. This is environment metadata, not a user message.",
            "Use it as factual UI context only.",
        ]
        for key in ("selected_media_id", "selected_media_type", "selected_media_label", "current_time_seconds", "current_frame_no"):
            value = normalized_snapshot.get(key, None)
            if isinstance(value, str):
                rendered_value = value if len(value) > 0 else "none"
            else:
                rendered_value = "none" if value is None else value
            lines.append(f"{key}: {rendered_value}")
        lines.append("</wangp_runtime_update>")
        self.session.runtime_status_note = "\n".join(lines)
        self.session.runtime_status_signature = signature
        if self.debug_enabled:
            self._log(f"Prepared runtime status update: {signature}")

    def _build_pending_user_message(self, user_text: str) -> dict[str, Any]:
        message = {"role": "user", "content": str(user_text or "").strip()}
        runtime_status_note = str(self.session.runtime_status_note or "").strip()
        if len(runtime_status_note) == 0:
            return message
        message["model_content"] = f"{runtime_status_note}\n\n{message['content']}".strip()
        self.session.runtime_status_note = ""
        if self.debug_enabled:
            self._log(f"Queued runtime status update inside hidden user content: {runtime_status_note}")
        return message

    def _record_live_context(self, log_message: str) -> str:
        if self.runtime is None:
            raise RuntimeError("Assistant runtime is not available for live-context recording.")
        current_seq = self.runtime._get_active_sequence()
        if current_seq is None or len(current_seq.token_ids) == 0:
            return self._canonicalize_context(sync_runtime="record_only")
        self.session.rendered_token_ids = [int(token_id) for token_id in current_seq.token_ids]
        self.session.runtime_snapshot = None
        self.session.pending_replay_reason = ""
        self._skip_pause_snapshot = False
        self._remember_render_state()
        self._log(log_message)
        return "recorded"

    def _send_chat(self, text: str) -> None:
        text = str(text or "").strip()
        if len(text) == 0:
            return
        self._emit_chat_event(assistant_chat.set_assistant_content(self.session, self._ensure_active_turn(), text))

    def _ensure_active_turn(self) -> str:
        if len(self._active_turn_id) == 0:
            self._active_turn_id = assistant_chat.create_assistant_turn(self.session)
            mark_assistant_turn_message(self.session, self._active_turn_id)
        return self._active_turn_id

    def _split_for_display(self, raw_text: str) -> tuple[str, str]:
        thinking_text, answer_text = qwen35_text._split_generated_text(raw_text)
        if self.debug_enabled and len(thinking_text) > 0:
            print("[Assistant][Thinking]")
            try:
                print(thinking_text)
            except UnicodeEncodeError:
                encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                safe_text = thinking_text.encode(encoding, errors="replace").decode(encoding, errors="replace")
                sys.stdout.write(safe_text + "\n")
                sys.stdout.flush()
        return thinking_text, answer_text

    def _start_stream_pass(self) -> None:
        self._ensure_active_turn()
        self._stream_answer_text = ""
        self._stream_reasoning_text = ""
        self._stream_reasoning_block_id = ""
        self._stream_thinking_unknown = self.runtime is not None and qwen35_text._prompt_enhancer_thinking_enabled(self.runtime.model, thinking_enabled=self.thinking_enabled)
        self._stream_thinking_open = False

    def _current_stream_content(self) -> str:
        return self._stream_answer_text

    def _split_streaming_text(self, raw_text: str, is_final: bool = False) -> tuple[str, str]:
        text = strip_trailing_stop_markup(str(raw_text or "")).replace("\r\n", "\n").replace("\r", "\n")
        lowered = text.lower()
        open_idx = lowered.find("<think>")
        close_idx = lowered.find("</think>")
        if open_idx >= 0 and (close_idx < 0 or open_idx < close_idx):
            self._stream_thinking_unknown = False
            if close_idx < 0:
                self._stream_thinking_open = True
                return qwen35_text._normalize_generated_text(text[open_idx + len("<think>") :]), ""
            self._stream_thinking_open = False
            thinking_text, answer_text = qwen35_text._split_generated_text(text)
            return thinking_text, qwen35_text._clean_answer_text(_strip_partial_tool_markup(answer_text))
        if self._stream_thinking_open and close_idx < 0:
            return qwen35_text._normalize_generated_text(text.replace("<think>", "\n")), ""
        if close_idx >= 0:
            self._stream_thinking_unknown = False
            self._stream_thinking_open = False
            thinking_text, answer_text = qwen35_text._split_generated_text(text)
            return thinking_text, qwen35_text._clean_answer_text(_strip_partial_tool_markup(answer_text))
        if self._stream_thinking_unknown and not is_final:
            return "", ""
        self._stream_thinking_unknown = False
        thinking_text, answer_text = qwen35_text._split_generated_text(text)
        return thinking_text, qwen35_text._clean_answer_text(_strip_partial_tool_markup(answer_text))

    def _stream_generation_update(self, *, raw_text: str, token_count: int, stop_reason: str | None, is_final: bool) -> None:
        turn_id = self._ensure_active_turn()
        thinking_text, answer_text = self._split_streaming_text(raw_text, is_final=is_final)
        if not is_final and len(thinking_text) < len(self._stream_reasoning_text):
            thinking_text = self._stream_reasoning_text
        if not is_final and len(answer_text) < len(self._stream_answer_text):
            answer_text = self._stream_answer_text
        if thinking_text != self._stream_reasoning_text and len(thinking_text) > 0:
            self._stream_reasoning_block_id, reasoning_event = assistant_chat.upsert_reasoning_block(self.session, turn_id, self._stream_reasoning_block_id, thinking_text)
            self._stream_reasoning_text = thinking_text
            self._emit_chat_event(reasoning_event)
        if answer_text != self._stream_answer_text and len(answer_text) > 0:
            self._stream_answer_text = answer_text
            self._emit_chat_event(assistant_chat.set_assistant_content(self.session, turn_id, self._stream_answer_text))

    def _handle_tool_progress(self, status: str | None = None, status_text: str | None = None, result: dict[str, Any] | None = None) -> None:
        if self._active_tool_context is None:
            return
        message_id, tool_id = self._active_tool_context
        self._emit_chat_event(assistant_chat.update_tool_call(self.session, message_id, tool_id, status=status, status_text=status_text, result=result))

    def _acquire_runtime(self) -> Qwen35AssistantRuntime:
        acquired_here = False
        if not self._gpu_acquired:
            self.runtime_hooks.clear_gpu_resident()
            self.session.release_vram_callback = None
            self.runtime_hooks.acquire_gpu()
            self._gpu_acquired = True
            acquired_here = True
        try:
            model, _tokenizer = self.runtime_hooks.ensure_loaded()
            model._prompt_enhancer_min_model_len_hint = self._get_context_window_tokens()
            if self.runtime is None or self.runtime.model is not model:
                self.runtime = Qwen35AssistantRuntime(model, debug_enabled=self.debug_enabled)
            return self.runtime
        except Exception:
            if acquired_here:
                self._gpu_acquired = False
                self.runtime_hooks.release_gpu()
            raise

    def _ensure_vision_loaded(self) -> tuple[Any, Any]:
        ensure_vision_loaded = self.runtime_hooks.ensure_vision_loaded
        if not callable(ensure_vision_loaded):
            raise RuntimeError("Deepy vision runtime is not available.")
        caption_model, caption_processor = ensure_vision_loaded()
        if caption_model is None or caption_processor is None:
            raise RuntimeError("Deepy vision runtime is not available.")
        return caption_model, caption_processor

    def _run_visual_query(self, media_record: dict[str, Any], question: str, frame_no: int | None = None) -> dict[str, Any]:
        if not self._gpu_acquired:
            self.runtime_hooks.clear_gpu_resident()
            self.session.release_vram_callback = None
            self.runtime_hooks.acquire_gpu()
            self._gpu_acquired = True
        media_path = str(media_record.get("path", "")).strip()
        if len(media_path) == 0 or not os.path.isfile(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")
        caption_model, caption_processor = self._ensure_vision_loaded()
        media_type = str(media_record.get("media_type", "")).strip().lower()
        if media_type == "video":
            image = get_video_frame(media_path, 0 if frame_no is None else int(frame_no), return_last_if_missing=True, return_PIL=True).convert("RGB")
        else:
            with Image.open(media_path) as image_handle:
                image = image_handle.convert("RGB")
        prompt_token_ids, prompt_embeds, prompt_position_ids, position_offset = deepy_vision.build_image_question_prompt(
            caption_model,
            caption_processor,
            image,
            question,
        )
        runtime = self._acquire_runtime()
        answer = runtime.generate_embedded_answer(
            prompt_token_ids,
            prompt_embeds,
            prompt_position_ids,
            position_offset,
            max_new_tokens=192,
            seed=0,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        return {
            "status": "done",
            "media_id": media_record.get("media_id", ""),
            "media_type": media_type,
            "label": media_record.get("label", ""),
            "frame_no": None if media_type != "video" else (0 if frame_no is None else int(frame_no)),
            "question": str(question or "").strip(),
            "answer": answer,
            "error": "",
        }

    def _force_release_vram(self) -> None:
        self.runtime_hooks.clear_gpu_resident()
        try:
            if self.runtime is not None and self.session.runtime_snapshot is None and len(self.session.rendered_token_ids) > 0:
                self.session.runtime_snapshot = self.runtime.snapshot_context()
        except Exception as exc:
            self._log(f"Resident snapshot before VRAM release failed: {exc}")
        try:
            self.runtime_hooks.unload_runtime()
        finally:
            self.runtime_hooks.unload_weights()
            self.runtime = None
            self.session.release_vram_callback = None

    def _pause_runtime(self, pause_reason: str = "idle") -> None:
        keep_loaded = self.vram_mode in (DEEPY_VRAM_ALWAYS, DEEPY_VRAM_UNLOAD_ON_REQUEST)
        if pause_reason == "vision":
            keep_loaded = False
        if pause_reason == "tool" and self.vram_mode != DEEPY_VRAM_ALWAYS:
            keep_loaded = False
        allow_force_release = keep_loaded and self.vram_mode == DEEPY_VRAM_UNLOAD_ON_REQUEST and pause_reason != "tool"
        release_callback = self._force_release_vram if keep_loaded else None
        if keep_loaded:
            self.session.release_vram_callback = release_callback
        else:
            self.session.release_vram_callback = None

        if not self._gpu_acquired:
            if self.session.drop_state_requested:
                if callable(self.session.release_vram_callback):
                    self.session.release_vram_callback()
                clear_assistant_session(self.session)
                self.session.drop_state_requested = False
            return
        try:
            if self.runtime is not None and not self.session.drop_state_requested and not self._skip_pause_snapshot:
                self.session.runtime_snapshot = self.runtime.snapshot_context()
            else:
                self.session.runtime_snapshot = None
        finally:
            try:
                if not keep_loaded:
                    self.runtime_hooks.unload_runtime()
            finally:
                try:
                    if not keep_loaded:
                        self.runtime_hooks.unload_weights()
                        self.runtime = None
                finally:
                    self.runtime_hooks.release_gpu(
                        keep_resident=allow_force_release,
                        release_vram_callback=release_callback,
                        force_release_on_acquire=allow_force_release,
                    )
                    self._gpu_acquired = False
                    self._skip_pause_snapshot = False
                    if self.session.drop_state_requested:
                        if keep_loaded and callable(self.session.release_vram_callback):
                            self.session.release_vram_callback()
                        clear_assistant_session(self.session)
                        self.session.drop_state_requested = False

    def _render_messages(self, add_generation_prompt: bool) -> list[int]:
        if self.runtime is None:
            raise RuntimeError("Assistant runtime is not available for prompt rendering.")
        messages = [{"role": "system", "content": self._build_system_prompt(log_injections=True)}]
        for message in self.session.messages:
            role = str(message.get("role", "")).strip().lower()
            if role == "assistant":
                model_message = {"role": "assistant"}
                model_content = message.get("model_content", None)
                if isinstance(model_content, str) and len(model_content) > 0:
                    model_message["content"] = model_content
                elif "content" in message:
                    model_message["content"] = message["content"]
                if "tool_calls" in message:
                    model_message["tool_calls"] = message["tool_calls"]
                messages.append(model_message)
                continue
            model_message = {"role": role}
            model_message["content"] = self._message_render_content(message)
            messages.append(model_message)
        thinking_enabled = qwen35_text._prompt_enhancer_thinking_enabled(self.runtime.model, thinking_enabled=self.thinking_enabled)
        return render_assistant_messages(
            self.runtime.tokenizer,
            messages,
            self.tool_box.get_tool_schemas(),
            add_generation_prompt=add_generation_prompt,
            thinking_enabled=thinking_enabled,
        )

    def _restore_or_replay_session(self) -> str:
        if self.runtime is None:
            raise RuntimeError("Assistant runtime is not available for restore.")
        runtime = self.runtime
        fallback_tokens = self.session.rendered_token_ids
        if len(fallback_tokens) == 0:
            return "empty"
        try:
            live_seq = runtime._get_active_sequence()
        except Exception:
            live_seq = None
        if live_seq is not None:
            live_token_ids = [int(token_id) for token_id in live_seq.token_ids]
            snapshot_seq = None if self.session.runtime_snapshot is None else self.session.runtime_snapshot.get("sequence", {})
            snapshot_token_ids = [] if not isinstance(snapshot_seq, dict) else [int(token_id) for token_id in snapshot_seq.get("token_ids", []) or []]
            if len(snapshot_token_ids) > 0 and snapshot_token_ids == live_token_ids:
                self._log("Session context reused live runtime. [no prefill redone]")
                self.session.runtime_snapshot = None
                self.session.pending_replay_reason = ""
                return "reused"
            if fallback_tokens[: len(live_token_ids)] == live_token_ids:
                self._log("Session context reused live runtime. [no prefill redone]")
                self.session.runtime_snapshot = None
                self.session.pending_replay_reason = ""
                return "reused"
        mode, runtime_replay_reason = runtime.restore_or_replay(self.session.runtime_snapshot, fallback_tokens)
        pending_replay_reason = str(self.session.pending_replay_reason or "").strip()
        runtime_replay_reason = str(runtime_replay_reason or "").strip()
        if len(pending_replay_reason) > 0 and runtime_replay_reason == "no exact runtime snapshot was available":
            replay_reason = pending_replay_reason
        elif len(pending_replay_reason) > 0 and len(runtime_replay_reason) > 0:
            replay_reason = f"{pending_replay_reason}; {runtime_replay_reason}"
        else:
            replay_reason = pending_replay_reason or runtime_replay_reason
        if mode == "replayed":
            if len(replay_reason) > 0:
                self._log(f"Session context replayed. Reason: {replay_reason} [prefill redone]")
            else:
                self._log("Session context replayed. [prefill redone]")
        elif mode == "restored":
            if len(replay_reason) > 0:
                self._log(f"Session context restored. Reason: {replay_reason} [no prefill redone]")
            else:
                self._log("Session context restored. [no prefill redone]")
        else:
            self._log(f"Session context {mode}.")
        self.session.runtime_snapshot = None
        self.session.pending_replay_reason = ""
        return mode

    def _discard_oldest_completed_turn(self) -> str:
        messages = self.session.messages
        user_indexes = [idx for idx, message in enumerate(messages) if str(message.get("role", "")).strip().lower() == "user"]
        if len(user_indexes) > 1:
            cut = user_indexes[1]
            del messages[:cut]
            return f"dropped oldest turn ({cut} messages)"
        return ""

    def _fit_rendered_messages_to_window(self, *, add_generation_prompt: bool, reserve_tokens: int = 0) -> list[int]:
        if self.runtime is None:
            raise RuntimeError("Assistant runtime is not available for context fitting.")
        max_model_len = self._get_context_window_tokens()
        hard_budget = max(1, max_model_len - max(0, int(reserve_tokens)))
        post_trim_budget = min(hard_budget, max(1, int(max_model_len * _POST_TRIM_WINDOW_FRACTION)))
        target_tokens = self._render_messages(add_generation_prompt=add_generation_prompt)
        if len(target_tokens) <= hard_budget:
            return target_tokens
        while len(target_tokens) > post_trim_budget:
            trim_reason = self._discard_oldest_completed_turn()
            if len(trim_reason) == 0:
                if len(target_tokens) > hard_budget:
                    raise RuntimeError(f"Current assistant turn alone exceeds the model window ({len(target_tokens)} > {hard_budget}) and will not be cut mid-turn.")
                break
            self._log(f"Trimming assistant context: {trim_reason}.")
            target_tokens = self._render_messages(add_generation_prompt=add_generation_prompt)
        if len(target_tokens) > hard_budget:
            raise RuntimeError(f"Assistant context exceeds the model window ({len(target_tokens)} > {hard_budget}) and cannot be trimmed further without cutting the current turn.")
        return target_tokens

    def _sync_generation_context(self) -> None:
        runtime = self._acquire_runtime()
        if len(self.session.rendered_token_ids) > 0:
            restore_mode = self._restore_or_replay_session()
            if restore_mode in ("reused", "restored") and self._can_append_pending_tool_suffix():
                thinking_enabled = qwen35_text._prompt_enhancer_thinking_enabled(self.runtime.model, thinking_enabled=self.thinking_enabled)
                suffix_tokens = render_tool_turn_suffix(runtime.tokenizer, self._pending_tool_render_contents(), thinking_enabled=thinking_enabled)
                if len(suffix_tokens) > 0:
                    mode = runtime.append_suffix(suffix_tokens)
                    self._record_live_context("Generation context extended from live runtime. [suffix append only]" if mode == "extended" else "Generation context prefilled from live runtime. [prefill redone]" if mode == "prefilled" else f"Generation context {mode} from live runtime.")
                    return
            if restore_mode in ("reused", "restored") and self._can_append_pending_user_suffix():
                thinking_enabled = qwen35_text._prompt_enhancer_thinking_enabled(self.runtime.model, thinking_enabled=self.thinking_enabled)
                suffix_tokens = render_text_user_turn_suffix(runtime.tokenizer, self._pending_user_render_content(), thinking_enabled=thinking_enabled)
                if len(suffix_tokens) > 0:
                    mode = runtime.append_suffix(suffix_tokens)
                    self._record_live_context("Generation context extended from live runtime. [suffix append only]" if mode == "extended" else "Generation context prefilled from live runtime. [prefill redone]" if mode == "prefilled" else f"Generation context {mode} from live runtime.")
                    return
        target_tokens = self._fit_rendered_messages_to_window(add_generation_prompt=True, reserve_tokens=128)
        if len(self.session.rendered_token_ids) > 0:
            mode = runtime.extend_context(target_tokens)
            self._remember_render_state()
            if mode == "prefilled":
                self._log("Generation context prefilled. [prefill redone]")
            elif mode == "extended":
                self._log("Generation context extended. [suffix append only]")
            elif mode == "replayed":
                self._log("Generation context replayed. [prefill redone]")
            else:
                self._log(f"Generation context {mode}.")
            return
        runtime.prime_context(target_tokens)
        self._remember_render_state()
        self._log("Generation context primed. [prefill redone]")

    def _canonicalize_context(self, sync_runtime: bool | str = True) -> str:
        if self.runtime is None:
            raise RuntimeError("Assistant runtime is not available for canonicalization.")
        target_tokens = self._fit_rendered_messages_to_window(add_generation_prompt=False)
        if not sync_runtime or sync_runtime == "record_only":
            self.session.rendered_token_ids = list(target_tokens)
            self.session.runtime_snapshot = None
            self.session.pending_replay_reason = "context canonicalization was recorded without syncing runtime"
            self._remember_render_state()
            self._skip_pause_snapshot = True
            self._log("Canonical context recorded without runtime sync.")
            return "recorded"
        if sync_runtime == "record_preserve_live":
            self.session.rendered_token_ids = list(target_tokens)
            self.session.runtime_snapshot = None
            self.session.pending_replay_reason = ""
            self._remember_render_state()
            self._skip_pause_snapshot = False
            self._log("Canonical context recorded while preserving live runtime.")
            return "recorded"
        current_seq = self.runtime._get_active_sequence()
        if sync_runtime == "if_cheap":
            if current_seq is None or len(current_seq.token_ids) == 0:
                self.session.rendered_token_ids = list(target_tokens)
                self.session.runtime_snapshot = None
                self.session.pending_replay_reason = "no active runtime sequence was available during canonicalization"
                self._remember_render_state()
                self._skip_pause_snapshot = True
                self._log("Canonical context recorded without runtime sync because no active sequence was available.")
                return "recorded"
            current_token_ids = [int(token_id) for token_id in current_seq.token_ids]
            if target_tokens[: len(current_token_ids)] != current_token_ids:
                self.session.rendered_token_ids = list(target_tokens)
                self.session.runtime_snapshot = None
                self.session.pending_replay_reason = _describe_prefix_mismatch(current_token_ids, target_tokens)
                self._remember_render_state()
                self._skip_pause_snapshot = True
                self._log("Canonical context recorded without runtime sync because it would require replay.")
                return "recorded"
        self._skip_pause_snapshot = False
        self.session.pending_replay_reason = ""
        if current_seq is None or len(current_seq.token_ids) == 0:
            self.runtime.prime_context(target_tokens)
            self._log("Canonical context rebuilt from scratch.")
            mode = "replayed"
        else:
            mode = self.runtime.extend_context(target_tokens)
            self._log(f"Canonical context {mode}.")
        self.session.rendered_token_ids = list(target_tokens)
        self._remember_render_state()
        return mode

    def _build_tool_error(self, tool_name: str, arguments: dict[str, Any], error_text: str) -> dict[str, Any]:
        return {
            "status": "error",
            "tool": tool_name,
            "arguments": dict(arguments or {}),
            "error": str(error_text),
        }

    def _execute_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(tool_call.get("name", "")).strip()
        tool_label = self.tool_box.get_tool_display_name(tool_name)
        tool_transcript_label = self.tool_box.get_tool_transcript_label(tool_name)
        tool_template = self.tool_box.get_tool_template_filename(tool_name)
        tool_policy = self.tool_box.get_tool_policy(tool_name)
        arguments = dict(tool_call.get("arguments", {}) or {})
        self._log(f"Tool call: {tool_name} {arguments}")
        message_id = self._ensure_active_turn()
        tool_id, tool_event = assistant_chat.add_tool_call(self.session, message_id, tool_name, arguments, tool_label=tool_transcript_label)
        self._emit_chat_event(tool_event)
        validation_error = self.tool_box.validate_tool_call(tool_name, arguments)
        if len(validation_error) > 0:
            result = self._build_tool_error(tool_name, arguments, validation_error)
            self._log(f"Tool validation error: {validation_error}")
            self._set_status(f"{tool_label} failed: {validation_error}", kind="error")
            self._emit_chat_event(assistant_chat.complete_tool_call(self.session, message_id, tool_id, result))
            self._emit_chat_event(assistant_chat.build_sync_event(self.session))
            return result
        if len(tool_template) > 0:
            self._set_status(f"Using {tool_label} ({tool_template})...", kind="tool")
        else:
            self._set_status(f"Using {tool_label}...", kind="tool")
        if tool_policy.get("pause_runtime", True):
            self._pause_runtime(pause_reason=tool_policy.get("pause_reason", "tool"))
        try:
            self._active_tool_context = (message_id, tool_id)
            result = self.tool_box.call(tool_name, arguments)
        except Exception as exc:
            result = self._build_tool_error(tool_name, arguments, str(exc))
            self._log(f"Tool error: {exc}")
        finally:
            self._active_tool_context = None
        self._log(f"Tool result: {_json_dumps(result)}")
        self._emit_chat_event(assistant_chat.complete_tool_call(self.session, message_id, tool_id, result))
        # Queue-backed tools can finish and immediately trigger another model pass; emit a full
        # transcript sync here so the UI materializes the final tool state and attachment first.
        self._emit_chat_event(assistant_chat.build_sync_event(self.session))
        return result

    def _append_assistant_message(self, raw_text: str, tool_calls: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        cleaned_text = strip_tool_blocks(raw_text)
        if tool_calls:
            cleaned_text = strip_inline_tool_call_text(cleaned_text)
        message = {"role": "assistant"}
        stripped_text = strip_trailing_stop_markup(cleaned_text)
        stripped_raw_text = strip_trailing_stop_markup(raw_text)
        thinking_text, answer_text = qwen35_text._split_generated_text(stripped_text)
        thinking_enabled = self.runtime is not None and qwen35_text._prompt_enhancer_thinking_enabled(self.runtime.model, thinking_enabled=self.thinking_enabled)
        if thinking_enabled and ("<think>" in stripped_text.lower() or "</think>" in stripped_text.lower() or len(thinking_text) > 0):
            content = "<think>\n"
            if len(thinking_text) > 0:
                content += f"{thinking_text}\n"
            content += "</think>"
            if len(answer_text) > 0:
                content += f"\n\n{answer_text}"
        else:
            content = answer_text if len(answer_text) > 0 else stripped_text
        if len(content) > 0:
            message["content"] = content
        if len(stripped_raw_text) > 0 and not tool_calls:
            message["model_content"] = stripped_raw_text
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{int(time.time() * 1000)}_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": dict(tool_call["arguments"]),
                    },
                }
                for idx, tool_call in enumerate(tool_calls)
            ]
        self.session.messages.append(message)
        return message.get("tool_calls", [])

    def _append_tool_message(self, payload: dict[str, Any], tool_call_id: str | None = None) -> None:
        message = {"role": "tool", "content": _json_dumps(payload)}
        if tool_call_id:
            message["tool_call_id"] = str(tool_call_id)
        self.session.messages.append(message)

    def run_turn(self, user_text: str, max_new_tokens: int = 1024, seed: int | None = 0, do_sample: bool = True, temperature: float | None = 0.6, top_p: float | None = 0.9, top_k: int | None = None) -> None:
        user_text = str(user_text or "").strip()
        if len(user_text) == 0:
            self._send_chat("Please enter a request.")
            return

        if self.debug_enabled:
            print("[User]")
            print(user_text)

        self._active_turn_id = ""
        self._refresh_runtime_status_note()
        self.session.messages.append(self._build_pending_user_message(user_text))
        recent_thoughts: list[str] = []
        model_passes = 0
        final_user_text = ""
        turn_completed = False
        try:
            while True:
                if self.session.interrupt_requested:
                    break
                show_loading_status = model_passes == 0 and (
                    self.session.force_loading_status_once
                    or (len(self.session.rendered_token_ids) == 0 and self.session.runtime_snapshot is None)
                )
                self._set_status("Loading Deepy..." if show_loading_status else "Thinking...", kind="loading" if show_loading_status else "thinking")
                self._sync_generation_context()
                if self.session.interrupt_requested:
                    break
                if show_loading_status:
                    self.session.force_loading_status_once = False
                    self._set_status("Thinking...", kind="thinking")
                self._start_stream_pass()
                result = self.runtime.generate_segment(
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    thinking_enabled=self.thinking_enabled,
                    stop_requested=lambda: bool(self.session.interrupt_requested),
                    stream_callback=self._stream_generation_update,
                    stream_interval_seconds=1.0,
                )
                model_passes += 1
                if self.session.interrupt_requested or result.stop_reason == "interrupted":
                    break
                raw_text = result.raw_text
                thinking_text, answer_text = self._split_for_display(raw_text)
                normalized_thinking = re.sub(r"\s+", " ", str(thinking_text or "")).strip()
                if len(normalized_thinking) == 0:
                    recent_thoughts.clear()
                else:
                    recent_thoughts.append(normalized_thinking)
                    if len(recent_thoughts) > 4:
                        recent_thoughts = recent_thoughts[-4:]
                    if len(recent_thoughts) >= 3 and recent_thoughts[-1] == recent_thoughts[-2] == recent_thoughts[-3]:
                        self._send_chat("Assistant stopped because the same thought repeated 3 times in a row.")
                        turn_completed = True
                        break
                    if (
                        len(recent_thoughts) >= 4
                        and recent_thoughts[-1] == recent_thoughts[-3]
                        and recent_thoughts[-2] == recent_thoughts[-4]
                        and recent_thoughts[-1] != recent_thoughts[-2]
                    ):
                        self._send_chat("Assistant stopped because the same two thoughts started alternating in a loop.")
                        turn_completed = True
                        break
                tool_calls = extract_tool_calls(raw_text)
                if len(tool_calls) == 0:
                    tool_calls = self.tool_box.infer_tool_calls(raw_text)
                if self.debug_enabled:
                    self._log(f"Model stop reason: {result.stop_reason}")
                    print("[Assistant][Raw]")
                    print(raw_text)
                if tool_calls:
                    stored_tool_calls = self._append_assistant_message(raw_text, tool_calls=tool_calls)
                    self._record_live_context("Assistant tool-call context recorded from live runtime.")
                    for tool_call, stored_tool_call in zip(tool_calls, stored_tool_calls):
                        if self.session.interrupt_requested:
                            break
                        tool_result = self._execute_tool(tool_call)
                        self._append_tool_message(tool_result, stored_tool_call.get("id"))
                    if self.session.interrupt_requested:
                        break
                    continue

                self._append_assistant_message(raw_text)
                self._record_live_context("Assistant context recorded from live runtime.")
                final_user_text = "" if len(self._stream_answer_text.strip()) > 0 else (answer_text or qwen35_text._clean_generated_text(raw_text))
                turn_completed = True
                break
        finally:
            self._hide_status()
            try:
                self._pause_runtime(pause_reason="idle")
            except Exception as exc:
                self._log(f"Pause-after-turn failed: {exc}")
            if self.session.interrupt_requested:
                rollback_assistant_turn(self.session)
            finish_assistant_turn(self.session)
            self.session.runtime_status_note = ""
        if not self.session.interrupt_requested and len(final_user_text.strip()) > 0:
            self._send_chat(final_user_text)
        if turn_completed and not self.session.interrupt_requested and len(self.session.interruption_notice.strip()) > 0:
            if self.debug_enabled:
                self._log("Clearing interruption notice after a successful follow-up turn.")
            self.session.interruption_notice = ""
