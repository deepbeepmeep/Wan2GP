from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
import time
import ffmpeg
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageColor

from shared.utils.audio_video import extract_audio_tracks
from shared.utils.utils import get_video_frame, get_video_info
from shared.deepy.config import (
    DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT,
    DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY,
    DEEPY_CONTEXT_TOKENS_DEFAULT,
    DEEPY_CONTEXT_TOKENS_KEY,
    DEEPY_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_VRAM_MODE_ALWAYS_LOADED,
    DEEPY_VRAM_MODE_UNLOAD,
    DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST,
    get_deepy_config_value,
    normalize_deepy_auto_cancel_queue_tasks,
    normalize_deepy_context_tokens,
    normalize_deepy_custom_system_prompt,
    normalize_deepy_vram_mode,
)
from shared.deepy import DEFAULT_SYSTEM_PROMPT as ASSISTANT_SYSTEM_PROMPT
from shared.deepy import media_registry, tool_settings as deepy_tool_settings, transcription as deepy_transcription, ui_settings as deepy_ui_settings, video_tools as deepy_video_tools, vision as deepy_vision
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
    "list": "array",
    "dict": "object",
}
_AI_GEN_NO = 0
_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
_DEEPY_DOCS = {
    "finetunes": {"title": "Finetunes", "path": _DOCS_DIR / "FINETUNES.md"},
    "getting_started": {"title": "Getting Started", "path": _DOCS_DIR / "GETTING_STARTED.md"},
    "loras": {"title": "Loras", "path": _DOCS_DIR / "LORAS.md"},
    "overview": {"title": "Overview", "path": _DOCS_DIR / "OVERVIEW.md"},
    "prompts": {"title": "Prompts", "path": _DOCS_DIR / "PROMPTS.md"},
    "vace": {"title": "VACE", "path": _DOCS_DIR / "VACE.md"},
}
_DOC_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_DOC_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SELECTED_REFERENCE_RE = re.compile(r"\b(selected|current(?:ly)?\s+selected|current\s+(?:item|media))\b", flags=re.IGNORECASE)
_RUNTIME_UPDATE_BLOCK_RE = re.compile(r"\s*<wangp_runtime_update>.*?</wangp_runtime_update>\s*", flags=re.DOTALL | re.IGNORECASE)
_POST_TRIM_WINDOW_FRACTION = 0.25
_INJECT_SELECTED_MEDIA_RUNTIME_UPDATES = False
_RUNTIME_STATUS_VISUAL_KEYS = (
    "selected_visual_media_id",
    "selected_visual_media_type",
    "selected_visual_media_label",
    "selected_visual_current_time_seconds",
    "selected_visual_current_frame_no",
)
_RUNTIME_STATUS_AUDIO_KEYS = (
    "selected_audio_media_id",
    "selected_audio_media_type",
    "selected_audio_media_label",
)
_RUNTIME_STATUS_ALL_KEYS = _RUNTIME_STATUS_VISUAL_KEYS + _RUNTIME_STATUS_AUDIO_KEYS


def set_assistant_debug(enabled: bool) -> None:
    global ASSISTANT_DEBUG
    ASSISTANT_DEBUG = bool(enabled)


def _json_type_from_annotation(annotation) -> str:
    annotation_name = getattr(annotation, "__name__", str(annotation))
    if annotation_name.startswith("list["):
        return "array"
    if annotation_name.startswith("dict["):
        return "object"
    return _TOOL_TYPE_MAP.get(annotation_name, "string")


def _build_tool_parameter_schema(annotations: dict[str, Any], param_name: str, param_meta: dict[str, Any]) -> dict[str, Any]:
    schema = {
        "type": param_meta.get("type") or _json_type_from_annotation(annotations.get(param_name, str)),
        "description": str(param_meta.get("description", "")).strip(),
    }
    for meta_key, meta_value in param_meta.items():
        if meta_key in {"description", "required", "type"}:
            continue
        schema[meta_key] = copy.deepcopy(meta_value)
    return schema


def _get_main_callable(name: str) -> Any:
    main_module = sys.modules.get("__main__")
    return None if main_module is None else getattr(main_module, str(name or "").strip(), None)


def _get_main_attribute(name: str) -> Any:
    lookup_name = str(name or "").strip()
    if len(lookup_name) == 0:
        return None
    for module_name in ("__main__", "wgp"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        value = getattr(module, lookup_name, None)
        if value is not None:
            return value
    return None


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


def _doc_relative_path(doc_path: Path) -> str:
    return str(doc_path.relative_to(_DOCS_DIR.parent)).replace("\\", "/")


def _normalize_doc_text(value: str) -> str:
    return " ".join(_DOC_TOKEN_RE.findall(str(value or "").lower()))


def _tokenize_doc_query(value: str) -> list[str]:
    return _DOC_TOKEN_RE.findall(str(value or "").lower())


def _extract_doc_sections(doc_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lookup_id = str(doc_id or "").strip().lower()
    doc_entry = _DEEPY_DOCS.get(lookup_id, None)
    if doc_entry is None:
        raise KeyError(lookup_id)
    doc_path = Path(doc_entry["path"])
    content = doc_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = content.split("\n") if len(content) > 0 else []
    headings = []
    in_code_block = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = _DOC_HEADING_RE.match(line)
        if match is None:
            continue
        headings.append((index, len(match.group(1)), match.group(2).strip()))
    include_top_level = not any(level > 1 for _line_no, level, _title in headings)
    sections = []
    stack: list[tuple[int, str]] = []
    for heading_index, (start_line, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        if not include_top_level and level == 1:
            continue
        end_line = len(lines)
        for next_start_line, next_level, _next_title in headings[heading_index + 1 :]:
            if next_level <= level:
                end_line = next_start_line
                break
        section_parts = [item_title for item_level, item_title in stack if include_top_level or item_level > 1]
        section_name = " > ".join(section_parts or [title])
        markdown = "\n".join(lines[start_line:end_line]).strip()
        body = "\n".join(lines[start_line + 1 : end_line]).strip()
        sections.append(
            {
                "section": section_name,
                "heading": title,
                "heading_level": int(level),
                "content": markdown,
                "body": body,
            }
        )
    if not sections and len(content) > 0:
        sections.append(
            {
                "section": str(doc_entry["title"]).strip() or lookup_id,
                "heading": str(doc_entry["title"]).strip() or lookup_id,
                "heading_level": 1,
                "content": content,
                "body": content,
            }
        )
    return {
        "doc_id": lookup_id,
        "title": str(doc_entry["title"]).strip() or lookup_id,
        "path": _doc_relative_path(doc_path),
    }, sections


def _build_doc_excerpt(section: dict[str, Any], query: str, query_tokens: list[str], limit: int = 260) -> str:
    lines = [line.strip() for line in str(section.get("body", "") or "").splitlines() if len(line.strip()) > 0]
    if not lines:
        lines = [line.strip() for line in str(section.get("content", "") or "").splitlines() if len(line.strip()) > 0]
    if not lines:
        return ""
    query_lower = str(query or "").strip().lower()
    best_line = ""
    if len(query_lower) > 0:
        best_line = next((line for line in lines if query_lower in line.lower()), "")
    if len(best_line) == 0 and query_tokens:
        best_line = max(lines, key=lambda line: sum(token in line.lower() for token in query_tokens))
    if len(best_line) == 0:
        best_line = lines[0]
    best_line = re.sub(r"\s+", " ", best_line).strip()
    return best_line if len(best_line) <= limit else best_line[: limit - 3].rstrip() + "..."


def _score_doc_section(query: str, query_tokens: list[str], doc_title: str, section: dict[str, Any]) -> int:
    query_lower = str(query or "").strip().lower()
    path_text = f"{doc_title} {section.get('section', '')}".lower()
    content_text = str(section.get("body", "") or section.get("content", "")).lower()
    score = 0
    if len(query_lower) > 0 and query_lower in path_text:
        score += 100
    if len(query_lower) > 0 and query_lower in content_text:
        score += 40
    for token in query_tokens:
        if token in path_text:
            score += 12
        if token in content_text:
            score += 3
    return score


def _resolve_doc_section(doc_id: str, section_name: str) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    doc_info, sections = _extract_doc_sections(doc_id)
    normalized_target = _normalize_doc_text(section_name)
    if len(normalized_target) == 0:
        return doc_info, {}, []
    exact_path_matches = [section for section in sections if _normalize_doc_text(section["section"]) == normalized_target]
    if len(exact_path_matches) == 1:
        return doc_info, exact_path_matches[0], []
    exact_heading_matches = [section for section in sections if _normalize_doc_text(section["heading"]) == normalized_target]
    if len(exact_path_matches) == 0 and len(exact_heading_matches) == 1:
        return doc_info, exact_heading_matches[0], []
    partial_matches = [section for section in sections if normalized_target in _normalize_doc_text(section["section"])]
    if len(exact_path_matches) == 0 and len(exact_heading_matches) == 0 and len(partial_matches) == 1:
        return doc_info, partial_matches[0], []
    candidate_matches = exact_path_matches or exact_heading_matches or partial_matches
    candidate_names = [str(section["section"]) for section in candidate_matches[:5]]
    return doc_info, {}, candidate_names


def _format_avg_tokens_per_second(value: float) -> str:
    try:
        speed = float(value or 0.0)
    except Exception:
        speed = 0.0
    if not math.isfinite(speed) or speed < 0.0:
        speed = 0.0
    return f"{speed:.1f}"


def build_assistant_chat_stats(
    session: AssistantSessionState,
    *,
    max_tokens: int,
    active_sequence_token_count: int | None = None,
    live_prefill_tokens: int = 0,
    live_prefill_seconds: float = 0.0,
    live_generated_tokens: int = 0,
    live_generation_seconds: float = 0.0,
) -> dict[str, Any]:
    max_tokens = max(0, int(max_tokens or 0))
    consumed_tokens = None if active_sequence_token_count is None else max(0, int(active_sequence_token_count))
    if consumed_tokens is None:
        snapshot_sequence = None if session.runtime_snapshot is None else session.runtime_snapshot.get("sequence", None)
        if isinstance(snapshot_sequence, dict):
            snapshot_token_ids = snapshot_sequence.get("token_ids", []) or []
            if len(snapshot_token_ids) > 0:
                consumed_tokens = len(snapshot_token_ids)
    if consumed_tokens is None:
        consumed_tokens = len(session.rendered_token_ids or [])
    total_prefill_tokens = max(0, int(session.prefill_token_total or 0)) + max(0, int(live_prefill_tokens or 0))
    total_prefill_seconds = max(0.0, float(session.prefill_seconds_total or 0.0)) + max(0.0, float(live_prefill_seconds or 0.0))
    total_generated_tokens = max(0, int(session.generated_token_total or 0)) + max(0, int(live_generated_tokens or 0))
    total_generation_seconds = max(0.0, float(session.generated_seconds_total or 0.0)) + max(0.0, float(live_generation_seconds or 0.0))
    avg_prefill_tokens_per_second = (float(total_prefill_tokens) / float(total_prefill_seconds)) if total_prefill_seconds > 1e-9 else 0.0
    avg_generated_tokens_per_second = (float(total_generated_tokens) / float(total_generation_seconds)) if total_generation_seconds > 1e-9 else 0.0
    return {
        "visible": True,
        "text": f"prefill {_format_avg_tokens_per_second(avg_prefill_tokens_per_second)} tk/s | gen {_format_avg_tokens_per_second(avg_generated_tokens_per_second)} tk/s | {int(consumed_tokens):,} / {int(max_tokens):,} tk",
        "avg_prefill_tokens_per_second": avg_prefill_tokens_per_second,
        "avg_generated_tokens_per_second": avg_generated_tokens_per_second,
        "consumed_tokens": int(consumed_tokens),
        "max_tokens": int(max_tokens),
    }


@dataclass(slots=True)
class AssistantSessionState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    rendered_token_ids: list[int] = field(default_factory=list)
    rendered_messages_len: int = 0
    runtime_snapshot: dict[str, Any] | None = None
    discard_runtime_snapshot_on_release: bool = False
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
    prefill_token_total: int = 0
    prefill_seconds_total: float = 0.0
    generated_token_total: int = 0
    generated_seconds_total: float = 0.0
    runtime_max_model_len: int = 0
    chat_stats_signature: str = ""
    seen_video_gallery_paths: list[str] = field(default_factory=list)
    seen_audio_gallery_paths: list[str] = field(default_factory=list)
    generated_client_ids: list[str] = field(default_factory=list)
    selected_visual_runtime_signature: str = ""
    selected_audio_runtime_signature: str = ""
    video_tool_runtime_variants: dict[str, str] = field(default_factory=dict)


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
    session.discard_runtime_snapshot_on_release = False
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
    session.prefill_token_total = 0
    session.prefill_seconds_total = 0.0
    session.generated_token_total = 0
    session.generated_seconds_total = 0.0
    session.runtime_max_model_len = 0
    session.chat_stats_signature = ""
    session.seen_video_gallery_paths = []
    session.seen_audio_gallery_paths = []
    session.generated_client_ids = []
    session.selected_visual_runtime_signature = ""
    session.selected_audio_runtime_signature = ""
    session.video_tool_runtime_variants = {}
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
        "chat_transcript": copy.deepcopy(session.chat_transcript),
        "chat_transcript_counter": int(session.chat_transcript_counter or 0),
    }


def mark_assistant_turn_message(session: AssistantSessionState, message_id: str) -> None:
    checkpoint = session.current_turn
    if not isinstance(checkpoint, dict):
        return
    checkpoint["assistant_message_id"] = str(message_id or "").strip()


def checkpoint_assistant_turn(session: AssistantSessionState) -> bool:
    checkpoint = session.current_turn
    if not isinstance(checkpoint, dict):
        return False
    checkpoint["messages_len"] = len(session.messages)
    checkpoint["rendered_token_ids"] = [int(token_id) for token_id in session.rendered_token_ids]
    checkpoint["rendered_messages_len"] = int(session.rendered_messages_len or 0)
    checkpoint["runtime_snapshot"] = None if session.runtime_snapshot is None else copy.deepcopy(session.runtime_snapshot)
    checkpoint["rendered_system_prompt_signature"] = str(session.rendered_system_prompt_signature or "")
    checkpoint["rendered_context_window_tokens"] = int(session.rendered_context_window_tokens or 0)
    checkpoint["chat_transcript"] = copy.deepcopy(session.chat_transcript)
    checkpoint["chat_transcript_counter"] = int(session.chat_transcript_counter or 0)
    return True


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
    transcript_snapshot = checkpoint.get("chat_transcript", None)
    if isinstance(transcript_snapshot, list):
        session.chat_transcript = copy.deepcopy(transcript_snapshot)
        try:
            session.chat_transcript_counter = int(checkpoint.get("chat_transcript_counter", session.chat_transcript_counter) or 0)
        except Exception:
            pass
    else:
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

    def _interrupted_result(self, client_id: str, task: dict[str, Any], *, force_cancel_queue: bool = False) -> dict[str, Any]:
        self._log(f"Generation interrupted for {client_id}")
        cancel_result = {}
        if (force_cancel_queue or self._auto_cancel_queue_tasks_enabled()) and len(str(client_id or "").strip()) > 0:
            queue = list((self.gen or {}).get("queue", []) or [])
            if self._queue_contains_client_id(queue, client_id):
                self.send_cmd("abort_client_id", str(client_id))
                cancel_result = {"client_id": str(client_id), "mode": "abort_client_id"}
            elif self._clear_inline_queue_client_id(client_id):
                cancel_result = {"client_id": str(client_id), "mode": "inline_queue"}
        result = {
            "status": "interrupted",
            "client_id": client_id,
            "output_file": "",
            "prompt": task["prompt"],
            "resolution": task["resolution"],
            "error": "Interrupted by user.",
        }
        if isinstance(cancel_result, dict) and len(cancel_result) > 0:
            result["queue_cancel"] = cancel_result
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

    def _auto_cancel_queue_tasks_enabled(self) -> bool:
        return normalize_deepy_auto_cancel_queue_tasks(self._server_config().get(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT))

    def _clear_inline_queue_client_id(self, client_id: str) -> bool:
        client_id = str(client_id or "").strip()
        if len(client_id) == 0 or not isinstance(self.gen, dict):
            return False
        def _matches(item):
            if not isinstance(item, dict):
                return False
            if str(item.get("client_id", "") or "").strip() == client_id:
                return True
            params = item.get("params", None)
            return isinstance(params, dict) and str(params.get("client_id", "") or "").strip() == client_id
        inline_queue = self.gen.get("inline_queue", None)
        if _matches(inline_queue):
            self.gen.pop("inline_queue", None)
            return True
        if isinstance(inline_queue, list):
            remaining_inline = [item for item in inline_queue if not _matches(item)]
            if len(remaining_inline) != len(inline_queue):
                if remaining_inline:
                    self.gen["inline_queue"] = remaining_inline
                else:
                    self.gen.pop("inline_queue", None)
                return True
        return False

    def _get_effective_tool_model_def(self, tool_name: str) -> dict[str, Any]:
        variant = self.get_tool_variant(tool_name)
        if len(variant) == 0:
            return {}
        try:
            model_def = deepy_tool_settings.get_tool_variant_model_def(tool_name, variant)
        except Exception:
            return {}
        return dict(model_def or {}) if isinstance(model_def, dict) else {}

    def _get_deepy_tool_config(self, tool_name: str) -> dict[str, Any]:
        deepy_tools = self._get_effective_tool_model_def(tool_name).get("deepy_tools", None)
        if not isinstance(deepy_tools, dict):
            return {}
        tool_config = deepy_tools.get(str(tool_name or "").strip(), None)
        return dict(tool_config or {}) if isinstance(tool_config, dict) else {}

    def _get_image_start_target(self, tool_name: str) -> str:
        target = str(self._get_deepy_tool_config(tool_name).get("image_start", "image_start") or "image_start").strip()
        return "image_refs" if target == "image_refs" else "image_start"

    def get_tool_variant(self, tool_name: str) -> str:
        lookup_name = str(tool_name or "").strip()
        setting_key = {
            "gen_image": "image_generator_variant",
            "edit_image": "image_editor_variant",
            "gen_video": "video_generator_variant",
            "gen_video_with_speech": "video_with_speech_variant",
            "gen_speech_from_description": "speech_from_description_variant",
            "gen_speech_from_sample": "speech_from_sample_variant",
        }.get(lookup_name, "")
        if len(setting_key) > 0:
            return str(self._get_tool_ui_settings().get(setting_key, "") or "").strip()
        return ""

    def get_tool_template_filename(self, tool_name: str) -> str:
        try:
            variant = self.get_tool_variant(tool_name)
        except Exception:
            variant = ""
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
        if str(tool_name or "").strip() not in {"gen_image", "edit_image", "gen_video", "gen_speech_from_description", "gen_speech_from_sample", "gen_video_with_speech"}:
            return label
        template_label = Path(self.get_tool_template_filename(tool_name)).stem.strip()
        return label if len(template_label) == 0 else f"{label} [{template_label}]"

    def _parse_generation_resolution(self, resolution: Any) -> tuple[int | None, int | None]:
        width_text, separator, height_text = str(resolution or "").strip().lower().partition("x")
        if separator != "x":
            return None, None
        try:
            return int(width_text), int(height_text)
        except Exception:
            return None, None

    def _is_video_generation_tool(self, tool_name: str) -> bool:
        return str(tool_name or "").strip() in {"gen_video", "gen_video_with_speech"}

    def _supports_inference_steps_override(self, tool_name: str) -> bool:
        return str(tool_name or "").strip() in {"gen_image", "edit_image", "gen_video", "gen_video_with_speech"}

    def _compute_effective_video_fps(self, task: dict[str, Any]) -> int | None:
        force_fps = str(task.get("force_fps", "") or "").strip()
        model_type = str(task.get("model_type", "") or task.get("base_model_type", "") or "").strip()
        video_guide = str(task.get("video_guide", "") or "").strip() or None
        video_source = str(task.get("video_source", "") or "").strip() or None
        get_computed_fps = _get_main_callable("get_computed_fps")
        if callable(get_computed_fps) and len(model_type) > 0:
            try:
                return int(round(float(get_computed_fps(force_fps, model_type, video_guide, video_source))))
            except Exception:
                pass
        if len(force_fps) > 0:
            try:
                return int(force_fps)
            except Exception:
                pass
        get_base_model_type = _get_main_callable("get_base_model_type")
        base_model_type = model_type
        if callable(get_base_model_type) and len(model_type) > 0:
            try:
                base_model_type = str(get_base_model_type(model_type) or model_type).strip() or model_type
            except Exception:
                base_model_type = model_type
        get_model_fps = _get_main_callable("get_model_fps")
        if callable(get_model_fps) and len(base_model_type) > 0:
            try:
                return int(round(float(get_model_fps(base_model_type))))
            except Exception:
                return None
        return None

    def _get_effective_video_latent_size(self, task: dict[str, Any]) -> int | None:
        model_type = str(task.get("model_type", "") or task.get("base_model_type", "") or "").strip()
        get_base_model_type = _get_main_callable("get_base_model_type")
        base_model_type = model_type
        if callable(get_base_model_type) and len(model_type) > 0:
            try:
                base_model_type = str(get_base_model_type(model_type) or model_type).strip() or model_type
            except Exception:
                base_model_type = model_type
        get_model_min_frames_and_step = _get_main_callable("get_model_min_frames_and_step")
        if callable(get_model_min_frames_and_step) and len(base_model_type) > 0:
            try:
                _frames_minimum, _frames_steps, latent_size = get_model_min_frames_and_step(base_model_type)
                latent_size = int(latent_size)
                if latent_size > 0:
                    return latent_size
            except Exception:
                pass
        get_model_def = _get_main_callable("get_model_def")
        if callable(get_model_def) and len(base_model_type) > 0:
            try:
                model_def = get_model_def(base_model_type)
            except Exception:
                model_def = None
            if isinstance(model_def, dict):
                try:
                    latent_size = int(model_def.get("latent_size", model_def.get("frames_steps", 0)) or 0)
                except Exception:
                    latent_size = 0
                if latent_size > 0:
                    return latent_size
        return None

    @staticmethod
    def _snap_video_frame_count_to_latent_grid(frame_count: int, latent_size: int | None) -> int:
        raw_frames = int(frame_count)
        if raw_frames <= 0:
            return raw_frames
        if latent_size is None or int(latent_size) <= 0:
            return raw_frames
        step = int(latent_size)
        return max(1, int(round((raw_frames - 1) / float(step))) * step + 1)

    def _get_effective_generation_defaults(self, tool_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        lookup_name = str(tool_name or "").strip()
        if lookup_name not in deepy_tool_settings.GENERATION_TOOL_IDS:
            return None, {
                "status": "error",
                "tool_id": lookup_name,
                "error": f"tool_id must be one of: {', '.join(deepy_tool_settings.GENERATION_TOOL_IDS)}.",
            }
        generator_variant = self.get_tool_variant(lookup_name)
        try:
            task = deepy_tool_settings.build_generation_task(lookup_name, generator_variant, prompt="", client_id="__deepy_defaults__")
        except Exception as exc:
            return None, {
                "status": "error",
                "tool_id": lookup_name,
                "template": generator_variant,
                "error": str(exc),
            }
        include_num_frames = self._is_video_generation_tool(lookup_name)
        task, error_result = self._apply_generation_overrides(lookup_name, task, include_num_frames=include_num_frames)
        if error_result is not None:
            error_result["tool_id"] = lookup_name
            error_result["template"] = generator_variant
            return None, error_result
        model_def = self._get_effective_tool_model_def(lookup_name)
        audio_only = bool(model_def.get("audio_only", False))
        width = height = None
        if not audio_only:
            width, height = self._parse_generation_resolution(task.get("resolution", ""))
        seed = task.get("seed", None)
        try:
            seed = None if seed is None or str(seed).strip() == "" else int(seed)
        except Exception:
            seed = None
        result = {
            "status": "ok",
            "tool_id": lookup_name,
            "template": generator_variant,
            "width": width,
            "height": height,
            "seed": seed,
        }
        if self._supports_inference_steps_override(lookup_name):
            try:
                num_inference_steps = task.get("num_inference_steps", None)
                result["num_inference_steps"] = None if num_inference_steps is None or str(num_inference_steps).strip() == "" else int(num_inference_steps)
            except Exception:
                result["num_inference_steps"] = None
        if include_num_frames:
            result["num_frames"] = None if task.get("video_length", None) is None else int(task.get("video_length"))
            result["fps"] = self._compute_effective_video_fps(task)
        if lookup_name == "gen_video":
            result["multimedia_generation"] = bool(model_def.get("multimedia_generation", False))
        return result, None

    def _apply_generation_overrides(
        self,
        tool_name: str,
        task: dict[str, Any],
        *,
        include_num_frames: bool,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        duration_seconds: float | None = None,
        fps: int | None = None,
        num_inference_steps: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        ui_settings = self._get_tool_ui_settings()
        if ui_settings["use_template_properties"]:
            base_resolution = str(task.get("resolution", "") or "").strip()
            base_num_frames = task.get("video_length", None) if include_num_frames else None
        else:
            base_resolution = f"{ui_settings['width']}x{ui_settings['height']}"
            task["seed"] = int(ui_settings["seed"])
            if include_num_frames:
                base_num_frames = int(ui_settings["num_frames"])
        default_width = default_height = None
        if len(base_resolution) > 0:
            default_width, default_height = self._parse_generation_resolution(base_resolution)
        try:
            width = None if width is None or str(width).strip() == "" else int(width)
            height = None if height is None or str(height).strip() == "" else int(height)
        except Exception:
            return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": base_resolution, "error": "width and height must be integers."}
        if width is None or height is None:
            if default_width is None or default_height is None or default_width <= 0 or default_height <= 0:
                if width is not None or height is not None:
                    return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": base_resolution, "error": "width and height must both be provided because the template/default settings do not define a valid resolution."}
                return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": base_resolution, "error": "Template/default settings do not define a valid resolution."}
            width = default_width if width is None else width
            height = default_height if height is None else height
        min_dim = int(deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MIN)
        max_dim = int(deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MAX)
        if width < min_dim or width > max_dim or height < min_dim or height > max_dim:
            return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": f"{width}x{height}", "error": f"width and height must stay between {min_dim} and {max_dim}."}
        parsed_duration_seconds = None
        if include_num_frames:
            parsed_duration_seconds, error_result = self._parse_time_value(duration_seconds, "duration_seconds", required=False)
            if error_result is not None:
                return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": f"{width}x{height}", "error": str(error_result.get("error", "") or "duration_seconds is invalid.")}
            if parsed_duration_seconds is not None:
                if parsed_duration_seconds <= 0:
                    return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": f"{width}x{height}", "error": "duration_seconds must be > 0."}
                if num_frames is not None and str(num_frames).strip() != "":
                    return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": f"{width}x{height}", "error": "Specify either num_frames or duration_seconds, not both."}
        task["resolution"] = f"{width}x{height}"
        if fps is not None:
            try:
                fps = int(fps)
            except Exception:
                return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": task["resolution"], "error": "fps must be an integer."}
            if fps < 15 or fps > 60:
                return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": task["resolution"], "error": "fps must stay between 15 and 60."}
            task["force_fps"] = str(int(fps))
        if include_num_frames:
            if parsed_duration_seconds is not None:
                effective_fps = int(fps) if fps is not None else self._compute_effective_video_fps(task)
                if effective_fps is None or effective_fps <= 0:
                    return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": task["resolution"], "error": "Could not determine FPS to convert duration_seconds. Pass fps explicitly."}
                num_frames = int(round(float(parsed_duration_seconds) * float(effective_fps)))
                num_frames = self._snap_video_frame_count_to_latent_grid(num_frames, self._get_effective_video_latent_size(task))
            try:
                num_frames = base_num_frames if num_frames is None or str(num_frames).strip() == "" else int(num_frames)
            except Exception:
                return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": task["resolution"], "error": "num_frames must be an integer."}
            min_frames = int(deepy_ui_settings.ASSISTANT_OVERRIDE_FRAMES_MIN)
            max_frames = int(deepy_ui_settings.ASSISTANT_OVERRIDE_FRAMES_MAX)
            if num_frames is None or num_frames < min_frames or num_frames > max_frames:
                return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": task["resolution"], "error": f"num_frames must stay between {min_frames} and {max_frames}."}
            task["video_length"] = int(num_frames)
        if num_inference_steps is not None:
            try:
                num_inference_steps = int(num_inference_steps)
            except Exception:
                return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": task["resolution"], "error": "num_inference_steps must be an integer."}
            if num_inference_steps <= 0:
                return None, {"status": "error", "client_id": str(task.get("client_id", "") or "").strip(), "output_file": "", "prompt": str(task.get("prompt", "") or "").strip(), "resolution": task["resolution"], "error": "num_inference_steps must be a positive integer."}
            task["num_inference_steps"] = int(num_inference_steps)
        return task, None

    def _build_generation_task(self, tool_name: str, variant: str, *, prompt: str, client_id: str, **kwargs) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        self._remember_generated_client_id(client_id)
        try:
            task = deepy_tool_settings.build_generation_task(tool_name, variant, prompt=prompt, client_id=client_id, **kwargs)
        except ValueError as exc:
            return None, {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": str(prompt or "").strip(),
                "error": str(exc),
            }
        return task, None

    def _sync_recent_media(self, max_items: int = 5) -> None:
        if self.session is None:
            return
        file_list, file_settings_list, audio_file_list, audio_file_settings_list = self.get_processed_queue(self.gen)
        media_registry.sync_recent_generated_media(self.session, file_list, file_settings_list, max_items=max_items)
        media_registry.sync_recent_generated_media(self.session, audio_file_list, audio_file_settings_list, max_items=max_items)

    def _remember_generated_client_id(self, client_id: str) -> None:
        if self.session is None:
            return
        normalized_client_id = str(client_id or "").strip()
        if len(normalized_client_id) == 0:
            return
        generated_client_ids = [str(value or "").strip() for value in list(self.session.generated_client_ids or []) if len(str(value or "").strip()) > 0]
        if normalized_client_id in generated_client_ids:
            return
        generated_client_ids.append(normalized_client_id)
        self.session.generated_client_ids = generated_client_ids

    def _register_gallery_media_record(self, media_path: str, settings: dict[str, Any] | None) -> dict[str, Any] | None:
        if self.session is None:
            return None
        normalized_path = str(media_path or "").strip()
        if len(normalized_path) == 0:
            return None
        resolved_settings = settings if isinstance(settings, dict) else None
        client_id = "" if resolved_settings is None else str(resolved_settings.get("client_id", "") or "").strip()
        return media_registry.register_media(
            self.session,
            normalized_path,
            settings=resolved_settings,
            source="deepy" if client_id in {str(value or "").strip() for value in list(self.session.generated_client_ids or []) if len(str(value or "").strip()) > 0} else "wangp",
            client_id=client_id,
        )

    def _get_new_user_gallery_media(self) -> dict[str, dict[str, Any]]:
        if self.session is None:
            return {}
        file_list, file_settings_list, audio_file_list, audio_file_settings_list = self.get_processed_queue(self.gen)
        generated_client_ids = {str(value or "").strip() for value in list(self.session.generated_client_ids or []) if len(str(value or "").strip()) > 0}
        media_updates = {}
        gallery_groups = (
            ("seen_video_gallery_paths", list(file_list or []), list(file_settings_list or [])),
            ("seen_audio_gallery_paths", list(audio_file_list or []), list(audio_file_settings_list or [])),
        )
        for session_attr, gallery_files, gallery_settings in gallery_groups:
            previous_files = [str(path or "").strip() for path in getattr(self.session, session_attr, []) if len(str(path or "").strip()) > 0]
            current_pairs = [(str(path or "").strip(), gallery_settings[index] if index < len(gallery_settings) and isinstance(gallery_settings[index], dict) else None) for index, path in enumerate(gallery_files) if len(str(path or "").strip()) > 0]
            current_files = [path for path, _settings in current_pairs]
            appended_start = len(previous_files) if len(previous_files) <= len(current_files) and current_files[: len(previous_files)] == previous_files else len(current_files)
            setattr(self.session, session_attr, list(current_files))
            if appended_start >= len(current_pairs):
                continue
            for media_path, settings in current_pairs[appended_start:]:
                client_id = "" if not isinstance(settings, dict) else str(settings.get("client_id", "") or "").strip()
                if len(client_id) > 0 and client_id in generated_client_ids:
                    continue
                media_record = self._register_gallery_media_record(media_path, settings)
                media_type = "" if media_record is None else str(media_record.get("media_type", "") or "").strip()
                if media_type in {"image", "video", "audio"}:
                    media_updates[media_type] = media_record
        return media_updates

    def _get_selected_gallery_media_updates(self) -> list[str]:
        if self.session is None:
            return []
        updates = []

        visual_media_record, _error_result = self._get_selected_media_record_from_source("video", "all")
        visual_signature = "" if visual_media_record is None else f"{str(visual_media_record.get('media_type', '') or '').strip()}:{str(visual_media_record.get('media_id', '') or '').strip()}"
        if visual_signature != str(self.session.selected_visual_runtime_signature or "") and visual_media_record is not None:
            visual_media_type = str(visual_media_record.get("media_type", "") or "").strip()
            if visual_media_type in {"image", "video"}:
                instruction = self._runtime_media_instruction(
                    visual_media_record,
                    action="selected",
                    gallery_label="Image / Video Gallery",
                    reference_label="selected",
                    why="matched selected media",
                    selected_payload=True,
                )
                if len(instruction) > 0:
                    updates.append(instruction)
        self.session.selected_visual_runtime_signature = visual_signature

        audio_media_record, _error_result = self._get_selected_media_record_from_source("audio", "audio")
        audio_signature = "" if audio_media_record is None else f"audio:{str(audio_media_record.get('media_id', '') or '').strip()}"
        if audio_signature != str(self.session.selected_audio_runtime_signature or "") and audio_media_record is not None:
            instruction = self._runtime_media_instruction(
                audio_media_record,
                action="selected",
                gallery_label="Audio Gallery",
                reference_label="selected",
                why="matched selected media",
                selected_payload=True,
            )
            if len(instruction) > 0:
                updates.append(instruction)
        self.session.selected_audio_runtime_signature = audio_signature

        return updates

    def _queue_contains_client_id(self, queue: list[Any], client_id: str) -> bool:
        lookup_client_id = str(client_id or "").strip()
        if len(lookup_client_id) == 0:
            return False
        return any(isinstance(item, dict) and isinstance(item.get("params"), dict) and str(item["params"].get("client_id", "") or "").strip() == lookup_client_id for item in list(queue or []))

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

    def _normalize_selected_media_type(self, media_type: str | None, reference: str | None = None) -> str:
        normalized = str(media_type or "").strip().lower()
        if normalized in {"image", "video", "audio"}:
            return normalized
        if normalized in {"", "any", "all"}:
            inferred = media_registry.normalize_media_type("any", reference=reference)
            return "all" if inferred == "any" else inferred
        return "all"

    def _selected_media_payload(self, media_record: dict[str, Any], why: str = "") -> dict[str, Any]:
        payload = self._compact_media_payload(media_record, why=why)
        payload["path"] = str(media_record.get("path", "")).strip()
        payload.update(self._get_selected_video_position(media_record))
        return payload

    def _runtime_media_instruction(self, media_record: dict[str, Any], *, action: str, gallery_label: str, reference_label: str, why: str = "", selected_payload: bool = False) -> str:
        media_type = str(media_record.get("media_type", "") or "").strip()
        media_id = str(media_record.get("media_id", "") or "").strip()
        if len(media_type) == 0 or len(media_id) == 0:
            return ""
        payload = self._selected_media_payload(media_record, why=why) if selected_payload else self._compact_media_payload(media_record, why=why)
        return f"The user has {action} {media_type} id {media_id} in the {gallery_label}, use this media id if the user asks you to work on the {reference_label} {media_type}. Media details: {_json_dumps(payload)}"

    def _get_selected_runtime_snapshot(self) -> dict[str, Any] | None:
        snapshot = {}

        visual_media_record, _error_result = self._get_selected_media_record_from_source("video", "all")
        if visual_media_record is not None:
            snapshot["selected_visual_media_id"] = str(visual_media_record.get("media_id", "") or "").strip()
            snapshot["selected_visual_media_type"] = str(visual_media_record.get("media_type", "") or "").strip()
            label = str(visual_media_record.get("label", "") or "").strip()
            if len(label) > 0:
                snapshot["selected_visual_media_label"] = label
            if snapshot["selected_visual_media_type"] == "video":
                video_position = self._get_selected_video_position(visual_media_record)
                if "current_time_seconds" in video_position:
                    snapshot["selected_visual_current_time_seconds"] = video_position["current_time_seconds"]
                if "current_frame_no" in video_position:
                    snapshot["selected_visual_current_frame_no"] = video_position["current_frame_no"]

        audio_media_record, _error_result = self._get_selected_media_record_from_source("audio", "audio")
        if audio_media_record is not None:
            snapshot["selected_audio_media_id"] = str(audio_media_record.get("media_id", "") or "").strip()
            snapshot["selected_audio_media_type"] = str(audio_media_record.get("media_type", "") or "").strip()
            label = str(audio_media_record.get("label", "") or "").strip()
            if len(label) > 0:
                snapshot["selected_audio_media_label"] = label

        return snapshot if len(snapshot) > 1 else None

    def _is_selected_reference(self, reference: str) -> bool:
        return _SELECTED_REFERENCE_RE.search(str(reference or "").strip()) is not None

    def _get_selected_media_record_from_source(self, source: str, requested_media_type: str = "all") -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        requested_label = self._normalize_selected_media_type(requested_media_type)
        if self.session is None:
            return None, {"status": "error", "media_type": requested_label, "error": "Assistant session is not available."}
        file_list, file_settings_list, audio_file_list, audio_file_settings_list = self.get_processed_queue(self.gen)
        source = "audio" if str(source or "").strip().lower() == "audio" else "video"
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
            gallery_label = "audio gallery" if source == "audio" else "image/video gallery"
            return None, {"status": "error", "media_type": requested_label, "error": f"No media is currently selected in the WanGP {gallery_label}."}
        selected_path = str(file_list[choice] or "").strip()
        selected_settings = file_settings_list[choice] if choice < len(file_settings_list) and isinstance(file_settings_list[choice], dict) else None
        selected_client_id = str((selected_settings or {}).get("client_id", "") or "").strip()
        selected_gallery_media_type = "audio" if source == "audio" else "video"
        if len(selected_client_id) > 0 and (source == "audio" or deepy_video_tools.has_video_extension(selected_path)):
            latest_path, latest_settings = media_registry.find_last_gallery_media_by_client(file_list, file_settings_list, selected_client_id, media_type=selected_gallery_media_type)
            if latest_path is not None:
                selected_path = latest_path
                selected_settings = latest_settings if isinstance(latest_settings, dict) else None
        media_record = media_registry.register_media(
            self.session,
            selected_path,
            settings=selected_settings,
            source="deepy" if str((selected_settings or {}).get("client_id", "") or "").strip().startswith("ai_") else "wangp",
            client_id=str((selected_settings or {}).get("client_id", "") or "").strip(),
        )
        if media_record is None:
            return None, {"status": "error", "media_type": requested_label, "error": "The currently selected gallery item is not a supported media file."}
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

    def _get_all_selected_media_records(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        visual_media_record, _visual_error = self._get_selected_media_record_from_source("video", "all")
        audio_media_record, _audio_error = self._get_selected_media_record_from_source("audio", "audio")
        if visual_media_record is None and audio_media_record is None:
            return None, None, {"status": "error", "media_type": "all", "error": "No media is currently selected in either WanGP gallery."}
        return visual_media_record, audio_media_record, None

    def _get_selected_media_record(self, requested_media_type: str = "all") -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        resolved_media_type = self._normalize_selected_media_type(requested_media_type)
        if resolved_media_type == "audio":
            return self._get_selected_media_record_from_source("audio", "audio")
        if resolved_media_type in {"image", "video"}:
            return self._get_selected_media_record_from_source("video", resolved_media_type)
        visual_media_record, audio_media_record, error_result = self._get_all_selected_media_records()
        if error_result is not None:
            return None, error_result
        if visual_media_record is None:
            return audio_media_record, None
        if audio_media_record is None:
            return visual_media_record, None
        return None, {
            "status": "error",
            "media_type": "all",
            "error": "Both a visual selection and an audio selection exist. Request image, video, or audio explicitly, or use Get Selected Media with media_type='all'.",
        }

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

    def _record_direct_media(self, output_path: str, settings: dict[str, Any], *, is_image: bool, audio_only: bool, label: str | None = None, persist_metadata: bool = True) -> dict[str, Any] | None:
        if not os.path.isfile(output_path):
            raise RuntimeError(f"Output file was not created: {output_path}")
        if not callable(self.record_file_metadata):
            raise RuntimeError("WanGP direct media recording is not available.")
        self.record_file_metadata(output_path, settings if persist_metadata else None, is_image, audio_only, self.gen)
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

    def _resolve_audio_or_video_media(self, media_id: str, parameter_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        media_id = str(media_id or "").strip()
        if len(media_id) == 0:
            return None, {"status": "error", parameter_name: media_id, "error": f"{parameter_name} is required."}
        if self.session is None:
            return None, {"status": "error", parameter_name: media_id, "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return None, {"status": "error", parameter_name: media_id, "error": f"Unknown media id for {parameter_name}."}
        if media_record.get("media_type") not in {"audio", "video"}:
            actual_media_type = str(media_record.get("media_type", "") or "").strip() or "unknown media type"
            return None, {
                "status": "error",
                parameter_name: media_record.get("media_id", ""),
                "actual_media_type": actual_media_type,
                "media_type": actual_media_type,
                "error": f"{parameter_name} must reference an audio or video file, not a {actual_media_type}.",
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

    def _parse_int_value(self, value: Any, parameter_name: str, *, required: bool = False) -> tuple[int | None, dict[str, Any] | None]:
        if value is None or str(value).strip() == "":
            return (None, {"status": "error", "error": f"{parameter_name} is required."}) if required else (None, None)
        try:
            resolved = int(value)
        except Exception:
            return None, {"status": "error", "error": f"{parameter_name} must be an integer."}
        if resolved < 0:
            return None, {"status": "error", "error": f"{parameter_name} must be >= 0."}
        return resolved, None

    def _parse_bool_value(self, value: Any, parameter_name: str, *, required: bool = False) -> tuple[bool | None, dict[str, Any] | None]:
        if value is None or str(value).strip() == "":
            return (None, {"status": "error", "error": f"{parameter_name} is required."}) if required else (None, None)
        if isinstance(value, bool):
            return value, None
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True, None
        if normalized in {"false", "0", "no", "off"}:
            return False, None
        return None, {"status": "error", "error": f"{parameter_name} must be true or false."}

    def _resolve_segment_args(
        self,
        source_media: dict[str, Any],
        *,
        start_time: Any = None,
        end_time: Any = None,
        duration: Any = None,
        start_frame: Any = None,
        end_frame: Any = None,
        num_frames: Any = None,
        allow_empty: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        time_inputs = (start_time, end_time, duration)
        frame_inputs = (start_frame, end_frame, num_frames)
        has_time_args = any(value is not None and str(value).strip() != "" for value in time_inputs)
        has_frame_args = any(value is not None and str(value).strip() != "" for value in frame_inputs)
        if has_time_args and has_frame_args:
            return None, {"status": "error", "error": "Use either time-based arguments or frame-based arguments, not both."}
        if not has_time_args and not has_frame_args:
            if allow_empty:
                return {"mode": "time", "start_time": None, "end_time": None, "duration": None, "start_frame": None, "end_frame": None, "num_frames": None}, None
            return None, {"status": "error", "error": "Provide at least one of start_time, end_time, duration, start_frame, end_frame, or num_frames."}
        if has_frame_args:
            if str(source_media.get("media_type", "") or "").strip() != "video":
                return None, {"status": "error", "error": "Frame-based extraction is only supported when media_id references a video."}
            start_frame, error_result = self._parse_int_value(start_frame, "start_frame")
            if error_result is not None:
                return None, error_result
            end_frame, error_result = self._parse_int_value(end_frame, "end_frame")
            if error_result is not None:
                return None, error_result
            num_frames, error_result = self._parse_int_value(num_frames, "num_frames")
            if error_result is not None:
                return None, error_result
            if end_frame is not None and num_frames is not None:
                return None, {"status": "error", "error": "Specify either end_frame or num_frames, not both."}
            start_frame = 0 if start_frame is None else start_frame
            if num_frames is not None and num_frames <= 0:
                return None, {"status": "error", "error": "num_frames must be > 0."}
            media_path = str(source_media.get("path", "")).strip()
            try:
                fps, _width, _height, frame_count = get_video_info(media_path)
            except Exception as exc:
                return None, {"status": "error", "error": str(exc)}
            precise_fps = deepy_video_tools.get_precise_video_fps(media_path)
            effective_fps = float(precise_fps) if precise_fps is not None and precise_fps > 0 else float(fps or 0)
            if effective_fps <= 0:
                return None, {"status": "error", "error": "Could not determine source video FPS for frame-based extraction."}
            max_frame = max(0, int(frame_count) - 1)
            if start_frame > max_frame:
                return None, {"status": "error", "error": f"start_frame must be between 0 and {max_frame}."}
            resolved_end_frame = max_frame
            if end_frame is not None:
                if end_frame < start_frame:
                    return None, {"status": "error", "error": "end_frame must be >= start_frame."}
                resolved_end_frame = min(end_frame, max_frame)
            elif num_frames is not None:
                resolved_end_frame = min(start_frame + num_frames - 1, max_frame)
            resolved_num_frames = max(0, resolved_end_frame - start_frame + 1)
            resolved_start_time = start_frame / effective_fps
            if end_frame is not None:
                resolved_end_time = (resolved_end_frame + 1) / effective_fps
                resolved_duration = None
            elif num_frames is not None:
                resolved_end_time = None
                resolved_duration = resolved_num_frames / effective_fps
            else:
                resolved_end_time = None
                resolved_duration = None
            return {
                "mode": "frame",
                "start_time": resolved_start_time,
                "end_time": resolved_end_time,
                "duration": resolved_duration,
                "start_frame": start_frame,
                "end_frame": resolved_end_frame,
                "num_frames": resolved_num_frames,
            }, None
        start_time, error_result = self._parse_time_value(start_time, "start_time")
        if error_result is not None:
            return None, error_result
        end_time, error_result = self._parse_time_value(end_time, "end_time")
        if error_result is not None:
            return None, error_result
        duration, error_result = self._parse_time_value(duration, "duration")
        if error_result is not None:
            return None, error_result
        if end_time is not None and duration is not None:
            return None, {"status": "error", "error": "Specify either end_time or duration, not both."}
        if start_time is None:
            start_time = 0.0
        return {"mode": "time", "start_time": start_time, "end_time": end_time, "duration": duration, "start_frame": None, "end_frame": None, "num_frames": None}, None

    def _build_deepy_settings(self, prompt: str, comments: str = "", **updates: Any) -> dict[str, Any]:
        wangp_version = str(_get_main_attribute("WanGP_version") or "").strip()
        settings = {
            "type": f"WanGP v{wangp_version} DeepBeepMeep - Deepy" if len(wangp_version) > 0 else "WanGP DeepBeepMeep - Deepy",
            "model_type": "Deepy",
            "prompt": str(prompt or "").strip(),
            "client_id": _next_ai_client_id(),
        }
        self._remember_generated_client_id(settings["client_id"])
        settings["comments"] = str(comments or "").strip()
        end_time = time.time()
        settings["creation_date"] = datetime.fromtimestamp(end_time).isoformat(timespec="seconds")
        settings["creation_timestamp"] = int(end_time)
        for key, value in updates.items():
            if value is not None:
                settings[key] = value
        return settings

    def _build_direct_media_settings(self, source_media: dict[str, Any], comments: str, fallback_prompt: str | None = None, **updates: Any) -> dict[str, Any]:
        settings = dict(source_media.get("settings", {}) or {})
        if fallback_prompt is not None and (len(settings) == 0 or str(settings.get("model_type", "") or "").strip() == "Deepy"):
            return self._build_deepy_settings(fallback_prompt, comments, **updates)
        settings["client_id"] = _next_ai_client_id()
        self._remember_generated_client_id(settings["client_id"])
        settings["comments"] = str(comments or "").strip()
        end_time = time.time()
        settings["creation_date"] = datetime.fromtimestamp(end_time).isoformat(timespec="seconds")
        settings["creation_timestamp"] = int(end_time)
        for key, value in updates.items():
            if value is not None:
                settings[key] = value
        return settings

    def _build_direct_image_settings(self, comments: str, width: int, height: int, **updates: Any) -> dict[str, Any]:
        return self._build_deepy_settings(updates.pop("prompt", f"An image at {int(width)}x{int(height)}."), comments, image_mode=1, resolution=f"{int(width)}x{int(height)}", **updates)

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

    def _get_output_duration_seconds(self, output_path: str, file_settings: dict[str, Any] | None = None) -> float | None:
        duration = deepy_video_tools.get_media_duration(output_path)
        return None if duration is None else round(duration, 3)

    def _queue_generation_task(self, task: dict[str, Any], *, activity_label: str, output_label: str | None = None, gallery_media_type: str = "image") -> dict[str, Any]:
        if not isinstance(self.gen, dict):
            raise RuntimeError("WanGP generation queue is not available.")
        client_id = str(task.get("client_id", "") or "").strip()
        prompt = str(task.get("prompt", "") or "").strip()
        resolution = str(task.get("resolution", "") or "").strip()
        gen = self.gen
        self.get_processed_queue(gen)
        self._set_status(f"Queueing {activity_label}...", kind="tool")
        self._update_tool_progress("running", "Queued", {"status": "queued", "client_id": client_id, "prompt": prompt, "resolution": resolution})
        task["priority"] = True
        gen["inline_queue"] = task
        self.send_cmd("load_queue_trigger", {"client_id": client_id})
        self._log(f"Queued {activity_label} for {client_id}")

        queue_admitted = False
        queue_wait_started_at = time.time()
        queue_wait_suspended = False
        queue_wait_suspend_logged = False
        activity_console_label = activity_label.capitalize()
        while True:
            if self._is_interrupted():
                return self._interrupted_result(client_id, task, force_cancel_queue=True)
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
            file_list, file_settings_list, audio_file_list, audio_file_settings_list = self.get_processed_queue(gen)
            media_file_list = list(audio_file_list or []) if gallery_media_type == "audio" else list(file_list or [])
            media_settings_list = list(audio_file_settings_list or []) if gallery_media_type == "audio" else list(file_settings_list or [])
            file_path, file_settings = media_registry.find_last_gallery_media_by_client(media_file_list, media_settings_list, client_id, media_type=gallery_media_type)
            if file_path is not None and isinstance(file_settings, dict):
                self._log(f"{activity_label.capitalize()} already completed before queue admission wait observed for {client_id}; skipping browser-style queue admission wait.")
                self._set_status(f"{activity_label.capitalize()} started...", kind="tool")
                self._update_tool_progress("running", "Running", {"status": "running", "client_id": client_id, "prompt": prompt, "resolution": resolution})
                break
            queue = list(gen.get("queue", []) or [])
            if self._queue_contains_client_id(queue, client_id):
                queue_admitted = True
                if queue_wait_suspended:
                    print(f"WanGP back in focus tool {activity_console_label} resumed")
                self._set_status(f"{activity_label.capitalize()} started...", kind="tool")
                self._update_tool_progress("running", "Running", {"status": "running", "client_id": client_id, "prompt": prompt, "resolution": resolution})
                break
            if not queue_wait_suspend_logged and time.time() - queue_wait_started_at >= 10:
                print(f"Tool {activity_console_label} suspended while waiting than WanGP Video Generator gets in focus")
                queue_wait_suspend_logged = True
                queue_wait_suspended = True
            time.sleep(0.25)

        while True:
            if self._is_interrupted():
                return self._interrupted_result(client_id, task, force_cancel_queue=True)
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
            file_list, file_settings_list, audio_file_list, audio_file_settings_list = self.get_processed_queue(gen)
            media_file_list = list(audio_file_list or []) if gallery_media_type == "audio" else list(file_list or [])
            media_settings_list = list(audio_file_settings_list or []) if gallery_media_type == "audio" else list(file_settings_list or [])
            queue = list(gen.get("queue", []) or [])
            client_id_still_in_queue = self._queue_contains_client_id(queue, client_id)
            if client_id_still_in_queue:
                time.sleep(0.5)
                continue
            file_path, file_settings = media_registry.find_last_gallery_media_by_client(media_file_list, media_settings_list, client_id, media_type=gallery_media_type)
            if file_path is not None and isinstance(file_settings, dict):
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
                if gallery_media_type in {"video", "audio"}:
                    result["output_duration"] = self._get_output_duration_seconds(str(file_path), file_settings)
                self._log(f"{activity_label.capitalize()} completed for {client_id}: {file_path}")
                self._set_status(f"{activity_label.capitalize()} finished.", kind="tool")
                self.send_cmd("refresh_gallery", {"path": str(file_path)})
                self._update_tool_progress("done", "Done", result)
                return result
            error_text = f"{activity_label.capitalize()} finished queue processing but no {gallery_media_type} output with client_id '{client_id}' was found in the gallery."
            self._log(error_text)
            self._set_status(error_text, kind="error")
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

    @assistant_tool(
        display_name="Get Loras",
        description="List the available LoRA filenames for one of Deepy's 6 generation tools.",
        parameters={
            "tool_id": {
                "type": "string",
                "description": "Generation tool id: gen_image, edit_image, gen_video, gen_video_with_speech, gen_speech_from_description, or gen_speech_from_sample.",
                "enum": list(deepy_tool_settings.GENERATION_TOOL_IDS),
            },
        },
        pause_runtime=False,
    )
    def get_loras(self, tool_id: str) -> dict[str, Any]:
        lookup_name = str(tool_id or "").strip()
        if lookup_name not in deepy_tool_settings.GENERATION_TOOL_IDS:
            return {
                "status": "error",
                "tool_id": lookup_name,
                "loras": [],
                "count": 0,
                "error": f"tool_id must be one of: {', '.join(deepy_tool_settings.GENERATION_TOOL_IDS)}.",
            }
        generator_variant = self.get_tool_variant(lookup_name)
        template_file = self.get_tool_template_filename(lookup_name)
        try:
            loras = deepy_tool_settings.list_tool_loras(lookup_name, generator_variant)
        except Exception as exc:
            return {
                "status": "error",
                "tool_id": lookup_name,
                "generator_variant": generator_variant,
                "template_file": template_file,
                "loras": [],
                "count": 0,
                "error": str(exc),
            }
        return {
            "status": "ok",
            "tool_id": lookup_name,
            "generator_variant": generator_variant,
            "template_file": template_file,
            "loras": loras,
            "count": len(loras),
        }

    @assistant_tool(
        display_name="Get Default Settings",
        description="Return the effective default generation settings for one of Deepy's 6 generation tools: the values that WanGP will use if those settings are omitted during generation.",
        parameters={
            "tool_id": {
                "type": "string",
                "description": "Generation tool id: gen_image, edit_image, gen_video, gen_video_with_speech, gen_speech_from_description, or gen_speech_from_sample.",
                "enum": list(deepy_tool_settings.GENERATION_TOOL_IDS),
            },
        },
        pause_runtime=False,
    )
    def get_default_settings(self, tool_id: str) -> dict[str, Any]:
        result, error_result = self._get_effective_generation_defaults(tool_id)
        return result if error_result is None else error_result

    @assistant_tool(
        display_name="Generate Image",
        description="Queue and generate an image from a text prompt inside WanGP, then wait until the output image is available.",
        parameters={
            "prompt": {
                "type": "string",
                "description": "The image generation prompt to send to WanGP.",
            },
            "width": {
                "type": "integer",
                "description": "Optional output width in pixels. Only pass this when the user explicitly asks for output size; otherwise omit it and use the current Deepy/template setting.",
                "required": False,
            },
            "height": {
                "type": "integer",
                "description": "Optional output height in pixels. Only pass this when the user explicitly asks for output size; otherwise omit it and use the current Deepy/template setting.",
                "required": False,
            },
            "num_inference_steps": {
                "type": "integer",
                "description": "Optional number of inference steps. If omitted, keep the template step count.",
                "required": False,
            },
        },
    )
    def gen_image(self, prompt: str, width: int | None = None, height: int | None = None, num_inference_steps: int | None = None) -> dict[str, Any]:
        client_id = _next_ai_client_id()
        generator_variant = self._get_tool_ui_settings()["image_generator_variant"]
        template_file = self.get_tool_template_filename("gen_image")
        task, error_result = self._build_generation_task("gen_image", generator_variant, prompt=prompt, client_id=client_id)
        if error_result is not None:
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            return error_result
        task, error_result = self._apply_generation_overrides("gen_image", task, include_num_frames=False, width=width, height=height, num_inference_steps=num_inference_steps)
        if error_result is not None:
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            return error_result
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
            "width": {
                "type": "integer",
                "description": "Optional output width in pixels. Only pass this when the user explicitly asks for output size; otherwise omit it and use the current Deepy/template setting.",
                "required": False,
            },
            "height": {
                "type": "integer",
                "description": "Optional output height in pixels. Only pass this when the user explicitly asks for output size; otherwise omit it and use the current Deepy/template setting.",
                "required": False,
            },
            "num_frames": {
                "type": "integer",
                "description": "Optional output frame count. If omitted, use the current Deepy/template setting.",
                "required": False,
            },
            "duration_seconds": {
                "type": "number",
                "description": "Optional output duration in seconds. Deepy converts it to num_frames using the effective FPS. Do not pass this together with num_frames.",
                "required": False,
            },
            "fps": {
                "type": "integer",
                "description": "Optional output FPS between 15 and 60. If omitted, keep the template FPS behavior.",
                "required": False,
            },
            "num_inference_steps": {
                "type": "integer",
                "description": "Optional number of inference steps. If omitted, keep the template step count.",
                "required": False,
            },
            "loras": {
                "type": "array",
                "description": "Optional list of LoRA filenames to apply. Each item must include `name` and may include `multiplier` as a number like 0.8 or a WanGP multiplier string like `0;1`. Omitted multipliers default to 1.",
                "required": False,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "LoRA filename returned by Get Loras."},
                        "multiplier": {"description": "Optional LoRA multiplier. Accepts a number or a WanGP multiplier string."},
                    },
                    "required": ["name"],
                },
            },
        },
    )
    def gen_video(
        self,
        prompt: str,
        image_start: str | None = None,
        image_end: str | None = None,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        duration_seconds: float | None = None,
        fps: int | None = None,
        num_inference_steps: int | None = None,
        loras: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
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
        task, error_result = self._build_generation_task(
            "gen_video",
            generator_variant,
            prompt=prompt,
            client_id=client_id,
            image_start=None if start_media is None else str(start_media.get("path", "")).strip(),
            image_end=None if end_media is None else str(end_media.get("path", "")).strip(),
        )
        if error_result is not None:
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            if start_media is not None:
                error_result["source_start_media_id"] = start_media.get("media_id", "")
            if end_media is not None:
                error_result["source_end_media_id"] = end_media.get("media_id", "")
            return error_result
        try:
            task = deepy_tool_settings.apply_tool_loras("gen_video", generator_variant, task, loras)
        except Exception as exc:
            error_result = {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": str(prompt or "").strip(),
                "resolution": str(task.get("resolution", "") or "").strip(),
                "error": str(exc),
            }
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            if start_media is not None:
                error_result["source_start_media_id"] = start_media.get("media_id", "")
            if end_media is not None:
                error_result["source_end_media_id"] = end_media.get("media_id", "")
            return error_result
        task, error_result = self._apply_generation_overrides("gen_video", task, include_num_frames=True, width=width, height=height, num_frames=num_frames, duration_seconds=duration_seconds, fps=fps, num_inference_steps=num_inference_steps)
        if error_result is not None:
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            if start_media is not None:
                error_result["source_start_media_id"] = start_media.get("media_id", "")
            if end_media is not None:
                error_result["source_end_media_id"] = end_media.get("media_id", "")
            return error_result
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
        result = self._queue_generation_task(task, activity_label="video generation", output_label="Generated video", gallery_media_type="video")
        result["generator_variant"] = generator_variant
        if len(template_file) > 0:
            result["template_file"] = template_file
        if start_media is not None:
            result["source_start_media_id"] = start_media.get("media_id", "")
        if end_media is not None:
            result["source_end_media_id"] = end_media.get("media_id", "")
        return result

    @assistant_tool(
        display_name="Generate Video With Speech",
        description="Queue and generate a talking video from a text prompt, a start image, and a speech audio clip inside WanGP, then wait until the output video is available.",
        parameters={
            "prompt": {
                "type": "string",
                "description": "The video generation prompt to send to WanGP.",
            },
            "image_start": {
                "type": "string",
                "description": "The media id of the start image returned by Resolve Media.",
            },
            "audio_media_id": {
                "type": "string",
                "description": "The media id of the speech audio returned by Resolve Media.",
            },
            "width": {
                "type": "integer",
                "description": "Optional output width in pixels. Only pass this when the user explicitly asks for output size; otherwise omit it and use the current Deepy/template setting.",
                "required": False,
            },
            "height": {
                "type": "integer",
                "description": "Optional output height in pixels. Only pass this when the user explicitly asks for output size; otherwise omit it and use the current Deepy/template setting.",
                "required": False,
            },
            "num_frames": {
                "type": "integer",
                "description": "Optional output frame count. If omitted, use the current Deepy/template setting.",
                "required": False,
            },
            "duration_seconds": {
                "type": "number",
                "description": "Optional output duration in seconds. Deepy converts it to num_frames using the effective FPS. Do not pass this together with num_frames.",
                "required": False,
            },
            "fps": {
                "type": "integer",
                "description": "Optional output FPS between 15 and 60. If omitted, keep the template FPS behavior.",
                "required": False,
            },
            "num_inference_steps": {
                "type": "integer",
                "description": "Optional number of inference steps. If omitted, keep the template step count.",
                "required": False,
            },
            "loras": {
                "type": "array",
                "description": "Optional list of LoRA filenames to apply. Each item must include `name` and may include `multiplier` as a number like 0.8 or a WanGP multiplier string like `0;1`. Omitted multipliers default to 1.",
                "required": False,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "LoRA filename returned by Get Loras."},
                        "multiplier": {"description": "Optional LoRA multiplier. Accepts a number or a WanGP multiplier string."},
                    },
                    "required": ["name"],
                },
            },
        },
    )
    def gen_video_with_speech(
        self,
        prompt: str,
        image_start: str,
        audio_media_id: str,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        duration_seconds: float | None = None,
        fps: int | None = None,
        num_inference_steps: int | None = None,
        loras: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._sync_recent_media()
        start_media, error_result = self._resolve_image_media(image_start, "image_start")
        if error_result is not None:
            error_result.update({"prompt": str(prompt or "").strip(), "output_file": ""})
            return error_result
        audio_media, error_result = self._resolve_audio_media(audio_media_id, "audio_media_id")
        if error_result is not None:
            error_result.update({"prompt": str(prompt or "").strip(), "output_file": ""})
            return error_result
        client_id = _next_ai_client_id()
        generator_variant = self.get_tool_variant("gen_video_with_speech")
        template_file = self.get_tool_template_filename("gen_video_with_speech")
        task, error_result = self._build_generation_task(
            "gen_video_with_speech",
            generator_variant,
            prompt=prompt,
            client_id=client_id,
            audio_guide=str(audio_media.get("path", "")).strip(),
            image_start_target=self._get_image_start_target("gen_video_with_speech"),
            image_start=str(start_media.get("path", "")).strip(),
        )
        if error_result is not None:
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            error_result["source_start_media_id"] = start_media.get("media_id", "")
            error_result["source_audio_media_id"] = audio_media.get("media_id", "")
            error_result["image_start_target"] = self._get_image_start_target("gen_video_with_speech")
            return error_result
        try:
            task = deepy_tool_settings.apply_tool_loras("gen_video_with_speech", generator_variant, task, loras)
        except Exception as exc:
            error_result = {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": str(prompt or "").strip(),
                "resolution": str(task.get("resolution", "") or "").strip(),
                "error": str(exc),
            }
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            error_result["source_start_media_id"] = start_media.get("media_id", "")
            error_result["source_audio_media_id"] = audio_media.get("media_id", "")
            error_result["image_start_target"] = self._get_image_start_target("gen_video_with_speech")
            return error_result
        task, error_result = self._apply_generation_overrides("gen_video_with_speech", task, include_num_frames=True, width=width, height=height, num_frames=num_frames, duration_seconds=duration_seconds, fps=fps, num_inference_steps=num_inference_steps)
        if error_result is not None:
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            error_result["source_start_media_id"] = start_media.get("media_id", "")
            error_result["source_audio_media_id"] = audio_media.get("media_id", "")
            error_result["image_start_target"] = self._get_image_start_target("gen_video_with_speech")
            return error_result
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
        if len(str(task.get("audio_guide", "") or "").strip()) == 0:
            return {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": str(prompt or "").strip(),
                "resolution": task.get("resolution", ""),
                "error": "Speech audio path is empty.",
            }
        result = self._queue_generation_task(task, activity_label="video generation", output_label="Generated video", gallery_media_type="video")
        result["generator_variant"] = generator_variant
        if len(template_file) > 0:
            result["template_file"] = template_file
        result["source_start_media_id"] = start_media.get("media_id", "")
        result["source_audio_media_id"] = audio_media.get("media_id", "")
        result["image_start_target"] = self._get_image_start_target("gen_video_with_speech")
        return result

    @assistant_tool(
        display_name="Generate Speech From Description",
        description="Queue and generate a speech audio clip from text, using a voice description stored in alt_prompt, then wait until the output audio is available.",
        parameters={
            "prompt": {
                "type": "string",
                "description": "The speech content to synthesize.",
            },
            "voice_description": {
                "type": "string",
                "description": "A short description of the desired voice, tone, or speaking style.",
            },
        },
    )
    def gen_speech_from_description(self, prompt: str, voice_description: str) -> dict[str, Any]:
        client_id = _next_ai_client_id()
        generator_variant = self.get_tool_variant("gen_speech_from_description")
        template_file = self.get_tool_template_filename("gen_speech_from_description")
        task, error_result = self._build_generation_task("gen_speech_from_description", generator_variant, prompt=prompt, client_id=client_id, alt_prompt=voice_description)
        if error_result is not None:
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            return error_result
        if len(task["prompt"]) == 0:
            self._set_status("Speech generation failed: prompt is empty.", kind="error")
            return {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": "",
                "error": "Prompt is empty.",
            }
        if len(str(task.get("alt_prompt", "") or "").strip()) == 0:
            self._set_status("Speech generation failed: voice description is empty.", kind="error")
            return {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": str(prompt or "").strip(),
                "error": "voice_description is required.",
            }
        result = self._queue_generation_task(task, activity_label="speech generation", output_label="Generated speech", gallery_media_type="audio")
        result["generator_variant"] = generator_variant
        if len(template_file) > 0:
            result["template_file"] = template_file
        result["voice_description"] = str(task.get("alt_prompt", "") or "").strip()
        return result

    @assistant_tool(
        display_name="Generate Speech From Sample",
        description="Queue and generate a speech audio clip from text, cloning the voice from a previously resolved audio sample, then wait until the output audio is available.",
        parameters={
            "prompt": {
                "type": "string",
                "description": "The speech content to synthesize.",
            },
            "media_id": {
                "type": "string",
                "description": "The media id of the audio sample returned by Resolve Media.",
            },
        },
    )
    def gen_speech_from_sample(self, prompt: str, media_id: str) -> dict[str, Any]:
        self._sync_recent_media()
        sample_media, error_result = self._resolve_audio_media(media_id, "media_id")
        if error_result is not None:
            error_result.update({"prompt": str(prompt or "").strip(), "output_file": ""})
            return error_result
        client_id = _next_ai_client_id()
        generator_variant = self.get_tool_variant("gen_speech_from_sample")
        template_file = self.get_tool_template_filename("gen_speech_from_sample")
        task, error_result = self._build_generation_task(
            "gen_speech_from_sample",
            generator_variant,
            prompt=prompt,
            client_id=client_id,
            audio_guide=str(sample_media.get("path", "")).strip(),
        )
        if error_result is not None:
            error_result["generator_variant"] = generator_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            error_result["source_media_id"] = sample_media.get("media_id", "")
            return error_result
        if len(task["prompt"]) == 0:
            self._set_status("Speech generation failed: prompt is empty.", kind="error")
            return {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": "",
                "error": "Prompt is empty.",
            }
        if len(str(task.get("audio_guide", "") or "").strip()) == 0:
            return {
                "status": "error",
                "client_id": client_id,
                "media_id": sample_media.get("media_id", ""),
                "output_file": "",
                "prompt": str(prompt or "").strip(),
                "error": "Audio sample path is empty.",
            }
        result = self._queue_generation_task(task, activity_label="speech generation", output_label="Generated speech", gallery_media_type="audio")
        result["generator_variant"] = generator_variant
        if len(template_file) > 0:
            result["template_file"] = template_file
        result["source_media_id"] = sample_media.get("media_id", "")
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
            "width": {
                "type": "integer",
                "description": "Optional output width in pixels. Only pass this when the user explicitly asks for output size; otherwise omit it and use the current Deepy/template setting.",
                "required": False,
            },
            "height": {
                "type": "integer",
                "description": "Optional output height in pixels. Only pass this when the user explicitly asks for output size; otherwise omit it and use the current Deepy/template setting.",
                "required": False,
            },
            "num_inference_steps": {
                "type": "integer",
                "description": "Optional number of inference steps. If omitted, keep the template step count.",
                "required": False,
            },
        },
    )
    def edit_image(
        self,
        media_id: str,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int | None = None,
    ) -> dict[str, Any]:
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
        task, error_result = self._build_generation_task(
            "edit_image",
            editor_variant,
            prompt=prompt,
            client_id=client_id,
            image_refs=[str(media_record.get("path", "")).strip()],
        )
        if error_result is not None:
            error_result["editor_variant"] = editor_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            error_result["media_id"] = media_record.get("media_id", "")
            return error_result
        task, error_result = self._apply_generation_overrides("edit_image", task, include_num_frames=False, width=width, height=height, num_inference_steps=num_inference_steps)
        if error_result is not None:
            error_result["editor_variant"] = editor_variant
            if len(template_file) > 0:
                error_result["template_file"] = template_file
            error_result["media_id"] = media_record.get("media_id", "")
            return error_result
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
        display_name="Create Color Frame",
        description="Create a solid-color image with the requested width and height, rounded to the nearest multiple of 16, and add it to WanGP galleries. Use this for blank frames, color cards, or transition plates.",
        parameters={
            "width": {
                "type": "integer",
                "description": "Output image width in pixels.",
            },
            "height": {
                "type": "integer",
                "description": "Output image height in pixels.",
            },
            "color": {
                "type": "string",
                "description": "Optional fill color. Accepts common names like black, white, red, or hex values like #000000.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def create_color_frame(self, width: int, height: int, color: str = "black") -> dict[str, Any]:
        try:
            width = int(width)
            height = int(height)
        except Exception:
            return {"status": "error", "width": width, "height": height, "color": str(color or "").strip() or "black", "output_file": "", "error": "width and height must be integers."}
        if width <= 0 or height <= 0:
            return {"status": "error", "width": width, "height": height, "color": str(color or "").strip() or "black", "output_file": "", "error": "width and height must be >= 1."}
        width = max(16, int(round(width / 16.0) * 16))
        height = max(16, int(round(height / 16.0) * 16))
        resolved_color = str(color or "black").strip() or "black"
        try:
            rgb_color = ImageColor.getrgb(resolved_color)
        except Exception:
            return {"status": "error", "width": width, "height": height, "color": resolved_color, "output_file": "", "error": "color must be a valid color name or hex value."}
        if len(rgb_color) == 4:
            rgb_color = tuple(rgb_color[:3])
        safe_color_name = re.sub(r"[^a-z0-9]+", "_", resolved_color.lower()).strip("_") or "color"
        output_name = f"color_{safe_color_name}_{width}x{height}.png"
        self._set_status("Creating color frame...", kind="tool")
        self._update_tool_progress("running", "Creating", {"status": "running", "width": width, "height": height, "color": resolved_color})
        output_path = self._resolve_direct_output_path(output_name, True, False)
        try:
            image = Image.new("RGB", (width, height), rgb_color)
            image.save(output_path)
        except Exception as exc:
            result = {"status": "error", "width": width, "height": height, "color": resolved_color, "output_file": "", "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Color frame creation failed: {exc}", kind="error")
            return result
        settings = self._build_direct_image_settings(f'Created solid {resolved_color} image at {width}x{height}', width, height, prompt=f"A solid {resolved_color} image at {width}x{height}.")
        media_record = self._record_direct_media(output_path, settings, is_image=True, audio_only=False, label="Color frame")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "color": resolved_color,
            "output_file": output_path,
            "error": "",
        }
        self._update_tool_progress("done", "Done", result)
        self._set_status("Color frame created.", kind="tool")
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
        prompt_summary = f"An image extracted from a video at {time_seconds:.3f} seconds." if time_seconds is not None else f"An image extracted from frame {resolved_frame_no} of a video."
        extracted_settings = self._build_direct_media_settings(source_media, comments, fallback_prompt=prompt_summary)
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
        description="Extract a video segment from a previously resolved video using either time-based arguments (start_time, end_time, duration) or frame-based arguments (start_frame, end_frame, num_frames).",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id for the source video returned by Resolve Media.",
            },
            "start_time": {
                "type": "number",
                "description": "Optional start time in seconds. Defaults to the beginning when only end_time or duration is provided.",
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
            "start_frame": {
                "type": "integer",
                "description": "Optional start frame number. Defaults to frame 0 when only end_frame or num_frames is provided.",
                "required": False,
            },
            "end_frame": {
                "type": "integer",
                "description": "Optional inclusive end frame number.",
                "required": False,
            },
            "num_frames": {
                "type": "integer",
                "description": "Optional number of frames to keep from start_frame.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def extract_video(
        self,
        media_id: str,
        start_time: float | None = None,
        end_time: float | None = None,
        duration: float | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        num_frames: int | None = None,
    ) -> dict[str, Any]:
        self._sync_recent_media()
        source_media, error_result = self._resolve_video_media(media_id, "media_id")
        if error_result is not None:
            return error_result
        segment_args, error_result = self._resolve_segment_args(
            source_media,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            start_frame=start_frame,
            end_frame=end_frame,
            num_frames=num_frames,
        )
        if error_result is not None:
            error_result.update({"media_id": source_media.get("media_id", ""), "output_file": ""})
            return error_result
        self._set_status("Extracting video...", kind="tool")
        progress_payload = {
            "status": "running",
            "media_id": source_media.get("media_id", ""),
            "mode": segment_args["mode"],
            "start_time": segment_args["start_time"],
            "end_time": segment_args["end_time"],
            "duration": segment_args["duration"],
        }
        if segment_args["mode"] == "frame":
            progress_payload.update({"start_frame": segment_args["start_frame"], "end_frame": segment_args["end_frame"], "num_frames": segment_args["num_frames"]})
        self._update_tool_progress("running", "Extracting", progress_payload)
        source_path = str(source_media.get("path", "")).strip()
        video_codec, video_container = self._get_video_output_settings()
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        output_path = self._resolve_direct_output_path(f"{base_name}_clip{deepy_video_tools.get_video_container_extension(video_container)}", False, False)
        try:
            output_path = deepy_video_tools.extract_video(
                source_path,
                output_path,
                start_time=segment_args["start_time"],
                end_time=segment_args["end_time"],
                duration=segment_args["duration"],
                video_codec=video_codec,
                video_container=video_container,
                audio_codec=self._get_video_audio_output_codec(),
            )
        except Exception as exc:
            result = {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Video extraction failed: {exc}", kind="error")
            return result
        if segment_args["mode"] == "frame":
            comments = f'Extracted video segment from "{os.path.basename(source_path)}" starting at frame {segment_args["start_frame"]} ({segment_args["start_time"]:.3f}s)'
            if start_frame is None and (end_frame is not None or num_frames is not None):
                comments = f'Extracted video segment from "{os.path.basename(source_path)}" starting at the beginning'
            if num_frames is not None:
                comments += f" with {segment_args['num_frames']} frame"
                if segment_args["num_frames"] != 1:
                    comments += "s"
            elif end_frame is not None:
                comments += f" ending at frame {segment_args['end_frame']} ({segment_args['end_time']:.3f}s)"
            else:
                comments += " through the end of the video"
        else:
            comments = f'Extracted video segment from "{os.path.basename(source_path)}" starting at {segment_args["start_time"]:.3f}s'
            if start_time is None and (end_time is not None or duration is not None):
                comments = f'Extracted video segment from "{os.path.basename(source_path)}" starting at the beginning'
            if segment_args["end_time"] is not None:
                comments += f" ending at {segment_args['end_time']:.3f}s"
            elif segment_args["duration"] is not None:
                comments += f" with duration {segment_args['duration']:.3f}s"
        prompt_summary = f"Audio extracted from a source media item."
        if audio_track_no is not None:
            prompt_summary = f"Audio extracted from track {audio_track_no} of a source media item."
        if segment_args["mode"] == "frame" and (start_frame is not None or end_frame is not None or num_frames is not None):
            prompt_summary += f" Keep audio aligned to frames starting at {segment_args['start_frame']}."
        elif segment_args["start_time"] is not None or segment_args["end_time"] is not None or segment_args["duration"] is not None:
            if segment_args["end_time"] is not None:
                prompt_summary += f" From {segment_args['start_time']:.3f} to {segment_args['end_time']:.3f} seconds."
            elif segment_args["duration"] is not None:
                prompt_summary += f" Starting at {segment_args['start_time']:.3f} seconds for {segment_args['duration']:.3f} seconds."
        extracted_settings = self._build_direct_media_settings(source_media, comments, fallback_prompt=prompt_summary)
        self._update_video_metadata_fields(output_path, extracted_settings)
        media_record = self._record_direct_media(output_path, extracted_settings, is_image=False, audio_only=False, label="Extracted video")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "source_media_id": source_media.get("media_id", ""),
            "mode": segment_args["mode"],
            "start_time": segment_args["start_time"],
            "end_time": segment_args["end_time"],
            "duration": segment_args["duration"],
            "output_file": output_path,
            "error": "",
        }
        if segment_args["mode"] == "frame":
            result.update({"start_frame": segment_args["start_frame"], "end_frame": segment_args["end_frame"], "num_frames": segment_args["num_frames"]})
        self._update_tool_progress("done", "Done", result)
        self._set_status("Video extracted.", kind="tool")
        return result

    @assistant_tool(
        display_name="Extract Audio",
        description="Extract audio from a previously resolved video or audio file using either time-based arguments (start_time, end_time, duration) or, for video sources, frame-based arguments (start_frame, end_frame, num_frames).",
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
            "start_frame": {
                "type": "integer",
                "description": "Optional start frame number when media_id refers to a video. Defaults to frame 0 when only end_frame or num_frames is provided.",
                "required": False,
            },
            "end_frame": {
                "type": "integer",
                "description": "Optional inclusive end frame number when media_id refers to a video.",
                "required": False,
            },
            "num_frames": {
                "type": "integer",
                "description": "Optional number of source video frames to keep when media_id refers to a video.",
                "required": False,
            },
            "audio_track_no": {
                "type": "integer",
                "description": "Optional 1-based audio track number to extract. Defaults to 1.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def extract_audio(
        self,
        media_id: str,
        start_time: float | None = None,
        end_time: float | None = None,
        duration: float | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        num_frames: int | None = None,
        audio_track_no: int | None = None,
    ) -> dict[str, Any]:
        self._sync_recent_media()
        if self.session is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "output_file": "", "error": "Assistant session is not available."}
        source_media = media_registry.get_media_record(self.session, media_id)
        if source_media is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "output_file": "", "error": "Unknown media id."}
        if source_media.get("media_type") not in {"audio", "video"}:
            actual_media_type = str(source_media.get("media_type", "") or "").strip() or "unknown media type"
            return {"status": "error", "media_id": source_media.get("media_id", ""), "actual_media_type": actual_media_type, "media_type": actual_media_type, "output_file": "", "error": f"media_id must reference audio or video, not a {actual_media_type}."}
        segment_args, error_result = self._resolve_segment_args(
            source_media,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            start_frame=start_frame,
            end_frame=end_frame,
            num_frames=num_frames,
            allow_empty=True,
        )
        if error_result is not None:
            error_result.update({"media_id": source_media.get("media_id", ""), "output_file": ""})
            return error_result
        try:
            audio_track_no = None if audio_track_no is None or str(audio_track_no).strip() == "" else int(audio_track_no)
        except Exception:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "audio_track_no": audio_track_no, "output_file": "", "error": "audio_track_no must be an integer."}
        if audio_track_no is not None and audio_track_no <= 0:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "audio_track_no": audio_track_no, "output_file": "", "error": "audio_track_no must be >= 1."}
        self._set_status("Extracting audio...", kind="tool")
        progress_payload = {
            "status": "running",
            "media_id": source_media.get("media_id", ""),
            "mode": segment_args["mode"],
            "start_time": segment_args["start_time"],
            "end_time": segment_args["end_time"],
            "duration": segment_args["duration"],
            "audio_track_no": audio_track_no,
        }
        if segment_args["mode"] == "frame":
            progress_payload.update({"start_frame": segment_args["start_frame"], "end_frame": segment_args["end_frame"], "num_frames": segment_args["num_frames"]})
        self._update_tool_progress("running", "Extracting", progress_payload)
        source_path = str(source_media.get("path", "")).strip()
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        audio_codec = self._get_standalone_audio_output_codec()
        output_path = self._resolve_direct_output_path(f"{base_name}_audio{deepy_video_tools.get_audio_standalone_extension(audio_codec)}", False, True)
        try:
            output_path = deepy_video_tools.extract_audio(
                source_path,
                output_path,
                start_time=segment_args["start_time"],
                end_time=segment_args["end_time"],
                duration=segment_args["duration"],
                audio_track_no=audio_track_no,
                audio_codec=audio_codec,
            )
        except Exception as exc:
            result = {"status": "error", "media_id": source_media.get("media_id", ""), "audio_track_no": audio_track_no, "output_file": "", "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Audio extraction failed: {exc}", kind="error")
            return result
        comments = f'Extracted audio from "{os.path.basename(source_path)}"'
        if audio_track_no is not None:
            comments += f" using audio track {audio_track_no}"
        if segment_args["mode"] == "frame":
            if start_frame is not None or end_frame is not None or num_frames is not None:
                comments += f" starting at frame {segment_args['start_frame']} ({segment_args['start_time']:.3f}s)"
                if start_frame is None and (end_frame is not None or num_frames is not None):
                    comments = comments.replace(f" starting at frame {segment_args['start_frame']} ({segment_args['start_time']:.3f}s)", " starting at the beginning")
                if num_frames is not None:
                    comments += f" with {segment_args['num_frames']} frame"
                    if segment_args["num_frames"] != 1:
                        comments += "s"
                elif end_frame is not None:
                    comments += f" ending at frame {segment_args['end_frame']} ({segment_args['end_time']:.3f}s)"
                else:
                    comments += " through the end of the source"
        else:
            if segment_args["start_time"] is not None:
                comments += f" starting at {segment_args['start_time']:.3f}s"
                if start_time is None and (end_time is not None or duration is not None):
                    comments = comments.replace(f" starting at {segment_args['start_time']:.3f}s", " starting at the beginning")
            if segment_args["end_time"] is not None:
                comments += f" ending at {segment_args['end_time']:.3f}s"
            elif segment_args["duration"] is not None:
                comments += f" with duration {segment_args['duration']:.3f}s"
        extracted_settings = self._build_direct_media_settings(source_media, comments)
        self._update_audio_metadata_fields(output_path, extracted_settings)
        media_record = self._record_direct_media(output_path, extracted_settings, is_image=False, audio_only=True, label="Extracted audio")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "source_media_id": source_media.get("media_id", ""),
            "mode": segment_args["mode"],
            "start_time": segment_args["start_time"],
            "end_time": segment_args["end_time"],
            "duration": segment_args["duration"],
            "audio_track_no": 1 if audio_track_no is None else audio_track_no,
            "output_file": output_path,
            "error": "",
        }
        if segment_args["mode"] == "frame":
            result.update({"start_frame": segment_args["start_frame"], "end_frame": segment_args["end_frame"], "num_frames": segment_args["num_frames"]})
        self._update_tool_progress("done", "Done", result)
        self._set_status("Audio extracted.", kind="tool")
        return result

    @assistant_tool(
        display_name="Transcribe Media",
        description="Transcribe the spoken content of a previously resolved audio or video media item with Whisper medium, returning segment timestamps by default and optionally using word timestamps instead.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id for the source audio or video returned by Resolve Media.",
            },
            "timestamp_type": {
                "type": "string",
                "description": "Optional timestamp detail to include. Use `segment` for segment timestamps, `word` for word timestamps, or `none` to disable timestamps. If omitted, segment timestamps are returned.",
                "required": False,
            },
            "audio_track_no": {
                "type": "integer",
                "description": "Optional 1-based audio track number when the source media contains multiple audio tracks.",
                "required": False,
            },
        },
    )
    def transcribe_media(self, media_id: str, timestamp_type: str | None = None, audio_track_no: int | None = None) -> dict[str, Any]:
        self._sync_recent_media()
        source_media, error_result = self._resolve_audio_or_video_media(media_id, "media_id")
        if error_result is not None:
            return error_result
        try:
            normalized_timestamp_type = deepy_transcription.normalize_timestamp_type(timestamp_type)
        except Exception as exc:
            return {
                "status": "error",
                "media_id": str(media_id or "").strip(),
                "timestamp_type": str(timestamp_type or "").strip(),
                "error": str(exc),
            }
        try:
            audio_track_no = None if audio_track_no is None or str(audio_track_no).strip() == "" else int(audio_track_no)
        except Exception:
            return {
                "status": "error",
                "media_id": source_media.get("media_id", ""),
                "timestamp_type": "" if normalized_timestamp_type is None else normalized_timestamp_type,
                "audio_track_no": audio_track_no,
                "error": "audio_track_no must be an integer.",
            }
        if audio_track_no is not None and audio_track_no <= 0:
            return {
                "status": "error",
                "media_id": source_media.get("media_id", ""),
                "timestamp_type": "" if normalized_timestamp_type is None else normalized_timestamp_type,
                "audio_track_no": audio_track_no,
                "error": "audio_track_no must be >= 1.",
            }
        self._set_status("Transcribing media...", kind="tool")
        progress_payload = {
            "status": "running",
            "media_id": source_media.get("media_id", ""),
            "media_type": source_media.get("media_type", ""),
            "timestamp_type": "" if normalized_timestamp_type is None else normalized_timestamp_type,
        }
        if audio_track_no is not None:
            progress_payload["audio_track_no"] = audio_track_no
        self._update_tool_progress("running", "Transcribing", progress_payload)
        source_path = str(source_media.get("path", "")).strip()
        try:
            payload = deepy_transcription.transcribe_media(source_path, timestamp_type=normalized_timestamp_type, audio_track_no=audio_track_no)
        except Exception as exc:
            result = {
                "status": "error",
                "media_id": source_media.get("media_id", ""),
                "label": source_media.get("label", ""),
                "media_type": source_media.get("media_type", ""),
                "path": source_path,
                "filename": os.path.basename(source_path),
                "timestamp_type": "" if normalized_timestamp_type is None else normalized_timestamp_type,
                "error": str(exc),
            }
            if audio_track_no is not None:
                result["audio_track_no"] = audio_track_no
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Transcription failed: {exc}", kind="error")
            return result
        result = {
            "status": "done",
            "media_id": source_media.get("media_id", ""),
            "label": source_media.get("label", ""),
            "media_type": source_media.get("media_type", ""),
            "path": source_path,
            "filename": os.path.basename(source_path),
            "error": "",
            **payload,
        }
        if audio_track_no is not None:
            result["audio_track_no"] = audio_track_no
        self._update_tool_progress("done", "Done", result)
        self._set_status("Transcription finished.", kind="tool")
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
        display_name="Resize Crop",
        description="Resize and crop a previously resolved image or video in one step. Crop values can be expressed in pixels or percent. When both width and height are provided, aspect ratio is preserved by default by cropping extra area instead of stretching.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id for the source image or video returned by Resolve Media.",
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
            "crop_anchor": {
                "type": "string",
                "description": "Optional. Defaults to center. Controls which area stays in frame when aspect-ratio-preserving auto-crop trims extra area.",
                "enum": ["center", "left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"],
                "required": False,
            },
            "stretch_to_fit": {
                "type": "boolean",
                "description": "Optional. Defaults to false. When width and height are both provided, set this to true only if the user explicitly wants stretching or distortion instead of cropping extra area.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def resize_crop(self, media_id: str, width: int | None = None, height: int | None = None, crop_left: float | None = None, crop_top: float | None = None, crop_right: float | None = None, crop_bottom: float | None = None, crop_unit: str | None = None, crop_anchor: str | None = None, stretch_to_fit: bool | None = None) -> dict[str, Any]:
        self._sync_recent_media()
        if self.session is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "output_file": "", "error": "Assistant session is not available."}
        source_media = media_registry.get_media_record(self.session, media_id)
        if source_media is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "output_file": "", "error": "Unknown media id."}
        if source_media.get("media_type") not in {"image", "video"}:
            actual_media_type = str(source_media.get("media_type", "") or "").strip() or "unknown media type"
            return {"status": "error", "media_id": source_media.get("media_id", ""), "actual_media_type": actual_media_type, "media_type": actual_media_type, "output_file": "", "error": f"media_id must reference an image or video, not a {actual_media_type}."}
        try:
            width = None if width is None or str(width).strip() == "" else int(width)
            height = None if height is None or str(height).strip() == "" else int(height)
            crop_left = 0 if crop_left is None or str(crop_left).strip() == "" else float(crop_left)
            crop_top = 0 if crop_top is None or str(crop_top).strip() == "" else float(crop_top)
            crop_right = 0 if crop_right is None or str(crop_right).strip() == "" else float(crop_right)
            crop_bottom = 0 if crop_bottom is None or str(crop_bottom).strip() == "" else float(crop_bottom)
        except Exception:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": "width and height must be integers, crop values must be numbers."}
        stretch_to_fit, error_result = self._parse_bool_value(stretch_to_fit, "stretch_to_fit")
        if error_result is not None:
            error_result["media_id"] = source_media.get("media_id", "")
            error_result["output_file"] = ""
            return error_result
        if stretch_to_fit is None:
            stretch_to_fit = False
        preserve_aspect_ratio = not bool(stretch_to_fit)
        crop_unit = str(crop_unit or "pixels").strip().lower() or "pixels"
        if crop_unit not in {"pixels", "percent"}:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": "crop_unit must be 'pixels' or 'percent'."}
        crop_anchor = str(crop_anchor or "center").strip().lower().replace("-", "_").replace(" ", "_") or "center"
        if crop_anchor not in {"center", "left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"}:
            return {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": "crop_anchor must be center, left, right, top, bottom, top_left, top_right, bottom_left, or bottom_right."}
        source_media_type = str(source_media.get("media_type", "") or "").strip() or "media"
        self._set_status(f"Resizing and cropping {source_media_type}...", kind="tool")
        self._update_tool_progress("running", "Processing", {"status": "running", "media_id": source_media.get("media_id", ""), "width": width, "height": height, "crop_left": crop_left, "crop_top": crop_top, "crop_right": crop_right, "crop_bottom": crop_bottom, "crop_unit": crop_unit, "crop_anchor": crop_anchor, "stretch_to_fit": stretch_to_fit, "preserve_aspect_ratio": preserve_aspect_ratio})
        source_path = str(source_media.get("path", "")).strip()
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        try:
            if source_media_type == "video":
                video_codec, video_container = self._get_video_output_settings()
                output_path = self._resolve_direct_output_path(f"{base_name}_resized{deepy_video_tools.get_video_container_extension(video_container)}", False, False)
                output_path = deepy_video_tools.resize_crop_video(source_path, output_path, width=width, height=height, crop_left=crop_left, crop_top=crop_top, crop_right=crop_right, crop_bottom=crop_bottom, crop_unit=crop_unit, preserve_aspect_ratio=preserve_aspect_ratio, crop_anchor=crop_anchor, video_codec=video_codec, video_container=video_container, audio_codec=self._get_video_audio_output_codec())
            else:
                image_ext = os.path.splitext(source_path)[1].lower()
                if image_ext not in {".png", ".jpg", ".jpeg", ".webp"}:
                    image_ext = ".png"
                output_path = self._resolve_direct_output_path(f"{base_name}_resized{image_ext}", True, False)
                output_path = deepy_video_tools.resize_crop_image(source_path, output_path, width=width, height=height, crop_left=crop_left, crop_top=crop_top, crop_right=crop_right, crop_bottom=crop_bottom, crop_unit=crop_unit, preserve_aspect_ratio=preserve_aspect_ratio, crop_anchor=crop_anchor)
        except Exception as exc:
            result = {"status": "error", "media_id": source_media.get("media_id", ""), "output_file": "", "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            self._set_status(f"Resize/crop failed: {exc}", kind="error")
            return result
        has_manual_crop = any(value > 0 for value in (crop_left, crop_top, crop_right, crop_bottom))
        uses_aspect_crop = width is not None and height is not None and not stretch_to_fit
        action_text = "cropped" if has_manual_crop or uses_aspect_crop else "resized" if width is not None or height is not None else "processed"
        action_label = action_text.capitalize()
        comments = f'{action_label} "{os.path.basename(source_path)}"'
        if width is not None or height is not None:
            comments += f" to {width if width is not None else 'auto'}x{height if height is not None else 'auto'}"
        if width is not None and height is not None:
            comments += " with stretching" if stretch_to_fit else " with preserved aspect ratio"
            if action_text == "cropped" and not stretch_to_fit and crop_anchor != "center":
                comments += f" anchored {crop_anchor}"
        if has_manual_crop:
            comments += f" with crop {crop_left}/{crop_top}/{crop_right}/{crop_bottom} {crop_unit}"
        prompt_summary = None
        if source_media_type == "image":
            if action_text == "cropped":
                prompt_summary = f"An image cropped to {width if width is not None else 'auto'}x{height if height is not None else 'auto'}."
                if width is not None and height is not None and crop_anchor != "center":
                    prompt_summary = f"An image cropped to {width}x{height}, keeping the {crop_anchor.replace('_', ' ')} area."
            elif action_text == "resized":
                prompt_summary = f"An image resized to {width if width is not None else 'auto'}x{height if height is not None else 'auto'}."
        resized_settings = self._build_direct_media_settings(source_media, comments, fallback_prompt=prompt_summary)
        if source_media_type == "video":
            self._update_video_metadata_fields(output_path, resized_settings)
        media_record = self._record_direct_media(output_path, resized_settings, is_image=source_media_type == "image", audio_only=False, label=f"{action_label} {source_media_type}")
        result = {
            "status": "done",
            "media_id": "" if media_record is None else media_record.get("media_id", ""),
            "source_media_id": source_media.get("media_id", ""),
            "output_file": output_path,
            "error": "",
        }
        self._update_tool_progress("done", "Done", result)
        self._set_status(f"{source_media_type.capitalize()} resize/crop finished.", kind="tool")
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
        self._remember_generated_client_id(merged_settings["client_id"])
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
        display_name="Search Doc",
        description="Search WanGP documentation by keywords and return the best matching sections.",
        parameters={
            "query": {
                "type": "string",
                "description": "Keywords or a short natural-language question to search for in WanGP docs.",
            },
            "doc_id": {
                "type": "string",
                "description": "Optional documentation id to limit the search to: finetunes, getting_started, loras, overview, prompts, or vace.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def search_doc(self, query: str, doc_id: str = "") -> dict[str, Any]:
        query = str(query or "").strip()
        lookup_id = str(doc_id or "").strip().lower()
        if len(query) == 0:
            return {"status": "error", "query": "", "doc_id": lookup_id, "matches": [], "error": "query is empty."}
        if len(lookup_id) > 0 and lookup_id not in _DEEPY_DOCS:
            return {
                "status": "error",
                "query": query,
                "doc_id": lookup_id,
                "matches": [],
                "available_doc_ids": sorted(_DEEPY_DOCS.keys()),
                "error": "Unknown documentation id.",
            }
        target_doc_ids = [lookup_id] if len(lookup_id) > 0 else sorted(_DEEPY_DOCS.keys())
        query_tokens = _tokenize_doc_query(query)
        self._set_status("Searching documentation...", kind="tool")
        self._update_tool_progress("running", "Searching", {"status": "running", "query": query, "doc_id": lookup_id})
        try:
            matches = []
            for current_doc_id in target_doc_ids:
                doc_info, sections = _extract_doc_sections(current_doc_id)
                for section in sections:
                    score = _score_doc_section(query, query_tokens, doc_info["title"], section)
                    if score <= 0:
                        continue
                    matches.append(
                        {
                            "doc_id": doc_info["doc_id"],
                            "title": doc_info["title"],
                            "path": doc_info["path"],
                            "section": section["section"],
                            "heading": section["heading"],
                            "heading_level": section["heading_level"],
                            "excerpt": _build_doc_excerpt(section, query, query_tokens),
                            "score": int(score),
                        }
                    )
        except Exception as exc:
            result = {"status": "error", "query": query, "doc_id": lookup_id, "matches": [], "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            return result
        matches.sort(key=lambda item: (-int(item["score"]), str(item["doc_id"]), len(str(item["section"]))))
        result = {
            "status": "done",
            "query": query,
            "doc_id": lookup_id,
            "searched_doc_ids": target_doc_ids,
            "matches": matches[:5],
            "error": "",
        }
        self._update_tool_progress("done", "Done", {"status": "done", "query": query, "doc_id": lookup_id, "match_count": len(result["matches"]), "error": ""})
        self._set_status("Documentation search finished.", kind="tool")
        return result

    @assistant_tool(
        display_name="Load Doc Section",
        description="Load one specific WanGP documentation section using the doc id and section path returned by Search Doc.",
        parameters={
            "doc_id": {
                "type": "string",
                "description": "Documentation id: finetunes, getting_started, loras, overview, prompts, or vace.",
            },
            "section": {
                "type": "string",
                "description": "The section path returned by Search Doc, for example `Prompt Enhancer > Automatic Versus On-Demand`.",
            },
        },
        pause_runtime=False,
    )
    def load_doc_section(self, doc_id: str, section: str) -> dict[str, Any]:
        lookup_id = str(doc_id or "").strip().lower()
        section = str(section or "").strip()
        if lookup_id not in _DEEPY_DOCS:
            return {
                "status": "error",
                "doc_id": lookup_id,
                "section": section,
                "available_doc_ids": sorted(_DEEPY_DOCS.keys()),
                "error": "Unknown documentation id.",
            }
        if len(section) == 0:
            return {"status": "error", "doc_id": lookup_id, "section": "", "error": "section is empty."}
        self._set_status("Loading documentation section...", kind="tool")
        self._update_tool_progress("running", "Loading", {"status": "running", "doc_id": lookup_id, "section": section})
        try:
            doc_info, resolved_section, candidate_sections = _resolve_doc_section(lookup_id, section)
        except Exception as exc:
            result = {"status": "error", "doc_id": lookup_id, "section": section, "error": str(exc)}
            self._update_tool_progress("error", "Error", result)
            return result
        if len(resolved_section) == 0:
            result = {
                "status": "error",
                "doc_id": lookup_id,
                "section": section,
                "matching_sections": candidate_sections,
                "error": "Section not found or ambiguous. Use the exact section path returned by Search Doc.",
            }
            self._update_tool_progress("error", "Error", result)
            return result
        result = {
            "status": "done",
            "doc_id": doc_info["doc_id"],
            "title": doc_info["title"],
            "path": doc_info["path"],
            "section": resolved_section["section"],
            "heading": resolved_section["heading"],
            "heading_level": resolved_section["heading_level"],
            "content": resolved_section["content"],
            "error": "",
        }
        self._update_tool_progress("done", "Loaded", {"status": "done", "doc_id": doc_info["doc_id"], "section": resolved_section["section"], "path": doc_info["path"], "error": ""})
        self._set_status("Documentation section loaded.", kind="tool")
        return result

    @assistant_tool(
        display_name="Get Selected Media",
        description="Return the current selected WanGP gallery media. With media_type=all, return both the selected visual media and the selected audio media. If the selected visual item is a video, also report the current player time and frame number.",
        parameters={
            "media_type": {
                "type": "string",
                "description": "Optional desired media type: image, video, audio, or all. all returns both gallery selections.",
                "required": False,
            },
        },
        pause_runtime=False,
    )
    def get_selected_media(self, media_type: str = "all") -> dict[str, Any]:
        self._sync_recent_media()
        resolved_media_type = self._normalize_selected_media_type(media_type)
        if resolved_media_type == "all":
            visual_media_record, audio_media_record, error_result = self._get_all_selected_media_records()
            if error_result is not None:
                return error_result
            return {
                "status": "done",
                "media_type": "all",
                "selected_visual_media": None if visual_media_record is None else self._selected_media_payload(visual_media_record),
                "selected_audio_media": None if audio_media_record is None else self._selected_media_payload(audio_media_record),
                "error": "",
            }
        media_record, error_result = self._get_selected_media_record(media_type)
        if error_result is not None:
            return error_result
        return {"status": "done", **self._selected_media_payload(media_record), "error": ""}

    @assistant_tool(
        display_name="Get Media Details",
        description="Return detailed local metadata for a previously resolved image, video, or audio.",
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
        if media_type not in {"image", "video", "audio"}:
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "media_type": media_type,
                "error": "Detailed media info currently supports images, videos, and audio.",
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
                    "sample_rate": None,
                    "channels": None,
                    "error": "",
                }
            elif media_type == "video":
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
                    "sample_rate": None,
                    "channels": None,
                    "error": "",
                }
            else:
                probe = ffmpeg.probe(media_path)
                audio_streams = [stream for stream in probe.get("streams", []) if str(stream.get("codec_type", "")).strip().lower() == "audio"]
                primary_stream = audio_streams[0] if audio_streams else {}
                sample_rate = primary_stream.get("sample_rate", None)
                channels = primary_stream.get("channels", None)
                duration_seconds = probe.get("format", {}).get("duration", None)
                try:
                    duration_seconds = None if duration_seconds in {None, "", "N/A"} else float(duration_seconds)
                except Exception:
                    duration_seconds = None
                try:
                    sample_rate = None if sample_rate in {None, "", "N/A"} else int(sample_rate)
                except Exception:
                    sample_rate = None
                try:
                    channels = None if channels in {None, "", "N/A"} else int(channels)
                except Exception:
                    channels = None
                result = {
                    "status": "done",
                    "media_id": media_record.get("media_id", ""),
                    "label": media_record.get("label", ""),
                    "media_type": "audio",
                    "path": media_path,
                    "filename": os.path.basename(media_path),
                    "width": None,
                    "height": None,
                    "resolution": None,
                    "frame_count": None,
                    "fps": None,
                    "duration_seconds": duration_seconds,
                    "has_audio": len(audio_streams) > 0,
                    "audio_track_count": int(len(audio_streams)),
                    "sample_rate": sample_rate,
                    "channels": channels,
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
        description="Look up previously generated WanGP media by a short reference such as 'last', 'previous', or 'selected' plus media_type, or by a short description.",
        parameters={
            "reference": {
                "type": "string",
                "description": "The media reference. Use short aliases such as 'last', 'previous', or 'selected' when media_type already specifies image, video, or audio. Descriptive references such as 'robot on the moon' also work.",
            },
            "media_type": {
                "type": "string",
                "description": "The desired media type: image, video, audio, or all. Pair this with short references such as reference='last' or reference='selected'.",
            },
        },
        pause_runtime=False,
    )
    def resolve_media_reference(self, reference: str, media_type: str) -> dict[str, Any]:
        self._sync_recent_media()
        resolved_reference = str(reference or "").strip()
        resolved_media_type_text = str(media_type or "all").strip() or "all"
        if self.session is None:
            return {"status": "error", "reference": resolved_reference, "media_type": resolved_media_type_text, "matches": [], "error": "Assistant session is not available."}
        if self._is_selected_reference(resolved_reference):
            resolved_media_type = self._normalize_selected_media_type(media_type, reference=resolved_reference)
            if resolved_media_type == "all":
                matches = []
                visual_media_record, audio_media_record, error_result = self._get_all_selected_media_records()
                if error_result is not None:
                    error_result.setdefault("reference", resolved_reference)
                    return error_result
                if visual_media_record is not None:
                    matches.append(self._selected_media_payload(visual_media_record, why="matched selected visual media"))
                if audio_media_record is not None:
                    matches.append(self._selected_media_payload(audio_media_record, why="matched selected audio media"))
                if len(matches) == 1:
                    return {"status": "resolved", "media_type": "all", "reference": resolved_reference, "media": matches[0], "error": ""}
                return {"status": "candidates", "media_type": "all", "reference": resolved_reference, "matches": matches, "error": ""}
            media_record, error_result = self._get_selected_media_record(resolved_media_type)
            if error_result is not None:
                error_result.setdefault("reference", resolved_reference)
                return error_result
            return {"status": "resolved", "media_type": resolved_media_type, "reference": resolved_reference, "media": self._selected_media_payload(media_record, why="matched selected media"), "error": ""}
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
                properties[param_name] = _build_tool_parameter_schema(annotations, param_name, param_meta)
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
    def __init__(self, session: AssistantSessionState, runtime_hooks: AssistantRuntimeHooks, tool_box: tools, send_cmd, debug_enabled: bool | None = None, thinking_enabled: bool = True, vram_mode: str = DEEPY_VRAM_MODE_UNLOAD):
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
        self._prefill_started_at: float | None = None
        self._live_prefill_tokens = 0
        self._segment_started_at: float | None = None
        self._segment_generated_tokens = 0
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
        self._emit_stats()

    def _hide_status(self) -> None:
        self._emit_chat_event(assistant_chat.build_status_event(None, visible=False))
        self._emit_stats(force=True)

    def _get_context_window_tokens(self) -> int:
        return normalize_deepy_context_tokens(get_deepy_config_value(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT))

    def _active_sequence_token_count(self) -> int | None:
        if self.runtime is None:
            return None
        try:
            current_seq = self.runtime._get_active_sequence()
        except Exception:
            return None
        if current_seq is None:
            return None
        try:
            return len(current_seq.token_ids or [])
        except Exception:
            return None

    def _resolved_chat_max_tokens(self) -> int:
        max_tokens = 0
        if self.runtime is not None:
            try:
                max_tokens = int(self.runtime.get_max_model_len() or 0)
            except Exception:
                max_tokens = 0
        if max_tokens > 0:
            self.session.runtime_max_model_len = max_tokens
            return max_tokens
        try:
            max_tokens = int(self.session.runtime_max_model_len or 0)
        except Exception:
            max_tokens = 0
        return max_tokens if max_tokens > 0 else self._get_context_window_tokens()

    def _chat_stats_payload(self) -> dict[str, Any]:
        live_prefill_seconds = 0.0 if self._prefill_started_at is None else max(0.0, time.perf_counter() - self._prefill_started_at)
        live_generation_seconds = 0.0 if self._segment_started_at is None else max(0.0, time.perf_counter() - self._segment_started_at)
        return build_assistant_chat_stats(
            self.session,
            max_tokens=self._resolved_chat_max_tokens(),
            active_sequence_token_count=self._active_sequence_token_count(),
            live_prefill_tokens=self._live_prefill_tokens,
            live_prefill_seconds=live_prefill_seconds,
            live_generated_tokens=self._segment_generated_tokens,
            live_generation_seconds=live_generation_seconds,
        )

    def _emit_stats(self, *, force: bool = False) -> None:
        stats = self._chat_stats_payload()
        signature = _json_dumps(stats)
        if not force and signature == str(self.session.chat_stats_signature or ""):
            return
        self.session.chat_stats_signature = signature
        self._emit_chat_event(assistant_chat.build_stats_event(stats))

    def _record_prefill_metrics(self, token_count: int, elapsed_seconds: float) -> None:
        tokens = max(0, int(token_count or 0))
        elapsed = max(0.0, float(elapsed_seconds or 0.0))
        if tokens <= 0 or elapsed <= 0.0:
            return
        self.session.prefill_token_total += tokens
        self.session.prefill_seconds_total += elapsed

    def _record_generation_metrics(self, token_count: int, elapsed_seconds: float) -> None:
        tokens = max(0, int(token_count or 0))
        elapsed = max(0.0, float(elapsed_seconds or 0.0))
        if tokens <= 0 or elapsed <= 0.0:
            return
        self.session.generated_token_total += tokens
        self.session.generated_seconds_total += elapsed

    def _run_prefill_call(self, token_count: int, callback: Callable[[], Any], *, record_if: bool | Callable[[Any], bool] = True) -> Any:
        tokens = max(0, int(token_count or 0))
        started_at = time.perf_counter()
        self._prefill_started_at = started_at if tokens > 0 else None
        self._live_prefill_tokens = tokens
        completed = False
        result = None
        try:
            result = callback()
            completed = True
            return result
        finally:
            elapsed_seconds = max(0.0, time.perf_counter() - started_at)
            self._prefill_started_at = None
            self._live_prefill_tokens = 0
            should_record = record_if(result) if callable(record_if) else bool(record_if)
            if completed and should_record:
                self._record_prefill_metrics(tokens, elapsed_seconds)
            self._emit_stats(force=True)

    def _finish_stream_pass(self, token_count: int | None = None) -> None:
        elapsed_seconds = 0.0 if self._segment_started_at is None else max(0.0, time.perf_counter() - self._segment_started_at)
        recorded_tokens = max(max(0, int(token_count or 0)), max(0, int(self._segment_generated_tokens or 0)))
        self._record_generation_metrics(recorded_tokens, elapsed_seconds)
        self._segment_started_at = None
        self._segment_generated_tokens = 0
        self._emit_stats(force=True)

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
            if str(message.get("role", "")).strip().lower() == "user":
                return model_content
            if _INJECT_SELECTED_MEDIA_RUNTIME_UPDATES:
                return model_content
            return _RUNTIME_UPDATE_BLOCK_RE.sub("\n", model_content).strip()
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

    def _get_runtime_tool_template_label(self, tool_name: str) -> str:
        try:
            variant = str(self.tool_box.get_tool_variant(tool_name) or "").strip()
        except Exception:
            variant = ""
        if len(variant) == 0:
            return ""
        template_label = Path(variant).name.strip()
        return template_label if len(template_label) > 0 else variant

    def _build_video_tool_runtime_instruction(self, tool_name: str, *, changed: bool) -> str:
        template_label = self._get_runtime_tool_template_label(tool_name)
        if len(template_label) == 0:
            return ""
        model_def = self.tool_box._get_effective_tool_model_def(tool_name)
        image_prompt_types_allowed = str(model_def.get("image_prompt_types_allowed", "") or "").strip()
        sentences = [
            f"The {tool_name} tool {'has changed and now uses' if changed else 'uses'} Settings '{template_label}'."
        ]
        if tool_name == "gen_video" and bool(model_def.get("multimedia_generation", False)):
            sentences.append(
                "The gen_video tool can generate a video with an audio output from a text prompt. So if the user provides only a text prompt and wants a talking or voiced video, you must use gen_video directly, keep the spoken words in the prompt, and do not call gen_speech_from_description, gen_speech_from_sample, or gen_video_with_speech first."
            )
        if "T" in image_prompt_types_allowed:
            sentences.append(
                f"The {tool_name} tool can generate a video even if a start image is not provided. So if the user does not provide a start image or asks you explicitly to generate the start image, do not create a start image; just describe the starting situation in the prompt."
            )
        elif "S" in image_prompt_types_allowed:
            sentences.append(
                f"The {tool_name} tool needs a start image. So if the user does not provide a start image, you will need to create a start image first to use this tool."
            )
        return " ".join(sentences).strip()

    def _get_video_tool_runtime_updates(self) -> list[str]:
        if self.session is None:
            return []
        previous_variants = dict(self.session.video_tool_runtime_variants or {})
        current_variants: dict[str, str] = {}
        updates = []
        for tool_name in ("gen_video", "gen_video_with_speech"):
            variant = str(self.tool_box.get_tool_variant(tool_name) or "").strip()
            if len(variant) == 0:
                continue
            current_variants[tool_name] = variant
            previous_variant = str(previous_variants.get(tool_name, "") or "").strip()
            if previous_variant == variant:
                continue
            instruction = self._build_video_tool_runtime_instruction(tool_name, changed=len(previous_variant) > 0)
            if len(instruction) > 0:
                updates.append(instruction)
        self.session.video_tool_runtime_variants = current_variants
        return updates

    def _refresh_runtime_status_note(self) -> None:
        note_blocks = []

        runtime_lines = []

        runtime_lines.extend(self._get_video_tool_runtime_updates())

        new_user_gallery_media = self.tool_box._get_new_user_gallery_media()
        if "image" in new_user_gallery_media:
            instruction = self.tool_box._runtime_media_instruction(
                new_user_gallery_media["image"],
                action="added",
                gallery_label="Image / Video Gallery",
                reference_label="last",
                why="matched last user-added image",
            )
            if len(instruction) > 0:
                runtime_lines.append(instruction)
        if "video" in new_user_gallery_media:
            instruction = self.tool_box._runtime_media_instruction(
                new_user_gallery_media["video"],
                action="added",
                gallery_label="Image / Video Gallery",
                reference_label="last",
                why="matched last user-added video",
            )
            if len(instruction) > 0:
                runtime_lines.append(instruction)
        if "audio" in new_user_gallery_media:
            instruction = self.tool_box._runtime_media_instruction(
                new_user_gallery_media["audio"],
                action="added",
                gallery_label="Audio Gallery",
                reference_label="last",
                why="matched last user-added audio",
            )
            if len(instruction) > 0:
                runtime_lines.append(instruction)

        runtime_lines.extend(self.tool_box._get_selected_gallery_media_updates())

        if len(runtime_lines) > 0:
            note_blocks.append(
                "\n".join(
                    [
                        "<wangp_runtime_update>",
                        "Hidden WanGP runtime state. This is environment metadata, not a user message.",
                        *runtime_lines,
                        "</wangp_runtime_update>",
                    ]
                )
            )
            if self.debug_enabled:
                self._log(f"Prepared runtime update with {len(runtime_lines)} instruction(s).")

        if _INJECT_SELECTED_MEDIA_RUNTIME_UPDATES:
            snapshot = self.tool_box._get_selected_runtime_snapshot()
            previous_snapshot = {}
            previous_signature = str(self.session.runtime_status_signature or "").strip()
            if len(previous_signature) > 0:
                try:
                    previous_snapshot = dict(json.loads(previous_signature) or {})
                except Exception:
                    previous_snapshot = {}
            if snapshot is None:
                if len(previous_signature) == 0:
                    normalized_snapshot = None
                else:
                    normalized_snapshot = {key: None for key in _RUNTIME_STATUS_ALL_KEYS}
            else:
                normalized_snapshot = {key: None for key in _RUNTIME_STATUS_ALL_KEYS}
                for key in ("selected_visual_media_id", "selected_visual_media_type", "selected_visual_media_label", "selected_audio_media_id", "selected_audio_media_type", "selected_audio_media_label"):
                    normalized_snapshot[key] = str(snapshot.get(key, "") or "").strip() or None
                for key in ("selected_visual_current_time_seconds", "selected_visual_current_frame_no"):
                    normalized_snapshot[key] = snapshot.get(key, None)
            if normalized_snapshot is not None:
                signature = _json_dumps(normalized_snapshot)
                if signature != self.session.runtime_status_signature:
                    changed_keys = [key for key in _RUNTIME_STATUS_ALL_KEYS if previous_snapshot.get(key, None) != normalized_snapshot.get(key, None)]
                    if len(previous_snapshot) == 0:
                        emitted_keys = list(_RUNTIME_STATUS_ALL_KEYS)
                    else:
                        emitted_keys = []
                        if any(key in changed_keys for key in _RUNTIME_STATUS_VISUAL_KEYS):
                            emitted_keys.extend(_RUNTIME_STATUS_VISUAL_KEYS)
                        if any(key in changed_keys for key in _RUNTIME_STATUS_AUDIO_KEYS):
                            emitted_keys.extend(_RUNTIME_STATUS_AUDIO_KEYS)
                    if len(emitted_keys) > 0:
                        lines = [
                            "<wangp_runtime_update>",
                            "Hidden WanGP runtime state. This is environment metadata, not a user message.",
                            "Use it as factual UI context only. Omitted keys keep their previous runtime-update values.",
                        ]
                        for key in emitted_keys:
                            value = normalized_snapshot.get(key, None)
                            if isinstance(value, str):
                                rendered_value = value if len(value) > 0 else "none"
                            else:
                                rendered_value = "none" if value is None else value
                            lines.append(f"{key}: {rendered_value}")
                        lines.append("</wangp_runtime_update>")
                        note_blocks.append("\n".join(lines))
                        if self.debug_enabled:
                            self._log(f"Prepared runtime status update: {signature}")
                self.session.runtime_status_signature = signature
        else:
            self.session.runtime_status_signature = ""

        self.session.runtime_status_note = "\n\n".join(note_blocks)

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
        self._emit_stats(force=True)
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
        self._segment_started_at = time.perf_counter()
        self._segment_generated_tokens = 0

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
        self._segment_generated_tokens = max(int(self._segment_generated_tokens or 0), max(0, int(token_count or 0)))
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
        self._emit_stats()

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
        if self.debug_enabled:
            prompt_embeds_shape = None if prompt_embeds is None else tuple(int(x) for x in prompt_embeds.shape)
            prompt_position_shape = None if prompt_position_ids is None else tuple(int(x) for x in prompt_position_ids.shape)
            prompt_embeds_dtype = None if prompt_embeds is None else str(prompt_embeds.dtype).replace("torch.", "")
            prompt_position_dtype = None if prompt_position_ids is None else str(prompt_position_ids.dtype).replace("torch.", "")
            self._log(
                "Inspect visual query "
                f"media_id={media_record.get('media_id', '')} media_type={media_type} image_size={image.size} "
                f"question={question!r} prompt_tokens={len(prompt_token_ids)} "
                f"prompt_embeds_shape={prompt_embeds_shape} prompt_embeds_dtype={prompt_embeds_dtype} "
                f"prompt_position_ids_shape={prompt_position_shape} prompt_position_ids_dtype={prompt_position_dtype} "
                f"position_offset={int(position_offset or 0)}"
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
        discard_runtime_snapshot = bool(self.session.discard_runtime_snapshot_on_release)
        try:
            if discard_runtime_snapshot:
                self.session.runtime_snapshot = None
                if len(self.session.rendered_token_ids) > 0:
                    self.session.pending_replay_reason = "Deepy RAM unload discarded the cached runtime snapshot"
            elif self.runtime is not None and self.session.runtime_snapshot is None and len(self.session.rendered_token_ids) > 0:
                self.session.runtime_snapshot = self.runtime.snapshot_context()
        except Exception as exc:
            self._log(f"Resident snapshot before VRAM release failed: {exc}")
        try:
            self.runtime_hooks.unload_runtime()
        finally:
            self.runtime_hooks.unload_weights()
            self.runtime = None
            self.session.release_vram_callback = None
            self.session.discard_runtime_snapshot_on_release = False

    def _pause_runtime(self, pause_reason: str = "idle") -> None:
        keep_loaded = self.vram_mode in (DEEPY_VRAM_MODE_ALWAYS_LOADED, DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST)
        if pause_reason == "vision":
            keep_loaded = False
        if pause_reason == "tool" and self.vram_mode != DEEPY_VRAM_MODE_ALWAYS_LOADED:
            keep_loaded = False
        allow_force_release = keep_loaded and self.vram_mode == DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST and pause_reason != "tool"
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
        mode, runtime_replay_reason = self._run_prefill_call(
            len(fallback_tokens),
            lambda: runtime.restore_or_replay(self.session.runtime_snapshot, fallback_tokens),
            record_if=lambda result: isinstance(result, tuple) and len(result) > 0 and result[0] == "replayed",
        )
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
                    prefix_tokens = self._active_sequence_token_count()
                    prefix_tokens = len(self.session.rendered_token_ids) if prefix_tokens is None else prefix_tokens
                    mode = self._run_prefill_call(prefix_tokens + len(suffix_tokens), lambda: runtime.append_suffix(suffix_tokens), record_if=lambda result: result == "prefilled")
                    self._record_live_context("Generation context extended from live runtime. [suffix append only]" if mode == "extended" else "Generation context prefilled from live runtime. [prefill redone]" if mode == "prefilled" else f"Generation context {mode} from live runtime.")
                    return
            if restore_mode in ("reused", "restored") and self._can_append_pending_user_suffix():
                thinking_enabled = qwen35_text._prompt_enhancer_thinking_enabled(self.runtime.model, thinking_enabled=self.thinking_enabled)
                suffix_tokens = render_text_user_turn_suffix(runtime.tokenizer, self._pending_user_render_content(), thinking_enabled=thinking_enabled)
                if len(suffix_tokens) > 0:
                    prefix_tokens = self._active_sequence_token_count()
                    prefix_tokens = len(self.session.rendered_token_ids) if prefix_tokens is None else prefix_tokens
                    mode = self._run_prefill_call(prefix_tokens + len(suffix_tokens), lambda: runtime.append_suffix(suffix_tokens), record_if=lambda result: result == "prefilled")
                    self._record_live_context("Generation context extended from live runtime. [suffix append only]" if mode == "extended" else "Generation context prefilled from live runtime. [prefill redone]" if mode == "prefilled" else f"Generation context {mode} from live runtime.")
                    return
        target_tokens = self._fit_rendered_messages_to_window(add_generation_prompt=True, reserve_tokens=128)
        if len(self.session.rendered_token_ids) > 0:
            mode = self._run_prefill_call(len(target_tokens), lambda: runtime.extend_context(target_tokens), record_if=lambda result: result in ("prefilled", "replayed"))
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
        self._run_prefill_call(len(target_tokens), lambda: runtime.prime_context(target_tokens))
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
            self._emit_chat_event(assistant_chat.build_sync_event(self.session, stats=self._chat_stats_payload()))
            return result
        if len(tool_template) > 0:
            self._set_status(f"Using {tool_label} ({Path(tool_template).stem})...", kind="tool")
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
        self._emit_chat_event(assistant_chat.build_sync_event(self.session, stats=self._chat_stats_payload()))
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
        checkpoint_assistant_turn(self.session)
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
                self._emit_stats(force=True)
                if self.session.interrupt_requested:
                    break
                if show_loading_status:
                    self.session.force_loading_status_once = False
                    self._set_status("Thinking...", kind="thinking")
                self._start_stream_pass()
                result = None
                try:
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
                finally:
                    self._finish_stream_pass(None if result is None else result.token_count)
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
                        checkpoint_assistant_turn(self.session)
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
            self._prefill_started_at = None
            self._live_prefill_tokens = 0
            self._segment_started_at = None
            self._segment_generated_tokens = 0
            self._emit_stats(force=True)
        if not self.session.interrupt_requested and len(final_user_text.strip()) > 0:
            self._send_chat(final_user_text)
        if turn_completed and not self.session.interrupt_requested and len(self.session.interruption_notice.strip()) > 0:
            if self.debug_enabled:
                self._log("Clearing interruption notice after a successful follow-up turn.")
            self.session.interruption_notice = ""
