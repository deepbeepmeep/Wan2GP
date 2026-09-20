from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import markdown
from markdown.extensions.tables import TableExtension, TableProcessor

from shared.deepy import video_tools as deepy_video_tools
from shared.utils.gallery_media import gallery_media_ids
from shared.deepy.config import DEEPY_TYPE_PRIME, normalize_deepy_type


CHAT_HOST_ID = "assistant_chat_html"
CHAT_EVENT_ID = "assistant_chat_event"
SYNC_BUTTON_ID = "assistant_chat_sync_button"
DOCK_ID = "assistant_chat_dock"
LAUNCHER_HOST_ID = "assistant_chat_launcher_host"
LAUNCHER_BUTTON_ID = "assistant_chat_toggle"
PANEL_ID = "assistant_chat_panel"
SETTINGS_LAUNCHER_HOST_ID = "assistant_chat_settings_launcher_host"
SETTINGS_TOGGLE_ID = "assistant_chat_settings_toggle"
SETTINGS_PANEL_ID = "assistant_chat_settings_panel"
CHAT_BLOCK_ID = "assistant_chat_shell_block"
STATS_BLOCK_ID = "assistant_chat_stats_block"
STATS_ID = "assistant_chat_stats"
CONTROLS_ID = "assistant_chat_controls"
REQUEST_ID = "assistant_chat_request"
ASK_BUTTON_ID = "assistant_chat_ask_button"
RESET_BUTTON_ID = "assistant_chat_reset_button"
PAUSE_BRIDGE_ID = "assistant_chat_pause_bridge"
STOP_BRIDGE_ID = "assistant_chat_stop_bridge"
PAUSE_TOGGLE_ACTION = "__toggle_pause__"
BUSY_QUEUE_INPUT_ID = "assistant_chat_busy_queue_input"
BUSY_QUEUE_SUBMISSION_ID = "assistant_chat_busy_queue_submission_id"
BUSY_QUEUE_BUTTON_ID = "assistant_chat_busy_queue_button"
SUBMISSION_ID = "assistant_chat_submission_id"
STEER_INPUT_ID = "assistant_chat_steer_input"
STEER_SUBMISSION_ID = "assistant_chat_steer_submission_id"
STEER_BUTTON_ID = "assistant_chat_steer_button"
QUEUED_ACTION_INPUT_ID = "assistant_chat_queued_action_input"
QUEUED_ACTION_BUTTON_ID = "assistant_chat_queued_action_button"
SAVE_SETTINGS_BUTTON_ID = "assistant_chat_save_settings_button"
WELCOME_SESSION_INPUT_ID = "assistant_chat_welcome_session_input"
WELCOME_SESSION_BUTTON_ID = "assistant_chat_welcome_session_button"
SESSION_REFRESH_BUTTON_ID = "assistant_chat_session_refresh_button"
SESSION_PREFILL_BUTTON_ID = "assistant_chat_session_prefill_button"
SESSION_RESUME_BUTTON_ID = "assistant_chat_session_resume_button"
SESSION_RENAME_BUTTON_ID = "assistant_chat_session_rename_button"
SESSION_DUPLICATE_BUTTON_ID = "assistant_chat_session_duplicate_button"
SESSION_EXPORT_BUTTON_ID = "assistant_chat_session_export_button"
SESSION_IMPORT_BUTTON_ID = "assistant_chat_session_import_button"
SESSION_DELETE_BUTTON_ID = "assistant_chat_session_delete_button"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jfif", ".pjpeg"}
_VIDEO_EXTENSIONS = deepy_video_tools.VIDEO_EXTENSIONS
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus"}
_ARCHIVE_EXTENSIONS = {".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz"}


class _ChatTableProcessor(TableProcessor):
    """Allow a table to start immediately after prose, as in generated answers."""

    def test(self, parent, block):
        lines = block.split("\n")
        for start in range(len(lines) - 1):
            if super().test(parent, "\n".join(lines[start:start + 2])) and super().test(parent, "\n".join(lines[start:])):
                self._table_start = start
                return True
        return False

    def run(self, parent, blocks):
        start = self._table_start
        if start == 0:
            return super().run(parent, blocks)
        lines = blocks.pop(0).split("\n")
        self.parser.parseBlocks(parent, ["\n".join(lines[:start]), "\n".join(lines[start:])])
        return True


class _ChatTableExtension(TableExtension):
    def extendMarkdown(self, md):
        super().extendMarkdown(md)
        md.parser.blockprocessors.register(_ChatTableProcessor(md.parser, self.getConfigs()), "table", 75)


_MARKDOWN_EXTENSIONS = ["extra", "nl2br", "sane_lists", "fenced_code", _ChatTableExtension()]
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
_DOWNLOAD_MARKDOWN_TOKEN_RE = re.compile(r"(?P<fence>```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$))|(?P<link>!?\[(?:\\.|`[^`\n]*`|[^\]\n])*\]\([^\n)]*\))|(?P<code>`[^`\n]+`)")
_DOWNLOAD_LINK_RE = re.compile(r"!?\[(?:\\.|`[^`\n]*`|[^\]\n])*\]\([^\n)]*\)")
_GALLERY_MEDIA_ID_RE = re.compile(r"(?:visual|audio):[a-f0-9]{12}", re.IGNORECASE)
_ABSOLUTE_PATH_START_RE = re.compile(r"(?<![\w:/])(?:[A-Za-z]:[\\/]|\\\\|/(?!/))")
_TOOL_RESULT_PATH_KEYS = {"destination", "generated_files", "output_file", "output_files", "path", "paths", "source", "sources"}
_DOWNLOAD_REFERENCE_LIMIT = 20
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_AUDIO_THUMBNAIL_PATH = os.path.join(_REPO_ROOT, "icons", "soundwave.jpg")
_ARCHIVE_THUMBNAIL_PATH = os.path.join(_REPO_ROOT, "icons", "zip.svg")
SERVER_INSTANCE_ID = uuid.uuid4().hex
_UNSET = object()


def _session_picker_markup(sessions: list[dict[str, Any]] | None, active_id: str = "", enabled: bool = False) -> str:
    if not enabled:
        return ""
    choices = []
    for item in list(sessions or []):
        storage_id = str(item.get("id", "") or "").strip()
        if not storage_id:
            continue
        title = str(item.get("title", "") or "Deepy session").strip()
        selected = " selected" if storage_id == str(active_id or "") else ""
        choices.append(f"<option value='{html.escape(storage_id, quote=True)}'{selected}>{html.escape(title)}</option>")
    disabled = " disabled" if not choices else ""
    options = "".join(choices) if choices else "<option value=''>No saved sessions</option>"
    return (
        "<div class='chat__session-picker'>"
        "<div class='chat__session-picker-spacer' aria-hidden='true'></div>"
        "<label><span>Saved sessions</span><span class='chat__session-picker-controls'>"
        f"<select aria-label='Deepy session' data-wac-session-picker{disabled}>{options}</select>"
        f"<button type='button' data-wac-session-resume aria-label='Resume selected session' title='Resume selected session'{disabled}>Resume</button>"
        "</span></label>"
        "</div>"
    )


def _empty_state_markup(deepy_type: str, sessions: list[dict[str, Any]] | None = None, active_id: str = "", multi_session_enabled: bool = False) -> str:
    if normalize_deepy_type(deepy_type) == DEEPY_TYPE_PRIME:
        title = "Deepy Prime"
        mode = "Advanced creative orchestration"
        intro = "Describe the result you want and Deepy Prime can plan the work, choose suitable models and tools, and connect multiple image, video, and audio steps into one creative workflow."
        benefits = (
            "Plan and complete multi-step projects that create, inspect, edit, and combine several pieces of media.",
            "Choose among available WanGP models and settings according to your goal, quality preference, and source media.",
            "Build on Gallery items or existing files, then extract, transcribe, resize, add sound, upscale, or continue generating.",
            "Extend the workflow with other connected services when they are available.",
        )
        examples = (
            "Create a character portrait and related keyframes, then turn them into a longer video with a soundtrack.",
            "Inspect the selected video, improve the weak sections, upscale it, and prepare a subtitled version.",
            "Design an album cover, write a matching song, and create a short promotional video from both.",
        )
    else:
        title = "Deepy Zero"
        mode = "Fast, focused creation"
        intro = "Deepy Zero is the lightweight assistant for straightforward requests. It uses the models and templates selected in Deepy Settings, making it a good match for smaller LLMs, quick responses, and familiar results."
        benefits = (
            "Generate an image, video, speech clip, or song with your preferred templates and defaults.",
            "Handle focused edits and practical media tasks without requiring a complex workflow.",
            "Refer naturally to the selected, latest, or previous Gallery item.",
            "See each generation and completed result in the normal WanGP queue and Galleries.",
        )
        examples = (
            "Generate a square album cover showing a robot jazz band.",
            "Animate the selected image as a five-second cinematic shot.",
            "Transcribe the last video or resize it for social media.",
        )
    benefit_items = "".join(f"<li>{html.escape(item)}</li>" for item in benefits)
    example_items = "".join(f"<li>{html.escape(item)}</li>" for item in examples)
    return (
        "<div class='chat__empty-card'>"
        "<header class='chat__empty-header'>"
        "<span class='chat__empty-eyebrow'>Current assistant</span>"
        f"<h2 class='chat__empty-title'>{html.escape(title)}</h2>"
        f"<span class='chat__empty-mode'>{html.escape(mode)}</span>"
        "</header>"
        f"<p class='chat__empty-intro'>{html.escape(intro)}</p>"
        "<div class='chat__empty-grid'>"
        f"<section class='chat__empty-section'><h3>What it does for you</h3><ul>{benefit_items}</ul></section>"
        f"<section class='chat__empty-section chat__empty-section--examples'><h3>Try asking</h3><ul>{example_items}</ul></section>"
        "</div>"
        "<p class='chat__empty-tip'>Start with the outcome you want. Deepy will ask only when an important choice is missing.</p>"
        f"{_session_picker_markup(sessions, active_id, multi_session_enabled)}"
        "</div>"
    )


def _shell_markup(deepy_type: str = "", sessions: list[dict[str, Any]] | None = None, active_id: str = "", multi_session_enabled: bool = False) -> str:
    return f"""
<section class="chat">
  <div class="chat__scroll">
    <div class="chat__empty">
      {_empty_state_markup(deepy_type, sessions, active_id, multi_session_enabled)}
    </div>
    <div class="chat__transcript"></div>
  </div>
  <div class="chat__status" aria-live="polite">
    <div class="chat__status-dots" aria-hidden="true"><span></span><span></span><span></span></div>
    <div class="chat__status-text"></div>
    <button class="chat__status-pause" type="button" aria-label="Pause Deepy" disabled>Pause</button>
    <button class="chat__status-stop" type="button" aria-label="Stop Deepy" disabled>Stop</button>
  </div>
  <button class="chat__jump-bottom" type="button" aria-label="Jump to latest messages" aria-hidden="true" tabindex="-1">
    <span aria-hidden="true"></span>
  </button>
</section>
""".strip()


def render_shell_html(deepy_type: str = "", sessions: list[dict[str, Any]] | None = None, active_id: str = "", multi_session_enabled: bool = False) -> str:
    catalog = html.escape(json.dumps(list(sessions or []), ensure_ascii=False), quote=True)
    return f"<div id='{CHAT_HOST_ID}' data-wangp-assistant-chat-mounted='true' data-deepy-type='{html.escape(normalize_deepy_type(deepy_type))}' data-session-catalog='{catalog}' data-active-session-id='{html.escape(active_id, quote=True)}' data-multi-session-enabled='{str(bool(multi_session_enabled)).lower()}'>{_shell_markup(deepy_type, sessions, active_id, multi_session_enabled)}</div>"


def render_stats_html() -> str:
    return f"<div id='{STATS_ID}' class='chat__stats' aria-hidden='true'><span class='chat__input-helper' aria-hidden='true'>Press Enter to Queue Requests / CTRL Enter to Steer Deepy</span><span class='chat__stats-text'></span></div>"


def render_launcher_html() -> str:
    return (
        f"<button id='{LAUNCHER_BUTTON_ID}' class='chat__toggle' type='button' "
        "aria-label='Toggle Deepy assistant' aria-expanded='false'>"
        "<span class='chat__toggle-text'>Ask Deepy</span>"
        "</button>"
    )


def render_settings_launcher_html() -> str:
    return (
        f"<button id='{SETTINGS_TOGGLE_ID}' class='chat__settings-toggle' type='button' "
        "aria-label='Toggle Deepy settings' aria-expanded='false'>"
        "<span class='chat__settings-toggle-text'>Settings</span>"
        "</button>"
    )


def get_css() -> str:
    return '\n'.join((Path(__file__).with_name("web") / name).read_text(encoding="utf-8") for name in ('chat.css', 'workspaces.css', 'workspace_viewer.css'))


def get_javascript() -> str:
    return (Path(__file__).parents[1] / 'gradio/form_sync.js').read_text(encoding='utf-8') + '\n' + "\n".join((Path(__file__).with_name("web") / name).read_text(encoding="utf-8") for name in ("transport.js", "gradio_transport.js", "chat.js", "compact_actions.js", "voice.js", "workspaces.js", "gallery_selection.js", "media_view.js", "workspace_viewer.js", "hybrid_transport.js"))


def _touch_chat(session) -> int:
    session.chat_revision = int(session.chat_revision or 0) + 1
    return session.chat_revision


def reset_session_chat(session) -> None:
    session.chat_turn_durations = {}
    session.chat_transcript.clear()
    session.chat_transcript_counter = 0
    session.chat_status = None
    _touch_chat(session)


def build_reset_event(session=None) -> str:
    return _event_payload({"type": "reset"}, session)


def _pause_aware_status(session, status: dict[str, Any] | None) -> dict[str, Any] | None:
    if session is None:
        return status
    if bool(getattr(session, "paused", False)):
        return {"visible": True, "kind": "paused", "text": "Deepy is paused."}
    if bool(getattr(session, "pause_requested", False)):
        text = "Pausing media processing at the next checkpoint..." if getattr(session, "media_tool_active", False) else ("Pausing after the current tool finishes..." if bool(getattr(session, "assistant_action_active", False)) else "Pausing Deepy...")
        return {"visible": True, "kind": "pause_pending", "text": text}
    return status


def build_status_event(text: str | None, kind: str = "status", visible: bool = True, stats: dict[str, Any] | None = None, session=None) -> str:
    status = None if not visible or not text else {"visible": True, "kind": str(kind or "status"), "text": str(text or "").strip()}
    status = _pause_aware_status(session, status)
    if session is not None:
        session.chat_status = status
    event = {"type": "status", "status": status}
    if stats is not None:
        event["stats"] = stats
    return _event_payload(event, session)


def build_stats_event(stats: dict[str, Any] | None = None) -> str:
    return _event_payload({"type": "stats", "stats": stats})


def build_session_catalog_event(sessions: list[dict[str, Any]], active_session_id: str = "", multi_session_enabled: bool = False) -> str:
    return _event_payload({"type": "session_catalog", "sessions": list(sessions or []), "active_session_id": str(active_session_id or ""), "multi_session_enabled": bool(multi_session_enabled)})


def build_session_resume_ready_event(request_id: str, session=None) -> str:
    return _event_payload({"type": "session_resume_ready", "request_id": str(request_id or "")})


def build_event_batch(payloads: list[str], *, replay: bool = False) -> str:
    envelopes = []
    for payload in payloads or []:
        payload_text = str(payload or "").strip()
        if len(payload_text) == 0:
            continue
        try:
            envelope = json.loads(payload_text)
        except Exception:
            continue
        if isinstance(envelope, dict):
            envelopes.append(envelope)
    if len(envelopes) == 0:
        return ""
    if len(envelopes) == 1:
        return json.dumps(envelopes[0], ensure_ascii=False)
    return json.dumps({"event_id": uuid.uuid4().hex, "instance_id": SERVER_INSTANCE_ID, "batch": envelopes, "replay": bool(replay)}, ensure_ascii=False)


def build_replay_batch(session, commands: list[dict[str, Any]]) -> str:
    payloads = [build_reset_event(session)]
    for command in list(commands or []):
        if not isinstance(command, dict) or str(command.get("cmd", "") or "") != "chat_output" or not isinstance(command.get("event"), dict):
            continue
        event = dict(command["event"])
        for key in ("chat_session_id", "revision", "sequence", "sequence_start"):
            event.pop(key, None)
        payloads.append(_event_payload(event, session))
    return build_event_batch(payloads, replay=True)


def _message_has_renderable_output(record: dict[str, Any]) -> bool:
    return str(record.get("role", "")).strip() != "assistant" or bool(_ensure_message_blocks(record)) or bool(record.get("attachments"))


def pending_upload_count(session) -> int:
    return sum(bool(item.get('chat_upload')) for item in session.pending_chat_media)


def build_pending_upload_event(session) -> str:
    return _event_payload({'type': 'pending_uploads', 'pending_upload_count': pending_upload_count(session)}, session, _touch_chat(session))


def build_sync_event(session, status: dict[str, Any] | None | object = _UNSET, stats: dict[str, Any] | None = None, acknowledged_submission_ids: list[str] | tuple[str, ...] | None = None) -> str:
    if status is _UNSET:
        status = getattr(session, "chat_status", None)
    status = _pause_aware_status(session, status)
    session.chat_status = status
    while True:
        revision = int(session.chat_revision or 0)
        messages = [_render_message_payload(record) for record in list(session.chat_transcript) if _message_has_renderable_output(record)]
        if revision == int(session.chat_revision or 0):
            break
    event = {"type": "sync", "messages": messages, "status": status, "pending_upload_count": pending_upload_count(session)}
    if stats is None:
        stored_stats = getattr(session, "remote_usage_stats", None)
        if isinstance(stored_stats, dict):
            stats = stored_stats
    if stats is not None:
        event["stats"] = stats
    if acknowledged_submission_ids is not None:
        event["acknowledged_submission_ids"] = [str(submission_id or "").strip() for submission_id in acknowledged_submission_ids if str(submission_id or "").strip()]
    return _event_payload(event, session, revision)


def _queued_tail_insert_index(session) -> int:
    records = list(session.chat_transcript or [])
    insert_index = len(records)
    while insert_index > 0:
        record = records[insert_index - 1]
        if not isinstance(record, dict):
            break
        if str(record.get("role", "")).strip() != "user":
            break
        if not bool(record.get("queued", False)):
            break
        insert_index -= 1
    return insert_index


def add_user_message(session, text: str, queued: bool = False, client_submission_id: str | None = None) -> tuple[str, str]:
    submission_id = str(client_submission_id or "").strip()[:128]
    record = {
        "id": _next_message_id(session, "user"),
        "role": "user",
        "author": "You",
        "created_at": _time_label(),
        "blocks": [],
        "attachments": [],
        "badge": "Queued" if queued else "",
        "queued": bool(queued),
    }
    if submission_id:
        record["client_submission_id"] = submission_id
    content = str(text or "").strip()
    if len(content) > 0:
        record["blocks"].append({"id": _next_block_id("content"), "type": "markdown", "text": content})
    with session.turn_lock:
        if session.pending_chat_media:
            record["runtime_media"] = [{key: value for key, value in item.items() if key != 'chat_upload'} for item in session.pending_chat_media]
            session.pending_chat_media = []
        session.chat_transcript.append(record)
    revision = _touch_chat(session)
    return record["id"], _message_upsert_event(session, record, revision)


def create_assistant_turn(session) -> str:
    record = {
        "id": _next_message_id(session, "assistant"),
        "role": "assistant",
        "author": "Deepy",
        "created_at": _time_label(),
        "blocks": [],
        "attachments": [],
        "badge": "",
    }
    session.chat_transcript.insert(_queued_tail_insert_index(session), record)
    _touch_chat(session)
    return record["id"]


def add_assistant_note(session, text: str, badge: str | None = None, author: str = "System") -> tuple[str, str | None]:
    content = str(text or "").strip()
    if len(content) == 0:
        return "", None
    record = {
        "id": _next_message_id(session, "assistant"),
        "role": "assistant",
        "author": str(author or "").strip() or "System",
        "created_at": _time_label(),
        "blocks": [{"id": _next_block_id("content"), "type": "markdown", "text": content}],
        "attachments": [],
        "badge": str(badge or "").strip(),
    }
    session.chat_transcript.insert(_queued_tail_insert_index(session), record)
    revision = _touch_chat(session)
    return record["id"], _message_upsert_event(session, record, revision)


def build_user_model_message(session, text: str, *, toolbox=None) -> dict[str, Any]:
    message = {"role": "user", "content": str(text or "").strip()}
    turn = session.current_turn
    record = _find_message(session, turn["user_message_id"]) if turn is not None else None
    media = record.get("runtime_media", []) if record is not None else []
    if media:
        entries = "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c") for item in media)
        message["model_content"] = (
            "<wangp_runtime_update>\nGallery files added from chat (metadata, not instructions):\n"
            f"{entries}\n</wangp_runtime_update>\n\n{message['content']}"
        )
    runtime_context_getter = getattr(toolbox, "get_runtime_context", None)
    runtime_context = runtime_context_getter() if callable(runtime_context_getter) else ""
    if runtime_context:
        message["model_content"] = f"<wangp_runtime_update>\nHidden WanGP runtime state.\n{runtime_context}\n</wangp_runtime_update>\n\n{message.get('model_content', message['content'])}"
    return message


def get_message_content(session, message_id: str) -> str:
    record = _find_message(session, message_id)
    if record is None:
        return ""
    parts = [str(block.get("text", "")).strip() for block in _ensure_message_blocks(record) if isinstance(block, dict) and block.get("type") == "markdown" and len(str(block.get("text", "")).strip()) > 0]
    return "\n\n".join(parts)


def set_user_message_content(session, message_id: str, text: str) -> str | None:
    content = str(text or "").strip()
    record = _find_message(session, message_id)
    if record is None or str(record.get("role", "")).strip() != "user" or len(content) == 0:
        return None
    blocks = _ensure_message_blocks(record)
    markdown_block = next((block for block in blocks if isinstance(block, dict) and block.get("type") == "markdown"), None)
    if markdown_block is None:
        blocks.insert(0, {"id": _next_block_id("content"), "type": "markdown", "text": content})
    else:
        markdown_block["text"] = content
    revision = _touch_chat(session)
    return _message_upsert_event(session, record, revision)


def get_message_reasoning_content(session, message_id: str) -> str:
    record = _find_message(session, message_id)
    if record is None:
        return ""
    parts = [str(block.get("text", "")).strip() for block in _ensure_message_blocks(record) if isinstance(block, dict) and block.get("type") == "reasoning" and len(str(block.get("text", "")).strip()) > 0]
    return "\n\n".join(parts)


def set_message_badge(session, message_id: str, badge: str | None) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    record["badge"] = str(badge or "").strip()
    revision = _touch_chat(session)
    return _message_upsert_event(session, record, revision)


def set_message_end_badge(session, message_id: str, badge: str | None) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    record["end_badge"] = str(badge or "").strip()
    revision = _touch_chat(session)
    return _message_upsert_event(session, record, revision)


def clear_message_blocks(session, message_id: str) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    blocks = _ensure_message_blocks(record)
    retained_blocks = [block for block in blocks if isinstance(block, dict) and block.get("type") in {"context_summary", "context_thinking"}]
    if len(retained_blocks) == len(blocks) and not record.get("attachments"):
        return None
    record["blocks"] = retained_blocks
    record["attachments"] = []
    revision = _touch_chat(session)
    return _message_upsert_event(session, record, revision)


def clear_assistant_content(session, message_id: str) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    blocks = _ensure_message_blocks(record)
    kept_blocks = [block for block in blocks if not (isinstance(block, dict) and block.get("type") == "markdown")]
    if len(kept_blocks) == len(blocks):
        return None
    record["blocks"] = kept_blocks
    record["content"] = ""
    revision = _touch_chat(session)
    return _message_upsert_event(session, record, revision)


def remove_message(session, message_id: str) -> str | None:
    target_id = str(message_id or "").strip()
    if len(target_id) == 0:
        return None
    original_len = len(session.chat_transcript)
    session.chat_transcript[:] = [record for record in session.chat_transcript if str(record.get("id", "")) != target_id]
    if len(session.chat_transcript) == original_len:
        return None
    revision = _touch_chat(session)
    return _event_payload({"type": "remove_message", "message_id": target_id}, session, revision)


def append_reasoning(session, message_id: str, text: str) -> str | None:
    _reasoning_id, payload = upsert_reasoning_block(session, message_id, None, text, streaming=False)
    return payload


def add_context_summary(session, message_id: str, text: str) -> tuple[str, str | None]:
    return upsert_context_summary(session, message_id, None, text, streaming=False)


def upsert_context_summary(session, message_id: str, summary_id: str | None, text: str, streaming: bool = False) -> tuple[str, str | None]:
    return _upsert_text_block(session, message_id, summary_id, "context_summary", text, streaming=streaming)


def upsert_context_thinking(session, message_id: str, thinking_id: str | None, text: str, streaming: bool = False) -> tuple[str, str | None]:
    return _upsert_text_block(session, message_id, thinking_id, "context_thinking", text, streaming=streaming)


def remove_message_block(session, message_id: str, block_id: str) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    target_id = str(block_id or "").strip()
    blocks = _ensure_message_blocks(record)
    removed = next((block for block in blocks if isinstance(block, dict) and str(block.get("id", "")) == target_id), None)
    retained = [block for block in blocks if not isinstance(block, dict) or str(block.get("id", "")) != target_id]
    if len(retained) == len(blocks):
        return None
    record["blocks"] = retained
    revision = _touch_chat(session)
    return _event_payload({"type": "remove_block", "message_id": str(message_id), "block_id": target_id, "block_type": str(removed.get("type", "markdown"))}, session, revision)


def steer_queued_message(session, message_id: str, *, status_text: str = "Steering accepted. Applying this request at the current boundary...", acknowledged_submission_ids: list[str] | tuple[str, ...] | None = None) -> str | None:
    return steer_queued_messages(session, [message_id], status_text=status_text, acknowledged_submission_ids=acknowledged_submission_ids)


def steer_queued_messages(session, message_ids: list[str] | tuple[str, ...], *, status_text: str = "Steering accepted. Applying this request at the current boundary...", acknowledged_submission_ids: list[str] | tuple[str, ...] | None = None) -> str | None:
    records = []
    for message_id in message_ids:
        record = _find_message(session, message_id)
        if record is None or str(record.get("role", "")).strip() != "user" or not bool(record.get("queued", False)):
            return None
        records.append(record)
    if not records:
        return None
    transcript = session.chat_transcript
    for record in records:
        transcript.remove(record)
    records[0]["badge"] = "Steered"
    records[0]["assistant_badge"] = "Steered"
    insert_index = _queued_tail_insert_index(session)
    transcript[insert_index:insert_index] = records
    _touch_chat(session)
    return build_sync_event(session, status={"visible": True, "kind": "queued", "text": status_text}, acknowledged_submission_ids=acknowledged_submission_ids)


def upsert_reasoning_block(session, message_id: str, reasoning_id: str | None, text: str, streaming: bool = True) -> tuple[str, str | None]:
    return _upsert_text_block(session, message_id, reasoning_id, "reasoning", text, streaming=streaming)


def add_tool_call(session, message_id: str, tool_name: str, arguments: dict[str, Any], tool_label: str | None = None, request_pending: bool = False) -> tuple[str, str | None]:
    record = _find_message(session, message_id)
    if record is None:
        return "", None
    tool_record = {
        "id": _next_tool_id(),
        "type": "tool",
        "name": str(tool_name or "").strip(),
        "label": str(tool_label or "").strip() or _friendly_tool_label(tool_name),
        "arguments": dict(arguments or {}),
        "result": None,
        "status": "running",
        "status_text": "Preparing" if request_pending else "Running",
        "request_pending": bool(request_pending),
        "attachment": None,
        "attachments": [],
    }
    _ensure_message_blocks(record).append(tool_record)
    revision = _touch_chat(session)
    return tool_record["id"], _block_upsert_event(session, record, tool_record, revision)


def update_tool_call(session, message_id: str, tool_id: str, status: str | None = None, result: dict[str, Any] | object = _UNSET, status_text: str | None = None, tool_name: str | None = None, tool_label: str | None = None, arguments: dict[str, Any] | object = _UNSET, request_pending: bool | None = None) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    for tool_record in _ensure_message_blocks(record):
        if not isinstance(tool_record, dict) or tool_record.get("type") != "tool" or tool_record.get("id") != tool_id:
            continue
        if status is not None:
            tool_record["status"] = str(status or "").strip().lower() or "running"
        if status_text is not None:
            tool_record["status_text"] = str(status_text or "").strip()
        if tool_name is not None:
            tool_record["name"] = str(tool_name or "").strip()
        if tool_label is not None:
            tool_record["label"] = str(tool_label or "").strip()
        if arguments is not _UNSET:
            tool_record["arguments"] = dict(arguments or {})
        if request_pending is not None:
            tool_record["request_pending"] = bool(request_pending)
        if result is not _UNSET:
            tool_record["result"] = None if result is None else dict(result or {})
            tool_record["attachments"] = _attachments_from_tool_result(tool_record.get("result"), getattr(session, "file_access_policy", None))
            tool_record["attachment"] = tool_record["attachments"][-1] if tool_record["attachments"] else None
            tool_record["presentation"] = _structured_tool_presentation(tool_record, getattr(session, "file_access_policy", None))
        revision = _touch_chat(session)
        return _block_upsert_event(session, record, tool_record, revision)
    return None


def complete_tool_call(session, message_id: str, tool_id: str, result: dict[str, Any]) -> str | None:
    status = str((result or {}).get("status", "")).strip().lower()
    failed = status in {"error", "failed", "interrupted"}
    return update_tool_call(session, message_id, tool_id, status="error" if failed else "done", result=result, status_text="Interrupted" if status == "interrupted" else ("Error" if failed else "Done"))


def upsert_assistant_content_block(session, message_id: str, content_id: str | None, text: str, streaming: bool = True) -> tuple[str, str | None]:
    return _upsert_text_block(session, message_id, content_id, "markdown", text, streaming=streaming)


def remove_assistant_content_block(session, message_id: str, content_id: str) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    blocks = _ensure_message_blocks(record)
    target_id = str(content_id or "").strip()
    for index, block in enumerate(blocks):
        if isinstance(block, dict) and block.get("type") == "markdown" and block.get("id", "") == target_id:
            del blocks[index]
            revision = _touch_chat(session)
            return _event_payload({"type": "remove_block", "message_id": str(message_id), "block_id": target_id, "block_type": "markdown"}, session, revision)
    return None


def set_assistant_content(session, message_id: str, text: str) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    content_text = str(text or "").strip()
    if len(content_text) == 0:
        return None
    blocks = _ensure_message_blocks(record)
    if len(blocks) > 0 and isinstance(blocks[-1], dict) and blocks[-1].get("type") == "markdown":
        if str(blocks[-1].get("text", "")).strip() == content_text:
            return None
        blocks[-1]["text"] = content_text
    else:
        blocks.append({"id": _next_block_id("content"), "type": "markdown", "text": content_text})
    revision = _touch_chat(session)
    return _message_upsert_event(session, record, revision)


def _next_message_id(session, prefix: str) -> str:
    session.chat_transcript_counter += 1
    return f"{prefix}_{session.chat_transcript_counter}"


def _next_tool_id() -> str:
    return f"tool_{uuid.uuid4().hex[:10]}"


def _next_block_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _message_frame_payload(record: dict[str, Any]) -> dict[str, Any]:
    return _compose_message_payload(record, "", "")


def _message_index(session, record: dict[str, Any]) -> int:
    record_id = str(record.get("id", ""))
    return next(index for index, candidate in enumerate(session.chat_transcript) if candidate is record or str(candidate.get("id", "")) == record_id)


def _message_upsert_event(session, record: dict[str, Any], revision: int) -> str:
    event = {"type": "upsert_message", "message": _render_message_payload(record), "message_index": _message_index(session, record)}
    if record['role'] == 'user':
        event['pending_upload_count'] = pending_upload_count(session)
    return _event_payload(event, session, revision)


def _block_index(record: dict[str, Any], block_id: str) -> int:
    return next(index for index, block in enumerate(_ensure_message_blocks(record)) if isinstance(block, dict) and str(block.get("id", "")) == str(block_id))


def _block_upsert_event(session, record: dict[str, Any], block: dict[str, Any], revision: int) -> str:
    block_type = str(block.get("type", "markdown"))
    return _event_payload({
        "type": "upsert_block",
        "message_id": str(record.get("id", "")),
        "message": _message_frame_payload(record),
        "message_index": _message_index(session, record),
        "block_id": str(block.get("id", "")),
        "block_type": block_type,
        "block_index": _block_index(record, str(block.get("id", ""))),
        "html": _render_block_html(record, block, streaming=False),
    }, session, revision)


def _upsert_text_block(session, message_id: str, block_id: str | None, block_type: str, text: str, *, streaming: bool) -> tuple[str, str | None]:
    canonical_text = str(text or "").lstrip() if streaming else str(text or "").strip()
    if not canonical_text.strip():
        return "", None
    record = _find_message(session, message_id)
    if record is None:
        return "", None
    blocks = _ensure_message_blocks(record)
    target_id = str(block_id or "").strip()
    block = next((item for item in blocks if isinstance(item, dict) and item.get("type") == block_type and item.get("id", "") == target_id), None)
    if block is None:
        target_id = target_id or _next_block_id("content" if block_type == "markdown" else block_type)
        block = {"id": target_id, "type": block_type, "text": canonical_text, "streaming": bool(streaming), "_published_text": canonical_text}
        blocks.append(block)
        revision = _touch_chat(session)
        event = {
            "type": "upsert_block",
            "message_id": str(message_id),
            "message": _message_frame_payload(record),
            "message_index": _message_index(session, record),
            "block_id": target_id,
            "block_type": block_type,
            "block_index": len(blocks) - 1,
            "html": _render_block_html(record, block, streaming=streaming),
            "text": canonical_text,
            "text_start": 0,
            "text_end": len(canonical_text),
            "streaming": bool(streaming),
        }
        return target_id, _event_payload(event, session, revision)

    previous_text = str(block.get("text", ""))
    previous_streaming = bool(block.get("streaming", False))
    published_text = str(block.get("_published_text", previous_text))
    if canonical_text == previous_text and bool(streaming) == previous_streaming:
        return target_id, None
    block["text"] = canonical_text
    block["streaming"] = bool(streaming)
    block["_published_text"] = canonical_text
    revision = _touch_chat(session)
    common = {
        "message_id": str(message_id),
        "block_id": target_id,
        "block_type": block_type,
        "block_index": _block_index(record, target_id),
    }
    if not streaming:
        return target_id, _event_payload({**common, "type": "finalize_block", "message": _message_frame_payload(record), "message_index": _message_index(session, record), "text": canonical_text, "text_end": len(canonical_text), "html": _render_block_html(record, block, streaming=False)}, session, revision)
    if canonical_text.startswith(published_text):
        suffix = canonical_text[len(published_text):]
        if len(suffix) == 0:
            return target_id, _event_payload({**common, "type": "replace_block_text", "text": canonical_text, "text_start": 0, "text_end": len(canonical_text)}, session, revision)
        return target_id, _event_payload({**common, "type": "append_block_text", "text": suffix, "text_start": len(published_text), "text_end": len(canonical_text)}, session, revision)
    return target_id, _event_payload({**common, "type": "replace_block_text", "text": canonical_text, "text_start": 0, "text_end": len(canonical_text)}, session, revision)


def _friendly_tool_label(tool_name: str | None) -> str:
    name = str(tool_name or "").strip()
    if len(name) == 0:
        return "Tool"
    if name.startswith("wangp_"):
        name = name[len("wangp_"):]
    return name.replace("_", " ").replace("-", " ").strip().title()


def _clean_display_filename(text: str) -> str:
    return re.sub(r"(?<![\w])\d{4}-\d{2}-\d{2}-\d{2}h\d{2}m\d{2}s_seed-?\d+_", "", text)


def _short_tool_label_value(value: Any, max_chars: int = 42) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    text = _clean_display_filename(re.sub(r"\s+", " ", str(value or "")).strip())
    if len(text) == 0:
        return ""
    if "/" in text or "\\" in text:
        text = os.path.basename(text.replace("\\", "/")) or text
    return text if len(text) <= max_chars else f"{text[:max_chars - 1].rstrip()}…"


def _humanize_tool_value(value: Any) -> str:
    text = _short_tool_label_value(value)
    if len(text) == 0:
        return ""
    aliases = {
        "edit_image": "Edit Image",
        "gen_image": "Generate Image",
        "gen_song": "Generate Song",
        "gen_speech_from_description": "Generate Speech From Description",
        "gen_speech_from_sample": "Generate Speech From Sample",
        "gen_video": "Generate Video",
        "gen_video_with_speech": "Generate Video With Speech",
    }
    if text.casefold() in aliases:
        return aliases[text.casefold()]
    words = re.sub(r"[_-]+", " ", text).split()
    special = {
        "api": "API",
        "audio": "Audio",
        "doc": "Doc",
        "id": "ID",
        "image": "Image",
        "lora": "LoRA",
        "loras": "LoRAs",
        "mcp": "MCP",
        "media": "Media",
        "ui": "UI",
        "url": "URL",
        "video": "Video",
        "wangp": "WanGP",
    }
    return " ".join(special.get(word.casefold(), word if any(character.isupper() for character in word) else word.title()) for word in words)


def _finish_tool_call_label(label: str) -> str:
    compact = _clean_display_filename(re.sub(r"\s+", " ", str(label or "")).strip()) or "Tool"
    return compact if len(compact) <= 96 else f"{compact[:95].rstrip()}…"


def _tool_filter_label(filters: Any) -> str:
    if not isinstance(filters, dict):
        return ""
    parts = []
    for key, value in filters.items():
        rendered = _short_tool_label_value(value, 24)
        if rendered:
            parts.append(f"{_humanize_tool_value(key)}: {rendered}")
    return ", ".join(parts)


def build_io_tool_call_label(action: str | None = None, arguments: dict[str, Any] | None = None) -> str:
    """Build the chat-only label for the compact IO toolbox."""

    action_name = str(action or "").strip()
    if not action_name:
        return "List IO Tools"
    action_label = {"list": "List Files", "info": "Get File Information", "read_text": "Read Text", "search_text": "Search Text", "write_text": "Write Text", "write_artifact_text": "Compile Artifact", "mkdir": "Create Directory", "copy": "Copy File", "move": "Move File or Directory", "delete": "Delete File or Directory", "zip": "Create ZIP", "unzip": "Extract ZIP", "download": "Prepare Download"}.get(action_name, _humanize_tool_value(action_name))
    if arguments is None:
        return _finish_tool_call_label(f"Get {action_label} Schema")

    arguments = dict(arguments)
    source = _short_tool_label_value(arguments.get("source") or arguments.get("path"))
    destination = _short_tool_label_value(arguments.get("destination"))
    if source and destination == source:
        destination_parts = str(arguments.get("destination") or "").replace("\\", "/").split("/")
        if len(destination_parts) > 1:
            destination = f"{destination_parts[-2]}/{destination}"
    if action_name == "list":
        label = "List Filesystem Roots" if not source else f"List {'Files Recursively' if arguments.get('recursive') else 'Files'} in {source}"
        pattern = _short_tool_label_value(arguments.get("pattern"))
        return _finish_tool_call_label(label if not pattern or pattern == "*" else f"{label} Matching {pattern}")
    if action_name == "info":
        return _finish_tool_call_label(action_label if not source else f"{action_label} for {source}")
    if action_name == "read_text":
        start, end = arguments.get("start_line"), arguments.get("end_line")
        if start is not None and end is not None:
            return _finish_tool_call_label(f"Read Lines {start}–{end}{f' from {source}' if source else ''}")
        return _finish_tool_call_label(f"Read {source or 'Text'}{f' from Line {start}' if start is not None else ''}")
    if action_name == "search_text":
        query = _short_tool_label_value(arguments.get("query"), 28)
        label = f"Search {source or 'Text Files'}" + (f" for “{query}”" if query else "")
        pattern = _short_tool_label_value(arguments.get("pattern"))
        return _finish_tool_call_label(label if not pattern or pattern == "*" else f"{label} Matching {pattern}")
    if action_name == "write_text":
        mode = str(arguments.get("mode", "create") or "create").casefold()
        verb = "Append to" if mode == "append" else "Create" if mode == "create" else "Overwrite"
        return _finish_tool_call_label(f"{verb} Text File{f' {source}' if source else ''}")
    if action_name == "write_artifact_text":
        return _finish_tool_call_label(f"Compile Artifact to Text File{f' {source}' if source else ''}")
    if action_name == "mkdir":
        return _finish_tool_call_label(action_label if not source else f"{action_label} {source}")
    if action_name == "copy":
        return _finish_tool_call_label(f"Copy {source or 'File'}{f' to {destination}' if destination else ''}")
    if action_name == "move":
        return _finish_tool_call_label(f"Move {source or 'File or Directory'}{f' to {destination}' if destination else ''}")
    if action_name == "delete":
        return _finish_tool_call_label(f"Delete {source or 'File or Directory'}{' Recursively' if arguments.get('recursive') else ''}")
    if action_name == "zip":
        sources = arguments.get("sources")
        count = len(sources) if isinstance(sources, list) else 0
        label = f"Create ZIP{f' {destination}' if destination else ''}"
        return _finish_tool_call_label(label if count == 0 else f"{label} from {count} Item{'s' if count != 1 else ''}")
    if action_name == "unzip":
        return _finish_tool_call_label(f"Extract {source or 'ZIP'}{f' to {destination}' if destination else ''}")
    if action_name == "download":
        return _finish_tool_call_label(action_label if not source else f"{action_label} for {source}")
    return _finish_tool_call_label(action_label)


def build_tool_call_label(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    base_label: str | None = None,
    model_label: str | None = None,
    media_label: str | None = None,
    variant_label: str | None = None,
) -> str:
    """Build the chat-only label shown as soon as a tool call starts."""

    arguments = dict(arguments or {})
    name = str(tool_name or "").strip()
    normalized_name = name[len("wangp_"):] if name.startswith("wangp_") else name
    base = str(base_label or "").strip() or _friendly_tool_label(name)
    model = _short_tool_label_value(model_label) or _humanize_tool_value(arguments.get("model_type"))
    media = _short_tool_label_value(media_label)
    variant = _short_tool_label_value(variant_label)

    model_actions = {
        "get_model": "Get Model Definition of",
        "get_model_metadata": "Get Model Information for",
        "get_model_availability": "Check Model Availability of",
        "get_default_settings": "Get Default Settings of",
        "get_model_schema": "Get Model Schema of",
    }
    if normalized_name == "model" and model:
        action = {"schema": "Get Model Schema of", "definition": "Get Model Definition of", "defaults": "Get Model Defaults of"}.get(str(arguments.get("view", "schema")), "Get Model Information for")
        return _finish_tool_call_label(f"{action} {model}")
    if normalized_name == "models":
        query = _short_tool_label_value(arguments.get("query"))
        filters = _tool_filter_label(arguments.get("filters"))
        label = "Find Models" if not query else f"Find Models for {query}"
        return _finish_tool_call_label(label if not filters else f"{label} ({filters})")
    if normalized_name == "model_settings" and model:
        setting_id = _short_tool_label_value(arguments.get("setting_id"))
        return _finish_tool_call_label(f"Get {setting_id} for {model}" if setting_id else f"List Settings for {model}")
    if normalized_name == "list_loras" and model:
        pattern = _short_tool_label_value(arguments.get("name"))
        return _finish_tool_call_label(f"List LoRAs for {model}" if not pattern else f"List LoRAs for {model} Matching {pattern}")
    if normalized_name in model_actions and model:
        return _finish_tool_call_label(f"{model_actions[normalized_name]} {model}")
    if normalized_name == "get_default_settings":
        target = _humanize_tool_value(arguments.get("tool_id"))
        if target:
            return _finish_tool_call_label(f"Get Default Settings for {target}")
    if normalized_name == "generate":
        label = f"Generate {media or 'Media'}"
        return _finish_tool_call_label(f"{label} Using {model}" if model else label)
    if normalized_name == "toolbox":
        action = _humanize_tool_value(arguments.get("action"))
        return _finish_tool_call_label("List Toolbox Content" if not action else f"Use Toolbox {action}")
    if normalized_name == "io":
        return build_io_tool_call_label(arguments.get("action"), arguments.get("arguments") if "arguments" in arguments else None)
    if normalized_name == "notify":
        title = _short_tool_label_value(arguments.get("title"))
        return _finish_tool_call_label("Send Notification" if not title or title == "Deepy notification" else f"Send Notification: {title}")
    if normalized_name == "search_models":
        query = _short_tool_label_value(arguments.get("query"))
        return _finish_tool_call_label("Search Models" if not query else f'Search Models for “{query}”')
    if normalized_name in {"list_models", "list_model_defs", "list_model_availability"}:
        labels = {"list_models": "List Models", "list_model_defs": "List Model Definitions", "list_model_availability": "List Model Availability"}
        query = _short_tool_label_value(arguments.get("query") or arguments.get("name") or arguments.get("family") or arguments.get("main_output"))
        return _finish_tool_call_label(labels[normalized_name] if not query else f"{labels[normalized_name]} Matching {query}")
    if normalized_name == "list_gallery":
        kind = _humanize_tool_value(arguments.get("media_type"))
        kind = "Media" if not kind or kind.casefold() == "all" else {"image": "Images", "video": "Videos", "audio": "Audio"}.get(kind.casefold(), kind)
        selected = "Selected " if bool(arguments.get("selected_only", False)) else ""
        return _finish_tool_call_label(f"List {selected}Gallery {kind}")
    if normalized_name == "get_media_settings":
        target = _short_tool_label_value(arguments.get("media_id") or arguments.get("path"))
        return _finish_tool_call_label("Get Media Settings" if not target else f"Get Media Settings for {target}")
    if normalized_name in {"list_files", "query_file"}:
        target = _short_tool_label_value(arguments.get("media_id") or arguments.get("path"))
        action = "List Files" if normalized_name == "list_files" else "Read File Information"
        return _finish_tool_call_label(action if not target else f"{action} for {target}")
    if normalized_name == "list_deepy_templates":
        target = _humanize_tool_value(arguments.get("tool_id"))
        return _finish_tool_call_label("List Deepy Templates" if not target else f"List {target} Templates")
    if normalized_name == "get_deepy_template_settings":
        tool = _humanize_tool_value(arguments.get("tool_id"))
        template = _short_tool_label_value(arguments.get("template"))
        if tool and template.casefold() == "default":
            return _finish_tool_call_label(f"Get Default Template Settings for {tool}")
        if tool:
            return _finish_tool_call_label(f"Get Template Settings for {tool}" if not template else f"Get {template} Template Settings for {tool}")
        return _finish_tool_call_label("Get Deepy Template Settings" if not template else f"Get Deepy Template Settings for {template}")
    if normalized_name in {"postprocess", "postprocessing"}:
        process = _humanize_tool_value(arguments.get("process"))
        return _finish_tool_call_label("List Postprocessing Options" if not process else f"Run {process} Postprocessing")
    if normalized_name in {"get_job", "cancel_job"}:
        job_id = _short_tool_label_value(arguments.get("job_id"))
        action = "Check Generation Job" if normalized_name == "get_job" else "Cancel Generation Job"
        return _finish_tool_call_label(action if not job_id else f"{action} {job_id}")
    if normalized_name == "get_loras":
        target = _humanize_tool_value(arguments.get("tool_id"))
        return _finish_tool_call_label("List LoRAs" if not target else f"List LoRAs for {target}")
    if normalized_name in {"gen_image", "edit_image", "gen_video", "gen_video_with_speech", "gen_song", "gen_speech_from_description", "gen_speech_from_sample"}:
        return _finish_tool_call_label(base if not variant else f"{base} Using {variant}")
    if normalized_name == "create_color_frame":
        width, height = arguments.get("width"), arguments.get("height")
        color = _short_tool_label_value(arguments.get("color"))
        details = f" {width}×{height}" if width is not None and height is not None else ""
        return _finish_tool_call_label(f"Create{details} {color or 'Color'} Frame")
    if normalized_name == "inspect_video":
        source = _short_tool_label_value(arguments.get("media_id"))
        start_time, end_time = arguments.get("start_time_seconds"), arguments.get("end_time_seconds")
        try:
            range_label = f" from {float(start_time):g}s to {float(end_time):g}s"
        except (TypeError, ValueError):
            range_label = ""
        action = "Inspect Mid-Res Video" if bool(arguments.get("mid_res_sampling", False)) else "Inspect Video"
        return _finish_tool_call_label(f"{action}{f' {source}' if source else ''}{range_label}")
    if normalized_name == "inspect_media":
        area = " (Selected Area)" if arguments.get("bbox") is not None else ""
        media_inputs = arguments.get("media_inputs")
        if isinstance(media_inputs, list):
            inputs = media_inputs
        else:
            media_ids = arguments.get("media_ids")
            inputs = [{"media_id": value} for value in media_ids] if isinstance(media_ids, list) else [{"media_id": arguments.get("media_id")}] if arguments.get("media_id") else []
        images, frames, unknown, video_names = 0, 0, 0, []
        for item in inputs:
            source = item.get("media_id") if isinstance(item, dict) else item
            source_basename = os.path.basename(str(source or "").strip().replace("\\", "/"))
            source_name = _short_tool_label_value(source)
            extension = os.path.splitext(source_basename)[1].casefold()
            if extension in _IMAGE_EXTENSIONS:
                images += 1
            elif extension in _VIDEO_EXTENSIONS or isinstance(item, dict) and any(item.get(key) is not None for key in ("frame_no", "time_seconds")):
                frames += 1
                if extension in _VIDEO_EXTENSIONS:
                    video_names.append(source_name)
            else:
                unknown += 1
        if unknown or images + frames == 0:
            count = images + frames + unknown
            label = "Inspect Media" if count == 0 else "Inspect Visual" if count == 1 else f"Inspect {count} Visuals"
            return _finish_tool_call_label(f"{label}{area}")
        if images and frames:
            image_text = "Image" if images == 1 else f"{images} Images"
            frame_text = "Frame" if frames == 1 else f"{frames} Frames"
            return _finish_tool_call_label(f"Inspect {image_text} and {frame_text}{area}")
        if images:
            label = "Inspect Image" if images == 1 else f"Inspect {images} Images"
            return _finish_tool_call_label(f"{label}{area}")
        frame_text = "Frame" if frames == 1 else f"{frames} Frames"
        if len(video_names) == frames and len(set(video_names)) == 1:
            return _finish_tool_call_label(f"Inspect {frame_text} from {video_names[0]}{area}")
        label = f"Inspect {frame_text}" if frames == 1 else f"Inspect {frames} Video Frames"
        return _finish_tool_call_label(f"{label}{area}")
    if normalized_name == "side_by_side":
        media_ids = arguments.get("media_ids")
        count = len(media_ids) if isinstance(media_ids, list) else 0
        label = "Compose Side by Side" if count == 0 else f"Compose {count} Visual{'s' if count != 1 else ''} Side by Side"
        layout = str(arguments.get("layout", "") or "").strip().casefold()
        layout_label = {"horizontal": " Horizontally", "vertical": " Vertically", "grid": " in a Grid"}.get(layout, f" in {layout.upper()}" if layout else "")
        legends = arguments.get("legends")
        legend_label = " with Legends" if isinstance(legends, list) and any(str(legend or "").strip() for legend in legends) else ""
        return _finish_tool_call_label(f"{label}{layout_label}{legend_label}")
    if normalized_name == "add_to_gallery":
        paths = arguments.get("paths")
        sources = paths if isinstance(paths, list) else [arguments.get("path")] if arguments.get("path") else []
        if len(sources) != 1:
            return _finish_tool_call_label("Add Media to Gallery" if not sources else f"Add {len(sources)} Media Items to Gallery")
        return _finish_tool_call_label(f"Add to Gallery for {_short_tool_label_value(sources[0])}")
    if normalized_name == "resize_crop":
        width, height = arguments.get("width"), arguments.get("height")
        cropping = any(arguments.get(key) is not None for key in ("crop_left", "crop_top", "crop_right", "crop_bottom"))
        action = "Resize and Crop Media" if cropping and (width is not None or height is not None) else "Crop Media" if cropping else "Resize Media"
        if width is not None and height is not None:
            action += f" to {width}×{height}"
        elif width is not None:
            action += f" to {width}px Wide"
        elif height is not None:
            action += f" to {height}px High"
        return _finish_tool_call_label(action)
    if normalized_name == "search_doc":
        query = _short_tool_label_value(arguments.get("query"))
        return _finish_tool_call_label("Search Documentation" if not query else f'Search Documentation for “{query}”')
    if normalized_name == "load_doc_section":
        section = _short_tool_label_value(arguments.get("section"))
        return _finish_tool_call_label("Read Documentation Section" if not section else f"Read {section} from Documentation")
    if normalized_name == "get_selected_media":
        kind = _humanize_tool_value(arguments.get("media_type"))
        return _finish_tool_call_label("Get Selected Media" if not kind or kind.casefold() == "all" else f"Get Selected {kind}")
    if normalized_name == "get_media_details":
        target = _short_tool_label_value(arguments.get("media_id"))
        return _finish_tool_call_label("Get Media Details" if not target else f"Get Media Details for {target}")
    if normalized_name == "resolve_media_reference":
        reference = _humanize_tool_value(arguments.get("reference"))
        kind = _humanize_tool_value(arguments.get("media_type"))
        return _finish_tool_call_label("Resolve Media" if not reference else f"Resolve {reference}{f' {kind}' if kind and kind.casefold() != 'all' else ''}")
    if normalized_name == "mcp_resource":
        server = _short_tool_label_value(arguments.get("server"))
        target = _humanize_tool_value(arguments.get("uri"))
        if not target:
            return _finish_tool_call_label("List MCP Documents" if not server else f"List {server} Documents")
        query = _short_tool_label_value(arguments.get("query"))
        if query:
            return _finish_tool_call_label(f'Search {target} for “{query}”')
        section = _short_tool_label_value(arguments.get("section"))
        return _finish_tool_call_label(f"Read {section} from {target}" if section else f"Read {target}")

    subjects = ("media_id", "path", "reference", "doc_id", "section", "query", "job_id", "server", "uri")
    subject = next((_short_tool_label_value(arguments.get(key)) for key in subjects if _short_tool_label_value(arguments.get(key))), "")
    if not subject:
        ignored_keys = {"prompt", "question", "content", "text", "source", "arguments", "parameters", "extra_settings", "limit", "offset", "wait", "timeout_s", "event_limit"}
        subject = next((_short_tool_label_value(value) for key, value in arguments.items() if key not in ignored_keys and not isinstance(value, bool) and _short_tool_label_value(value)), "")
    return _finish_tool_call_label(base if not subject else f"{base} for {subject}")


def _find_message(session, message_id: str) -> dict[str, Any] | None:
    target_id = str(message_id or "")
    for record in session.chat_transcript:
        if record.get("id") == target_id:
            return record
    return None


def resolve_message_id(session, message_reference: str) -> str:
    reference = str(message_reference or "").strip()
    if len(reference) == 0:
        return ""
    record = _find_message(session, reference)
    if record is None:
        record = next((item for item in session.chat_transcript if str(item.get("client_submission_id", "") or "").strip() == reference), None)
    return "" if record is None else str(record.get("id", "") or "").strip()


def _ensure_message_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = record.get("blocks", None)
    if isinstance(blocks, list):
        return blocks
    blocks = []
    content = str(record.get("content", "") or "").strip()
    if len(content) > 0:
        blocks.append({"id": _next_block_id("content"), "type": "markdown", "text": content})
    for reasoning_block in record.get("reasoning", []) or []:
        if isinstance(reasoning_block, dict):
            reasoning_id = str(reasoning_block.get("id", "")).strip() or _next_block_id("reasoning")
            reasoning_text = str(reasoning_block.get("text", "")).strip()
        else:
            reasoning_id = _next_block_id("reasoning")
            reasoning_text = str(reasoning_block or "").strip()
        if len(reasoning_text) > 0:
            blocks.append({"id": reasoning_id, "type": "reasoning", "text": reasoning_text})
    for tool_block in record.get("tools", []) or []:
        if not isinstance(tool_block, dict):
            continue
        migrated_block = dict(tool_block)
        migrated_block["type"] = "tool"
        migrated_block["id"] = str(migrated_block.get("id", "")).strip() or _next_tool_id()
        blocks.append(migrated_block)
    record["blocks"] = blocks
    return blocks


def _time_label() -> str:
    return time.strftime("%H:%M")


def _event_payload(event: dict[str, Any], session=None, revision: int | None = None) -> str:
    payload = dict(event)
    if payload.get("status"):
        payload["status"] = {**payload["status"], "text": _clean_display_filename(str(payload["status"]["text"]))}
    if session is not None:
        payload["chat_session_id"] = str(session.chat_session_id)
        payload["revision"] = int(session.chat_revision if revision is None else revision)
        # Status/stats can be coalesced away and are not transcript mutations.
        if event["type"] not in {"status", "stats", "session_catalog", "session_resume_ready"}:
            session.chat_event_sequence = int(getattr(session, "chat_event_sequence", 0) or 0) + 1
            payload["sequence"] = session.chat_event_sequence
            payload["sequence_start"] = session.chat_event_sequence
        if event["type"] in {"sync", "status", "reset"}:
            turn = getattr(session, "current_turn", None)
            durations = getattr(session, "chat_turn_durations", {})
            if event["type"] == "status" and durations:
                latest = next(reversed(durations))
                durations = {latest: durations[latest]}
            payload["presentation"] = {"active_message_id": turn.get("assistant_message_id", "") if turn else "", "active_user_id": turn.get("user_message_id", "") if turn else "", "durations": durations}
            if turn and "presentation_started_at" in turn:
                elapsed = turn.get("presentation_pause_started_at", time.monotonic()) - turn["presentation_started_at"] - turn["presentation_paused_seconds"]
                payload["presentation"].update(elapsed_seconds=max(0.0, elapsed), paused="presentation_pause_started_at" in turn)
    return json.dumps({"event_id": uuid.uuid4().hex, "instance_id": SERVER_INSTANCE_ID, "event": payload}, ensure_ascii=False)


def _markdown_to_html(text: str) -> str:
    text = str(text or "").strip()
    if len(text) == 0:
        return ""
    text = html.escape(text, quote=False)
    rendered = markdown.markdown(text, extensions=_MARKDOWN_EXTENSIONS, output_format="html5")
    rendered = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener noreferrer"', rendered)
    rendered = re.sub(r'<a href="(/wangp_api/gallery/media/[^"]+)"', r'<a href="\1" target="_blank" rel="noopener noreferrer"', rendered)
    rendered = re.sub(r'<a href="(/wangp_api/download/[a-f0-9]+)"', r'<a href="\1" download', rendered)
    # Add wrap opportunities to visible prose, never HTML attributes or code.
    return re.sub(r"(<pre\b[^>]*>.*?</pre>|<code\b[^>]*>.*?</code>|<[^>]*>)|_", lambda match: match[1] or "_<wbr>", rendered, flags=re.DOTALL)


def _authorized_download_path(value: Any, file_access_policy) -> str | None:
    if file_access_policy is None:
        return None
    try:
        path_text = str(value or "").strip().strip("\"'")
        if os.name == "nt" and path_text.rstrip(" .") != path_text:
            return None
        candidate = Path(path_text).expanduser()
        if candidate.is_absolute() and file_access_policy.virtualized:
            return None
        path = file_access_policy.resolve_path(path_text)
        return str(path) if file_access_policy.can_read(path) and path.is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


def _existing_download_path(value: Any) -> str | None:
    try:
        path = os.path.abspath(os.path.expanduser(str(value or "").strip()))
        return path if os.path.isfile(path) else None
    except (OSError, RuntimeError, ValueError):
        return None


def _tool_result_paths(value: Any, key: str = ""):
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _tool_result_paths(child_value, str(child_key or "").strip().casefold())
    elif isinstance(value, (list, tuple)):
        for child_value in value:
            yield from _tool_result_paths(child_value, key)
    elif (key in _TOOL_RESULT_PATH_KEYS or key.endswith(("_path", "_paths", "_file", "_files"))) and isinstance(value, (str, os.PathLike)):
        yield str(value)


def _download_reference_targets(session, record: dict[str, Any], file_access_policy) -> dict[str, str]:
    targets: dict[str, tuple[str, str] | None] = {}

    def add(reference: Any, path: str) -> None:
        text = str(reference or "").strip()
        if not text or "\n" in text:
            return
        key = text.casefold()
        existing = targets.get(key)
        if existing is None and key in targets:
            return
        if existing is not None and os.path.normcase(existing[1]) != os.path.normcase(path):
            targets[key] = None
        else:
            targets[key] = (text, path)

    for media in list(getattr(session, "media_registry", []) or []):
        if not isinstance(media, dict):
            continue
        media_path = str(media.get("path", "") or "").strip()
        path = _existing_download_path(media_path)
        if path is None:
            continue
        add(media.get("media_id"), path)
        media_type = str(media.get("media_type", "") or "").strip().lower()
        if media_type in {"image", "video", "audio"}:
            gallery = "audio" if media_type == "audio" else "visual"
            for media_id in gallery_media_ids(media_path, gallery, media.get("settings")):
                add(media_id, path)
        add(media.get("filename") or os.path.basename(path), path)
        if file_access_policy.can_read(Path(path)):
            add(file_access_policy.virtualize_path(path), path)
            if not file_access_policy.virtualized:
                add(media.get("path"), path)
                add(path, path)
    for media_id, media_path in dict(getattr(session, "gallery_download_registry", {}) or {}).items():
        path = _existing_download_path(media_path)
        if path is not None:
            add(media_id, path)
    for block in _ensure_message_blocks(record):
        if not isinstance(block, dict) or block.get("type") != "tool":
            continue
        for value in _tool_result_paths({"arguments": block.get("arguments"), "result": block.get("result")}):
            path = _authorized_download_path(value, file_access_policy)
            if path is None:
                continue
            add(value, path)
            add(file_access_policy.virtualize_path(path), path)
            if not file_access_policy.virtualized:
                add(path, path)
            if os.path.splitext(os.path.basename(path))[1]:
                add(os.path.basename(path), path)
    return {value[0]: value[1] for value in targets.values() if value is not None}


def _markdown_download_link(label: str, path: str, state: dict[str, Any], *, code: bool = False) -> str | None:
    if state["count"] >= _DOWNLOAD_REFERENCE_LIMIT:
        return None
    reference = label[1:-1].strip() if code else label.strip()
    gallery_media_id = reference.casefold() if _GALLERY_MEDIA_ID_RE.fullmatch(reference) else ""
    path_key = f"gallery:{gallery_media_id}" if gallery_media_id else os.path.normcase(path)
    url = state["urls"].get(path_key)
    if url is None:
        from shared.utils.downloads import register_file_download, register_gallery_download

        url = (register_gallery_download(gallery_media_id, path) if gallery_media_id else register_file_download(path))["url"]
        state["urls"][path_key] = url
    state["count"] += 1
    escaped_label = label if code else re.sub(r"([\\`*{}\[\]()#+\-.!_|>])", r"\\\1", label)
    return f"[{escaped_label}]({url})"


def _rewrite_sandbox_download_link(token: str, file_access_policy) -> str:
    marker = token.rfind("](")
    if marker < 0 or not token.endswith(")"):
        return token
    target = urllib.parse.unquote(token[marker + 2:-1].strip())
    if not target.casefold().startswith("sandbox:"):
        return token
    path = _authorized_download_path(target[len("sandbox:"):], file_access_policy)
    if path is None:
        return token
    from shared.utils.downloads import register_file_download

    return f"{token[:marker + 2]}{register_file_download(path)['url']})"


def _absolute_path_prefix(text: str, start: int, file_access_policy) -> tuple[int, str] | None:
    line_end = text.find("\n", start)
    tail = text[start : min(len(text) if line_end < 0 else line_end, start + 4096)]
    endpoints = {len(tail), *(match.start() for match in re.finditer(r"\s+", tail))}
    for end in sorted(endpoints, reverse=True):
        candidate = tail[:end].rstrip()
        variants, trimmed = [candidate], candidate
        while trimmed and trimmed[-1] in ".,;:!?)]}'\"`*_":
            trimmed = trimmed[:-1].rstrip()
            variants.append(trimmed)
        for variant in variants:
            path = _authorized_download_path(variant, file_access_policy)
            if path is not None:
                return len(variant), path
    return None


def _linkify_absolute_paths(text: str, file_access_policy, state: dict[str, Any]) -> str:
    if file_access_policy is None or not file_access_policy.read_enabled or state["count"] >= _DOWNLOAD_REFERENCE_LIMIT:
        return text
    rendered, cursor = [], 0
    while state["count"] < _DOWNLOAD_REFERENCE_LIMIT:
        match = _ABSOLUTE_PATH_START_RE.search(text, cursor)
        if match is None:
            break
        resolved = _absolute_path_prefix(text, match.start(), file_access_policy)
        if resolved is None:
            rendered.append(text[cursor : match.end()])
            cursor = match.end()
            continue
        length, path = resolved
        label = text[match.start() : match.start() + length]
        link = _markdown_download_link(label, path, state)
        if link is None:
            break
        rendered.extend((text[cursor : match.start()], link))
        cursor = match.start() + length
    rendered.append(text[cursor:])
    return "".join(rendered)


def _linkify_plain_download_references(text: str, references: dict[str, str], file_access_policy, state: dict[str, Any]) -> str:
    if references and state["count"] < _DOWNLOAD_REFERENCE_LIMIT:
        lookup = {reference.casefold(): path for reference, path in references.items()}
        alternatives = "|".join(re.escape(reference) for reference in sorted(references, key=len, reverse=True))
        pattern = re.compile(rf"(?<![\w\\/])({alternatives})(?![\w\\/])", flags=re.IGNORECASE)

        def replace(match: re.Match[str]) -> str:
            path = lookup[match.group(1).casefold()]
            return _markdown_download_link(match.group(1), path, state) or match.group(0)

        text = pattern.sub(replace, text)
    rendered, cursor = [], 0
    for match in _DOWNLOAD_LINK_RE.finditer(text):
        rendered.append(_linkify_absolute_paths(text[cursor : match.start()], file_access_policy, state))
        rendered.append(match.group(0))
        cursor = match.end()
    rendered.append(_linkify_absolute_paths(text[cursor:], file_access_policy, state))
    return "".join(rendered)


def _linkify_download_markdown(text: str, references: dict[str, str], file_access_policy, state: dict[str, Any] | None = None) -> str:
    lookup = {reference.casefold(): path for reference, path in references.items()}
    state = {"count": 0, "urls": {}} if state is None else state
    rendered, cursor = [], 0
    for match in _DOWNLOAD_MARKDOWN_TOKEN_RE.finditer(text):
        rendered.append(_linkify_plain_download_references(text[cursor : match.start()], references, file_access_policy, state))
        token = match.group(0)
        if match.lastgroup == "code" and state["count"] < _DOWNLOAD_REFERENCE_LIMIT:
            value = token[1:-1].strip()
            path = lookup.get(value.casefold()) or _authorized_download_path(value, file_access_policy)
            token = _markdown_download_link(token, path, state, code=True) if path is not None else token
        elif match.lastgroup == "link":
            token = _rewrite_sandbox_download_link(token, file_access_policy)
        rendered.append(token)
        cursor = match.end()
    rendered.append(_linkify_plain_download_references(text[cursor:], references, file_access_policy, state))
    return "".join(rendered)


def linkify_message_download_references(session, message_id: str, file_access_policy) -> str | None:
    if file_access_policy is None:
        return None
    record = _find_message(session, message_id)
    if record is None or str(record.get("role", "")).strip().lower() != "assistant":
        return None
    references = _download_reference_targets(session, record, file_access_policy)
    changed, state = False, {"count": 0, "urls": {}}
    for block in _ensure_message_blocks(record):
        if not isinstance(block, dict) or block.get("type") != "markdown":
            continue
        original = str(block.get("text", "") or "")
        linked = _linkify_download_markdown(original, references, file_access_policy, state)
        if linked != original:
            block["text"] = linked
            changed = True
    if not changed:
        return None
    record.setdefault("download_links", {}).update({url: path for path, url in state["urls"].items() if not path.startswith("gallery:")})
    revision = _touch_chat(session)
    return _message_upsert_event(session, record, revision)


def _structured_tool_presentation(tool_record: dict[str, Any], file_access_policy) -> dict[str, Any] | None:
    result = tool_record.get("result")
    if not isinstance(result, dict):
        return None
    rows = result.get("entries") if isinstance(result.get("entries"), list) else result.get("items") if isinstance(result.get("items"), list) else None
    if rows is None and str(tool_record.get("name", "")) == "wangp_artifact" and isinstance(result.get("data"), dict):
        rows = [{"field": key, "value": value} for key, value in result["data"].items()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        return None
    preferred = ["index", "name", "title", "path", "type", "duration_seconds", "size_bytes", "modified", "outline", "prompt", "content", "field", "value"]
    available = []
    for key in preferred:
        if any(key in row for row in rows):
            available.append(key)
    for row in rows:
        for key in row:
            if key not in available:
                available.append(key)
    columns = available[:8]
    rendered_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value if value is not None else "")
            href = ""
            if column == "path" and file_access_policy is not None:
                path = _authorized_download_path(value, file_access_policy)
                if path is not None:
                    from shared.utils.downloads import register_file_download
                    href = register_file_download(path)["url"]
            cells.append({"text": text, "href": href})
        rendered_rows.append(cells)
    total = result.get("matched")
    offset = int(result.get("offset", 0) or 0)
    end = offset + len(rows)
    summary = f"{offset + 1}-{end} of {int(total)}" if total is not None else f"{offset + 1}-{end}" if offset else f"{len(rows)} item{'s' if len(rows) != 1 else ''}"
    if result.get("has_more"):
        summary += f"; next offset {result.get('next_offset')}"
    return {"type": "records", "title": str(tool_record.get("label", "") or "Structured result"), "summary": summary, "columns": columns, "rows": rendered_rows}


def _render_structured_tool_presentation(presentation: dict[str, Any] | None) -> str:
    if not isinstance(presentation, dict) or presentation.get("type") != "records":
        return ""
    columns, rows = list(presentation.get("columns", []) or []), list(presentation.get("rows", []) or [])
    if not columns or not rows:
        return ""
    header = "".join(f"<th>{html.escape(str(column).replace('_', ' ').title())}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for cell in list(row or []):
            text = html.escape(str(cell.get("text", "")))
            href = str(cell.get("href", "") or "").strip()
            content = f"<a href='{html.escape(href)}' download>{text}</a>" if href else text
            cells.append(f"<td>{content}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<section class='chat__structured-result'>"
        f"<div class='chat__structured-result-header'><strong>{html.escape(str(presentation.get('title', 'Structured result')))}</strong><span>{html.escape(str(presentation.get('summary', '')))}</span></div>"
        f"<div class='chat__structured-result-scroll'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        "</section>"
    )


def _plain_text_to_html(text: str) -> str:
    escaped = html.escape(str(text or "").strip(), quote=False)
    return "" if not escaped else f"<p>{escaped.replace(chr(10), '<br>')}</p>"


def _extract_attachments_from_markdown(text: str) -> tuple[str, list[dict[str, Any]]]:
    attachments = []

    def replace_match(match: re.Match[str]) -> str:
        attachment = _attachment_from_path(match.group("path"), match.group("alt"))
        if attachment is not None:
            attachments.append(attachment)
        return ""

    stripped = _MARKDOWN_IMAGE_RE.sub(replace_match, str(text or ""))
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped, attachments


def _attachments_from_tool_result(result: dict[str, Any] | None, file_access_policy=None) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    download = result.get("download")
    if isinstance(download, dict) and str(download.get("url", "")).strip():
        filename = str(download.get("filename", "Download file") or "Download file").strip()
        size_bytes = download.get("size_bytes")
        subtitle = f"{int(size_bytes):,} bytes" if isinstance(size_bytes, (int, float)) else ""
        url = str(download["url"]).strip()
        kind = _attachment_kind(filename, "download")
        source_path = _physical_attachment_path(result.get("output_file") or result.get("path"), file_access_policy)
        preview = _attachment_from_path(source_path) if source_path and kind in {"image", "video", "audio"} else None
        bundled_thumbnail = {"audio": _AUDIO_THUMBNAIL_PATH, "archive": _ARCHIVE_THUMBNAIL_PATH}.get(kind)
        thumb_url = str(preview.get("thumb_url", "")) if preview is not None else (_bundled_thumbnail_url(bundled_thumbnail) if bundled_thumbnail else "")
        return [{"href": url, "label": f"Download {filename}", "subtitle": subtitle, "thumb_url": thumb_url, "kind": kind, "path_key": url, "path": source_path, "download": filename}]

    output_paths = []

    def collect(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        for key in ("generated_files", "output_files"):
            values = payload.get(key)
            if isinstance(values, (list, tuple)):
                output_paths.extend(str(value).strip() for value in values if str(value).strip())
        nested_result = payload.get("result")
        if isinstance(nested_result, dict):
            collect(nested_result)
        output_file = str(payload.get("output_file", "") or "").strip()
        if output_file:
            output_paths.append(output_file)

    collect(result)
    attachments = []
    seen_paths = set()
    for output_path in output_paths:
        output_file = _physical_attachment_path(output_path, file_access_policy)
        path_key = os.path.normcase(os.path.normpath(output_file))
        if not output_file or path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        ext = os.path.splitext(output_file)[1].lower()
        label = "Generated image" if ext in _IMAGE_EXTENSIONS else ("Generated video" if ext in _VIDEO_EXTENSIONS else ("Generated audio" if ext in _AUDIO_EXTENSIONS else "Generated file"))
        attachment = _attachment_from_path(output_file, label)
        if attachment is not None:
            attachments.append(attachment)
    return attachments


def _attachment_from_tool_result(result: dict[str, Any] | None, file_access_policy=None) -> dict[str, Any] | None:
    attachments = _attachments_from_tool_result(result, file_access_policy)
    return attachments[-1] if attachments else None


def _physical_attachment_path(value: Any, file_access_policy=None) -> str:
    path = str(value or "").strip()
    if not path or file_access_policy is None:
        return path
    try:
        resolved = file_access_policy.resolve_path(path)
        return str(resolved) if resolved.is_file() else ""
    except (OSError, PermissionError, ValueError):
        return path


def _attachment_from_path(path: str, label: str | None = None) -> dict[str, Any] | None:
    clean_path = str(path or "").strip()
    if len(clean_path) == 0:
        return None
    normalized_path = clean_path
    if normalized_path.startswith("/gradio_api/file="):
        normalized_path = normalized_path.split("=", 1)[1]
    normalized_path = urllib.parse.unquote(normalized_path).replace("\\", "/")
    normalized_path = os.path.normpath(normalized_path).replace("\\", "/")
    path_key = normalized_path.lower()
    from shared.utils.downloads import register_file_download
    href = register_file_download(normalized_path)["url"]
    ext = os.path.splitext(normalized_path)[1].lower()
    resolved_label = str(label or os.path.basename(normalized_path) or "Open file").strip()
    subtitle = os.path.basename(normalized_path)
    if resolved_label == subtitle:
        subtitle = ""
    thumb_url = ""
    kind = _attachment_kind(normalized_path)
    if ext in _IMAGE_EXTENSIONS:
        kind = "image"
        thumb_url = href + '/thumbnail'
    elif ext in _VIDEO_EXTENSIONS:
        kind = "video"
        thumb_url = href + '/thumbnail'
    elif ext in _AUDIO_EXTENSIONS:
        kind = "audio"
        thumb_url = _audio_thumbnail_url()
    return {
        "path_key": path_key,
        "path": normalized_path,
        "href": href,
        "label": resolved_label,
        "subtitle": subtitle,
        "kind": kind,
        "thumb_url": thumb_url,
    }


def _attachment_kind(path: str, default: str = "file") -> str:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in _ARCHIVE_EXTENSIONS:
        return "archive"
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    if ext in _AUDIO_EXTENSIONS:
        return "audio"
    return "file" if ext else default


def _audio_thumbnail_url() -> str:
    return _bundled_thumbnail_url(_AUDIO_THUMBNAIL_PATH)


def _bundled_thumbnail_url(thumbnail_path: str) -> str:
    if not os.path.isfile(thumbnail_path):
        return ""
    path = os.path.normpath(thumbnail_path).replace("\\", "/")
    from shared.utils.downloads import register_file_download
    return register_file_download(path)["url"]


def _attachment_icon(kind: str) -> str:
    icons = {
        "archive": "<rect x='4' y='6' width='40' height='10' rx='2'></rect><path d='M8 16v22a4 4 0 0 0 4 4h24a4 4 0 0 0 4-4V16'></path><path d='M19 25h10'></path>",
        "audio": "<path d='M18 36V13l19-4v23'></path><path d='M18 19l19-4'></path><ellipse cx='13' cy='37' rx='5' ry='4'></ellipse><ellipse cx='32' cy='33' rx='5' ry='4'></ellipse>",
        "download": "<path d='M24 6v24'></path><path d='m15 22 9 9 9-9'></path><path d='M10 39h28'></path>",
        "image": "<rect x='6' y='8' width='36' height='32' rx='4'></rect><circle cx='17' cy='19' r='3'></circle><path d='m10 35 9-9 6 6 5-5 8 8'></path>",
        "video": "<rect x='6' y='10' width='36' height='28' rx='4'></rect><path d='m20 18 11 6-11 6z'></path>",
        "file": "<path d='M13 5h15l7 7v31H13z'></path><path d='M28 5v8h7M19 23h10M19 29h10M19 35h7'></path>",
    }
    icon_kind = kind if kind in icons else "file"
    return f"<div class='chat__attachment-thumb chat__attachment-thumb--icon chat__attachment-thumb--{icon_kind}' data-attachment-icon='{icon_kind}'><svg viewBox='0 0 48 48' aria-hidden='true' focusable='false'>{icons[icon_kind]}</svg></div>"


def _render_copy_button(source: str, label: str, text: str | None = None) -> str:
    copy_text = "" if text is None else f" data-copy-text='{html.escape(str(text), quote=True)}'"
    return (
        f"<button type='button' class='chat__copy-button' data-copy-source='{html.escape(source, quote=True)}'{copy_text} aria-label='{html.escape(label, quote=True)}' title='{html.escape(label, quote=True)}'>"
        "<svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><rect x='5' y='5' width='8' height='8' rx='1.5'></rect><path d='M3.5 10.5H3A1.5 1.5 0 0 1 1.5 9V3A1.5 1.5 0 0 1 3 1.5h6A1.5 1.5 0 0 1 10.5 3v.5'></path></svg>"
        "</button>"
    )


def _render_queued_request_actions() -> str:
    steer_label = html.escape("Steer with this queued request", quote=True)
    edit_label = html.escape("Edit queued request", quote=True)
    remove_label = html.escape("Remove queued request", quote=True)
    return (
        f"<button type='button' class='chat__message-action-button' data-message-action='steer' aria-label='{steer_label}' title='{steer_label}'>"
        "<svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M2.5 8h9M8.5 4l4 4-4 4'></path></svg>"
        "</button>"
        f"<button type='button' class='chat__message-action-button' data-message-action='edit' aria-label='{edit_label}' title='{edit_label}'>"
        "<svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M3 11.8 3.5 9l6.8-6.8a1.4 1.4 0 0 1 2 0l1.5 1.5a1.4 1.4 0 0 1 0 2L7 12.5l-2.8.5Z'></path><path d='m9.4 3.1 3.5 3.5'></path></svg>"
        "</button>"
        f"<button type='button' class='chat__message-action-button' data-message-action='remove' aria-label='{remove_label}' title='{remove_label}'>"
        "<svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M3 4.5h10M6 4.5V2.7h4v1.8M4.7 4.5l.6 8.3h5.4l.6-8.3M7 7v3.4M9 7v3.4'></path></svg>"
        "</button>"
    )


def _render_collapse_button(label: str) -> str:
    escaped_label = html.escape(f"Collapse {label}", quote=True)
    return f"<button type='button' class='chat__collapse-button' data-disclosure-action='collapse' aria-label='{escaped_label}' title='{escaped_label}'><span aria-hidden='true'>▾</span></button>"


def _compose_message_payload(record: dict[str, Any], body_html: str, copy_text: str) -> dict[str, Any]:
    role = str(record.get("role", "assistant"))
    badge_text = str(record.get("badge", "")).strip()
    end_badge_text = str(record.get("end_badge", "")).strip()
    badge_html = "" if len(badge_text) == 0 else f"<span class='chat__badge'>{html.escape(badge_text)}</span>"
    copy_button_html = _render_copy_button("user", "Copy request", copy_text) if role == "user" else ""
    queued_actions_html = _render_queued_request_actions() if role == "user" and badge_text == "Queued" else ""
    actions_html = f"<div class='chat__message-actions'>{copy_button_html}{queued_actions_html}</div>" if role == "user" else ""
    end_badge_html = "" if len(end_badge_text) == 0 else f"<div class='chat__message-end'><span class='chat__message-end-badge'>{html.escape(end_badge_text)}</span></div>"
    card_html = (
        f"<article class='chat__message chat__message--{html.escape(role)}' data-message-id='{html.escape(str(record.get('id', '')))}'>"
        f"<div class='chat__avatar'>{html.escape('You' if role == 'user' else 'Deepy')}</div>"
        f"<div class='chat__message-card'>"
        f"<div class='chat__meta'>"
        f"<div class='chat__meta-left'>{badge_html}</div>"
        f"<div class='chat__meta-right'>{actions_html}<div class='chat__time'>{html.escape(str(record.get('created_at', '')))}</div></div>"
        f"</div>"
        f"<div class='chat__body'>{body_html}</div>"
        f"{end_badge_html}"
        f"</div>"
        f"</article>"
    )
    payload = {"id": record.get("id", ""), "role": role, "html": card_html, "badge": badge_text, "queued": bool(record.get("queued", False))}
    client_submission_id = str(record.get("client_submission_id", "") or "").strip()
    if client_submission_id:
        payload["client_submission_id"] = client_submission_id
    return payload


def _render_message_payload(record: dict[str, Any]) -> dict[str, Any]:
    blocks_html, rendered_attachment_keys = _render_message_blocks(record)
    attachments_html = _render_attachments(
        [
            attachment
            for attachment in list(record.get("attachments", []))
            if isinstance(attachment, dict) and (attachment.get("path_key", "") or attachment.get("href", "")) not in rendered_attachment_keys
        ]
    )
    copy_text = "\n\n".join(str(block.get("text", "")).strip() for block in _ensure_message_blocks(record) if isinstance(block, dict) and block.get("type") == "markdown" and len(str(block.get("text", "")).strip()) > 0)
    return _compose_message_payload(record, f"{blocks_html}{attachments_html}", copy_text)


def _render_message_blocks(record: dict[str, Any]) -> tuple[str, set[str]]:
    blocks = _ensure_message_blocks(record)
    if len(blocks) == 0:
        return "", set()
    rendered = []
    rendered_attachment_keys = set()
    reasoning_total = sum(1 for block in blocks if isinstance(block, dict) and block.get("type") == "reasoning" and len(str(block.get("text", "")).strip()) > 0)
    reasoning_no = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "markdown")).strip().lower() or "markdown"
        if block_type == "markdown":
            rendered.append(_render_markdown_block(record, block, rendered_attachment_keys, streaming=bool(block.get("streaming", False))))
            continue
        if block_type == "reasoning":
            reasoning_text = str(block.get("text", "")).strip()
            if len(reasoning_text) == 0:
                continue
            reasoning_no += 1
            rendered.append(_render_reasoning_block(block, reasoning_no, reasoning_total, streaming=bool(block.get("streaming", False))))
            continue
        if block_type in {"context_summary", "context_thinking"}:
            summary_text = str(block.get("text", "")).strip()
            if len(summary_text) > 0:
                rendered.append(_render_context_summary_block(block))
            continue
        if block_type == "tool":
            attachments = block.get("attachments") if isinstance(block.get("attachments"), list) else [block.get("attachment")] if isinstance(block.get("attachment"), dict) else []
            attachment_html = _render_attachments(_dedupe_attachments(attachments, rendered_attachment_keys))
            rendered.append(_render_tool_block(block, attachment_html))
    return "".join(rendered), rendered_attachment_keys


def _render_markdown_block(record: dict[str, Any], block: dict[str, Any], rendered_attachment_keys: set[str], *, streaming: bool) -> str:
    block_id = html.escape(str(block.get("id", "")), quote=True)
    if streaming:
        content_html = f"<div class='chat__stream-text'>{html.escape(str(block.get('text', '')))}</div>"
    else:
        content_source, attachments = _extract_attachments_from_markdown(block.get("text", ""))
        content_html = _plain_text_to_html(content_source) if str(record.get("role", "")).strip().lower() == "user" else _markdown_to_html(content_source)
        content_html += _render_attachments(_dedupe_attachments(attachments, rendered_attachment_keys))
    return f"<div class='chat__content-block' data-block-id='{block_id}' data-block-type='markdown'>{content_html}</div>"


def _render_block_html(record: dict[str, Any], block: dict[str, Any], *, streaming: bool) -> str:
    rendered_attachment_keys: set[str] = set()
    for prior in _ensure_message_blocks(record):
        if prior is block:
            break
        if not isinstance(prior, dict):
            continue
        if prior.get("type") == "markdown":
            _source, attachments = _extract_attachments_from_markdown(prior.get("text", ""))
            _dedupe_attachments(attachments, rendered_attachment_keys)
        elif prior.get("type") == "tool":
            attachments = prior.get("attachments") if isinstance(prior.get("attachments"), list) else [prior.get("attachment")] if isinstance(prior.get("attachment"), dict) else []
            _dedupe_attachments(attachments, rendered_attachment_keys)
    block_type = str(block.get("type", "markdown"))
    if block_type == "markdown":
        return _render_markdown_block(record, block, rendered_attachment_keys, streaming=streaming)
    if block_type == "reasoning":
        return _render_reasoning_block(block, 1, 1, streaming=streaming)
    if block_type in {"context_summary", "context_thinking"}:
        return _render_context_summary_block(block, streaming=streaming)
    if block_type == "tool":
        attachments = block.get("attachments") if isinstance(block.get("attachments"), list) else [block.get("attachment")] if isinstance(block.get("attachment"), dict) else []
        return _render_tool_block(block, _render_attachments(_dedupe_attachments(attachments, rendered_attachment_keys)))
    raise ValueError(f"Unsupported assistant chat block type: {block_type}")


def _dedupe_attachments(attachments: list[dict[str, Any]], rendered_attachment_keys: set[str]) -> list[dict[str, Any]]:
    unique = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        dedupe_key = attachment.get("path_key", "") or attachment.get("href", "")
        if len(dedupe_key) == 0 or dedupe_key in rendered_attachment_keys:
            continue
        rendered_attachment_keys.add(dedupe_key)
        unique.append(attachment)
    return unique


def _render_reasoning_block(block: dict[str, Any], block_no: int, total_blocks: int, streaming: bool = False) -> str:
    label = "Thought process"
    block_id = html.escape(str(block.get("id", "")), quote=True)
    content_html = f"<div class='chat__stream-text'>{html.escape(str(block.get('text', '')))}</div>" if streaming else _markdown_to_html(block.get("text", ""))
    return (
        f"<details class='chat__disclosure chat__disclosure--reasoning' data-block-id='{block_id}' data-block-type='reasoning' data-reasoning-id='{block_id}'>"
        f"<summary><span class='chat__tool-title'><span class='chat__tool-chip'>Thought</span>{html.escape(label)}</span></summary>"
        f"<div class='chat__disclosure-body'><div class='chat__reasoning-block'>{content_html}</div>{_render_collapse_button('thought')}</div>"
        "</details>"
    )


def _render_context_summary_block(block: dict[str, Any], streaming: bool | None = None) -> str:
    streaming = bool(block.get("streaming", False)) if streaming is None else bool(streaming)
    block_id = html.escape(str(block.get("id", "")), quote=True)
    thinking = block["type"] == "context_thinking"
    label = "Compaction thoughts" if thinking else "Summarizing earlier history…" if streaming else "Earlier history summarized"
    content_html = f"<div class='chat__stream-text'>{html.escape(str(block.get('text', '')))}</div>" if streaming else _markdown_to_html(block.get("text", ""))
    return (
        f"<details class='chat__disclosure chat__disclosure--context-summary' data-block-id='{block_id}' data-block-type='{block['type']}' data-context-summary-id='{block_id}'>"
        f"<summary><span class='chat__tool-title'><span class='chat__tool-chip'>{'Thought' if thinking else 'Context'}</span>{label}</span></summary>"
        f"<div class='chat__disclosure-body'><div class='chat__context-summary'>{content_html}</div>{_render_collapse_button('summary')}</div>"
        "</details>"
    )


def _render_tool_block(tool_record: dict[str, Any], attachment_html: str = "") -> str:
    name = str(tool_record.get("name", "tool")).strip() or "tool"
    label = _clean_display_filename(str(tool_record.get("label", "")).strip()) or _friendly_tool_label(name)
    status = str(tool_record.get("status", "running")).strip().lower()
    status_label = str(tool_record.get("status_text", "")).strip() or {"running": "Running", "done": "Done", "error": "Error"}.get(status, status.title() or "Running")
    status_class = {"running": "running", "done": "done", "error": "error"}.get(status, "running")
    label_html = html.escape(label).replace("_", "_<wbr>")
    status_html = html.escape(_clean_display_filename(status_label)).replace("_", "_<wbr>")
    request_pending = bool(tool_record.get("request_pending", False))
    arguments_text = html.escape(json.dumps(tool_record.get("arguments", {}), ensure_ascii=False, indent=2, sort_keys=True))
    result_payload = tool_record.get("result", {})
    result_text = html.escape(json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True)) if result_payload is not None else ""
    arguments_copy_button = _render_copy_button("json", f"Copy {label} arguments")
    result_copy_button = "" if result_payload is None else _render_copy_button("json", "Copy result")
    presentation_html = _render_structured_tool_presentation(tool_record.get("presentation"))
    pending_body = "<div class='chat__disclosure-body'><div class='chat__tool-json chat__tool-pending'>Deepy is preparing the tool request.</div></div>"
    completed_body = (
        "<div class='chat__disclosure-body'>"
        "<div class='chat__tool-grid'>"
        f"<div class='chat__tool-json'><div class='chat__tool-section-header'><div class='chat__tool-section-title'>{html.escape(label)} Arguments</div>{arguments_copy_button}</div><pre class='chat__pre'>{arguments_text}</pre></div>"
        f"<div class='chat__tool-json'><div class='chat__tool-section-header'><div class='chat__tool-section-title'>Result</div>{result_copy_button}</div><pre class='chat__pre'>{result_text or html.escape('Pending...')}</pre></div>"
        "</div>"
        f"{presentation_html}"
        f"{_render_collapse_button('tool')}"
        "</div>"
    )
    details = (
        f"<details class='chat__disclosure chat__disclosure--tool' data-tool-id='{html.escape(str(tool_record.get('id', '')))}'>"
        f"<summary><span class='chat__tool-title'><span class='chat__tool-chip'>Tool</span><span>{label_html}</span></span><span class='chat__tool-status chat__tool-status--{status_class}'>{status_html}</span></summary>"
        f"{pending_body if request_pending else completed_body}"
        "</details>"
    )
    block_id = html.escape(str(tool_record.get("id", "")), quote=True)
    return f"<div class='chat__tool-block' data-block-id='{block_id}' data-block-type='tool'>{details}{attachment_html}</div>"


def _render_attachments(attachments: list[dict[str, Any]]) -> str:
    if len(attachments) == 0:
        return ""
    cards = []
    for attachment in attachments:
        href = str(attachment.get("href", "")).strip()
        if len(href) == 0:
            continue
        label = html.escape(str(attachment.get("label", "Open file")))
        subtitle = html.escape(str(attachment.get("subtitle", "")))
        thumb_url = str(attachment.get("thumb_url", "")).strip()
        subtitle_html = f"<span class='chat__attachment-subtitle'>{subtitle}</span>" if len(subtitle) > 0 else ""
        thumb_html = (
            f"<img class='chat__attachment-thumb' loading='lazy' src='{html.escape(thumb_url)}' alt='{label}'>"
            if len(thumb_url) > 0
            else _attachment_icon(str(attachment.get("kind", "file")).strip().lower())
        )
        download_name = str(attachment.get("download", "")).strip()
        link_attributes = f" download='{html.escape(download_name)}'" if download_name else " target='_blank' rel='noopener'"
        cards.append(
            f"<a class='chat__attachment' data-media-kind='{html.escape(str(attachment.get('kind', 'file')))}' href='{html.escape(href)}'{link_attributes}>"
            f"{thumb_html}"
            "<span class='chat__attachment-meta'>"
            f"<span class='chat__attachment-title'>{label}</span>"
            f"{subtitle_html}"
            "</span>"
            "</a>"
        )
    if len(cards) == 0:
        return ""
    return f"<div class='chat__attachments'>{''.join(cards)}</div>"
