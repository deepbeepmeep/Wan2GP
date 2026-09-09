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
_MARKDOWN_EXTENSIONS = ["extra", "nl2br", "sane_lists", "fenced_code", "tables"]
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
    return r"""
#assistant_chat_dock {
    --dock-gap: 14px;
    --dock-launcher-width: 41px;
    --dock-panel-width: 548px;
    --dock-settings-panel-width: 660px;
    --dock-settings-panel-offset: 44px;
    --dock-font-scale: 0.9;
    --chat-history-min-height: 112px;
    --chat-request-max-height: 320px;
    position: fixed !important;
    top: 50%;
    left: 0;
    z-index: 1500;
    width: var(--dock-launcher-width);
    transform: translateY(-50%);
    pointer-events: none;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}

#assistant_chat_dock:not(:has(#assistant_chat_toggle)) {
    display: none !important;
}

#assistant_chat_dock > * {
    flex: 0 0 auto !important;
}

#assistant_chat_launcher_host,
#assistant_chat_panel,
#assistant_chat_settings_launcher_host,
#assistant_chat_settings_panel {
    pointer-events: auto;
}

#assistant_chat_launcher_host {
    flex: 0 0 var(--dock-launcher-width) !important;
    position: relative;
    width: var(--dock-launcher-width) !important;
    min-width: var(--dock-launcher-width) !important;
    max-width: var(--dock-launcher-width) !important;
    min-height: 188px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
    min-width: 0 !important;
}

#assistant_chat_launcher_host .html-container,
#assistant_chat_shell_block .html-container,
#assistant_chat_stats_block .html-container {
    padding: 0 !important;
}

#assistant_chat_launcher_host .prose,
#assistant_chat_shell_block .prose,
#assistant_chat_stats_block .prose {
    max-width: none !important;
    margin: 0 !important;
}

#assistant_chat_toggle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: var(--dock-launcher-width);
    min-width: var(--dock-launcher-width);
    min-height: 188px;
    padding: 18px 6px;
    border: 1px solid rgba(73, 87, 99, 0.18);
    border-left: 0;
    border-radius: 0 22px 22px 0;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(239, 242, 246, 0.98) 100%);
    box-shadow: 0 18px 34px rgba(8, 33, 49, 0.16);
    cursor: pointer;
    transform: translateX(-4px);
    transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}

#assistant_chat_toggle:hover {
    transform: translateX(0);
    box-shadow: 0 22px 38px rgba(8, 33, 49, 0.2);
}

#assistant_chat_dock.is-open #assistant_chat_toggle {
    background: linear-gradient(180deg, rgba(13, 79, 113, 0.98) 0%, rgba(7, 50, 72, 0.98) 100%);
}

#assistant_chat_dock.is-open #assistant_chat_toggle .chat__toggle-text {
    color: #f4fbff;
}

.chat__toggle-text {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: #4d6070;
    font-size: calc(0.76rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
}

#assistant_chat_panel {
    position: absolute !important;
    top: 50%;
    left: calc(var(--dock-launcher-width) + var(--dock-gap));
    flex: 0 0 auto !important;
    width: min(var(--dock-panel-width), calc(100vw - 92px));
    padding: 12px;
    border: 1px solid rgba(16, 78, 109, 0.16);
    border-radius: 24px;
    background: #ffffff;
    box-shadow: 0 30px 60px rgba(8, 34, 50, 0.2);
    opacity: 0;
    visibility: hidden;
    transform: translateY(-50%) translateX(-30px) scale(0.98);
    transform-origin: left center !important;
    transition: opacity 0.22s ease, transform 0.22s ease, visibility 0.22s step-end;
    pointer-events: none;
    overflow: visible !important;
}

#assistant_chat_dock:not(.is-open) #assistant_chat_panel {
    display: none;
}

#assistant_chat_dock.is-open #assistant_chat_panel {
    display: block;
    opacity: 1;
    visibility: visible;
    transform: translateY(-50%) translateX(0) scale(1);
    transition: opacity 0.22s ease, transform 0.22s ease, visibility 0.22s step-start;
    pointer-events: auto;
}

#assistant_chat_panel.has-fixed-composer-layout {
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    max-height: calc(100vh - 36px);
}

#assistant_chat_panel.has-fixed-composer-layout #assistant_chat_shell_block {
    flex: 1 1 auto !important;
    min-height: var(--chat-history-min-height) !important;
    overflow: hidden !important;
}

#assistant_chat_panel.has-fixed-composer-layout #assistant_chat_shell_block > .html-container,
#assistant_chat_panel.has-fixed-composer-layout #assistant_chat_shell_block .prose,
#assistant_chat_panel.has-fixed-composer-layout #assistant_chat_html,
#assistant_chat_panel.has-fixed-composer-layout .chat {
    height: 100% !important;
    min-height: 0 !important;
}

#assistant_chat_panel.has-fixed-composer-layout #assistant_chat_controls,
#assistant_chat_panel.has-fixed-composer-layout #assistant_chat_stats_block {
    flex: 0 0 auto !important;
}

#assistant_chat_settings_launcher_host {
    position: absolute !important;
    top: 20px;
    right: -30px;
    z-index: 3;
    width: 30px !important;
    min-width: 30px !important;
    max-width: 30px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

#assistant_chat_settings_launcher_host .html-container,
#assistant_chat_settings_launcher_host .prose {
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
}

#assistant_chat_settings_toggle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    min-width: 30px;
    min-height: 156px;
    padding: 14px 4px;
    border: 1px solid rgba(16, 78, 109, 0.18);
    border-left: 0;
    border-radius: 0 18px 18px 0;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(237, 245, 250, 0.98) 100%);
    box-shadow: 0 16px 28px rgba(8, 34, 50, 0.12);
    cursor: pointer;
    transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}

#assistant_chat_settings_toggle:hover {
    box-shadow: 0 18px 30px rgba(8, 34, 50, 0.16);
}

#assistant_chat_panel.is-settings-open #assistant_chat_settings_toggle {
    background: linear-gradient(180deg, rgba(13, 79, 113, 0.98) 0%, rgba(7, 50, 72, 0.98) 100%);
}

#assistant_chat_panel.is-settings-open #assistant_chat_settings_toggle .chat__settings-toggle-text {
    color: #f4fbff;
}

.chat__settings-toggle-text {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #0f5375;
    font-size: calc(0.68rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
}

#assistant_chat_settings_panel {
    position: absolute !important;
    top: 0;
    left: calc(100% + var(--dock-settings-panel-offset));
    z-index: 2;
    width: min(var(--dock-settings-panel-width), calc(100vw - 150px));
    height: 100%;
    padding: 0;
    display: block;
    border: 0;
    border-radius: 24px;
    background: transparent;
    box-shadow: none;
    opacity: 0;
    visibility: hidden;
    transform: translateX(-24px) scale(0.98);
    transition: opacity 0.22s ease, transform 0.22s ease, visibility 0.22s step-end;
    pointer-events: none;
    overflow: visible !important;
}

#assistant_chat_panel.is-settings-open #assistant_chat_settings_panel {
    opacity: 1;
    visibility: visible;
    transform: translateX(0) scale(1);
    transition: opacity 0.22s ease, transform 0.22s ease, visibility 0.22s step-start;
    pointer-events: auto;
}

#assistant_chat_settings_panel .form,
#assistant_chat_settings_panel .wrap,
#assistant_chat_settings_panel .block,
#assistant_chat_settings_panel .gradio-container,
#assistant_chat_settings_panel .accordion {
    min-width: 0 !important;
}

#assistant_chat_settings_panel > .chat__settings-card {
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    display: flex !important;
    flex-direction: column !important;
    min-width: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
    border: 1px solid rgba(14, 71, 99, 0.18) !important;
    border-radius: 22px !important;
    background: #ffffff !important;
    box-shadow: 0 28px 56px rgba(8, 33, 49, 0.2) !important;
    overflow: hidden !important;
}

#assistant_chat_settings_panel > .chat__settings-card > .form {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel > .chat__settings-card > .chat__settings-scroll {
    display: block !important;
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 12px 12px 12px;
}

#assistant_chat_settings_panel > .chat__settings-card > .chat__settings-scroll > .block {
    display: block !important;
    margin: 0 0 12px !important;
    overflow: visible;
}

#assistant_chat_settings_panel > .chat__settings-card > .chat__settings-scroll > .block > .label-wrap {
    align-items: center;
    padding: 10px 14px;
    border: 1px solid rgba(23, 90, 125, 0.16);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(236, 244, 249, 0.98) 0%, rgba(224, 237, 245, 0.98) 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
}

#assistant_chat_settings_panel > .chat__settings-card > .chat__settings-scroll > .block > .label-wrap.open {
    margin-bottom: 8px;
}

#assistant_chat_settings_panel > .chat__settings-card > .chat__settings-scroll > .block > .label-wrap span {
    color: #174a67;
}

#assistant_chat_settings_panel > .chat__settings-card > .chat__settings-scroll > .block > div:last-child {
    overflow: visible;
}

#assistant_chat_settings_panel .label-wrap {
    gap: 6px;
}

#assistant_chat_shell_block,
#assistant_chat_stats_block,
#assistant_chat_controls {
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#assistant_chat_shell_block {
    margin-bottom: 8px !important;
}

#assistant_chat_stats_block {
    margin-top: 2px !important;
    margin-bottom: 4px !important;
}

#assistant_chat_controls,
#assistant_chat_controls > .form,
#assistant_chat_request {
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_controls > .form {
    padding: 0 !important;
    border: 0 !important;
    min-width: 0 !important;
}

#assistant_chat_controls {
    display: flex;
    align-items: center;
    flex-wrap: nowrap;
    justify-content: flex-start;
    gap: 10px;
}

#assistant_chat_request {
    order: 0;
    flex: 1 1 auto !important;
    width: auto !important;
    min-width: 0;
    padding: 0 !important;
}

#assistant_chat_request span[data-testid="block-info"],
#assistant_chat_controls span[data-testid="block-info"] {
    display: none !important;
}

#assistant_chat_request > .form,
#assistant_chat_request > .wrap {
    width: 100% !important;
    min-width: 0 !important;
    height: 100% !important;
    padding: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

#assistant_chat_request textarea,
#assistant_chat_request input {
    box-sizing: border-box !important;
    width: 100% !important;
    min-height: 48px !important;
    max-height: var(--chat-request-max-height) !important;
    overflow-y: auto !important;
    resize: none !important;
    font-size: calc(0.92rem * var(--dock-font-scale)) !important;
    line-height: 1.45;
    border: 1px solid rgba(23, 90, 125, 0.18) !important;
    border-radius: 15px !important;
    background: linear-gradient(180deg, rgba(248, 252, 255, 0.94) 0%, rgba(239, 246, 251, 0.95) 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7), 0 8px 18px rgba(14, 53, 75, 0.04) !important;
}

#assistant_chat_request textarea:focus,
#assistant_chat_request input:focus {
    border-color: rgba(23, 110, 154, 0.34) !important;
    box-shadow: 0 0 0 3px rgba(57, 145, 189, 0.16), 0 10px 20px rgba(14, 53, 75, 0.09) !important;
}

#assistant_chat_request label,
#assistant_chat_request .input-container {
    width: 100% !important;
    min-height: 52px !important;
    display: flex !important;
    align-items: center !important;
}

#assistant_chat_ask_button,
#assistant_chat_reset_button {
    order: 0;
    flex: 0 0 auto !important;
    align-self: center;
    min-width: 0 !important;
    height: 48px;
    min-height: 48px;
    padding: 0 14px;
    border-radius: 15px;
    font-size: calc(1.12rem * var(--dock-font-scale));
    font-weight: 700;
    box-shadow: 0 12px 22px rgba(11, 43, 63, 0.12);
    border: 0;
}

#assistant_chat_ask_button {
    width: 86px;
    background: linear-gradient(180deg, #0e5b81 0%, #0a415e 100%);
    color: #f3fbff;
}

#assistant_chat_reset_button {
    width: 82px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(239, 246, 250, 0.98) 100%);
    color: #164f70;
    border: 1px solid rgba(20, 82, 113, 0.14);
}

#assistant_chat_pause_bridge,
#assistant_chat_stop_bridge {
    display: none !important;
}

#assistant_chat_settings_panel .chat__settings-actions {
    margin-top: 10px;
}

#assistant_chat_settings_panel .chat__settings-actions > .form {
    width: 100%;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_save_settings_button {
    width: 100%;
    min-height: 42px;
    border-radius: 14px;
    background: linear-gradient(180deg, #0e5b81 0%, #0a415e 100%);
    color: #f3fbff;
    border: 0;
    box-shadow: 0 12px 22px rgba(11, 43, 63, 0.12);
}

#assistant_chat_html {
    min-height: 520px;
}

#assistant_chat_html > .chat {
    --chat-border: transparent;
    --chat-shadow: none;
    --chat-surface: #ffffff;
    --chat-status-offset: 18px;
    --chat-status-reserved-height: 0px;
    --chat-status-gap: 7px;
    --assistant-bg: linear-gradient(180deg, #145171 0%, #0c3954 100%);
    --assistant-border: rgba(8, 40, 57, 0.42);
    --assistant-text: #f2fbff;
    --user-bg: linear-gradient(180deg, #ffffff 0%, #f5fbff 100%);
    --user-border: rgba(55, 131, 180, 0.18);
    --user-text: #163f58;
    --muted-text: #5b7282;
    --soft-text: #6d8090;
    --tool-bg: rgba(234, 245, 251, 0.92);
    --tool-border: rgba(40, 108, 153, 0.16);
    --status-bg: linear-gradient(180deg, rgba(19, 51, 71, 0.95) 0%, rgba(10, 31, 47, 0.94) 100%);
    --status-text: #fbfeff;
    --empty-border: rgba(31, 94, 132, 0.12);
    position: relative;
    display: flex;
    flex-direction: column;
    height: 520px;
    overflow: hidden;
    border: 1px solid var(--chat-border);
    border-radius: 26px;
    background: var(--chat-surface);
    box-shadow: var(--chat-shadow);
    isolation: isolate;
}

#assistant_chat_html > .chat:has(.chat__status.is-visible) {
    --chat-status-reserved-height: 58px;
}

#assistant_chat_html > .chat::before {
    content: "";
    position: absolute;
    inset: 0;
    background: none;
    pointer-events: none;
}

.chat__scroll {
    position: relative;
    flex: 1;
    min-width: 0;
    overflow-x: hidden;
    overflow-y: auto;
    background: transparent;
}

.chat__scroll::-webkit-scrollbar {
    width: 10px;
}

.chat__scroll::-webkit-scrollbar-thumb {
    border-radius: 999px;
    border: 2px solid transparent;
    background: rgba(29, 92, 128, 0.2);
    background-clip: padding-box;
}

.chat__empty {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    min-height: 100%;
    box-sizing: border-box;
    padding: 22px 24px 16px;
    border: 0;
    border-radius: 0;
    color: var(--muted-text);
    text-align: left;
    font-size: calc(0.9rem * var(--dock-font-scale));
    line-height: 1.48;
    background: transparent;
    backdrop-filter: none;
}

#deepy_type_value {
    display: none !important;
}

.chat__empty-card {
    width: min(100%, 482px);
}

.chat__empty-header {
    position: relative;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 2px 12px;
    align-items: center;
    padding: 14px 16px;
    overflow: hidden;
    border: 1px solid rgba(24, 101, 144, 0.2);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(12, 79, 115, 0.98) 0%, rgba(19, 111, 151, 0.92) 56%, rgba(54, 151, 178, 0.88) 100%);
    box-shadow: 0 12px 26px rgba(15, 83, 119, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.18);
}

.chat__empty-header::after {
    content: "";
    position: absolute;
    top: -38px;
    right: -22px;
    width: 126px;
    height: 126px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.1);
}

.chat__empty-eyebrow {
    position: relative;
    z-index: 1;
    grid-column: 1 / -1;
    color: rgba(237, 249, 255, 0.76);
    font-size: calc(0.61rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.17em;
    line-height: 1.2;
    text-transform: uppercase;
}

.chat__empty-title {
    position: relative;
    z-index: 1;
    margin: 0;
    color: #ffffff;
    font-size: calc(1.48rem * var(--dock-font-scale));
    font-weight: 820;
    letter-spacing: -0.02em;
    line-height: 1.08;
}

.chat__empty-mode {
    position: relative;
    z-index: 1;
    max-width: 174px;
    padding: 5px 9px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 999px;
    color: #f2fbff;
    font-size: calc(0.65rem * var(--dock-font-scale));
    font-weight: 700;
    line-height: 1.15;
    text-align: center;
    background: rgba(4, 47, 71, 0.3);
}

.chat__empty-intro {
    margin: 13px 3px 12px;
    color: #3f5f72;
    font-size: calc(0.86rem * var(--dock-font-scale));
    line-height: 1.48;
}

.chat__empty-grid {
    display: grid;
    grid-template-columns: 1fr 0.92fr;
    gap: 10px;
}

.chat__input-helper {
    display: none;
    flex: 1 1 auto;
    min-width: 0;
    color: rgba(66, 88, 103, 0.68);
    font-size: calc(0.68rem * var(--dock-font-scale));
    line-height: 1.15;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.chat__input-helper.is-visible {
    display: block;
}

.chat__empty-section {
    padding: 12px 13px 11px;
    border: 1px solid rgba(31, 94, 132, 0.12);
    border-radius: 15px;
    background: linear-gradient(180deg, rgba(244, 250, 253, 0.96) 0%, rgba(235, 246, 251, 0.88) 100%);
}

.chat__empty-section--examples {
    background: linear-gradient(180deg, rgba(248, 251, 253, 0.98) 0%, rgba(241, 247, 250, 0.9) 100%);
}

.chat__empty-section h3 {
    margin: 0 0 7px;
    color: #194d70;
    font-size: calc(0.72rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.04em;
    line-height: 1.2;
    text-transform: uppercase;
}

.chat__empty-section ul {
    display: grid;
    gap: 6px;
    margin: 0;
    padding: 0;
    list-style: none;
}

.chat__empty-section li {
    position: relative;
    margin: 0;
    padding-left: 12px;
    color: #526d7e;
    font-size: calc(0.74rem * var(--dock-font-scale));
    line-height: 1.36;
}

.chat__empty-section li::before {
    content: "";
    position: absolute;
    top: 0.5em;
    left: 0;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #2c93bd;
    box-shadow: 0 0 0 3px rgba(44, 147, 189, 0.1);
}

.chat__empty-tip {
    margin: 10px 3px 0;
    color: #587486;
    font-size: calc(0.7rem * var(--dock-font-scale));
    font-weight: 650;
    line-height: 1.35;
}

.chat__session-picker {
    margin: 8px 3px 0;
    padding-top: 8px;
    border-top: 1px solid rgba(31, 94, 132, 0.12);
}

.chat__session-picker-spacer {
    height: calc(0.72rem * var(--dock-font-scale));
}

.chat__session-picker label {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    gap: 6px;
    color: #46677a;
    font-size: calc(0.62rem * var(--dock-font-scale));
    font-weight: 750;
}

.chat__session-picker label > span:first-child {
    white-space: nowrap;
}

.chat__session-picker-controls {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
}

.chat__session-picker select {
    box-sizing: border-box;
    flex: 0 1 240px;
    width: min(240px, 100%);
    min-width: 0;
    height: 28px;
    min-height: 28px;
    margin: 0;
    padding: 2px 24px 2px 7px;
    border: 1px solid rgba(31, 94, 132, 0.22);
    border-radius: 7px;
    color: #244b62;
    background: rgba(255, 255, 255, 0.92);
    cursor: pointer;
    font-size: calc(0.64rem * var(--dock-font-scale));
    line-height: 1.1;
}

.chat__session-picker button {
    box-sizing: border-box;
    flex: 0 0 auto;
    height: 28px;
    min-height: 28px;
    margin: 0;
    padding: 2px 8px;
    border: 1px solid rgba(20, 102, 140, 0.28);
    border-radius: 7px;
    color: #f7fcff;
    background: linear-gradient(180deg, #197da6 0%, #0d5f83 100%);
    box-shadow: 0 8px 16px rgba(10, 72, 103, 0.14);
    cursor: pointer;
    font-size: calc(0.62rem * var(--dock-font-scale));
    line-height: 1;
    font-weight: 800;
}

.chat__session-picker button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 11px 20px rgba(10, 72, 103, 0.2);
}

.chat__session-picker select:disabled {
    opacity: 0.62;
    cursor: default;
}

.chat__session-picker button:disabled {
    opacity: 0.52;
    cursor: default;
}

@media (min-width: 901px) {
    .chat__empty-card:has(.chat__session-picker) .chat__empty-intro {
        margin: 9px 3px 8px;
        line-height: 1.35;
    }

    .chat__empty-card:has(.chat__session-picker) .chat__empty-section {
        padding: 9px 11px 8px;
    }

    .chat__empty-card:has(.chat__session-picker) .chat__empty-section h3 {
        margin-bottom: 5px;
    }

    .chat__empty-card:has(.chat__session-picker) .chat__empty-section ul {
        gap: 4px;
    }

    .chat__empty-card:has(.chat__session-picker) .chat__empty-section li {
        line-height: 1.24;
    }

    .chat__empty-card:has(.chat__session-picker) .chat__empty-tip {
        margin-top: 6px;
        line-height: 1.2;
    }

    .chat__empty-card:has(.chat__session-picker) .chat__session-picker {
        margin-top: 4px;
        padding-top: 4px;
    }
}

#assistant_chat_settings_panel .chat__session-selector {
    align-items: flex-end;
}

#assistant_chat_settings_panel .chat__session-action-buttons {
    align-items: stretch;
    flex-wrap: nowrap;
    gap: 6px;
}

#assistant_chat_settings_panel .chat__session-action-buttons > .form,
#assistant_chat_settings_panel .chat__session-action-buttons > .block {
    flex: 1 1 0 !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
}

#assistant_chat_settings_panel .chat__session-action-buttons .chat__template-tool-icon-btn {
    width: 100% !important;
    min-width: 0 !important;
    max-width: none !important;
}

.chat__transcript {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 22px 18px calc(var(--chat-status-offset) + var(--chat-status-reserved-height) + var(--chat-status-gap));
}

#assistant_chat_html > .chat.is-replaying *,
#assistant_chat_html > .chat.is-replaying *::before,
#assistant_chat_html > .chat.is-replaying *::after {
    animation: none !important;
    transition: none !important;
    scroll-behavior: auto !important;
}

.chat__stats {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    min-height: calc(0.78rem * var(--dock-font-scale));
    padding: 0 2px;
    font-size: calc(0.64rem * var(--dock-font-scale));
    line-height: 1.15;
    white-space: nowrap;
    overflow: hidden;
    color: #8d9aa5;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.18s ease;
}

.chat__stats.is-visible,
.chat__stats.has-input-helper {
    opacity: 0.96;
}

.chat__stats-text {
    flex: 0 1 auto;
    min-width: 0;
    margin-left: auto;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
}

.chat__message {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    width: 100%;
    min-width: 0;
}

.chat__message--user {
    flex-direction: row-reverse;
}

.chat__avatar {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 54px;
    height: 54px;
    border-radius: 50%;
    font-size: calc(0.8rem * var(--dock-font-scale));
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    box-shadow: 0 12px 22px rgba(18, 61, 88, 0.12);
    margin-top: 10px;
}

.chat__message--assistant .chat__avatar {
    color: #eefbff;
    background: linear-gradient(180deg, rgba(11, 72, 103, 0.96) 0%, rgba(7, 48, 70, 0.96) 100%);
    border: 1px solid rgba(7, 39, 57, 0.35);
}

.chat__message--user .chat__avatar {
    color: #0e4564;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(245, 251, 255, 0.99) 100%);
    border: 1px solid rgba(47, 124, 170, 0.14);
}

.chat__message-card {
    position: relative;
    width: min(82%, 860px);
    min-width: 0;
    box-sizing: border-box;
    border-radius: 22px;
    padding: 16px 16px 14px;
    box-shadow: 0 18px 34px rgba(11, 36, 54, 0.08);
}

.chat__message--assistant .chat__message-card {
    border: 1px solid var(--assistant-border);
    background: var(--assistant-bg);
    color: var(--assistant-text);
}

.chat__message--user .chat__message-card {
    border: 1px solid var(--user-border);
    background: var(--user-bg);
    color: var(--user-text);
}

.chat__meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
    font-size: calc(0.82rem * var(--dock-font-scale));
    color: var(--soft-text);
}

.chat__message--assistant .chat__meta {
    color: rgba(242, 251, 255, 0.74);
}

.chat__message--assistant .chat__time {
    color: #f4fbff;
}

.chat__meta-left {
    display: inline-flex;
    align-items: center;
    min-height: 1em;
}

.chat__meta-right {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: 7px;
    min-width: 0;
}

.chat__message--user .chat__meta-right {
    padding-right: 0;
}

.chat__message--user .chat__message-actions {
    display: inline-flex;
    flex-direction: row;
    align-items: center;
    justify-content: flex-start;
    gap: 3px;
}

.chat__message--user .chat__body {
    padding-right: 0;
}

.chat__copy-button,
.chat__message-action-button,
.chat__collapse-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    min-width: 24px;
    padding: 0;
    border: 1px solid rgba(40, 104, 142, 0.16);
    border-radius: 7px;
    color: #346f92;
    background: rgba(255, 255, 255, 0.54);
    cursor: pointer;
    transition: opacity 0.16s ease, transform 0.16s ease, color 0.16s ease, background 0.16s ease;
}

.chat__copy-button svg,
.chat__message-action-button svg {
    width: 13px;
    height: 13px;
    fill: none;
    stroke: currentColor;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: 1.5;
}

.chat__message-actions .chat__copy-button,
.chat__message-action-button {
    margin: 0 !important;
}

.chat__copy-button:hover,
.chat__copy-button:focus-visible,
.chat__message-action-button:hover,
.chat__message-action-button:focus-visible {
    color: #0d5d89;
    background: rgba(255, 255, 255, 0.92);
    outline: none;
}

.chat__copy-button.is-copied {
    color: #19734a;
    border-color: rgba(25, 115, 74, 0.28);
    background: rgba(221, 249, 234, 0.96);
}

.chat__copy-button.is-copy-error {
    color: #a13f3f;
    border-color: rgba(161, 63, 63, 0.28);
}

.chat__message--assistant .chat__copy-button {
    color: rgba(245, 251, 255, 0.86);
    border-color: rgba(255, 255, 255, 0.14);
    background: rgba(255, 255, 255, 0.08);
}

.chat__message--assistant .chat__copy-button:hover,
.chat__message--assistant .chat__copy-button:focus-visible {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.16);
}

.chat__message--user .chat__copy-button,
.chat__message--user .chat__message-action-button,
.chat__tool-json .chat__copy-button {
    opacity: 0;
    pointer-events: none;
    transform: translateY(-2px);
}

.chat__tool-json .chat__copy-button {
    transform: none;
    transition: color 0.16s ease, background 0.16s ease;
}

.chat__message--user .chat__message-card:hover .chat__copy-button,
.chat__message--user .chat__copy-button:focus-visible,
.chat__message--user .chat__message-card:hover .chat__message-action-button,
.chat__message--user .chat__message-action-button:focus-visible,
.chat__tool-json:hover .chat__copy-button,
.chat__tool-json .chat__copy-button:focus-visible {
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0);
}

.chat__message.is-pending-queue-action .chat__message-action-button {
    opacity: 0.45;
    pointer-events: none;
}

.chat__message--user.is-editing .chat__message-card {
    outline: 2px solid rgba(31, 126, 177, 0.34);
    outline-offset: 2px;
}

.chat__author {
    font-weight: 700;
    letter-spacing: 0.03em;
}

.chat__time {
    opacity: 0.9;
    white-space: nowrap;
}

.chat__badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-left: 8px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: calc(0.72rem * var(--dock-font-scale));
    font-weight: 700;
    letter-spacing: 0.02em;
    background: rgba(31, 110, 154, 0.1);
    color: #20658f;
}

.chat__message--assistant .chat__badge {
    background: rgba(255, 255, 255, 0.12);
    color: #eff9ff;
}

.chat__message-end {
    display: flex;
    justify-content: flex-end;
    margin-top: 12px;
}

.chat__message-end-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    color: #eff9ff;
    font-size: calc(0.72rem * var(--dock-font-scale));
    font-weight: 700;
    letter-spacing: 0.02em;
}

.chat__message--assistant .chat__tool-title,
.chat__message--assistant .chat__disclosure summary {
    color: var(--assistant-text);
}

.chat__body {
    min-width: 0;
    font-size: calc(0.97rem * var(--dock-font-scale));
    line-height: 1.68;
}

.chat__message--assistant .chat__body,
.chat__message--assistant .chat__body p,
.chat__message--assistant .chat__body li,
.chat__message--assistant .chat__body strong,
.chat__message--assistant .chat__body em,
.chat__message--assistant .chat__body blockquote,
.chat__message--assistant .chat__body h1,
.chat__message--assistant .chat__body h2,
.chat__message--assistant .chat__body h3,
.chat__message--assistant .chat__body h4 {
    color: var(--assistant-text);
}

.chat__body > :first-child {
    margin-top: 0;
}

.chat__body > :last-child {
    margin-bottom: 0;
}

.chat__content-block,
.chat__tool-block {
    display: contents;
}

.chat__content-block:first-child > :first-child {
    margin-top: 0;
}

.chat__content-block:last-child > :last-child {
    margin-bottom: 0;
}

.chat__stream-text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    color: inherit;
    font: inherit;
}

.chat__stream-tail {
    white-space: pre-wrap;
    color: inherit !important;
    background: transparent !important;
    font: inherit;
}

.chat__stream-text > span {
    background: transparent !important;
}

.chat__message--assistant .chat__stream-text,
.chat__message--assistant .chat__stream-text * {
    color: var(--assistant-text) !important;
}

.chat__message--user .chat__stream-text,
.chat__message--user .chat__stream-text * {
    color: var(--user-text) !important;
}

.chat__body p,
.chat__body ul,
.chat__body ol,
.chat__body pre,
.chat__body blockquote {
    margin: 0 0 0.85em;
}

.chat__body ul,
.chat__body ol {
    padding-left: 1.4em;
    list-style-position: outside;
}

.chat__body code {
    padding: 0.12em 0.34em;
    border-radius: 8px;
    font-size: 0.92em;
    background: rgba(16, 73, 104, 0.08);
}

.chat__message--assistant .chat__body code {
    color: var(--assistant-text);
    background: rgba(255, 255, 255, 0.12);
}

.chat__body pre {
    overflow-x: auto;
    padding: 12px 13px;
    border-radius: 14px;
    border: 1px solid rgba(26, 84, 117, 0.12);
    background: rgba(239, 247, 251, 0.96);
}

.chat__message--assistant .chat__body pre {
    color: var(--assistant-text);
    border-color: rgba(255, 255, 255, 0.12);
    background: rgba(7, 33, 48, 0.38);
}

.chat__body a {
    color: inherit;
    font-weight: 600;
}

.chat__disclosure {
    margin-top: 12px;
    border: 1px solid var(--tool-border);
    border-radius: 16px;
    background: var(--tool-bg);
    overflow: hidden;
}

.chat__message--assistant .chat__disclosure {
    border-color: rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.08);
}

.chat__disclosure summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 9px 12px;
    cursor: pointer;
    list-style: none;
    font-weight: 700;
    font-size: calc(0.7rem * var(--dock-font-scale));
    line-height: 1.3;
}

.chat__disclosure > summary {
    display: flex;
}

.chat__disclosure summary::-webkit-details-marker {
    display: none;
}

.chat__disclosure summary::after {
    content: "\25B8";
    font-size: calc(0.78rem * var(--dock-font-scale));
    transition: color 0.18s ease;
    color: #2f769f;
}

.chat__disclosure[open] summary::after {
    content: "\25BE";
}

.chat__message--assistant .chat__disclosure summary::after {
    color: rgba(245, 251, 255, 0.86);
}

.chat__disclosure-body {
    padding: 0 14px 14px;
    font-size: calc(0.84rem * var(--dock-font-scale));
    line-height: 1.52;
    color: #385363;
}

.chat__reasoning-block > :last-child {
    margin-bottom: 0;
}

.chat__collapse-button {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    box-sizing: border-box;
    margin: 0 -14px -14px;
    width: calc(100% + 28px);
    height: 20px;
    min-width: 0;
    padding: 0 12px;
    border: 0;
    border-radius: 0;
    color: #ffffff !important;
    background: transparent !important;
    box-shadow: none !important;
    line-height: 1;
    transition: none;
}

.chat__collapse-button > span {
    display: block;
    color: #ffffff !important;
    font-size: calc(0.78rem * var(--dock-font-scale));
    line-height: 1;
    transform: rotate(180deg);
}

.chat__collapse-button:focus-visible {
    outline: 1px solid currentColor;
    outline-offset: 1px;
}

.chat__disclosure:not([open]) > .chat__disclosure-body {
    display: none;
}

.chat__disclosure[open] > .chat__disclosure-body {
    display: block;
}

.chat__message--assistant .chat__disclosure-body {
    color: var(--assistant-text);
}

.chat__message--assistant .chat__reasoning-block,
.chat__message--assistant .chat__context-summary {
    color: var(--assistant-text) !important;
    background: transparent !important;
}

.chat__context-summary > :first-child {
    margin-top: 0;
}

.chat__context-summary > :last-child {
    margin-bottom: 0;
}

.chat__tool-title {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: calc(0.72rem * var(--dock-font-scale));
}

.chat__tool-chip {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: calc(0.54rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #205f86;
    background: rgba(33, 109, 153, 0.12);
}

.chat__message--assistant .chat__tool-chip {
    color: #eff9ff;
    background: rgba(255, 255, 255, 0.14);
}

.chat__tool-status {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 999px;
    font-size: calc(0.55rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.02em;
}

.chat__tool-status--running {
    background: rgba(229, 160, 38, 0.14);
    color: #90600f;
}

.chat__tool-status--done {
    background: rgba(72, 208, 128, 0.16);
    color: #5df0a0;
}

.chat__tool-status--error {
    background: rgba(183, 62, 62, 0.12);
    color: #973232;
}

.chat__pre {
    margin: 10px 0 0;
    padding: 12px 13px;
    border-radius: 14px;
    overflow-x: auto;
    background: rgba(247, 251, 253, 0.95);
    border: 1px solid rgba(30, 92, 127, 0.1);
    font-size: calc(0.72rem * var(--dock-font-scale));
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
}

.chat__message--assistant .chat__pre {
    color: var(--assistant-text);
    background: rgba(7, 33, 48, 0.38);
    border-color: rgba(255, 255, 255, 0.12);
}

.chat__tool-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
}

.chat__tool-json {
    min-width: 0;
}

.chat__message--assistant .chat__tool-pending {
    color: var(--assistant-text);
}

.chat__tool-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    min-height: 24px;
}

.chat__tool-section-title {
    margin-bottom: 6px;
    font-size: calc(0.67rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #557385;
}

.chat__message--assistant .chat__tool-section-title {
    color: rgba(233, 246, 255, 0.76);
}

.chat__tool-section-header .chat__tool-section-title {
    margin-bottom: 0;
}

@media (hover: none) {
    .chat__message--user .chat__copy-button,
    .chat__message--user .chat__message-action-button,
    .chat__tool-json .chat__copy-button {
        opacity: 0.72;
        pointer-events: auto;
        transform: none;
    }
}

.chat__structured-result {
    margin-top: 10px;
    border: 1px solid rgba(116, 190, 230, 0.22);
    border-radius: 12px;
    overflow: hidden;
    color: var(--assistant-text);
    background: rgba(7, 33, 48, 0.28);
}

.chat__structured-result-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 9px 11px;
    font-size: calc(0.76rem * var(--dock-font-scale));
    border-bottom: 1px solid rgba(116, 190, 230, 0.18);
    color: #f5fbff !important;
    background: linear-gradient(180deg, rgba(20, 89, 123, 0.98) 0%, rgba(10, 60, 88, 0.98) 100%);
}

.chat__structured-result-header > * {
    color: #f5fbff !important;
}

.chat__structured-result-scroll {
    max-height: 360px;
    overflow: auto;
}

#assistant_chat_html table {
    width: 100%;
    table-layout: fixed;
    margin: 0 0 0.85em;
    border-collapse: collapse;
    border: 1px solid rgba(116, 190, 230, 0.18);
    font-size: 0.92em;
    color: var(--assistant-text);
}

#assistant_chat_html th,
#assistant_chat_html td {
    padding: 7px 9px;
    border: 1px solid rgba(116, 190, 230, 0.18);
    text-align: left;
    vertical-align: top;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    color: var(--assistant-text);
}

#assistant_chat_html table a {
    color: #bfe9ff;
}

#assistant_chat_html th {
    color: #f5fbff !important;
    background: rgba(11, 65, 94, 0.98) !important;
}

#assistant_chat_html .chat__structured-result table {
    margin: 0;
    table-layout: auto;
    font-size: calc(0.72rem * var(--dock-font-scale));
}

.chat__structured-result th {
    position: sticky;
    top: 0;
    z-index: 1;
}

.chat__attachments {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
    margin-top: 12px;
}

.chat__attachment {
    display: flex;
    gap: 12px;
    align-items: center;
    min-width: 0;
    padding: 12px;
    border: 1px solid rgba(31, 101, 141, 0.12);
    border-radius: 16px;
    color: inherit;
    text-decoration: none;
    background: rgba(255, 255, 255, 0.78);
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}

.chat__attachment:hover {
    transform: translateY(-1px);
    border-color: rgba(31, 101, 141, 0.22);
    box-shadow: 0 14px 28px rgba(12, 45, 67, 0.1);
}

.chat__attachment-thumb {
    flex: 0 0 88px;
    width: 88px;
    height: 88px;
    box-sizing: border-box;
    object-fit: cover;
    border-radius: 14px;
    border: 1px solid rgba(26, 82, 114, 0.12);
    background: rgba(234, 245, 251, 0.9);
    overflow: hidden;
}

.chat__attachment-thumb--icon {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #2b7299;
    background: linear-gradient(145deg, rgba(239, 249, 254, 0.98), rgba(219, 239, 249, 0.96));
}

.chat__attachment-thumb--icon svg {
    width: 46px;
    height: 46px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.chat__attachment-thumb--archive {
    color: #93651a;
    background: linear-gradient(145deg, rgba(255, 249, 226, 0.98), rgba(247, 229, 171, 0.96));
}

.chat__attachment-meta {
    min-width: 0;
}

.chat__attachment-title {
    display: block;
    font-weight: 700;
    color: #1b587e;
}

.chat__attachment-subtitle {
    display: block;
    margin-top: 4px;
    color: #667d8c;
    font-size: calc(0.84rem * var(--dock-font-scale));
    line-height: 1.45;
    word-break: break-word;
}

.chat__status {
    position: absolute;
    left: 18px;
    right: 18px;
    bottom: var(--chat-status-offset);
    z-index: 3;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0;
    padding: 12px 14px;
    border-radius: 18px;
    background: var(--status-bg);
    color: var(--status-text);
    box-shadow: 0 16px 34px rgba(10, 30, 46, 0.18);
    transform: translateY(8px);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.18s ease, transform 0.18s ease;
}

.chat__status,
.chat__status-text,
.chat__status-pause,
.chat__status-stop {
    color: var(--status-text);
}

.chat__status.is-visible {
    opacity: 1;
    transform: translateY(0);
}

.chat__status-text {
    flex: 1;
    min-width: 0;
    font-size: calc(0.92rem * var(--dock-font-scale));
    line-height: 1.45;
    font-weight: 600;
    pointer-events: none;
}

.chat__status-dots {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    pointer-events: none;
}

.chat__status-dots span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.9);
    animation: wangp-assistant-chat-pulse 1.18s infinite ease-in-out;
}

.chat__status-dots span:nth-child(2) {
    animation-delay: 0.15s;
}

.chat__status-dots span:nth-child(3) {
    animation-delay: 0.3s;
}

.chat__status-pause,
.chat__status-stop {
    display: inline-flex;
    align-items: center;
    align-self: center;
    justify-content: center;
    min-width: 62px;
    min-height: 34px;
    margin: 0;
    padding: 0 12px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 999px;
    background: rgba(179, 58, 58, 0.9);
    box-shadow: 0 10px 18px rgba(6, 18, 28, 0.16);
    font-size: calc(0.74rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    cursor: pointer;
    pointer-events: none;
    transition: transform 0.16s ease, background 0.16s ease, opacity 0.16s ease;
}

.chat__status.is-visible .chat__status-pause,
.chat__status.is-visible .chat__status-stop {
    pointer-events: auto;
}

.chat__status-pause {
    min-width: 70px;
    background: rgba(25, 111, 154, 0.94);
}

.chat__status-pause[data-mode="resume"] {
    background: rgba(185, 125, 30, 0.94);
}

.chat__status-pause:hover:not(:disabled) {
    transform: translateY(-1px);
    background: rgba(35, 133, 181, 0.98);
}

.chat__status-pause[data-mode="resume"]:hover:not(:disabled) {
    background: rgba(207, 145, 42, 0.98);
}

.chat__status-stop:hover:not(:disabled) {
    transform: translateY(-1px);
    background: rgba(197, 72, 72, 0.96);
}

.chat__status-pause:disabled,
.chat__status-stop:disabled {
    opacity: 0.55;
    cursor: default;
}

.chat__status-pause[hidden],
.chat__status-stop[hidden] {
    display: none;
}

.chat__status[data-kind="paused"] .chat__status-dots span {
    animation: none;
    opacity: 0.72;
}

.chat__jump-bottom {
    position: absolute;
    left: 50%;
    bottom: calc(var(--chat-status-offset) + 8px);
    z-index: 4;
    width: 42px;
    height: 42px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 2px solid rgba(251, 254, 255, 0.88);
    border-radius: 999px;
    background: transparent;
    color: transparent;
    box-shadow: none;
    backdrop-filter: none;
    transform: translate(-50%, 10px);
    opacity: 0;
    pointer-events: none;
    filter: drop-shadow(0 2px 4px rgba(9, 31, 46, 0.28));
    transition: opacity 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
}

.chat__jump-bottom.is-visible {
    opacity: 1;
    pointer-events: auto;
    transform: translate(-50%, 0);
}

.chat__jump-bottom span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    box-sizing: border-box;
    border-right: 3px solid rgba(251, 254, 255, 0.98);
    border-bottom: 3px solid rgba(251, 254, 255, 0.98);
    transform: translateY(-2px) rotate(45deg);
}

.chat__jump-bottom:hover {
    border-color: rgba(251, 254, 255, 1);
}

.chat__jump-bottom:hover span {
    border-right-color: rgba(251, 254, 255, 1);
    border-bottom-color: rgba(251, 254, 255, 1);
}

#assistant_chat_settings_panel .chat__template-tool-grid {
    position: relative;
    gap: 12px;
}

#assistant_chat_settings_panel .chat__template-tool-grid-row {
    gap: 12px;
    align-items: stretch;
}

#assistant_chat_settings_panel .chat__template-tool-card {
    min-width: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .chat__template-tool-card > .form {
    min-width: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .chat__template-tool-row {
    gap: 10px;
    align-items: flex-end;
    flex-wrap: nowrap;
}

#assistant_chat_settings_panel .chat__template-tool-dropdown {
    flex: 1 1 auto !important;
    min-width: 0 !important;
}

#assistant_chat_settings_panel .chat__template-tool-row > .form,
#assistant_chat_settings_panel .chat__template-tool-dropdown {
    min-width: 0 !important;
}

#assistant_chat_settings_panel .chat__template-tool-row > .form,
#assistant_chat_settings_panel .chat__template-tool-dropdown,
#assistant_chat_settings_panel .chat__template-tool-dropdown .wrap {
    overflow: visible !important;
}

#assistant_chat_settings_panel .chat__template-tool-dropdown .wrap > ul.options[role="listbox"] {
    position: absolute !important;
    inset: calc(100% - 8px) auto auto 0 !important;
    width: 100% !important;
    max-height: min(280px, 40vh) !important;
    z-index: 2147483647 !important;
}

#assistant_chat_settings_panel .chat__template-tool-actions {
    flex: 0 0 auto !important;
    gap: 4px;
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
}

#assistant_chat_settings_panel .chat__template-tool-actions > .form {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .chat__template-tool-icon-btn {
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    height: 34px;
    min-height: 34px;
    padding: 0 !important;
    border: 1px solid rgba(17, 84, 118, 0.14);
    border-radius: 12px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(236, 244, 249, 0.99) 100%);
    color: #155574;
    box-shadow: 0 10px 18px rgba(11, 44, 63, 0.08);
    font-size: calc(0.88rem * var(--dock-font-scale));
    line-height: 1;
    font-weight: 700;
    transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
}

#assistant_chat_settings_panel .chat__template-tool-icon-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 24px rgba(11, 44, 63, 0.12);
}

#assistant_chat_settings_panel .chat__template-tool-icon-btn--danger {
    color: #8b2d2d;
    background: linear-gradient(180deg, rgba(255, 252, 252, 0.99) 0%, rgba(249, 239, 239, 0.99) 100%);
    border-color: rgba(156, 62, 62, 0.16);
}

#assistant_chat_settings_panel .chat__template-modal-wrap.hide {
    display: none !important;
    pointer-events: none !important;
}

#assistant_chat_settings_panel .chat__template-modal-wrap:not(.hide) {
    position: absolute !important;
    inset: 0;
    z-index: 40;
    display: flex !important;
    align-items: center;
    justify-content: center;
    margin: 0 !important;
    padding: 12px !important;
    border: 0 !important;
    background: rgba(10, 38, 53, 0.18) !important;
    backdrop-filter: blur(3px);
    overflow: hidden !important;
    box-sizing: border-box;
}

#assistant_chat_settings_panel .chat__template-modal-wrap > .form {
    width: 100% !important;
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .chat__template-modal-wrap > .styler {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: min(100%, 450px) !important;
    max-width: 450px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
    flex: 0 1 auto !important;
}

#assistant_chat_settings_panel .chat__template-modal-card {
    width: 100% !important;
    max-width: 450px !important;
    min-width: 0 !important;
    flex: 0 1 auto !important;
    padding: 0 !important;
    gap: 0 !important;
    border: 1px solid rgba(14, 71, 99, 0.18) !important;
    border-radius: 22px !important;
    background: #ffffff !important;
    box-shadow: 0 28px 56px rgba(8, 33, 49, 0.2) !important;
    overflow: hidden !important;
}

#assistant_chat_settings_panel > .chat__settings-card.chat__template-modal-card {
    width: 100% !important;
    max-width: none !important;
    flex: 1 1 auto !important;
}

#assistant_chat_settings_panel .tab-nav button,
#assistant_chat_settings_panel button[role="tab"] {
    font-size: calc(0.82rem * var(--dock-font-scale));
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}

#assistant_chat_settings_panel .chat__template-modal-card > .form {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .chat__template-modal-card .html-container {
    padding: 0 !important;
}

#assistant_chat_settings_panel .chat__template-modal-card .prose {
    margin: 0 !important;
    max-width: none !important;
}

.chat__template-modal-titlebar {
    padding: 10px 16px 9px;
    background: linear-gradient(180deg, rgba(16, 86, 121, 0.98) 0%, rgba(10, 59, 84, 0.98) 100%);
    color: #f3fbff;
}

.chat__template-modal-kicker {
    font-size: calc(0.66rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    opacity: 0.78;
}

.chat__template-modal-heading {
    font-size: calc(0.9rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.02em;
    color: #f3fbff !important;
}

.chat__template-modal-context {
    margin: 16px 18px 0;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
}

.chat__template-modal-context-label {
    font-size: calc(0.7rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5b7282;
}

.chat__template-modal-context-value {
    margin-top: 5px;
    color: #174a67;
    font-size: calc(0.95rem * var(--dock-font-scale));
    font-weight: 700;
    word-break: break-word;
}

.chat__template-modal-message {
    margin: 14px 18px 0;
    padding: 0;
    border-radius: 0;
    font-size: calc(0.9rem * var(--dock-font-scale));
    line-height: 1.5;
    font-weight: 600;
    background: transparent !important;
}

.chat__template-modal-message.is-info {
    color: #164f70;
}

.chat__template-modal-message.is-warning {
    color: #7a5415;
}

.chat__template-modal-message.is-error {
    color: #b33434;
}

#assistant_chat_settings_panel .chat__template-modal-input {
    width: calc(100% - 36px) !important;
    margin: 14px 18px 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .chat__template-modal-input input,
#assistant_chat_settings_panel .chat__template-modal-input textarea {
    border: 1px solid rgba(17, 84, 118, 0.2) !important;
    border-radius: 11px !important;
    background: rgba(246, 251, 253, 0.98) !important;
    color: #174a67 !important;
}

.chat__template-modal-actions {
    justify-content: flex-end;
    gap: 10px;
    padding: 18px;
}

.chat__template-modal-btn {
    min-width: 92px;
    height: 40px;
    min-height: 40px;
    border-radius: 14px;
    border: 1px solid rgba(17, 84, 118, 0.14);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(237, 245, 250, 0.99) 100%);
    color: #155574;
    box-shadow: 0 10px 18px rgba(11, 44, 63, 0.08);
    font-weight: 700;
}

.chat__template-modal-btn--primary {
    color: #f4fbff;
    border-color: rgba(10, 59, 84, 0.12);
    background: linear-gradient(180deg, rgba(16, 86, 121, 0.98) 0%, rgba(10, 59, 84, 0.98) 100%);
}

#assistant_chat_dock.is-dark #assistant_chat_toggle {
    border-color: rgba(28, 104, 145, 0.28);
    background: linear-gradient(180deg, rgba(13, 79, 113, 0.98) 0%, rgba(7, 50, 72, 0.98) 100%);
    box-shadow: 0 18px 34px rgba(0, 0, 0, 0.34);
}

#assistant_chat_dock.is-dark #assistant_chat_toggle .chat__toggle-text {
    color: #f4fbff;
}

#assistant_chat_dock.is-dark.is-open #assistant_chat_toggle {
    border-color: rgba(115, 120, 126, 0.6);
    background: linear-gradient(180deg, rgba(92, 96, 102, 0.98) 0%, rgba(58, 61, 66, 0.98) 100%);
}

#assistant_chat_dock.is-dark.is-open #assistant_chat_toggle .chat__toggle-text {
    color: #f4fbff;
}

#assistant_chat_dock.is-dark #assistant_chat_settings_toggle {
    border-color: rgba(28, 104, 145, 0.28);
    background: linear-gradient(180deg, rgba(13, 79, 113, 0.98) 0%, rgba(7, 50, 72, 0.98) 100%);
    box-shadow: 0 16px 28px rgba(0, 0, 0, 0.3);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_toggle .chat__settings-toggle-text {
    color: #f4fbff;
}

#assistant_chat_dock.is-dark #assistant_chat_panel.is-settings-open #assistant_chat_settings_toggle {
    border-color: rgba(115, 120, 126, 0.6);
    background: linear-gradient(180deg, rgba(92, 96, 102, 0.98) 0%, rgba(58, 61, 66, 0.98) 100%);
}

#assistant_chat_dock.is-dark #assistant_chat_panel.is-settings-open #assistant_chat_settings_toggle .chat__settings-toggle-text {
    color: #f4fbff;
}

#assistant_chat_dock.is-dark #assistant_chat_panel,
#assistant_chat_dock.is-dark #assistant_chat_settings_panel {
    border-color: rgba(92, 96, 102, 0.78);
    background: #000000;
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.46), inset 0 0 0 1px rgba(70, 73, 78, 0.42);
    color: #eaf2f7;
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel > .chat__settings-card > .chat__settings-scroll > .block > .label-wrap {
    border-color: rgba(112, 138, 156, 0.18);
    background: linear-gradient(180deg, rgba(9, 9, 9, 0.98) 0%, rgba(20, 20, 20, 0.98) 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel > .chat__settings-card > .chat__settings-scroll > .block > .label-wrap span {
    color: #e6eef4;
}

#assistant_chat_dock.is-dark #assistant_chat_request textarea,
#assistant_chat_dock.is-dark #assistant_chat_request input {
    color: #eef6fb !important;
    caret-color: #eef6fb !important;
    border-color: rgba(103, 132, 151, 0.24) !important;
    background: linear-gradient(180deg, rgba(10, 10, 10, 0.96) 0%, rgba(19, 19, 19, 0.96) 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 8px 18px rgba(0, 0, 0, 0.22) !important;
}

#assistant_chat_dock.is-dark #assistant_chat_request textarea::placeholder,
#assistant_chat_dock.is-dark #assistant_chat_request input::placeholder {
    color: #93a6b4 !important;
}

#assistant_chat_dock.is-dark .chat__input-helper {
    color: rgba(174, 190, 201, 0.68);
}

#assistant_chat_dock.is-dark #assistant_chat_reset_button {
    color: #e8f1f6;
    border-color: rgba(103, 132, 151, 0.22);
    background: linear-gradient(180deg, rgba(12, 12, 12, 0.98) 0%, rgba(22, 22, 22, 0.98) 100%);
    box-shadow: 0 12px 22px rgba(0, 0, 0, 0.22);
}

#assistant_chat_dock.is-dark .chat {
    --chat-surface: #000000;
    --assistant-bg: linear-gradient(180deg, #0f4a69 0%, #082f45 100%);
    --assistant-border: rgba(67, 114, 143, 0.34);
    --assistant-text: #f2fbff;
    --user-bg: linear-gradient(180deg, #12181d 0%, #090d10 100%);
    --user-border: rgba(101, 127, 145, 0.2);
    --user-text: #edf4f9;
    --muted-text: #b3c1cb;
    --soft-text: #98a9b5;
    --tool-bg: rgba(17, 24, 30, 0.96);
    --tool-border: rgba(103, 132, 151, 0.18);
    --empty-border: rgba(103, 132, 151, 0.16);
}

#assistant_chat_dock.is-dark .chat__empty-intro,
#assistant_chat_dock.is-dark .chat__empty-section li,
#assistant_chat_dock.is-dark .chat__empty-tip,
#assistant_chat_dock.is-dark .chat__body,
#assistant_chat_dock.is-dark .chat__body p,
#assistant_chat_dock.is-dark .chat__body li,
#assistant_chat_dock.is-dark .chat__body strong,
#assistant_chat_dock.is-dark .chat__body em,
#assistant_chat_dock.is-dark .chat__body blockquote,
#assistant_chat_dock.is-dark .chat__body h1,
#assistant_chat_dock.is-dark .chat__body h2,
#assistant_chat_dock.is-dark .chat__body h3,
#assistant_chat_dock.is-dark .chat__body h4 {
    color: #edf4f9;
}

#assistant_chat_dock.is-dark .chat__empty-header {
    border-color: rgba(100, 171, 205, 0.28);
    background: linear-gradient(135deg, rgba(7, 49, 72, 0.98) 0%, rgba(10, 75, 104, 0.96) 58%, rgba(18, 98, 125, 0.92) 100%);
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

#assistant_chat_dock.is-dark .chat__empty-section {
    border-color: rgba(103, 132, 151, 0.2);
    background: linear-gradient(180deg, rgba(18, 24, 29, 0.98) 0%, rgba(10, 15, 19, 0.98) 100%);
}

#assistant_chat_dock.is-dark .chat__empty-section h3 {
    color: #9edaf3;
}

#assistant_chat_dock.is-dark .chat__stats {
    color: #9eb0bd;
}

#assistant_chat_dock.is-dark .chat__message--user .chat__avatar {
    color: #eef6fb;
    background: linear-gradient(180deg, rgba(24, 31, 37, 0.99) 0%, rgba(10, 12, 14, 0.99) 100%);
    border-color: rgba(103, 132, 151, 0.2);
}

#assistant_chat_dock.is-dark .chat__message--user .chat__copy-button {
    color: #b9d9ea;
    border-color: rgba(148, 185, 205, 0.2);
    background: rgba(255, 255, 255, 0.07);
}

#assistant_chat_dock.is-dark .chat__body code {
    background: rgba(130, 162, 183, 0.12);
}

#assistant_chat_dock.is-dark .chat__body pre {
    color: #eaf2f7;
    border-color: rgba(103, 132, 151, 0.16);
    background: rgba(10, 14, 17, 0.96);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel .chat__template-tool-icon-btn,
#assistant_chat_dock.is-dark .chat__template-modal-btn {
    color: #ecf4f9;
    border-color: rgba(103, 132, 151, 0.22);
    background: linear-gradient(180deg, rgba(10, 10, 10, 0.99) 0%, rgba(21, 21, 21, 0.99) 100%);
    box-shadow: 0 10px 18px rgba(0, 0, 0, 0.22);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel .chat__template-tool-icon-btn--danger {
    color: #ffb1b1;
    border-color: rgba(173, 84, 84, 0.24);
    background: linear-gradient(180deg, rgba(22, 10, 10, 0.99) 0%, rgba(32, 14, 14, 0.99) 100%);
}

#assistant_chat_dock.is-dark .chat__session-picker label {
    color: #adc7d6;
}

#assistant_chat_dock.is-dark .chat__session-picker select {
    color: #e8f1f6;
    border-color: rgba(125, 166, 189, 0.24);
    background: rgba(13, 18, 22, 0.96);
}

#assistant_chat_dock.is-dark .chat__session-picker button {
    color: #f2f9fc;
    border-color: rgba(95, 174, 210, 0.34);
    background: linear-gradient(180deg, #176c91 0%, #0c4b69 100%);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel .chat__template-modal-wrap:not(.hide) {
    background: rgba(0, 0, 0, 0.52) !important;
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel .chat__template-modal-card {
    border-color: rgba(92, 96, 102, 0.82) !important;
    background: #000000 !important;
    box-shadow: 0 28px 56px rgba(0, 0, 0, 0.42), inset 0 0 0 1px rgba(70, 73, 78, 0.44) !important;
}

#assistant_chat_dock.is-dark .chat__template-modal-context-label {
    color: #9fb1be;
}

#assistant_chat_dock.is-dark .chat__template-modal-context-value,
#assistant_chat_dock.is-dark .chat__template-modal-message.is-info {
    color: #ecf4f9;
}

#assistant_chat_dock.is-dark .chat__template-modal-message.is-warning {
    color: #f3d189;
}

#assistant_chat_dock.is-dark .chat__template-modal-message.is-error {
    color: #ff9e9e;
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel .chat__template-modal-input input,
#assistant_chat_dock.is-dark #assistant_chat_settings_panel .chat__template-modal-input textarea {
    border-color: rgba(125, 166, 189, 0.28) !important;
    background: rgba(17, 22, 26, 0.98) !important;
    color: #ecf4f9 !important;
}

@keyframes wangp-assistant-chat-pulse {
    0%, 80%, 100% { transform: scale(0.66); opacity: 0.46; }
    40% { transform: scale(1); opacity: 1; }
}

@media (max-width: 900px) {
    #assistant_chat_dock {
        top: auto;
        bottom: 18px;
        width: 36px;
        transform: none;
    }

    #assistant_chat_toggle {
        min-height: 152px;
        width: 36px;
        min-width: 36px;
        padding: 14px 5px;
        border-radius: 0 18px 18px 0;
    }

    #assistant_chat_panel {
        top: auto;
        bottom: 0;
        left: calc(36px + var(--dock-gap));
        width: min(360px, calc(100vw - 72px));
        padding: 12px;
        transform: translateX(-20px) scale(0.98);
    }

    .chat {
        height: 471px;
        border-radius: 20px;
    }

    #assistant_chat_html {
        min-height: 471px;
    }

    .chat__scroll {
        padding: 0;
    }

    .chat__message-card {
        width: min(92%, 100%);
        padding: 14px 14px 12px;
    }

    .chat__avatar {
        width: 46px;
        height: 46px;
        margin-top: 9px;
    }

    .chat__empty {
        padding: 18px 14px 14px;
    }

    .chat__empty-grid {
        grid-template-columns: 1fr;
    }

    .chat__empty-header {
        grid-template-columns: 1fr;
    }

    .chat__empty-mode {
        max-width: none;
        justify-self: start;
        margin-top: 5px;
    }

    .chat__transcript {
        padding: 16px 12px calc(var(--chat-status-offset) + var(--chat-status-reserved-height) + var(--chat-status-gap));
    }

    .chat__attachments {
        grid-template-columns: 1fr;
    }

    .chat__attachment-thumb {
        width: 72px;
        height: 72px;
        flex-basis: 72px;
    }

    #assistant_chat_controls {
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    #assistant_chat_request {
        flex: 1 1 100% !important;
        width: 100% !important;
        order: 1;
    }

    #assistant_chat_ask_button,
    #assistant_chat_reset_button {
        order: 2;
        flex: 1 1 calc(50% - 5px) !important;
        width: auto;
    }

    #assistant_chat_dock.is-open #assistant_chat_panel {
        transform: translateX(0) scale(1);
    }

    #assistant_chat_settings_launcher_host {
        top: 14px;
        right: 12px;
        width: auto !important;
        min-width: 0 !important;
        max-width: none !important;
    }

    #assistant_chat_settings_toggle {
        min-height: 30px;
        width: auto;
        min-width: 30px;
        padding: 8px 12px;
        border-radius: 14px;
        border-left: 1px solid rgba(16, 78, 109, 0.18);
    }

    .chat__settings-toggle-text {
        writing-mode: horizontal-tb;
        transform: none;
        letter-spacing: 0.08em;
    }

    #assistant_chat_settings_panel {
        top: 12px;
        left: 12px;
        width: calc(100% - 24px);
        height: calc(100% - 24px);
        transform: translateY(10px) scale(0.98);
    }

    #assistant_chat_panel.is-settings-open #assistant_chat_settings_panel {
        transform: translateY(0) scale(1);
    }

    #assistant_chat_settings_panel .chat__template-tool-grid-row,
    #assistant_chat_settings_panel .chat__template-tool-row,
    #assistant_chat_settings_panel .chat__session-action-buttons,
    #assistant_chat_settings_panel .chat__session-options {
        flex-wrap: wrap;
    }

    #assistant_chat_settings_panel .chat__session-selector > .form,
    #assistant_chat_settings_panel .chat__session-options > .form,
    #assistant_chat_settings_panel .chat__session-options > .block {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 0 !important;
        max-width: none !important;
    }

    #assistant_chat_settings_panel .chat__session-options > .form > .block {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 0 !important;
        max-width: none !important;
    }

    #assistant_chat_settings_panel .chat__session-action-buttons > input[type="file"] {
        display: none !important;
    }

    #assistant_chat_settings_panel .chat__session-action-buttons > .chat__template-tool-icon-btn {
        flex: 1 1 38px !important;
        width: auto !important;
        min-width: 38px !important;
    }

    #assistant_chat_settings_panel .chat__template-tool-actions {
        width: 100%;
        min-width: 0 !important;
        max-width: none !important;
        flex-direction: row;
    }

    #assistant_chat_settings_panel .chat__template-tool-actions > .form {
        width: 100%;
    }

    #assistant_chat_settings_panel .chat__template-tool-icon-btn {
        flex: 1 1 calc(50% - 4px);
    }

    #assistant_chat_settings_panel .chat__template-modal-wrap {
        inset: 0;
        padding: 8px !important;
    }

    #assistant_chat_settings_panel .chat__template-modal-wrap > .styler {
        width: 100% !important;
        max-width: none !important;
    }

    #assistant_chat_settings_panel .chat__template-modal-card {
        width: 100% !important;
        max-width: none !important;
    }
}
"""


def get_javascript() -> str:
    return r"""
window.__wangpAssistantChatNS = window.__wangpAssistantChatNS || {};
window.__wangpAssistantChatPending = window.__wangpAssistantChatPending || [];
const WAC = window.__wangpAssistantChatNS;
WAC.replayDepth = Number(WAC.replayDepth || 0);
window.WAC = WAC;

WAC.state = WAC.state || { order: [], messages: {}, status: null, stats: null };
WAC.blockState = WAC.blockState || {};
WAC.init = WAC.init || false;
WAC.observer = WAC.observer || null;
WAC.eventNode = WAC.eventNode || null;
WAC.pollTimer = WAC.pollTimer || null;
WAC.lastPayloadId = WAC.lastPayloadId || '';
WAC.lastPayloadText = WAC.lastPayloadText || '';
WAC.dockBridgeInstalled = WAC.dockBridgeInstalled || false;
WAC.dockOpen = typeof WAC.dockOpen === 'boolean' ? WAC.dockOpen : false;
WAC.settingsOpen = typeof WAC.settingsOpen === 'boolean' ? WAC.settingsOpen : false;
WAC.disclosureNode = WAC.disclosureNode || null;
WAC.disclosureState = WAC.disclosureState || {};
WAC.composerLayoutMode = WAC.composerLayoutMode || '';
WAC.composerResizeScrollState = WAC.composerResizeScrollState || null;
WAC.composerResizeFrame = WAC.composerResizeFrame || 0;
WAC.followSubmissionId = WAC.followSubmissionId || '';
WAC.followSubmissionScrollFrame = WAC.followSubmissionScrollFrame || 0;
WAC.sessionCatalog = Array.isArray(WAC.sessionCatalog) ? WAC.sessionCatalog : [];
WAC.activeSessionId = WAC.activeSessionId || '';
WAC.multiSessionEnabled = !!WAC.multiSessionEnabled;
WAC.sessionCatalogInitialized = !!WAC.sessionCatalogInitialized;

WAC.dock = function () {
  return document.querySelector('#assistant_chat_dock');
};

WAC.panel = function () {
  return document.querySelector('#assistant_chat_panel');
};

WAC.launcher = function () {
  return document.querySelector('#assistant_chat_toggle');
};

WAC.settingsPanel = function () {
  return document.querySelector('#assistant_chat_settings_panel');
};

WAC.settingsLauncher = function () {
  return document.querySelector('#assistant_chat_settings_toggle');
};

WAC.requestInput = function () {
  return document.querySelector('#assistant_chat_request textarea, #assistant_chat_request input');
};

WAC.resetComposerLayout = function () {
  const panel = WAC.panel();
  if (!panel) return;
  panel.classList.remove('has-fixed-composer-layout');
  panel.style.removeProperty('height');
  panel.style.removeProperty('--chat-request-max-height');
  panel.dataset.composerLayoutMode = '';
  WAC.composerLayoutMode = '';
};

WAC.syncComposerLayout = function () {
  if (WAC.replayDepth > 0) return;
  const dock = WAC.dock();
  const panel = WAC.panel();
  const shellBlock = document.querySelector('#assistant_chat_shell_block');
  const input = WAC.requestInput();
  if (!dock || !panel || !shellBlock || !input || !WAC.dockOpen || window.getComputedStyle(panel).display === 'none') return;
  const mode = window.innerWidth <= 900 ? 'mobile' : 'desktop';
  if (panel.dataset.composerLayoutMode === mode && panel.classList.contains('has-fixed-composer-layout')) return;
  WAC.resetComposerLayout();
  const panelRect = panel.getBoundingClientRect();
  const shellRect = shellBlock.getBoundingClientRect();
  const inputRect = input.getBoundingClientRect();
  if (panelRect.height <= 0 || shellRect.height <= 0 || inputRect.height <= 0) return;
  const historyMinHeight = parseFloat(window.getComputedStyle(dock).getPropertyValue('--chat-history-min-height')) || 112;
  const viewportLimit = Math.max(320, window.innerHeight - 36);
  const panelHeight = Math.min(panelRect.height, viewportLimit);
  const nonHistoryHeight = Math.max(0, panelRect.height - shellRect.height);
  const availableHistoryHeight = Math.max(historyMinHeight, panelHeight - nonHistoryHeight);
  const maxRequestHeight = Math.max(inputRect.height, inputRect.height + availableHistoryHeight - historyMinHeight);
  panel.style.height = `${panelHeight}px`;
  panel.style.setProperty('--chat-request-max-height', `${maxRequestHeight}px`);
  panel.dataset.composerLayoutMode = mode;
  panel.classList.add('has-fixed-composer-layout');
  WAC.composerLayoutMode = mode;
};

WAC.scheduleComposerLayout = function (scrollState) {
  if (scrollState) WAC.composerResizeScrollState = scrollState;
  else if (!WAC.composerResizeScrollState) WAC.composerResizeScrollState = WAC.captureAutoscrollState();
  if (WAC.composerResizeFrame) window.cancelAnimationFrame(WAC.composerResizeFrame);
  WAC.composerResizeFrame = window.requestAnimationFrame(() => {
    WAC.composerResizeFrame = window.requestAnimationFrame(() => {
      WAC.composerResizeFrame = 0;
      WAC.syncComposerLayout();
      const state = WAC.composerResizeScrollState;
      WAC.composerResizeScrollState = null;
      WAC.applyAutoscrollState(state);
    });
  });
};

WAC.escapeHtml = function (value) {
  return String(value || '').replace(/[&<>\"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[char] || char));
};

WAC.timeLabel = function (timestamp) {
  const value = Number(timestamp);
  return Number.isFinite(value) && value > 0 ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
};

WAC.bottomThreshold = function () {
  return 1;
};

WAC.captureAutoscrollState = function () {
  if (WAC.replayDepth > 0) return null;
  const scroll = WAC.scroll();
  if (!scroll) return { atBottom: true, top: 0 };
  return {
    atBottom: WAC.isNearBottom(),
    top: Math.max(0, scroll.scrollTop),
  };
};

WAC.applyAutoscrollState = function (state) {
  if (WAC.replayDepth > 0) return;
  const scroll = WAC.scroll();
  if (!scroll) return;
  if (state && state.atBottom) {
    scroll.scrollTop = scroll.scrollHeight;
    WAC.syncJumpToBottom();
    return;
  }
  if (!state) {
    WAC.syncJumpToBottom();
    return;
  }
  scroll.scrollTop = Math.max(0, Number(state.top || 0));
  WAC.syncJumpToBottom();
};

const optimisticProtocolVersion = 2;
if (WAC.optimisticProtocolVersion !== optimisticProtocolVersion) {
  WAC.optimisticSubmits = [];
  WAC.state.order = WAC.state.order.filter((messageId) => {
    const optimistic = String(messageId || '').startsWith('optimistic_');
    if (optimistic) delete WAC.state.messages[messageId];
    return !optimistic;
  });
  WAC.optimisticProtocolVersion = optimisticProtocolVersion;
}
WAC.optimisticSubmits = Array.isArray(WAC.optimisticSubmits) ? WAC.optimisticSubmits : [];
WAC.serverInstanceId = WAC.serverInstanceId || '';
WAC.chatSessionId = WAC.chatSessionId || '';
WAC.chatRevision = Number.isFinite(Number(WAC.chatRevision)) ? Number(WAC.chatRevision) : -1;
WAC.chatSequence = Number.isFinite(Number(WAC.chatSequence)) ? Number(WAC.chatSequence) : -1;
WAC.syncRequired = !!WAC.syncRequired;
WAC.syncRecoveryPending = !!WAC.syncRecoveryPending;
WAC.optimisticMaxAgeMs = 30000;
WAC.pendingSteeringId = WAC.pendingSteeringId || '';
WAC.queuedEditMessageId = WAC.queuedEditMessageId || '';
WAC.queuedEditDraft = typeof WAC.queuedEditDraft === 'string' ? WAC.queuedEditDraft : '';

WAC.normalizeText = function (value) {
  return String(value || '').replace(/\r\n?/g, '\n').replace(/\u00a0/g, ' ').trim();
};

WAC.gradioConfig = function () {
  return window.gradio_config || window.__gradio_config__ || null;
};

WAC.componentNode = function (id) {
  if (id === null || typeof id === 'undefined') return null;
  return document.getElementById(`component-${id}`);
};

WAC.isVisibleNode = function (node) {
  if (!node) return false;
  const style = window.getComputedStyle(node);
  if (style.display === 'none' || style.visibility === 'hidden') return false;
  const rect = node.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};

WAC.dropdownChoiceTexts = function (component) {
  const rawChoices = component && component.props ? component.props.choices : [];
  if (!Array.isArray(rawChoices)) return [];
  const texts = [];
  for (const choice of rawChoices) {
    if (Array.isArray(choice)) {
      texts.push(String(choice[0] || '').toLowerCase());
      texts.push(String(choice[1] || '').toLowerCase());
      continue;
    }
    if (choice && typeof choice === 'object') {
      texts.push(String(choice.label || choice.name || '').toLowerCase());
      texts.push(String(choice.value || '').toLowerCase());
      continue;
    }
    texts.push(String(choice || '').toLowerCase());
  }
  return texts;
};

WAC.findWanGpSettingsDropdown = function () {
  const cfg = WAC.gradioConfig();
  const components = cfg && Array.isArray(cfg.components) ? cfg.components : [];
  let fallback = null;
  for (const component of components) {
    if (!component || String(component.type || '').toLowerCase() !== 'dropdown') continue;
    const texts = WAC.dropdownChoiceTexts(component);
    const hasSettings = texts.some((text) => text.includes('>settings'));
    const hasProfiles = texts.some((text) => text.includes('>profiles'));
    const hasLoraPresetHint = texts.some((text) => text.includes('lora preset'));
    if (!hasSettings || (!hasProfiles && !hasLoraPresetHint)) continue;
    const node = WAC.componentNode(component.id);
    if (node && WAC.isVisibleNode(node)) return { component, node };
    if (!fallback) fallback = { component, node };
  }
  return fallback;
};

WAC.getWanGpSettingsSelection = function () {
  const located = WAC.findWanGpSettingsDropdown();
  if (!located || !located.component) return { value: '', label: '' };
  const component = located.component;
  const node = located.node || WAC.componentNode(component.id);
  const input = node ? node.querySelector('input[role="listbox"], input, textarea') : null;
  const label = WAC.normalizeText(input ? (input.value || input.getAttribute('value') || '') : '');
  const value = WAC.normalizeText(component && component.props ? component.props.value : '');
  return { value, label };
};

WAC.buildOptimisticUserMessage = function (optimisticId, content, timestamp, badgeText) {
  const contentHtml = WAC.escapeHtml(content).replace(/\n/g, '<br>');
  const badge = WAC.normalizeText(badgeText);
  const badgeHtml = badge ? `<span class='chat__badge'>${WAC.escapeHtml(badge)}</span>` : '';
  const copyButton = `<button type='button' class='chat__copy-button' data-copy-source='user' data-copy-text='${WAC.escapeHtml(content)}' aria-label='Copy request' title='Copy request'><svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><rect x='5' y='5' width='8' height='8' rx='1.5'></rect><path d='M3.5 10.5H3A1.5 1.5 0 0 1 1.5 9V3A1.5 1.5 0 0 1 3 1.5h6A1.5 1.5 0 0 1 10.5 3v.5'></path></svg></button>`;
  const queuedActions = badge === 'Queued' ? `<button type='button' class='chat__message-action-button' data-message-action='steer' aria-label='Steer with this queued request' title='Steer with this queued request'><svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M2.5 8h9M8.5 4l4 4-4 4'></path></svg></button><button type='button' class='chat__message-action-button' data-message-action='edit' aria-label='Edit queued request' title='Edit queued request'><svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M3 11.8 3.5 9l6.8-6.8a1.4 1.4 0 0 1 2 0l1.5 1.5a1.4 1.4 0 0 1 0 2L7 12.5l-2.8.5Z'></path><path d='m9.4 3.1 3.5 3.5'></path></svg></button><button type='button' class='chat__message-action-button' data-message-action='remove' aria-label='Remove queued request' title='Remove queued request'><svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M3 4.5h10M6 4.5V2.7h4v1.8M4.7 4.5l.6 8.3h5.4l.6-8.3M7 7v3.4M9 7v3.4'></path></svg></button>` : '';
  const html = [
    `<article class='chat__message chat__message--user' data-message-id='${optimisticId}'>`,
    "<div class='chat__avatar'>You</div>",
    "<div class='chat__message-card'>",
    `<div class='chat__meta'><div class='chat__meta-left'>${badgeHtml}</div>`,
    `<div class='chat__meta-right'><div class='chat__message-actions'>${copyButton}${queuedActions}</div><div class='chat__time'>${WAC.escapeHtml(WAC.timeLabel(timestamp))}</div></div></div>`,
    `<div class='chat__body'><p>${contentHtml}</p></div>`,
    "</div></article>",
  ].join('');
  return { id: optimisticId, role: 'user', html, badge, queued: badge === 'Queued' || badge === 'Steered', client_submission_id: optimisticId };
};

WAC.dropOptimisticSubmit = function (optimisticId) {
  const targetId = String(optimisticId || '');
  WAC.optimisticSubmits = (WAC.optimisticSubmits || []).filter((item) => String(item && item.id || '') !== targetId);
  if (WAC.followSubmissionId === targetId) WAC.followSubmissionId = '';
};

WAC.queuedTailInsertIndex = function () {
  let index = WAC.state.order.length;
  while (index > 0) {
    const message = WAC.state.messages[WAC.state.order[index - 1]];
    if (!message || message.role !== 'user' || !message.queued) break;
    index -= 1;
  }
  return index;
};

WAC.clearRequestInput = function (expectedText) {
  const input = WAC.requestInput();
  if (!input) return;
  const current = WAC.normalizeText(input.value || '');
  const expected = WAC.normalizeText(expectedText || '');
  if (expected && current && current !== expected) return;
  input.value = '';
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
};

WAC.reconcileOptimisticSubmits = function (acknowledgedSubmissionIds) {
  const acknowledged = new Set((Array.isArray(acknowledgedSubmissionIds) ? acknowledgedSubmissionIds : []).map((value) => String(value || '').trim()).filter(Boolean));
  for (const messageId of WAC.state.order) {
    const message = WAC.state.messages[messageId];
    const submissionId = String(message && message.client_submission_id || '').trim();
    if (submissionId) acknowledged.add(submissionId);
  }
  const now = Date.now();
  WAC.optimisticSubmits = (Array.isArray(WAC.optimisticSubmits) ? WAC.optimisticSubmits : []).filter((item) => {
    const submissionId = String(item && item.id || '').trim();
    const timestamp = Number(item && item.ts || 0);
    return submissionId && !acknowledged.has(submissionId) && timestamp > 0 && now - timestamp < WAC.optimisticMaxAgeMs;
  });
  for (const item of WAC.optimisticSubmits) {
    const optimisticId = String(item && item.id || '').trim();
    const content = WAC.normalizeText(item && item.text || '');
    if (!optimisticId || !content || WAC.state.messages[optimisticId]) continue;
    const message = WAC.buildOptimisticUserMessage(optimisticId, content, item.ts, item.badge);
    const messageIndex = message.badge === 'Steered' ? WAC.queuedTailInsertIndex() : WAC.state.order.length;
    WAC.state.order.splice(messageIndex, 0, optimisticId);
    WAC.state.messages[optimisticId] = message;
  }
};

WAC.newSubmissionId = function () {
  const randomId = window.crypto && typeof window.crypto.randomUUID === 'function' ? window.crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`;
  return `optimistic_${randomId}`;
};

WAC.pushOptimisticUserMessage = function (text, badgeText) {
  const content = WAC.normalizeText(text);
  if (!content) return '';
  const now = Date.now();
  const optimisticId = WAC.newSubmissionId();
  const badge = WAC.normalizeText(badgeText);
  WAC.followSubmissionId = optimisticId;
  WAC.optimisticSubmits.push({ id: optimisticId, text: content, ts: now, badge });
  const message = WAC.buildOptimisticUserMessage(optimisticId, content, now, badge);
  WAC.upsertMessage(message, false, badge === 'Steered' ? WAC.queuedTailInsertIndex() : undefined);
  WAC.scrollToBottomAfterLayout();
  window.setTimeout(() => {
    if (!(WAC.optimisticSubmits || []).some((item) => String(item && item.id || '') === optimisticId)) return;
    WAC.dropOptimisticSubmit(optimisticId);
    WAC.removeMessage(optimisticId);
  }, WAC.optimisticMaxAgeMs);
  return optimisticId;
};

WAC.host = function () {
  return document.querySelector('#assistant_chat_html');
};

WAC.shell = function () {
  return document.querySelector('#assistant_chat_html .chat');
};

WAC.scroll = function () {
  return document.querySelector('#assistant_chat_html .chat__scroll');
};

WAC.transcript = function () {
  return document.querySelector('#assistant_chat_html .chat__transcript');
};

WAC.empty = function () {
  return document.querySelector('#assistant_chat_html .chat__empty');
};

WAC.acknowledgeOptimisticSubmits = function (submissionIds) {
  const acknowledged = new Set((Array.isArray(submissionIds) ? submissionIds : []).map((value) => String(value || '').trim()).filter(Boolean));
  if (acknowledged.size === 0) return;
  WAC.optimisticSubmits = (WAC.optimisticSubmits || []).filter((item) => !acknowledged.has(String(item && item.id || '').trim()));
  for (const submissionId of acknowledged) delete WAC.state.messages[submissionId];
  WAC.state.order = WAC.state.order.filter((messageId) => !acknowledged.has(String(messageId || '')));
};

WAC.mergeStaleSync = function (messages, acknowledgedSubmissionIds) {
  WAC.ensureShell();
  const followSubmittedRequest = WAC.syncAcknowledgesFollowedSubmission(messages, acknowledgedSubmissionIds);
  const scrollState = followSubmittedRequest ? { atBottom: true, top: 0 } : WAC.captureAutoscrollState();
  WAC.acknowledgeOptimisticSubmits(acknowledgedSubmissionIds);
  const snapshotOrder = [];
  for (const message of (Array.isArray(messages) ? messages : [])) {
    if (!message || !message.id) continue;
    const messageId = String(message.id);
    snapshotOrder.push(messageId);
    if (!WAC.state.messages[messageId]) WAC.state.messages[messageId] = message;
  }
  const snapshotIds = new Set(snapshotOrder);
  WAC.state.order = snapshotOrder.concat(WAC.state.order.filter((messageId) => !snapshotIds.has(String(messageId))));
  WAC.hydrate(scrollState);
  if (followSubmittedRequest) {
    WAC.followSubmissionId = '';
    WAC.scrollToBottomAfterLayout();
  }
};

WAC.readSessionCatalogFromHost = function () {
  if (WAC.sessionCatalogInitialized) return;
  const host = WAC.host();
  if (!host) return;
  try {
    const catalog = JSON.parse(String(host.dataset.sessionCatalog || '[]'));
    if (Array.isArray(catalog)) WAC.sessionCatalog = catalog;
  } catch (_error) {}
  WAC.activeSessionId = String(host.dataset.activeSessionId || WAC.activeSessionId || '');
  WAC.multiSessionEnabled = String(host.dataset.multiSessionEnabled || '').toLowerCase() === 'true';
  WAC.sessionCatalogInitialized = true;
};

WAC.syncSessionActionTooltips = function () {
  const labels = {
    assistant_chat_session_resume_button: 'Resume the selected session',
    assistant_chat_session_rename_button: 'Rename the selected session',
    assistant_chat_session_duplicate_button: 'Duplicate the selected session',
    assistant_chat_session_export_button: 'Export the selected session',
    assistant_chat_session_import_button: 'Import a session archive',
    assistant_chat_session_delete_button: 'Delete the selected session',
  };
  for (const [id, label] of Object.entries(labels)) {
    const host = document.getElementById(id);
    if (!host) continue;
    const button = host.matches && host.matches('button') ? host : host.querySelector('button');
    if (!button) continue;
    button.title = label;
    button.setAttribute('aria-label', label);
  }
};

WAC.sessionPickerMarkup = function () {
  if (!WAC.multiSessionEnabled) return '';
  const options = [];
  for (const item of WAC.sessionCatalog) {
    const id = String(item && item.id || '').trim();
    if (!id) continue;
    const title = String(item && item.title || 'Deepy session').trim();
    const selected = id === WAC.activeSessionId ? ' selected' : '';
    options.push(`<option value="${WAC.escapeHtml(id)}"${selected}>${WAC.escapeHtml(title)}</option>`);
  }
  const disabled = options.length === 0 ? ' disabled' : '';
  const choices = options.length ? options.join('') : '<option value="">No saved sessions</option>';
  return `<div class="chat__session-picker"><div class="chat__session-picker-spacer" aria-hidden="true"></div><label><span>Saved sessions</span><span class="chat__session-picker-controls"><select aria-label="Deepy session" data-wac-session-picker${disabled}>${choices}</select><button type="button" data-wac-session-resume aria-label="Resume selected session" title="Resume selected session"${disabled}>Resume</button></span></label></div>`;
};

WAC.resumeSelectedSession = function (trigger) {
  const container = trigger && trigger.closest ? trigger.closest('.chat__session-picker') : null;
  const picker = container ? container.querySelector('[data-wac-session-picker]') : null;
  const storageId = String(picker && picker.value || '').trim();
  const bridgeHost = document.getElementById('assistant_chat_welcome_session_button');
  const bridge = bridgeHost && bridgeHost.matches && bridgeHost.matches('button') ? bridgeHost : bridgeHost && bridgeHost.querySelector('button');
  if (!storageId || !bridge || typeof bridge.click !== 'function') return false;
  if (!WAC.setBridgeValue('#assistant_chat_welcome_session_input textarea, #assistant_chat_welcome_session_input input', storageId)) return false;
  window.setTimeout(() => bridge.click(), 0);
  return true;
};

WAC.prefillResumedSession = function (requestId) {
  const normalizedRequestId = String(requestId || '').trim();
  if (!normalizedRequestId || normalizedRequestId === WAC.lastSessionResumeRequestId) return false;
  WAC.lastSessionResumeRequestId = normalizedRequestId;
  window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
    const bridgeHost = document.getElementById('assistant_chat_session_prefill_button');
    const bridge = bridgeHost && bridgeHost.matches && bridgeHost.matches('button') ? bridgeHost : bridgeHost && bridgeHost.querySelector('button');
    if (bridge && typeof bridge.click === 'function') bridge.click();
  }));
  return true;
};

WAC.bindSessionPickerControls = function (root) {
  const scope = root && root.querySelectorAll ? root : document;
  for (const button of scope.querySelectorAll('[data-wac-session-resume]')) {
    if (button.dataset.wacSessionResumeBound === 'true') continue;
    button.dataset.wacSessionResumeBound = 'true';
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      WAC.resumeSelectedSession(button);
    });
  }
};

WAC.refreshSessionPickers = function () {
  const markup = WAC.sessionPickerMarkup();
  const existing = document.querySelectorAll('.chat__session-picker');
  if (!markup) {
    for (const node of existing) node.remove();
    return;
  }
  if (existing.length) {
    for (const node of existing) node.outerHTML = markup;
    WAC.bindSessionPickerControls();
    return;
  }
  const card = document.querySelector('#assistant_chat_html .chat__empty-card');
  if (card) {
    card.insertAdjacentHTML('beforeend', markup);
    WAC.bindSessionPickerControls(card);
  }
};

WAC.emptyMarkup = function (mode) {
  if (mode === 'prime') {
    return `<div class="chat__empty-card">
      <header class="chat__empty-header">
        <span class="chat__empty-eyebrow">Current assistant</span>
        <h2 class="chat__empty-title">Deepy Prime</h2>
        <span class="chat__empty-mode">Advanced creative orchestration</span>
      </header>
      <p class="chat__empty-intro">Describe the result you want and Deepy Prime can plan the work, choose suitable models and tools, and connect multiple image, video, and audio steps into one creative workflow.</p>
      <div class="chat__empty-grid">
        <section class="chat__empty-section"><h3>What it does for you</h3><ul>
          <li>Plan and complete multi-step projects that create, inspect, edit, and combine several pieces of media.</li>
          <li>Choose among available WanGP models and settings according to your goal, quality preference, and source media.</li>
          <li>Build on Gallery items or existing files, then extract, transcribe, resize, add sound, upscale, or continue generating.</li>
          <li>Extend the workflow with other connected services when they are available.</li>
        </ul></section>
        <section class="chat__empty-section chat__empty-section--examples"><h3>Try asking</h3><ul>
          <li>Create a character portrait and related keyframes, then turn them into a longer video with a soundtrack.</li>
          <li>Inspect the selected video, improve the weak sections, upscale it, and prepare a subtitled version.</li>
          <li>Design an album cover, write a matching song, and create a short promotional video from both.</li>
        </ul></section>
      </div>
      <p class="chat__empty-tip">Start with the outcome you want. Deepy will ask only when an important choice is missing.</p>
      ${WAC.sessionPickerMarkup()}
    </div>`;
  }
  return `<div class="chat__empty-card">
    <header class="chat__empty-header">
      <span class="chat__empty-eyebrow">Current assistant</span>
      <h2 class="chat__empty-title">Deepy Zero</h2>
      <span class="chat__empty-mode">Fast, focused creation</span>
    </header>
    <p class="chat__empty-intro">Deepy Zero is the lightweight assistant for straightforward requests. It uses the models and templates selected in Deepy Settings, making it a good match for smaller LLMs, quick responses, and familiar results.</p>
    <div class="chat__empty-grid">
      <section class="chat__empty-section"><h3>What it does for you</h3><ul>
        <li>Generate an image, video, speech clip, or song with your preferred templates and defaults.</li>
        <li>Handle focused edits and practical media tasks without requiring a complex workflow.</li>
        <li>Refer naturally to the selected, latest, or previous Gallery item.</li>
        <li>See each generation and completed result in the normal WanGP queue and Galleries.</li>
      </ul></section>
      <section class="chat__empty-section chat__empty-section--examples"><h3>Try asking</h3><ul>
        <li>Generate a square album cover showing a robot jazz band.</li>
        <li>Animate the selected image as a five-second cinematic shot.</li>
        <li>Transcribe the last video or resize it for social media.</li>
      </ul></section>
    </div>
    <p class="chat__empty-tip">Start with the outcome you want. Deepy will ask only when an important choice is missing.</p>
    ${WAC.sessionPickerMarkup()}
  </div>`;
};

WAC.syncDeepyTypePreview = function () {
  const canonical = document.querySelector('#deepy_type_value');
  const value = String((canonical && canonical.textContent) || '').trim().toLowerCase();
  if (!value) return;
  const mode = value === 'prime' ? 'prime' : 'zero';
  const host = WAC.host();
  const empty = WAC.empty();
  if (!host || !empty || host.dataset.deepyType === mode) return;
  host.dataset.deepyType = mode;
  empty.innerHTML = WAC.emptyMarkup(mode);
};

WAC.statusNode = function () {
  return document.querySelector('#assistant_chat_html .chat__status');
};

WAC.jumpBottomNode = function () {
  return document.querySelector('#assistant_chat_html .chat__jump-bottom');
};

WAC.statsNode = function () {
  return document.getElementById('assistant_chat_stats');
};

WAC.disclosureKey = function (node) {
  if (!node || !node.getAttribute) return '';
  const reasoningId = String(node.getAttribute('data-reasoning-id') || '').trim();
  if (reasoningId) return `reasoning:${reasoningId}`;
  const toolId = String(node.getAttribute('data-tool-id') || '').trim();
  if (toolId) return `tool:${toolId}`;
  const contextSummaryId = String(node.getAttribute('data-context-summary-id') || '').trim();
  if (contextSummaryId) return `context-summary:${contextSummaryId}`;
  return '';
};

WAC.captureDisclosureState = function (root) {
  if (WAC.replayDepth > 0) return;
  const scope = root || WAC.transcript();
  if (!scope || !scope.querySelectorAll) return;
  const nodes = [...scope.querySelectorAll('.chat__disclosure')];
  if (scope.matches && scope.matches('.chat__disclosure')) nodes.unshift(scope);
  nodes.forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (!key) return;
    WAC.disclosureState[key] = !!node.open;
  });
};

WAC.applyDisclosureState = function (root) {
  if (WAC.replayDepth > 0) return;
  const scope = root || WAC.transcript();
  if (!scope || !scope.querySelectorAll) return;
  const nodes = [...scope.querySelectorAll('.chat__disclosure')];
  if (scope.matches && scope.matches('.chat__disclosure')) nodes.unshift(scope);
  nodes.forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (!key || !(key in WAC.disclosureState)) return;
    node.open = !!WAC.disclosureState[key];
  });
};

WAC.handleDisclosureToggle = function (event) {
  if (WAC.replayDepth > 0) return;
  const node = event && event.target;
  if (!node || !node.classList || !node.classList.contains('chat__disclosure')) return;
  const key = WAC.disclosureKey(node);
  if (!key) return;
  WAC.disclosureState[key] = !!node.open;
};

WAC.toggleDisclosure = function (node) {
  if (!node || !node.classList || !node.classList.contains('chat__disclosure')) return;
  const scrollState = WAC.captureAutoscrollState();
  node.open = !node.open;
  const key = WAC.disclosureKey(node);
  if (key) WAC.disclosureState[key] = !!node.open;
  WAC.applyAutoscrollState(scrollState);
};

WAC.closeDisclosure = function (node) {
  if (!node || !node.open) return;
  WAC.toggleDisclosure(node);
  const summary = node.querySelector(':scope > summary');
  if (summary && typeof summary.focus === 'function') summary.focus({ preventScroll: true });
};

WAC.markCopyButton = function (button, state) {
  if (!button) return;
  const originalLabel = String(button.dataset.copyLabel || button.getAttribute('aria-label') || 'Copy');
  button.dataset.copyLabel = originalLabel;
  button.classList.toggle('is-copied', state === 'copied');
  button.classList.toggle('is-copy-error', state === 'error');
  const label = state === 'copied' ? 'Copied' : state === 'error' ? 'Copy failed' : originalLabel;
  button.setAttribute('aria-label', label);
  button.setAttribute('title', label);
  if (button.copyResetTimer) window.clearTimeout(button.copyResetTimer);
  if (state) button.copyResetTimer = window.setTimeout(() => { WAC.markCopyButton(button, ''); }, 1200);
};

WAC.handleCopyButtonClick = function (event) {
  const button = event && event.target && event.target.closest ? event.target.closest('.chat__copy-button') : null;
  if (!button) return false;
  event.preventDefault();
  event.stopPropagation();
  const source = String(button.getAttribute('data-copy-source') || '');
  const jsonNode = source === 'json' ? button.closest('.chat__tool-json') : null;
  const pre = jsonNode ? jsonNode.querySelector('pre') : null;
  const text = source === 'json' ? String((pre && pre.textContent) || '') : String(button.getAttribute('data-copy-text') || '');
  if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
    WAC.markCopyButton(button, 'error');
    return true;
  }
  navigator.clipboard.writeText(text).then(() => { WAC.markCopyButton(button, 'copied'); }).catch(() => { WAC.markCopyButton(button, 'error'); });
  return true;
};

WAC.handleCollapseButtonClick = function (event) {
  const button = event && event.target && event.target.closest ? event.target.closest('[data-disclosure-action="collapse"]') : null;
  if (!button) return false;
  event.preventDefault();
  event.stopPropagation();
  if (button.dataset.collapsePointerHandled === 'true') {
    delete button.dataset.collapsePointerHandled;
    return true;
  }
  WAC.closeDisclosure(button.closest('.chat__disclosure'));
  return true;
};

WAC.handleCollapseButtonPointerDown = function (event) {
  const button = event && event.target && event.target.closest ? event.target.closest('[data-disclosure-action="collapse"]') : null;
  if (!button) return false;
  const isPrimaryPointer = event.button === 0 || event.pointerType === 'touch' || event.pointerType === 'pen';
  if (!isPrimaryPointer) return false;
  event.preventDefault();
  event.stopPropagation();
  button.dataset.collapsePointerHandled = 'true';
  WAC.closeDisclosure(button.closest('.chat__disclosure'));
  return true;
};

WAC.handleDisclosurePointerDown = function (event) {
  const summary = event && event.target && event.target.closest ? event.target.closest('summary') : null;
  if (!summary) return false;
  const disclosureNode = summary.parentElement;
  if (!disclosureNode || !disclosureNode.classList || !disclosureNode.classList.contains('chat__disclosure')) return false;
  event.preventDefault();
  event.stopPropagation();
  WAC.toggleDisclosure(disclosureNode);
  return true;
};

WAC.handleAttachmentPointerDown = function (event) {
  const link = event && event.target && event.target.closest ? event.target.closest('a.chat__attachment') : null;
  if (!link) return false;
  const isPrimaryPointer = event.button === 0 || event.pointerType === 'touch' || event.pointerType === 'pen';
  if (!isPrimaryPointer || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return false;
  const href = String(link.href || '').trim();
  if (!href) return false;
  event.preventDefault();
  event.stopPropagation();
  const target = String(link.target || '_blank').trim() || '_blank';
  if (target === '_blank') {
    const opened = window.open(href, '_blank', 'noopener');
    if (opened) opened.opener = null;
    return true;
  }
  window.location.assign(href);
  return true;
};

WAC.stopBridgeTargets = function () {
  const wrapper = document.querySelector('#assistant_chat_stop_bridge');
  if (!wrapper) return [];
  const targets = [wrapper];
  const button = wrapper.querySelector('button');
  if (button) targets.unshift(button);
  return targets.filter((target, index, items) => !!target && items.indexOf(target) === index);
};

WAC.pauseBridgeTargets = function () {
  const wrapper = document.querySelector('#assistant_chat_pause_bridge');
  if (!wrapper) return [];
  const targets = [wrapper];
  const button = wrapper.querySelector('button');
  if (button) targets.unshift(button);
  return targets.filter((target, index, items) => !!target && items.indexOf(target) === index);
};

WAC.requestCanonicalSync = function () {
  const button = document.querySelector('#assistant_chat_sync_button button, #assistant_chat_sync_button');
  if (!button || typeof button.click !== 'function') return false;
  button.click();
  return true;
};

WAC.setBridgeValue = function (selector, value) {
  const input = document.querySelector(selector);
  if (!input) return false;
  input.value = String(value || '');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
};

WAC.queueBusyRequest = function (text, submissionId) {
  const input = document.querySelector('#assistant_chat_busy_queue_input textarea, #assistant_chat_busy_queue_input input');
  const button = document.querySelector('#assistant_chat_busy_queue_button button, #assistant_chat_busy_queue_button');
  if (!input || !button) return false;
  WAC.setBridgeValue('#assistant_chat_busy_queue_input textarea, #assistant_chat_busy_queue_input input', text);
  WAC.setBridgeValue('#assistant_chat_busy_queue_submission_id textarea, #assistant_chat_busy_queue_submission_id input', submissionId);
  if (typeof button.click === 'function') button.click();
  return true;
};

WAC.steerRequest = function (text, submissionId) {
  const button = document.querySelector('#assistant_chat_steer_button button, #assistant_chat_steer_button');
  if (!button) return false;
  WAC.setBridgeValue('#assistant_chat_steer_input textarea, #assistant_chat_steer_input input', text);
  WAC.setBridgeValue('#assistant_chat_steer_submission_id textarea, #assistant_chat_steer_submission_id input', submissionId);
  if (typeof button.click === 'function') button.click();
  return true;
};

WAC.queuedRequestAction = function (action, messageId, text) {
  const button = document.querySelector('#assistant_chat_queued_action_button button, #assistant_chat_queued_action_button');
  if (!button) return false;
  const payload = JSON.stringify({ action: String(action || ''), message_id: String(messageId || ''), text: String(text || '') });
  WAC.setBridgeValue('#assistant_chat_queued_action_input textarea, #assistant_chat_queued_action_input input', payload);
  if (typeof button.click === 'function') button.click();
  return true;
};

WAC.setRequestInputValue = function (value) {
  const input = WAC.requestInput();
  if (!input) return;
  input.value = String(value || '');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
};

WAC.setQueuedEditButtonLabels = function (editing) {
  const askButton = document.querySelector('#assistant_chat_ask_button button, #assistant_chat_ask_button');
  const resetButton = document.querySelector('#assistant_chat_reset_button button, #assistant_chat_reset_button');
  if (askButton) askButton.textContent = editing ? 'Save' : 'Ask';
  if (resetButton) {
    if (editing) {
      if (resetButton.textContent !== 'Cancel') resetButton.dataset.idleLabel = resetButton.textContent || 'Reset';
      resetButton.textContent = 'Cancel';
    } else if (resetButton.textContent === 'Cancel') {
      resetButton.textContent = resetButton.dataset.idleLabel || 'Reset';
    }
  }
};

WAC.finishQueuedRequestEdit = function () {
  const messageId = String(WAC.queuedEditMessageId || '');
  if (messageId) {
    const messageNode = WAC.transcript() && WAC.transcript().querySelector(`[data-message-id="${CSS.escape(messageId)}"]`);
    if (messageNode) messageNode.classList.remove('is-editing');
  }
  WAC.setRequestInputValue(WAC.queuedEditDraft);
  WAC.queuedEditMessageId = '';
  WAC.queuedEditDraft = '';
  WAC.setQueuedEditButtonLabels(false);
};

WAC.startQueuedRequestEdit = function (messageNode) {
  if (!messageNode) return;
  const copyButton = messageNode.querySelector('[data-copy-source="user"]');
  const input = WAC.requestInput();
  if (!copyButton || !input || !messageNode.querySelector('[data-message-action="edit"]')) return;
  if (WAC.queuedEditMessageId) WAC.finishQueuedRequestEdit();
  WAC.queuedEditMessageId = String(messageNode.getAttribute('data-message-id') || '');
  WAC.queuedEditDraft = String(input.value || '');
  const text = String(copyButton.getAttribute('data-copy-text') || '');
  messageNode.classList.add('is-editing');
  WAC.setRequestInputValue(text);
  WAC.setQueuedEditButtonLabels(true);
  input.focus({ preventScroll: true });
  input.setSelectionRange(input.value.length, input.value.length);
};

WAC.syncQueuedRequestEdit = function () {
  if (WAC.replayDepth > 0) return;
  const messageId = String(WAC.queuedEditMessageId || '');
  if (!messageId) return;
  const messageNode = WAC.transcript() && WAC.transcript().querySelector(`[data-message-id="${CSS.escape(messageId)}"]`);
  if (!messageNode || !messageNode.querySelector('[data-message-action="edit"]')) {
    WAC.finishQueuedRequestEdit();
    return;
  }
  messageNode.classList.add('is-editing');
  WAC.setQueuedEditButtonLabels(true);
};

WAC.submitQueuedRequestAction = function (messageNode, action, text) {
  if (!messageNode) return;
  const messageId = String(messageNode.getAttribute('data-message-id') || '');
  const normalizedAction = String(action || '').trim().toLowerCase();
  const content = String(text || '').trim();
  if (!messageId || !['edit', 'remove', 'steer'].includes(normalizedAction) || (normalizedAction === 'edit' && !content)) return;
  if (WAC.queuedEditMessageId === messageId) WAC.finishQueuedRequestEdit();
  messageNode.classList.add('is-pending-queue-action');
  if (!WAC.queuedRequestAction(normalizedAction, messageId, content)) {
    messageNode.classList.remove('is-pending-queue-action');
    return;
  }
  window.setTimeout(() => {
    if (messageNode.isConnected) messageNode.classList.remove('is-pending-queue-action');
  }, 3000);
};

WAC.handleQueuedRequestClick = function (event) {
  const messageAction = event && event.target && event.target.closest ? event.target.closest('[data-message-action]') : null;
  if (!messageAction) return false;
  const messageNode = messageAction.closest('.chat__message--user');
  if (!messageNode) return false;
  event.preventDefault();
  event.stopPropagation();
  const action = String(messageAction.getAttribute('data-message-action') || '');
  if (action === 'edit') WAC.startQueuedRequestEdit(messageNode);
  else if (action === 'remove') WAC.submitQueuedRequestAction(messageNode, 'remove', '');
  else if (action === 'steer') WAC.submitQueuedRequestAction(messageNode, 'steer', '');
  return true;
};

WAC.isAssistantBusy = function () {
  if (WAC.state && WAC.state.status && WAC.state.status.visible && WAC.state.status.text) return true;
  const stopButton = document.querySelector('#assistant_chat_html .chat__status-stop');
  return !!(stopButton && !stopButton.disabled);
};

WAC.setBusyInputHelper = function (visible) {
  const stats = WAC.statsNode();
  if (!stats) return;
  let helper = stats.querySelector('.chat__input-helper');
  if (!helper) {
    helper = document.createElement('span');
    helper.className = 'chat__input-helper';
    helper.textContent = 'Press Enter to Queue Requests / CTRL Enter to Steer Deepy';
    stats.prepend(helper);
  }
  helper.classList.toggle('is-visible', !!visible);
  helper.setAttribute('aria-hidden', visible ? 'false' : 'true');
  stats.classList.toggle('has-input-helper', !!visible);
  stats.setAttribute('aria-hidden', visible || stats.classList.contains('is-visible') ? 'false' : 'true');
};

WAC.eventSource = function () {
  return document.querySelector('#assistant_chat_event textarea, #assistant_chat_event input');
};

WAC.consumePayload = function (payload) {
  if (!payload) return [];
  let envelope = payload;
  if (typeof payload === 'string') {
    if (payload === WAC.lastPayloadText) return [];
    try {
      envelope = JSON.parse(payload);
    } catch (_error) {
      return [];
    }
  }
  const payloadId = envelope && typeof envelope.event_id === 'string' ? envelope.event_id : '';
  const payloadText = typeof payload === 'string' ? payload : (payloadId ? '' : JSON.stringify(envelope));
  if ((payloadId && payloadId === WAC.lastPayloadId) || (!payloadId && payloadText === WAC.lastPayloadText)) return [];
  WAC.lastPayloadId = payloadId;
  WAC.lastPayloadText = payloadText;
  if (Array.isArray(envelope.batch)) {
    const replaying = !!envelope.replay;
    let transcript = null;
    let shell = null;
    let previousVisibility = '';
    if (replaying) {
      WAC.ensureShell();
      shell = typeof WAC.shell === 'function' ? WAC.shell() : null;
      transcript = WAC.transcript();
      previousVisibility = transcript ? transcript.style.visibility : '';
      if (transcript) transcript.style.visibility = 'hidden';
      if (shell) {
        shell.classList.add('is-replaying');
        shell.setAttribute('aria-busy', 'true');
      }
      WAC.replayDepth += 1;
    }
    try {
      for (const item of envelope.batch) WAC.consumePayload(item);
    } finally {
      if (replaying) {
        WAC.replayDepth = Math.max(0, WAC.replayDepth - 1);
        transcript = WAC.transcript() || transcript;
        WAC.applyDisclosureState(transcript);
        WAC.showEmptyIfNeeded();
        const scroll = WAC.scroll();
        if (scroll) scroll.scrollTop = scroll.scrollHeight;
        WAC.syncJumpToBottom();
        if (transcript) transcript.style.visibility = previousVisibility;
        if (shell) {
          shell.classList.remove('is-replaying');
          shell.removeAttribute('aria-busy');
        }
      }
    }
    WAC.lastPayloadId = payloadId;
    WAC.lastPayloadText = payloadText;
    return [];
  }
  const instanceId = envelope && typeof envelope.instance_id === 'string' ? envelope.instance_id : '';
  if (instanceId) {
    if (WAC.serverInstanceId && WAC.serverInstanceId !== instanceId) {
      WAC.reset();
    }
    WAC.serverInstanceId = instanceId;
  }
  const event = envelope && envelope.event ? envelope.event : envelope;
  if (!event || typeof event !== 'object') return [];
  if (event.type === 'session_catalog') {
    WAC.sessionCatalog = Array.isArray(event.sessions) ? event.sessions : [];
    WAC.activeSessionId = String(event.active_session_id || '');
    WAC.multiSessionEnabled = !!event.multi_session_enabled;
    WAC.refreshSessionPickers();
    return [];
  }
  const chatSessionId = typeof event.chat_session_id === 'string' ? event.chat_session_id : '';
  if (chatSessionId && WAC.chatSessionId && chatSessionId !== WAC.chatSessionId) WAC.reset();
  if (chatSessionId) WAC.chatSessionId = chatSessionId;
  if (event.type === 'session_resume_ready') {
    WAC.prefillResumedSession(event.request_id);
    return [];
  }
  const blockEventTypes = ['upsert_block', 'append_block_text', 'replace_block_text', 'finalize_block', 'remove_block'];
  const transcriptEvent = event.type === 'sync' || event.type === 'upsert_message' || event.type === 'remove_message' || blockEventTypes.includes(event.type);
  const revision = Number(event.revision);
  const hasRevision = transcriptEvent && Number.isFinite(revision);
  const sequence = Number(event.sequence);
  const sequenceStart = Number.isFinite(Number(event.sequence_start)) ? Number(event.sequence_start) : sequence;
  const hasSequence = transcriptEvent && Number.isFinite(sequence);
  const acknowledgedSubmissionIds = event.type === 'sync' ? (event.acknowledged_submission_ids || []) : [];
  const canonicalSubmissionIds = event.type === 'sync' ? (event.messages || []).map((message) => String(message && message.client_submission_id || '').trim()).filter(Boolean) : [];
  const steeringAcknowledged = !!WAC.pendingSteeringId && [...acknowledgedSubmissionIds, ...canonicalSubmissionIds].some((submissionId) => String(submissionId || '').trim() === WAC.pendingSteeringId);
  if (hasRevision && revision < WAC.chatRevision) {
    if (event.type === 'sync') {
      WAC.mergeStaleSync(event.messages || [], acknowledgedSubmissionIds);
      if (steeringAcknowledged) {
        WAC.pendingSteeringId = '';
        WAC.setStatus(event.status || null);
      }
    }
    return [];
  }
  if (event.type !== 'sync' && hasSequence) {
    if (sequence <= WAC.chatSequence) return [];
    if (WAC.chatSequence >= 0 && sequenceStart > WAC.chatSequence + 1) {
      if (event.type !== 'upsert_block' && event.type !== 'finalize_block') {
        WAC.markSyncRequired(event);
        return [];
      }
    }
    WAC.chatSequence = sequence;
  }
  if (event.type === 'sync' && hasSequence) {
    WAC.chatSequence = sequence;
    WAC.syncRequired = false;
    WAC.syncRecoveryPending = false;
  }
  if (hasRevision) WAC.chatRevision = revision;
  if (event.type === 'reset') {
    WAC.reset();
    if (chatSessionId) WAC.chatSessionId = chatSessionId;
    if (Number.isFinite(revision)) WAC.chatRevision = revision;
    if (Number.isFinite(sequence)) WAC.chatSequence = sequence;
    return [];
  }
  if (event.type === 'upsert_message') {
    const message = event.message || {};
    const submissionId = String(message.client_submission_id || '').trim();
    const followSubmittedRequest = !!submissionId && submissionId === WAC.followSubmissionId;
    if (submissionId) {
      WAC.acknowledgeOptimisticSubmits([submissionId]);
      if (submissionId === WAC.pendingSteeringId) {
        WAC.pendingSteeringId = '';
        WAC.setStatus({ visible: true, kind: 'queued', text: 'Steering accepted. Waiting for the current thought/action boundary...' });
      }
    }
    WAC.upsertMessage(message, false, event.message_index);
    if (followSubmittedRequest) WAC.scrollToBottomAfterLayout();
    return [];
  }
  if (event.type === 'remove_message') {
    WAC.removeMessage(event.message_id);
    return [];
  }
  if (event.type === 'upsert_block') {
    WAC.upsertBlock(event);
    return [];
  }
  if (event.type === 'append_block_text') {
    WAC.appendBlockText(event);
    return [];
  }
  if (event.type === 'replace_block_text') {
    WAC.replaceBlockText(event);
    return [];
  }
  if (event.type === 'finalize_block') {
    WAC.finalizeBlock(event);
    return [];
  }
  if (event.type === 'remove_block') {
    WAC.removeBlock(event);
    return [];
  }
  if (event.type === 'status') {
    if (WAC.pendingSteeringId) return [];
    WAC.setStatus(event.status || null);
    if (Object.prototype.hasOwnProperty.call(event, 'stats')) WAC.setStats(event.stats || null);
    return [];
  }
  if (event.type === 'stats') {
    WAC.setStats(event.stats || null);
    return [];
  }
  if (event.type === 'sync') {
    const status = WAC.pendingSteeringId && !steeringAcknowledged ? WAC.state.status : (event.status || null);
    if (steeringAcknowledged) WAC.pendingSteeringId = '';
    WAC.sync(event.messages || [], status, Object.prototype.hasOwnProperty.call(event, 'stats') ? (event.stats || null) : WAC.state.stats, acknowledgedSubmissionIds);
    return [];
  }
  return [];
};

WAC.readEventSource = function () {
  const node = WAC.eventSource();
  if (!node) return;
  const value = typeof node.value === 'string' ? node.value.trim() : '';
  if (!value) return;
  WAC.consumePayload(value);
};

WAC.observeEventSourceValue = function (node) {
  if (!node || node.__wangpAssistantEventValueObserved) return;
  let prototype = node;
  let descriptor = null;
  while ((prototype = Object.getPrototypeOf(prototype))) {
    descriptor = Object.getOwnPropertyDescriptor(prototype, 'value');
    if (descriptor && typeof descriptor.get === 'function' && typeof descriptor.set === 'function') break;
  }
  if (!descriptor) return;
  try {
    Object.defineProperty(node, 'value', {
      configurable: true,
      enumerable: descriptor.enumerable,
      get() { return descriptor.get.call(this); },
      set(value) {
        descriptor.set.call(this, value);
        WAC.consumePayload(String(descriptor.get.call(this) || '').trim());
      },
    });
    Object.defineProperty(node, '__wangpAssistantEventValueObserved', { configurable: true, value: true });
  } catch (_error) {}
};

WAC.handleEventNodeMutation = function () {
  const node = WAC.eventSource();
  if (!node) return;
  WAC.observeEventSourceValue(node);
  if (node === WAC.eventNode) return;
  if (WAC.eventNode && WAC.eventNodeHandler) {
    WAC.eventNode.removeEventListener('input', WAC.eventNodeHandler, true);
    WAC.eventNode.removeEventListener('change', WAC.eventNodeHandler, true);
  }
  WAC.eventNode = node;
  const handler = function () { WAC.readEventSource(); };
  WAC.eventNodeHandler = handler;
  node.addEventListener('input', handler, true);
  node.addEventListener('change', handler, true);
  setTimeout(handler, 0);
};

WAC.replaceState = function (messages, status, stats) {
  const nextState = { order: [], messages: {}, status: status || null, stats: typeof stats === 'undefined' ? (WAC.state ? (WAC.state.stats || null) : null) : (stats || null) };
  const items = Array.isArray(messages) ? messages : [];
  for (const message of items) {
    if (!message || !message.id) continue;
    const key = String(message.id);
    nextState.order.push(key);
    nextState.messages[key] = message;
  }
  WAC.state = nextState;
};

WAC.syncDockVisibility = function () {
  document.querySelectorAll('#assistant_chat_dock').forEach((dock) => {
    const hasLauncher = !!dock.querySelector('#assistant_chat_toggle');
    dock.style.display = hasLauncher ? 'flex' : 'none';
  });
};

WAC.parseThemeColor = function (value) {
  const match = String(value || '').trim().match(/^rgba?\(([^)]+)\)$/i);
  if (!match) return null;
  const parts = match[1].split(',').map((part) => parseFloat(part.trim()));
  if (parts.length < 3 || parts.slice(0, 3).some((part) => !Number.isFinite(part))) return null;
  const alpha = Number.isFinite(parts[3]) ? parts[3] : 1;
  if (alpha <= 0.01) return null;
  return { r: parts[0], g: parts[1], b: parts[2], a: alpha };
};

WAC.resolveThemeBackground = function (node) {
  let current = node;
  while (current) {
    const resolved = WAC.parseThemeColor(window.getComputedStyle(current).backgroundColor);
    if (resolved) return resolved;
    current = current.parentElement;
  }
  return WAC.parseThemeColor(window.getComputedStyle(document.body).backgroundColor);
};

WAC.relativeLuminance = function (rgb) {
  if (!rgb) return 1;
  const normalize = function (value) {
    const channel = Math.max(0, Math.min(255, Number(value || 0))) / 255;
    return channel <= 0.03928 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * normalize(rgb.r) + 0.7152 * normalize(rgb.g) + 0.0722 * normalize(rgb.b);
};

WAC.isDarkTheme = function () {
  const nodes = [
    document.querySelector('.gradio-container'),
    document.body,
    document.documentElement,
    document.querySelector('gradio-app'),
  ].filter(Boolean);
  if (nodes.some((node) => node.classList && node.classList.contains('dark'))) return true;
  if (nodes.some((node) => String(node.getAttribute('data-theme') || node.getAttribute('theme') || '').toLowerCase().includes('dark'))) return true;
  const sample = document.querySelector('.gradio-container') || document.body;
  const background = WAC.resolveThemeBackground(sample);
  const foreground = WAC.parseThemeColor(window.getComputedStyle(sample).color) || WAC.parseThemeColor(window.getComputedStyle(document.body).color);
  const backgroundLuminance = WAC.relativeLuminance(background);
  const foregroundLuminance = WAC.relativeLuminance(foreground);
  return backgroundLuminance < 0.18 || (foreground && backgroundLuminance < foregroundLuminance);
};

WAC.syncThemeState = function () {
  const dock = WAC.dock();
  if (!dock) return;
  dock.classList.toggle('is-dark', !!WAC.isDarkTheme());
};

WAC.syncDockState = function () {
  WAC.syncDockVisibility();
  WAC.syncThemeState();
  const dock = WAC.dock();
  const launcher = WAC.launcher();
  if (dock) dock.classList.toggle('is-open', !!WAC.dockOpen);
  if (launcher) launcher.setAttribute('aria-expanded', WAC.dockOpen ? 'true' : 'false');
  WAC.syncSettingsState();
};

WAC.syncSettingsState = function () {
  const panel = WAC.panel();
  const launcher = WAC.settingsLauncher();
  const open = !!WAC.dockOpen && !!WAC.settingsOpen;
  if (panel) panel.classList.toggle('is-settings-open', open);
  if (launcher) launcher.setAttribute('aria-expanded', open ? 'true' : 'false');
};

WAC.syncDockLayout = function () {
  const dock = WAC.dock();
  if (!dock) return;
  if (window.innerWidth <= 900) {
    dock.style.removeProperty('--dock-panel-width');
    dock.style.removeProperty('--dock-settings-panel-width');
    return;
  }
  const candidates = [
    dock.parentElement,
    dock.parentElement ? dock.parentElement.closest('.column') : null,
    dock.parentElement && dock.parentElement.parentElement ? dock.parentElement.parentElement.closest('.column') : null,
  ].filter((node) => node && node !== dock);
  const flowColumn = candidates
    .map((node) => ({ node, rect: node.getBoundingClientRect() }))
    .filter((entry) => entry.rect.width > 180)
    .sort((a, b) => a.rect.width - b.rect.width)[0];
  const flowRect = flowColumn ? flowColumn.rect : null;
  const dockStyle = window.getComputedStyle(dock);
  const launcherWidth = parseFloat(dockStyle.getPropertyValue('--dock-launcher-width')) || 41;
  const dockGap = parseFloat(dockStyle.getPropertyValue('--dock-gap')) || 14;
  const panelLeft = launcherWidth + dockGap;
  const measuredWidth = flowRect ? Math.round(flowRect.width) : 0;
  const columnBoundWidth = flowRect ? Math.round(flowRect.right - panelLeft - 12) : 0;
  const maxWidth = Math.max(320, window.innerWidth - panelLeft - 28);
  const panelWidth = Math.max(Math.min(320, maxWidth), Math.min(measuredWidth || 548, columnBoundWidth || measuredWidth || 548, maxWidth));
  const settingsPanelOffset = parseFloat(dockStyle.getPropertyValue('--dock-settings-panel-offset')) || 44;
  const maxSettingsWidth = Math.max(320, window.innerWidth - panelLeft - panelWidth - settingsPanelOffset - 12);
  const settingsWidth = Math.min(maxSettingsWidth, Math.max(320, Math.min(panelWidth + 112, 660)));
  dock.style.setProperty('--dock-panel-width', `${panelWidth}px`);
  dock.style.setProperty('--dock-settings-panel-width', `${settingsWidth}px`);
};

WAC.setDockOpen = function (open) {
  WAC.dockOpen = !!open;
  WAC.syncDockState();
  WAC.syncDockLayout();
  if (WAC.dockOpen) {
    window.setTimeout(() => {
      WAC.syncComposerLayout();
      const input = WAC.requestInput();
      if (input) input.focus();
    }, 140);
  }
};

WAC.toggleDock = function (forceOpen) {
  const nextOpen = typeof forceOpen === 'boolean' ? forceOpen : !WAC.dockOpen;
  WAC.setDockOpen(nextOpen);
};

WAC.setSettingsOpen = function (open) {
  WAC.settingsOpen = !!open;
  if (WAC.settingsOpen && !WAC.dockOpen) WAC.dockOpen = true;
  WAC.syncDockState();
  WAC.syncDockLayout();
  if (WAC.settingsOpen) {
    const refreshButton = document.querySelector('#assistant_chat_session_refresh_button button, #assistant_chat_session_refresh_button');
    if (refreshButton && typeof refreshButton.click === 'function') refreshButton.click();
  }
};

WAC.toggleSettings = function (forceOpen) {
  const nextOpen = typeof forceOpen === 'boolean' ? forceOpen : !WAC.settingsOpen;
  WAC.setSettingsOpen(nextOpen);
};

WAC.ensureShell = function () {
  const host = WAC.host();
  if (!host) return false;
  WAC.readSessionCatalogFromHost();
  WAC.syncSessionActionTooltips();
  WAC.bindSessionPickerControls(host);
  if (host.dataset.wangpAssistantChatMounted === 'true' && WAC.shell()) {
    if (WAC.replayDepth <= 0) {
      WAC.showEmptyIfNeeded();
      WAC.syncDockState();
      WAC.syncDockLayout();
      WAC.syncComposerLayout();
    }
    return true;
  }
  host.innerHTML = `
    <section class="chat">
      <div class="chat__scroll">
        <div class="chat__empty">
          ${WAC.emptyMarkup('zero')}
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
  `;
  host.dataset.wangpAssistantChatMounted = 'true';
  WAC.bindSessionPickerControls(host);
  WAC.hydrate();
  WAC.syncDockVisibility();
  WAC.syncDockState();
  WAC.syncDockLayout();
  WAC.syncComposerLayout();
  WAC.syncDisclosureBridge();
  WAC.syncScrollBridge();
  return true;
};

WAC.isNearBottom = function () {
  const scroll = WAC.scroll();
  if (!scroll) return true;
  return (scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight) <= WAC.bottomThreshold();
};

WAC.syncJumpToBottom = function () {
  if (WAC.replayDepth > 0) return;
  const node = WAC.jumpBottomNode();
  if (!node) return;
  const show = WAC.state.order.length > 0 && !WAC.isNearBottom();
  node.classList.toggle('is-visible', show);
  node.setAttribute('aria-hidden', show ? 'false' : 'true');
  node.tabIndex = show ? 0 : -1;
};

WAC.scrollToBottom = function () {
  const scroll = WAC.scroll();
  if (!scroll) return;
  scroll.scrollTop = scroll.scrollHeight;
  WAC.syncJumpToBottom();
};

WAC.scrollToBottomAfterLayout = function () {
  WAC.scrollToBottom();
  if (WAC.followSubmissionScrollFrame) window.cancelAnimationFrame(WAC.followSubmissionScrollFrame);
  WAC.followSubmissionScrollFrame = window.requestAnimationFrame(() => {
    WAC.followSubmissionScrollFrame = window.requestAnimationFrame(() => {
      WAC.followSubmissionScrollFrame = 0;
      WAC.scrollToBottom();
    });
  });
};

WAC.hideEmpty = function () {
  if (WAC.replayDepth > 0) return;
  const empty = WAC.empty();
  if (empty) empty.style.display = 'none';
};

WAC.showEmptyIfNeeded = function () {
  if (WAC.replayDepth > 0) return;
  const empty = WAC.empty();
  const transcript = WAC.transcript();
  const isEmpty = WAC.state.order.length === 0;
  if (empty) empty.style.display = isEmpty ? 'flex' : 'none';
  if (transcript) transcript.style.display = isEmpty ? 'none' : 'flex';
  WAC.syncJumpToBottom();
};

WAC.createMessageNode = function (message) {
  const tpl = document.createElement('template');
  tpl.innerHTML = (message && message.html) ? String(message.html).trim() : '';
  return tpl.content.firstElementChild;
};

WAC.markSyncRequired = function (event) {
  WAC.syncRequired = true;
  window.dispatchEvent(new CustomEvent('wangp-assistant-chat-sync-required', { detail: { chatSessionId: WAC.chatSessionId, sequence: WAC.chatSequence, event: event || null } }));
  if (!WAC.syncRecoveryPending) {
    WAC.syncRecoveryPending = true;
    if (!WAC.requestCanonicalSync()) WAC.syncRecoveryPending = false;
  }
};

WAC.syncAcknowledgesFollowedSubmission = function (messages, acknowledgedSubmissionIds) {
  const followed = String(WAC.followSubmissionId || '').trim();
  if (!followed) return false;
  if ((Array.isArray(acknowledgedSubmissionIds) ? acknowledgedSubmissionIds : []).some((value) => String(value || '').trim() === followed)) return true;
  return (Array.isArray(messages) ? messages : []).some((message) => String(message && message.client_submission_id || '').trim() === followed);
};

WAC.messageNode = function (messageId) {
  const transcript = WAC.transcript();
  return transcript ? transcript.querySelector(`[data-message-id="${CSS.escape(String(messageId || ''))}"]`) : null;
};

WAC.blockNode = function (messageId, blockId) {
  const messageNode = WAC.messageNode(messageId);
  return messageNode ? messageNode.querySelector(`[data-block-id="${CSS.escape(String(blockId || ''))}"]`) : null;
};

WAC.liveTextNode = function (blockNode) {
  return blockNode && blockNode.querySelector ? blockNode.querySelector('.chat__stream-text') : null;
};

WAC.safeStreamingMarkdownUrl = function (value, image) {
  const raw = String(value || '').trim().replace(/^<|>$/g, '');
  if (!raw || /^javascript:/i.test(raw) || /^data:/i.test(raw)) return '';
  if (raw.startsWith('/wangp_api/gallery/media/') || (!image && raw.startsWith('/wangp_api/download/'))) return raw;
  try {
    const resolved = new URL(raw, document.baseURI);
    return resolved.protocol === 'http:' || resolved.protocol === 'https:' ? resolved.href : '';
  } catch (_error) {
    return '';
  }
};

WAC.streamingMarkdownDelimiterFlags = function (text, index, width, marker) {
  const before = index > 0 ? text[index - 1] : '';
  const after = index + width < text.length ? text[index + width] : '';
  const beforeWhitespace = !before || /\s/u.test(before);
  const afterWhitespace = !after || /\s/u.test(after);
  const beforePunctuation = !!before && /[\p{P}\p{S}]/u.test(before);
  const afterPunctuation = !!after && /[\p{P}\p{S}]/u.test(after);
  const leftFlanking = !afterWhitespace && (!afterPunctuation || beforeWhitespace || beforePunctuation);
  const rightFlanking = !beforeWhitespace && (!beforePunctuation || afterWhitespace || afterPunctuation);
  if (marker === '_') {
    return { open: leftFlanking && (!rightFlanking || beforePunctuation), close: rightFlanking && (!leftFlanking || afterPunctuation) };
  }
  return { open: leftFlanking, close: rightFlanking };
};

WAC.findStreamingMarkdownCloser = function (text, marker, start) {
  const width = marker.length;
  const markerChar = marker[0];
  let index = text.indexOf(marker, start);
  while (index >= 0) {
    let backslashes = 0;
    for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) backslashes += 1;
    const exactRun = text[index - 1] !== markerChar && text[index + width] !== markerChar;
    if (backslashes % 2 === 0 && exactRun && WAC.streamingMarkdownDelimiterFlags(text, index, width, markerChar).close) return index;
    index = text.indexOf(marker, index + width);
  }
  return -1;
};

WAC.appendStreamingInlineMarkdown = function (parent, value) {
  const text = String(value || '');
  let plainStart = 0;
  let index = 0;
  const flushPlain = (end) => {
    if (end > plainStart) parent.appendChild(document.createTextNode(text.slice(plainStart, end)));
  };
  while (index < text.length) {
    let consumed = 0;
    let rendered = null;
    if (text[index] === '\\' && index + 1 < text.length) {
      flushPlain(index);
      parent.appendChild(document.createTextNode(text[index + 1]));
      index += 2;
      plainStart = index;
      continue;
    }
    if ((text.startsWith('![', index) || text[index] === '[')) {
      const image = text.startsWith('![', index);
      const labelStart = index + (image ? 2 : 1);
      const labelEnd = text.indexOf('](', labelStart);
      const targetEnd = labelEnd < 0 ? -1 : text.indexOf(')', labelEnd + 2);
      if (labelEnd >= 0 && targetEnd >= 0) {
        const label = text.slice(labelStart, labelEnd);
        const rawTarget = text.slice(labelEnd + 2, targetEnd).trim().split(/\s+/)[0];
        const href = WAC.safeStreamingMarkdownUrl(rawTarget, image);
        if (href) {
          if (image) {
            rendered = document.createElement('img');
            rendered.src = href;
            rendered.alt = label;
            rendered.loading = 'lazy';
          } else {
            rendered = document.createElement('a');
            rendered.href = href;
            rendered.target = '_blank';
            rendered.rel = 'noopener noreferrer';
            WAC.appendStreamingInlineMarkdown(rendered, label);
          }
          consumed = targetEnd - index + 1;
        }
      }
    }
    if (!rendered && text[index] === '`') {
      const end = text.indexOf('`', index + 1);
      if (end > index + 1) {
        rendered = document.createElement('code');
        rendered.textContent = text.slice(index + 1, end);
        consumed = end - index + 1;
      }
    }
    if (!rendered && (text.startsWith('**', index) || text.startsWith('__', index)) && text[index - 1] !== text[index] && text[index + 2] !== text[index] && WAC.streamingMarkdownDelimiterFlags(text, index, 2, text[index]).open) {
      const marker = text.slice(index, index + 2);
      const end = WAC.findStreamingMarkdownCloser(text, marker, index + 2);
      if (end > index + 2) {
        rendered = document.createElement('strong');
        WAC.appendStreamingInlineMarkdown(rendered, text.slice(index + 2, end));
        consumed = end - index + 2;
      }
    }
    if (!rendered && (text[index] === '*' || text[index] === '_') && text[index - 1] !== text[index] && text[index + 1] !== text[index] && WAC.streamingMarkdownDelimiterFlags(text, index, 1, text[index]).open) {
      const end = WAC.findStreamingMarkdownCloser(text, text[index], index + 1);
      if (end > index + 1) {
        rendered = document.createElement('em');
        WAC.appendStreamingInlineMarkdown(rendered, text.slice(index + 1, end));
        consumed = end - index + 1;
      }
    }
    if (!rendered) {
      index += 1;
      continue;
    }
    flushPlain(index);
    parent.appendChild(rendered);
    index += consumed;
    plainStart = index;
  }
  flushPlain(text.length);
};

WAC.resetStreamingMarkdown = function (node) {
  node.replaceChildren();
  const state = { source: '', buffer: '', inFence: false, code: null, list: null, listType: '', blockBoundary: true, tail: null };
  node.__wangpStreamingMarkdown = state;
  return state;
};

WAC.appendStreamingListItem = function (node, state, line) {
  const match = line.match(/^[ \t]*(?:([-+*])|(\d+)\.)[ \t]+(.*)$/);
  if (!match) return null;
  const listType = match[1] ? 'ul' : 'ol';
  const continuing = state.listType === listType && state.list && state.list === node.lastElementChild;
  if (!state.blockBoundary && !continuing) return null;
  const list = continuing ? state.list : document.createElement(listType);
  if (!continuing) {
    if (listType === 'ol' && Number(match[2]) !== 1) list.start = Number(match[2]);
    node.appendChild(list);
  }
  const item = document.createElement('li');
  WAC.appendStreamingInlineMarkdown(item, match[3]);
  list.appendChild(item);
  state.list = list;
  state.listType = listType;
  state.blockBoundary = false;
  return item;
};

WAC.renderStreamingMarkdownLine = function (node, state, line) {
  const fence = line.match(/^\s*```\s*([^\s`]*)\s*$/);
  if (state.inFence) {
    if (fence) {
      state.inFence = false;
      state.code = null;
      state.blockBoundary = true;
    } else {
      state.code.appendChild(document.createTextNode(`${line}\n`));
    }
    return;
  }
  if (fence) {
    state.list = null;
    state.listType = '';
    const pre = document.createElement('pre');
    const code = document.createElement('code');
    if (fence[1]) code.className = `language-${fence[1].replace(/[^a-z0-9_-]/gi, '')}`;
    pre.appendChild(code);
    node.appendChild(pre);
    state.inFence = true;
    state.code = code;
    state.blockBoundary = true;
    return;
  }
  const heading = line.match(/^(#{1,6})\s+(.+)$/);
  if (heading) {
    state.list = null;
    state.listType = '';
    const element = document.createElement(`h${heading[1].length}`);
    WAC.appendStreamingInlineMarkdown(element, heading[2]);
    node.appendChild(element);
    state.blockBoundary = true;
    return;
  }
  const quote = line.match(/^\s*>\s?(.*)$/);
  if (quote) {
    state.list = null;
    state.listType = '';
    const element = document.createElement('blockquote');
    WAC.appendStreamingInlineMarkdown(element, quote[1]);
    node.appendChild(element);
    state.blockBoundary = true;
    return;
  }
  if (WAC.appendStreamingListItem(node, state, line)) return;
  if (!line.trim()) {
    if (!state.list) node.appendChild(document.createElement('br'));
    state.blockBoundary = true;
    return;
  }
  state.list = null;
  state.listType = '';
  const rule = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line);
  if (rule) {
    node.appendChild(document.createElement('hr'));
    state.blockBoundary = true;
    return;
  }
  const span = document.createElement('span');
  WAC.appendStreamingInlineMarkdown(span, line);
  node.appendChild(span);
  node.appendChild(document.createElement('br'));
  state.blockBoundary = false;
};

WAC.renderStreamingMarkdown = function (node, value) {
  if (!node) return;
  const text = String(value || '');
  let state = node.__wangpStreamingMarkdown;
  if (!state || !text.startsWith(state.source)) state = WAC.resetStreamingMarkdown(node);
  if (text === state.source) return;
  const delta = text.slice(state.source.length);
  const previousBuffer = state.buffer;
  state.buffer += delta;
  state.source = text;
  let completedLine = false;
  let newline = state.buffer.indexOf('\n');
  while (newline >= 0) {
    completedLine = true;
    if (state.tail) state.tail.remove();
    state.tail = null;
    WAC.renderStreamingMarkdownLine(node, state, state.buffer.slice(0, newline));
    state.buffer = state.buffer.slice(newline + 1);
    newline = state.buffer.indexOf('\n');
  }
  const delimiterArrived = ['\\', '`', '*', '_', '[', ']', '(', ')', '!'].some((marker) => delta.includes(marker)) || previousBuffer.endsWith('\\');
  const listMarker = /^[ \t]*(?:[-+*]|\d+\.)[ \t]+/;
  const listMarkerArrived = listMarker.test(state.buffer) && !listMarker.test(previousBuffer);
  if (state.tail && !completedLine && !delimiterArrived && !listMarkerArrived && state.buffer === previousBuffer + delta) {
    const target = ['OL', 'UL'].includes(state.tail.tagName) ? state.tail.lastElementChild : state.tail;
    target.appendChild(document.createTextNode(delta));
    return;
  }
  if (state.tail) state.tail.remove();
  if (!state.inFence) {
    const preview = { ...state };
    const item = WAC.appendStreamingListItem(node, preview, state.buffer);
    if (item) {
      state.tail = preview.list === state.list ? item : preview.list;
      return;
    }
  }
  state.tail = document.createElement('span');
  state.tail.className = 'chat__stream-tail';
  if (state.inFence && state.code) {
    state.tail.textContent = state.buffer;
    state.code.appendChild(state.tail);
  } else {
    WAC.appendStreamingInlineMarkdown(state.tail, state.buffer);
    node.appendChild(state.tail);
  }
};

WAC.incrementalMessageState = function (messageId) {
  const key = String(messageId || '');
  if (!WAC.blockState[key]) WAC.blockState[key] = { order: [], blocks: {} };
  return WAC.blockState[key];
};

WAC.createBlockNode = function (html) {
  const tpl = document.createElement('template');
  tpl.innerHTML = String(html || '').trim();
  return tpl.content.firstElementChild;
};

WAC.positionBlockNode = function (messageNode, blockNode, blockIndex) {
  const body = messageNode && messageNode.querySelector ? messageNode.querySelector('.chat__body') : null;
  if (!body || !blockNode) return;
  const index = Number.isFinite(Number(blockIndex)) ? Number(blockIndex) : body.querySelectorAll(':scope > [data-block-id]').length;
  blockNode.dataset.blockIndex = String(index);
  const before = Array.from(body.querySelectorAll(':scope > [data-block-id]')).find((node) => node !== blockNode && Number(node.dataset.blockIndex || -1) > index);
  if (before) body.insertBefore(blockNode, before);
  else if (blockNode.parentElement !== body || blockNode !== body.lastElementChild) body.appendChild(blockNode);
};

WAC.rememberBlock = function (event, node, text, finalized) {
  const messageState = WAC.incrementalMessageState(event.message_id);
  const blockId = String(event.block_id || '');
  if (!messageState.blocks[blockId]) messageState.order.push(blockId);
  messageState.blocks[blockId] = {
    id: blockId,
    type: String(event.block_type || (node && node.dataset.blockType) || ''),
    index: Number.isFinite(Number(event.block_index)) ? Number(event.block_index) : messageState.order.length - 1,
    html: node ? node.outerHTML : String(event.html || ''),
    text: String(typeof text === 'undefined' ? '' : text),
    finalized: !!finalized,
  };
  messageState.order.sort((left, right) => Number(messageState.blocks[left].index) - Number(messageState.blocks[right].index));
};

WAC.upsertBlock = function (event) {
  if (!event || !event.message_id || !event.block_id) return;
  if (!WAC.messageNode(event.message_id) && event.message) WAC.upsertMessage(event.message, true, event.message_index);
  const messageNode = WAC.messageNode(event.message_id);
  if (!messageNode) return WAC.markSyncRequired(event);
  WAC.captureDisclosureState(messageNode);
  const scrollState = WAC.captureAutoscrollState();
  const next = WAC.createBlockNode(event.html);
  if (!next) return WAC.markSyncRequired(event);
  const initialText = String(event.text || '');
  const nextLive = WAC.liveTextNode(next);
  if (nextLive) WAC.renderStreamingMarkdown(nextLive, initialText);
  const existing = WAC.blockNode(event.message_id, event.block_id);
  let node = next;
  if (existing) {
    const key = WAC.disclosureKey(existing.querySelector && existing.matches('.chat__disclosure') ? existing : existing.querySelector('.chat__disclosure'));
    if (key) WAC.disclosureState[key] = !!(existing.matches('.chat__disclosure') ? existing.open : existing.querySelector('.chat__disclosure').open);
    existing.replaceWith(next);
  }
  WAC.positionBlockNode(messageNode, node, event.block_index);
  WAC.rememberBlock(event, node, initialText, !event.streaming);
  WAC.hideEmpty();
  WAC.applyDisclosureState(messageNode);
  WAC.applyAutoscrollState(scrollState);
};

WAC.currentBlockText = function (event, node) {
  const messageState = WAC.incrementalMessageState(event.message_id);
  const known = messageState.blocks[String(event.block_id || '')];
  if (known) return String(known.text || '');
  const live = WAC.liveTextNode(node);
  return live ? String(live.textContent || '') : '';
};

WAC.appendBlockText = function (event) {
  const node = WAC.blockNode(event.message_id, event.block_id);
  const live = WAC.liveTextNode(node);
  if (!node || !live) return WAC.markSyncRequired(event);
  const current = WAC.currentBlockText(event, node);
  const start = Number(event.text_start);
  const end = Number(event.text_end);
  if (Number.isFinite(end) && current.length >= end) return;
  if (!Number.isFinite(start) || current.length !== start) return WAC.markSyncRequired(event);
  const scrollState = WAC.captureAutoscrollState();
  const suffix = String(event.text || '');
  WAC.renderStreamingMarkdown(live, current + suffix);
  WAC.rememberBlock(event, node, current + suffix, false);
  WAC.applyAutoscrollState(scrollState);
};

WAC.replaceBlockText = function (event) {
  const node = WAC.blockNode(event.message_id, event.block_id);
  const live = WAC.liveTextNode(node);
  if (!node || !live) return WAC.markSyncRequired(event);
  const scrollState = WAC.captureAutoscrollState();
  WAC.renderStreamingMarkdown(live, String(event.text || ''));
  WAC.rememberBlock(event, node, String(event.text || ''), false);
  WAC.applyAutoscrollState(scrollState);
};

WAC.finalizeBlock = function (event) {
  if (!WAC.messageNode(event.message_id) && event.message) WAC.upsertMessage(event.message, true, event.message_index);
  const messageNode = WAC.messageNode(event.message_id);
  const current = WAC.blockNode(event.message_id, event.block_id);
  const next = WAC.createBlockNode(event.html);
  if (!messageNode || !next) return WAC.markSyncRequired(event);
  if (current) WAC.captureDisclosureState(current);
  const scrollState = WAC.captureAutoscrollState();
  if (current) current.replaceWith(next);
  WAC.positionBlockNode(messageNode, next, event.block_index);
  WAC.rememberBlock(event, next, String(event.text || ''), true);
  WAC.applyDisclosureState(next);
  WAC.applyAutoscrollState(scrollState);
};

WAC.removeBlock = function (event) {
  const node = WAC.blockNode(event.message_id, event.block_id);
  const scrollState = WAC.captureAutoscrollState();
  if (node) node.remove();
  const messageState = WAC.blockState[String(event.message_id || '')];
  if (messageState) {
    delete messageState.blocks[String(event.block_id || '')];
    messageState.order = messageState.order.filter((blockId) => blockId !== String(event.block_id || ''));
  }
  WAC.applyAutoscrollState(scrollState);
};

WAC.replayMessageBlocks = function (messageId, messageNode) {
  const messageState = WAC.blockState[String(messageId || '')];
  if (!messageState || !messageNode) return;
  for (const blockId of messageState.order) {
    const block = messageState.blocks[blockId];
    if (!block) continue;
    const next = WAC.createBlockNode(block.html);
    if (!next) continue;
    const live = WAC.liveTextNode(next);
    if (live && !block.finalized) WAC.renderStreamingMarkdown(live, String(block.text || ''));
    const existing = messageNode.querySelector(`[data-block-id="${CSS.escape(blockId)}"]`);
    if (existing) existing.replaceWith(next);
    WAC.positionBlockNode(messageNode, next, block.index);
  }
};

WAC.syncAttributes = function (target, source) {
  if (!target || !source || !target.getAttributeNames || !source.getAttributeNames) return;
  const sourceNames = new Set(source.getAttributeNames());
  for (const name of target.getAttributeNames()) {
    if (!sourceNames.has(name)) target.removeAttribute(name);
  }
  for (const name of sourceNames) {
    const nextValue = source.getAttribute(name);
    if (target.getAttribute(name) !== nextValue) target.setAttribute(name, nextValue);
  }
};

WAC.patchDisclosureNode = function (current, next) {
  if (!current || !next) return;
  const wasOpen = !!current.open;
  WAC.syncAttributes(current, next);
  current.open = wasOpen;
  current.className = next.className;
  const currentSummary = current.querySelector(':scope > summary');
  const nextSummary = next.querySelector(':scope > summary');
  if (currentSummary && nextSummary && currentSummary.innerHTML !== nextSummary.innerHTML) currentSummary.innerHTML = nextSummary.innerHTML;
  const currentBody = current.querySelector(':scope > .chat__disclosure-body');
  const nextBody = next.querySelector(':scope > .chat__disclosure-body');
  if (currentBody && nextBody && currentBody.innerHTML !== nextBody.innerHTML) currentBody.innerHTML = nextBody.innerHTML;
};

WAC.patchMessageBody = function (currentBody, nextBody) {
  if (!currentBody || !nextBody) return;
  const existingByKey = new Map();
  currentBody.querySelectorAll(':scope > .chat__disclosure').forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (key) existingByKey.set(key, node);
  });
  let cursor = currentBody.firstChild;
  for (const nextNode of Array.from(nextBody.childNodes)) {
    const key = nextNode.nodeType === 1 && nextNode.classList.contains('chat__disclosure') ? WAC.disclosureKey(nextNode) : '';
    const reusable = key ? existingByKey.get(key) : null;
    if (reusable) {
      WAC.patchDisclosureNode(reusable, nextNode);
      existingByKey.delete(key);
      if (reusable === cursor) cursor = cursor.nextSibling;
      else currentBody.insertBefore(reusable, cursor);
      continue;
    }
    const cursorIsDisclosure = cursor && cursor.nodeType === 1 && cursor.classList.contains('chat__disclosure');
    if (cursor && !cursorIsDisclosure) {
      const replaced = cursor;
      cursor = cursor.nextSibling;
      replaced.replaceWith(nextNode);
    } else {
      currentBody.insertBefore(nextNode, cursor);
    }
  }
  while (cursor) {
    const removed = cursor;
    cursor = cursor.nextSibling;
    removed.remove();
  }
};

WAC.patchMessageNode = function (current, next) {
  if (!current || !next) return;
  const preserveQueuedEdit = String(WAC.queuedEditMessageId || '') === String(current.getAttribute('data-message-id') || '') && !!next.querySelector('[data-message-action="edit"]');
  WAC.syncAttributes(current, next);
  current.className = next.className;
  if (preserveQueuedEdit) current.classList.add('is-editing');
  const currentAvatar = current.querySelector(':scope > .chat__avatar');
  const nextAvatar = next.querySelector(':scope > .chat__avatar');
  if (currentAvatar && nextAvatar) {
    WAC.syncAttributes(currentAvatar, nextAvatar);
    if (currentAvatar.innerHTML !== nextAvatar.innerHTML) currentAvatar.innerHTML = nextAvatar.innerHTML;
  }
  const currentCard = current.querySelector(':scope > .chat__message-card');
  const nextCard = next.querySelector(':scope > .chat__message-card');
  if (!currentCard || !nextCard) {
    current.replaceChildren(...Array.from(next.childNodes));
    return;
  }
  WAC.syncAttributes(currentCard, nextCard);
  currentCard.className = nextCard.className;
  const currentMeta = currentCard.querySelector(':scope > .chat__meta');
  const nextMeta = nextCard.querySelector(':scope > .chat__meta');
  if (currentMeta && nextMeta) {
    WAC.syncAttributes(currentMeta, nextMeta);
    currentMeta.className = nextMeta.className;
    if (currentMeta.innerHTML !== nextMeta.innerHTML) currentMeta.innerHTML = nextMeta.innerHTML;
  }
  const currentBody = currentCard.querySelector(':scope > .chat__body');
  const nextBody = nextCard.querySelector(':scope > .chat__body');
  if (currentBody && nextBody) {
    WAC.syncAttributes(currentBody, nextBody);
    currentBody.className = nextBody.className;
    WAC.patchMessageBody(currentBody, nextBody);
  }
  const currentEnd = currentCard.querySelector(':scope > .chat__message-end');
  const nextEnd = nextCard.querySelector(':scope > .chat__message-end');
  if (currentEnd && nextEnd) {
    WAC.syncAttributes(currentEnd, nextEnd);
    currentEnd.className = nextEnd.className;
    if (currentEnd.innerHTML !== nextEnd.innerHTML) currentEnd.innerHTML = nextEnd.innerHTML;
  } else if (currentEnd) {
    currentEnd.remove();
  } else if (nextEnd) {
    currentCard.appendChild(nextEnd);
  }
};

WAC.messageBodyText = function (node) {
  const body = node && node.querySelector ? node.querySelector('.chat__body') : null;
  return body ? WAC.normalizeText(body.innerText || body.textContent || '') : '';
};

WAC.upsertMessage = function (message, preserveIncrementalState, messageIndex) {
  if (!message || !message.id) return;
  WAC.ensureShell();
  const transcript = WAC.transcript();
  if (!transcript) return;
  WAC.captureDisclosureState(transcript);
  const scrollState = WAC.captureAutoscrollState();
  const node = WAC.createMessageNode(message);
  if (!node) return;
  const existing = transcript.querySelector(`[data-message-id="${CSS.escape(String(message.id))}"]`);
  const incomingId = String(message.id);
  if (!preserveIncrementalState) delete WAC.blockState[incomingId];
  const positioned = Number.isInteger(Number(messageIndex)) && Number(messageIndex) >= 0;
  if (positioned) {
    const nextOrder = WAC.state.order.filter((messageId) => String(messageId) !== incomingId);
    nextOrder.splice(Math.min(Number(messageIndex), nextOrder.length), 0, incomingId);
    WAC.state.order = nextOrder;
  } else if (!existing && !WAC.state.order.includes(incomingId)) {
    WAC.state.order.push(incomingId);
  }
  if (existing) {
    WAC.patchMessageNode(existing, node);
  } else {
    transcript.appendChild(node);
  }
  if (positioned) {
    const inserted = existing || node;
    const orderIndex = WAC.state.order.indexOf(incomingId);
    const beforeId = WAC.state.order[orderIndex + 1];
    const beforeNode = beforeId ? WAC.messageNode(beforeId) : null;
    transcript.insertBefore(inserted, beforeNode);
  }
  WAC.state.messages[incomingId] = message;
  WAC.hideEmpty();
  WAC.applyDisclosureState(transcript);
  WAC.syncQueuedRequestEdit();
  WAC.applyAutoscrollState(scrollState);
};

WAC.removeMessage = function (messageId) {
  const transcript = WAC.transcript();
  if (!transcript) return;
  const scrollState = WAC.captureAutoscrollState();
  const existing = transcript.querySelector(`[data-message-id="${CSS.escape(String(messageId))}"]`);
  if (existing) existing.remove();
  delete WAC.state.messages[String(messageId)];
  delete WAC.blockState[String(messageId)];
  WAC.state.order = WAC.state.order.filter(id => id !== String(messageId));
  WAC.syncQueuedRequestEdit();
  WAC.showEmptyIfNeeded();
  WAC.applyAutoscrollState(scrollState);
};

WAC.setStatus = function (status, restoreAnchor) {
  WAC.ensureShell();
  const scrollState = WAC.captureAutoscrollState();
  WAC.state.status = status || null;
  const node = WAC.statusNode();
  if (!node) return;
  const textNode = node.querySelector('.chat__status-text');
  const pauseNode = node.querySelector('.chat__status-pause');
  const stopNode = node.querySelector('.chat__status-stop');
  if (!status || !status.visible || !status.text) {
    node.classList.remove('is-visible');
    node.removeAttribute('data-kind');
    if (textNode) textNode.textContent = '';
    if (pauseNode) {
      pauseNode.textContent = 'Pause';
      pauseNode.setAttribute('aria-label', 'Pause Deepy');
      pauseNode.dataset.mode = 'pause';
      pauseNode.hidden = false;
      pauseNode.disabled = true;
    }
    if (stopNode) {
      stopNode.hidden = false;
      stopNode.disabled = true;
    }
    WAC.setBusyInputHelper(false);
    WAC.applyAutoscrollState(scrollState);
    return;
  }
  if (textNode) textNode.textContent = String(status.text);
  const kind = String(status.kind || 'status');
  node.dataset.kind = kind;
  if (pauseNode) {
    const isPaused = kind === 'paused';
    pauseNode.hidden = kind === 'session_loading';
    pauseNode.textContent = isPaused ? 'Resume' : kind === 'pause_pending' ? 'Pausing…' : kind === 'resuming' ? 'Resuming…' : 'Pause';
    pauseNode.setAttribute('aria-label', isPaused ? 'Resume Deepy' : 'Pause Deepy');
    pauseNode.dataset.mode = isPaused ? 'resume' : 'pause';
    pauseNode.disabled = kind === 'pause_pending' || kind === 'resuming' || kind === 'session_loading';
  }
  if (stopNode) {
    stopNode.hidden = kind === 'session_loading';
    stopNode.disabled = kind === 'session_loading';
  }
  node.classList.add('is-visible');
  WAC.setBusyInputHelper(true);
  WAC.applyAutoscrollState(scrollState);
};

WAC.setStats = function (stats) {
  WAC.ensureShell();
  WAC.state.stats = stats || null;
  const node = WAC.statsNode();
  if (!node) return;
  let textNode = node.querySelector('.chat__stats-text');
  if (!textNode) {
    textNode = document.createElement('span');
    textNode.className = 'chat__stats-text';
    node.appendChild(textNode);
  }
  if (!stats || stats.visible === false || !stats.text) {
    node.classList.remove('is-visible');
    textNode.textContent = '';
    textNode.removeAttribute('title');
    node.setAttribute('aria-hidden', node.classList.contains('has-input-helper') ? 'false' : 'true');
    return;
  }
  textNode.textContent = String(stats.text);
  textNode.title = String(stats.text);
  node.classList.add('is-visible');
  node.setAttribute('aria-hidden', 'false');
};

WAC.sync = function (messages, status, stats, acknowledgedSubmissionIds) {
  WAC.ensureShell();
  WAC.captureDisclosureState(WAC.transcript());
  const followSubmittedRequest = WAC.syncAcknowledgesFollowedSubmission(messages, acknowledgedSubmissionIds);
  const scrollState = followSubmittedRequest ? { atBottom: true, top: 0 } : WAC.captureAutoscrollState();
  WAC.blockState = {};
  WAC.replaceState(messages, status, stats);
  WAC.reconcileOptimisticSubmits(acknowledgedSubmissionIds);
  WAC.hydrate(scrollState);
  if (followSubmittedRequest) {
    WAC.followSubmissionId = '';
    WAC.scrollToBottomAfterLayout();
  }
};

WAC.reset = function () {
  if (WAC.queuedEditMessageId) WAC.finishQueuedRequestEdit();
  WAC.state = { order: [], messages: {}, status: null, stats: null };
  WAC.blockState = {};
  WAC.optimisticSubmits = [];
  WAC.pendingSteeringId = '';
  WAC.followSubmissionId = '';
  WAC.chatSessionId = '';
  WAC.chatRevision = -1;
  WAC.chatSequence = -1;
  WAC.syncRequired = false;
  WAC.syncRecoveryPending = false;
  WAC.disclosureState = {};
  WAC.ensureShell();
  const transcript = WAC.transcript();
  if (transcript) transcript.innerHTML = '';
  WAC.showEmptyIfNeeded();
  WAC.setStatus(null);
  WAC.setStats(null);
};

WAC.hydrate = function (scrollState) {
  const transcript = WAC.transcript();
  if (!transcript) return;
  const existingById = new Map();
  transcript.querySelectorAll(':scope > [data-message-id]').forEach((node) => {
    const messageId = String(node.getAttribute('data-message-id') || '');
    if (messageId) existingById.set(messageId, node);
  });
  let cursor = transcript.firstElementChild;
  for (const messageId of WAC.state.order) {
    const message = WAC.state.messages[messageId];
    if (!message) continue;
    const node = WAC.createMessageNode(message);
    if (!node) continue;
    const existing = existingById.get(String(messageId));
    if (existing) {
      WAC.patchMessageNode(existing, node);
      existingById.delete(String(messageId));
      if (existing === cursor) cursor = cursor.nextElementSibling;
      else transcript.insertBefore(existing, cursor);
    } else {
      transcript.insertBefore(node, cursor);
    }
    WAC.replayMessageBlocks(messageId, existing || node);
  }
  for (const obsolete of existingById.values()) obsolete.remove();
  WAC.applyDisclosureState(transcript);
  WAC.syncQueuedRequestEdit();
  WAC.showEmptyIfNeeded();
  WAC.setStatus(WAC.state.status, null);
  WAC.setStats(WAC.state.stats);
  WAC.applyAutoscrollState(scrollState);
};

WAC.applyEvent = function (payload) {
  return WAC.consumePayload(payload);
};

WAC.syncDisclosureBridge = function () {
  const transcript = WAC.transcript();
  if (!transcript || transcript === WAC.disclosureNode) return;
  if (WAC.disclosureNode) WAC.disclosureNode.removeEventListener('toggle', WAC.handleDisclosureToggle, true);
  WAC.disclosureNode = transcript;
  WAC.disclosureNode.addEventListener('toggle', WAC.handleDisclosureToggle, true);
};

WAC.handleScroll = function () {
  WAC.syncJumpToBottom();
};

WAC.syncScrollBridge = function () {
  const scroll = WAC.scroll();
  if (!scroll || scroll === WAC.scrollNode) {
    WAC.syncJumpToBottom();
    return;
  }
  if (WAC.scrollNode) WAC.scrollNode.removeEventListener('scroll', WAC.handleScroll, { passive: true });
  WAC.scrollNode = scroll;
  WAC.scrollNode.addEventListener('scroll', WAC.handleScroll, { passive: true });
  WAC.syncJumpToBottom();
};

WAC.installObserver = function () {
  if (WAC.observer) return;
  const target = document.querySelector('gradio-app') || document.body;
  if (!target) return;
  WAC.observer = new MutationObserver(() => {
      if (WAC.observerScheduled) return;
      WAC.observerScheduled = true;
      window.requestAnimationFrame(() => {
        WAC.observerScheduled = false;
        if (WAC.host()) WAC.ensureShell();
        WAC.syncScrollBridge();
        WAC.syncThemeState();
        WAC.syncDockLayout();
        WAC.syncDeepyTypePreview();
        WAC.handleEventNodeMutation();
        WAC.readEventSource();
      });
  });
  WAC.observer.observe(target, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'data-theme', 'theme', 'style'] });
};

WAC.installEventBridge = function () {
  WAC.handleEventNodeMutation();
  WAC.syncDisclosureBridge();
  WAC.syncScrollBridge();
  if (!WAC.pollTimer) WAC.pollTimer = window.setInterval(() => { WAC.readEventSource(); }, 250);
  window.addEventListener('focus', () => { WAC.readEventSource(); }, { passive: true });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) WAC.readEventSource();
  });
  window.addEventListener('resize', () => {
    WAC.resetComposerLayout();
    WAC.syncDockLayout();
    window.requestAnimationFrame(WAC.syncComposerLayout);
  }, { passive: true });
};

WAC.installDockBridge = function () {
  if (WAC.dockBridgeInstalled) return;
  WAC.dockBridgeInstalled = true;
  WAC.dockOpen = false;
  try { window.localStorage.removeItem('wangp-assistant-chat-open'); } catch (_error) {}
  document.addEventListener('beforeinput', (event) => {
    const input = WAC.requestInput();
    if (input && event.target === input) WAC.composerResizeScrollState = WAC.captureAutoscrollState();
  }, true);
  document.addEventListener('input', (event) => {
    if (event.target && event.target.closest && event.target.closest('#deepy_type_choice')) WAC.syncDeepyTypePreview();
    const input = WAC.requestInput();
    if (input && event.target === input) WAC.scheduleComposerLayout();
  }, true);
  document.addEventListener('focusin', (event) => {
    const input = WAC.requestInput();
    if (input && event.target === input) WAC.syncComposerLayout();
  }, true);
  document.addEventListener('change', (event) => {
    if (event.target && event.target.closest && event.target.closest('#deepy_type_choice')) WAC.syncDeepyTypePreview();
    const sessionPicker = event.target && event.target.closest ? event.target.closest('[data-wac-session-picker]') : null;
    if (sessionPicker) {
      const container = sessionPicker.closest('.chat__session-picker');
      const resumeButton = container ? container.querySelector('[data-wac-session-resume]') : null;
      if (resumeButton) resumeButton.disabled = !String(sessionPicker.value || '').trim();
    }
  }, true);
  document.addEventListener('click', (event) => {
    if (event.target && event.target.closest && event.target.closest('#deepy_type_choice')) window.setTimeout(WAC.syncDeepyTypePreview, 0);
  }, true);
  document.addEventListener('pointerdown', (event) => {
    if (WAC.handleCollapseButtonPointerDown(event)) return;
    if (WAC.handleDisclosurePointerDown(event)) return;
    if (WAC.handleAttachmentPointerDown(event)) return;
  }, true);
  document.addEventListener('click', (event) => {
    if (WAC.handleCopyButtonClick(event)) return;
    if (WAC.handleCollapseButtonClick(event)) return;
    if (WAC.handleQueuedRequestClick(event)) return;
    const attachmentLink = event.target && event.target.closest ? event.target.closest('.chat__attachment, .chat__body a') : null;
    if (attachmentLink) return;
    const disclosureSummary = event.target && event.target.closest ? event.target.closest('summary') : null;
    if (disclosureSummary) {
      const disclosureNode = disclosureSummary.parentElement;
      if (disclosureNode && disclosureNode.classList && disclosureNode.classList.contains('chat__disclosure')) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
    }
    const toggle = event.target && event.target.closest ? event.target.closest('#assistant_chat_toggle') : null;
    if (toggle) {
      event.preventDefault();
      WAC.toggleDock();
      return;
    }
    const settingsToggle = event.target && event.target.closest ? event.target.closest('#assistant_chat_settings_toggle') : null;
    if (settingsToggle) {
      event.preventDefault();
      WAC.toggleSettings();
      return;
    }
    const pauseButton = event.target && event.target.closest ? event.target.closest('.chat__status-pause') : null;
    if (pauseButton) {
      event.preventDefault();
      event.stopPropagation();
      const target = WAC.pauseBridgeTargets()[0];
      if (target && typeof target.click === 'function') target.click();
      return;
    }
    const stopButton = event.target && event.target.closest ? event.target.closest('.chat__status-stop') : null;
    if (stopButton) {
      event.preventDefault();
      event.stopPropagation();
      const target = WAC.stopBridgeTargets()[0];
      if (target && typeof target.click === 'function') target.click();
      return;
    }
    const jumpBottomButton = event.target && event.target.closest ? event.target.closest('.chat__jump-bottom') : null;
    if (jumpBottomButton) {
      event.preventDefault();
      WAC.scrollToBottom();
      return;
    }
    const resetButton = event.target && event.target.closest ? event.target.closest('#assistant_chat_reset_button') : null;
    if (resetButton && WAC.queuedEditMessageId) {
      event.preventDefault();
      event.stopPropagation();
      WAC.finishQueuedRequestEdit();
      return;
    }
    const askButton = event.target && event.target.closest ? event.target.closest('#assistant_chat_ask_button') : null;
    if (!askButton) return;
    const input = WAC.requestInput();
    const text = input ? String(input.value || '').trim() : '';
    if (WAC.queuedEditMessageId) {
      event.preventDefault();
      event.stopPropagation();
      if (!text) {
        if (input) input.focus({ preventScroll: true });
        return;
      }
      const messageNode = WAC.transcript() && WAC.transcript().querySelector(`[data-message-id="${CSS.escape(WAC.queuedEditMessageId)}"]`);
      WAC.submitQueuedRequestAction(messageNode, 'edit', text);
      return;
    }
    if (!text) return;
    WAC.setDockOpen(true);
    const busy = WAC.isAssistantBusy();
    const submissionId = WAC.pushOptimisticUserMessage(text, busy ? 'Queued' : '');
    WAC.setBusyInputHelper(true);
    WAC.setBridgeValue('#assistant_chat_submission_id textarea, #assistant_chat_submission_id input', submissionId);
    if (busy) {
      if (WAC.queueBusyRequest(text, submissionId)) {
        event.preventDefault();
        event.stopPropagation();
        WAC.clearRequestInput(text);
        return;
      }
    }
  }, true);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && WAC.queuedEditMessageId) {
      event.preventDefault();
      event.stopPropagation();
      WAC.finishQueuedRequestEdit();
      return;
    }
    if (event.key !== 'Escape') return;
    if (WAC.settingsOpen) {
      WAC.setSettingsOpen(false);
      return;
    }
    if (!WAC.dockOpen) return;
    WAC.setDockOpen(false);
  }, true);
  document.addEventListener('keydown', (event) => {
    const input = WAC.requestInput();
    if (!input || event.target !== input || event.key !== 'Enter' || event.shiftKey || event.altKey) return;
    const text = String(input.value || '').trim();
    if (WAC.queuedEditMessageId) {
      event.preventDefault();
      event.stopPropagation();
      if (!text) {
        input.focus({ preventScroll: true });
        return;
      }
      const messageNode = WAC.transcript() && WAC.transcript().querySelector(`[data-message-id="${CSS.escape(WAC.queuedEditMessageId)}"]`);
      WAC.submitQueuedRequestAction(messageNode, 'edit', text);
      return;
    }
    if (!text) return;
    event.preventDefault();
    event.stopPropagation();
    WAC.setDockOpen(true);
    if (event.ctrlKey || event.metaKey) {
      const submissionId = WAC.pushOptimisticUserMessage(text, 'Steered');
      if (WAC.steerRequest(text, submissionId)) {
        WAC.pendingSteeringId = submissionId;
        WAC.setStatus({ visible: true, kind: 'queued', text: 'Steering requested. Waiting for the current thought/action boundary...' });
        WAC.setBusyInputHelper(true);
        window.setTimeout(() => { if (WAC.pendingSteeringId === submissionId) WAC.pendingSteeringId = ''; }, WAC.optimisticMaxAgeMs);
        window.setTimeout(() => { WAC.clearRequestInput(text); }, 0);
      } else {
        WAC.dropOptimisticSubmit(submissionId);
        WAC.removeMessage(submissionId);
      }
      return;
    }
    const askButton = document.querySelector('#assistant_chat_ask_button button, #assistant_chat_ask_button');
    if (askButton && typeof askButton.click === 'function') askButton.click();
  }, true);
  WAC.syncDockState();
  WAC.syncDockLayout();
};

if (!WAC.init) {
  WAC.installObserver();
  WAC.installEventBridge();
  WAC.installDockBridge();
  WAC.init = true;
}

setTimeout(() => { WAC.ensureShell(); WAC.syncDeepyTypePreview(); WAC.handleEventNodeMutation(); WAC.readEventSource(); WAC.syncDockState(); WAC.syncDockLayout(); WAC.syncComposerLayout(); }, 50);
if (window.__wangpAssistantChatPending.length > 0) {
  const pending = window.__wangpAssistantChatPending.slice();
  window.__wangpAssistantChatPending.length = 0;
  for (const payload of pending) WAC.consumePayload(payload);
}
window.applyAssistantChatEvent = function (payload) {
  return WAC.consumePayload(payload);
};
"""


def _touch_chat(session) -> int:
    session.chat_revision = int(session.chat_revision or 0) + 1
    return session.chat_revision


def reset_session_chat(session) -> None:
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
        text = "Pausing after the current tool finishes..." if bool(getattr(session, "assistant_action_active", False)) else "Pausing Deepy..."
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
    return _event_payload(event)


def build_stats_event(stats: dict[str, Any] | None = None) -> str:
    return _event_payload({"type": "stats", "stats": stats})


def build_session_catalog_event(sessions: list[dict[str, Any]], active_session_id: str = "", multi_session_enabled: bool = False) -> str:
    return _event_payload({"type": "session_catalog", "sessions": list(sessions or []), "active_session_id": str(active_session_id or ""), "multi_session_enabled": bool(multi_session_enabled)})


def build_session_resume_ready_event(request_id: str, session=None) -> str:
    return _event_payload({"type": "session_resume_ready", "request_id": str(request_id or "")}, session)


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
    event = {"type": "sync", "messages": messages, "status": status}
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
    retained_blocks = [block for block in blocks if isinstance(block, dict) and block.get("type") == "context_summary"]
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
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record), "message_index": _message_index(session, record)}, session, revision)


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
    canonical_text = str(text or "").strip()
    if len(canonical_text) == 0:
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


def _short_tool_label_value(value: Any, max_chars: int = 42) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
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
    compact = re.sub(r"\s+", " ", str(label or "")).strip() or "Tool"
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
    if session is not None:
        session.chat_event_sequence = int(getattr(session, "chat_event_sequence", 0) or 0) + 1
        payload["chat_session_id"] = str(session.chat_session_id)
        payload["revision"] = int(session.chat_revision if revision is None else revision)
        payload["sequence"] = session.chat_event_sequence
        payload["sequence_start"] = session.chat_event_sequence
    return json.dumps({"event_id": uuid.uuid4().hex, "instance_id": SERVER_INSTANCE_ID, "event": payload}, ensure_ascii=False)


def _markdown_to_html(text: str) -> str:
    text = str(text or "").strip()
    if len(text) == 0:
        return ""
    text = html.escape(text, quote=False)
    rendered = markdown.markdown(text, extensions=_MARKDOWN_EXTENSIONS, output_format="html5")
    rendered = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener noreferrer"', rendered)
    rendered = re.sub(r'<a href="(/wangp_api/gallery/media/[^"]+)"', r'<a href="\1" target="_blank" rel="noopener noreferrer"', rendered)
    return re.sub(r'<a href="(/wangp_api/download/[a-f0-9]+)"', r'<a href="\1" download', rendered)


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
        from shared.gradio.downloads import register_file_download, register_gallery_download

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
    from shared.gradio.downloads import register_file_download

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
                    from shared.gradio.downloads import register_file_download
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
        return [{"href": url, "label": f"Download {filename}", "subtitle": subtitle, "thumb_url": thumb_url, "kind": kind, "path_key": url, "download": filename}]

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
    href = f"/gradio_api/file={urllib.parse.quote(normalized_path, safe='/')}"
    ext = os.path.splitext(normalized_path)[1].lower()
    resolved_label = str(label or os.path.basename(normalized_path) or "Open file").strip()
    subtitle = os.path.basename(normalized_path)
    if resolved_label == subtitle:
        subtitle = ""
    thumb_url = ""
    kind = _attachment_kind(normalized_path)
    if ext in _IMAGE_EXTENSIONS:
        kind = "image"
        thumb_url = href
    elif ext in _VIDEO_EXTENSIONS:
        kind = "video"
        try:
            thumb_url = deepy_video_tools.get_video_thumbnail_data_url(normalized_path)
        except Exception:
            thumb_url = ""
    elif ext in _AUDIO_EXTENSIONS:
        kind = "audio"
        thumb_url = _audio_thumbnail_url()
    return {
        "path_key": path_key,
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
    return f"/gradio_api/file={urllib.parse.quote(path, safe='/')}"


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
        if block_type == "context_summary":
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
    if block_type == "context_summary":
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
    content_html = f"<div class='chat__stream-text'>{html.escape(str(block.get('text', '')))}</div>" if streaming else _markdown_to_html(block.get("text", ""))
    return (
        f"<details class='chat__disclosure chat__disclosure--context-summary' data-block-id='{block_id}' data-block-type='context_summary' data-context-summary-id='{block_id}'>"
        f"<summary><span class='chat__tool-title'><span class='chat__tool-chip'>Context</span>{'Summarizing earlier history…' if streaming else 'Earlier history summarized'}</span></summary>"
        f"<div class='chat__disclosure-body'><div class='chat__context-summary'>{content_html}</div>{_render_collapse_button('summary')}</div>"
        "</details>"
    )


def _render_tool_block(tool_record: dict[str, Any], attachment_html: str = "") -> str:
    name = str(tool_record.get("name", "tool")).strip() or "tool"
    label = str(tool_record.get("label", "")).strip() or _friendly_tool_label(name)
    status = str(tool_record.get("status", "running")).strip().lower()
    status_label = str(tool_record.get("status_text", "")).strip() or {"running": "Running", "done": "Done", "error": "Error"}.get(status, status.title() or "Running")
    status_class = {"running": "running", "done": "done", "error": "error"}.get(status, "running")
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
        f"<summary><span class='chat__tool-title'><span class='chat__tool-chip'>Tool</span>{html.escape(label)}</span><span class='chat__tool-status chat__tool-status--{status_class}'>{html.escape(status_label)}</span></summary>"
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
            f"<a class='chat__attachment' href='{html.escape(href)}'{link_attributes}>"
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
