from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.parse
import uuid
from typing import Any

import markdown

from shared.deepy import video_tools as deepy_video_tools
from shared.deepy.config import DEEPY_TYPE_PRIME, normalize_deepy_type


CHAT_HOST_ID = "assistant_chat_html"
CHAT_EVENT_ID = "assistant_chat_event"
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
STOP_BRIDGE_ID = "assistant_chat_stop_bridge"
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
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jfif", ".pjpeg"}
_VIDEO_EXTENSIONS = deepy_video_tools.VIDEO_EXTENSIONS
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus"}
_MARKDOWN_EXTENSIONS = ["extra", "nl2br", "sane_lists", "fenced_code", "tables"]
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_AUDIO_THUMBNAIL_PATH = os.path.join(_REPO_ROOT, "icons", "soundwave.jpg")
SERVER_INSTANCE_ID = uuid.uuid4().hex
_UNSET = object()


def _empty_state_markup(deepy_type: str) -> str:
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
        "<div class='wangp-assistant-chat__empty-card'>"
        "<header class='wangp-assistant-chat__empty-header'>"
        "<span class='wangp-assistant-chat__empty-eyebrow'>Current assistant</span>"
        f"<h2 class='wangp-assistant-chat__empty-title'>{html.escape(title)}</h2>"
        f"<span class='wangp-assistant-chat__empty-mode'>{html.escape(mode)}</span>"
        "</header>"
        f"<p class='wangp-assistant-chat__empty-intro'>{html.escape(intro)}</p>"
        "<div class='wangp-assistant-chat__empty-grid'>"
        f"<section class='wangp-assistant-chat__empty-section'><h3>What it does for you</h3><ul>{benefit_items}</ul></section>"
        f"<section class='wangp-assistant-chat__empty-section wangp-assistant-chat__empty-section--examples'><h3>Try asking</h3><ul>{example_items}</ul></section>"
        "</div>"
        "<p class='wangp-assistant-chat__empty-tip'>Start with the outcome you want. Deepy will ask only when an important choice is missing.</p>"
        "</div>"
    )


def _shell_markup(deepy_type: str = "") -> str:
    return f"""
<section class="wangp-assistant-chat">
  <div class="wangp-assistant-chat__scroll">
    <div class="wangp-assistant-chat__empty">
      {_empty_state_markup(deepy_type)}
    </div>
    <div class="wangp-assistant-chat__transcript"></div>
  </div>
  <div class="wangp-assistant-chat__status" aria-live="polite">
    <div class="wangp-assistant-chat__status-dots" aria-hidden="true"><span></span><span></span><span></span></div>
    <div class="wangp-assistant-chat__status-text"></div>
    <button class="wangp-assistant-chat__status-stop" type="button" aria-label="Stop Deepy" disabled>Stop</button>
  </div>
  <button class="wangp-assistant-chat__jump-bottom" type="button" aria-label="Jump to latest messages" aria-hidden="true" tabindex="-1">
    <span aria-hidden="true"></span>
  </button>
</section>
""".strip()


def render_shell_html(deepy_type: str = "") -> str:
    return f"<div id='{CHAT_HOST_ID}' data-wangp-assistant-chat-mounted='true' data-deepy-type='{html.escape(normalize_deepy_type(deepy_type))}'>{_shell_markup(deepy_type)}</div>"


def render_stats_html() -> str:
    return f"<div id='{STATS_ID}' class='wangp-assistant-chat__stats' aria-hidden='true'><span class='wangp-assistant-chat__input-helper' aria-hidden='true'>Press Enter to Queue Requests / CTRL Enter to Steer Deepy</span><span class='wangp-assistant-chat__stats-text'></span></div>"


def render_launcher_html() -> str:
    return (
        f"<button id='{LAUNCHER_BUTTON_ID}' class='wangp-assistant-chat__toggle' type='button' "
        "aria-label='Toggle Deepy assistant' aria-expanded='false'>"
        "<span class='wangp-assistant-chat__toggle-text'>Ask Deepy</span>"
        "</button>"
    )


def render_settings_launcher_html() -> str:
    return (
        f"<button id='{SETTINGS_TOGGLE_ID}' class='wangp-assistant-chat__settings-toggle' type='button' "
        "aria-label='Toggle Deepy settings' aria-expanded='false'>"
        "<span class='wangp-assistant-chat__settings-toggle-text'>Settings</span>"
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

#assistant_chat_dock.is-open #assistant_chat_toggle .wangp-assistant-chat__toggle-text {
    color: #f4fbff;
}

.wangp-assistant-chat__toggle-text {
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
#assistant_chat_panel.has-fixed-composer-layout .wangp-assistant-chat {
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

#assistant_chat_panel.is-settings-open #assistant_chat_settings_toggle .wangp-assistant-chat__settings-toggle-text {
    color: #f4fbff;
}

.wangp-assistant-chat__settings-toggle-text {
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

#assistant_chat_settings_panel > .wangp-assistant-chat__settings-card {
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

#assistant_chat_settings_panel > .wangp-assistant-chat__settings-card > .form {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel > .wangp-assistant-chat__settings-card > .wangp-assistant-chat__settings-scroll {
    display: block !important;
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 12px 12px 12px;
}

#assistant_chat_settings_panel > .wangp-assistant-chat__settings-card > .wangp-assistant-chat__settings-scroll > .block {
    display: block !important;
    margin: 0 0 12px !important;
    overflow: visible;
}

#assistant_chat_settings_panel > .wangp-assistant-chat__settings-card > .wangp-assistant-chat__settings-scroll > .block > .label-wrap {
    align-items: center;
    padding: 10px 14px;
    border: 1px solid rgba(23, 90, 125, 0.16);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(236, 244, 249, 0.98) 0%, rgba(224, 237, 245, 0.98) 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
}

#assistant_chat_settings_panel > .wangp-assistant-chat__settings-card > .wangp-assistant-chat__settings-scroll > .block > .label-wrap.open {
    margin-bottom: 8px;
}

#assistant_chat_settings_panel > .wangp-assistant-chat__settings-card > .wangp-assistant-chat__settings-scroll > .block > .label-wrap span {
    color: #174a67;
}

#assistant_chat_settings_panel > .wangp-assistant-chat__settings-card > .wangp-assistant-chat__settings-scroll > .block > div:last-child {
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

#assistant_chat_stop_bridge {
    display: none !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__settings-actions {
    margin-top: 10px;
}

#assistant_chat_settings_panel .wangp-assistant-chat__settings-actions > .form {
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
    min-height: 495px;
}

.wangp-assistant-chat {
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
    height: 495px;
    overflow: hidden;
    border: 1px solid var(--chat-border);
    border-radius: 26px;
    background: var(--chat-surface);
    box-shadow: var(--chat-shadow);
    isolation: isolate;
}

.wangp-assistant-chat:has(.wangp-assistant-chat__status.is-visible) {
    --chat-status-reserved-height: 58px;
}

.wangp-assistant-chat::before {
    content: "";
    position: absolute;
    inset: 0;
    background: none;
    pointer-events: none;
}

.wangp-assistant-chat__scroll {
    position: relative;
    flex: 1;
    overflow-y: auto;
    background: transparent;
}

.wangp-assistant-chat__scroll::-webkit-scrollbar {
    width: 10px;
}

.wangp-assistant-chat__scroll::-webkit-scrollbar-thumb {
    border-radius: 999px;
    border: 2px solid transparent;
    background: rgba(29, 92, 128, 0.2);
    background-clip: padding-box;
}

.wangp-assistant-chat__empty {
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

.wangp-assistant-chat__empty-card {
    width: min(100%, 482px);
}

.wangp-assistant-chat__empty-header {
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

.wangp-assistant-chat__empty-header::after {
    content: "";
    position: absolute;
    top: -38px;
    right: -22px;
    width: 126px;
    height: 126px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.1);
}

.wangp-assistant-chat__empty-eyebrow {
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

.wangp-assistant-chat__empty-title {
    position: relative;
    z-index: 1;
    margin: 0;
    color: #ffffff;
    font-size: calc(1.48rem * var(--dock-font-scale));
    font-weight: 820;
    letter-spacing: -0.02em;
    line-height: 1.08;
}

.wangp-assistant-chat__empty-mode {
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

.wangp-assistant-chat__empty-intro {
    margin: 13px 3px 12px;
    color: #3f5f72;
    font-size: calc(0.86rem * var(--dock-font-scale));
    line-height: 1.48;
}

.wangp-assistant-chat__empty-grid {
    display: grid;
    grid-template-columns: 1fr 0.92fr;
    gap: 10px;
}

.wangp-assistant-chat__input-helper {
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

.wangp-assistant-chat__input-helper.is-visible {
    display: block;
}

.wangp-assistant-chat__empty-section {
    padding: 12px 13px 11px;
    border: 1px solid rgba(31, 94, 132, 0.12);
    border-radius: 15px;
    background: linear-gradient(180deg, rgba(244, 250, 253, 0.96) 0%, rgba(235, 246, 251, 0.88) 100%);
}

.wangp-assistant-chat__empty-section--examples {
    background: linear-gradient(180deg, rgba(248, 251, 253, 0.98) 0%, rgba(241, 247, 250, 0.9) 100%);
}

.wangp-assistant-chat__empty-section h3 {
    margin: 0 0 7px;
    color: #194d70;
    font-size: calc(0.72rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.04em;
    line-height: 1.2;
    text-transform: uppercase;
}

.wangp-assistant-chat__empty-section ul {
    display: grid;
    gap: 6px;
    margin: 0;
    padding: 0;
    list-style: none;
}

.wangp-assistant-chat__empty-section li {
    position: relative;
    margin: 0;
    padding-left: 12px;
    color: #526d7e;
    font-size: calc(0.74rem * var(--dock-font-scale));
    line-height: 1.36;
}

.wangp-assistant-chat__empty-section li::before {
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

.wangp-assistant-chat__empty-tip {
    margin: 10px 3px 0;
    color: #587486;
    font-size: calc(0.7rem * var(--dock-font-scale));
    font-weight: 650;
    line-height: 1.35;
}

.wangp-assistant-chat__transcript {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 22px 18px calc(var(--chat-status-offset) + var(--chat-status-reserved-height) + var(--chat-status-gap));
}

.wangp-assistant-chat__stats {
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

.wangp-assistant-chat__stats.is-visible,
.wangp-assistant-chat__stats.has-input-helper {
    opacity: 0.96;
}

.wangp-assistant-chat__stats-text {
    flex: 0 1 auto;
    min-width: 0;
    margin-left: auto;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
}

.wangp-assistant-chat__message {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    width: 100%;
}

.wangp-assistant-chat__message--user {
    flex-direction: row-reverse;
}

.wangp-assistant-chat__avatar {
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

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__avatar {
    color: #eefbff;
    background: linear-gradient(180deg, rgba(11, 72, 103, 0.96) 0%, rgba(7, 48, 70, 0.96) 100%);
    border: 1px solid rgba(7, 39, 57, 0.35);
}

.wangp-assistant-chat__message--user .wangp-assistant-chat__avatar {
    color: #0e4564;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(245, 251, 255, 0.99) 100%);
    border: 1px solid rgba(47, 124, 170, 0.14);
}

.wangp-assistant-chat__message-card {
    position: relative;
    width: min(82%, 860px);
    border-radius: 22px;
    padding: 16px 16px 14px;
    box-shadow: 0 18px 34px rgba(11, 36, 54, 0.08);
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__message-card {
    border: 1px solid var(--assistant-border);
    background: var(--assistant-bg);
    color: var(--assistant-text);
}

.wangp-assistant-chat__message--user .wangp-assistant-chat__message-card {
    border: 1px solid var(--user-border);
    background: var(--user-bg);
    color: var(--user-text);
}

.wangp-assistant-chat__meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
    font-size: calc(0.82rem * var(--dock-font-scale));
    color: var(--soft-text);
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__meta {
    color: rgba(242, 251, 255, 0.74);
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__time {
    color: #f4fbff;
}

.wangp-assistant-chat__meta-left {
    display: inline-flex;
    align-items: center;
    min-height: 1em;
}

.wangp-assistant-chat__meta-right {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: 7px;
    min-width: 0;
}

.wangp-assistant-chat__message--user .wangp-assistant-chat__meta-right {
    padding-right: 0;
}

.wangp-assistant-chat__message--user .wangp-assistant-chat__meta-left {
    padding-left: 78px;
}

.wangp-assistant-chat__message--user .wangp-assistant-chat__message-actions {
    position: absolute;
    top: 12px;
    left: 16px;
    display: inline-flex;
    flex-direction: row;
    align-items: center;
    justify-content: flex-start;
    gap: 3px;
    transform: none;
}

.wangp-assistant-chat__message--user .wangp-assistant-chat__body {
    padding-right: 0;
}

.wangp-assistant-chat__copy-button,
.wangp-assistant-chat__message-action-button,
.wangp-assistant-chat__collapse-button {
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

.wangp-assistant-chat__copy-button svg,
.wangp-assistant-chat__message-action-button svg {
    width: 13px;
    height: 13px;
    fill: none;
    stroke: currentColor;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: 1.5;
}

.wangp-assistant-chat__message-actions .wangp-assistant-chat__copy-button,
.wangp-assistant-chat__message-action-button {
    margin: 0 !important;
}

.wangp-assistant-chat__copy-button:hover,
.wangp-assistant-chat__copy-button:focus-visible,
.wangp-assistant-chat__message-action-button:hover,
.wangp-assistant-chat__message-action-button:focus-visible {
    color: #0d5d89;
    background: rgba(255, 255, 255, 0.92);
    outline: none;
}

.wangp-assistant-chat__copy-button.is-copied {
    color: #19734a;
    border-color: rgba(25, 115, 74, 0.28);
    background: rgba(221, 249, 234, 0.96);
}

.wangp-assistant-chat__copy-button.is-copy-error {
    color: #a13f3f;
    border-color: rgba(161, 63, 63, 0.28);
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__copy-button {
    color: rgba(245, 251, 255, 0.86);
    border-color: rgba(255, 255, 255, 0.14);
    background: rgba(255, 255, 255, 0.08);
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__copy-button:hover,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__copy-button:focus-visible {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.16);
}

.wangp-assistant-chat__message--user .wangp-assistant-chat__copy-button,
.wangp-assistant-chat__message--user .wangp-assistant-chat__message-action-button,
.wangp-assistant-chat__tool-json .wangp-assistant-chat__copy-button {
    opacity: 0;
    pointer-events: none;
    transform: translateY(-2px);
}

.wangp-assistant-chat__tool-json .wangp-assistant-chat__copy-button {
    transform: none;
    transition: color 0.16s ease, background 0.16s ease;
}

.wangp-assistant-chat__message--user .wangp-assistant-chat__message-card:hover .wangp-assistant-chat__copy-button,
.wangp-assistant-chat__message--user .wangp-assistant-chat__copy-button:focus-visible,
.wangp-assistant-chat__message--user .wangp-assistant-chat__message-card:hover .wangp-assistant-chat__message-action-button,
.wangp-assistant-chat__message--user .wangp-assistant-chat__message-action-button:focus-visible,
.wangp-assistant-chat__tool-json:hover .wangp-assistant-chat__copy-button,
.wangp-assistant-chat__tool-json .wangp-assistant-chat__copy-button:focus-visible {
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0);
}

.wangp-assistant-chat__message.is-pending-queue-action .wangp-assistant-chat__message-action-button {
    opacity: 0.45;
    pointer-events: none;
}

.wangp-assistant-chat__message--user.is-editing .wangp-assistant-chat__message-card {
    outline: 2px solid rgba(31, 126, 177, 0.34);
    outline-offset: 2px;
}

.wangp-assistant-chat__author {
    font-weight: 700;
    letter-spacing: 0.03em;
}

.wangp-assistant-chat__time {
    opacity: 0.9;
    white-space: nowrap;
}

.wangp-assistant-chat__badge {
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

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__badge {
    background: rgba(255, 255, 255, 0.12);
    color: #eff9ff;
}

.wangp-assistant-chat__message-end {
    display: flex;
    justify-content: flex-end;
    margin-top: 12px;
}

.wangp-assistant-chat__message-end-badge {
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

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__tool-title,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__disclosure summary {
    color: var(--assistant-text);
}

.wangp-assistant-chat__body {
    font-size: calc(0.97rem * var(--dock-font-scale));
    line-height: 1.68;
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body p,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body li,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body strong,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body em,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body blockquote,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body h1,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body h2,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body h3,
.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body h4 {
    color: var(--assistant-text);
}

.wangp-assistant-chat__body > :first-child {
    margin-top: 0;
}

.wangp-assistant-chat__body > :last-child {
    margin-bottom: 0;
}

.wangp-assistant-chat__body p,
.wangp-assistant-chat__body ul,
.wangp-assistant-chat__body ol,
.wangp-assistant-chat__body pre,
.wangp-assistant-chat__body blockquote {
    margin: 0 0 0.85em;
}

.wangp-assistant-chat__body ul,
.wangp-assistant-chat__body ol {
    padding-left: 1.2em;
}

.wangp-assistant-chat__body code {
    padding: 0.12em 0.34em;
    border-radius: 8px;
    font-size: 0.92em;
    background: rgba(16, 73, 104, 0.08);
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body code {
    color: var(--assistant-text);
    background: rgba(255, 255, 255, 0.12);
}

.wangp-assistant-chat__body pre {
    overflow-x: auto;
    padding: 12px 13px;
    border-radius: 14px;
    border: 1px solid rgba(26, 84, 117, 0.12);
    background: rgba(239, 247, 251, 0.96);
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__body pre {
    color: var(--assistant-text);
    border-color: rgba(255, 255, 255, 0.12);
    background: rgba(7, 33, 48, 0.38);
}

.wangp-assistant-chat__body a {
    color: inherit;
    font-weight: 600;
}

.wangp-assistant-chat__disclosure {
    margin-top: 12px;
    border: 1px solid var(--tool-border);
    border-radius: 16px;
    background: var(--tool-bg);
    overflow: hidden;
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__disclosure {
    border-color: rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.08);
}

.wangp-assistant-chat__disclosure summary {
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

.wangp-assistant-chat__disclosure > summary {
    display: flex;
}

.wangp-assistant-chat__disclosure summary::-webkit-details-marker {
    display: none;
}

.wangp-assistant-chat__disclosure summary::after {
    content: "\25B8";
    font-size: calc(0.78rem * var(--dock-font-scale));
    transition: color 0.18s ease;
    color: #2f769f;
}

.wangp-assistant-chat__disclosure[open] summary::after {
    content: "\25BE";
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__disclosure summary::after {
    color: rgba(245, 251, 255, 0.86);
}

.wangp-assistant-chat__disclosure-body {
    padding: 0 14px 14px;
    font-size: calc(0.84rem * var(--dock-font-scale));
    line-height: 1.52;
    color: #385363;
}

.wangp-assistant-chat__reasoning-block > :last-child {
    margin-bottom: 0;
}

.wangp-assistant-chat__collapse-button {
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

.wangp-assistant-chat__collapse-button > span {
    display: block;
    color: #ffffff !important;
    font-size: calc(0.78rem * var(--dock-font-scale));
    line-height: 1;
    transform: rotate(180deg);
}

.wangp-assistant-chat__collapse-button:focus-visible {
    outline: 1px solid currentColor;
    outline-offset: 1px;
}

.wangp-assistant-chat__disclosure:not([open]) > .wangp-assistant-chat__disclosure-body {
    display: none;
}

.wangp-assistant-chat__disclosure[open] > .wangp-assistant-chat__disclosure-body {
    display: block;
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__disclosure-body {
    color: var(--assistant-text);
}

.wangp-assistant-chat__context-summary > :first-child {
    margin-top: 0;
}

.wangp-assistant-chat__context-summary > :last-child {
    margin-bottom: 0;
}

.wangp-assistant-chat__tool-title {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: calc(0.72rem * var(--dock-font-scale));
}

.wangp-assistant-chat__tool-chip {
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

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__tool-chip {
    color: #eff9ff;
    background: rgba(255, 255, 255, 0.14);
}

.wangp-assistant-chat__tool-status {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 999px;
    font-size: calc(0.55rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.02em;
}

.wangp-assistant-chat__tool-status--running {
    background: rgba(229, 160, 38, 0.14);
    color: #90600f;
}

.wangp-assistant-chat__tool-status--done {
    background: rgba(72, 208, 128, 0.16);
    color: #5df0a0;
}

.wangp-assistant-chat__tool-status--error {
    background: rgba(183, 62, 62, 0.12);
    color: #973232;
}

.wangp-assistant-chat__pre {
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

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__pre {
    color: var(--assistant-text);
    background: rgba(7, 33, 48, 0.38);
    border-color: rgba(255, 255, 255, 0.12);
}

.wangp-assistant-chat__tool-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
}

.wangp-assistant-chat__tool-json {
    min-width: 0;
}

.wangp-assistant-chat__tool-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    min-height: 24px;
}

.wangp-assistant-chat__tool-section-title {
    margin-bottom: 6px;
    font-size: calc(0.67rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #557385;
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__tool-section-title {
    color: rgba(233, 246, 255, 0.76);
}

.wangp-assistant-chat__tool-section-header .wangp-assistant-chat__tool-section-title {
    margin-bottom: 0;
}

@media (hover: none) {
    .wangp-assistant-chat__message--user .wangp-assistant-chat__copy-button,
    .wangp-assistant-chat__message--user .wangp-assistant-chat__message-action-button,
    .wangp-assistant-chat__tool-json .wangp-assistant-chat__copy-button {
        opacity: 0.72;
        pointer-events: auto;
        transform: none;
    }
}

.wangp-assistant-chat__attachments {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
    margin-top: 12px;
}

.wangp-assistant-chat__attachment {
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

.wangp-assistant-chat__attachment:hover {
    transform: translateY(-1px);
    border-color: rgba(31, 101, 141, 0.22);
    box-shadow: 0 14px 28px rgba(12, 45, 67, 0.1);
}

.wangp-assistant-chat__attachment-thumb {
    flex: 0 0 88px;
    width: 88px;
    height: 88px;
    object-fit: cover;
    border-radius: 14px;
    border: 1px solid rgba(26, 82, 114, 0.12);
    background: rgba(234, 245, 251, 0.9);
}

.wangp-assistant-chat__attachment-meta {
    min-width: 0;
}

.wangp-assistant-chat__attachment-title {
    display: block;
    font-weight: 700;
    color: #1b587e;
}

.wangp-assistant-chat__attachment-subtitle {
    display: block;
    margin-top: 4px;
    color: #667d8c;
    font-size: calc(0.84rem * var(--dock-font-scale));
    line-height: 1.45;
    word-break: break-word;
}

.wangp-assistant-chat__status {
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

.wangp-assistant-chat__status,
.wangp-assistant-chat__status-text,
.wangp-assistant-chat__status-stop {
    color: var(--status-text);
}

.wangp-assistant-chat__status.is-visible {
    opacity: 1;
    transform: translateY(0);
}

.wangp-assistant-chat__status-text {
    flex: 1;
    min-width: 0;
    font-size: calc(0.92rem * var(--dock-font-scale));
    line-height: 1.45;
    font-weight: 600;
    pointer-events: none;
}

.wangp-assistant-chat__status-dots {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    pointer-events: none;
}

.wangp-assistant-chat__status-dots span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.9);
    animation: wangp-assistant-chat-pulse 1.18s infinite ease-in-out;
}

.wangp-assistant-chat__status-dots span:nth-child(2) {
    animation-delay: 0.15s;
}

.wangp-assistant-chat__status-dots span:nth-child(3) {
    animation-delay: 0.3s;
}

.wangp-assistant-chat__status-stop {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 62px;
    min-height: 34px;
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
    pointer-events: auto;
    transition: transform 0.16s ease, background 0.16s ease, opacity 0.16s ease;
}

.wangp-assistant-chat__status-stop:hover:not(:disabled) {
    transform: translateY(-1px);
    background: rgba(197, 72, 72, 0.96);
}

.wangp-assistant-chat__status-stop:disabled {
    opacity: 0.55;
    cursor: default;
}

.wangp-assistant-chat__jump-bottom {
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

.wangp-assistant-chat__jump-bottom.is-visible {
    opacity: 1;
    pointer-events: auto;
    transform: translate(-50%, 0);
}

.wangp-assistant-chat__jump-bottom span {
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

.wangp-assistant-chat__jump-bottom:hover {
    border-color: rgba(251, 254, 255, 1);
}

.wangp-assistant-chat__jump-bottom:hover span {
    border-right-color: rgba(251, 254, 255, 1);
    border-bottom-color: rgba(251, 254, 255, 1);
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-grid {
    position: relative;
    gap: 12px;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-grid-row {
    gap: 12px;
    align-items: stretch;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-card {
    min-width: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-card > .form {
    min-width: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-row {
    gap: 10px;
    align-items: flex-end;
    flex-wrap: nowrap;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-dropdown {
    flex: 1 1 auto !important;
    min-width: 0 !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-row > .form,
#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-dropdown {
    min-width: 0 !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-row > .form,
#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-dropdown,
#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-dropdown .wrap {
    overflow: visible !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-dropdown .wrap > ul.options[role="listbox"] {
    position: absolute !important;
    inset: calc(100% - 8px) auto auto 0 !important;
    width: 100% !important;
    max-height: min(280px, 40vh) !important;
    z-index: 2147483647 !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-actions {
    flex: 0 0 auto !important;
    gap: 4px;
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-actions > .form {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-icon-btn {
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

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-icon-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 24px rgba(11, 44, 63, 0.12);
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-tool-icon-btn--danger {
    color: #8b2d2d;
    background: linear-gradient(180deg, rgba(255, 252, 252, 0.99) 0%, rgba(249, 239, 239, 0.99) 100%);
    border-color: rgba(156, 62, 62, 0.16);
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-modal-wrap.hide {
    display: none !important;
    pointer-events: none !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-modal-wrap:not(.hide) {
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

#assistant_chat_settings_panel .wangp-assistant-chat__template-modal-wrap > .form {
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

#assistant_chat_settings_panel .wangp-assistant-chat__template-modal-wrap > .styler {
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

#assistant_chat_settings_panel .wangp-assistant-chat__template-modal-card {
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

#assistant_chat_settings_panel > .wangp-assistant-chat__settings-card.wangp-assistant-chat__template-modal-card {
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

#assistant_chat_settings_panel .wangp-assistant-chat__template-modal-card > .form {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-modal-card .html-container {
    padding: 0 !important;
}

#assistant_chat_settings_panel .wangp-assistant-chat__template-modal-card .prose {
    margin: 0 !important;
    max-width: none !important;
}

.wangp-assistant-chat__template-modal-titlebar {
    padding: 10px 16px 9px;
    background: linear-gradient(180deg, rgba(16, 86, 121, 0.98) 0%, rgba(10, 59, 84, 0.98) 100%);
    color: #f3fbff;
}

.wangp-assistant-chat__template-modal-kicker {
    font-size: calc(0.66rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    opacity: 0.78;
}

.wangp-assistant-chat__template-modal-heading {
    font-size: calc(0.9rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.02em;
    color: #f3fbff !important;
}

.wangp-assistant-chat__template-modal-context {
    margin: 16px 18px 0;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
}

.wangp-assistant-chat__template-modal-context-label {
    font-size: calc(0.7rem * var(--dock-font-scale));
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5b7282;
}

.wangp-assistant-chat__template-modal-context-value {
    margin-top: 5px;
    color: #174a67;
    font-size: calc(0.95rem * var(--dock-font-scale));
    font-weight: 700;
    word-break: break-word;
}

.wangp-assistant-chat__template-modal-message {
    margin: 14px 18px 0;
    padding: 0;
    border-radius: 0;
    font-size: calc(0.9rem * var(--dock-font-scale));
    line-height: 1.5;
    font-weight: 600;
    background: transparent !important;
}

.wangp-assistant-chat__template-modal-message.is-info {
    color: #164f70;
}

.wangp-assistant-chat__template-modal-message.is-warning {
    color: #7a5415;
}

.wangp-assistant-chat__template-modal-message.is-error {
    color: #b33434;
}

.wangp-assistant-chat__template-modal-actions {
    justify-content: flex-end;
    gap: 10px;
    padding: 18px;
}

.wangp-assistant-chat__template-modal-btn {
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

.wangp-assistant-chat__template-modal-btn--primary {
    color: #f4fbff;
    border-color: rgba(10, 59, 84, 0.12);
    background: linear-gradient(180deg, rgba(16, 86, 121, 0.98) 0%, rgba(10, 59, 84, 0.98) 100%);
}

#assistant_chat_dock.is-dark #assistant_chat_toggle {
    border-color: rgba(28, 104, 145, 0.28);
    background: linear-gradient(180deg, rgba(13, 79, 113, 0.98) 0%, rgba(7, 50, 72, 0.98) 100%);
    box-shadow: 0 18px 34px rgba(0, 0, 0, 0.34);
}

#assistant_chat_dock.is-dark #assistant_chat_toggle .wangp-assistant-chat__toggle-text {
    color: #f4fbff;
}

#assistant_chat_dock.is-dark.is-open #assistant_chat_toggle {
    border-color: rgba(115, 120, 126, 0.6);
    background: linear-gradient(180deg, rgba(92, 96, 102, 0.98) 0%, rgba(58, 61, 66, 0.98) 100%);
}

#assistant_chat_dock.is-dark.is-open #assistant_chat_toggle .wangp-assistant-chat__toggle-text {
    color: #f4fbff;
}

#assistant_chat_dock.is-dark #assistant_chat_settings_toggle {
    border-color: rgba(28, 104, 145, 0.28);
    background: linear-gradient(180deg, rgba(13, 79, 113, 0.98) 0%, rgba(7, 50, 72, 0.98) 100%);
    box-shadow: 0 16px 28px rgba(0, 0, 0, 0.3);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_toggle .wangp-assistant-chat__settings-toggle-text {
    color: #f4fbff;
}

#assistant_chat_dock.is-dark #assistant_chat_panel.is-settings-open #assistant_chat_settings_toggle {
    border-color: rgba(115, 120, 126, 0.6);
    background: linear-gradient(180deg, rgba(92, 96, 102, 0.98) 0%, rgba(58, 61, 66, 0.98) 100%);
}

#assistant_chat_dock.is-dark #assistant_chat_panel.is-settings-open #assistant_chat_settings_toggle .wangp-assistant-chat__settings-toggle-text {
    color: #f4fbff;
}

#assistant_chat_dock.is-dark #assistant_chat_panel,
#assistant_chat_dock.is-dark #assistant_chat_settings_panel {
    border-color: rgba(92, 96, 102, 0.78);
    background: #000000;
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.46), inset 0 0 0 1px rgba(70, 73, 78, 0.42);
    color: #eaf2f7;
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel > .wangp-assistant-chat__settings-card > .wangp-assistant-chat__settings-scroll > .block > .label-wrap {
    border-color: rgba(112, 138, 156, 0.18);
    background: linear-gradient(180deg, rgba(9, 9, 9, 0.98) 0%, rgba(20, 20, 20, 0.98) 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel > .wangp-assistant-chat__settings-card > .wangp-assistant-chat__settings-scroll > .block > .label-wrap span {
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

#assistant_chat_dock.is-dark .wangp-assistant-chat__input-helper {
    color: rgba(174, 190, 201, 0.68);
}

#assistant_chat_dock.is-dark #assistant_chat_reset_button {
    color: #e8f1f6;
    border-color: rgba(103, 132, 151, 0.22);
    background: linear-gradient(180deg, rgba(12, 12, 12, 0.98) 0%, rgba(22, 22, 22, 0.98) 100%);
    box-shadow: 0 12px 22px rgba(0, 0, 0, 0.22);
}

#assistant_chat_dock.is-dark .wangp-assistant-chat {
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

#assistant_chat_dock.is-dark .wangp-assistant-chat__empty-intro,
#assistant_chat_dock.is-dark .wangp-assistant-chat__empty-section li,
#assistant_chat_dock.is-dark .wangp-assistant-chat__empty-tip,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body p,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body li,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body strong,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body em,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body blockquote,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body h1,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body h2,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body h3,
#assistant_chat_dock.is-dark .wangp-assistant-chat__body h4 {
    color: #edf4f9;
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__empty-header {
    border-color: rgba(100, 171, 205, 0.28);
    background: linear-gradient(135deg, rgba(7, 49, 72, 0.98) 0%, rgba(10, 75, 104, 0.96) 58%, rgba(18, 98, 125, 0.92) 100%);
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__empty-section {
    border-color: rgba(103, 132, 151, 0.2);
    background: linear-gradient(180deg, rgba(18, 24, 29, 0.98) 0%, rgba(10, 15, 19, 0.98) 100%);
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__empty-section h3 {
    color: #9edaf3;
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__stats {
    color: #9eb0bd;
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__message--user .wangp-assistant-chat__avatar {
    color: #eef6fb;
    background: linear-gradient(180deg, rgba(24, 31, 37, 0.99) 0%, rgba(10, 12, 14, 0.99) 100%);
    border-color: rgba(103, 132, 151, 0.2);
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__message--user .wangp-assistant-chat__copy-button {
    color: #b9d9ea;
    border-color: rgba(148, 185, 205, 0.2);
    background: rgba(255, 255, 255, 0.07);
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__body code {
    background: rgba(130, 162, 183, 0.12);
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__body pre {
    color: #eaf2f7;
    border-color: rgba(103, 132, 151, 0.16);
    background: rgba(10, 14, 17, 0.96);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel .wangp-assistant-chat__template-tool-icon-btn,
#assistant_chat_dock.is-dark .wangp-assistant-chat__template-modal-btn {
    color: #ecf4f9;
    border-color: rgba(103, 132, 151, 0.22);
    background: linear-gradient(180deg, rgba(10, 10, 10, 0.99) 0%, rgba(21, 21, 21, 0.99) 100%);
    box-shadow: 0 10px 18px rgba(0, 0, 0, 0.22);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel .wangp-assistant-chat__template-tool-icon-btn--danger {
    color: #ffb1b1;
    border-color: rgba(173, 84, 84, 0.24);
    background: linear-gradient(180deg, rgba(22, 10, 10, 0.99) 0%, rgba(32, 14, 14, 0.99) 100%);
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel .wangp-assistant-chat__template-modal-wrap:not(.hide) {
    background: rgba(0, 0, 0, 0.52) !important;
}

#assistant_chat_dock.is-dark #assistant_chat_settings_panel .wangp-assistant-chat__template-modal-card {
    border-color: rgba(92, 96, 102, 0.82) !important;
    background: #000000 !important;
    box-shadow: 0 28px 56px rgba(0, 0, 0, 0.42), inset 0 0 0 1px rgba(70, 73, 78, 0.44) !important;
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__template-modal-context-label {
    color: #9fb1be;
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__template-modal-context-value,
#assistant_chat_dock.is-dark .wangp-assistant-chat__template-modal-message.is-info {
    color: #ecf4f9;
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__template-modal-message.is-warning {
    color: #f3d189;
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__template-modal-message.is-error {
    color: #ff9e9e;
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

    .wangp-assistant-chat {
        height: 449px;
        border-radius: 20px;
    }

    #assistant_chat_html {
        min-height: 449px;
    }

    .wangp-assistant-chat__scroll {
        padding: 0;
    }

    .wangp-assistant-chat__message-card {
        width: min(92%, 100%);
        padding: 14px 14px 12px;
    }

    .wangp-assistant-chat__avatar {
        width: 46px;
        height: 46px;
        margin-top: 9px;
    }

    .wangp-assistant-chat__empty {
        padding: 18px 14px 14px;
    }

    .wangp-assistant-chat__empty-grid {
        grid-template-columns: 1fr;
    }

    .wangp-assistant-chat__empty-header {
        grid-template-columns: 1fr;
    }

    .wangp-assistant-chat__empty-mode {
        max-width: none;
        justify-self: start;
        margin-top: 5px;
    }

    .wangp-assistant-chat__transcript {
        padding: 16px 12px calc(var(--chat-status-offset) + var(--chat-status-reserved-height) + var(--chat-status-gap));
    }

    .wangp-assistant-chat__attachments {
        grid-template-columns: 1fr;
    }

    .wangp-assistant-chat__attachment-thumb {
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

    .wangp-assistant-chat__settings-toggle-text {
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

    #assistant_chat_settings_panel .wangp-assistant-chat__template-tool-grid-row,
    #assistant_chat_settings_panel .wangp-assistant-chat__template-tool-row {
        flex-wrap: wrap;
    }

    #assistant_chat_settings_panel .wangp-assistant-chat__template-tool-actions {
        width: 100%;
        min-width: 0 !important;
        max-width: none !important;
        flex-direction: row;
    }

    #assistant_chat_settings_panel .wangp-assistant-chat__template-tool-actions > .form {
        width: 100%;
    }

    #assistant_chat_settings_panel .wangp-assistant-chat__template-tool-icon-btn {
        flex: 1 1 calc(50% - 4px);
    }

    #assistant_chat_settings_panel .wangp-assistant-chat__template-modal-wrap {
        inset: 0;
        padding: 8px !important;
    }

    #assistant_chat_settings_panel .wangp-assistant-chat__template-modal-wrap > .styler {
        width: 100% !important;
        max-width: none !important;
    }

    #assistant_chat_settings_panel .wangp-assistant-chat__template-modal-card {
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
window.WAC = WAC;

WAC.state = WAC.state || { order: [], messages: {}, status: null, stats: null };
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
  const scroll = WAC.scroll();
  if (!scroll) return { atBottom: true, top: 0 };
  return {
    atBottom: WAC.isNearBottom(),
    top: Math.max(0, scroll.scrollTop),
  };
};

WAC.applyAutoscrollState = function (state) {
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

WAC.buildOptimisticUserMessage = function (optimisticId, content, timestamp) {
  const contentHtml = WAC.escapeHtml(content).replace(/\n/g, '<br>');
  const copyButton = `<button type='button' class='wangp-assistant-chat__copy-button' data-copy-source='user' data-copy-text='${WAC.escapeHtml(content)}' aria-label='Copy request' title='Copy request'><svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><rect x='5' y='5' width='8' height='8' rx='1.5'></rect><path d='M3.5 10.5H3A1.5 1.5 0 0 1 1.5 9V3A1.5 1.5 0 0 1 3 1.5h6A1.5 1.5 0 0 1 10.5 3v.5'></path></svg></button>`;
  const html = [
    `<article class='wangp-assistant-chat__message wangp-assistant-chat__message--user' data-message-id='${optimisticId}'>`,
    "<div class='wangp-assistant-chat__avatar'>You</div>",
    "<div class='wangp-assistant-chat__message-card'>",
    "<div class='wangp-assistant-chat__meta'><div class='wangp-assistant-chat__meta-left'></div>",
    `<div class='wangp-assistant-chat__meta-right'><div class='wangp-assistant-chat__message-actions'>${copyButton}</div><div class='wangp-assistant-chat__time'>${WAC.escapeHtml(WAC.timeLabel(timestamp))}</div></div></div>`,
    `<div class='wangp-assistant-chat__body'><p>${contentHtml}</p></div>`,
    "</div></article>",
  ].join('');
  return { id: optimisticId, role: 'user', html };
};

WAC.dropOptimisticSubmit = function (optimisticId) {
  const targetId = String(optimisticId || '');
  WAC.optimisticSubmits = (WAC.optimisticSubmits || []).filter((item) => String(item && item.id || '') !== targetId);
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
    WAC.state.order.push(optimisticId);
    WAC.state.messages[optimisticId] = WAC.buildOptimisticUserMessage(optimisticId, content, item.ts);
  }
};

WAC.newSubmissionId = function () {
  const randomId = window.crypto && typeof window.crypto.randomUUID === 'function' ? window.crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`;
  return `optimistic_${randomId}`;
};

WAC.pushOptimisticUserMessage = function (text) {
  const content = WAC.normalizeText(text);
  if (!content) return '';
  const now = Date.now();
  const optimisticId = WAC.newSubmissionId();
  WAC.optimisticSubmits.push({ id: optimisticId, text: content, ts: now });
  WAC.upsertMessage(WAC.buildOptimisticUserMessage(optimisticId, content, now));
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
  return document.querySelector('#assistant_chat_html .wangp-assistant-chat');
};

WAC.scroll = function () {
  return document.querySelector('#assistant_chat_html .wangp-assistant-chat__scroll');
};

WAC.transcript = function () {
  return document.querySelector('#assistant_chat_html .wangp-assistant-chat__transcript');
};

WAC.empty = function () {
  return document.querySelector('#assistant_chat_html .wangp-assistant-chat__empty');
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
  const scrollState = WAC.captureAutoscrollState();
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
};

WAC.emptyMarkup = function (mode) {
  if (mode === 'prime') {
    return `<div class="wangp-assistant-chat__empty-card">
      <header class="wangp-assistant-chat__empty-header">
        <span class="wangp-assistant-chat__empty-eyebrow">Current assistant</span>
        <h2 class="wangp-assistant-chat__empty-title">Deepy Prime</h2>
        <span class="wangp-assistant-chat__empty-mode">Advanced creative orchestration</span>
      </header>
      <p class="wangp-assistant-chat__empty-intro">Describe the result you want and Deepy Prime can plan the work, choose suitable models and tools, and connect multiple image, video, and audio steps into one creative workflow.</p>
      <div class="wangp-assistant-chat__empty-grid">
        <section class="wangp-assistant-chat__empty-section"><h3>What it does for you</h3><ul>
          <li>Plan and complete multi-step projects that create, inspect, edit, and combine several pieces of media.</li>
          <li>Choose among available WanGP models and settings according to your goal, quality preference, and source media.</li>
          <li>Build on Gallery items or existing files, then extract, transcribe, resize, add sound, upscale, or continue generating.</li>
          <li>Extend the workflow with other connected services when they are available.</li>
        </ul></section>
        <section class="wangp-assistant-chat__empty-section wangp-assistant-chat__empty-section--examples"><h3>Try asking</h3><ul>
          <li>Create a character portrait and related keyframes, then turn them into a longer video with a soundtrack.</li>
          <li>Inspect the selected video, improve the weak sections, upscale it, and prepare a subtitled version.</li>
          <li>Design an album cover, write a matching song, and create a short promotional video from both.</li>
        </ul></section>
      </div>
      <p class="wangp-assistant-chat__empty-tip">Start with the outcome you want. Deepy will ask only when an important choice is missing.</p>
    </div>`;
  }
  return `<div class="wangp-assistant-chat__empty-card">
    <header class="wangp-assistant-chat__empty-header">
      <span class="wangp-assistant-chat__empty-eyebrow">Current assistant</span>
      <h2 class="wangp-assistant-chat__empty-title">Deepy Zero</h2>
      <span class="wangp-assistant-chat__empty-mode">Fast, focused creation</span>
    </header>
    <p class="wangp-assistant-chat__empty-intro">Deepy Zero is the lightweight assistant for straightforward requests. It uses the models and templates selected in Deepy Settings, making it a good match for smaller LLMs, quick responses, and familiar results.</p>
    <div class="wangp-assistant-chat__empty-grid">
      <section class="wangp-assistant-chat__empty-section"><h3>What it does for you</h3><ul>
        <li>Generate an image, video, speech clip, or song with your preferred templates and defaults.</li>
        <li>Handle focused edits and practical media tasks without requiring a complex workflow.</li>
        <li>Refer naturally to the selected, latest, or previous Gallery item.</li>
        <li>See each generation and completed result in the normal WanGP queue and Galleries.</li>
      </ul></section>
      <section class="wangp-assistant-chat__empty-section wangp-assistant-chat__empty-section--examples"><h3>Try asking</h3><ul>
        <li>Generate a square album cover showing a robot jazz band.</li>
        <li>Animate the selected image as a five-second cinematic shot.</li>
        <li>Transcribe the last video or resize it for social media.</li>
      </ul></section>
    </div>
    <p class="wangp-assistant-chat__empty-tip">Start with the outcome you want. Deepy will ask only when an important choice is missing.</p>
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
  return document.querySelector('#assistant_chat_html .wangp-assistant-chat__status');
};

WAC.jumpBottomNode = function () {
  return document.querySelector('#assistant_chat_html .wangp-assistant-chat__jump-bottom');
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
  const scope = root || WAC.transcript();
  if (!scope || !scope.querySelectorAll) return;
  scope.querySelectorAll('.wangp-assistant-chat__disclosure').forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (!key) return;
    WAC.disclosureState[key] = !!node.open;
  });
};

WAC.applyDisclosureState = function (root) {
  const scope = root || WAC.transcript();
  if (!scope || !scope.querySelectorAll) return;
  scope.querySelectorAll('.wangp-assistant-chat__disclosure').forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (!key || !(key in WAC.disclosureState)) return;
    node.open = !!WAC.disclosureState[key];
  });
};

WAC.handleDisclosureToggle = function (event) {
  const node = event && event.target;
  if (!node || !node.classList || !node.classList.contains('wangp-assistant-chat__disclosure')) return;
  const key = WAC.disclosureKey(node);
  if (!key) return;
  WAC.disclosureState[key] = !!node.open;
};

WAC.toggleDisclosure = function (node) {
  if (!node || !node.classList || !node.classList.contains('wangp-assistant-chat__disclosure')) return;
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
  const button = event && event.target && event.target.closest ? event.target.closest('.wangp-assistant-chat__copy-button') : null;
  if (!button) return false;
  event.preventDefault();
  event.stopPropagation();
  const source = String(button.getAttribute('data-copy-source') || '');
  const jsonNode = source === 'json' ? button.closest('.wangp-assistant-chat__tool-json') : null;
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
  WAC.closeDisclosure(button.closest('.wangp-assistant-chat__disclosure'));
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
  WAC.closeDisclosure(button.closest('.wangp-assistant-chat__disclosure'));
  return true;
};

WAC.handleDisclosurePointerDown = function (event) {
  const summary = event && event.target && event.target.closest ? event.target.closest('summary') : null;
  if (!summary) return false;
  const disclosureNode = summary.parentElement;
  if (!disclosureNode || !disclosureNode.classList || !disclosureNode.classList.contains('wangp-assistant-chat__disclosure')) return false;
  event.preventDefault();
  event.stopPropagation();
  WAC.toggleDisclosure(disclosureNode);
  return true;
};

WAC.handleAttachmentPointerDown = function (event) {
  const link = event && event.target && event.target.closest ? event.target.closest('a.wangp-assistant-chat__attachment') : null;
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
  if (resetButton) resetButton.textContent = editing ? 'Cancel' : 'Reset';
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
  if (!messageId || !['edit', 'remove'].includes(normalizedAction) || (normalizedAction === 'edit' && !content)) return;
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
  const messageNode = messageAction.closest('.wangp-assistant-chat__message--user');
  if (!messageNode) return false;
  event.preventDefault();
  event.stopPropagation();
  const action = String(messageAction.getAttribute('data-message-action') || '');
  if (action === 'edit') WAC.startQueuedRequestEdit(messageNode);
  else if (action === 'remove') WAC.submitQueuedRequestAction(messageNode, 'remove', '');
  return true;
};

WAC.isAssistantBusy = function () {
  if (WAC.state && WAC.state.status && WAC.state.status.visible && WAC.state.status.text) return true;
  const stopButton = document.querySelector('#assistant_chat_html .wangp-assistant-chat__status-stop');
  return !!(stopButton && !stopButton.disabled);
};

WAC.setBusyInputHelper = function (visible) {
  const stats = WAC.statsNode();
  if (!stats) return;
  let helper = stats.querySelector('.wangp-assistant-chat__input-helper');
  if (!helper) {
    helper = document.createElement('span');
    helper.className = 'wangp-assistant-chat__input-helper';
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
    try {
      envelope = JSON.parse(payload);
    } catch (_error) {
      return [];
    }
  }
  const payloadId = envelope && typeof envelope.event_id === 'string' ? envelope.event_id : '';
  const payloadText = typeof payload === 'string' ? payload : JSON.stringify(envelope);
  if ((payloadId && payloadId === WAC.lastPayloadId) || (!payloadId && payloadText === WAC.lastPayloadText)) return [];
  WAC.lastPayloadId = payloadId;
  WAC.lastPayloadText = payloadText;
  if (Array.isArray(envelope.batch)) {
    for (const item of envelope.batch) WAC.consumePayload(item);
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
  const chatSessionId = typeof event.chat_session_id === 'string' ? event.chat_session_id : '';
  if (chatSessionId && WAC.chatSessionId && chatSessionId !== WAC.chatSessionId) WAC.reset();
  if (chatSessionId) WAC.chatSessionId = chatSessionId;
  const transcriptEvent = event.type === 'sync' || event.type === 'upsert_message' || event.type === 'remove_message';
  const revision = Number(event.revision);
  const hasRevision = transcriptEvent && Number.isFinite(revision);
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
  if (hasRevision) WAC.chatRevision = revision;
  if (event.type === 'reset') {
    WAC.reset();
    if (chatSessionId) WAC.chatSessionId = chatSessionId;
    if (Number.isFinite(revision)) WAC.chatRevision = revision;
    return [];
  }
  if (event.type === 'upsert_message') {
    const message = event.message || {};
    const submissionId = String(message.client_submission_id || '').trim();
    if (submissionId) {
      WAC.acknowledgeOptimisticSubmits([submissionId]);
      if (submissionId === WAC.pendingSteeringId) {
        WAC.pendingSteeringId = '';
        WAC.setStatus({ visible: true, kind: 'queued', text: 'Steering accepted. Waiting for the current thought/action boundary...' });
      }
    }
    WAC.upsertMessage(message);
    return [];
  }
  if (event.type === 'remove_message') {
    WAC.removeMessage(event.message_id);
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

WAC.handleEventNodeMutation = function () {
  const node = WAC.eventSource();
  if (!node || node === WAC.eventNode) return;
  WAC.eventNode = node;
  const handler = function () { WAC.readEventSource(); };
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
  const maxSettingsWidth = Math.max(panelWidth, window.innerWidth - panelLeft - 44);
  const settingsWidth = Math.min(maxSettingsWidth, Math.max(panelWidth, Math.min(panelWidth + 112, 660)));
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
};

WAC.toggleSettings = function (forceOpen) {
  const nextOpen = typeof forceOpen === 'boolean' ? forceOpen : !WAC.settingsOpen;
  WAC.setSettingsOpen(nextOpen);
};

WAC.ensureShell = function () {
  const host = WAC.host();
  if (!host) return false;
  if (host.dataset.wangpAssistantChatMounted === 'true' && WAC.shell()) {
    WAC.showEmptyIfNeeded();
    WAC.syncDockState();
    WAC.syncDockLayout();
    WAC.syncComposerLayout();
    return true;
  }
  host.innerHTML = `
    <section class="wangp-assistant-chat">
      <div class="wangp-assistant-chat__scroll">
        <div class="wangp-assistant-chat__empty">
          ${WAC.emptyMarkup('zero')}
        </div>
        <div class="wangp-assistant-chat__transcript"></div>
      </div>
      <div class="wangp-assistant-chat__status" aria-live="polite">
        <div class="wangp-assistant-chat__status-dots" aria-hidden="true"><span></span><span></span><span></span></div>
        <div class="wangp-assistant-chat__status-text"></div>
        <button class="wangp-assistant-chat__status-stop" type="button" aria-label="Stop Deepy" disabled>Stop</button>
      </div>
      <button class="wangp-assistant-chat__jump-bottom" type="button" aria-label="Jump to latest messages" aria-hidden="true" tabindex="-1">
        <span aria-hidden="true"></span>
      </button>
    </section>
  `;
  host.dataset.wangpAssistantChatMounted = 'true';
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

WAC.hideEmpty = function () {
  const empty = WAC.empty();
  if (empty) empty.style.display = 'none';
};

WAC.showEmptyIfNeeded = function () {
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
  const currentBody = current.querySelector(':scope > .wangp-assistant-chat__disclosure-body');
  const nextBody = next.querySelector(':scope > .wangp-assistant-chat__disclosure-body');
  if (currentBody && nextBody && currentBody.innerHTML !== nextBody.innerHTML) currentBody.innerHTML = nextBody.innerHTML;
};

WAC.patchMessageBody = function (currentBody, nextBody) {
  if (!currentBody || !nextBody) return;
  const existingByKey = new Map();
  currentBody.querySelectorAll(':scope > .wangp-assistant-chat__disclosure').forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (key) existingByKey.set(key, node);
  });
  let cursor = currentBody.firstChild;
  for (const nextNode of Array.from(nextBody.childNodes)) {
    const key = nextNode.nodeType === 1 && nextNode.classList.contains('wangp-assistant-chat__disclosure') ? WAC.disclosureKey(nextNode) : '';
    const reusable = key ? existingByKey.get(key) : null;
    if (reusable) {
      WAC.patchDisclosureNode(reusable, nextNode);
      existingByKey.delete(key);
      if (reusable === cursor) cursor = cursor.nextSibling;
      else currentBody.insertBefore(reusable, cursor);
      continue;
    }
    const cursorIsDisclosure = cursor && cursor.nodeType === 1 && cursor.classList.contains('wangp-assistant-chat__disclosure');
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
  const currentAvatar = current.querySelector(':scope > .wangp-assistant-chat__avatar');
  const nextAvatar = next.querySelector(':scope > .wangp-assistant-chat__avatar');
  if (currentAvatar && nextAvatar) {
    WAC.syncAttributes(currentAvatar, nextAvatar);
    if (currentAvatar.innerHTML !== nextAvatar.innerHTML) currentAvatar.innerHTML = nextAvatar.innerHTML;
  }
  const currentCard = current.querySelector(':scope > .wangp-assistant-chat__message-card');
  const nextCard = next.querySelector(':scope > .wangp-assistant-chat__message-card');
  if (!currentCard || !nextCard) {
    current.replaceChildren(...Array.from(next.childNodes));
    return;
  }
  WAC.syncAttributes(currentCard, nextCard);
  currentCard.className = nextCard.className;
  const currentMeta = currentCard.querySelector(':scope > .wangp-assistant-chat__meta');
  const nextMeta = nextCard.querySelector(':scope > .wangp-assistant-chat__meta');
  if (currentMeta && nextMeta) {
    WAC.syncAttributes(currentMeta, nextMeta);
    currentMeta.className = nextMeta.className;
    if (currentMeta.innerHTML !== nextMeta.innerHTML) currentMeta.innerHTML = nextMeta.innerHTML;
  }
  const currentBody = currentCard.querySelector(':scope > .wangp-assistant-chat__body');
  const nextBody = nextCard.querySelector(':scope > .wangp-assistant-chat__body');
  if (currentBody && nextBody) {
    WAC.syncAttributes(currentBody, nextBody);
    currentBody.className = nextBody.className;
    WAC.patchMessageBody(currentBody, nextBody);
  }
  const currentEnd = currentCard.querySelector(':scope > .wangp-assistant-chat__message-end');
  const nextEnd = nextCard.querySelector(':scope > .wangp-assistant-chat__message-end');
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
  const body = node && node.querySelector ? node.querySelector('.wangp-assistant-chat__body') : null;
  return body ? WAC.normalizeText(body.innerText || body.textContent || '') : '';
};

WAC.upsertMessage = function (message) {
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
  if (existing) {
    WAC.patchMessageNode(existing, node);
  } else {
    WAC.state.order.push(incomingId);
    transcript.appendChild(node);
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
  const textNode = node.querySelector('.wangp-assistant-chat__status-text');
  const stopNode = node.querySelector('.wangp-assistant-chat__status-stop');
  if (!status || !status.visible || !status.text) {
    node.classList.remove('is-visible');
    node.removeAttribute('data-kind');
    if (textNode) textNode.textContent = '';
    if (stopNode) stopNode.disabled = true;
    WAC.setBusyInputHelper(false);
    WAC.applyAutoscrollState(scrollState);
    return;
  }
  if (textNode) textNode.textContent = String(status.text);
  node.dataset.kind = String(status.kind || 'status');
  if (stopNode) stopNode.disabled = false;
  node.classList.add('is-visible');
  WAC.setBusyInputHelper(true);
  WAC.applyAutoscrollState(scrollState);
};

WAC.setStats = function (stats) {
  WAC.ensureShell();
  WAC.state.stats = stats || null;
  const node = WAC.statsNode();
  if (!node) return;
  let textNode = node.querySelector('.wangp-assistant-chat__stats-text');
  if (!textNode) {
    textNode = document.createElement('span');
    textNode.className = 'wangp-assistant-chat__stats-text';
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
  const scrollState = WAC.captureAutoscrollState();
  WAC.replaceState(messages, status, stats);
  WAC.reconcileOptimisticSubmits(acknowledgedSubmissionIds);
  WAC.hydrate(scrollState);
};

WAC.reset = function () {
  if (WAC.queuedEditMessageId) WAC.finishQueuedRequestEdit();
  WAC.state = { order: [], messages: {}, status: null, stats: null };
  WAC.optimisticSubmits = [];
  WAC.pendingSteeringId = '';
  WAC.chatSessionId = '';
  WAC.chatRevision = -1;
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
    const attachmentLink = event.target && event.target.closest ? event.target.closest('.wangp-assistant-chat__attachment, .wangp-assistant-chat__body a') : null;
    if (attachmentLink) return;
    const disclosureSummary = event.target && event.target.closest ? event.target.closest('summary') : null;
    if (disclosureSummary) {
      const disclosureNode = disclosureSummary.parentElement;
      if (disclosureNode && disclosureNode.classList && disclosureNode.classList.contains('wangp-assistant-chat__disclosure')) {
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
    const stopButton = event.target && event.target.closest ? event.target.closest('.wangp-assistant-chat__status-stop') : null;
    if (stopButton) {
      event.preventDefault();
      event.stopPropagation();
      const target = WAC.stopBridgeTargets()[0];
      if (target && typeof target.click === 'function') target.click();
      return;
    }
    const jumpBottomButton = event.target && event.target.closest ? event.target.closest('.wangp-assistant-chat__jump-bottom') : null;
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
    const submissionId = WAC.pushOptimisticUserMessage(text);
    WAC.setBusyInputHelper(true);
    WAC.setBridgeValue('#assistant_chat_submission_id textarea, #assistant_chat_submission_id input', submissionId);
    if (WAC.isAssistantBusy()) {
      event.preventDefault();
      event.stopPropagation();
      WAC.clearRequestInput(text);
      WAC.queueBusyRequest(text, submissionId);
      return;
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
      const submissionId = WAC.pushOptimisticUserMessage(text);
      WAC.pendingSteeringId = submissionId;
      WAC.setStatus({ visible: true, kind: 'queued', text: 'Steering requested. Waiting for the current thought/action boundary...' });
      WAC.setBusyInputHelper(true);
      window.setTimeout(() => { if (WAC.pendingSteeringId === submissionId) WAC.pendingSteeringId = ''; }, WAC.optimisticMaxAgeMs);
      window.setTimeout(() => { WAC.clearRequestInput(text); }, 0);
      WAC.steerRequest(text, submissionId);
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
    _touch_chat(session)


def build_reset_event(session=None) -> str:
    return _event_payload({"type": "reset"}, session)


def build_status_event(text: str | None, kind: str = "status", visible: bool = True, stats: dict[str, Any] | None = None) -> str:
    status = None if not visible or not text else {"visible": True, "kind": str(kind or "status"), "text": str(text or "").strip()}
    event = {"type": "status", "status": status}
    if stats is not None:
        event["stats"] = stats
    return _event_payload(event)


def build_stats_event(stats: dict[str, Any] | None = None) -> str:
    return _event_payload({"type": "stats", "stats": stats})


def build_event_batch(payloads: list[str]) -> str:
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
    return json.dumps({"event_id": uuid.uuid4().hex, "instance_id": SERVER_INSTANCE_ID, "batch": envelopes}, ensure_ascii=False)


def _message_has_renderable_output(record: dict[str, Any]) -> bool:
    return str(record.get("role", "")).strip() != "assistant" or bool(_ensure_message_blocks(record)) or bool(record.get("attachments"))


def build_sync_event(session, status: dict[str, Any] | None = None, stats: dict[str, Any] | None = None, acknowledged_submission_ids: list[str] | tuple[str, ...] | None = None) -> str:
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
    return record["id"], _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


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
    return record["id"], _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


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
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


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
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


def set_message_end_badge(session, message_id: str, badge: str | None) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    record["end_badge"] = str(badge or "").strip()
    revision = _touch_chat(session)
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


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
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


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
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


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
    _reasoning_id, payload = upsert_reasoning_block(session, message_id, None, text)
    return payload


def add_context_summary(session, message_id: str, text: str) -> tuple[str, str | None]:
    summary_text = str(text or "").strip()
    if len(summary_text) == 0:
        return "", None
    record = _find_message(session, message_id)
    if record is None:
        return "", None
    block_id = _next_block_id("context_summary")
    _ensure_message_blocks(record).append({"id": block_id, "type": "context_summary", "text": summary_text})
    revision = _touch_chat(session)
    return block_id, _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


def upsert_reasoning_block(session, message_id: str, reasoning_id: str | None, text: str) -> tuple[str, str | None]:
    reasoning_text = str(text or "").strip()
    if len(reasoning_text) == 0:
        return "", None
    record = _find_message(session, message_id)
    if record is None:
        return "", None
    blocks = _ensure_message_blocks(record)
    target_id = str(reasoning_id or "").strip()
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "reasoning" or block.get("id", "") != target_id:
            continue
        if str(block.get("text", "")).strip() == reasoning_text:
            return target_id, None
        block["text"] = reasoning_text
        revision = _touch_chat(session)
        return target_id, _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)
    target_id = target_id or f"reasoning_{uuid.uuid4().hex[:10]}"
    blocks.append({"id": target_id, "type": "reasoning", "text": reasoning_text})
    revision = _touch_chat(session)
    return target_id, _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


def add_tool_call(session, message_id: str, tool_name: str, arguments: dict[str, Any], tool_label: str | None = None) -> tuple[str, str | None]:
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
        "status_text": "Running",
        "attachment": None,
    }
    _ensure_message_blocks(record).append(tool_record)
    revision = _touch_chat(session)
    return tool_record["id"], _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


def update_tool_call(session, message_id: str, tool_id: str, status: str | None = None, result: dict[str, Any] | object = _UNSET, status_text: str | None = None) -> str | None:
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
        if result is not _UNSET:
            tool_record["result"] = None if result is None else dict(result or {})
            tool_record["attachment"] = _attachment_from_tool_result(tool_record.get("result"))
        revision = _touch_chat(session)
        return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)
    return None


def complete_tool_call(session, message_id: str, tool_id: str, result: dict[str, Any]) -> str | None:
    status = str((result or {}).get("status", "")).strip().lower()
    failed = status in {"error", "failed", "interrupted"}
    return update_tool_call(session, message_id, tool_id, status="error" if failed else "done", result=result, status_text="Interrupted" if status == "interrupted" else ("Error" if failed else "Done"))


def upsert_assistant_content_block(session, message_id: str, content_id: str | None, text: str) -> tuple[str, str | None]:
    content_text = str(text or "").strip()
    if len(content_text) == 0:
        return "", None
    record = _find_message(session, message_id)
    if record is None:
        return "", None
    blocks = _ensure_message_blocks(record)
    target_id = str(content_id or "").strip()
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "markdown" or block.get("id", "") != target_id:
            continue
        if str(block.get("text", "")).strip() == content_text:
            return target_id, None
        block["text"] = content_text
        revision = _touch_chat(session)
        return target_id, _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)
    target_id = target_id or _next_block_id("content")
    blocks.append({"id": target_id, "type": "markdown", "text": content_text})
    revision = _touch_chat(session)
    return target_id, _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


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
            return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)
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
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)}, session, revision)


def _next_message_id(session, prefix: str) -> str:
    session.chat_transcript_counter += 1
    return f"{prefix}_{session.chat_transcript_counter}"


def _next_tool_id() -> str:
    return f"tool_{uuid.uuid4().hex[:10]}"


def _next_block_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


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
        "list_loras": "List LoRAs for",
    }
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
            return _finish_tool_call_label("Inspect Media" if count == 0 else "Inspect Visual" if count == 1 else f"Inspect {count} Visuals")
        if images and frames:
            image_text = "Image" if images == 1 else f"{images} Images"
            frame_text = "Frame" if frames == 1 else f"{frames} Frames"
            return _finish_tool_call_label(f"Inspect {image_text} and {frame_text}")
        if images:
            return _finish_tool_call_label("Inspect Image" if images == 1 else f"Inspect {images} Images")
        frame_text = "Frame" if frames == 1 else f"{frames} Frames"
        if len(video_names) == frames and len(set(video_names)) == 1:
            return _finish_tool_call_label(f"Inspect {frame_text} from {video_names[0]}")
        return _finish_tool_call_label(f"Inspect {frame_text}" if frames == 1 else f"Inspect {frames} Video Frames")
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
    if normalized_name == "mcp_list_resources":
        server = _short_tool_label_value(arguments.get("server"))
        return _finish_tool_call_label("List MCP Documents" if not server else f"List {server} Documents")
    if normalized_name == "mcp_read_resource":
        target = _humanize_tool_value(arguments.get("uri"))
        section = _short_tool_label_value(arguments.get("section"))
        return _finish_tool_call_label("Read MCP Document" if not target else f"Read {section} from {target}" if section else f"Read {target}")
    if normalized_name == "mcp_search_resource":
        query = _short_tool_label_value(arguments.get("query"))
        return _finish_tool_call_label("Search MCP Documents" if not query else f'Search MCP Documents for “{query}”')

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
        payload["chat_session_id"] = str(session.chat_session_id)
        payload["revision"] = int(session.chat_revision if revision is None else revision)
    return json.dumps({"event_id": uuid.uuid4().hex, "instance_id": SERVER_INSTANCE_ID, "event": payload}, ensure_ascii=False)


def _markdown_to_html(text: str) -> str:
    text = str(text or "").strip()
    if len(text) == 0:
        return ""
    text = html.escape(text, quote=False)
    rendered = markdown.markdown(text, extensions=_MARKDOWN_EXTENSIONS, output_format="html5")
    return re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener noreferrer"', rendered)


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


def _attachment_from_tool_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    output_file = str(result.get("output_file", "")).strip()
    if len(output_file) == 0:
        return None
    ext = os.path.splitext(output_file)[1].lower()
    label = "Generated image" if ext in _IMAGE_EXTENSIONS else ("Generated video" if ext in _VIDEO_EXTENSIONS else ("Generated audio" if ext in _AUDIO_EXTENSIONS else "Generated file"))
    return _attachment_from_path(output_file, label)


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
    kind = "file"
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
        if os.path.isfile(_AUDIO_THUMBNAIL_PATH):
            audio_thumb_path = os.path.normpath(_AUDIO_THUMBNAIL_PATH).replace("\\", "/")
            thumb_url = f"/gradio_api/file={urllib.parse.quote(audio_thumb_path, safe='/')}"
    return {
        "path_key": path_key,
        "href": href,
        "label": resolved_label,
        "subtitle": subtitle,
        "kind": kind,
        "thumb_url": thumb_url,
    }


def _render_copy_button(source: str, label: str, text: str | None = None) -> str:
    copy_text = "" if text is None else f" data-copy-text='{html.escape(str(text), quote=True)}'"
    return (
        f"<button type='button' class='wangp-assistant-chat__copy-button' data-copy-source='{html.escape(source, quote=True)}'{copy_text} aria-label='{html.escape(label, quote=True)}' title='{html.escape(label, quote=True)}'>"
        "<svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><rect x='5' y='5' width='8' height='8' rx='1.5'></rect><path d='M3.5 10.5H3A1.5 1.5 0 0 1 1.5 9V3A1.5 1.5 0 0 1 3 1.5h6A1.5 1.5 0 0 1 10.5 3v.5'></path></svg>"
        "</button>"
    )


def _render_queued_request_actions() -> str:
    edit_label = html.escape("Edit queued request", quote=True)
    remove_label = html.escape("Remove queued request", quote=True)
    return (
        f"<button type='button' class='wangp-assistant-chat__message-action-button' data-message-action='edit' aria-label='{edit_label}' title='{edit_label}'>"
        "<svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M3 11.8 3.5 9l6.8-6.8a1.4 1.4 0 0 1 2 0l1.5 1.5a1.4 1.4 0 0 1 0 2L7 12.5l-2.8.5Z'></path><path d='m9.4 3.1 3.5 3.5'></path></svg>"
        "</button>"
        f"<button type='button' class='wangp-assistant-chat__message-action-button' data-message-action='remove' aria-label='{remove_label}' title='{remove_label}'>"
        "<svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M3 4.5h10M6 4.5V2.7h4v1.8M4.7 4.5l.6 8.3h5.4l.6-8.3M7 7v3.4M9 7v3.4'></path></svg>"
        "</button>"
    )


def _render_collapse_button(label: str) -> str:
    escaped_label = html.escape(f"Collapse {label}", quote=True)
    return f"<button type='button' class='wangp-assistant-chat__collapse-button' data-disclosure-action='collapse' aria-label='{escaped_label}' title='{escaped_label}'><span aria-hidden='true'>▾</span></button>"


def _render_message_payload(record: dict[str, Any]) -> dict[str, Any]:
    role = str(record.get("role", "assistant"))
    badge_text = str(record.get("badge", "")).strip()
    end_badge_text = str(record.get("end_badge", "")).strip()
    blocks_html, rendered_attachment_keys = _render_message_blocks(record)
    attachments_html = _render_attachments(
        [
            attachment
            for attachment in list(record.get("attachments", []))
            if isinstance(attachment, dict) and (attachment.get("path_key", "") or attachment.get("href", "")) not in rendered_attachment_keys
        ]
    )
    badge_html = "" if len(badge_text) == 0 else f"<span class='wangp-assistant-chat__badge'>{html.escape(badge_text)}</span>"
    copy_text = "\n\n".join(str(block.get("text", "")).strip() for block in _ensure_message_blocks(record) if isinstance(block, dict) and block.get("type") == "markdown" and len(str(block.get("text", "")).strip()) > 0)
    copy_button_html = _render_copy_button("user", "Copy request", copy_text) if role == "user" else ""
    queued_actions_html = _render_queued_request_actions() if role == "user" and badge_text == "Queued" else ""
    actions_html = f"<div class='wangp-assistant-chat__message-actions'>{copy_button_html}{queued_actions_html}</div>" if role == "user" else ""
    end_badge_html = "" if len(end_badge_text) == 0 else f"<div class='wangp-assistant-chat__message-end'><span class='wangp-assistant-chat__message-end-badge'>{html.escape(end_badge_text)}</span></div>"
    body_html = f"{blocks_html}{attachments_html}"
    card_html = (
        f"<article class='wangp-assistant-chat__message wangp-assistant-chat__message--{html.escape(role)}' data-message-id='{html.escape(str(record.get('id', '')))}'>"
        f"<div class='wangp-assistant-chat__avatar'>{html.escape('You' if role == 'user' else 'Deepy')}</div>"
        f"<div class='wangp-assistant-chat__message-card'>"
        f"<div class='wangp-assistant-chat__meta'>"
        f"<div class='wangp-assistant-chat__meta-left'>{badge_html}</div>"
        f"<div class='wangp-assistant-chat__meta-right'>{actions_html}<div class='wangp-assistant-chat__time'>{html.escape(str(record.get('created_at', '')))}</div></div>"
        f"</div>"
        f"<div class='wangp-assistant-chat__body'>{body_html}</div>"
        f"{end_badge_html}"
        f"</div>"
        f"</article>"
    )
    payload = {"id": record.get("id", ""), "role": role, "html": card_html}
    client_submission_id = str(record.get("client_submission_id", "") or "").strip()
    if client_submission_id:
        payload["client_submission_id"] = client_submission_id
    return payload


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
            content_source, attachments = _extract_attachments_from_markdown(block.get("text", ""))
            content_html = _plain_text_to_html(content_source) if str(record.get("role", "")).strip().lower() == "user" else _markdown_to_html(content_source)
            if len(content_html) > 0:
                rendered.append(content_html)
            attachment_html = _render_attachments(_dedupe_attachments(attachments, rendered_attachment_keys))
            if len(attachment_html) > 0:
                rendered.append(attachment_html)
            continue
        if block_type == "reasoning":
            reasoning_text = str(block.get("text", "")).strip()
            if len(reasoning_text) == 0:
                continue
            reasoning_no += 1
            rendered.append(_render_reasoning_block(block, reasoning_no, reasoning_total))
            continue
        if block_type == "context_summary":
            summary_text = str(block.get("text", "")).strip()
            if len(summary_text) > 0:
                rendered.append(_render_context_summary_block(block))
            continue
        if block_type == "tool":
            rendered.append(_render_tool_block(block))
            attachment_html = _render_attachments(_dedupe_attachments([block.get("attachment")] if isinstance(block.get("attachment"), dict) else [], rendered_attachment_keys))
            if len(attachment_html) > 0:
                rendered.append(attachment_html)
    return "".join(rendered), rendered_attachment_keys


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


def _render_reasoning_block(block: dict[str, Any], block_no: int, total_blocks: int) -> str:
    label = "Thought process"
    return (
        f"<details class='wangp-assistant-chat__disclosure wangp-assistant-chat__disclosure--reasoning' data-reasoning-id='{html.escape(str(block.get('id', '')))}'>"
        f"<summary><span class='wangp-assistant-chat__tool-title'><span class='wangp-assistant-chat__tool-chip'>Thought</span>{html.escape(label)}</span></summary>"
        f"<div class='wangp-assistant-chat__disclosure-body'><div class='wangp-assistant-chat__reasoning-block'>{_markdown_to_html(block.get('text', ''))}</div>{_render_collapse_button('thought')}</div>"
        "</details>"
    )


def _render_context_summary_block(block: dict[str, Any]) -> str:
    return (
        f"<details class='wangp-assistant-chat__disclosure wangp-assistant-chat__disclosure--context-summary' data-context-summary-id='{html.escape(str(block.get('id', '')))}'>"
        "<summary><span class='wangp-assistant-chat__tool-title'><span class='wangp-assistant-chat__tool-chip'>Context</span>Earlier history summarized</span></summary>"
        f"<div class='wangp-assistant-chat__disclosure-body'><div class='wangp-assistant-chat__context-summary'>{_markdown_to_html(block.get('text', ''))}</div></div>"
        "</details>"
    )


def _render_tool_block(tool_record: dict[str, Any]) -> str:
    name = str(tool_record.get("name", "tool")).strip() or "tool"
    label = str(tool_record.get("label", "")).strip() or _friendly_tool_label(name)
    status = str(tool_record.get("status", "running")).strip().lower()
    status_label = str(tool_record.get("status_text", "")).strip() or {"running": "Running", "done": "Done", "error": "Error"}.get(status, status.title() or "Running")
    status_class = {"running": "running", "done": "done", "error": "error"}.get(status, "running")
    arguments_text = html.escape(json.dumps(tool_record.get("arguments", {}), ensure_ascii=False, indent=2, sort_keys=True))
    result_payload = tool_record.get("result", {})
    result_text = html.escape(json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True)) if result_payload is not None else ""
    arguments_copy_button = _render_copy_button("json", f"Copy {label} arguments")
    result_copy_button = "" if result_payload is None else _render_copy_button("json", "Copy result")
    return (
        f"<details class='wangp-assistant-chat__disclosure wangp-assistant-chat__disclosure--tool' data-tool-id='{html.escape(str(tool_record.get('id', '')))}'>"
        f"<summary><span class='wangp-assistant-chat__tool-title'><span class='wangp-assistant-chat__tool-chip'>Tool</span>{html.escape(label)}</span><span class='wangp-assistant-chat__tool-status wangp-assistant-chat__tool-status--{status_class}'>{html.escape(status_label)}</span></summary>"
        "<div class='wangp-assistant-chat__disclosure-body'>"
        "<div class='wangp-assistant-chat__tool-grid'>"
        f"<div class='wangp-assistant-chat__tool-json'><div class='wangp-assistant-chat__tool-section-header'><div class='wangp-assistant-chat__tool-section-title'>{html.escape(label)} Arguments</div>{arguments_copy_button}</div><pre class='wangp-assistant-chat__pre'>{arguments_text}</pre></div>"
        f"<div class='wangp-assistant-chat__tool-json'><div class='wangp-assistant-chat__tool-section-header'><div class='wangp-assistant-chat__tool-section-title'>Result</div>{result_copy_button}</div><pre class='wangp-assistant-chat__pre'>{result_text or html.escape('Pending...')}</pre></div>"
        "</div>"
        f"{_render_collapse_button('tool')}"
        "</div>"
        "</details>"
    )


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
        subtitle_html = f"<span class='wangp-assistant-chat__attachment-subtitle'>{subtitle}</span>" if len(subtitle) > 0 else ""
        thumb_html = (
            f"<img class='wangp-assistant-chat__attachment-thumb' loading='lazy' src='{html.escape(thumb_url)}' alt='{label}'>"
            if len(thumb_url) > 0
            else "<div class='wangp-assistant-chat__attachment-thumb'></div>"
        )
        cards.append(
            f"<a class='wangp-assistant-chat__attachment' href='{html.escape(href)}' target='_blank' rel='noopener'>"
            f"{thumb_html}"
            "<span class='wangp-assistant-chat__attachment-meta'>"
            f"<span class='wangp-assistant-chat__attachment-title'>{label}</span>"
            f"{subtitle_html}"
            "</span>"
            "</a>"
        )
    if len(cards) == 0:
        return ""
    return f"<div class='wangp-assistant-chat__attachments'>{''.join(cards)}</div>"
