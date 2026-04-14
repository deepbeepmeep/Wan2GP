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
BUSY_QUEUE_BUTTON_ID = "assistant_chat_busy_queue_button"
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


def _shell_markup() -> str:
    return """
<section class="wangp-assistant-chat">
  <div class="wangp-assistant-chat__scroll">
    <div class="wangp-assistant-chat__empty">
      <div>
        <strong>Dialogue With Deepy</strong>
        Ask for an image or video idea, then inspect the assistant's reasoning and tool usage without losing the live transcript.
      </div>
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


def render_shell_html() -> str:
    return f"<div id='{CHAT_HOST_ID}' data-wangp-assistant-chat-mounted='true'>{_shell_markup()}</div>"


def render_stats_html() -> str:
    return f"<div id='{STATS_ID}' class='wangp-assistant-chat__stats' aria-hidden='true'></div>"


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
    width: 100% !important;
    min-height: 48px !important;
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
    min-height: 430px;
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
    height: 430px;
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
    align-items: center;
    justify-content: center;
    height: 100%;
    box-sizing: border-box;
    padding: 36px 34px 102px;
    border: 0;
    border-radius: 0;
    color: var(--muted-text);
    text-align: center;
    font-size: calc(0.98rem * var(--dock-font-scale));
    line-height: 1.6;
    background: transparent;
    backdrop-filter: none;
}

.wangp-assistant-chat__empty strong {
    display: block;
    margin-bottom: 6px;
    color: #194d70;
    font-size: calc(1rem * var(--dock-font-scale));
}

.wangp-assistant-chat__transcript {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 22px 18px calc(var(--chat-status-offset) + var(--chat-status-reserved-height) + var(--chat-status-gap));
}

.wangp-assistant-chat__stats {
    min-height: calc(0.78rem * var(--dock-font-scale));
    padding: 0 2px;
    font-size: calc(0.64rem * var(--dock-font-scale));
    line-height: 1.15;
    white-space: nowrap;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #8d9aa5;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.18s ease;
}

.wangp-assistant-chat__stats.is-visible {
    opacity: 0.96;
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

.wangp-assistant-chat__disclosure:not([open]) > .wangp-assistant-chat__disclosure-body {
    display: none;
}

.wangp-assistant-chat__disclosure[open] > .wangp-assistant-chat__disclosure-body {
    display: block;
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__disclosure-body {
    color: var(--assistant-text);
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

#assistant_chat_dock.is-dark .wangp-assistant-chat__empty strong,
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

#assistant_chat_dock.is-dark .wangp-assistant-chat__stats {
    color: #9eb0bd;
}

#assistant_chat_dock.is-dark .wangp-assistant-chat__message--user .wangp-assistant-chat__avatar {
    color: #eef6fb;
    background: linear-gradient(180deg, rgba(24, 31, 37, 0.99) 0%, rgba(10, 12, 14, 0.99) 100%);
    border-color: rgba(103, 132, 151, 0.2);
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
        height: 390px;
        border-radius: 20px;
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
        padding: 28px 20px 88px;
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

WAC.escapeHtml = function (value) {
  return String(value || '').replace(/[&<>\"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[char] || char));
};

WAC.timeLabel = function () {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
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

WAC.optimisticSubmits = Array.isArray(WAC.optimisticSubmits) ? WAC.optimisticSubmits : [];
WAC.serverInstanceId = WAC.serverInstanceId || '';

WAC.normalizeText = function (value) {
  return String(value || '').replace(/\r\n?/g, '\n').replace(/\u00a0/g, ' ').trim();
};

WAC.splitRequestBlocks = function (value) {
  const normalized = String(value || '').replace(/\r\n?/g, '\n').trim();
  if (!normalized) return [];
  const blocks = [];
  let current = [];
  for (const rawLine of normalized.split('\n')) {
    if (!String(rawLine).trim()) {
      if (current.length > 0) {
        const block = current.join('\n').trim();
        if (block) blocks.push(block);
        current = [];
      }
      continue;
    }
    current.push(String(rawLine).replace(/\s+$/, ''));
  }
  if (current.length > 0) {
    const block = current.join('\n').trim();
    if (block) blocks.push(block);
  }
  return blocks;
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

WAC.buildOptimisticUserMessage = function (optimisticId, content) {
  const contentHtml = WAC.escapeHtml(content).replace(/\n/g, '<br>');
  const html = [
    `<article class='wangp-assistant-chat__message wangp-assistant-chat__message--user' data-message-id='${optimisticId}'>`,
    "<div class='wangp-assistant-chat__avatar'>You</div>",
    "<div class='wangp-assistant-chat__message-card'>",
    "<div class='wangp-assistant-chat__meta'><div class='wangp-assistant-chat__meta-left'></div>",
    `<div class='wangp-assistant-chat__time'>${WAC.escapeHtml(WAC.timeLabel())}</div></div>`,
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

WAC.reconcileOptimisticSubmits = function () {
  const optimistic = Array.isArray(WAC.optimisticSubmits) ? WAC.optimisticSubmits.slice() : [];
  if (optimistic.length === 0) return;
  const serverUserTexts = [];
  for (const messageId of WAC.state.order) {
    const message = WAC.state.messages[messageId];
    if (!message || message.role !== 'user' || String(message.id || '').startsWith('optimistic_')) continue;
    const node = WAC.createMessageNode(message);
    serverUserTexts.push(WAC.messageBodyText(node));
  }
  let matchedPrefix = 0;
  const maxMatch = Math.min(serverUserTexts.length, optimistic.length);
  for (let count = maxMatch; count > 0; count -= 1) {
    const serverSuffix = serverUserTexts.slice(serverUserTexts.length - count);
    const optimisticPrefix = optimistic.slice(0, count).map((item) => WAC.normalizeText(item && item.text || ''));
    if (serverSuffix.length === optimisticPrefix.length && serverSuffix.every((text, index) => text === optimisticPrefix[index])) {
      matchedPrefix = count;
      break;
    }
    const flattenedPrefix = [];
    for (const item of optimistic.slice(0, count)) {
      const content = WAC.normalizeText(item && item.text || '');
      if (!content) continue;
      const blocks = WAC.splitRequestBlocks(content);
      flattenedPrefix.push(...(blocks.length > 1 ? blocks : [content]).map((block) => WAC.normalizeText(block)));
    }
    const splitServerSuffix = flattenedPrefix.length > count ? serverUserTexts.slice(serverUserTexts.length - flattenedPrefix.length) : [];
    if (splitServerSuffix.length === flattenedPrefix.length && splitServerSuffix.every((text, index) => text === flattenedPrefix[index])) {
      matchedPrefix = count;
      break;
    }
  }
  WAC.optimisticSubmits = optimistic.slice(matchedPrefix);
  for (const item of WAC.optimisticSubmits) {
    const optimisticId = String(item && item.id || '').trim();
    const content = WAC.normalizeText(item && item.text || '');
    if (!optimisticId || !content || WAC.state.messages[optimisticId]) continue;
    WAC.state.order.push(optimisticId);
    WAC.state.messages[optimisticId] = WAC.buildOptimisticUserMessage(optimisticId, content);
  }
};

WAC.pushOptimisticUserMessage = function (text) {
  const content = WAC.normalizeText(text);
  if (!content) return;
  const now = Date.now();
  const lastOptimistic = (WAC.optimisticSubmits || [])[WAC.optimisticSubmits.length - 1] || { text: '', ts: 0 };
  if (WAC.normalizeText(lastOptimistic.text || '') === content && (now - Number(lastOptimistic.ts || 0)) < 900) return;
  const optimisticId = `optimistic_${now}`;
  WAC.optimisticSubmits.push({ id: optimisticId, text: content, ts: now });
  WAC.upsertMessage(WAC.buildOptimisticUserMessage(optimisticId, content));
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

WAC.queueBusyRequest = function (text) {
  const input = document.querySelector('#assistant_chat_busy_queue_input textarea, #assistant_chat_busy_queue_input input');
  const button = document.querySelector('#assistant_chat_busy_queue_button button, #assistant_chat_busy_queue_button');
  if (!input || !button) return false;
  input.value = String(text || '');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
  if (typeof button.click === 'function') button.click();
  return true;
};

WAC.isAssistantBusy = function () {
  if (WAC.state && WAC.state.status && WAC.state.status.visible && WAC.state.status.text) return true;
  const stopButton = document.querySelector('#assistant_chat_html .wangp-assistant-chat__status-stop');
  return !!(stopButton && !stopButton.disabled);
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
  if (event.type === 'reset') {
    WAC.reset();
    return [];
  }
  if (event.type === 'upsert_message') {
    WAC.upsertMessage(event.message || {});
    return [];
  }
  if (event.type === 'remove_message') {
    WAC.removeMessage(event.message_id);
    return [];
  }
  if (event.type === 'status') {
    WAC.setStatus(event.status || null);
    if (Object.prototype.hasOwnProperty.call(event, 'stats')) WAC.setStats(event.stats || null);
    return [];
  }
  if (event.type === 'stats') {
    WAC.setStats(event.stats || null);
    return [];
  }
  if (event.type === 'sync') {
    WAC.sync(event.messages || [], event.status || null, Object.prototype.hasOwnProperty.call(event, 'stats') ? (event.stats || null) : WAC.state.stats);
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
    return true;
  }
  host.innerHTML = `
    <section class="wangp-assistant-chat">
      <div class="wangp-assistant-chat__scroll">
        <div class="wangp-assistant-chat__empty">
          <div>
            <strong>Dialogue With Deepy</strong>
            Ask for an image or video idea, then inspect the assistant's reasoning and tool usage without losing the live transcript.
          </div>
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
  WAC.syncAttributes(current, next);
  current.className = next.className;
  const currentSummary = current.querySelector(':scope > summary');
  const nextSummary = next.querySelector(':scope > summary');
  if (currentSummary && nextSummary && currentSummary.innerHTML !== nextSummary.innerHTML) currentSummary.innerHTML = nextSummary.innerHTML;
  const currentBody = current.querySelector(':scope > .wangp-assistant-chat__disclosure-body');
  const nextBody = next.querySelector(':scope > .wangp-assistant-chat__disclosure-body');
  if (currentBody && nextBody && currentBody.innerHTML !== nextBody.innerHTML) currentBody.innerHTML = nextBody.innerHTML;
};

WAC.reuseDisclosureNodes = function (currentBody, nextBody) {
  if (!currentBody || !nextBody) return;
  const existingByKey = new Map();
  currentBody.querySelectorAll(':scope > .wangp-assistant-chat__disclosure').forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (key) existingByKey.set(key, node);
  });
  nextBody.querySelectorAll(':scope > .wangp-assistant-chat__disclosure').forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (!key) return;
    const current = existingByKey.get(key);
    if (!current) return;
    WAC.patchDisclosureNode(current, node);
    node.replaceWith(current);
  });
};

WAC.patchMessageNode = function (current, next) {
  if (!current || !next) return;
  WAC.syncAttributes(current, next);
  current.className = next.className;
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
    WAC.reuseDisclosureNodes(currentBody, nextBody);
    currentBody.replaceChildren(...Array.from(nextBody.childNodes));
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
  if (!existing && message.role === 'user' && !incomingId.startsWith('optimistic_') && Array.isArray(WAC.optimisticSubmits) && WAC.optimisticSubmits.length > 0) {
    const incomingText = WAC.messageBodyText(node);
    const optimistic = WAC.optimisticSubmits.find((item) => WAC.normalizeText(item && item.text || '') === incomingText);
    const optimisticId = String(optimistic && optimistic.id || '');
    const optimisticNode = optimisticId ? transcript.querySelector(`[data-message-id="${CSS.escape(optimisticId)}"]`) : null;
    if (optimisticNode && optimisticId && incomingText) {
      optimisticNode.replaceWith(node);
      delete WAC.state.messages[optimisticId];
      WAC.state.order = WAC.state.order.map((id) => id === optimisticId ? incomingId : id);
      WAC.state.messages[incomingId] = message;
      WAC.dropOptimisticSubmit(optimisticId);
      WAC.hideEmpty();
      WAC.applyDisclosureState(transcript);
      WAC.applyAutoscrollState(scrollState);
      return;
    }
  }
  if (existing) {
    WAC.patchMessageNode(existing, node);
  } else {
    WAC.state.order.push(incomingId);
    transcript.appendChild(node);
  }
  WAC.state.messages[incomingId] = message;
  WAC.hideEmpty();
  WAC.applyDisclosureState(transcript);
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
    WAC.applyAutoscrollState(scrollState);
    return;
  }
  if (textNode) textNode.textContent = String(status.text);
  node.dataset.kind = String(status.kind || 'status');
  if (stopNode) stopNode.disabled = false;
  node.classList.add('is-visible');
  WAC.applyAutoscrollState(scrollState);
};

WAC.setStats = function (stats) {
  WAC.ensureShell();
  WAC.state.stats = stats || null;
  const node = WAC.statsNode();
  if (!node) return;
  if (!stats || stats.visible === false || !stats.text) {
    node.classList.remove('is-visible');
    node.textContent = '';
    return;
  }
  node.textContent = String(stats.text);
  node.classList.add('is-visible');
};

WAC.sync = function (messages, status, stats) {
  WAC.ensureShell();
  WAC.captureDisclosureState(WAC.transcript());
  const scrollState = WAC.captureAutoscrollState();
  WAC.replaceState(messages, status, stats);
  WAC.reconcileOptimisticSubmits();
  WAC.hydrate(scrollState);
};

WAC.reset = function () {
  WAC.state = { order: [], messages: {}, status: null, stats: null };
  WAC.optimisticSubmits = [];
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
  const fragment = document.createDocumentFragment();
  for (const messageId of WAC.state.order) {
    const message = WAC.state.messages[messageId];
    if (!message) continue;
    const node = WAC.createMessageNode(message);
    if (!node) continue;
    const existing = existingById.get(String(messageId));
    if (existing) {
      WAC.patchMessageNode(existing, node);
      fragment.appendChild(existing);
      existingById.delete(String(messageId));
      continue;
    }
    fragment.appendChild(node);
  }
  transcript.replaceChildren(fragment);
  WAC.applyDisclosureState(transcript);
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
  window.addEventListener('resize', () => { WAC.syncDockLayout(); }, { passive: true });
};

WAC.installDockBridge = function () {
  if (WAC.dockBridgeInstalled) return;
  WAC.dockBridgeInstalled = true;
  WAC.dockOpen = false;
  try { window.localStorage.removeItem('wangp-assistant-chat-open'); } catch (_error) {}
  document.addEventListener('pointerdown', (event) => {
    if (WAC.handleDisclosurePointerDown(event)) return;
    if (WAC.handleAttachmentPointerDown(event)) return;
  }, true);
  document.addEventListener('click', (event) => {
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
      for (const target of WAC.stopBridgeTargets()) {
        target.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
        if (typeof target.click === 'function') target.click();
      }
      return;
    }
    const jumpBottomButton = event.target && event.target.closest ? event.target.closest('.wangp-assistant-chat__jump-bottom') : null;
    if (jumpBottomButton) {
      event.preventDefault();
      WAC.scrollToBottom();
      return;
    }
    const askButton = event.target && event.target.closest ? event.target.closest('#assistant_chat_ask_button') : null;
    if (!askButton) return;
    const input = WAC.requestInput();
    const text = input ? String(input.value || '').trim() : '';
    if (!text) return;
    WAC.setDockOpen(true);
    WAC.pushOptimisticUserMessage(text);
    window.setTimeout(() => { WAC.clearRequestInput(text); }, 0);
    if (WAC.isAssistantBusy()) {
      event.preventDefault();
      event.stopPropagation();
      WAC.queueBusyRequest(text);
      return;
    }
  }, true);
  document.addEventListener('keydown', (event) => {
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
    if (!input || event.target !== input || event.key !== 'Enter' || event.shiftKey || event.ctrlKey || event.altKey || event.metaKey) return;
    const text = String(input.value || '').trim();
    if (!text) return;
    event.preventDefault();
    event.stopPropagation();
    WAC.setDockOpen(true);
    WAC.pushOptimisticUserMessage(text);
    window.setTimeout(() => { WAC.clearRequestInput(text); }, 0);
    if (WAC.isAssistantBusy()) {
      WAC.queueBusyRequest(text);
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

setTimeout(() => { WAC.ensureShell(); WAC.handleEventNodeMutation(); WAC.readEventSource(); WAC.syncDockState(); WAC.syncDockLayout(); }, 50);
if (window.__wangpAssistantChatPending.length > 0) {
  const pending = window.__wangpAssistantChatPending.slice();
  window.__wangpAssistantChatPending.length = 0;
  for (const payload of pending) WAC.consumePayload(payload);
}
window.applyAssistantChatEvent = function (payload) {
  return WAC.consumePayload(payload);
};
"""


def reset_session_chat(session) -> None:
    session.chat_transcript.clear()
    session.chat_transcript_counter = 0


def build_reset_event() -> str:
    return _event_payload({"type": "reset"})


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


def build_sync_event(session, status: dict[str, Any] | None = None, stats: dict[str, Any] | None = None) -> str:
    messages = [_render_message_payload(record) for record in session.chat_transcript]
    event = {"type": "sync", "messages": messages, "status": status}
    if stats is not None:
        event["stats"] = stats
    return _event_payload(event)


def _queued_tail_insert_index(session) -> int:
    records = list(session.chat_transcript or [])
    insert_index = len(records)
    while insert_index > 0:
        record = records[insert_index - 1]
        if not isinstance(record, dict):
            break
        if str(record.get("role", "")).strip() != "user":
            break
        if str(record.get("badge", "")).strip() != "Queued":
            break
        insert_index -= 1
    return insert_index


def add_user_message(session, text: str, queued: bool = False) -> tuple[str, str]:
    record = {
        "id": _next_message_id(session, "user"),
        "role": "user",
        "author": "You",
        "created_at": _time_label(),
        "blocks": [],
        "attachments": [],
        "badge": "Queued" if queued else "",
    }
    content = str(text or "").strip()
    if len(content) > 0:
        record["blocks"].append({"id": _next_block_id("content"), "type": "markdown", "text": content})
    session.chat_transcript.append(record)
    return record["id"], _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


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
    return record["id"], _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


def get_message_content(session, message_id: str) -> str:
    record = _find_message(session, message_id)
    if record is None:
        return ""
    parts = [str(block.get("text", "")).strip() for block in _ensure_message_blocks(record) if isinstance(block, dict) and block.get("type") == "markdown" and len(str(block.get("text", "")).strip()) > 0]
    return "\n\n".join(parts)


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
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


def clear_message_blocks(session, message_id: str) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    record["blocks"] = []
    record["attachments"] = []
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


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
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


def remove_message(session, message_id: str) -> str | None:
    target_id = str(message_id or "").strip()
    if len(target_id) == 0:
        return None
    original_len = len(session.chat_transcript)
    session.chat_transcript[:] = [record for record in session.chat_transcript if str(record.get("id", "")) != target_id]
    if len(session.chat_transcript) == original_len:
        return None
    return _event_payload({"type": "remove_message", "message_id": target_id})


def append_reasoning(session, message_id: str, text: str) -> str | None:
    _reasoning_id, payload = upsert_reasoning_block(session, message_id, None, text)
    return payload


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
        return target_id, _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})
    target_id = target_id or f"reasoning_{uuid.uuid4().hex[:10]}"
    blocks.append({"id": target_id, "type": "reasoning", "text": reasoning_text})
    return target_id, _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


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
    return tool_record["id"], _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


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
        return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})
    return None


def complete_tool_call(session, message_id: str, tool_id: str, result: dict[str, Any]) -> str | None:
    status = str((result or {}).get("status", "")).strip().lower()
    failed = status in {"error", "failed", "interrupted"}
    return update_tool_call(session, message_id, tool_id, status="error" if failed else "done", result=result, status_text="Interrupted" if status == "interrupted" else ("Error" if failed else "Done"))


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
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


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
    return name.replace("_", " ").replace("-", " ").strip().title()


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


def _event_payload(event: dict[str, Any]) -> str:
    return json.dumps({"event_id": uuid.uuid4().hex, "instance_id": SERVER_INSTANCE_ID, "event": event}, ensure_ascii=False)


def _markdown_to_html(text: str) -> str:
    text = str(text or "").strip()
    if len(text) == 0:
        return ""
    text = html.escape(text, quote=False)
    return markdown.markdown(text, extensions=_MARKDOWN_EXTENSIONS, output_format="html5")


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


def _render_message_payload(record: dict[str, Any]) -> dict[str, Any]:
    role = str(record.get("role", "assistant"))
    badge_text = str(record.get("badge", "")).strip()
    blocks_html, rendered_attachment_keys = _render_message_blocks(record)
    attachments_html = _render_attachments(
        [
            attachment
            for attachment in list(record.get("attachments", []))
            if isinstance(attachment, dict) and (attachment.get("path_key", "") or attachment.get("href", "")) not in rendered_attachment_keys
        ]
    )
    badge_html = "" if len(badge_text) == 0 else f"<span class='wangp-assistant-chat__badge'>{html.escape(badge_text)}</span>"
    body_html = f"{blocks_html}{attachments_html}"
    if len(body_html) == 0 and role == "assistant":
        body_html = "<p>Working through the request.</p>"
    card_html = (
        f"<article class='wangp-assistant-chat__message wangp-assistant-chat__message--{html.escape(role)}' data-message-id='{html.escape(str(record.get('id', '')))}'>"
        f"<div class='wangp-assistant-chat__avatar'>{html.escape('You' if role == 'user' else 'Deepy')}</div>"
        f"<div class='wangp-assistant-chat__message-card'>"
        f"<div class='wangp-assistant-chat__meta'>"
        f"<div class='wangp-assistant-chat__meta-left'>{badge_html}</div>"
        f"<div class='wangp-assistant-chat__time'>{html.escape(str(record.get('created_at', '')))}</div>"
        f"</div>"
        f"<div class='wangp-assistant-chat__body'>{body_html}</div>"
        f"</div>"
        f"</article>"
    )
    return {"id": record.get("id", ""), "role": role, "html": card_html}


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
            content_html = _markdown_to_html(content_source)
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
        f"<div class='wangp-assistant-chat__disclosure-body'><div class='wangp-assistant-chat__reasoning-block'>{_markdown_to_html(block.get('text', ''))}</div></div>"
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
    return (
        f"<details class='wangp-assistant-chat__disclosure wangp-assistant-chat__disclosure--tool' data-tool-id='{html.escape(str(tool_record.get('id', '')))}'>"
        f"<summary><span class='wangp-assistant-chat__tool-title'><span class='wangp-assistant-chat__tool-chip'>Tool</span>{html.escape(label)}</span><span class='wangp-assistant-chat__tool-status wangp-assistant-chat__tool-status--{status_class}'>{html.escape(status_label)}</span></summary>"
        "<div class='wangp-assistant-chat__disclosure-body'>"
        "<div class='wangp-assistant-chat__tool-grid'>"
        f"<div><div class='wangp-assistant-chat__tool-section-title'>{html.escape(label)} Arguments</div><pre class='wangp-assistant-chat__pre'>{arguments_text}</pre></div>"
        f"<div><div class='wangp-assistant-chat__tool-section-title'>Result</div><pre class='wangp-assistant-chat__pre'>{result_text or html.escape('Pending...')}</pre></div>"
        "</div>"
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
