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


CHAT_HOST_ID = "assistant_chat_html"
CHAT_EVENT_ID = "assistant_chat_event"
DOCK_ID = "assistant_chat_dock"
LAUNCHER_HOST_ID = "assistant_chat_launcher_host"
LAUNCHER_BUTTON_ID = "assistant_chat_toggle"
PANEL_ID = "assistant_chat_panel"
CHAT_BLOCK_ID = "assistant_chat_shell_block"
CONTROLS_ID = "assistant_chat_controls"
REQUEST_ID = "assistant_chat_request"
ASK_BUTTON_ID = "assistant_chat_ask_button"
RESET_BUTTON_ID = "assistant_chat_reset_button"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jfif", ".pjpeg"}
_MARKDOWN_EXTENSIONS = ["extra", "nl2br", "sane_lists", "fenced_code", "tables"]
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


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
    <div class="wangp-assistant-chat__status-kind"></div>
  </div>
</section>
""".strip()


def render_shell_html() -> str:
    return f"<div id='{CHAT_HOST_ID}' data-wangp-assistant-chat-mounted='true'>{_shell_markup()}</div>"


def render_launcher_html() -> str:
    return (
        f"<button id='{LAUNCHER_BUTTON_ID}' class='wangp-assistant-chat__toggle' type='button' "
        "aria-label='Toggle Deepy assistant' aria-expanded='false'>"
        "<span class='wangp-assistant-chat__toggle-text'>Ask Deepy</span>"
        "</button>"
    )


def get_css() -> str:
    return r"""
#assistant_chat_dock {
    --dock-gap: 14px;
    --dock-launcher-width: 41px;
    --dock-panel-width: 588px;
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
#assistant_chat_panel {
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
#assistant_chat_shell_block .html-container {
    padding: 0 !important;
}

#assistant_chat_launcher_host .prose,
#assistant_chat_shell_block .prose {
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
    border: 1px solid rgba(13, 74, 105, 0.22);
    border-left: 0;
    border-radius: 0 22px 22px 0;
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(230, 245, 251, 0.98) 100%);
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
    color: #0f5375;
    font-size: 0.76rem;
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
    padding: 14px;
    border: 1px solid rgba(16, 78, 109, 0.16);
    border-radius: 28px;
    background: #ffffff;
    box-shadow: 0 30px 60px rgba(8, 34, 50, 0.2);
    opacity: 0;
    visibility: hidden;
    transform: translateY(-50%) translateX(-30px) scale(0.98);
    transform-origin: left center !important;
    transition: opacity 0.22s ease, transform 0.22s ease, visibility 0.22s step-end;
    pointer-events: none;
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

#assistant_chat_shell_block,
#assistant_chat_controls {
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#assistant_chat_shell_block {
    margin-bottom: 12px !important;
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
    align-items: stretch;
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
    min-height: 52px !important;
    border: 1px solid rgba(23, 90, 125, 0.18) !important;
    border-radius: 16px !important;
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
    min-width: 0 !important;
    min-height: 52px;
    padding: 0 16px;
    border-radius: 16px;
    font-weight: 700;
    box-shadow: 0 12px 22px rgba(11, 43, 63, 0.12);
    border: 0;
}

#assistant_chat_ask_button {
    width: 112px;
    background: linear-gradient(180deg, #0e5b81 0%, #0a415e 100%);
    color: #f3fbff;
}

#assistant_chat_reset_button {
    width: 88px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(239, 246, 250, 0.98) 100%);
    color: #164f70;
    border: 1px solid rgba(20, 82, 113, 0.14);
}

#assistant_chat_html {
    min-height: 430px;
}

.wangp-assistant-chat {
    --chat-border: transparent;
    --chat-shadow: none;
    --chat-surface: #ffffff;
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
    font-size: 0.98rem;
    line-height: 1.6;
    background: transparent;
    backdrop-filter: none;
}

.wangp-assistant-chat__empty strong {
    display: block;
    margin-bottom: 6px;
    color: #194d70;
    font-size: 1rem;
}

.wangp-assistant-chat__transcript {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 22px 18px 94px;
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
    font-size: 0.8rem;
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
    width: min(84%, 920px);
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
    font-size: 0.82rem;
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
    font-size: 0.72rem;
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
    font-size: 0.97rem;
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
    font-size: 0.8rem;
    line-height: 1.3;
}

.wangp-assistant-chat__disclosure summary::-webkit-details-marker {
    display: none;
}

.wangp-assistant-chat__disclosure summary::after {
    content: "▾";
    font-size: 0.78rem;
    transition: transform 0.18s ease;
    color: #2f769f;
}

.wangp-assistant-chat__disclosure[open] summary::after {
    transform: rotate(180deg);
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__disclosure summary::after {
    color: rgba(245, 251, 255, 0.86);
}

.wangp-assistant-chat__disclosure-body {
    padding: 0 14px 14px;
    color: #385363;
}

.wangp-assistant-chat__message--assistant .wangp-assistant-chat__disclosure-body {
    color: var(--assistant-text);
}

.wangp-assistant-chat__tool-title {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
}

.wangp-assistant-chat__tool-chip {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.64rem;
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
    font-size: 0.64rem;
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
    font-size: 0.84rem;
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
    font-size: 0.76rem;
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
    font-size: 0.84rem;
    line-height: 1.45;
    word-break: break-word;
}

.wangp-assistant-chat__status {
    position: absolute;
    left: 18px;
    right: 18px;
    bottom: 18px;
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
.wangp-assistant-chat__status-kind {
    color: var(--status-text);
}

.wangp-assistant-chat__status.is-visible {
    opacity: 1;
    transform: translateY(0);
}

.wangp-assistant-chat__status-text {
    flex: 1;
    min-width: 0;
    font-size: 0.92rem;
    line-height: 1.45;
    font-weight: 600;
}

.wangp-assistant-chat__status-dots {
    display: inline-flex;
    align-items: center;
    gap: 5px;
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

.wangp-assistant-chat__status-kind {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.18);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
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
        padding: 16px 12px 88px;
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
}
"""


def get_javascript() -> str:
    return r"""
window.__wangpAssistantChatNS = window.__wangpAssistantChatNS || {};
window.__wangpAssistantChatPending = window.__wangpAssistantChatPending || [];
const WAC = window.__wangpAssistantChatNS;

WAC.state = WAC.state || { order: [], messages: {}, status: null };
WAC.init = WAC.init || false;
WAC.observer = WAC.observer || null;
WAC.eventNode = WAC.eventNode || null;
WAC.pollTimer = WAC.pollTimer || null;
WAC.lastPayloadId = WAC.lastPayloadId || '';
WAC.lastPayloadText = WAC.lastPayloadText || '';
WAC.dockBridgeInstalled = WAC.dockBridgeInstalled || false;
WAC.dockOpen = typeof WAC.dockOpen === 'boolean' ? WAC.dockOpen : false;
WAC.scrollFrame = WAC.scrollFrame || 0;

WAC.dock = function () {
  return document.querySelector('#assistant_chat_dock');
};

WAC.panel = function () {
  return document.querySelector('#assistant_chat_panel');
};

WAC.launcher = function () {
  return document.querySelector('#assistant_chat_toggle');
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

WAC.captureScrollAnchor = function () {
  const scroll = WAC.scroll();
  if (!scroll) return { stick: true, top: 0 };
  return {
    stick: WAC.isNearBottom(),
    top: Math.max(0, scroll.scrollTop),
  };
};

WAC.restoreScrollAnchor = function (anchor, forceBottom) {
  const scroll = WAC.scroll();
  if (!scroll) return;
  if (WAC.scrollFrame) cancelAnimationFrame(WAC.scrollFrame);
  WAC.scrollFrame = requestAnimationFrame(() => {
    WAC.scrollFrame = 0;
    if (forceBottom || !anchor || anchor.stick) {
      scroll.scrollTop = scroll.scrollHeight;
      return;
    }
    const maxTop = Math.max(0, scroll.scrollHeight - scroll.clientHeight);
    const targetTop = Math.max(0, Math.min(maxTop, Number(anchor.top || 0)));
    scroll.scrollTop = targetTop;
  });
};

WAC.lastOptimisticSubmit = WAC.lastOptimisticSubmit || { id: '', text: '', ts: 0 };

WAC.pushOptimisticUserMessage = function (text) {
  const content = String(text || '').trim();
  if (!content) return;
  const now = Date.now();
  if (WAC.lastOptimisticSubmit.text === content && (now - WAC.lastOptimisticSubmit.ts) < 900) return;
  const optimisticId = `optimistic_${now}`;
  WAC.lastOptimisticSubmit = { id: optimisticId, text: content, ts: now };
  const html = [
    `<article class='wangp-assistant-chat__message wangp-assistant-chat__message--user' data-message-id='${optimisticId}'>`,
    "<div class='wangp-assistant-chat__avatar'>You</div>",
    "<div class='wangp-assistant-chat__message-card'>",
    "<div class='wangp-assistant-chat__meta'><div class='wangp-assistant-chat__meta-left'></div>",
    `<div class='wangp-assistant-chat__time'>${WAC.escapeHtml(WAC.timeLabel())}</div></div>`,
    `<div class='wangp-assistant-chat__body'><p>${WAC.escapeHtml(content)}</p></div>`,
    "</div></article>",
  ].join('');
  WAC.upsertMessage({ id: optimisticId, role: 'user', html });
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
    return [];
  }
  if (event.type === 'sync') {
    WAC.sync(event.messages || [], event.status || null);
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

WAC.replaceState = function (messages, status) {
  const nextState = { order: [], messages: {}, status: status || null };
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

WAC.syncDockState = function () {
  WAC.syncDockVisibility();
  const dock = WAC.dock();
  const launcher = WAC.launcher();
  if (dock) dock.classList.toggle('is-open', !!WAC.dockOpen);
  if (launcher) launcher.setAttribute('aria-expanded', WAC.dockOpen ? 'true' : 'false');
};

WAC.syncDockLayout = function () {
  const dock = WAC.dock();
  if (!dock) return;
  if (window.innerWidth <= 900) {
    dock.style.removeProperty('--dock-panel-width');
    return;
  }
  const flowColumn = dock.parentElement;
  const measuredWidth = flowColumn ? Math.round(flowColumn.getBoundingClientRect().width) : 0;
  const maxWidth = Math.max(420, window.innerWidth - 92);
  const panelWidth = Math.max(420, Math.min(measuredWidth || 588, maxWidth));
  dock.style.setProperty('--dock-panel-width', `${panelWidth}px`);
};

WAC.setDockOpen = function (open, persist) {
  WAC.dockOpen = !!open;
  WAC.syncDockState();
  WAC.syncDockLayout();
  if (persist !== false) {
    try { window.localStorage.setItem('wangp-assistant-chat-open', WAC.dockOpen ? '1' : '0'); } catch (_error) {}
  }
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
        <div class="wangp-assistant-chat__status-kind"></div>
      </div>
    </section>
  `;
  host.dataset.wangpAssistantChatMounted = 'true';
  WAC.hydrate();
  WAC.syncDockVisibility();
  WAC.syncDockState();
  WAC.syncDockLayout();
  return true;
};

WAC.isNearBottom = function () {
  const scroll = WAC.scroll();
  if (!scroll) return true;
  return (scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight) < 120;
};

WAC.scrollToBottom = function (force) {
  const scroll = WAC.scroll();
  if (!scroll) return;
  if (!force && !WAC.isNearBottom()) return;
  requestAnimationFrame(() => { scroll.scrollTop = scroll.scrollHeight; });
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
};

WAC.createMessageNode = function (message) {
  const tpl = document.createElement('template');
  tpl.innerHTML = (message && message.html) ? String(message.html).trim() : '';
  return tpl.content.firstElementChild;
};

WAC.messageBodyText = function (node) {
  const body = node && node.querySelector ? node.querySelector('.wangp-assistant-chat__body') : null;
  return body ? String(body.innerText || body.textContent || '').trim() : '';
};

WAC.upsertMessage = function (message) {
  if (!message || !message.id) return;
  WAC.ensureShell();
  const transcript = WAC.transcript();
  if (!transcript) return;
  const anchor = WAC.captureScrollAnchor();
  const node = WAC.createMessageNode(message);
  if (!node) return;
  const existing = transcript.querySelector(`[data-message-id="${CSS.escape(String(message.id))}"]`);
  const optimistic = WAC.lastOptimisticSubmit || { id: '', text: '', ts: 0 };
  const incomingId = String(message.id);
  if (!existing && message.role === 'user' && !incomingId.startsWith('optimistic_') && optimistic.id) {
    const optimisticNode = transcript.querySelector(`[data-message-id="${CSS.escape(String(optimistic.id))}"]`);
    const incomingText = WAC.messageBodyText(node);
    if (optimisticNode && incomingText && incomingText === optimistic.text) {
      optimisticNode.replaceWith(node);
      delete WAC.state.messages[String(optimistic.id)];
      WAC.state.order = WAC.state.order.map((id) => id === String(optimistic.id) ? incomingId : id);
      WAC.state.messages[incomingId] = message;
      WAC.lastOptimisticSubmit = { id: '', text: '', ts: 0 };
      WAC.hideEmpty();
      WAC.restoreScrollAnchor(anchor, true);
      return;
    }
  }
  if (existing) {
    existing.replaceWith(node);
  } else {
    WAC.state.order.push(incomingId);
    transcript.appendChild(node);
  }
  WAC.state.messages[incomingId] = message;
  WAC.hideEmpty();
  WAC.restoreScrollAnchor(anchor, message.role === 'user');
};

WAC.removeMessage = function (messageId) {
  const transcript = WAC.transcript();
  if (!transcript) return;
  const existing = transcript.querySelector(`[data-message-id="${CSS.escape(String(messageId))}"]`);
  if (existing) existing.remove();
  delete WAC.state.messages[String(messageId)];
  WAC.state.order = WAC.state.order.filter(id => id !== String(messageId));
  WAC.showEmptyIfNeeded();
};

WAC.setStatus = function (status, restoreAnchor) {
  WAC.ensureShell();
  WAC.state.status = status || null;
  const node = WAC.statusNode();
  if (!node) return;
  const anchor = restoreAnchor === undefined ? WAC.captureScrollAnchor() : restoreAnchor;
  const textNode = node.querySelector('.wangp-assistant-chat__status-text');
  const kindNode = node.querySelector('.wangp-assistant-chat__status-kind');
  if (!status || !status.visible || !status.text) {
    node.classList.remove('is-visible');
    if (textNode) textNode.textContent = '';
    if (kindNode) kindNode.textContent = '';
    if (anchor !== null) WAC.restoreScrollAnchor(anchor, false);
    return;
  }
  if (textNode) textNode.textContent = String(status.text);
  if (kindNode) kindNode.textContent = String(status.kind || 'status');
  node.classList.add('is-visible');
  if (anchor !== null) WAC.restoreScrollAnchor(anchor, false);
};

WAC.sync = function (messages, status) {
  WAC.ensureShell();
  const anchor = WAC.captureScrollAnchor();
  WAC.replaceState(messages, status);
  WAC.hydrate(anchor);
};

WAC.reset = function () {
  WAC.state = { order: [], messages: {}, status: null };
  WAC.ensureShell();
  const transcript = WAC.transcript();
  if (transcript) transcript.innerHTML = '';
  WAC.showEmptyIfNeeded();
  WAC.setStatus(null);
};

WAC.hydrate = function (anchor) {
  const transcript = WAC.transcript();
  if (!transcript) return;
  transcript.innerHTML = '';
  for (const messageId of WAC.state.order) {
    const message = WAC.state.messages[messageId];
    if (!message) continue;
    const node = WAC.createMessageNode(message);
    if (node) transcript.appendChild(node);
  }
  WAC.showEmptyIfNeeded();
  WAC.setStatus(WAC.state.status, null);
  WAC.restoreScrollAnchor(anchor, false);
};

WAC.applyEvent = function (payload) {
  return WAC.consumePayload(payload);
};

WAC.installObserver = function () {
  if (WAC.observer) return;
  const target = document.querySelector('gradio-app') || document.body;
  if (!target) return;
  WAC.observer = new MutationObserver(() => {
    if (WAC.host()) WAC.ensureShell();
    WAC.syncDockLayout();
    WAC.handleEventNodeMutation();
    WAC.readEventSource();
  });
  WAC.observer.observe(target, { childList: true, subtree: true });
};

WAC.installEventBridge = function () {
  WAC.handleEventNodeMutation();
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
  try {
    WAC.dockOpen = window.localStorage.getItem('wangp-assistant-chat-open') === '1';
  } catch (_error) {
    WAC.dockOpen = false;
  }
  document.addEventListener('click', (event) => {
    const toggle = event.target && event.target.closest ? event.target.closest('#assistant_chat_toggle') : null;
    if (toggle) {
      event.preventDefault();
      WAC.toggleDock();
      return;
    }
    const askButton = event.target && event.target.closest ? event.target.closest('#assistant_chat_ask_button') : null;
    if (!askButton) return;
    const input = WAC.requestInput();
    const text = input ? String(input.value || '').trim() : '';
    if (!text) return;
    WAC.setDockOpen(true);
    WAC.pushOptimisticUserMessage(text);
  }, true);
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !WAC.dockOpen) return;
    WAC.setDockOpen(false);
  }, true);
  document.addEventListener('keydown', (event) => {
    const input = WAC.requestInput();
    if (!input || event.target !== input || event.key !== 'Enter' || event.shiftKey || event.ctrlKey || event.altKey || event.metaKey) return;
    const text = String(input.value || '').trim();
    if (!text) return;
    WAC.setDockOpen(true);
    WAC.pushOptimisticUserMessage(text);
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


def build_status_event(text: str | None, kind: str = "status", visible: bool = True) -> str:
    status = None if not visible or not text else {"visible": True, "kind": str(kind or "status"), "text": str(text or "").strip()}
    return _event_payload({"type": "status", "status": status})


def build_sync_event(session, status: dict[str, Any] | None = None) -> str:
    messages = [_render_message_payload(record) for record in session.chat_transcript]
    return _event_payload({"type": "sync", "messages": messages, "status": status})


def add_user_message(session, text: str, queued: bool = False) -> tuple[str, str]:
    record = {
        "id": _next_message_id(session, "user"),
        "role": "user",
        "author": "You",
        "created_at": _time_label(),
        "content": str(text or "").strip(),
        "reasoning": [],
        "tools": [],
        "attachments": [],
        "badge": "Queued" if queued else "",
    }
    session.chat_transcript.append(record)
    return record["id"], _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


def create_assistant_turn(session) -> str:
    record = {
        "id": _next_message_id(session, "assistant"),
        "role": "assistant",
        "author": "Deepy",
        "created_at": _time_label(),
        "content": "",
        "reasoning": [],
        "tools": [],
        "attachments": [],
        "badge": "",
    }
    session.chat_transcript.append(record)
    return record["id"]


def set_message_badge(session, message_id: str, badge: str | None) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    record["badge"] = str(badge or "").strip()
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


def append_reasoning(session, message_id: str, text: str) -> str | None:
    reasoning_text = str(text or "").strip()
    if len(reasoning_text) == 0:
        return None
    record = _find_message(session, message_id)
    if record is None:
        return None
    reasoning_blocks = record.setdefault("reasoning", [])
    if len(reasoning_blocks) > 0 and reasoning_blocks[-1] == reasoning_text:
        return None
    reasoning_blocks.append(reasoning_text)
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


def add_tool_call(session, message_id: str, tool_name: str, arguments: dict[str, Any], tool_label: str | None = None) -> tuple[str, str | None]:
    record = _find_message(session, message_id)
    if record is None:
        return "", None
    tool_record = {
        "id": _next_tool_id(),
        "name": str(tool_name or "").strip(),
        "label": str(tool_label or "").strip() or _friendly_tool_label(tool_name),
        "arguments": dict(arguments or {}),
        "result": None,
        "status": "running",
        "attachment": None,
    }
    record.setdefault("tools", []).append(tool_record)
    return tool_record["id"], _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


def complete_tool_call(session, message_id: str, tool_id: str, result: dict[str, Any]) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    for tool_record in record.setdefault("tools", []):
        if tool_record.get("id") != tool_id:
            continue
        tool_record["result"] = dict(result or {})
        status = str(result.get("status", "")).strip().lower()
        tool_record["status"] = "error" if status in {"error", "failed"} else "done"
        attachment = _attachment_from_tool_result(result)
        if attachment is not None:
            tool_record["attachment"] = attachment
            _merge_attachments(record, [attachment])
        return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})
    return None


def set_assistant_content(session, message_id: str, text: str) -> str | None:
    record = _find_message(session, message_id)
    if record is None:
        return None
    record["content"] = str(text or "").strip()
    _merge_attachments(record, _extract_attachments_from_markdown(record["content"])[1])
    return _event_payload({"type": "upsert_message", "message": _render_message_payload(record)})


def _next_message_id(session, prefix: str) -> str:
    session.chat_transcript_counter += 1
    return f"{prefix}_{session.chat_transcript_counter}"


def _next_tool_id() -> str:
    return f"tool_{uuid.uuid4().hex[:10]}"


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


def _time_label() -> str:
    return time.strftime("%H:%M")


def _event_payload(event: dict[str, Any]) -> str:
    return json.dumps({"event_id": uuid.uuid4().hex, "event": event}, ensure_ascii=False)


def _markdown_to_html(text: str) -> str:
    text = str(text or "").strip()
    if len(text) == 0:
        return ""
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
    label = "Generated image" if ext in _IMAGE_EXTENSIONS else "Generated file"
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
    return {
        "path_key": path_key,
        "href": href,
        "label": resolved_label,
        "subtitle": subtitle,
        "kind": "image" if ext in _IMAGE_EXTENSIONS else "file",
        "thumb_url": href if ext in _IMAGE_EXTENSIONS else "",
    }


def _merge_attachments(record: dict[str, Any], new_attachments: list[dict[str, Any]]) -> None:
    if len(new_attachments) == 0:
        return
    existing = {attachment.get("path_key", "") or attachment.get("href", "") for attachment in record.setdefault("attachments", [])}
    for attachment in new_attachments:
        dedupe_key = attachment.get("path_key", "") or attachment.get("href", "")
        if len(dedupe_key) == 0 or dedupe_key in existing:
            continue
        record["attachments"].append(attachment)
        existing.add(dedupe_key)


def _render_message_payload(record: dict[str, Any]) -> dict[str, Any]:
    role = str(record.get("role", "assistant"))
    content_source, markdown_attachments = _extract_attachments_from_markdown(record.get("content", ""))
    _merge_attachments(record, markdown_attachments)
    badge_text = str(record.get("badge", "")).strip()
    content_html = _markdown_to_html(content_source)
    reasoning_html = _render_reasoning_blocks(record.get("reasoning", []))
    tools_html = _render_tool_blocks(record.get("tools", []))
    attachments_html = _render_attachments(list(record.get("attachments", [])))
    badge_html = "" if len(badge_text) == 0 else f"<span class='wangp-assistant-chat__badge'>{html.escape(badge_text)}</span>"
    body_html = content_html
    if len(body_html) == 0 and role == "assistant" and (len(reasoning_html) > 0 or len(tools_html) > 0):
        body_html = "<p>Working through the request.</p>"
    card_html = (
        f"<article class='wangp-assistant-chat__message wangp-assistant-chat__message--{html.escape(role)}' data-message-id='{html.escape(str(record.get('id', '')))}'>"
        f"<div class='wangp-assistant-chat__avatar'>{html.escape('You' if role == 'user' else 'Deepy')}</div>"
        f"<div class='wangp-assistant-chat__message-card'>"
        f"<div class='wangp-assistant-chat__meta'>"
        f"<div class='wangp-assistant-chat__meta-left'>{badge_html}</div>"
        f"<div class='wangp-assistant-chat__time'>{html.escape(str(record.get('created_at', '')))}</div>"
        f"</div>"
        f"<div class='wangp-assistant-chat__body'>{body_html}{reasoning_html}{tools_html}{attachments_html}</div>"
        f"</div>"
        f"</article>"
    )
    return {"id": record.get("id", ""), "role": role, "html": card_html}


def _render_reasoning_blocks(reasoning_blocks: list[str]) -> str:
    blocks = [str(block or "").strip() for block in reasoning_blocks if len(str(block or "").strip()) > 0]
    if len(blocks) == 0:
        return ""
    body_html = "".join(f"<div class='wangp-assistant-chat__reasoning-block'>{_markdown_to_html(block)}</div>" for block in blocks)
    label = "Thought process" if len(blocks) == 1 else f"Thought process ({len(blocks)} thoughts)"
    return (
        "<details class='wangp-assistant-chat__disclosure wangp-assistant-chat__disclosure--reasoning'>"
        f"<summary><span class='wangp-assistant-chat__tool-title'><span class='wangp-assistant-chat__tool-chip'>Thought</span>{html.escape(label)}</span></summary>"
        f"<div class='wangp-assistant-chat__disclosure-body'>{body_html}</div>"
        "</details>"
    )


def _render_tool_blocks(tool_records: list[dict[str, Any]]) -> str:
    blocks = []
    for tool_record in tool_records:
        name = str(tool_record.get("name", "tool")).strip() or "tool"
        label = str(tool_record.get("label", "")).strip() or _friendly_tool_label(name)
        status = str(tool_record.get("status", "running")).strip().lower()
        status_label = {"running": "Running", "done": "Done", "error": "Error"}.get(status, status.title() or "Running")
        status_class = {"running": "running", "done": "done", "error": "error"}.get(status, "running")
        arguments_text = html.escape(json.dumps(tool_record.get("arguments", {}), ensure_ascii=False, indent=2, sort_keys=True))
        result_payload = tool_record.get("result", {})
        result_text = html.escape(json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True)) if result_payload is not None else ""
        blocks.append(
            "<details class='wangp-assistant-chat__disclosure wangp-assistant-chat__disclosure--tool' "
            + ("open" if status == "running" else "")
            + ">"
            f"<summary><span class='wangp-assistant-chat__tool-title'><span class='wangp-assistant-chat__tool-chip'>Tool</span>{html.escape(label)}</span><span class='wangp-assistant-chat__tool-status wangp-assistant-chat__tool-status--{status_class}'>{html.escape(status_label)}</span></summary>"
            "<div class='wangp-assistant-chat__disclosure-body'>"
            "<div class='wangp-assistant-chat__tool-grid'>"
            f"<div><div class='wangp-assistant-chat__tool-section-title'>{html.escape(label)} Arguments</div><pre class='wangp-assistant-chat__pre'>{arguments_text}</pre></div>"
            f"<div><div class='wangp-assistant-chat__tool-section-title'>Result</div><pre class='wangp-assistant-chat__pre'>{result_text or html.escape('Pending...')}</pre></div>"
            "</div>"
            "</div>"
            "</details>"
        )
    return "".join(blocks)


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
"""
"""
