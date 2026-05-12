import html
import re


def _render_inline_markdown(text: str) -> str:
    rendered = html.escape(text, quote=False)
    rendered = re.sub(r"`([^`]+)`", lambda match: f"<code>{match.group(1)}</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", lambda match: f"<strong>{match.group(1)}</strong>", rendered)
    return rendered


def _render_markdown(markdown: str) -> str:
    lines = str(markdown or "").strip().splitlines()
    parts, paragraph, list_items, code_lines = [], [], [], []
    in_code = False
    code_lang = ""

    def flush_paragraph():
        if paragraph:
            parts.append(f"<p>{_render_inline_markdown(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list():
        if list_items:
            parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>")
            list_items.clear()

    def flush_code():
        if code_lines:
            lang_class = f" class='language-{html.escape(code_lang, quote=True)}'" if code_lang else ""
            parts.append(f"<pre><code{lang_class}>{html.escape(chr(10).join(code_lines), quote=False)}</code></pre>")
            code_lines.clear()

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
                code_lang = ""
            else:
                flush_paragraph()
                flush_list()
                in_code = True
                code_lang = re.sub(r"[^A-Za-z0-9_-]", "", stripped[3:].strip())
            continue
        if in_code:
            code_lines.append(raw_line.rstrip())
            continue
        if not stripped:
            flush_paragraph()
            flush_list()
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1)) + 1
            parts.append(f"<h{level}>{_render_inline_markdown(heading.group(2).strip())}</h{level}>")
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            list_items.append(_render_inline_markdown(stripped[2:].strip()))
            continue
        flush_list()
        paragraph.append(stripped)

    flush_code()
    flush_paragraph()
    flush_list()
    return "\n".join(parts)


def _normalize_infos(infos, model_name: str) -> tuple[str, str]:
    return str(model_name or "Model"), str(infos or "")


def render_model_description(description: str, infos=None, *, model_type: str = "", model_name: str = "Model", height: int = 40) -> str:
    if not infos:
        return f"<div style='height:{int(height)}px'>{description}</div>"
    title, markdown = _normalize_infos(infos, model_name)
    if not markdown.strip():
        return f"<div style='height:{int(height)}px'>{description}</div>"
    popup_id = "wangp-model-info-" + re.sub(r"[^A-Za-z0-9_-]", "-", str(model_type or model_name)).strip("-").lower()
    title_attr = html.escape(title, quote=True)
    title_html = html.escape(title)
    return (
        f"<div class='wangp-model-info-host' style='min-height:{int(height)}px'>"
        f"<div class='wangp-model-info-description'>{description}</div>"
        f"<button type='button' class='wangp-model-info-trigger' title='{title_attr}' aria-label='{title_attr}' data-wangp-model-info-open='{popup_id}'>&#9432;</button>"
        f"<div id='{popup_id}' class='wangp-model-info-popup' role='dialog' aria-label='{title_attr}' data-wangp-model-info-popup hidden>"
        "<div class='wangp-model-info-card'>"
        "<div class='wangp-model-info-titlebar' data-wangp-model-info-drag>"
        f"<div class='wangp-model-info-heading'>{title_html}</div>"
        "<button type='button' class='wangp-model-info-close' aria-label='Close information' data-wangp-model-info-close>&times;</button>"
        "</div>"
        f"<div class='wangp-model-info-content'>{_render_markdown(markdown)}</div>"
        "</div>"
        "</div>"
        "</div>"
    )


def get_css() -> str:
    return """
.wangp-model-info-host {
    position: relative;
    padding-right: 0;
}
.header-markdown-group .html-container {
    padding: 0 !important;
}
.wangp-model-info-description {
    line-height: 1.35;
}
.wangp-model-info-trigger {
    position: absolute;
    top: 1px;
    right: 2px;
    width: 26px;
    height: 26px;
    min-width: 26px;
    min-height: 26px;
    padding: 0;
    border: 1px solid rgba(17, 84, 118, 0.18);
    border-radius: 999px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(238, 247, 252, 0.98) 100%);
    color: #155574;
    box-shadow: 0 8px 18px rgba(11, 44, 63, 0.10);
    cursor: pointer;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.wangp-model-info-trigger:hover {
    border-color: rgba(16, 86, 121, 0.36);
    box-shadow: 0 10px 22px rgba(11, 44, 63, 0.16);
}
.wangp-model-info-popup[hidden] {
    display: none !important;
}
.wangp-model-info-popup {
    position: fixed;
    top: 96px;
    right: 32px;
    width: min(680px, calc(100vw - 34px));
    max-height: min(78vh, 720px);
    z-index: 1200;
    pointer-events: none;
}
.wangp-model-info-card {
    pointer-events: auto;
    overflow: hidden;
    border-radius: 18px;
    border: 1px solid rgba(17, 84, 118, 0.16);
    background: rgba(255, 255, 255, 0.99);
    box-shadow: 0 28px 62px rgba(7, 31, 48, 0.24);
}
.wangp-model-info-titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 10px 14px 9px 16px;
    background: linear-gradient(180deg, rgba(16, 86, 121, 0.98) 0%, rgba(10, 59, 84, 0.98) 100%);
    color: #f3fbff;
    cursor: grab;
    user-select: none;
    touch-action: none;
}
.wangp-model-info-titlebar:active {
    cursor: grabbing;
}
.wangp-model-info-heading {
    color: #f3fbff !important;
    font-size: 0.92rem;
    font-weight: 800;
}
.wangp-model-info-close {
    width: 26px;
    height: 26px;
    min-width: 26px;
    min-height: 26px;
    padding: 0;
    border: 1px solid rgba(255, 255, 255, 0.24);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    color: #f3fbff;
    cursor: pointer;
    font-size: 20px;
    line-height: 1;
}
.wangp-model-info-content {
    max-height: calc(min(78vh, 720px) - 46px);
    overflow: auto;
    padding: 16px 18px 18px;
    color: #174a67;
    font-size: 0.92rem;
    line-height: 1.5;
}
.wangp-model-info-content h2,
.wangp-model-info-content h3,
.wangp-model-info-content h4 {
    margin: 12px 0 7px;
    color: #103f59;
    font-weight: 800;
}
.wangp-model-info-content h2:first-child,
.wangp-model-info-content h3:first-child,
.wangp-model-info-content h4:first-child {
    margin-top: 0;
}
.wangp-model-info-content p,
.wangp-model-info-content ul {
    margin: 0 0 11px;
}
.wangp-model-info-content ul {
    padding-left: 20px;
}
.wangp-model-info-content code {
    padding: 1px 4px;
    border-radius: 5px;
    background: rgba(16, 86, 121, 0.08);
    color: #0f4967;
}
.wangp-model-info-content pre {
    margin: 8px 0 13px;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid rgba(17, 84, 118, 0.12);
    background: #f4f9fc;
    overflow: auto;
}
.wangp-model-info-content pre code {
    padding: 0;
    border-radius: 0;
    background: transparent;
    color: #123f58;
}
"""


def get_javascript() -> str:
    return """
    window.wangpModelInfo = window.wangpModelInfo || {};
    window.wangpModelInfo.open = function(button) {
        const popupId = button?.getAttribute("data-wangp-model-info-open");
        const popup = popupId ? document.getElementById(popupId) : null;
        if (!popup) return;
        document.querySelectorAll("[data-wangp-model-info-popup]").forEach((other) => {
            if (other !== popup) other.hidden = true;
        });
        popup.hidden = false;
        popup.style.left = "auto";
        popup.style.right = "32px";
        popup.style.top = "96px";
    };
    window.wangpModelInfo.close = function(closeButton) {
        const popup = closeButton?.closest("[data-wangp-model-info-popup]");
        if (popup) popup.hidden = true;
    };
    let wangpModelInfoDrag = null;
    document.addEventListener("click", (event) => {
        const openButton = event.target.closest("[data-wangp-model-info-open]");
        if (openButton) {
            event.preventDefault();
            event.stopPropagation();
            window.wangpModelInfo.open(openButton);
            return;
        }
        const closeButton = event.target.closest("[data-wangp-model-info-close]");
        if (closeButton) {
            event.preventDefault();
            event.stopPropagation();
            window.wangpModelInfo.close(closeButton);
        }
    });
    document.addEventListener("pointerdown", (event) => {
        const handle = event.target.closest("[data-wangp-model-info-drag]");
        if (!handle || event.target.closest("[data-wangp-model-info-close]")) return;
        const popup = handle.closest("[data-wangp-model-info-popup]");
        if (!popup) return;
        const rect = popup.getBoundingClientRect();
        popup.style.left = rect.left + "px";
        popup.style.top = rect.top + "px";
        popup.style.right = "auto";
        wangpModelInfoDrag = { popup, pointerId: event.pointerId, offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top };
        handle.setPointerCapture?.(event.pointerId);
        event.preventDefault();
    });
    document.addEventListener("pointermove", (event) => {
        if (!wangpModelInfoDrag || wangpModelInfoDrag.pointerId !== event.pointerId) return;
        const margin = 10;
        const popup = wangpModelInfoDrag.popup;
        const rect = popup.getBoundingClientRect();
        const left = Math.min(Math.max(margin, event.clientX - wangpModelInfoDrag.offsetX), Math.max(margin, window.innerWidth - rect.width - margin));
        const top = Math.min(Math.max(margin, event.clientY - wangpModelInfoDrag.offsetY), Math.max(margin, window.innerHeight - 48));
        popup.style.left = left + "px";
        popup.style.top = top + "px";
        event.preventDefault();
    });
    document.addEventListener("pointerup", (event) => {
        if (wangpModelInfoDrag && wangpModelInfoDrag.pointerId === event.pointerId) wangpModelInfoDrag = null;
    });
    document.addEventListener("pointercancel", (event) => {
        if (wangpModelInfoDrag && wangpModelInfoDrag.pointerId === event.pointerId) wangpModelInfoDrag = null;
    });
"""
