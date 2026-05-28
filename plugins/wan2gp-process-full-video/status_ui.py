from __future__ import annotations

import html
import time

from shared.api import extract_status_phase_label


def phase_label_from_status(status: str = "") -> str:
    return extract_status_phase_label(status)


def phase_label_from_update(update=None, *, status: str = "", phase: str = "", raw_phase: str = "") -> str:
    status_phase = phase_label_from_status(status or getattr(update, "status", ""))
    raw_phase_text = str(raw_phase or getattr(update, "raw_phase", "") or phase or "").strip()
    if len(status_phase) > 0:
        return status_phase
    return raw_phase_text


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = int(round(float(seconds)))
    if total_seconds < 0:
        total_seconds = 0
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds_only = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_only:02d}" if hours > 0 else f"{minutes:02d}:{seconds_only:02d}"


class ChunkCallbacks:
    def __init__(self) -> None:
        self.phase_label = "Queued in WanGP..."
        self.status_text = "Queued in WanGP..."
        self.current_step = None
        self.total_steps = None
        self._last_explicit_status_at = 0.0

    def on_status(self, status):
        status_text = str(status or "").strip()
        if len(status_text) == 0:
            return
        self.status_text = status_text
        status_phase = phase_label_from_status(self.status_text)
        if len(status_phase) > 0:
            if status_phase != self.phase_label:
                self.current_step = None
                self.total_steps = None
            self.phase_label = status_phase
            self._last_explicit_status_at = time.time()

    def on_progress(self, update):
        incoming_status = str(getattr(update, "status", "") or "").strip()
        incoming_phase = phase_label_from_update(update, status=incoming_status or self.status_text)
        incoming_step = getattr(update, "current_step", None)
        incoming_total = getattr(update, "total_steps", None)
        if time.time() - self._last_explicit_status_at <= 1.0 and len(self.phase_label) > 0 and len(incoming_phase) > 0 and incoming_phase.lower() != self.phase_label.lower() and incoming_step is None:
            return
        if len(incoming_status) > 0:
            self.status_text = incoming_status
        if len(incoming_phase) > 0:
            self.phase_label = incoming_phase
        self.current_step = incoming_step
        self.total_steps = incoming_total


def render_chunk_status_html(total_chunks: int, completed_chunks: int, current_chunk: int, phase_label: str, status_text: str, *, continued: bool = False, phase_current_step=None, phase_total_steps=None, elapsed_seconds: float | None = None, eta_seconds: float | None = None, prefer_status_phase: bool = False) -> str:
    total_chunks = int(total_chunks)
    completed_chunks = int(completed_chunks)
    current_chunk = int(current_chunk)
    if total_chunks > 0:
        top_ratio = completed_chunks / total_chunks
        chunks_text = f"{completed_chunks} / {total_chunks}"
    else:
        top_ratio = 0.0
        chunks_text = "- / -"
    top_width = f"{100.0 * top_ratio:.2f}%"
    raw_status_text = status_text.strip()
    raw_phase_text = phase_label.strip()
    if prefer_status_phase:
        derived_phase = phase_label_from_status(raw_status_text)
        if len(derived_phase) > 0:
            raw_phase_text = derived_phase
    phase_html = html.escape(raw_phase_text or "Queued in WanGP...")
    status_html = html.escape(raw_status_text or raw_phase_text or "")
    continued_suffix = " (Continued)" if continued else ""
    has_phase_progress = phase_current_step is not None and phase_total_steps is not None and phase_total_steps > 0
    phase_ratio = float(phase_current_step) / float(phase_total_steps) if has_phase_progress else None
    phase_width = f"{100.0 * phase_ratio:.2f}%" if phase_ratio is not None else "0%"
    phase_suffix = f" ({phase_current_step} / {phase_total_steps})" if has_phase_progress else ""
    elapsed_html = html.escape(_format_elapsed(elapsed_seconds))
    eta_html = html.escape(_format_elapsed(eta_seconds))
    normalized_phase = raw_phase_text.lower()
    normalized_status = raw_status_text
    show_status_line = (not prefer_status_phase) and len(normalized_status) > 0 and (len(normalized_phase) == 0 or normalized_phase not in normalized_status.lower())
    status_line_html = f"<div style='font-size:0.9em;color:#4b5563'>{status_html}</div>" if show_status_line else ""
    return (
        "<div style='display:flex;flex-direction:column;gap:8px'>"
        f"<div style='font-weight:600'>Chunks Processed: {chunks_text}{continued_suffix}</div>"
        "<div style='height:12px;border-radius:999px;background:#d7dce3;overflow:hidden'>"
        f"<div style='height:100%;width:{top_width};background:linear-gradient(90deg,#2f7de1,#5db0ff)'></div>"
        "</div>"
        f"<div style='font-size:0.95em'><b>Phase:</b> {phase_html}{phase_suffix}</div>"
        "<div style='height:12px;border-radius:999px;background:#d7dce3;overflow:hidden'>"
        f"<div style='height:100%;width:{phase_width};background:linear-gradient(90deg,#e37a2f,#ffb05d)'></div>"
        "</div>"
        f"<div style='font-size:0.9em;color:#4b5563'><b>Elapsed:</b> {elapsed_html} <span style='padding-left:12px'><b>ETA:</b> {eta_html}</span></div>"
        f"{status_line_html}"
        "</div>"
    )


def render_output_file_html(output_path: str) -> str:
    value = html.escape(output_path, quote=False)
    return (
        "<div style='display:flex;flex-direction:column;gap:6px'>"
        "<div style='font-size:var(--block-label-text-size);font-weight:var(--block-label-text-weight);line-height:var(--line-sm)'>Output File</div>"
        f"<textarea readonly onclick='this.select()' spellcheck='false' rows='1' "
        "style='width:100%;min-height:35.64px;resize:none;overflow:hidden;padding:calc(8px * var(--wangp-ui-scale)) calc(12px * var(--wangp-ui-scale));"
        "border:1px solid var(--input-border-color);border-radius:var(--input-radius);background:var(--input-background-fill);color:var(--body-text-color);"
        "font:inherit;line-height:1.5;box-sizing:border-box'>"
        f"{value}</textarea>"
        "</div>"
    )


