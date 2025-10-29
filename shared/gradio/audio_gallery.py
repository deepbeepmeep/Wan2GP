import gradio as gr
from pathlib import Path
from datetime import datetime
import os
import json
import uuid


class AudioGallery:
    """
    A custom Gradio component that displays an audio gallery with thumbnails.

    Args:
        audio_paths: List of audio file paths
        selected_index: Initially selected index (default: 0)
        max_thumbnails: Maximum number of thumbnails to display (default: 10)
        height: Height of the gallery in pixels (default: 400)
        label: Label for the component (default: "Audio Gallery")
        update_only: If True, only render the inner HTML/Audio (internal use)
    """

    def __init__(self, audio_paths=None, selected_index=0, max_thumbnails=10, height=400, label="Audio Gallery", update_only=False):
        self.audio_paths = audio_paths or []
        self.selected_index = selected_index
        self.max_thumbnails = max_thumbnails
        self.height = height
        self.label = label
        self._render(update_only)

    # -------------------------------
    # Public API
    # -------------------------------

    @staticmethod
    def get_javascript():
        """
        Returns JavaScript code to append into Blocks(js=...).
        This code assumes it will be concatenated with other JS (no IIFE wrapper).
        """
        return r"""
// ===== AudioGallery: global-safe setup (no IIFE) =====
window.__agNS = window.__agNS || {};
const AG = window.__agNS;

// Singleton-ish state & guards
AG.state = AG.state || { manual: false, prevScroll: 0 };
AG.init  = AG.init  || false;
AG.moMap = AG.moMap || {}; // by element id, if needed later

// Helpers scoped on the namespace
AG.root = function () { return document.querySelector('#audio_gallery_html'); };
AG.container = function () { const r = AG.root(); return r ? r.querySelector('.thumbnails-container') : null; };

// Install global listeners once
if (!AG.init) {
  // Track scroll position of the thumbnails container
  document.addEventListener('scroll', (e) => {
    const c = AG.container();
    if (c && e.target === c) AG.state.prevScroll = c.scrollLeft;
  }, true);

  // Delegate click handling to select thumbnails (manual interaction)
  document.addEventListener('click', function (e) {
    const thumb = e.target.closest('.audio-thumbnail');
    if (!thumb) return;
    const c = AG.container();
    if (c) AG.state.prevScroll = c.scrollLeft;  // snapshot BEFORE backend updates
    const idx = thumb.getAttribute('data-index');
    window.selectAudioThumbnail(idx);
  });

  AG.init = true;
}

// Manual selection trigger (used by click handler and callable from elsewhere)
window.selectAudioThumbnail = function (index) {
  const c = AG.container();
  if (c) AG.state.prevScroll = c.scrollLeft;  // snapshot BEFORE Gradio re-renders
  AG.state.manual = true;

  const hiddenTextbox = document.querySelector('#audio_gallery_click_data textarea');
  const hiddenButton  = document.querySelector('#audio_gallery_click_trigger');
  if (hiddenTextbox && hiddenButton) {
    hiddenTextbox.value = String(index);
    hiddenTextbox.dispatchEvent(new Event('input', { bubbles: true }));
    hiddenButton.click();
  }

  // Brief window marking this as manual to suppress auto-scroll
  setTimeout(() => { AG.state.manual = false; }, 500);
};

// Ensure selected thumbnail is fully visible (programmatic only)
AG.ensureVisibleIfNeeded = function () {
  const c = AG.container();
  const r = AG.root();
  if (!c || !r) return;

  const sel = r.querySelector('.audio-thumbnail.selected');
  if (!sel) return;

  const left = sel.offsetLeft;
  const right = left + sel.clientWidth;
  const viewLeft  = c.scrollLeft;
  const viewRight = viewLeft + c.clientWidth;

  // Already fully visible
  if (left >= viewLeft && right <= viewRight) return;

  // Animate from CURRENT position (which we restore first)
  const target = left - (c.clientWidth / 2) + (sel.clientWidth / 2);
  const start  = c.scrollLeft;
  const dist   = target - start;
  const duration = 300;
  let t0 = null;

  function step(ts) {
    if (t0 === null) t0 = ts;
    const p = Math.min((ts - t0) / duration, 1);
    const ease = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
    c.scrollLeft = start + dist * ease;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
};

// Observe Gradio's DOM replacement of the HTML component and restore scroll
AG.installObserver = function () {
  const rootEl = AG.root();
  if (!rootEl) return;

  // Reuse/detach previous observer tied to this element id if needed
  const key = 'audio_gallery_html';
  if (AG.moMap[key]) { try { AG.moMap[key].disconnect(); } catch (_) {} }

  const mo = new MutationObserver(() => {
    const c = AG.container();
    if (!c) return;

    // 1) Always restore the last known scroll immediately (prevents jump-to-0)
    if (typeof AG.state.prevScroll === 'number') c.scrollLeft = AG.state.prevScroll;

    // 2) Only auto-scroll for programmatic changes
    if (!AG.state.manual) requestAnimationFrame(AG.ensureVisibleIfNeeded);
  });

  mo.observe(rootEl, { childList: true, subtree: true });
  AG.moMap[key] = mo;
};

// Try to install immediately; if not present yet, retry shortly
AG.tryInstall = function () {
  if (AG.root()) {
    AG.installObserver();
  } else {
    setTimeout(AG.tryInstall, 50);
  }
};

// Kick things off
AG.tryInstall();
// ===== end AudioGallery JS =====
"""

    def get_state(self):
        """Get the state components for use in other Gradio events. Returns: (state_paths, state_selected, refresh_trigger)"""
        return self.state_paths, self.state_selected, self.refresh_trigger

    def update(
        self,
        new_audio_paths=None,
        new_selected_index=None,
        current_paths_json=None,
        current_selected=None,
    ):
        """
        Programmatically update the gallery with new audio paths and/or selected index.
        Returns: (state_paths_json, state_selected_index, refresh_trigger_id)
        """
        # Decide which paths to use
        if new_audio_paths is not None:
            paths = new_audio_paths
            paths_json = json.dumps(paths)
        elif current_paths_json:
            paths_json = current_paths_json
            paths = json.loads(current_paths_json)
        else:
            paths = []
            paths_json = json.dumps([])

        audio_infos = self._process_audio_paths(paths)

        # Decide which selected index to use
        if new_selected_index is not None:
            try:
                selected_idx = int(new_selected_index)
            except Exception:
                selected_idx = 0
        elif current_selected is not None:
            try:
                selected_idx = int(current_selected)
            except Exception:
                selected_idx = 0
        else:
            selected_idx = 0

        if not audio_infos or selected_idx >= len(audio_infos) or selected_idx < 0:
            selected_idx = 0

        # Trigger id to notify the frontend to refresh (observed via MutationObserver on HTML rerender)
        refresh_id = str(uuid.uuid4())
        return paths_json, selected_idx, refresh_id

    # -------------------------------
    # Internal plumbing
    # -------------------------------

    def _render(self, update_only):
        """Internal render method called during initialization."""
        with gr.Column() as self.component:
            # Persistent state components
            self.state_paths = gr.Textbox(
                value=json.dumps(self.audio_paths),
                visible=False,
                elem_id="audio_gallery_state_paths",
            )

            selected_index = self.selected_index
            self.state_selected = gr.Number(
                value=selected_index,
                visible=False,
                elem_id="audio_gallery_state_selected",
            )

            # Trigger for refreshing the gallery (programmatic)
            self.refresh_trigger = gr.Textbox(
                value="",
                visible=False,
                elem_id="audio_gallery_refresh_trigger",
            )

            # Process audio paths (keep provided order)
            audio_infos = self._process_audio_paths(self.audio_paths)

            # Default selection
            default_audio = (
                audio_infos[selected_index]["path"]
                if audio_infos and selected_index < len(audio_infos)
                else (audio_infos[0]["path"] if audio_infos else None)
            )

            # Store for later use
            self.current_audio_infos = audio_infos

            # Wrapper for audio player with fixed height
            with gr.Column(elem_classes="audio-player-wrapper"):
                self.audio_player = gr.Audio(
                    value=default_audio, label=self.label, type="filepath"
                )

            # Create the gallery HTML (filename + thumbnails)
            self.gallery_html = gr.HTML(
                value=self._create_gallery_html(audio_infos, selected_index),
                elem_id="audio_gallery_html",  # stable anchor for MutationObserver
            )

            if update_only:
                return

            # Hidden textbox to capture clicks
            self.click_data = gr.Textbox(
                value="", visible=False, elem_id="audio_gallery_click_data"
            )

            # Hidden button to trigger the update
            self.click_trigger = gr.Button(
                visible=False, elem_id="audio_gallery_click_trigger"
            )

            # Set up the click handler
            self.click_trigger.click(
                fn=self._select_audio,
                inputs=[self.click_data, self.state_paths, self.state_selected],
                outputs=[
                    self.audio_player,
                    self.gallery_html,
                    self.state_paths,
                    self.state_selected,
                    self.click_data,
                ],
                show_progress="hidden",
            )

            # Set up the refresh handler (programmatic updates)
            self.refresh_trigger.change(
                fn=self._refresh_gallery,
                inputs=[self.refresh_trigger, self.state_paths, self.state_selected],
                outputs=[self.audio_player, self.gallery_html],
                show_progress="hidden",
            )

    def _select_audio(self, click_value, paths_json, current_selected):
        """Handle thumbnail selection (manual)."""
        if not click_value:
            return self._render_from_state(paths_json, current_selected)

        try:
            paths = json.loads(paths_json) if paths_json else []
            audio_infos = self._process_audio_paths(paths)

            if not audio_infos:
                return None, self._create_gallery_html([], 0), paths_json, 0, ""

            new_index = int(click_value)
            if 0 <= new_index < len(audio_infos):
                selected_path = audio_infos[new_index]["path"]
                return (
                    selected_path,
                    self._create_gallery_html(audio_infos, new_index),
                    paths_json,
                    new_index,
                    "",
                )
        except Exception:
            pass

        return self._render_from_state(paths_json, current_selected)

    def _refresh_gallery(self, refresh_id, paths_json, selected_idx):
        """Refresh gallery based on state (programmatic)."""
        if not refresh_id:
            return self._render_from_state(paths_json, selected_idx)[:2]

        try:
            paths = json.loads(paths_json) if paths_json else []
            audio_infos = self._process_audio_paths(paths)

            if not audio_infos:
                return None, self._create_gallery_html([], 0)

            selected_idx = int(selected_idx) if selected_idx is not None else 0
            if selected_idx >= len(audio_infos) or selected_idx < 0:
                selected_idx = 0

            selected_path = audio_infos[selected_idx]["path"]
            gallery_html_content = self._create_gallery_html(audio_infos, selected_idx)

            return selected_path, gallery_html_content
        except Exception:
            return None, self._create_gallery_html([], 0)

    def _get_audio_duration(self, audio_path):
        """Get audio duration in seconds. Returns formatted string."""
        try:
            import wave
            import contextlib

            # Try WAV format first
            try:
                with contextlib.closing(wave.open(audio_path, "r")) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    return self._format_duration(duration)
            except Exception:
                pass

            # For other formats, try using mutagen if available
            try:
                from mutagen import File
                audio = File(audio_path)
                if audio and getattr(audio, "info", None):
                    return self._format_duration(audio.info.length)
            except Exception:
                pass

            # Fallback to file size estimation (very rough)
            file_size = os.path.getsize(audio_path)
            estimated_duration = file_size / 32000  # bytes per second guess
            return self._format_duration(estimated_duration)

        except Exception:
            return "0:00"

    def _format_duration(self, seconds):
        """Format duration in seconds to MM:SS format."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def _get_file_info(self, audio_path):
        """Get file information: basename, date/time, duration."""
        p = Path(audio_path)
        basename = p.name

        # Get modification time
        mtime = os.path.getmtime(audio_path)
        dt = datetime.fromtimestamp(mtime)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")

        # Get duration
        duration = self._get_audio_duration(audio_path)

        return {
            "basename": basename,
            "date": date_str,
            "time": time_str,
            "duration": duration,
            "path": audio_path,
            "timestamp": mtime,
        }

    def _create_thumbnail_html(self, info, index, is_selected):
        """Create HTML for a thumbnail."""
        selected_class = "selected" if is_selected else ""
        return f"""
        <div class="audio-thumbnail {selected_class}"
             data-index="{index}"
             data-path="{info['path']}"
             title="{info['basename']}">
            <div class="thumbnail-date">{info['date']}</div>
            <div class="thumbnail-time">{info['time']}</div>
            <div class="thumbnail-duration">{info['duration']}</div>
        </div>
        """

    def _create_gallery_html(self, audio_infos, selected_index):
        """Create the complete gallery HTML."""
        thumbnails_html = ""
        num_thumbnails = len(audio_infos)

        # Calculate thumbnail width based on number of thumbnails
        if num_thumbnails > 0:
            thumbnail_width = max(80, min(150, 100 - (num_thumbnails - 1) * 2))
        else:
            thumbnail_width = 100

        for i, info in enumerate(audio_infos):
            is_selected = i == selected_index
            thumbnails_html += self._create_thumbnail_html(info, i, is_selected)

        selected_basename = (
            audio_infos[selected_index]["basename"]
            if (audio_infos and 0 <= selected_index < len(audio_infos))
            else ""
        )

        gallery_html = f"""
        <style>
            :root {{
                --bg-primary: #fafafa;
                --bg-secondary: white;
                --bg-selected-filename: #f0f0f0;
                --bg-selected-thumbnail: #E3F2FD;
                --bg-tooltip: rgba(0, 0, 0, 0.85);
                --text-primary: #333;
                --text-secondary: #666;
                --text-tooltip: white;
                --border-primary: #e0e0e0;
                --border-secondary: #d0d0d0;
                --accent-color: #2196F3;
                --shadow-color: rgba(33, 150, 243, 0.3);
                --shadow-color-selected: rgba(33, 150, 243, 0.4);
            }}

            @media (prefers-color-scheme: dark) {{
                :root {{
                    --bg-primary: #27272a;
                    --bg-secondary: #52525b;
                    --bg-selected-filename: #2c2c2c;
                    --bg-selected-thumbnail: #0d2a40;
                    --bg-tooltip: rgba(255, 255, 255, 0.85);
                    --text-primary: #e0e0e0;
                    --text-secondary: #a0a0a0;
                    --text-tooltip: black;
                    --border-primary: #333;
                    --border-secondary: #444;
                    --shadow-color: rgba(33, 150, 243, 0.5);
                    --shadow-color-selected: rgba(33, 150, 243, 0.6);
                }}
            }}

            /* Fix audio player height */
            .audio-player-wrapper {{
                min-height: 200px;
                max-height: 200px;
                overflow: hidden;
                margin-bottom: 0 !important;
                padding-bottom: 0 !important;
            }}

            .audio-player-wrapper .audio {{
                height: 100% !important;
                margin-bottom: 0 !important;
            }}

            .audio-gallery-container {{
                display: flex;
                flex-direction: column;
                overflow: hidden;
                margin-top: 0 !important;
                padding-top: 0 !important;
            }}

            .selected-filename {{
                padding: 4px 12px;
                background: var(--bg-selected-filename);
                border-radius: 4px;
                font-size: 14px;
                font-weight: 500;
                margin: 0 0 8px 0;
                text-align: center;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                color: var(--text-primary);
            }}

            .thumbnails-container {{
                display: flex;
                justify-content: center;
                gap: 8px;
                padding: 12px;
                overflow-x: auto;
                overflow-y: hidden;
                flex-direction: row;
                border: 1px solid var(--border-primary);
                border-radius: 8px;
                background: var(--bg-primary);
                min-height: 120px;
            }}

            .audio-thumbnail {{
                position: relative;
                min-width: {thumbnail_width}px;
                width: {thumbnail_width}px;
                flex-shrink: 0;
                padding: 12px;
                border: 2px solid var(--border-secondary);
                border-radius: 8px;
                cursor: pointer;
                background: var(--bg-secondary);
                transition: all 0.2s ease;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 4px;
            }}

            .audio-thumbnail:hover {{
                border-color: var(--accent-color);
                box-shadow: 0 2px 8px var(--shadow-color);
                transform: translateY(-2px);
            }}

            .audio-thumbnail:hover::after {{
                content: attr(title);
                position: absolute;
                bottom: calc(100% + 5px);
                left: 50%;
                transform: translateX(-50%);
                background: var(--bg-tooltip);
                color: var(--text-tooltip);
                padding: 6px 10px;
                border-radius: 4px;
                font-size: 12px;
                white-space: nowrap;
                z-index: 1000;
                pointer-events: none;
                max-width: {thumbnail_width}px;
                overflow: hidden;
                text-overflow: ellipsis;
            }}

            .audio-thumbnail.selected {{
                border-color: var(--accent-color);
                background: var(--bg-selected-thumbnail);
                box-shadow: 0 2px 12px var(--shadow-color-selected);
            }}

            .thumbnail-date {{
                font-size: 13px;
                font-weight: 600;
                color: var(--text-primary);
                white-space: nowrap;
            }}

            .thumbnail-time {{
                font-size: 12px;
                color: var(--text-secondary);
                white-space: nowrap;
            }}

            .thumbnail-duration {{
                font-size: 13px;
                font-weight: bold;
                color: var(--accent-color);
                margin-top: 4px;
                white-space: nowrap;
            }}
        </style>

        <div class="audio-gallery-container">
            <div class="selected-filename" id="selected-filename">{selected_basename}</div>

            <div class="thumbnails-container" id="thumbnails-container">
                {thumbnails_html}
            </div>

            <script>
                /* No-op; JS is injected globally via Blocks(js=AudioGallery.get_javascript()) */
            </script>
        </div>
        """
        return gallery_html

    def _process_audio_paths(self, paths):
        """Process audio paths and return audio infos in the same order."""
        audio_infos = []
        if paths:
            for path in paths:
                try:
                    if os.path.exists(path):
                        audio_infos.append(self._get_file_info(path))
                except Exception:
                    continue
            audio_infos = audio_infos[: self.max_thumbnails]
        return audio_infos

    def _render_from_state(self, paths_json, selected_idx):
        """Render gallery from state."""
        try:
            paths = json.loads(paths_json) if paths_json else []
            audio_infos = self._process_audio_paths(paths)

            if not audio_infos:
                return None, self._create_gallery_html([], 0), paths_json, 0, ""

            selected_idx = int(selected_idx) if selected_idx is not None else 0
            if selected_idx >= len(audio_infos) or selected_idx < 0:
                selected_idx = 0

            selected_path = audio_infos[selected_idx]["path"]
            gallery_html_content = self._create_gallery_html(audio_infos, selected_idx)

            return selected_path, gallery_html_content, paths_json, selected_idx, ""
        except Exception:
            return None, self._create_gallery_html([], 0), paths_json, 0, ""
