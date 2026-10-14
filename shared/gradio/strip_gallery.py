# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Galleries shown as a player above a strip of thumbnails (audio files, 3D assets).

StripGallery holds the state components, the thumbnail strip and the selection plumbing shared by every strip gallery; a subclass
only says how a file is played (the player component and its value) and how its thumbnail looks. Element ids derive from the
gallery source id (`<id>_gallery_html`, `<id>_gallery_state_paths`, ...), so each gallery keeps stable anchors for scripts and tests.
"""
import html
import json
import os
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from shared.utils.gallery_sources import AUDIO, MESH


def _get_selected_idx(infos, selected_idx):
    selected_idx = int(selected_idx) if selected_idx is not None else 0
    if selected_idx >= len(infos):
        selected_idx = len(infos) - 1
    elif selected_idx < 0:
        selected_idx = 0
    if len(infos) == 0:
        selected_idx = -1
    return selected_idx


def gradio_file_url(path):
    return "/gradio_api/file=" + quote(os.path.abspath(path), safe="")


class StripGallery:
    """A strip gallery bound to a gallery source.

    Args:
        paths: initial file paths
        selected_index: initially selected index
        max_thumbnails: maximum number of thumbnails (None: no limit)
        label: player label
        update_only: if True, only render the player and the strip (form refresh)
    """

    source = None
    _timestamp_cache = OrderedDict()
    _timestamp_cache_max_entries = 50
    _gradio_upload_mtime_patch_installed = False

    def __init__(self, paths=None, selected_index=-1, max_thumbnails=10, label=None, update_only=False):
        self.paths = paths or []
        self.selected_index = selected_index
        self.max_thumbnails = max_thumbnails
        self.label = label or self.source.label
        self._render(update_only)

    def element_id(self, suffix):
        return f"{self.source.id}_gallery_{suffix}"

    # -------------------------------
    # Subclass hooks
    # -------------------------------

    def _create_player(self, path):
        raise NotImplementedError

    def _player_value(self, path):
        return path

    def _thumbnail_body(self, info):
        raise NotImplementedError

    def _extra_file_info(self, path):
        return {}

    # -------------------------------
    # Public API
    # -------------------------------

    @staticmethod
    def get_javascript():
        """JavaScript for Blocks(js=...): the strip plumbing of every strip gallery, then each gallery's own script."""
        ids = json.dumps(list(STRIP_GALLERY_CLASSES))
        return StripGallery._javascript.replace("__STRIP_GALLERY_IDS__", ids) + "".join(gallery._own_javascript for gallery in STRIP_GALLERY_CLASSES.values())

    _own_javascript = ""
    _javascript = r"""
// ===== StripGallery: global-safe setup (no IIFE) =====
window.__stripGalleryNS = window.__stripGalleryNS || {};
const SG = window.__stripGalleryNS;
SG.state = SG.state || {};  // per gallery id: {manual, prevScroll}
SG.moMap = SG.moMap || {};
SG.uploadPatchInit = SG.uploadPatchInit || false;

SG.galleryState = function (id) { return SG.state[id] = SG.state[id] || { manual: false, prevScroll: 0 }; };
SG.root = function (id) { return document.querySelector('#' + id + '_gallery_html'); };
SG.container = function (id) { const r = SG.root(id); return r ? r.querySelector('.thumbnails-container') : null; };

if (!SG.init) {
  // Track the scroll position of each thumbnail strip
  document.addEventListener('scroll', (e) => {
    const strip = e.target.closest && e.target.closest('.strip-gallery-container');
    if (strip && e.target.classList.contains('thumbnails-container')) SG.galleryState(strip.dataset.gallery).prevScroll = e.target.scrollLeft;
  }, true);
  // Delegate thumbnail clicks (manual selection)
  document.addEventListener('click', function (e) {
    const thumb = e.target.closest('.strip-thumbnail');
    if (!thumb) return;
    const id = thumb.dataset.gallery, c = SG.container(id);
    if (c) SG.galleryState(id).prevScroll = c.scrollLeft;  // snapshot BEFORE backend updates
    window.selectStripThumbnail(id, thumb.getAttribute('data-index'));
  });
  SG.init = true;
}

// Patch upload requests once to forward browser lastModified timestamps.
if (!SG.uploadPatchInit) {
  SG._origFetch = SG._origFetch || window.fetch.bind(window);
  window.fetch = function (input, init) {
    let nextInit = init;
    try {
      const reqUrl = (typeof input === 'string') ? input : (input && input.url ? input.url : '');
      const reqMethod = ((init && init.method) || (input && input.method) || 'GET').toUpperCase();
      const reqBody = init && init.body;
      const isUploadRequest = reqMethod === 'POST' && typeof reqUrl === 'string' && (reqUrl.includes('/upload') || reqUrl.includes('/gradio_api/upload'));
      if (isUploadRequest && reqBody instanceof FormData && reqBody.has('files')) {
        const lastModifiedValues = reqBody.getAll('files').map(oneFile => (typeof File !== 'undefined' && oneFile instanceof File) ? Number(oneFile.lastModified || 0) : 0);
        if (lastModifiedValues.length > 0) {
          const headers = new Headers((init && init.headers) || {});
          headers.set('x-wangp-file-last-modified', JSON.stringify(lastModifiedValues));
          nextInit = Object.assign({}, init || {}, { headers });
        }
      }
    } catch (_) {}
    return SG._origFetch(input, nextInit);
  };
  SG.uploadPatchInit = true;
}

// Manual selection trigger (used by the click handler and callable from elsewhere)
window.selectStripThumbnail = function (id, index) {
  if (window.WanGPGallerySelection && document.querySelector('#wangp-gallery-view')) {
    window.WanGPGallerySelection.select(id, index);
    return;
  }
  const c = SG.container(id), state = SG.galleryState(id);
  if (c) state.prevScroll = c.scrollLeft;  // snapshot BEFORE Gradio re-renders
  state.manual = true;
  const hiddenTextbox = document.querySelector('#' + id + '_gallery_click_data textarea');
  const hiddenButton = document.querySelector('#' + id + '_gallery_click_trigger');
  if (hiddenTextbox && hiddenButton) {
    hiddenTextbox.value = String(index);
    hiddenTextbox.dispatchEvent(new Event('input', { bubbles: true }));
    hiddenButton.click();
  }
  // Brief window marking this as manual to suppress auto-scroll
  setTimeout(() => { state.manual = false; }, 500);
};

// Ensure the selected thumbnail is fully visible (programmatic changes only)
SG.ensureVisibleIfNeeded = function (id) {
  const c = SG.container(id), r = SG.root(id);
  const sel = r && r.querySelector('.strip-thumbnail.selected');
  if (!c || !sel) return;
  const left = sel.offsetLeft, right = left + sel.clientWidth;
  if (left >= c.scrollLeft && right <= c.scrollLeft + c.clientWidth) return;
  const target = left - (c.clientWidth / 2) + (sel.clientWidth / 2), start = c.scrollLeft, dist = target - start, duration = 300;
  let t0 = null;
  function step(ts) {
    if (t0 === null) t0 = ts;
    const p = Math.min((ts - t0) / duration, 1);
    c.scrollLeft = start + dist * (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
};

// Observe Gradio's DOM replacement of a gallery strip and restore its scroll
SG.installObserver = function (id) {
  const rootEl = SG.root(id);
  if (!rootEl) return false;
  if (SG.moMap[id]) { try { SG.moMap[id].disconnect(); } catch (_) {} }
  const mo = new MutationObserver(() => {
    const c = SG.container(id), state = SG.galleryState(id);
    if (!c) return;
    c.scrollLeft = state.prevScroll;  // prevents the jump to 0
    if (!state.manual) requestAnimationFrame(() => SG.ensureVisibleIfNeeded(id));
  });
  mo.observe(rootEl, { childList: true, subtree: true });
  SG.moMap[id] = mo;
  return true;
};

// Call back once an element exists (a gallery tab renders its content when first shown)
SG.whenElement = function (selector, callback, delay = 50) {
  const el = document.querySelector(selector);
  if (el) callback(el);
  else setTimeout(() => SG.whenElement(selector, callback, Math.min(delay * 2, 500)), delay);
};
for (const id of __STRIP_GALLERY_IDS__) SG.whenElement('#' + id + '_gallery_html', () => SG.installObserver(id));
// ===== end StripGallery JS =====
"""

    @classmethod
    def install_gradio_upload_mtime_patch(cls):
        """Monkey-patch Gradio multipart parser to keep browser file lastModified as temp-file mtime."""
        if cls._gradio_upload_mtime_patch_installed:
            return True
        try:
            from gradio.route_utils import GradioMultiPartParser, GradioUploadFile
        except Exception:
            return False

        original_parse = getattr(GradioMultiPartParser, "parse", None)
        if original_parse is None:
            return False
        if getattr(original_parse, "__wangp_upload_mtime_patch__", False):
            cls._gradio_upload_mtime_patch_installed = True
            return True

        header_name = "x-wangp-file-last-modified"

        async def patched_parse(parser_self, *args, **kwargs):
            form = await original_parse(parser_self, *args, **kwargs)
            try:
                header_payload = None
                if hasattr(parser_self, "headers") and parser_self.headers is not None:
                    header_payload = parser_self.headers.get(header_name)
                if not header_payload:
                    return form

                mtime_values = json.loads(header_payload)
                if not isinstance(mtime_values, list) or len(mtime_values) == 0:
                    return form

                uploaded_files = form.getlist("files")
                for idx, file_obj in enumerate(uploaded_files):
                    if idx >= len(mtime_values):
                        break
                    ts_ms = mtime_values[idx]
                    if isinstance(ts_ms, dict):
                        ts_ms = ts_ms.get("lastModified", ts_ms.get("last_modified", 0))
                    try:
                        ts_ms = float(ts_ms)
                    except Exception:
                        continue
                    if ts_ms <= 0:
                        continue
                    ts = ts_ms / 1000.0 if ts_ms > 100_000_000_000 else ts_ms

                    tmp_path = None
                    if isinstance(file_obj, GradioUploadFile):
                        tmp_path = getattr(getattr(file_obj, "file", None), "name", None)
                    elif hasattr(file_obj, "file"):
                        tmp_path = getattr(file_obj.file, "name", None)
                    if tmp_path and os.path.exists(tmp_path):
                        os.utime(tmp_path, (ts, ts))
            except Exception:
                pass
            return form

        patched_parse.__wangp_upload_mtime_patch__ = True
        GradioMultiPartParser.parse = patched_parse
        cls._gradio_upload_mtime_patch_installed = True
        return True

    def get_state(self):
        """State components for other Gradio events: (state_paths, state_selected, refresh_trigger)."""
        return self.state_paths, self.state_selected, self.refresh_trigger

    # -------------------------------
    # Internal plumbing
    # -------------------------------

    def _render(self, update_only):
        with gr.Column() as self.component:
            self.state_paths = gr.Textbox(value=json.dumps(self.paths), visible=False, elem_id=self.element_id("state_paths"))
            selected_index = self.selected_index
            self.state_selected = gr.Number(value=selected_index, visible=False, elem_id=self.element_id("state_selected"))
            # Trigger for refreshing the gallery (programmatic)
            self.refresh_trigger = gr.Textbox(value="", visible=False, elem_id=self.element_id("refresh_trigger"))

            infos = self._process_paths(self.paths)
            default_path = infos[selected_index]["path"] if infos and 0 <= selected_index < len(infos) else (infos[0]["path"] if infos else None)
            with gr.Column(elem_classes=["strip-player-wrapper", f"{self.source.id}-player-wrapper"]):
                self.player = self._create_player(default_path)

            # The thumbnails strip; its id is the stable anchor of the MutationObserver
            self.gallery_html = gr.HTML(value=self._create_gallery_html(infos, selected_index), elem_id=self.element_id("html"), elem_classes="strip-gallery-root")

            if update_only:
                return

            # Hidden textbox and button capturing thumbnail clicks
            self.click_data = gr.Textbox(value="", visible=False, elem_id=self.element_id("click_data"))
            self.click_trigger = gr.Button(visible=False, elem_id=self.element_id("click_trigger"))
            self.click_trigger.click(fn=self._select, inputs=[self.click_data, self.state_paths, self.state_selected], outputs=[self.player, self.gallery_html, self.state_paths, self.state_selected, self.click_data], show_progress="hidden")
            # Programmatic updates
            self.refresh_trigger.change(fn=self._refresh_gallery, inputs=[self.refresh_trigger, self.state_paths, self.state_selected], outputs=[self.player, self.gallery_html, self.state_selected], show_progress="hidden")

    def _select(self, click_value, paths_json, current_selected):
        """Handle a thumbnail selection (manual)."""
        if not click_value:
            return self._render_from_state(paths_json, current_selected)
        try:
            paths = json.loads(paths_json) if paths_json else []
            infos = self._process_paths(paths)
            if not infos:
                return self._player_value(None), self._create_gallery_html([], 0), paths_json, -1, ""
            new_index = int(click_value)
            if 0 <= new_index < len(infos):
                selected_path = infos[new_index]["path"]
                if not os.path.exists(selected_path):
                    selected_path = None
                return self._player_value(selected_path), self._create_gallery_html(infos, new_index), paths_json, new_index, ""
        except Exception:
            pass
        return self._render_from_state(paths_json, current_selected)

    def _refresh_gallery(self, refresh_id, paths_json, selected_idx):
        rendered = self._render_from_state(paths_json, selected_idx)
        return rendered[0], rendered[1], rendered[3]

    @classmethod
    def _get_cached_creation_datetime(cls, path):
        try:
            stat = os.stat(path)
            cache_key = (path, int(stat.st_size), int(stat.st_mtime))
        except Exception:
            return None
        cached_dt = cls._timestamp_cache.get(cache_key)
        if cached_dt is not None:
            cls._timestamp_cache.move_to_end(cache_key)
            return cached_dt
        dt = cls._read_creation_datetime(path)
        if dt is None:
            try:
                if os.name == "nt":
                    ts = os.path.getctime(path)
                else:
                    stat = os.stat(path)
                    ts = stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime
                dt = datetime.fromtimestamp(ts)
            except Exception:
                return None
        cls._timestamp_cache[cache_key] = dt
        while len(cls._timestamp_cache) > cls._timestamp_cache_max_entries:
            cls._timestamp_cache.popitem(last=False)
        return dt

    @classmethod
    def _read_creation_datetime(cls, path):
        return None

    def _get_file_info(self, path, not_found=False):
        info = {"basename": Path(path).name, "path": path}
        if not_found:
            info.update(date="Deleted", time="00:00:00", timestamp="", deleted=True)
            return info
        dt = self._get_cached_creation_datetime(path) or datetime.fromtimestamp(os.path.getmtime(path))
        info.update(date=dt.strftime("%Y-%m-%d"), time=dt.strftime("%H:%M:%S"), timestamp=dt.timestamp(), deleted=False)
        info.update(self._extra_file_info(path))
        return info

    def _create_thumbnail_html(self, info, index, is_selected):
        selected_class = "selected" if is_selected else ""
        return f"""
        <div class="strip-thumbnail {self.source.id}-thumbnail {selected_class}" data-gallery="{self.source.id}" data-index="{index}" data-path="{html.escape(info['path'])}" title="{html.escape(info['basename'])}">
            {self._thumbnail_body(info)}
        </div>
        """

    def _create_gallery_html(self, infos, selected_index, offset=0):
        thumbnail_width = max(80, min(150, 100 - (len(infos) - 1) * 2)) if infos else 100
        thumbnails_html = "".join(self._create_thumbnail_html(info, i + offset, i == selected_index) for i, info in enumerate(infos))
        selected_basename = infos[selected_index]["basename"] if infos and 0 <= selected_index < len(infos) else ""
        return f"""
        <style>
            .strip-gallery-container {{
                --bg-primary: var(--studio-surface);
                --bg-secondary: var(--studio-input);
                --bg-selected-filename: var(--studio-soft);
                --bg-selected-thumbnail: var(--studio-button);
                --bg-tooltip: var(--body-text-color);
                --text-primary: var(--body-text-color);
                --text-secondary: var(--body-text-color-subdued);
                --text-tooltip: var(--background-fill-primary);
                --border-primary: var(--studio-border);
                --border-secondary: var(--studio-border);
                --accent-color: var(--studio-accent);
                --shadow-color: color-mix(in srgb, var(--studio-accent) 20%, transparent);
                --shadow-color-selected: color-mix(in srgb, var(--studio-accent) 25%, transparent);
                display: flex; flex-direction: column; overflow: hidden; margin-top: 0 !important; padding-top: 0 !important;
            }}
            .strip-player-wrapper {{ margin-bottom: 0 !important; padding-bottom: 0 !important; }}
            .audio-player-wrapper {{ min-height: 200px; max-height: 200px; overflow: hidden; }}
            .audio-player-wrapper .audio {{ height: 100% !important; margin-bottom: 0 !important; }}
            .strip-gallery-container .selected-filename {{
                padding: 4px 12px; background: var(--bg-selected-filename); border-radius: 4px; font-size: 14px; font-weight: 500;
                margin: 0 0 8px 0; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-primary);
            }}
            .strip-gallery-container .thumbnails-container {{
                display: flex; justify-content: center; gap: 8px; padding: 12px; overflow-x: auto; overflow-y: hidden; flex-direction: row;
                border: 1px solid var(--border-primary); border-radius: 8px; background: var(--bg-primary); min-height: 120px;
            }}
            .strip-thumbnail {{
                position: relative; min-width: {thumbnail_width}px; width: {thumbnail_width}px; flex-shrink: 0; padding: 12px;
                border: 2px solid var(--border-secondary); border-radius: 8px; cursor: pointer; background: var(--bg-secondary);
                transition: all 0.2s ease; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px;
            }}
            .strip-thumbnail:hover {{ border-color: var(--accent-color); box-shadow: 0 2px 8px var(--shadow-color); transform: translateY(-2px); }}
            .strip-thumbnail:hover::after {{
                content: attr(title); position: absolute; bottom: calc(100% + 5px); left: 50%; transform: translateX(-50%);
                background: var(--bg-tooltip); color: var(--text-tooltip); padding: 6px 10px; border-radius: 4px; font-size: 12px;
                white-space: nowrap; z-index: 1000; pointer-events: none; max-width: {thumbnail_width}px; overflow: hidden; text-overflow: ellipsis;
            }}
            .strip-thumbnail.selected {{ border-color: var(--accent-color); background: var(--bg-selected-thumbnail); box-shadow: 0 2px 12px var(--shadow-color-selected); }}
            .strip-thumbnail .thumbnail-date {{ font-size: 13px; font-weight: 600; color: var(--text-primary); white-space: nowrap; }}
            .strip-thumbnail .thumbnail-time {{ font-size: 12px; color: var(--text-secondary); white-space: nowrap; }}
            .strip-thumbnail .thumbnail-detail {{ font-size: 13px; font-weight: bold; color: var(--accent-color); margin-top: 4px; white-space: nowrap; }}
            .strip-thumbnail .thumbnail-image {{ width: 100%; aspect-ratio: 1; object-fit: contain; border-radius: 4px; }}
            .mesh-thumbnail {{ padding: 6px; }}
            .mesh-thumbnail .thumbnail-time {{ font-size: 11px; line-height: 1.2; text-align: center; white-space: nowrap; }}
            .mesh-thumbnail .thumbnail-detail {{ margin-top: 0; font-size: 10.5px; max-width: 100%; overflow: hidden; text-overflow: ellipsis; }}
            @keyframes wangp-mesh-ready {{ from {{ opacity: .99; }} to {{ opacity: 1; }} }}
            .mesh-viewer {{ display: block; position: relative; border-radius: 8px; background: var(--studio-input); }}
            wangp-mesh-viewer {{ animation: wangp-mesh-ready 1ms; }}
            .mesh-viewer-status {{ position: absolute; inset: auto 8px 8px; padding: 8px; border-radius: 6px; background: var(--background-fill-primary); color: var(--body-text-color); font-size: 14px; }}
            .mesh-viewer-status[hidden] {{ display: none; }}
            .mesh-viewer-status button {{ margin-left: 8px; text-decoration: underline; }}
            .mesh-viewer-fallback {{ width: 100%; height: 100%; object-fit: contain; }}
            .mesh-viewer:fullscreen {{ height: 100vh !important; width: 100vw; border-radius: 0; }}
            .mesh-viewer-actions {{ position: absolute; top: 8px; right: 8px; display: flex; gap: 6px; }}
            .mesh-viewer-actions > * {{
                display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 6px; cursor: pointer;
                border: 1px solid var(--studio-border); background: var(--studio-surface); color: var(--body-text-color); opacity: 0.85;
            }}
            .mesh-viewer-actions > *:hover {{ opacity: 1; border-color: var(--studio-accent); }}
            .mesh-viewer-actions > [hidden] {{ display: none; }}
            .mesh-viewer-actions button:disabled {{ cursor: progress; opacity: 0.5; }}
            .mesh-snapshot-popup {{
                position: absolute; top: 48px; right: 8px; z-index: 5; display: flex; flex-direction: column; gap: 8px; padding: 10px 12px;
                width: 300px; max-width: calc(100% - 16px); box-sizing: border-box; border-radius: 8px; border: 1px solid var(--studio-border);
                background: var(--studio-surface); color: var(--body-text-color); box-shadow: 0 4px 16px var(--shadow-color); font-size: 13px;
            }}
            .mesh-snapshot-popup .mesh-snapshot-title {{ font-weight: 600; }}
            .mesh-snapshot-popup select {{
                width: 100%; padding: 4px 6px; border-radius: 6px; border: 1px solid var(--studio-border); background: var(--studio-input); color: inherit; font-size: 13px;
            }}
            .mesh-snapshot-popup .mesh-snapshot-hint {{ font-size: 11px; opacity: 0.75; }}
            .mesh-snapshot-popup .mesh-snapshot-buttons {{ display: flex; gap: 6px; justify-content: flex-end; align-items: center; }}
            .mesh-snapshot-popup button {{
                padding: 4px 10px; border-radius: 6px; border: 1px solid var(--studio-border); background: var(--studio-surface); color: inherit; cursor: pointer;
                height: 28px; font-size: 13px; line-height: 18px;
            }}
            .mesh-snapshot-popup button:hover, .mesh-snapshot-popup button[data-snapshot="add"] {{ border-color: var(--studio-accent); }}
            .mesh-snapshot-popup button:disabled {{ cursor: progress; opacity: 0.5; }}
            .mesh-viewer-fallback {{ height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; padding: 12px; box-sizing: border-box; }}
            .mesh-viewer-fallback img {{ min-height: 0; max-width: 100%; flex: 1; object-fit: contain; }}
            .mesh-viewer-fallback div {{ font-size: 0.85em; opacity: 0.75; text-align: center; }}
        </style>
        <div class="strip-gallery-container" data-gallery="{self.source.id}">
            <div class="selected-filename">{html.escape(selected_basename)}</div>
            <div class="thumbnails-container">
                {thumbnails_html}
            </div>
        </div>
        """

    def _process_paths(self, paths):
        """File infos in the same order as the paths."""
        infos = []
        for path in paths or []:
            try:
                infos.append(self._get_file_info(path, not os.path.exists(path)))
            except Exception:
                continue
        return infos[: self.max_thumbnails]

    def _render_from_state(self, paths_json, selected_idx):
        value = json.loads(paths_json) if paths_json else []
        projected = isinstance(value, dict)
        paths, offset = (value['paths'], value['offset']) if projected else (value, 0)
        infos = self._process_paths(paths)
        local_index = selected_idx - offset
        if not projected:
            local_index = _get_selected_idx(infos, local_index)
            selected_idx = local_index
        selected_path = infos[local_index]['path'] if 0 <= local_index < len(infos) else None
        if selected_path and not os.path.exists(selected_path):
            selected_path = None
        return self._player_value(selected_path), self._create_gallery_html(infos, local_index, offset), paths_json, selected_idx, ''


class AudioGallery(StripGallery):
    source = AUDIO

    def _create_player(self, path):
        return gr.Audio(value=path, label=self.label, type="filepath")

    @classmethod
    def _read_creation_datetime(cls, path):
        try:
            from shared.utils.audio_metadata import read_audio_metadata, resolve_audio_creation_datetime
            return resolve_audio_creation_datetime(path, wangp_metadata=read_audio_metadata(path))
        except Exception:
            return None

    def _extra_file_info(self, path):
        return {"duration": self._get_audio_duration(path)}

    def _thumbnail_body(self, info):
        duration = "0:00" if info["deleted"] else info["duration"]
        return f'<div class="thumbnail-date">{info["date"]}</div><div class="thumbnail-time">{info["time"]}</div><div class="thumbnail-detail">{duration}</div>'

    def _get_audio_duration(self, audio_path):
        """Audio duration formatted as M:SS."""
        try:
            import contextlib
            import wave
            try:
                with contextlib.closing(wave.open(audio_path, "r")) as f:
                    return self._format_duration(f.getnframes() / float(f.getframerate()))
            except Exception:
                pass
            try:
                from mutagen import File
                audio = File(audio_path)
                if audio and getattr(audio, "info", None):
                    return self._format_duration(audio.info.length)
            except Exception:
                pass
            return self._format_duration(os.path.getsize(audio_path) / 32000)  # rough estimate from the file size
        except Exception:
            return "0:00"

    def _format_duration(self, seconds):
        mins, secs = divmod(int(seconds + 0.5), 60)
        return f"{mins}:{secs:02d}"


class MeshGallery(StripGallery):
    """3D assets (GLB meshes, PLY Gaussian splats): an interactive viewer above a strip of preview thumbnails (`<stem>.png` beside each asset)."""

    source = MESH
    viewer_height = 420

    @staticmethod
    def get_viewer_javascript(vendor_base):
        return MeshGallery._own_javascript.replace("'/gradio_api/file=shared/gradio/vendor/'", f"new URL({json.dumps(vendor_base)}, document.baseURI).href")

    def _create_player(self, path):
        return gr.HTML(value=self._player_value(path), elem_id=self.element_id("viewer"))

    def _player_value(self, path):
        if path is None:
            return f'<div class="mesh-viewer mesh-viewer-empty" style="height:{self.viewer_height}px"></div>'
        url, name = html.escape(gradio_file_url(path)), html.escape(Path(path).name)
        preview = self.preview_path(path)
        preview_url = html.escape(gradio_file_url(preview)) if os.path.isfile(preview) else ""
        if Path(path).suffix.lower() == ".ply":
            viewer, kind = f'<wangp-splat-viewer src="{url}" style="display:block;width:100%;height:100%"></wangp-splat-viewer>', "Gaussian Splats (PLY)"
        else:
            viewer, kind = f'<model-viewer src="{url}" alt="{name}" camera-controls touch-action="pan-y" interaction-prompt="none" shadow-intensity="0.6" environment-image="neutral" exposure="1" tone-mapping="neutral" style="width:100%;height:100%;background:transparent"></model-viewer>', "GLB"
        snapshot_icon = '<path d="M4 8h3l2-3h6l2 3h3v11H4V8Z"/><circle cx="12" cy="13" r="3.5"/>'
        download_icon = '<path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19h14"/>'
        fullscreen_icon = '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>'
        play_icon = '<path d="M8 5v14l11-7z"/>'
        icon = lambda drawing: f'<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{drawing}</svg>'
        return f"""<wangp-mesh-viewer class="mesh-viewer" data-path="{html.escape(path)}" data-preview="{preview_url}" style="height:{self.viewer_height}px">
{viewer}
<div class="mesh-viewer-actions" role="group" aria-label="3D Asset Actions">
<button type="button" data-mesh-action="play" title="Play the Animation" aria-label="Play the Animation" hidden>{icon(play_icon)}</button>
<button type="button" data-mesh-action="snapshot" title="Add a Snapshot of the Current View to the Image Gallery" aria-label="Add Snapshot to Image Gallery">{icon(snapshot_icon)}</button>
<a href="{url}" download="{name}" title="Download the 3D Asset ({kind})" aria-label="Download {name}">{icon(download_icon)}</a>
<button type="button" data-mesh-action="fullscreen" title="Full Screen (Esc to leave)" aria-label="Full Screen">{icon(fullscreen_icon)}</button>
</div>
</wangp-mesh-viewer>"""

    @staticmethod
    def preview_path(path):
        return os.path.splitext(path)[0] + ".png"

    def _extra_file_info(self, path):
        preview = self.preview_path(path)
        info = {"preview": preview if os.path.isfile(preview) else None}
        try:
            from shared.utils.mesh_outputs import asset_stats
            stats = asset_stats(path)
            count, unit = (stats["gaussians"], "splats") if stats["kind"] == "splats" else (stats["faces"], "faces")
            info["detail"] = f"{count / 1e6:.1f}M {unit}" if count >= 1e6 else f"{count / 1000:.0f}K {unit}" if count >= 1000 else f"{count} {unit}"
        except Exception:
            info["detail"] = f"{os.path.getsize(path) / 1048576:.1f} MB"
        return info

    def _thumbnail_body(self, info):
        if info["deleted"]:
            return f'<div class="thumbnail-date">{info["date"]}</div><div class="thumbnail-time">{info["time"]}</div>'
        image = f'<img class="thumbnail-image" src="{html.escape(gradio_file_url(info["preview"]))}" alt="" loading="lazy">' if info["preview"] else '<div class="thumbnail-detail">3D</div>'
        return f'{image}<div class="thumbnail-time">{info["date"]}<br>{info["time"]}</div><div class="thumbnail-detail">{info["detail"]}</div>'

    # The vendored viewers load the first time the viewer shows an asset (<model-viewer> for GLB, three.js + Spark for splats):
    # users without 3D outputs pay nothing. Both need WebGL; without it (e.g. browser hardware acceleration disabled) the preview is shown.
    _own_javascript = r"""
// ===== MeshGallery: lazy 3D viewers =====
window.__meshGalleryNS = window.__meshGalleryNS || {};
(function (MG) {
  if (MG.init) return;
  MG.init = true;
  const VENDOR = '/gradio_api/file=shared/gradio/vendor/';
  const SPARK = VENDOR + 'spark/';
  MG.visible = element => {
    const box = element.getBoundingClientRect();
    return !document.hidden && box.width > 0 && box.height > 0 && box.bottom > 0 && box.right > 0
      && box.top < window.innerHeight && box.left < window.innerWidth && getComputedStyle(element).visibility !== 'hidden';
  };
  MG.webgl = function () {
    if (MG.hasWebGL !== true) {
      const gl = document.createElement('canvas').getContext('webgl2');
      MG.hasWebGL = !!gl;
      gl?.getExtension('WEBGL_lose_context')?.loseContext();
    }
    return MG.hasWebGL;
  };
  MG.load = function () {
    if (customElements.get('model-viewer')) return Promise.resolve();
    const retry = MG.moduleAttempt ? '?retry=' + MG.moduleAttempt : '';
    return MG.loading ||= import(VENDOR + 'model-viewer.min.js' + retry).catch(error => {
      MG.loading = null; MG.moduleAttempt = (MG.moduleAttempt || 0) + 1; throw error;
    });
  };
  // Gaussian splats: one WebGL context while the asset is shown, released when the viewer shows another asset.
  class SplatViewer extends HTMLElement {
    async load() {
      const token = this.token = {};
      this.ready = this.inView = false;
      const [THREE, {SparkRenderer, SplatMesh}, {OrbitControls}] = await Promise.all([import(SPARK + 'three.module.js'), import(SPARK + 'spark.module.min.js'), import(SPARK + 'OrbitControls.js')]);
      if (this.token !== token) return;
      const renderer = new THREE.WebGLRenderer({antialias: false, alpha: true});
      renderer.domElement.addEventListener('webglcontextlost', () => {
        if (this.token === token) this.dispatchEvent(new Event('error'));
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.domElement.style.cssText = 'width:100%;height:100%;display:block';
      this.appendChild(renderer.domElement);
      const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000), controls = new OrbitControls(camera, renderer.domElement);
      // Redraw for interaction and Spark's asynchronous sorting, then leave static views idle.
      const requestRender = () => {
        if (this.renderFrame || this.token !== token || !this.ready || !this.inView || document.hidden) return;
        this.renderFrame = requestAnimationFrame(() => {
          this.renderFrame = 0;
          if (this.token !== token || !this.inView || document.hidden || !this.clientWidth || !this.clientHeight) return;
          renderer.render(scene, camera);
        });
      };
      const splat = new SplatMesh({url: this.getAttribute('src')});
      const spark = new SparkRenderer({renderer, onDirty: requestRender});
      scene.add(spark, splat);
      this.state = {renderer, scene, camera, controls, splat, spark, requestRender};
      controls.addEventListener('change', requestRender);
      const resize = () => {
        const width = this.clientWidth, height = this.clientHeight;
        if (!width || !height) return;
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        requestRender();
      };
      this.state.resize = resize;
      await splat.initialized;
      if (this.token !== token) return;
      // Same 3/4 framing as the preview: the whole asset in view, seen from the front right and slightly above.
      const box = splat.getBoundingBox(true), center = box.getCenter(new THREE.Vector3()), radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
      camera.position.copy(center).addScaledVector(new THREE.Vector3(0.55, 0.35, 0.75).normalize(), radius / Math.sin(THREE.MathUtils.degToRad(20)));
      camera.near = radius / 100;
      camera.far = radius * 100;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
      this.ready = true;
      this.refresh();
    }
    refresh() {
      this.inView = MG.visible(this);
      if (!this.inView) {
        cancelAnimationFrame(this.renderFrame);
        this.renderFrame = 0;
      } else if (this.ready) this.state.resize();
    }
    disconnectedCallback() {
      this.token = null;
      cancelAnimationFrame(this.renderFrame);
      this.renderFrame = 0;
      const state = this.state;
      if (!state) return;
      this.state = null;
      state.controls.removeEventListener('change', state.requestRender);
      state.controls.dispose();
      state.splat.dispose();
      state.spark.dispose();
      state.renderer.dispose();
      state.renderer.forceContextLoss();
      state.renderer.domElement.remove();
    }
    toBlob({mimeType = 'image/png', pixelRatio = window.devicePixelRatio} = {}) {
      const {renderer, scene, camera} = this.state;
      const previousRatio = renderer.getPixelRatio();
      renderer.setPixelRatio(pixelRatio);
      renderer.render(scene, camera);  // the canvas is read in the same task as the render
      const blob = new Promise(resolve => renderer.domElement.toBlob(resolve, mimeType));
      renderer.setPixelRatio(previousRatio);
      return blob;
    }
  }
  customElements.define('wangp-splat-viewer', SplatViewer);
  // Animated GLBs behave like a video: paused until Play, then Pause / Resume (the button only shows when the asset has an animation).
  const PLAY_ICON = '<path d="M8 5v14l11-7z"/>', PAUSE_ICON = '<path d="M7 5h4v14H7zM13 5h4v14h-4z"/>';
  const setPlayButton = (button, playing) => {
    button.querySelector('svg').innerHTML = playing ? PAUSE_ICON : PLAY_ICON;
    button.title = button.ariaLabel = playing ? 'Pause the Animation' : 'Play the Animation';
  };
  class MeshViewer extends HTMLElement {
    connectedCallback() {
      const mount = this.mount = {};
      // HTML components insert their children in the same task as the host.
      queueMicrotask(() => { if (this.isConnected && this.mount === mount) this.start(); });
    }
    disconnectedCallback() {
      this.mount = null;
      this.events?.abort();
      this.viewer?.pause?.();
    }
    async start() {
      this.events?.abort();
      this.events = new AbortController();
      const viewer = this.viewer = this.querySelector('model-viewer, wangp-splat-viewer');
      this.playing = false;
      const snapshot = this.querySelector('[data-mesh-action="snapshot"]');
      if (snapshot) snapshot.disabled = true;
      const status = this.status ||= this.appendChild(document.createElement('div'));
      status.className = 'mesh-viewer-status'; status.setAttribute('role', 'status');
      status.hidden = false; status.textContent = 'Loading 3D Asset…';
      const fail = message => {
        if (!this.isConnected || this.viewer !== viewer) return;
        const replacement = viewer.cloneNode(false);
        const preview = document.createElement('img'); preview.className = 'mesh-viewer-fallback'; preview.alt = '3D Asset Preview';
        if (this.dataset.preview) preview.src = this.dataset.preview;
        else preview.hidden = true;
        viewer.pause?.();
        viewer.replaceWith(preview);
        this.viewer = null;
        this.querySelector('[data-mesh-action="play"]').hidden = true;
        if (snapshot) snapshot.disabled = true;
        status.hidden = false; status.textContent = message;
        const retry = document.createElement('button'); retry.type = 'button'; retry.textContent = 'Retry 3D';
        retry.onclick = () => {preview.replaceWith(replacement); this.start();};
        status.append(retry);
      };
      const loaded = () => {
        if (!this.isConnected || this.viewer !== viewer) return;
        status.hidden = true;
        if (snapshot) snapshot.disabled = false;
        const play = this.querySelector('[data-mesh-action="play"]');
        play.hidden = !viewer.availableAnimations?.length;
        setPlayButton(play, false);
        this.refresh();
      };
      viewer.addEventListener('error', () => fail('Unable to Load This 3D Asset. You Can Still Download It.'), {once: true, signal: this.events.signal});
      viewer.addEventListener('load', loaded, {once: true, signal: this.events.signal});
      try {
        // A temporary context failure must not disable all later attempts.
        if (!MG.webgl()) {fail('Interactive 3D Is Unavailable in This Browser. You Can Still Download the Asset.'); return;}
        if (viewer.localName === 'model-viewer') {
          await MG.load();
          if (viewer.loaded) loaded();
        } else {
          await viewer.load();
          loaded();
        }
      } catch (error) {
        console.error('3D viewer', error);
        fail('Unable to Start the 3D Viewer. You Can Still Download the Asset.');
      }
    }
    refresh() {
      const viewer = this.viewer;
      if (!viewer) return;
      if (viewer.localName === 'wangp-splat-viewer') viewer.refresh();
      else if (typeof viewer.pause === 'function') {
        if (this.playing && MG.visible(this)) { if (viewer.paused) viewer.play(); }
        else if (!viewer.paused) viewer.pause();
      }
    }
  }
  customElements.define('wangp-mesh-viewer', MeshViewer);
  let layoutFrame = 0;
  MG.refresh = () => {
    if (layoutFrame) return;
    layoutFrame = requestAnimationFrame(() => {
      layoutFrame = 0;
      document.querySelectorAll('wangp-mesh-viewer').forEach(viewer => viewer.refresh());
    });
  };
  window.addEventListener('resize', MG.refresh);
  document.addEventListener('scroll', MG.refresh, {capture: true, passive: true});
  document.addEventListener('fullscreenchange', MG.refresh);
  document.addEventListener('visibilitychange', () => document.querySelectorAll('wangp-mesh-viewer').forEach(viewer => viewer.refresh()));
  document.addEventListener('animationstart', event => {if (event.animationName === 'wangp-mesh-ready') MG.refresh();});
  document.addEventListener('click', MG.refresh);
  document.addEventListener('click', event => {
    const button = event.target.closest('.mesh-viewer [data-mesh-action="play"]');
    if (!button) return;
    const container = button.closest('wangp-mesh-viewer');
    container.playing = !container.playing;
    container.refresh();
    setPlayButton(button, container.playing);
  });
  // Full screen: the viewer of the selected asset (both viewers follow their container's size)
  document.addEventListener('click', event => {
    const button = event.target.closest('.mesh-viewer [data-mesh-action="fullscreen"]');
    if (!button) return;
    const container = button.closest('.mesh-viewer');
    if (document.fullscreenElement === container) document.exitFullscreen();
    else container.requestFullscreen();
  });
  // Snapshot: the current view of the 3D viewer becomes an image of the image gallery. A popup sets its size: the pixel budget of a
  // resolution (by default the one selected in the settings) with the shape of the viewer, the view being rendered again at that size.
  const SNAPSHOT_SIDES = [512, 768, 1024, 1440, 2048, 2880, 4096], SNAPSHOT_MAX_SIDE = 8192;
  const snapshotRatio = (viewer, budget) => {
    const width = viewer.clientWidth || 1, height = viewer.clientHeight || 1;
    return budget ? Math.min(Math.sqrt(budget / (width * height)), SNAPSHOT_MAX_SIDE / Math.max(width, height)) : window.devicePixelRatio;
  };
  // model-viewer renders at the page's pixel ratio, read at each frame: raised while it renders the snapshot (minimum scale 1, no throttling)
  MG.renderSnapshot = async function (viewer, pixelRatio) {
    if (viewer.localName === 'wangp-splat-viewer') return viewer.toBlob({mimeType: 'image/png', pixelRatio});
    const ModelViewer = customElements.get('model-viewer'), minimum = ModelViewer.minimumRenderScale;
    const own = Object.getOwnPropertyDescriptor(window, 'devicePixelRatio');
    Object.defineProperty(window, 'devicePixelRatio', {configurable: true, get: () => pixelRatio});
    ModelViewer.minimumRenderScale = 1;
    try {
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      return await viewer.toBlob({mimeType: 'image/png'});
    } finally {
      if (own) Object.defineProperty(window, 'devicePixelRatio', own);
      else delete window.devicePixelRatio;
      ModelViewer.minimumRenderScale = minimum;
    }
  };
  const addSnapshot = async (container, pixelRatio) => {
    const viewer = container.querySelector('model-viewer, wangp-splat-viewer');
    const request = document.querySelector('#wangp-gallery-add-snapshot textarea'), view = document.querySelector('#wangp-gallery-view textarea');
    if (!viewer || typeof viewer.toBlob !== 'function' || !request || !view?.value) return;
    const blob = await MG.renderSnapshot(viewer, pixelRatio);
    const image = await new Promise((resolve, reject) => {const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(blob);});
    const payload = {workspace: JSON.parse(view.value).workspace, path: container.dataset.path, image, id: Date.now()};
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(request, JSON.stringify(payload));
    request.dispatchEvent(new Event('input', {bubbles: true}));
  };
  MG.openSnapshotPopup = function (container) {
    const viewer = container.querySelector('model-viewer, wangp-splat-viewer');
    if (!viewer) return;
    const choices = [], settings = document.querySelector('#wangp-resolution input')?.value.match(/(\d+)\s*x\s*(\d+)/);
    if (settings) choices.push([`Settings: ${settings[1]}x${settings[2]}`, settings[1] * settings[2]]);
    choices.push(['Viewer', 0], ...SNAPSHOT_SIDES.map(side => [`as ${side}x${side}`, side * side]));
    const popup = document.createElement('div');
    popup.className = 'mesh-snapshot-popup';
    popup.innerHTML = '<div class="mesh-snapshot-title">Snapshot Size</div><select aria-label="Snapshot Size"></select>'
      + '<div class="mesh-snapshot-hint">As many pixels as the resolution in brackets, with the shape of the viewer.</div>'
      + '<div class="mesh-snapshot-buttons"><button type="button" data-snapshot="add">Add to Image Gallery</button><button type="button" data-snapshot="cancel">Cancel</button></div>';
    const select = popup.querySelector('select');
    for (const [label, budget] of choices) {
      const ratio = snapshotRatio(viewer, budget);
      select.add(new Option(`${Math.ceil(viewer.clientWidth * ratio)} x ${Math.ceil(viewer.clientHeight * ratio)} (${label})`, ratio));
    }
    popup.addEventListener('click', async event => {
      const action = event.target.closest('[data-snapshot]')?.dataset.snapshot;
      if (action === 'cancel') popup.remove();
      if (action !== 'add') return;
      popup.querySelectorAll('button, select').forEach(element => { element.disabled = true; });
      try { await addSnapshot(container, Number(select.value)); } finally { popup.remove(); }
    });
    popup.addEventListener('keydown', event => { if (event.key === 'Escape') popup.remove(); });
    container.appendChild(popup);
    select.focus();
  };
  document.addEventListener('click', event => {
    const button = event.target.closest('.mesh-viewer [data-mesh-action="snapshot"]');
    if (!button) return;
    const container = button.closest('.mesh-viewer'), popup = container.querySelector('.mesh-snapshot-popup');
    if (popup) popup.remove();
    else MG.openSnapshotPopup(container);
  });
})(window.__meshGalleryNS);
// ===== end MeshGallery JS =====
"""


STRIP_GALLERY_CLASSES = {gallery.source.id: gallery for gallery in (AudioGallery, MeshGallery)}
