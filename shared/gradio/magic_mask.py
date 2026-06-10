import html
import queue
import threading
import uuid
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import gradio as gr

from shared import magic_mask


MAGIC_WAND_LABEL = "\U0001FA84"
MAX_MAGIC_MASK_OBJECTS = 5
_ABORT_EVENTS: dict[str, threading.Event] = {}
_ORIGINAL_IMAGE_EDITOR = None


class MagicMaskAbort(Exception):
    pass


def magic_mask_button_updates(image_mode, video_prompt_type):
    mask_visible = "V" in video_prompt_type and "A" in video_prompt_type and "U" not in video_prompt_type
    image_outputs = image_mode > 0
    return gr.update(visible=mask_visible and image_outputs), gr.update(visible=mask_visible and not image_outputs)


def _status_html(message, tone="info"):
    if not message:
        return ""
    tone_class = "is-error" if tone == "error" else ""
    return f"<div class='wangp-magic-mask-message {tone_class}'>{html.escape(str(message))}</div>"


def _progress_html(message, percent=0):
    if not message:
        return ""
    percent = max(0, min(100, int(percent)))
    return (
        "<div class='wangp-magic-mask-progress'>"
        f"<div class='wangp-magic-mask-progress-label'>{html.escape(str(message))}</div>"
        "<div class='wangp-magic-mask-progress-track'>"
        f"<div class='wangp-magic-mask-progress-bar' style='width:{percent}%;'></div>"
        "</div></div>"
    )


def _abort_event(token):
    token = str(token or "")
    if len(token) == 0:
        token = "default"
    if token not in _ABORT_EVENTS:
        _ABORT_EVENTS[token] = threading.Event()
    return _ABORT_EVENTS[token]


def _open_panel():
    return gr.update(visible=True), "", "", gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), None, None


def _close_panel():
    return (
        gr.update(visible=False),
        "",
        "",
        gr.update(visible=False, interactive=False),
        None,
        None,
    )


def _magic_mask_object_colors(model_def):
    colors = model_def.get("magic_mask_object_colors", []) if isinstance(model_def, dict) else []
    return colors if isinstance(colors, (list, tuple)) and len(colors) > 0 else []


def _max_objects_limit(value, object_count=None):
    if value in (None, "", "all"):
        return None
    limit = MAX_MAGIC_MASK_OBJECTS if object_count is None else min(int(object_count), MAX_MAGIC_MASK_OBJECTS)
    return min(max(1, int(value)), limit)


def _max_object_choices():
    return [(str(index), index) for index in range(1, MAX_MAGIC_MASK_OBJECTS + 1)] + [("All", "all")]


def _max_time_seconds(value):
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    seconds = float(text)
    if seconds <= 0:
        raise ValueError("Max Time must be greater than 0 seconds.")
    return seconds


def _abort_magic_mask(abort_token):
    _abort_event(abort_token).set()
    return _status_html("Aborting Magic Mask..."), gr.update(visible=True, interactive=False)


def _exit_button_running():
    return gr.update(visible=False, interactive=False)


def _exit_button_idle():
    return gr.update(visible=True, interactive=True)


def _abort_button_running():
    return gr.update(visible=True, interactive=True)


def _abort_button_idle():
    return gr.update(visible=False, interactive=False)


def _raise_if_aborted(abort_event):
    if abort_event.is_set():
        raise MagicMaskAbort()


def _image_source(image_mask_guide, image_guide):
    if isinstance(image_mask_guide, dict) and image_mask_guide.get("background") is not None:
        return image_mask_guide["background"]
    return image_guide


def _keywords_processed_html(processed, total):
    return _status_html(f"Masks generated: {processed}/{total}")


def _mask_progress_html(keyword, done, total):
    percent = int(done * 100 / max(int(total), 1))
    return _progress_html(f'Generating Mask "{keyword}"', percent)


def _current_keyword_progress(keywords, done, total):
    keyword_count = max(len(keywords), 1)
    total_steps = max(float(total), 1.0)
    done = max(0.0, min(float(done), total_steps))
    current_index = min(keyword_count - 1, int(done * keyword_count / total_steps))
    keyword_steps = total_steps / keyword_count
    keyword_done = max(0.0, min(keyword_steps, done - current_index * keyword_steps))
    return keywords[current_index], keyword_done, keyword_steps


def _run_keyword_mask(video, keywords, abort_event, colorize_objects=False, color_palette=None, max_colored_objects=None):
    progress_events = queue.Queue()

    def progress_callback(done, total):
        if abort_event.is_set():
            raise MagicMaskAbort()
        progress_events.put(("progress", int(done), int(total)))

    def worker():
        try:
            if abort_event.is_set():
                raise MagicMaskAbort()
            progress_events.put(("done", magic_mask.generate_keyword_masks(video, keywords, progress_callback=progress_callback, colorize_objects=colorize_objects, color_palette=color_palette, max_colored_objects=max_colored_objects)))
        except MagicMaskAbort as exc:
            progress_events.put(("abort", exc))
        except Exception as exc:
            progress_events.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    pending_event = None
    while True:
        event = pending_event or progress_events.get()
        pending_event = None
        if event[0] == "progress":
            latest_event = event
            while True:
                try:
                    next_event = progress_events.get_nowait()
                except queue.Empty:
                    break
                if next_event[0] == "progress":
                    latest_event = next_event
                else:
                    pending_event = next_event
                    break
            event = latest_event
            yield event[1], event[2]
        elif event[0] == "done":
            thread.join()
            return event[1]
        elif event[0] == "abort":
            thread.join()
            raise event[1]
        else:
            thread.join()
            raise event[1]


def _generate_magic_mask(
    state,
    keywords_text,
    negative_mask,
    image_mode,
    video_guide,
    image_mask_guide,
    image_guide,
    abort_token,
    max_objects,
    max_time,
    *,
    download_assets: Callable[[dict[str, Any]], Any],
    acquire_gpu: Callable[[Any, str, str], Any],
    release_gpu: Callable[[Any, str], Any],
    get_model_settings: Callable[[Any], dict],
    get_model_def: Callable[[Any], dict] | None = None,
):
    source_image = None
    if image_mode > 0:
        source_image = _image_source(image_mask_guide, image_guide)
        if source_image is None:
            yield gr.update(), gr.update(), gr.update(), _status_html("Magic Mask needs a control image.", "error"), gr.update(visible=True), "", _exit_button_idle(), _abort_button_idle(), None, None
            return
    elif video_guide is None:
        yield gr.update(), gr.update(), gr.update(), _status_html("Magic Mask needs a control video.", "error"), gr.update(visible=True), "", _exit_button_idle(), _abort_button_idle(), None, None
        return

    keywords = magic_mask.parse_keywords(keywords_text)
    if len(keywords) == 0:
        yield gr.update(), gr.update(), gr.update(), _status_html("Enter at least one keyword.", "error"), gr.update(visible=True), "", _exit_button_idle(), _abort_button_idle(), None, None
        return
    try:
        max_time = _max_time_seconds(max_time) if image_mode <= 0 else None
    except ValueError as exc:
        yield gr.update(), gr.update(), gr.update(), _status_html(exc, "error"), gr.update(visible=True), "", _exit_button_idle(), _abort_button_idle(), None, None
        return
    keywords_label = ", ".join(keywords)
    abort_event = _abort_event(abort_token)
    abort_event.clear()
    acquired = False
    try:
        yield gr.update(), gr.update(), gr.update(), _status_html("Initializing Magic Mask"), gr.update(visible=True), _progress_html("Preparing files", 0), _exit_button_running(), _abort_button_running(), None, None
        download_assets(magic_mask.query_download_def())
        _raise_if_aborted(abort_event)
        yield gr.update(), gr.update(), gr.update(), _status_html("Initializing Magic Mask"), gr.update(visible=True), _progress_html("Initializing", 0), _exit_button_running(), _abort_button_running(), None, None
        acquire_gpu(state, magic_mask.PROCESS_ID, magic_mask.PROCESS_NAME)
        _raise_if_aborted(abort_event)
        acquired = True
        ui_settings = get_model_settings(state)
        model_def = get_model_def(state) if callable(get_model_def) else {}
        object_colors = _magic_mask_object_colors(model_def)
        colorize_objects = len(object_colors) > 0
        max_colored_objects = _max_objects_limit(max_objects, len(object_colors) if colorize_objects else None)
        if image_mode > 0:
            _raise_if_aborted(abort_event)
            background, video = magic_mask.prepare_image_mask_input(source_image)
            total = len(keywords)
            mask_generator = _run_keyword_mask(video, keywords, abort_event, colorize_objects=colorize_objects, color_palette=object_colors, max_colored_objects=max_colored_objects)
            progress_started = False
            try:
                while True:
                    done, frame_total = next(mask_generator)
                    if done <= 0:
                        continue
                    progress_started = True
                    processed = min(total, int(done * total / max(frame_total, 1)))
                    current_keyword, keyword_done, keyword_total = _current_keyword_progress(keywords, done, frame_total)
                    yield gr.update(), gr.update(), gr.update(), _keywords_processed_html(processed, total), gr.update(visible=True), _mask_progress_html(current_keyword, keyword_done, keyword_total), _exit_button_running(), _abort_button_running(), None, None
            except StopIteration as stop:
                merged_mask = stop.value[0]
            if not progress_started:
                yield gr.update(), gr.update(), gr.update(), _keywords_processed_html(0, total), gr.update(visible=True), _mask_progress_html(keywords[0], 1, 1), _exit_button_running(), _abort_button_running(), None, None
            yield gr.update(), gr.update(), gr.update(), _keywords_processed_html(total, total), gr.update(visible=True), _mask_progress_html(keywords[-1], 1, 1), _exit_button_running(), _abort_button_running(), None, None
            yield gr.update(), gr.update(), gr.update(), _status_html("Saving Image Mask..."), gr.update(visible=True), "", _exit_button_running(), _abort_button_running(), None, None
            _raise_if_aborted(abort_event)
            mask_image = magic_mask.mask_to_image(magic_mask.finalize_masks(merged_mask, negative_mask=negative_mask))
            image_mask_guide_value = magic_mask.build_image_editor_value(background, mask_image)
            if isinstance(ui_settings, dict):
                ui_settings["image_guide"] = background
                ui_settings["image_mask"] = mask_image
            gr.Info(f"Magic Mask generated {'a negative ' if negative_mask else 'an '}image mask for: {keywords_label}.")
            yield gr.update(value=image_mask_guide_value), gr.update(value=mask_image), gr.update(), "", gr.update(visible=False), "", _exit_button_idle(), _abort_button_idle(), None, None
            return
        _raise_if_aborted(abort_event)
        video_path, video, fps = magic_mask.prepare_video_mask_input(video_guide, max_time_seconds=max_time)
        total = len(keywords)
        mask_generator = _run_keyword_mask(video, keywords, abort_event, colorize_objects=colorize_objects, color_palette=object_colors, max_colored_objects=max_colored_objects)
        progress_started = False
        try:
            while True:
                done, frame_total = next(mask_generator)
                if done <= 0:
                    continue
                progress_started = True
                processed = min(total, int(done * total / max(frame_total, 1)))
                current_keyword, keyword_done, keyword_total = _current_keyword_progress(keywords, done, frame_total)
                yield gr.update(), gr.update(), gr.update(), _keywords_processed_html(processed, total), gr.update(visible=True), _mask_progress_html(current_keyword, keyword_done, keyword_total), _exit_button_running(), _abort_button_running(), None, None
        except StopIteration as stop:
            merged_mask = stop.value
        if not progress_started:
            yield gr.update(), gr.update(), gr.update(), _keywords_processed_html(0, total), gr.update(visible=True), _mask_progress_html(keywords[0], 1, 1), _exit_button_running(), _abort_button_running(), None, None
        yield gr.update(), gr.update(), gr.update(), _keywords_processed_html(total, total), gr.update(visible=True), _mask_progress_html(keywords[-1], 1, 1), _exit_button_running(), _abort_button_running(), None, None
        yield gr.update(), gr.update(), gr.update(), _status_html("Saving Video Mask..."), gr.update(visible=True), "", _exit_button_running(), _abort_button_running(), None, None
        background_color = model_def.get("video_mask_background_color", None)
        mask_path = magic_mask.save_mask_video(video_path, magic_mask.finalize_masks(merged_mask, negative_mask=negative_mask), fps, keywords, abort_callback=lambda: _raise_if_aborted(abort_event), background_color=background_color)
        if isinstance(ui_settings, dict):
            ui_settings["video_mask"] = mask_path
        gr.Info(f"Magic Mask generated {'a negative ' if negative_mask else 'a '}video mask for: {keywords_label}.")
        yield gr.update(), gr.update(), gr.update(value=mask_path), "", gr.update(visible=False), "", _exit_button_idle(), _abort_button_idle(), None, None
    except MagicMaskAbort:
        yield gr.update(), gr.update(), gr.update(), _status_html("Magic Mask aborted."), gr.update(visible=True), "", _exit_button_idle(), _abort_button_idle(), None, None
    except Exception as exc:
        yield gr.update(), gr.update(), gr.update(), _status_html(exc, "error"), gr.update(visible=True), "", _exit_button_idle(), _abort_button_idle(), None, None
    finally:
        if acquired:
            release_gpu(state, magic_mask.PROCESS_ID)


@dataclass
class MagicMaskUI:
    trigger: gr.Button | None = None
    panel: gr.Group | None = None
    keywords: gr.Textbox | None = None
    negative_mask: gr.Checkbox | None = None
    status: gr.HTML | None = None
    progress_html: gr.HTML | None = None
    cancel_btn: gr.Button | None = None
    close_btn: gr.Button | None = None
    abort_btn: gr.Button | None = None
    generate_btn: gr.Button | None = None
    max_objects: gr.Dropdown | None = None
    max_time: gr.Textbox | None = None
    abort_token: gr.State | None = None
    pending_image_mask_guide: gr.State | None = None
    pending_image_mask: gr.State | None = None

    @staticmethod
    def hidden_trigger():
        return gr.Button(MAGIC_WAND_LABEL, size="sm", min_width=1, visible=False, elem_classes=["wangp-magic-mask-trigger", "wangp-magic-mask-trigger--hidden"])

    @staticmethod
    def button_updates(image_mode, video_prompt_type):
        return magic_mask_button_updates(image_mode, video_prompt_type)

    @staticmethod
    def patch_image_editor():
        global _ORIGINAL_IMAGE_EDITOR
        if _ORIGINAL_IMAGE_EDITOR is not None:
            return True
        original = gr.ImageEditor
        original_init = original.__init__
        if getattr(original_init, "__wangp_magic_mask_patch__", False):
            _ORIGINAL_IMAGE_EDITOR = original_init
            return True

        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._wangp_magic_mask_patch_enabled = True

        patched_init.__wangp_magic_mask_patch__ = True
        _ORIGINAL_IMAGE_EDITOR = original_init
        original.__init__ = patched_init
        return True

    @staticmethod
    def get_css():
        return r"""
.wangp-magic-mask-anchor {
    position: relative;
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

.wangp-magic-mask-anchor--image-editor {
    position: relative;
}

.wangp-magic-mask-anchor > .form,
.wangp-magic-mask-anchor > .styler {
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

.wangp-magic-mask-anchor:not(:has(> .block:not(.hide):not(.hidden), > button:not(.hide):not(.hidden), > .gr-group:not(.hide):not(.hidden))) {
    display: none !important;
}

.wangp-magic-mask-trigger,
.wangp-magic-mask-trigger button {
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    height: 34px;
    min-height: 34px;
    padding: 0 !important;
    border: 1px solid var(--button-secondary-border-color, rgba(17, 84, 118, 0.14)) !important;
    border-radius: 12px !important;
    background: var(--button-secondary-background-fill, linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(236, 244, 249, 0.99) 100%)) !important;
    color: var(--button-secondary-text-color, #155574) !important;
    box-shadow: var(--shadow-drop, 0 10px 18px rgba(11, 44, 63, 0.08)) !important;
    font-weight: 700;
    line-height: 1;
}

.wangp-magic-mask-trigger--overlay {
    position: absolute !important;
    top: 28px;
    right: 8px;
    z-index: 35;
}

.wangp-magic-mask-trigger--editor {
    display: none !important;
}

.wangp-magic-mask-toolbar-button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex: 0 0 auto !important;
    margin: var(--spacing-xxs, 2px) !important;
    width: 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
    height: 28px !important;
    min-height: 28px !important;
    padding: 0 !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-xs, 4px) !important;
    background: transparent !important;
    color: var(--block-label-text-color) !important;
    box-shadow: none !important;
    font-size: 16px !important;
    line-height: 1 !important;
}

.wangp-magic-mask-toolbar-button:hover {
    cursor: pointer !important;
    background: var(--background-fill-secondary) !important;
    color: var(--color-accent) !important;
    transform: none !important;
}

.wangp-magic-mask-toolbar-button.wangp-magic-mask-unavailable {
    cursor: not-allowed !important;
    opacity: 0.52;
    filter: grayscale(1);
}

.wangp-magic-mask-toolbar-button.wangp-magic-mask-unavailable:hover {
    cursor: not-allowed !important;
    color: var(--body-text-color-subdued) !important;
}

.wangp-magic-mask-toolbar-button[hidden] {
    display: none !important;
}

.wangp-magic-mask-synthetic-toolbar {
    position: absolute !important;
    top: var(--block-label-margin, 8px) !important;
    right: var(--block-label-margin, 8px) !important;
    z-index: var(--layer-3, 1000) !important;
}

.wangp-magic-mask-trigger:hover,
.wangp-magic-mask-trigger button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 24px rgba(11, 44, 63, 0.12);
}

.wangp-magic-mask-trigger:hover::after {
    content: "Magic Mask";
    position: absolute;
    top: 40px;
    right: 0;
    width: max-content;
    max-width: 160px;
    padding: 5px 7px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.82);
    color: #ffffff;
    font-size: calc(12px * var(--wangp-ui-scale));
    font-weight: 600;
    line-height: 1.2;
    pointer-events: none;
}

.wangp-magic-mask-panel.hide {
    display: none !important;
}

.wangp-magic-mask-panel.wangp-model-info-popup:not(.hide) {
    display: block !important;
    position: fixed !important;
    left: 12px !important;
    top: 96px !important;
    right: auto !important;
    width: min(680px, calc(100vw - 24px)) !important;
    height: min(496px, calc(100vh - 24px)) !important;
    max-height: calc(100vh - 24px) !important;
    z-index: 2147483000 !important;
    padding: 0 !important;
    overflow: visible !important;
    pointer-events: auto !important;
    resize: none !important;
}

.wangp-magic-mask-panel,
.wangp-magic-mask-panel *,
.wangp-magic-mask-card,
.wangp-magic-mask-card * {
    box-sizing: border-box;
}

.wangp-magic-mask-panel > .form,
.wangp-magic-mask-panel > .styler,
.wangp-magic-mask-card {
    resize: none !important;
}

.wangp-magic-mask-panel::-webkit-resizer {
    display: none !important;
}

.wangp-magic-mask-panel > .form,
.wangp-magic-mask-panel > .styler {
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    align-items: stretch !important;
    justify-content: stretch !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

.wangp-magic-mask-card.wangp-model-info-card {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    min-width: 0 !important;
    height: 100% !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
    background: var(--block-background-fill, var(--background-fill-primary, #ffffff)) !important;
}

.wangp-magic-mask-card > .form {
    display: flex !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

.wangp-magic-mask-card .html-container,
.wangp-magic-mask-card .prose {
    max-width: 100% !important;
    overflow: hidden !important;
}

.wangp-magic-mask-titlebar {
    flex: 0 0 auto;
    margin: 0 !important;
}

.wangp-magic-mask-titlebar > .form {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 14px !important;
    width: 100% !important;
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.wangp-magic-mask-heading,
.wangp-magic-mask-heading > .html-container {
    min-height: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.wangp-magic-mask-close {
    flex: 0 0 26px !important;
}

.wangp-magic-mask-close button,
.wangp-magic-mask-close {
    width: 26px !important;
    height: 26px !important;
    min-width: 26px !important;
    min-height: 26px !important;
    padding: 0 !important;
}

.wangp-magic-mask-body {
    flex: 0 0 auto !important;
    padding: 14px 18px 0;
}

.wangp-magic-mask-body > .form {
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.wangp-magic-mask-intro {
    margin: 0 0 12px;
    color: var(--body-text-color, #164f70);
    font-size: calc(0.88rem * var(--wangp-ui-scale));
    line-height: 1.45;
}

.wangp-magic-mask-keyword-row {
    align-items: center;
}

.wangp-magic-mask-keyword-row > .form {
    display: flex !important;
    align-items: center;
    gap: 10px !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

.wangp-magic-mask-options-row {
    align-items: center;
    margin-top: 8px;
}

.wangp-magic-mask-options-row > .form {
    display: flex !important;
    align-items: center;
    gap: 10px !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

.wangp-magic-mask-options-row > .form > .block {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.wangp-magic-mask-keywords {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.wangp-magic-mask-max-objects,
.wangp-magic-mask-max-time,
.wangp-magic-mask-max-objects > .form,
.wangp-magic-mask-max-time > .form,
.wangp-magic-mask-max-objects > .styler,
.wangp-magic-mask-max-time > .styler {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.wangp-magic-mask-max-objects label,
.wangp-magic-mask-max-time label {
    background: transparent !important;
    box-shadow: none !important;
}

.wangp-magic-mask-max-time textarea {
    resize: none !important;
}

.wangp-magic-mask-keywords textarea {
    min-height: 38px !important;
    height: 38px !important;
    overflow-y: hidden !important;
    resize: none !important;
}

.wangp-magic-mask-negative {
    flex: 0 0 150px !important;
    min-width: 150px !important;
}

.wangp-magic-mask-negative,
.wangp-magic-mask-negative > .form,
.wangp-magic-mask-negative > .styler {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.wangp-magic-mask-negative .block,
.wangp-magic-mask-negative .form,
.wangp-magic-mask-negative .styler,
.wangp-magic-mask-negative label {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.wangp-magic-mask-negative label {
    white-space: nowrap !important;
}

.wangp-magic-mask-message {
    flex: 0 0 auto !important;
    margin: 12px 18px 0;
    color: var(--body-text-color, #164f70);
    font-size: calc(0.9rem * var(--wangp-ui-scale));
    line-height: 1.5;
    font-weight: 600;
}

.wangp-magic-mask-message.is-error {
    color: var(--error-text-color, #b33434);
}

.wangp-magic-mask-progress {
    flex: 0 0 auto !important;
    margin: 12px 18px 0;
}

.wangp-magic-mask-progress-label {
    margin-bottom: 6px;
    color: var(--body-text-color, #164f70);
    font-size: calc(0.82rem * var(--wangp-ui-scale));
    font-weight: 700;
}

.wangp-magic-mask-progress-track {
    width: 100%;
    height: 8px;
    overflow: hidden;
    border-radius: 999px;
    background: var(--background-fill-secondary, rgba(19, 91, 126, 0.14));
}

.wangp-magic-mask-progress-bar {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2d89b7 0%, #56b18e 100%);
    transition: width 0.22s ease;
}

.wangp-magic-mask-spacer {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    overflow: hidden !important;
}

.wangp-magic-mask-actions {
    flex: 0 0 auto !important;
    justify-content: flex-end;
    align-items: flex-end;
    gap: 10px;
    margin-top: auto !important;
    padding: 18px;
    overflow: hidden !important;
}

.wangp-magic-mask-actions > .form {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: flex-end !important;
    gap: 10px !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

.wangp-magic-mask-btn,
.wangp-magic-mask-btn button {
    min-width: 92px;
    height: 40px;
    min-height: 40px;
    border-radius: 14px;
    border: 1px solid var(--button-secondary-border-color, rgba(17, 84, 118, 0.14));
    background: var(--button-secondary-background-fill, linear-gradient(180deg, rgba(255, 255, 255, 0.99) 0%, rgba(237, 245, 250, 0.99) 100%));
    color: var(--button-secondary-text-color, #155574);
    box-shadow: var(--shadow-drop, 0 10px 18px rgba(11, 44, 63, 0.08));
    font-weight: 700;
}

.wangp-magic-mask-btn--primary,
.wangp-magic-mask-btn--primary button {
    color: var(--button-primary-text-color, #f4fbff);
    border-color: var(--button-primary-border-color, rgba(10, 59, 84, 0.12));
    background: var(--button-primary-background-fill, linear-gradient(180deg, rgba(16, 86, 121, 0.98) 0%, rgba(10, 59, 84, 0.98) 100%));
}

.wangp-magic-mask-btn--danger,
.wangp-magic-mask-btn--danger button {
    color: #ffffff;
    border-color: rgba(142, 45, 45, 0.16);
    background: linear-gradient(180deg, rgba(188, 67, 67, 0.98) 0%, rgba(132, 41, 41, 0.98) 100%);
}

.wangp-magic-mask-btn:disabled,
.wangp-magic-mask-btn button:disabled {
    cursor: not-allowed !important;
    filter: grayscale(0.9);
    opacity: 0.48;
}

@media (prefers-color-scheme: dark) {
    .wangp-magic-mask-message.is-error {
        color: #ff9e9e;
    }
}
"""

    @staticmethod
    def get_javascript():
        return r"""
window.__wangpMagicMaskNS = window.__wangpMagicMaskNS || {};
const WMM = window.__wangpMagicMaskNS;
WMM.init = WMM.init || false;
WMM.observer = WMM.observer || null;
WMM.raf = WMM.raf || null;
WMM.interval = WMM.interval || null;

WMM.isVisible = function (element) {
    if (!element) return false;
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return rect.width > 2 && rect.height > 2 && style.display !== 'none' && style.visibility !== 'hidden';
};

WMM.findImageEditorToolbar = function (editor) {
    const synthetic = editor.querySelector('.wangp-magic-mask-synthetic-toolbar');
    const topControls = editor.querySelector('.icon-button-wrapper.top-panel:not(.wangp-magic-mask-synthetic-toolbar), .icon-button-wrapper:not(.wangp-magic-mask-synthetic-toolbar)');
    if (topControls) {
        if (synthetic) synthetic.remove();
        return topControls;
    }

    const imageToolbars = Array.from(editor.querySelectorAll('.toolbar-wrap'));
    const primaryToolbar = imageToolbars.find((toolbar) => !toolbar.closest('.toolbar-wrap-wrap')) || imageToolbars[0];
    if (primaryToolbar) {
        if (synthetic) synthetic.remove();
        return primaryToolbar;
    }

    const explicit = editor.querySelector('[role="toolbar"], [class*="toolbar"], [class*="Toolbar"], [class*="tools"], [class*="Tools"]');
    if (explicit && !explicit.classList.contains('wangp-magic-mask-synthetic-toolbar')) {
        if (synthetic) synthetic.remove();
        return explicit;
    }

    const editorRect = editor.getBoundingClientRect();
    let best = null;
    let bestScore = -Infinity;
    const candidates = new Map();
    editor.querySelectorAll('button').forEach((button) => {
        let node = button.parentElement;
        for (let depth = 0; node && node !== editor && depth < 5; depth += 1, node = node.parentElement) {
            candidates.set(node, node.querySelectorAll('button').length);
        }
    });
    candidates.forEach((buttonCount, node) => {
        const rect = node.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0 || rect.top > editorRect.top + 100) return;
        const topDistance = Math.abs(rect.top - editorRect.top);
        const rightBias = rect.left > editorRect.left + editorRect.width * 0.45 ? 16 : 0;
        const heightPenalty = Math.max(0, rect.height - 52) * 2;
        const score = buttonCount * 24 + rightBias - topDistance - heightPenalty;
        if (score > bestScore) {
            best = node;
            bestScore = score;
        }
    });
    if (best) {
        if (synthetic) synthetic.remove();
        return best;
    }

    let fallback = synthetic;
    if (!fallback) {
        fallback = document.createElement('div');
        fallback.className = 'icon-button-wrapper top-panel wangp-magic-mask-synthetic-toolbar';
        fallback.setAttribute('role', 'toolbar');
        fallback.setAttribute('aria-label', 'ImageEditor tools');
        if (getComputedStyle(editor).position === 'static') editor.style.position = 'relative';
        editor.appendChild(fallback);
    }
    return fallback;
};

WMM.visibleImageEditors = function () {
    return Array.from(document.querySelectorAll('#img_editor, [data-testid="image-editor"], .imageeditor, .image-editor'))
        .map((candidate) => candidate.closest('.block') || candidate.closest('[id^="component-"]') || candidate)
        .filter((editor, index, editors) => WMM.isVisible(editor) && editors.indexOf(editor) === index);
};

WMM.focusImageEditor = function (editor, clickBrush) {
    if (!editor || !WMM.isVisible(editor)) return false;
    const focusTarget = editor.querySelector('.pixi-target canvas, canvas, .pixi-target, [data-testid="image"]') || editor;
    if (!focusTarget.hasAttribute('tabindex')) focusTarget.setAttribute('tabindex', '-1');
    try {
        focusTarget.focus({ preventScroll: true });
    } catch (_) {
        focusTarget.focus();
    }
    if (clickBrush) {
        const brushButton = editor.querySelector('button[aria-label="Brush"]') || document.querySelector('button[aria-label="Brush"]');
        if (brushButton && !brushButton.disabled) brushButton.click();
    }
    return true;
};

WMM.focusVisibleImageEditor = function (clickBrush) {
    const editor = WMM.visibleImageEditors()[0];
    return WMM.focusImageEditor(editor, clickBrush);
};

WMM.refocusImageEditorAfterMagicMask = function () {
    const openPanel = Array.from(document.querySelectorAll('.wangp-magic-mask-panel:not(.hide)')).some((panel) => WMM.isVisible(panel));
    if (openPanel) return;
    setTimeout(() => WMM.focusVisibleImageEditor(true), 150);
    setTimeout(() => WMM.focusVisibleImageEditor(true), 700);
};

WMM.installImageEditorFocusPatch = function () {
    if (WMM.imageEditorFocusPatchInstalled) return;
    WMM.imageEditorFocusPatchInstalled = true;
    document.addEventListener('pointerdown', (event) => {
        const editor = event.target?.closest?.('#img_editor, [data-testid="image-editor"], .imageeditor, .image-editor')?.closest?.('.block') || event.target?.closest?.('#img_editor, [data-testid="image-editor"], .imageeditor, .image-editor');
        if (!editor || !WMM.isVisible(editor)) return;
        if (event.target?.closest?.('button, input, textarea, select, [role="button"]')) return;
        WMM.focusImageEditor(editor, false);
    }, true);
};

WMM.openMagicMaskPanelInAnchor = function (anchor) {
    const panel = anchor?.querySelector?.('.wangp-magic-mask-panel');
    if (!panel) return false;
    panel.hidden = false;
    panel.classList.remove('hide');
    panel.style.display = '';
    WMM.positionMagicMaskPopup(panel);
    return true;
};

WMM.positionMagicMaskPopup = function (panel) {
    if (!panel || !WMM.isVisible(panel)) return false;
    const anchor = panel.closest('.wangp-magic-mask-anchor');
    const rect = anchor?.getBoundingClientRect?.();
    const viewportW = Math.max(320, window.innerWidth || document.documentElement.clientWidth || 0);
    const viewportH = Math.max(320, window.innerHeight || document.documentElement.clientHeight || 0);
    const width = Math.min(Math.max(rect?.width || 0, 420), viewportW - 24, 780);
    const height = Math.min(496, viewportH - 24);
    const left = Math.min(Math.max(12, rect?.left || 12), Math.max(12, viewportW - width - 12));
    const top = Math.min(Math.max(12, rect?.top || 96), Math.max(12, viewportH - height - 12));
    panel.style.setProperty('left', `${Math.round(left)}px`, 'important');
    panel.style.setProperty('top', `${Math.round(top)}px`, 'important');
    panel.style.setProperty('right', 'auto', 'important');
    panel.style.setProperty('width', `${Math.round(width)}px`, 'important');
    panel.style.setProperty('height', `${Math.round(height)}px`, 'important');
    return true;
};

WMM.positionOpenMagicMaskPopups = function () {
    document.querySelectorAll('.wangp-magic-mask-panel:not(.hide)').forEach((panel) => WMM.positionMagicMaskPopup(panel));
};

WMM.decoratePopupShells = function () {
    document.querySelectorAll('.wangp-magic-mask-titlebar').forEach((titlebar) => titlebar.setAttribute('data-wangp-model-info-drag', ''));
    document.querySelectorAll('.wangp-magic-mask-close').forEach((closeButton) => closeButton.setAttribute('data-wangp-model-info-close', ''));
};

WMM.installOverlayTriggerPatch = function () {
    document.querySelectorAll('.wangp-magic-mask-trigger--overlay').forEach((trigger) => {
        if (trigger.dataset.wangpMagicMaskOverlayBound === '1') return;
        trigger.dataset.wangpMagicMaskOverlayBound = '1';
        trigger.addEventListener('click', () => {
            WMM.openMagicMaskPanelInAnchor(trigger.closest('.wangp-magic-mask-anchor'));
        }, true);
    });
};

WMM.findImageEditorForTrigger = function (trigger) {
    const roots = [
        trigger.closest('.wangp-magic-mask-anchor--image-editor'),
        trigger.parentElement,
        trigger.closest('.column'),
        trigger.closest('[id^="component-"]')?.parentElement,
    ].filter(Boolean);
    for (const root of roots) {
        const editor = Array.from(root.querySelectorAll('#img_editor, [data-testid="image-editor"], .imageeditor, .image-editor'))
            .find((candidate) => {
                const block = candidate.closest('.block') || candidate.closest('[id^="component-"]') || candidate;
                return WMM.isVisible(block);
            });
        if (editor) return editor.closest('.block') || editor.closest('[id^="component-"]') || editor;
    }
    const previousEditors = Array.from(document.querySelectorAll('#img_editor, [data-testid="image-editor"], .imageeditor, .image-editor'))
        .map((candidate) => candidate.closest('.block') || candidate.closest('[id^="component-"]') || candidate)
        .filter((editor) => WMM.isVisible(editor) && editor.compareDocumentPosition(trigger) & Node.DOCUMENT_POSITION_FOLLOWING);
    return previousEditors.pop() || null;
};

WMM.mountImageEditorTriggers = function () {
    WMM.decoratePopupShells();
    WMM.installOverlayTriggerPatch();
    document.querySelectorAll('.wangp-magic-mask-trigger--editor').forEach((trigger) => {
        const anchor = trigger.closest('.wangp-magic-mask-anchor--image-editor') || trigger.parentElement || document.body;
        if (trigger.classList.contains('hidden') || !!trigger.closest('.hidden') || !!trigger.closest('.hide') || !WMM.isVisible(anchor)) return;
        const editor = WMM.findImageEditorForTrigger(trigger);
        if (!editor || !trigger) return;
        const toolbar = WMM.findImageEditorToolbar(editor);
        if (!toolbar) {
            return;
        }
        anchor.querySelectorAll('.wangp-magic-mask-toolbar-button').forEach((button) => {
            if (button.parentElement !== toolbar) button.remove();
        });
        let toolbarButton = toolbar.querySelector('.wangp-magic-mask-toolbar-button');
        if (!toolbarButton) {
            toolbarButton = document.createElement('button');
            toolbarButton.type = 'button';
            toolbarButton.className = 'wangp-magic-mask-toolbar-button';
            toolbarButton.setAttribute('aria-label', 'Magic Mask');
            toolbarButton.setAttribute('title', 'Magic Mask');
            toolbarButton.textContent = '\u{1FA84}';
            toolbar.appendChild(toolbarButton);
        }
        toolbar.classList.add('wangp-magic-mask-toolbar');
        toolbarButton.hidden = trigger.classList.contains('hidden') || !!trigger.closest('.hidden');
        toolbarButton.disabled = trigger.disabled;
        const needsImage = /Upload an image/i.test(editor.innerText || '') && /select the draw tool to start/i.test(editor.innerText || '');
        toolbarButton.classList.toggle('wangp-magic-mask-unavailable', needsImage);
        toolbarButton.title = needsImage ? 'Magic Mask needs a control image' : 'Magic Mask';
        toolbarButton.onclick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (needsImage) return;
            trigger.click();
        };
    });
};

WMM.scheduleMount = function () {
    if (WMM.raf) cancelAnimationFrame(WMM.raf);
    WMM.raf = requestAnimationFrame(WMM.mountImageEditorTriggers);
};

if (!WMM.init) {
    WMM.init = true;
    WMM.observer = new MutationObserver(WMM.scheduleMount);
    const root = document.querySelector('gradio-app') || document.body;
    if (root) WMM.observer.observe(root, { childList: true, subtree: true });
    window.addEventListener('resize', WMM.scheduleMount);
    window.addEventListener('load', WMM.scheduleMount);
    WMM.interval = window.setInterval(WMM.scheduleMount, 500);
}
WMM.installImageEditorFocusPatch();
WMM.scheduleMount();
"""

    @staticmethod
    def focus_image_editor_javascript():
        return "() => { window.__wangpMagicMaskNS?.refocusImageEditorAfterMagicMask?.(); }"

    @staticmethod
    def position_popup_javascript():
        return "() => { window.__wangpMagicMaskNS?.positionOpenMagicMaskPopups?.(); }"

    def render(self, visible=False, trigger_mode="overlay"):
        self.abort_token = gr.State(str(uuid.uuid4()))
        self.pending_image_mask_guide = gr.State(None)
        self.pending_image_mask = gr.State(None)
        self.trigger = gr.Button(MAGIC_WAND_LABEL, size="sm", min_width=1, visible=visible, elem_classes=["wangp-magic-mask-trigger", f"wangp-magic-mask-trigger--{trigger_mode}"])
        with gr.Column(visible=False, elem_classes=["wangp-magic-mask-panel", "wangp-model-info-popup"]) as self.panel:
            with gr.Column(elem_classes=["wangp-magic-mask-card", "wangp-model-info-card"]):
                with gr.Row(elem_classes=["wangp-model-info-titlebar", "wangp-magic-mask-titlebar"]):
                    gr.HTML("<div class='wangp-model-info-heading'>Magic Mask</div>", elem_classes=["wangp-magic-mask-heading"])
                    self.close_btn = gr.Button("x", elem_classes=["wangp-model-info-close", "wangp-magic-mask-close"], min_width=26, scale=0)
                with gr.Column(elem_classes=["wangp-magic-mask-body"]):
                    gr.HTML("<div class='wangp-magic-mask-intro'>Enter the list of Object or Persons to track and that will be used to build the Mask. Each object / person should be separated by a \",\". For example: \"blue car, woman to the right\"</div>")
                    with gr.Row(elem_classes=["wangp-magic-mask-keyword-row"]):
                        self.keywords = gr.Textbox(show_label=False, placeholder="person, car, sky", lines=1, scale=6, elem_classes=["wangp-magic-mask-keywords"])
                        with gr.Group(elem_classes=["wangp-magic-mask-negative"]):
                            self.negative_mask = gr.Checkbox(label="Negative Mask", value=False, container=False, min_width=1, elem_classes=["wangp-magic-mask-negative-checkbox"])
                    with gr.Row(elem_classes=["wangp-magic-mask-options-row"]):
                        self.max_objects = gr.Dropdown(choices=[("All", "all")], value="all", label="Max Objects", scale=1, visible=False, elem_classes=["wangp-magic-mask-max-objects"])
                        self.max_time = gr.Textbox(value="", label="Max Time (s)", placeholder="", lines=1, scale=1, elem_classes=["wangp-magic-mask-max-time"])
                self.status = gr.HTML("")
                self.progress_html = gr.HTML("")
                gr.HTML("", elem_classes=["wangp-magic-mask-spacer"], padding=False)
                with gr.Row(elem_classes=["wangp-magic-mask-actions"]):
                    self.cancel_btn = gr.Button("Exit", size="sm", elem_classes=["wangp-magic-mask-btn", "wangp-magic-mask-exit-btn"])
                    self.abort_btn = gr.Button("Abort", size="sm", visible=False, elem_classes=["wangp-magic-mask-btn", "wangp-magic-mask-btn--danger"])
                    self.generate_btn = gr.Button("Generate", size="sm", elem_classes=["wangp-magic-mask-btn", "wangp-magic-mask-btn--primary"])
        return self

    def mount(
        self,
        *,
        state,
        image_mode,
        video_guide,
        image_mask_guide,
        image_guide,
        image_mask,
        video_mask,
        download_assets: Callable[[dict[str, Any]], Any],
        acquire_gpu: Callable[[Any, str, str], Any],
        release_gpu: Callable[[Any, str], Any],
        get_model_settings: Callable[[Any], dict],
        get_model_def: Callable[[Any], dict] | None = None,
        model_def: dict | None = None,
    ):
        def model_def_resolver(state_value):
            return model_def if isinstance(model_def, dict) else get_model_def(state_value) if callable(get_model_def) else {}

        def open_panel(state_value, image_mode_value, max_objects_value):
            is_image_mask = int(image_mode_value or 0) > 0
            max_time_update = gr.update(visible=not is_image_mask)
            if max_objects_value != "all":
                try:
                    max_objects_value = _max_objects_limit(max_objects_value)
                except (TypeError, ValueError):
                    max_objects_value = "all"
            return (*_open_panel(), gr.update(choices=_max_object_choices(), value=max_objects_value, visible=True), max_time_update)

        def close_panel():
            return (*_close_panel(), gr.update(), gr.update())

        open_event = self.trigger.click(fn=open_panel, inputs=[state, image_mode, self.max_objects], outputs=[self.panel, self.status, self.progress_html, self.cancel_btn, self.abort_btn, self.pending_image_mask_guide, self.pending_image_mask, self.max_objects, self.max_time], show_progress="hidden")
        open_event.then(fn=None, inputs=[], outputs=[], js=MagicMaskUI.position_popup_javascript())
        close_outputs = [self.panel, self.status, self.progress_html, self.abort_btn, self.pending_image_mask_guide, self.pending_image_mask, self.max_objects, self.max_time]
        self.close_btn.click(fn=close_panel, inputs=[], outputs=close_outputs, show_progress="hidden")
        self.cancel_btn.click(fn=close_panel, inputs=[], outputs=close_outputs, show_progress="hidden")
        self.abort_btn.click(fn=_abort_magic_mask, inputs=[self.abort_token], outputs=[self.status, self.abort_btn], show_progress="hidden")

        def generate(state_value, keywords_text, negative_mask_value, image_mode_value, video_guide_value, image_mask_guide_value, image_guide_value, abort_token_value, max_objects_value, max_time_value):
            yield from _generate_magic_mask(
                state_value,
                keywords_text,
                negative_mask_value,
                image_mode_value,
                video_guide_value,
                image_mask_guide_value,
                image_guide_value,
                abort_token_value,
                max_objects_value,
                max_time_value,
                download_assets=download_assets,
                acquire_gpu=acquire_gpu,
                release_gpu=release_gpu,
                get_model_settings=get_model_settings,
                get_model_def=model_def_resolver,
            )

        generate_event = self.generate_btn.click(
            fn=generate,
            inputs=[state, self.keywords, self.negative_mask, image_mode, video_guide, image_mask_guide, image_guide, self.abort_token, self.max_objects, self.max_time],
            outputs=[image_mask_guide, image_mask, video_mask, self.status, self.panel, self.progress_html, self.cancel_btn, self.abort_btn, self.pending_image_mask_guide, self.pending_image_mask],
            show_progress="hidden",
        )
        generate_event.then(fn=None, inputs=[], outputs=[], js=MagicMaskUI.focus_image_editor_javascript())
