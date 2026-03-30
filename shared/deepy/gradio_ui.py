from __future__ import annotations

import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gradio as gr

from shared.deepy import tool_settings as deepy_tool_settings
from shared.deepy import ui_settings as deepy_ui_settings
from shared.gradio import assistant_chat


_TEMPLATE_TOOL_LAYOUT = (
    ("gen_video", "gen_video_with_speech"),
    ("gen_image", "edit_image"),
    ("gen_speech_from_description", "gen_speech_from_sample"),
)
_TEMPLATE_TOOL_ORDER = tuple(tool_name for row in _TEMPLATE_TOOL_LAYOUT for tool_name in row)
_TEMPLATE_TOOL_SELECTOR_CHOICE_KEY = {
    "gen_video": "video_generator_choices",
    "gen_video_with_speech": "video_with_speech_choices",
    "gen_image": "image_generator_choices",
    "edit_image": "image_editor_choices",
    "gen_speech_from_description": "speech_from_description_choices",
    "gen_speech_from_sample": "speech_from_sample_choices",
}
_TEMPLATE_TOOL_SELECTOR_SELECTED_KEY = {
    "gen_video": "selected_video_generator",
    "gen_video_with_speech": "selected_video_with_speech",
    "gen_image": "selected_image_generator",
    "edit_image": "selected_image_editor",
    "gen_speech_from_description": "selected_speech_from_description",
    "gen_speech_from_sample": "selected_speech_from_sample",
}
_TEMPLATE_TOOL_UI_KEY = {
    "gen_video": "video_generator_variant",
    "gen_video_with_speech": "video_with_speech_variant",
    "gen_image": "image_generator_variant",
    "edit_image": "image_editor_variant",
    "gen_speech_from_description": "speech_from_description_variant",
    "gen_speech_from_sample": "speech_from_sample_variant",
}
_TEMPLATE_TOOL_DEFAULT_GETTER = {
    "gen_video": deepy_tool_settings.get_default_video_generator_variant,
    "gen_video_with_speech": deepy_tool_settings.get_default_video_with_speech_variant,
    "gen_image": deepy_tool_settings.get_default_image_generator_variant,
    "edit_image": deepy_tool_settings.get_default_image_editor_variant,
    "gen_speech_from_description": deepy_tool_settings.get_default_speech_from_description_variant,
    "gen_speech_from_sample": deepy_tool_settings.get_default_speech_from_sample_variant,
}
_TEMPLATE_ADD_SELECTION_ERROR = "Please Select User Settings in the Lora / Settings Dropdown Box"
_TEMPLATE_DELETE_BUILTIN_ERROR = "You cant delete a Built in Template"
_TEMPLATE_CAPTURE_JS = """() => {
  const selection = window.WAC && typeof window.WAC.getWanGpSettingsSelection === 'function'
    ? window.WAC.getWanGpSettingsSelection()
    : { value: '', label: '' };
  return [selection.value || '', selection.label || ''];
}"""


@dataclass(slots=True)
class DeepyTemplateToolControl:
    tool_name: str
    dropdown: Any
    add_btn: Any
    delete_btn: Any


@dataclass(slots=True)
class DeepyChatUI:
    dock: Any
    launcher_host: Any
    panel: Any
    settings_launcher_host: Any
    settings_save_btn: Any
    html_output: Any
    chat_event: Any
    stats_output: Any
    stop_btn: Any
    request: Any
    ask_btn: Any
    reset_btn: Any
    auto_cancel_queue_tasks: Any
    use_template_properties: Any
    override_height: Any
    override_width: Any
    override_num_frames: Any
    override_seed: Any
    default_video_with_speech: Any
    default_image_generator: Any
    default_image_editor: Any
    default_video_generator: Any
    default_speech_from_description: Any
    default_speech_from_sample: Any
    template_controls: tuple[DeepyTemplateToolControl, ...]
    template_selection_history: Any
    template_modal_state: Any
    captured_lset_value: Any
    captured_lset_label: Any
    template_modal: Any
    template_modal_title: Any
    template_modal_body: Any
    template_modal_yes_btn: Any
    template_modal_no_btn: Any
    template_modal_close_btn: Any


@dataclass(slots=True)
class DeepyChatHandlers:
    prepare_request_context: Callable[[Any, Any, Any, Any, Any], Any]
    update_tool_ui_settings: Callable[..., Any]
    store_selected_video_time: Callable[[Any, Any], Any]
    ask_ai: Callable[[Any, str], Any]
    stop_ai: Callable[[Any], Any]
    reset_ai: Callable[[Any], Any]


def _tool_values_from_inputs(current_video_generator: Any, current_video_with_speech: Any, current_image_generator: Any, current_image_editor: Any, current_speech_from_description: Any, current_speech_from_sample: Any) -> dict[str, Any]:
    return {
        "gen_video": current_video_generator,
        "gen_video_with_speech": current_video_with_speech,
        "gen_image": current_image_generator,
        "edit_image": current_image_editor,
        "gen_speech_from_description": current_speech_from_description,
        "gen_speech_from_sample": current_speech_from_sample,
    }


def _tool_values_from_ui_settings(tool_ui_state: dict[str, Any]) -> dict[str, Any]:
    return {tool_name: tool_ui_state[_TEMPLATE_TOOL_UI_KEY[tool_name]] for tool_name in _TEMPLATE_TOOL_ORDER}


def _normalize_tool_variant(tool_name: str, value: Any) -> str:
    resolved = deepy_tool_settings.find_tool_variant(tool_name, value)
    if resolved is not None:
        return resolved
    try:
        fallback = _TEMPLATE_TOOL_DEFAULT_GETTER[tool_name]()
    except Exception:
        variants = deepy_tool_settings.list_tool_variants(tool_name)
        fallback = variants[0] if len(variants) > 0 else ""
    return str(fallback or "")


def _build_template_selection_history(tool_values: dict[str, Any]) -> dict[str, dict[str, str]]:
    history: dict[str, dict[str, str]] = {}
    for tool_name in _TEMPLATE_TOOL_ORDER:
        current = _normalize_tool_variant(tool_name, tool_values.get(tool_name))
        history[tool_name] = {"current": current, "previous": current}
    return history


def _normalize_template_selection_history(history: Any, tool_values: dict[str, Any]) -> dict[str, dict[str, str]]:
    raw_history = history if isinstance(history, dict) else {}
    normalized: dict[str, dict[str, str]] = {}
    for tool_name in _TEMPLATE_TOOL_ORDER:
        current = _normalize_tool_variant(tool_name, tool_values.get(tool_name))
        previous = None
        record = raw_history.get(tool_name)
        if isinstance(record, dict):
            previous = deepy_tool_settings.find_tool_variant(tool_name, record.get("previous"))
        if previous is None:
            previous = current
        normalized[tool_name] = {"current": current, "previous": str(previous or current or "")}
    return normalized


def _modal_title_html(tool_name: str) -> str:
    display_name = deepy_tool_settings.TOOL_DISPLAY_NAMES.get(tool_name, tool_name.replace("_", " ").title())
    return (
        "<div class='wangp-assistant-chat__template-modal-titlebar'>"
        f"<div class='wangp-assistant-chat__template-modal-heading'>{html.escape(display_name)} Tool</div>"
        "</div>"
    )


def _settings_title_html() -> str:
    return (
        "<div class='wangp-assistant-chat__template-modal-titlebar'>"
        "<div class='wangp-assistant-chat__template-modal-heading'>Deepy Settings</div>"
        "</div>"
    )


def _tool_display_name(tool_name: str) -> str:
    return deepy_tool_settings.TOOL_DISPLAY_NAMES.get(tool_name, tool_name.replace("_", " ").title())


def _modal_context_html(label: str, value: str) -> str:
    return (
        "<div class='wangp-assistant-chat__template-modal-context'>"
        f"<div class='wangp-assistant-chat__template-modal-context-label'>{html.escape(label)}</div>"
        f"<div class='wangp-assistant-chat__template-modal-context-value'>{html.escape(value)}</div>"
        "</div>"
    )


def _wangp_settings_placeholders() -> set[str]:
    get_new_preset_msg = getattr(sys.modules.get("__main__"), "get_new_preset_msg", None)
    if not callable(get_new_preset_msg):
        return set()
    placeholders: set[str] = set()
    for advanced in (True, False):
        label = str(get_new_preset_msg(advanced) or "").strip()
        if len(label) > 0:
            placeholders.add(label)
    return placeholders


def _current_wangp_settings_context_html(value: str) -> str:
    selected = str(value or "").strip()
    if len(selected) == 0 or selected in _wangp_settings_placeholders():
        return ""
    return _modal_context_html("Current WanGP Settings", selected)


def _modal_message_html(message: str, *, tone: str = "info") -> str:
    tone_class = {"info": "is-info", "warning": "is-warning", "error": "is-error"}.get(str(tone or "").strip().lower(), "is-info")
    return f"<div class='wangp-assistant-chat__template-modal-message {tone_class}'>{html.escape(message)}</div>"


def _closed_template_modal() -> tuple[dict[str, Any], Any, Any, Any, Any, Any, Any]:
    return ({}, gr.update(visible=False), gr.update(value=""), gr.update(value=""), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))


def _open_template_modal(modal_state: dict[str, Any], title_html: str, body_html: str, *, yes_visible: bool = False, no_visible: bool = False, close_visible: bool = True) -> tuple[dict[str, Any], Any, Any, Any, Any, Any, Any]:
    return (
        dict(modal_state or {}),
        gr.update(visible=True),
        gr.update(value=title_html),
        gr.update(value=body_html),
        gr.update(visible=yes_visible),
        gr.update(visible=no_visible),
        gr.update(visible=close_visible),
    )


def _template_dropdown_updates(tool_values: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, str]]:
    refreshed = deepy_ui_settings.refresh_template_selector_state(
        tool_values.get("gen_image"),
        tool_values.get("edit_image"),
        tool_values.get("gen_video"),
        tool_values.get("gen_video_with_speech"),
        tool_values.get("gen_speech_from_description"),
        tool_values.get("gen_speech_from_sample"),
    )
    selected_values: dict[str, str] = {}
    dropdown_updates: list[Any] = []
    for tool_name in _TEMPLATE_TOOL_ORDER:
        selected_value = deepy_tool_settings.find_tool_variant(tool_name, refreshed.get(_TEMPLATE_TOOL_SELECTOR_SELECTED_KEY[tool_name]))
        if selected_value is None:
            selected_value = _normalize_tool_variant(tool_name, tool_values.get(tool_name))
        selected_values[tool_name] = str(selected_value or "")
        dropdown_updates.append(gr.update(choices=refreshed[_TEMPLATE_TOOL_SELECTOR_CHOICE_KEY[tool_name]], value=selected_values[tool_name]))
    return tuple(dropdown_updates), selected_values


def build_deepy_chat_ui(*, deepy_visible: bool) -> DeepyChatUI:
    template_selector_state = deepy_ui_settings.get_template_selector_state()
    tool_ui_state = deepy_ui_settings.get_persisted_assistant_tool_ui_settings()
    initial_tool_values = _tool_values_from_ui_settings(tool_ui_state)
    template_controls: list[DeepyTemplateToolControl] = []
    controls_by_tool: dict[str, DeepyTemplateToolControl] = {}
    with gr.Column(elem_id=assistant_chat.DOCK_ID) as dock:
        launcher_host = gr.HTML(assistant_chat.render_launcher_html() if deepy_visible else "", elem_id=assistant_chat.LAUNCHER_HOST_ID, visible=deepy_visible)
        with gr.Column(elem_id=assistant_chat.PANEL_ID, visible=deepy_visible) as panel:
            settings_launcher_host = gr.HTML(assistant_chat.render_settings_launcher_html(), elem_id=assistant_chat.SETTINGS_LAUNCHER_HOST_ID)
            html_output = gr.HTML(assistant_chat.render_shell_html(), elem_id=assistant_chat.CHAT_BLOCK_ID)
            chat_event = gr.Text(value="", interactive=False, visible=False, elem_id=assistant_chat.CHAT_EVENT_ID)
            stop_btn = gr.Button("Stop", elem_id=assistant_chat.STOP_BRIDGE_ID)
            with gr.Row(elem_id=assistant_chat.CONTROLS_ID):
                request = gr.Text(value="", label="Request", scale=3, show_label=False, elem_id=assistant_chat.REQUEST_ID)
                ask_btn = gr.Button("Ask", scale=1, min_width=10, elem_id=assistant_chat.ASK_BUTTON_ID)
                reset_btn = gr.Button("Reset", scale=1, min_width=10, elem_id=assistant_chat.RESET_BUTTON_ID)
            stats_output = gr.HTML(assistant_chat.render_stats_html(), elem_id=assistant_chat.STATS_BLOCK_ID)
            with gr.Column(elem_id=assistant_chat.SETTINGS_PANEL_ID):
                with gr.Column(elem_classes=["wangp-assistant-chat__template-modal-card", "wangp-assistant-chat__settings-card"]):
                    gr.HTML(_settings_title_html())
                    with gr.Column(elem_classes=["wangp-assistant-chat__settings-scroll"]):
                        with gr.Tabs():
                            with gr.Tab("Generation Properties"):
                                auto_cancel_queue_tasks = gr.Checkbox(
                                    value=tool_ui_state["auto_cancel_queue_tasks"],
                                    label="Auto-abort or remove Deepy-started generation on Stop/Reset.",
                                )
                                use_template_properties = gr.Checkbox(value=tool_ui_state["use_template_properties"], label="Use Properties defined in Templates Settings files.")
                                with gr.Row():
                                    override_width = gr.Slider(
                                        deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MIN,
                                        deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MAX,
                                        value=tool_ui_state["width"],
                                        step=deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_STEP,
                                        label="Default Width",
                                        interactive=not tool_ui_state["use_template_properties"],
                                    )
                                    override_height = gr.Slider(
                                        deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MIN,
                                        deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MAX,
                                        value=tool_ui_state["height"],
                                        step=deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_STEP,
                                        label="Default Height",
                                        interactive=not tool_ui_state["use_template_properties"],
                                    )
                                override_num_frames = gr.Slider(
                                    deepy_ui_settings.ASSISTANT_OVERRIDE_FRAMES_MIN,
                                    deepy_ui_settings.ASSISTANT_OVERRIDE_FRAMES_MAX,
                                    value=tool_ui_state["num_frames"],
                                    step=1,
                                    label="Default Number of Frames",
                                    interactive=not tool_ui_state["use_template_properties"],
                                )
                                override_seed = gr.Slider(
                                    -1,
                                    999999999,
                                    value=tool_ui_state["seed"],
                                    step=1,
                                    label="Seed (-1 for random)",
                                    interactive=not tool_ui_state["use_template_properties"],
                                )
                            with gr.Tab("Templates Settings used by Tools"):
                                with gr.Column(elem_classes=["wangp-assistant-chat__template-tool-grid"]):
                                    gr.Markdown("Please Match here Prerecorded Models Settings to each Generation Tool used by Deepy.")
                                    for tool_pair in _TEMPLATE_TOOL_LAYOUT:
                                        with gr.Row(elem_classes=["wangp-assistant-chat__template-tool-grid-row"]):
                                            for tool_name in tool_pair:
                                                with gr.Column(elem_classes=["wangp-assistant-chat__template-tool-card"]):
                                                    with gr.Row(elem_classes=["wangp-assistant-chat__template-tool-row"]):
                                                        dropdown = gr.Dropdown(
                                                            choices=template_selector_state[_TEMPLATE_TOOL_SELECTOR_CHOICE_KEY[tool_name]],
                                                            value=tool_ui_state[_TEMPLATE_TOOL_UI_KEY[tool_name]],
                                                            label=deepy_tool_settings.TOOL_DISPLAY_NAMES[tool_name],
                                                            elem_classes=["wangp-assistant-chat__template-tool-dropdown"],
                                                        )
                                                        with gr.Column(scale=0, min_width=34, elem_classes=["wangp-assistant-chat__template-tool-actions"]):
                                                            add_btn = gr.Button("\u2795", size="sm", min_width=1, elem_classes=["wangp-assistant-chat__template-tool-icon-btn"])
                                                            delete_btn = gr.Button("\U0001F5D1\uFE0F", size="sm", min_width=1, elem_classes=["wangp-assistant-chat__template-tool-icon-btn", "wangp-assistant-chat__template-tool-icon-btn--danger"])
                                                control = DeepyTemplateToolControl(tool_name=tool_name, dropdown=dropdown, add_btn=add_btn, delete_btn=delete_btn)
                                                controls_by_tool[tool_name] = control
                                                template_controls.append(control)
                        with gr.Row(elem_classes=["wangp-assistant-chat__settings-actions"]):
                            settings_save_btn = gr.Button("Save Deepy Settings", variant="primary", elem_id=assistant_chat.SAVE_SETTINGS_BUTTON_ID)
                template_selection_history = gr.State(_build_template_selection_history(initial_tool_values))
                template_modal_state = gr.State({})
                captured_lset_value = gr.Text(value="", interactive=False, visible=False)
                captured_lset_label = gr.Text(value="", interactive=False, visible=False)
                with gr.Group(visible=False, elem_classes=["wangp-assistant-chat__template-modal-wrap"]) as template_modal:
                    with gr.Column(elem_classes=["wangp-assistant-chat__template-modal-card"]):
                        template_modal_title = gr.HTML("")
                        template_modal_body = gr.HTML("")
                        with gr.Row(elem_classes=["wangp-assistant-chat__template-modal-actions"]):
                            template_modal_yes_btn = gr.Button("Yes", size="sm", visible=False, elem_classes=["wangp-assistant-chat__template-modal-btn", "wangp-assistant-chat__template-modal-btn--primary"])
                            template_modal_no_btn = gr.Button("No", size="sm", visible=False, elem_classes=["wangp-assistant-chat__template-modal-btn"])
                            template_modal_close_btn = gr.Button("Close", size="sm", visible=False, elem_classes=["wangp-assistant-chat__template-modal-btn"])
    return DeepyChatUI(
        dock=dock,
        launcher_host=launcher_host,
        panel=panel,
        settings_launcher_host=settings_launcher_host,
        settings_save_btn=settings_save_btn,
        html_output=html_output,
        chat_event=chat_event,
        stats_output=stats_output,
        stop_btn=stop_btn,
        request=request,
        ask_btn=ask_btn,
        reset_btn=reset_btn,
        auto_cancel_queue_tasks=auto_cancel_queue_tasks,
        use_template_properties=use_template_properties,
        override_height=override_height,
        override_width=override_width,
        override_num_frames=override_num_frames,
        override_seed=override_seed,
        default_video_with_speech=controls_by_tool["gen_video_with_speech"].dropdown,
        default_image_generator=controls_by_tool["gen_image"].dropdown,
        default_image_editor=controls_by_tool["edit_image"].dropdown,
        default_video_generator=controls_by_tool["gen_video"].dropdown,
        default_speech_from_description=controls_by_tool["gen_speech_from_description"].dropdown,
        default_speech_from_sample=controls_by_tool["gen_speech_from_sample"].dropdown,
        template_controls=tuple(template_controls),
        template_selection_history=template_selection_history,
        template_modal_state=template_modal_state,
        captured_lset_value=captured_lset_value,
        captured_lset_label=captured_lset_label,
        template_modal=template_modal,
        template_modal_title=template_modal_title,
        template_modal_body=template_modal_body,
        template_modal_yes_btn=template_modal_yes_btn,
        template_modal_no_btn=template_modal_no_btn,
        template_modal_close_btn=template_modal_close_btn,
    )


def bind_deepy_chat_ui(
    ui: DeepyChatUI,
    *,
    state: Any,
    output: Any,
    last_choice: Any,
    audio_files_paths: Any,
    audio_file_selected: Any,
    selected_video_time_input: Any,
    load_queue_trigger: Any,
    output_trigger: Any,
    abort_client_id: Any,
    handlers: DeepyChatHandlers,
) -> None:
    template_modal_outputs = [
        ui.template_modal_state,
        ui.template_modal,
        ui.template_modal_title,
        ui.template_modal_body,
        ui.template_modal_yes_btn,
        ui.template_modal_no_btn,
        ui.template_modal_close_btn,
    ]
    template_dropdown_inputs = [
        ui.default_video_generator,
        ui.default_video_with_speech,
        ui.default_image_generator,
        ui.default_image_editor,
        ui.default_speech_from_description,
        ui.default_speech_from_sample,
    ]
    template_dropdown_outputs = list(template_dropdown_inputs)

    def toggle_override_controls(use_template_properties):
        interactive = not deepy_ui_settings.normalize_assistant_use_template_properties(use_template_properties)
        return gr.update(interactive=interactive), gr.update(interactive=interactive), gr.update(interactive=interactive), gr.update(interactive=interactive)

    def track_template_selection(tool_name, selection_history, current_video_generator, current_video_with_speech, current_image_generator, current_image_editor, current_speech_from_description, current_speech_from_sample):
        raw_history = selection_history if isinstance(selection_history, dict) else {}
        previous_current = None
        record = raw_history.get(tool_name)
        if isinstance(record, dict):
            previous_current = deepy_tool_settings.find_tool_variant(tool_name, record.get("current"))
        tool_values = _tool_values_from_inputs(current_video_generator, current_video_with_speech, current_image_generator, current_image_editor, current_speech_from_description, current_speech_from_sample)
        normalized_history = _normalize_template_selection_history(selection_history, tool_values)
        current_value = normalized_history[tool_name]["current"]
        if previous_current is not None and previous_current != current_value:
            normalized_history[tool_name]["previous"] = previous_current
        return normalized_history

    def close_template_modal():
        return _closed_template_modal()

    def ask_ai_with_ui_settings(
        state_value,
        output_value,
        last_choice_value,
        audio_files_paths_value,
        audio_file_selected_value,
        ask_request,
        auto_cancel_queue_tasks,
        use_template_properties,
        override_height,
        override_width,
        override_num_frames,
        override_seed,
        default_video_generator,
        default_video_with_speech,
        default_image_generator,
        default_image_editor,
        default_speech_from_description,
        default_speech_from_sample,
    ):
        handlers.prepare_request_context(state_value, output_value, last_choice_value, audio_files_paths_value, audio_file_selected_value)
        update_session_ui_settings(
            state_value,
            auto_cancel_queue_tasks,
            use_template_properties,
            override_height,
            override_width,
            override_num_frames,
            override_seed,
            default_video_generator,
            default_video_with_speech,
            default_image_generator,
            default_image_editor,
            default_speech_from_description,
            default_speech_from_sample,
        )
        yield from handlers.ask_ai(state_value, ask_request)

    def _apply_ui_settings(
        state_value,
        auto_cancel_queue_tasks,
        use_template_properties,
        override_height,
        override_width,
        override_num_frames,
        override_seed,
        default_video_generator,
        default_video_with_speech,
        default_image_generator,
        default_image_editor,
        default_speech_from_description,
        default_speech_from_sample,
        *,
        persist,
    ):
        return handlers.update_tool_ui_settings(
            state_value,
            auto_cancel_queue_tasks=auto_cancel_queue_tasks,
            use_template_properties=use_template_properties,
            width=override_width,
            height=override_height,
            num_frames=override_num_frames,
            seed=override_seed,
            video_with_speech_variant=default_video_with_speech,
            image_generator_variant=default_image_generator,
            image_editor_variant=default_image_editor,
            video_generator_variant=default_video_generator,
            speech_from_description_variant=default_speech_from_description,
            speech_from_sample_variant=default_speech_from_sample,
            persist=persist,
        )

    def update_session_ui_settings(
        state_value,
        auto_cancel_queue_tasks,
        use_template_properties,
        override_height,
        override_width,
        override_num_frames,
        override_seed,
        default_video_generator,
        default_video_with_speech,
        default_image_generator,
        default_image_editor,
        default_speech_from_description,
        default_speech_from_sample,
    ):
        return _apply_ui_settings(
            state_value,
            auto_cancel_queue_tasks,
            use_template_properties,
            override_height,
            override_width,
            override_num_frames,
            override_seed,
            default_video_generator,
            default_video_with_speech,
            default_image_generator,
            default_image_editor,
            default_speech_from_description,
            default_speech_from_sample,
            persist=False,
        )

    def persist_ui_settings(
        state_value,
        auto_cancel_queue_tasks,
        use_template_properties,
        override_height,
        override_width,
        override_num_frames,
        override_seed,
        default_video_generator,
        default_video_with_speech,
        default_image_generator,
        default_image_editor,
        default_speech_from_description,
        default_speech_from_sample,
    ):
        _apply_ui_settings(
            state_value,
            auto_cancel_queue_tasks,
            use_template_properties,
            override_height,
            override_width,
            override_num_frames,
            override_seed,
            default_video_generator,
            default_video_with_speech,
            default_image_generator,
            default_image_editor,
            default_speech_from_description,
            default_speech_from_sample,
            persist=True,
        )

    def stop_ai_with_ui(state_value):
        return handlers.stop_ai(state_value)

    def reset_ai_with_ui(state_value):
        return handlers.reset_ai(state_value)

    def open_add_template_modal(tool_name, state_value, lset_value, lset_label, current_variant):
        selected_label = str(Path(str(lset_value or "").strip()).name or str(lset_label or "").strip() or "Nothing selected").strip()
        title_html = _modal_title_html(tool_name)
        source_path = deepy_tool_settings.resolve_wangp_settings_file(state_value, lset_value)
        if source_path is None:
            body_html = _current_wangp_settings_context_html(selected_label)
            body_html += _modal_message_html(_TEMPLATE_ADD_SELECTION_ERROR, tone="error")
            return _open_template_modal({}, title_html, body_html, close_visible=True)
        try:
            validation_error = deepy_tool_settings.validate_wangp_settings_for_tool(tool_name, source_path)
        except Exception as exc:
            validation_error = str(exc)
        source_label = source_path.stem
        if validation_error is not None and len(str(validation_error).strip()) > 0:
            body_html = _modal_context_html("Selected WanGP Settings", source_label)
            body_html += _modal_message_html(str(validation_error).strip(), tone="error")
            return _open_template_modal({}, title_html, body_html, close_visible=True)
        linked_variant = deepy_tool_settings.build_linked_tool_variant(state_value, source_path)
        body_html = _modal_context_html("Selected WanGP Settings", source_label)
        body_html += _modal_message_html(f'You are about to link Tool {_tool_display_name(tool_name)} to Settings "{source_label}". Are you sure ?', tone="info")
        modal_state = {
            "action": "add",
            "tool_name": tool_name,
            "variant_name": linked_variant,
            "source_path": source_label,
            "previous_variant": deepy_tool_settings.find_tool_variant(tool_name, current_variant) or "",
        }
        return _open_template_modal(modal_state, title_html, body_html, yes_visible=True, no_visible=True, close_visible=False)

    def open_delete_template_modal(tool_name, current_variant):
        title_html = _modal_title_html(tool_name)
        selected_variant = str(current_variant or "").strip()
        selected_label = str(Path(selected_variant).name or selected_variant or "Nothing selected").strip()
        body_html = _modal_context_html("Selected Deepy Template", selected_label)
        if not deepy_tool_settings.is_linked_tool_variant(selected_variant):
            body_html += _modal_message_html(_TEMPLATE_DELETE_BUILTIN_ERROR, tone="error")
            return _open_template_modal({}, title_html, body_html, close_visible=True)
        body_html += _modal_message_html(f"You are about to remove the link to {selected_label}. Are you sure ?", tone="warning")
        modal_state = {"action": "delete", "tool_name": tool_name, "variant_name": selected_variant}
        return _open_template_modal(modal_state, title_html, body_html, yes_visible=True, no_visible=True, close_visible=False)

    def confirm_template_modal_action(template_modal_state, selection_history, current_video_generator, current_video_with_speech, current_image_generator, current_image_editor, current_speech_from_description, current_speech_from_sample):
        tool_values = _tool_values_from_inputs(current_video_generator, current_video_with_speech, current_image_generator, current_image_editor, current_speech_from_description, current_speech_from_sample)
        normalized_history = _normalize_template_selection_history(selection_history, tool_values)
        modal_state = template_modal_state if isinstance(template_modal_state, dict) else {}
        action = str(modal_state.get("action", "")).strip().lower()
        tool_name = str(modal_state.get("tool_name", "")).strip()
        if action not in {"add", "delete"} or tool_name not in _TEMPLATE_TOOL_ORDER:
            dropdown_updates, selected_values = _template_dropdown_updates(tool_values)
            normalized_history = _normalize_template_selection_history(normalized_history, selected_values)
            return (*dropdown_updates, normalized_history, *_closed_template_modal())
        try:
            if action == "add":
                previous_variant = deepy_tool_settings.find_tool_variant(tool_name, modal_state.get("previous_variant")) or normalized_history[tool_name]["current"]
                new_variant = str(modal_state.get("variant_name", "")).strip()
                tool_values[tool_name] = new_variant
                dropdown_updates, selected_values = _template_dropdown_updates(tool_values)
                normalized_history = _normalize_template_selection_history(normalized_history, selected_values)
                normalized_history[tool_name]["current"] = selected_values[tool_name]
                normalized_history[tool_name]["previous"] = deepy_tool_settings.find_tool_variant(tool_name, previous_variant) or selected_values[tool_name]
                return (*dropdown_updates, normalized_history, *_closed_template_modal())
            restored_variant = deepy_tool_settings.find_tool_variant(tool_name, normalized_history[tool_name]["previous"])
            if restored_variant is None:
                restored_variant = _normalize_tool_variant(tool_name, "")
            tool_values[tool_name] = restored_variant
            dropdown_updates, selected_values = _template_dropdown_updates(tool_values)
            normalized_history = _normalize_template_selection_history(normalized_history, selected_values)
            normalized_history[tool_name]["current"] = selected_values[tool_name]
            normalized_history[tool_name]["previous"] = selected_values[tool_name]
            return (*dropdown_updates, normalized_history, *_closed_template_modal())
        except Exception as exc:
            dropdown_noops = tuple(gr.update() for _ in _TEMPLATE_TOOL_ORDER)
            if action == "add":
                context_label = str(Path(str(modal_state.get("source_path", "")).strip()).name or str(modal_state.get("variant_name", "")).strip() or "Unknown").strip()
                body_html = _modal_context_html("Selected WanGP Settings", context_label)
            else:
                body_html = _modal_context_html("Selected Deepy Template", str(modal_state.get("variant_name", "")).strip() or "Unknown")
            body_html += _modal_message_html(str(exc), tone="error")
            modal_updates = _open_template_modal({}, _modal_title_html(tool_name), body_html, close_visible=True)
            return (*dropdown_noops, normalized_history, *modal_updates)

    ui.use_template_properties.change(
        fn=toggle_override_controls,
        inputs=[ui.use_template_properties],
        outputs=[ui.override_height, ui.override_width, ui.override_num_frames, ui.override_seed],
        show_progress="hidden",
        queue=False,
    )
    for control in ui.template_controls:
        control.dropdown.change(
            fn=track_template_selection,
            inputs=[gr.State(control.tool_name), ui.template_selection_history, *template_dropdown_inputs],
            outputs=[ui.template_selection_history],
            show_progress="hidden",
            queue=False,
        )
        control.add_btn.click(
            fn=open_add_template_modal,
            inputs=[gr.State(control.tool_name), state, ui.captured_lset_value, ui.captured_lset_label, control.dropdown],
            outputs=template_modal_outputs,
            js="""(toolName, stateValue, _capturedValue, _capturedLabel, currentVariant) => {
                const selection = window.WAC && typeof window.WAC.getWanGpSettingsSelection === 'function'
                    ? window.WAC.getWanGpSettingsSelection()
                    : { value: '', label: '' };
                return [toolName, stateValue, selection.value || '', selection.label || '', currentVariant];
            }""",
            show_progress="hidden",
            queue=False,
        )
        control.delete_btn.click(
            fn=open_delete_template_modal,
            inputs=[gr.State(control.tool_name), control.dropdown],
            outputs=template_modal_outputs,
            show_progress="hidden",
            queue=False,
        )
    ui.template_modal_no_btn.click(fn=close_template_modal, inputs=[], outputs=template_modal_outputs, show_progress="hidden", queue=False)
    ui.template_modal_close_btn.click(fn=close_template_modal, inputs=[], outputs=template_modal_outputs, show_progress="hidden", queue=False)
    ui.template_modal_yes_btn.click(
        fn=confirm_template_modal_action,
        inputs=[ui.template_modal_state, ui.template_selection_history, *template_dropdown_inputs],
        outputs=[*template_dropdown_outputs, ui.template_selection_history, *template_modal_outputs],
        show_progress="hidden",
        queue=False,
    )
    selected_video_time_input.change(
        fn=handlers.store_selected_video_time,
        inputs=[state, selected_video_time_input],
        outputs=None,
        show_progress="hidden",
        queue=False,
    )
    ui.settings_save_btn.click(
        fn=persist_ui_settings,
        inputs=[
            state,
            ui.auto_cancel_queue_tasks,
            ui.use_template_properties,
            ui.override_height,
            ui.override_width,
            ui.override_num_frames,
            ui.override_seed,
            ui.default_video_generator,
            ui.default_video_with_speech,
            ui.default_image_generator,
            ui.default_image_editor,
            ui.default_speech_from_description,
            ui.default_speech_from_sample,
        ],
        outputs=None,
        show_progress="hidden",
    )
    ui.ask_btn.click(
        fn=ask_ai_with_ui_settings,
        inputs=[
            state,
            output,
            last_choice,
            audio_files_paths,
            audio_file_selected,
            ui.request,
            ui.auto_cancel_queue_tasks,
            ui.use_template_properties,
            ui.override_height,
            ui.override_width,
            ui.override_num_frames,
            ui.override_seed,
            ui.default_video_generator,
            ui.default_video_with_speech,
            ui.default_image_generator,
            ui.default_image_editor,
            ui.default_speech_from_description,
            ui.default_speech_from_sample,
        ],
        outputs=[ui.chat_event, load_queue_trigger, ui.request, output_trigger, abort_client_id],
        show_progress="hidden",
    )
    ui.stop_btn.click(fn=stop_ai_with_ui, inputs=[state], outputs=[ui.chat_event, load_queue_trigger, ui.request, abort_client_id], show_progress="hidden", queue=False)
    ui.reset_btn.click(fn=reset_ai_with_ui, inputs=[state], outputs=[ui.chat_event, load_queue_trigger, ui.request, abort_client_id], show_progress="hidden")


__all__ = ["DeepyChatHandlers", "DeepyChatUI", "DeepyTemplateToolControl", "bind_deepy_chat_ui", "build_deepy_chat_ui"]
