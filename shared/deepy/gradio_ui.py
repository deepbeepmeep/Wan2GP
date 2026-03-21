from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr

from shared.deepy import ui_settings as deepy_ui_settings
from shared.gradio import assistant_chat


@dataclass(slots=True)
class DeepyChatUI:
    dock: Any
    launcher_host: Any
    panel: Any
    settings_launcher_host: Any
    html_output: Any
    chat_event: Any
    stop_btn: Any
    request: Any
    ask_btn: Any
    reset_btn: Any
    use_template_properties: Any
    priority: Any
    override_height: Any
    override_width: Any
    override_num_frames: Any
    default_image_generator: Any
    default_image_editor: Any
    default_video_generator: Any
    refresh_presets_btn: Any


@dataclass(slots=True)
class DeepyChatHandlers:
    prepare_request_context: Callable[[Any, Any, Any, Any, Any], Any]
    update_tool_ui_settings: Callable[..., Any]
    store_selected_video_time: Callable[[Any, Any], Any]
    ask_ai: Callable[[Any, str], Any]
    stop_ai: Callable[[Any], Any]
    reset_ai: Callable[[Any], Any]


def build_deepy_chat_ui(*, deepy_visible: bool) -> DeepyChatUI:
    template_selector_state = deepy_ui_settings.get_template_selector_state()
    tool_ui_state = deepy_ui_settings.get_persisted_assistant_tool_ui_settings()
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
            with gr.Column(elem_id=assistant_chat.SETTINGS_PANEL_ID):
                with gr.Column(elem_classes=["wangp-assistant-chat__settings-scroll"]):
                    with gr.Accordion("Generation Properties", open=True):
                        use_template_properties = gr.Checkbox(value=tool_ui_state["use_template_properties"], label="Use Properties defined in Settings Templates files.")
                        priority = gr.Checkbox(value=False, label="Priority queue insertion (place next after current task).")
                        with gr.Row():
                            override_width = gr.Slider(
                                deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MIN,
                                deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MAX,
                                value=tool_ui_state["width"],
                                step=deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_STEP,
                                label="Width",
                                interactive=not tool_ui_state["use_template_properties"],
                            )
                            override_height = gr.Slider(
                                deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MIN,
                                deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_MAX,
                                value=tool_ui_state["height"],
                                step=deepy_ui_settings.ASSISTANT_OVERRIDE_DIMENSION_STEP,
                                label="Height",
                                interactive=not tool_ui_state["use_template_properties"],
                            )
                        override_num_frames = gr.Slider(
                            deepy_ui_settings.ASSISTANT_OVERRIDE_FRAMES_MIN,
                            deepy_ui_settings.ASSISTANT_OVERRIDE_FRAMES_MAX,
                            value=tool_ui_state["num_frames"],
                            step=1,
                            label="Number of Frames",
                            interactive=not tool_ui_state["use_template_properties"],
                        )
                    with gr.Accordion("Template Settings used by Tools", open=False):
                        default_image_generator = gr.Dropdown(
                            choices=template_selector_state["image_generator_choices"],
                            value=tool_ui_state["image_generator_variant"],
                            label="Image Generator",
                        )
                        default_image_editor = gr.Dropdown(
                            choices=template_selector_state["image_editor_choices"],
                            value=tool_ui_state["image_editor_variant"],
                            label="Image Editor",
                        )
                        default_video_generator = gr.Dropdown(
                            choices=template_selector_state["video_generator_choices"],
                            value=tool_ui_state["video_generator_variant"],
                            label="Video Generator",
                        )
                        refresh_presets_btn = gr.Button("Refresh Presets", size="sm")
    return DeepyChatUI(
        dock=dock,
        launcher_host=launcher_host,
        panel=panel,
        settings_launcher_host=settings_launcher_host,
        html_output=html_output,
        chat_event=chat_event,
        stop_btn=stop_btn,
        request=request,
        ask_btn=ask_btn,
        reset_btn=reset_btn,
        use_template_properties=use_template_properties,
        priority=priority,
        override_height=override_height,
        override_width=override_width,
        override_num_frames=override_num_frames,
        default_image_generator=default_image_generator,
        default_image_editor=default_image_editor,
        default_video_generator=default_video_generator,
        refresh_presets_btn=refresh_presets_btn,
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
    handlers: DeepyChatHandlers,
) -> None:
    def refresh_preset_dropdowns(current_image_generator, current_image_editor, current_video_generator):
        refreshed = deepy_ui_settings.refresh_template_selector_state(current_image_generator, current_image_editor, current_video_generator)
        return (
            gr.update(choices=refreshed["image_generator_choices"], value=refreshed["selected_image_generator"]),
            gr.update(choices=refreshed["image_editor_choices"], value=refreshed["selected_image_editor"]),
            gr.update(choices=refreshed["video_generator_choices"], value=refreshed["selected_video_generator"]),
        )

    def toggle_override_controls(use_template_properties):
        interactive = not deepy_ui_settings.normalize_assistant_use_template_properties(use_template_properties)
        return gr.update(interactive=interactive), gr.update(interactive=interactive), gr.update(interactive=interactive)

    def ask_ai_with_ui_settings(
        state_value,
        output_value,
        last_choice_value,
        audio_files_paths_value,
        audio_file_selected_value,
        ask_request,
        use_template_properties,
        priority,
        override_height,
        override_width,
        override_num_frames,
        default_image_generator,
        default_image_editor,
        default_video_generator,
    ):
        handlers.prepare_request_context(state_value, output_value, last_choice_value, audio_files_paths_value, audio_file_selected_value)
        handlers.update_tool_ui_settings(
            state_value,
            use_template_properties=use_template_properties,
            priority=priority,
            width=override_width,
            height=override_height,
            num_frames=override_num_frames,
            image_generator_variant=default_image_generator,
            image_editor_variant=default_image_editor,
            video_generator_variant=default_video_generator,
            persist=True,
        )
        yield from handlers.ask_ai(state_value, ask_request)

    def stop_ai_with_ui(state_value):
        return handlers.stop_ai(state_value)

    def reset_ai_with_ui(state_value):
        return handlers.reset_ai(state_value)

    ui.use_template_properties.change(
        fn=toggle_override_controls,
        inputs=[ui.use_template_properties],
        outputs=[ui.override_height, ui.override_width, ui.override_num_frames],
        show_progress="hidden",
        queue=False,
    )
    ui.refresh_presets_btn.click(
        fn=refresh_preset_dropdowns,
        inputs=[ui.default_image_generator, ui.default_image_editor, ui.default_video_generator],
        outputs=[ui.default_image_generator, ui.default_image_editor, ui.default_video_generator],
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
    ui.ask_btn.click(
        fn=ask_ai_with_ui_settings,
        inputs=[
            state,
            output,
            last_choice,
            audio_files_paths,
            audio_file_selected,
            ui.request,
            ui.use_template_properties,
            ui.priority,
            ui.override_height,
            ui.override_width,
            ui.override_num_frames,
            ui.default_image_generator,
            ui.default_image_editor,
            ui.default_video_generator,
        ],
        outputs=[ui.chat_event, load_queue_trigger, ui.request, output_trigger],
        show_progress="hidden",
    )
    ui.stop_btn.click(fn=stop_ai_with_ui, inputs=[state], outputs=[ui.chat_event, load_queue_trigger, ui.request], show_progress="hidden", queue=False)
    ui.reset_btn.click(fn=reset_ai_with_ui, inputs=[state], outputs=[ui.chat_event, load_queue_trigger, ui.request], show_progress="hidden")


__all__ = ["DeepyChatHandlers", "DeepyChatUI", "bind_deepy_chat_ui", "build_deepy_chat_ui"]
