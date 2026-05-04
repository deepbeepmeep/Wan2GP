from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from . import common
from . import constants as ui_constants
from . import process_catalog as catalog
from . import prompt_schedule as prompts
from . import status_ui
from .form_controller import FormComponentValues, ProcessFormController
from .process_library import ProcessLibrary
from .process_runner import ProcessRunner


def create_config_ui(self, api_session):
    if catalog.PROCESS_DEFINITIONS_ERROR is not None:
        with gr.Blocks() as plugin_blocks:
            gr.Markdown(f"Process settings configuration error: {html.escape(catalog.PROCESS_DEFINITIONS_ERROR)}")
        return plugin_blocks
    get_model_def = self.get_model_def
    get_lora_dir = self.get_lora_dir
    get_base_model_type = self.get_base_model_type
    library = ProcessLibrary(get_model_def=get_model_def, get_lora_dir=get_lora_dir, get_base_model_type=get_base_model_type)
    output_resolution_choices = [("1080p", "1080p"), ("900p", "900p"), ("720p", "720p"), ("540p", "540p"), ("480p", "480p"), ("384p", "384p"), ("320p", "320p"), ("256p", "256p")]
    output_resolution_values = {value for _, value in output_resolution_choices}
    source_audio_track_choices = [("Auto", "")] + [(f"Audio Track {track_no}", str(track_no)) for track_no in range(1, 10)]
    source_audio_track_values = {value for _, value in source_audio_track_choices}
    ratio_values = {value for _, value in ui_constants.RATIO_CHOICES}

    form_controller = ProcessFormController(library=library, get_model_def=get_model_def, output_resolution_values=output_resolution_values, source_audio_track_values=source_audio_track_values, ratio_values=ratio_values)
    saved_ui_settings = catalog.load_saved_process_full_video_settings()
    initial_user_refs = catalog.get_saved_user_settings_refs(saved_ui_settings)
    initial_form = form_controller.build_initial_form(saved_ui_settings, self.state.value, initial_user_refs)
    default_model_type = initial_form.model_type
    default_process_choices = initial_form.process_choices
    default_process_name = initial_form.process_name
    default_state = initial_form.form_state
    active_job = {"job": None, "running": False, "cancel_requested": False, "write_state": None}
    preview_state = {"image": None}
    ui_skip = object()

    def refresh_preview(_refresh_id):
        return preview_state["image"]

    def _button_update(label: str, enabled: bool | None):
        return gr.skip() if enabled is None else gr.update(value=label, interactive=enabled)

    def _ui_update(status=ui_skip, output=ui_skip, preview_refresh=ui_skip, *, start_enabled: bool | None = None, abort_enabled: bool | None = None):
        status_update = gr.skip() if status is ui_skip else status
        output_update = gr.skip() if output is ui_skip else status_ui.render_output_file_html(output)
        preview_update = gr.skip() if preview_refresh is ui_skip else preview_refresh
        start_update = _button_update("Start Process", start_enabled)
        abort_update = _button_update("Stop", abort_enabled)
        return status_update, output_update, preview_update, start_update, abort_update

    def _info_exit(message: str, *, output=ui_skip, total_chunks: int = 1, completed_chunks: int = 0, current_chunk: int = 1, continued: bool = False):
        gr.Info(str(message or "").strip())
        return _ui_update(status_ui.render_chunk_status_html(total_chunks, completed_chunks, current_chunk, "Info", str(message or "").strip(), continued=continued), output, ui_skip, start_enabled=True, abort_enabled=False)

    def _reset_live_chunk_status(state: dict) -> None:
        gen = state.get("gen") if isinstance(state, dict) else None
        if not isinstance(gen, dict):
            return
        gen["status"] = ""
        gen["status_display"] = False
        gen["progress_args"] = None
        gen["progress_phase"] = None
        gen["progress_status"] = ""
        gen["preview"] = None

    def _add_user_process_link(process_value: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str] | None):
        if str(process_value or "").strip() == ui_constants.NO_USER_SETTINGS_VALUE:
            raise gr.Error("No user settings are available to add.")
        process_definition = library.process_definition(process_value, main_state, user_refs)
        if process_definition is None:
            raise gr.Error("The selected user settings file could not be found.")
        problems = library.validate_user_process_definition(process_definition)
        if len(problems) > 0:
            raise gr.Error(library.format_user_process_validation_error(process_definition, problems))
        ref = catalog.normalize_user_settings_ref(process_definition.get("ref"))
        if len(ref) == 0:
            raise gr.Error("The selected user settings file could not be linked.")
        refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs})
        model_label = library.model_type_label(library.process_definition_model_type(process_definition))
        if ref.casefold() not in {item.casefold() for item in refs}:
            refs.append(ref)
            catalog.store_user_settings_refs(refs)
            gr.Info(f'User settings "{process_definition.get("name")}" have been added for {model_label}.')
        else:
            gr.Info(f'User settings "{process_definition.get("name")}" are already linked for {model_label}.')
        process_choices, selected = library.current_user_settings_choices(main_state, main_lset_name)
        if str(process_value or "").strip() in {value for _label, value in process_choices}:
            selected = str(process_value or "").strip()
        return (
            refs,
            gr.update(choices=library.model_type_choices(refs), value=ui_constants.ADD_USER_SETTINGS_MODEL_TYPE),
            gr.update(choices=process_choices, value=selected),
            form_controller.user_settings_hint_update(process_choices),
            selected,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _delete_user_process_link(memory_state: dict | None, process_value: str, main_state: dict | None, user_refs: list[str] | None, source_path: str):
        process_value = str(process_value or "").strip()
        ref = catalog.user_process_ref_from_value(process_value)
        if len(ref) == 0:
            raise gr.Error("Choose a linked user settings process to remove.")
        old_refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs})
        deleted_definition = library.build_user_process_definition(ref)
        deleted_model_type = library.process_definition_model_type(deleted_definition)
        new_refs = [item for item in old_refs if item.casefold() != ref.casefold()]
        catalog.store_user_settings_refs(new_refs)
        model_type, next_process_name, process_choices = library.select_after_user_process_delete(process_value, deleted_model_type, old_refs, new_refs)
        restored = form_controller.restore_state(memory_state, next_process_name, source_path, main_state, new_refs)
        gr.Info(f'Removed user settings "{Path(ref).stem}".')
        action_updates = form_controller.settings_action_updates(model_type, next_process_name)
        return (
            new_refs,
            gr.update(choices=library.model_type_choices(new_refs), value=model_type),
            gr.update(choices=process_choices, value=next_process_name),
            form_controller.user_settings_hint_update(process_choices),
            next_process_name,
            *action_updates,
            *restored,
        )

    model_type_choices = initial_form.model_type_choices
    initial_process_form_memory = {default_process_name: default_state.to_dict()}

    process_runner = ProcessRunner(
        plugin=self,
        api_session=api_session,
        library=library,
        get_model_def=get_model_def,
        active_job=active_job,
        preview_state=preview_state,
        ui_skip=ui_skip,
        ui_update=_ui_update,
        info_exit=_info_exit,
        reset_live_chunk_status=_reset_live_chunk_status,
    )

    def stop_process():
        active_job["cancel_requested"] = True
        write_state = active_job.get("write_state")
        if write_state is not None:
            write_state.stopped = True
        job = active_job.get("job")
        if job is not None and not job.done:
            job.cancel()
            common.plugin_info("Stopping current processing job...")
            return gr.update(value="Start Process", interactive=False), gr.update(value="Stop", interactive=False)
        if active_job.get("running"):
            return gr.update(value="Start Process", interactive=False), gr.update(value="Stop", interactive=False)
        return gr.update(value="Start Process", interactive=True), gr.update(value="Stop", interactive=False)

    def _form_values(source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        return FormComponentValues(source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)

    def _store_memory(memory_state, current_process_name, main_state, refs, source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        values = _form_values(source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)
        return form_controller.store_memory(memory_state, current_process_name, main_state, refs, values)

    def _change_process_model_type(memory_state, current_process_name, next_model_type, main_state, main_lset_name, refs, source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        values = _form_values(source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)
        return form_controller.change_process_model_type(memory_state, current_process_name, next_model_type, main_state, main_lset_name, refs, values)

    def _change_process_name(memory_state, current_process_name, next_process_name, process_model_type_value, main_state, refs, source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        values = _form_values(source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)
        return form_controller.change_process_name(memory_state, current_process_name, next_process_name, process_model_type_value, main_state, refs, values)

    def _refresh_from_main(refresh_id, memory_state, current_process_name, process_model_type_value, main_state, main_lset_name, refs, source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        values = _form_values(source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)
        return form_controller.refresh_from_main(refresh_id, memory_state, current_process_name, process_model_type_value, main_state, main_lset_name, refs, values)

    process_form_memory = gr.State(initial_process_form_memory)
    active_process_name_state = gr.State(default_process_name)
    user_process_refs = gr.State(initial_user_refs)
    with gr.Column():
        gr.HTML(
            """
            <style>
            #process-full-video-settings-actions {
                align-self: flex-end !important;
                margin-bottom: 1px;
                padding-bottom: 4px !important;
                gap: 4px;
                width: 34px !important;
                min-width: 34px !important;
                max-width: 34px !important;
            }
            #process-full-video-settings-actions > .form {
                padding: 0 !important;
                border: 0 !important;
                background: transparent !important;
                box-shadow: none !important;
            }
            #process-full-video-settings-actions button {
                width: 34px !important;
                min-width: 34px !important;
                max-width: 34px !important;
                height: 34px;
                min-height: 34px;
                padding: 0 !important;
            }
            #process-full-video-settings-actions .process-full-video-settings-action-placeholder {
                width: 34px;
                min-width: 34px;
                max-width: 34px;
                height: 34px;
                min-height: 34px;
            }
            #process-full-video-user-settings-hint-row {
                height: 12px !important;
                min-height: 0 !important;
                max-height: 12px !important;
                margin-top: -10px !important;
                margin-bottom: -4px !important;
                padding: 0 !important;
                overflow: visible !important;
            }
            #process-full-video-user-settings-hint-row > .form {
                padding: 0 !important;
                border: 0 !important;
                background: transparent !important;
                box-shadow: none !important;
                min-height: 0 !important;
                overflow: visible !important;
            }
            #process-full-video-user-settings-hint-row .block,
            #process-full-video-user-settings-hint-row .html-container,
            #process-full-video-user-settings-hint-row .prose {
                height: auto !important;
                margin: 0 !important;
                min-height: 0 !important;
                padding: 0 !important;
                overflow: visible !important;
            }
            </style>
            """
        )
        with gr.Row():
            gr.Markdown("This PlugIn is a *Super Sliding Windows* mode with *Low RAM requirements*, lossless Audio Copy and no risk to explode your Web Browser and the *Video Gallery* with huge files. You can stop a Process and Resume it later. You can define different prompts for different time range. However quite often the prompt should have little impact on the ouput.")
        with gr.Row():
            process_model_type = gr.Dropdown(model_type_choices, value=default_model_type, label="Model", scale=1)
            process_name = gr.Dropdown(default_process_choices, value=default_process_name, label="Process", scale=3)
            with gr.Column(scale=0, min_width=34, elem_id="process-full-video-settings-actions"):
                add_user_settings_btn = gr.Button("\u2795", size="sm", min_width=1, visible=default_model_type == ui_constants.ADD_USER_SETTINGS_MODEL_TYPE, elem_classes=["wangp-assistant-chat__template-tool-icon-btn"])
                delete_user_settings_btn = gr.Button("\U0001F5D1\uFE0F", size="sm", min_width=1, visible=catalog.is_user_process_value(default_process_name), elem_classes=["wangp-assistant-chat__template-tool-icon-btn", "wangp-assistant-chat__template-tool-icon-btn--danger"])
                settings_actions_placeholder = gr.HTML("<div class='process-full-video-settings-action-placeholder'></div>", visible=default_model_type != ui_constants.ADD_USER_SETTINGS_MODEL_TYPE and not catalog.is_user_process_value(default_process_name))
        with gr.Row(visible=library.process_choices_have_user_settings(default_process_choices), elem_id="process-full-video-user-settings-hint-row") as process_user_settings_hint_row:
            gr.HTML(value=ui_constants.USER_SETTINGS_HINT_HTML)
        with gr.Row():
            source_path = gr.Textbox(label="Source Video Path File", value=default_state.source_path, scale=3)
        with gr.Row():
            output_path = gr.Textbox(label="Output File Path File (None for auto, Full Name or Target Folder)", value=default_state.output_path, scale=3)
            continue_enabled = gr.Checkbox(label="Continue", value=default_state.continue_enabled, elem_classes="cbx_bottom", scale=1)
        with gr.Row():
            output_resolution = gr.Dropdown(output_resolution_choices, value=default_state.output_resolution, label="Output Resolution")
            target_ratio = gr.Dropdown(ui_constants.RATIO_CHOICES if initial_form.target_ratio_visible else ui_constants.RATIO_CHOICES_WITH_EMPTY, value=default_state.target_ratio if initial_form.target_ratio_visible else "", label="Target Ratio", visible=initial_form.target_ratio_visible)
            default_process_strength = 1.0 if initial_form.target_ratio_visible else default_state.process_strength
            process_strength = gr.Slider(label="Process Strength (LoRA Multiplier)", minimum=min(0.0, default_process_strength), maximum=max(3.0, default_process_strength), step=0.01, value=default_process_strength, visible=initial_form.process_strength_visible)
        with gr.Row():
            chunk_size_seconds = gr.Number(label="Chunk Size (seconds)", value=default_state.chunk_size_seconds, precision=2)
            sliding_window_overlap = gr.Slider(label="Sliding Window Overlap", minimum=1, maximum=initial_form.overlap_max, step=initial_form.overlap_step, value=default_state.sliding_window_overlap)
        with gr.Row():
            start_seconds = gr.Textbox(label="Start (s/MM:SS(.xx)/HH:MM:SS(.xx))", value=default_state.start_seconds, placeholder="seconds, MM:SS(.xx), or HH:MM:SS(.xx)")
            end_seconds = gr.Textbox(label="End (s/MM:SS(.xx)/HH:MM:SS(.xx))", value=default_state.end_seconds, placeholder="seconds, MM:SS(.xx), or HH:MM:SS(.xx)")
            source_audio_track = gr.Dropdown(source_audio_track_choices, value=default_state.source_audio_track, label="Source Audio Track")
        with gr.Row():
            prompt_text = gr.Textbox(
                label="Prompt (timed blocks supported: MM:SS(.xx) / HH:MM:SS(.xx))",
                value=default_state.prompt,
                lines=1,
                placeholder=prompts.TIMED_PROMPT_EXAMPLE,
            )
        with gr.Row():
            start_btn = gr.Button("Start Process")
            abort_btn = gr.Button("Stop", interactive=False)
        status_html = gr.HTML(value=status_ui.render_chunk_status_html(0, 0, 0, "Idle", "Waiting to start..."))
        preview_image = gr.Image(label="Last Frame Preview", type="pil")
        output_file = gr.HTML(value=status_ui.render_output_file_html(""))
        preview_refresh = gr.Textbox(value="", visible=False)
        tab_refresh_trigger = gr.Textbox(value="", visible=False)

    self.on_tab_outputs = [tab_refresh_trigger]

    gr.on(
        [
            source_path.change,
            process_strength.change,
            output_path.change,
            prompt_text.change,
            continue_enabled.change,
            source_audio_track.change,
            output_resolution.change,
            target_ratio.change,
            chunk_size_seconds.change,
            sliding_window_overlap.change,
            start_seconds.change,
            end_seconds.change,
        ],
        fn=_store_memory,
        inputs=[process_form_memory, active_process_name_state, self.state, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_form_memory],
        queue=False,
        show_progress="hidden",
    )
    process_model_type.change(
        fn=_change_process_model_type,
        inputs=[process_form_memory, active_process_name_state, process_model_type, self.state, self.lset_name, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_form_memory, active_process_name_state, process_name, process_user_settings_hint_row, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        queue=False,
        show_progress="hidden",
    )
    process_name.change(
        fn=_change_process_name,
        inputs=[process_form_memory, active_process_name_state, process_name, process_model_type, self.state, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_form_memory, active_process_name_state, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        queue=False,
        show_progress="hidden",
    )
    add_user_settings_btn.click(
        fn=_add_user_process_link,
        inputs=[process_name, self.state, self.lset_name, user_process_refs],
        outputs=[user_process_refs, process_model_type, process_name, process_user_settings_hint_row, active_process_name_state, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder],
        show_progress="hidden",
    )
    delete_user_settings_btn.click(
        fn=_delete_user_process_link,
        inputs=[process_form_memory, process_name, self.state, user_process_refs, source_path],
        outputs=[user_process_refs, process_model_type, process_name, process_user_settings_hint_row, active_process_name_state, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        show_progress="hidden",
    )
    self.refresh_form_trigger.change(
        fn=_refresh_from_main,
        inputs=[self.refresh_form_trigger, process_form_memory, active_process_name_state, process_model_type, self.state, self.lset_name, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_model_type, process_name, process_user_settings_hint_row, process_form_memory, active_process_name_state, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        queue=False,
        show_progress="hidden",
    )
    tab_refresh_trigger.change(
        fn=_refresh_from_main,
        inputs=[tab_refresh_trigger, process_form_memory, active_process_name_state, process_model_type, self.state, self.lset_name, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_model_type, process_name, process_user_settings_hint_row, process_form_memory, active_process_name_state, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        queue=False,
        show_progress="hidden",
    )
    start_btn.click(
        fn=process_runner.start_process,
        inputs=[self.state, process_name, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[status_html, output_file, preview_refresh, start_btn, abort_btn],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    preview_refresh.change(fn=refresh_preview, inputs=[preview_refresh], outputs=[preview_image], queue=False, show_progress="hidden")
    abort_btn.click(fn=stop_process, outputs=[start_btn, abort_btn], queue=False, show_progress="hidden")
