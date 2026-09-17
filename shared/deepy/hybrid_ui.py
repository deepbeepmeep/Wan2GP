"""Gradio view adapters for the shared Deepy service."""
from __future__ import annotations

import json

import gradio as gr

from shared.deepy import ui_settings
from shared.deepy.errors import DeepyBusy
from shared.gradio.form_sync import GradioForm
from shared.utils.form_sync import Saved
from shared.utils.gallery_view import gallery_window


def bind_gallery_sync(service, state, render_gallery, outputs, *, gallery, main):
    gr.HTML('<span data-deepy-hybrid="/deepy/"></span>', visible=False)
    error = gr.Textbox(visible=False, elem_id='deepy_hybrid_error')

    def show_error(payload):
        _, options = json.loads(payload)
        if options.pop('level', 'error') == 'info':
            gr.Info(options['message'], duration=options['duration'])
            return
        raise gr.Error(**options, print_exception=False)

    # Gradio only preserves custom error display options on its queued path.
    error.input(show_error, inputs=[error], outputs=None, queue=True, show_progress='hidden', trigger_mode='multiple', api_name=False)
    trigger = gr.Button(visible=False, elem_id='deepy_hybrid_gallery_sync')
    revision = gr.State(-1)
    restored = gr.Textbox(visible=False)
    view = gr.Textbox(visible=False, elem_id='wangp-gallery-view')
    interaction = gr.Textbox(visible=False, elem_id='wangp-gallery-interaction')

    def select(payload):
        service.select_gallery_view(json.loads(payload))

    interaction.input(select, inputs=[interaction], outputs=None, queue=False, show_progress='hidden', trigger_mode='always_last')

    def refresh(state_value, seen, previous_restoration):
        current = service.gallery_revision
        restoration = json.dumps(service._restoration)
        if seen == current and previous_restoration == restoration:
            return *((gr.update(),) * len(outputs)), current, restoration, gr.update()
        with service._mutation_lock:
            gen = state_value['gen']
            limit = service._deps.get_server_config()['clear_file_list']
            video, selected, video_offset = gallery_window(gen['file_list'], gen['selected'], limit)
            audio, _, audio_offset = gallery_window(gen['audio_file_list'], gen['audio_selected'], limit)
            gallery_view = json.dumps({'workspace': service.workspace_id, 'video': video, 'audio': audio, 'video_offset': video_offset, 'audio_offset': audio_offset, 'selected': selected})
            return *render_gallery(state_value), current, restoration, gallery_view

    def restore_selection(state_value):
        # Gradio resets selection to zero when an empty gallery first receives files.
        # Apply the canonical selection after that value update has been rendered.
        gen = state_value['gen']
        _, selected, _ = gallery_window(gen['file_list'], gen['selected'], service._deps.get_server_config()['clear_file_list'])
        return gr.update(selected_index=selected)

    gr.on([main.load, trigger.click], refresh, inputs=[state, revision, restored], outputs=[*outputs, revision, restored, view], queue=False, show_progress='hidden', trigger_mode='always_last').then(restore_selection, inputs=[state], outputs=[gallery], queue=False, show_progress='hidden').then(fn=None, inputs=[restored], outputs=None, js='payload => window.__wangpAssistantChatNS.galleryRestored?.(JSON.parse(payload))')

    # Reuse the existing preview renderer without requiring a generation request
    # in this page. New/reconnected pages also read the current shared preview.
    [preview] = [fn for fn in main.fns.values() if fn.name == 'refresh_preview' and fn.inputs == [state]]
    preview_trigger = gr.Button(visible=False, elem_id='deepy_hybrid_preview_sync')
    gr.on([main.load, preview_trigger.click], preview.fn, inputs=preview.inputs, outputs=preview.outputs, queue=False, show_progress='hidden', trigger_mode='always_last')


def bind_settings_sync(service, state, ui, dropdown_updates, catalog_event, preference_view):
    trigger = gr.Button(visible=False, elem_id='deepy_hybrid_settings_sync')
    revision = gr.State(-1)
    props = [ui.compact_actions, ui.auto_cancel_queue_tasks, ui.separate_requests_with_empty_line, ui.use_template_properties, ui.model_speed, ui.model_size, ui.override_height, ui.override_width, ui.override_num_frames, ui.override_audio_duration, ui.override_seed]
    keys = ['compact_actions', 'auto_cancel_queue_tasks', 'separate_requests_with_empty_line', 'use_template_properties', 'model_speed', 'model_size', 'height', 'width', 'num_frames', 'audio_duration', 'seed']
    tools = [ui.default_video_generator, ui.default_video_with_speech, ui.default_image_generator, ui.default_image_editor, ui.default_song, ui.default_with_refs, ui.default_speech_from_description, ui.default_speech_from_sample]
    tool_keys = ['video_generator_variant', 'video_with_speech_variant', 'image_generator_variant', 'image_editor_variant', 'song_variant', 'with_refs_variant', 'speech_from_description_variant', 'speech_from_sample_variant']
    preference_keys = ['multi_session', 'gallery_media_mode']
    all_keys = keys + tool_keys + preference_keys
    components = [*props, *tools, ui.multi_session, ui.session_gallery_media_mode]

    def read():
        values = ui_settings.get_persisted_assistant_tool_ui_settings(service._deps.get_server_config())
        values.update(service._session.tool_ui_settings)
        values.update(service._deps.controller.get_session_ui_settings())
        values.update(service.display_settings())
        return {key: values[key] for key in all_keys}

    def render_fields(values):
        updates, _ = dropdown_updates({tool: values[key] for tool, key in ui_settings.TEMPLATE_TOOL_UI_KEY.items()})
        return dict(zip((ui_settings.TEMPLATE_TOOL_UI_KEY[tool] for row in ui_settings.TEMPLATE_TOOL_LAYOUT for tool in row), updates))

    form = GradioForm(service.forms, 'deepy-gradio', components, keys=all_keys, read=read, scope=lambda: service._session.chat_session_id, render_fields=render_fields, view_events=[preference_view], lock=service._mutation_lock)
    outputs = [ui.session_dropdown, ui.reset_btn]

    def refresh(state_value, seen):
        current = service.settings_revision
        if seen == current:
            return *((gr.update(),) * len(outputs)), current
        _, sessions = catalog_event(service._session.storage_session_id, service._deps.controller.multi_session_enabled())
        service.forms.refresh('deepy-gradio')
        return sessions, gr.update(value='New' if service._deps.controller.multi_session_enabled() else 'Reset'), current

    trigger.click(refresh, inputs=[state, revision], outputs=[*outputs, revision], queue=False, show_progress='hidden', trigger_mode='always_last')

    for component, key in zip([*props, *tools], [*keys, *tool_keys]):
        def change(state_value, *values, _key=key):
            selected = dict(zip(all_keys, values))
            if _key == "compact_actions":
                service.update_display_settings({_key: selected[_key]})
            else:
                service.update_settings({_key: selected[_key]}, simplified=False, persist=False)
            return Saved(())
        form.bind_save(component.input, change, inputs=[state, *components], outputs=[], fields=[key], queue=False, show_progress='hidden', trigger_mode='always_last')
    return form



def bind_handlers(service, handlers):
    """Keep full Gradio settings/session management, publishing their committed changes."""
    def settings_update(*args, **kwargs):
        result = original_settings(*args, **kwargs)
        service.publish('settings', service.settings())
        service.host_changed()
        return result

    original_settings = handlers.update_tool_ui_settings
    handlers.update_tool_ui_settings = settings_update
    for name in ('update_session_ui_settings', 'rename_saved_session', 'duplicate_saved_session', 'import_saved_session', 'delete_saved_session'):
        original = getattr(handlers, name)

        def changed(*args, _handler=original, _name=name, **kwargs):
            with service._mutation_lock:
                if _name == 'delete_saved_session':
                    # An inactive session may own the gallery currently used by
                    # a Gradio generation. Deletion can now remove that workspace.
                    service.require_idle()
                result = _handler(*args, **kwargs)
            if isinstance(result, dict) and result.get('event'):
                service.publish('chat', result['event'])
            service.publish_sessions()
            service.host_changed()
            return result

        setattr(handlers, name, changed)


def control_updates(service, action, payload=None):
    try:
        service.control(action, {} if payload is None else payload)
    except DeepyBusy as exc:
        gr.Info(str(exc))
        return (gr.update(),) * 4
    return gr.update(), gr.update(), gr.update(value='') if action == 'reset' else gr.update(), gr.update()
