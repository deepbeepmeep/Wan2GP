"""Gradio view adapters for the shared Deepy service."""
from __future__ import annotations

import json

import gradio as gr

from shared.deepy import ui_settings
from shared.deepy.errors import DeepyBusy
from shared.gradio.form_sync import GradioForm
from shared.gradio.gallery_frames import bind_gallery_frames
from shared.utils.form_sync import Saved
from shared.utils.gallery_view import gallery_offset, gallery_window


def bind_workspace_extract(service, state, validate_prompt, prompt_inputs, save_inputs, save_values, use_settings, outputs):
    from shared.deepy.workspace_viewer_api import ViewerAction

    trigger = gr.Textbox(visible=False, elem_id='wangp-workspace-extract-settings')

    def extract(payload, state_value):
        try:
            selection = ViewerAction.model_validate_json(payload)
            if len(selection.keys) != 1:
                raise ValueError('Select one media item to extract its settings.')
            with service._mutation_lock:
                with service.gallery_lock:
                    viewer = service.workspace_viewer
                    entry = viewer._entry(selection.workspace, selection.source, selection.keys[0])
                    if selection.revision != viewer._catalog(selection.source)['revision']:
                        raise ValueError('The gallery changed. Reopen the workspace manager and select the media again.')
                # use_settings acquires the same non-reentrant gallery lock itself.
                # The original action reads the full lists from state; audio still needs a packed input.
                return use_settings(state_value, '[]' if selection.source == 'audio' else [], entry['index'], selection.source)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    trigger.input(validate_prompt, inputs=prompt_inputs, outputs=[prompt_inputs[3]], show_progress='hidden').then(
        save_inputs, inputs=save_values, outputs=None, show_progress='hidden',
    ).then(extract, inputs=[trigger, state], outputs=outputs, show_progress='hidden')


def bind_gallery_sync(service, state, render_gallery, outputs, *, gallery, main):
    bind_gallery_frames(service)
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
    # A stale response can be discarded by the browser's selection guard. Only
    # acknowledge revisions applied there, not responses completed on the server.
    revision = gr.Number(-1, visible=False, precision=0)
    restored = gr.Textbox(visible=False)
    view = gr.Textbox(visible=False, elem_id='wangp-gallery-view')
    interaction = gr.Textbox(visible=False, elem_id='wangp-gallery-interaction')

    def select(payload, state_value):
        payload = json.loads(payload)
        with service._mutation_lock:
            if payload['sequence'] <= state_value.get('gallery_interaction_sequence', 0):
                return
            state_value['gallery_interaction_sequence'] = payload['sequence']
            service.select_gallery_view(payload)

    interaction.input(select, inputs=[interaction, state], outputs=None, queue=False, show_progress='hidden', trigger_mode='always_last')

    def refresh(state_value, seen, previous_restoration, previous_view):
        current = service.gallery_revision
        restoration = json.dumps(service._restoration)
        if seen == current and previous_restoration == restoration:
            return *((gr.update(),) * len(outputs)), current, restoration, gr.update()
        with service._mutation_lock:
            updates = list(render_gallery(state_value))
            gen = state_value['gen']
            limit = service._deps.get_server_config()['clear_file_list']
            video, selected, video_offset = gallery_window(gen['file_list'], gen['selected'], limit, keep_selected=True)
            if video_offset != gallery_offset(len(gen['file_list']), limit):
                updates[outputs.index(gallery)] = gr.update(value=video, selected_index=selected)
            audio, _, audio_offset = gallery_window(gen['audio_file_list'], gen['audio_selected'], limit)
            gallery_view = {'workspace': service.workspace_id, 'video': video, 'audio': audio, 'video_offset': video_offset, 'audio_offset': audio_offset, 'selected': selected, 'audio_selected': gen['audio_selected']}
            gallery_view['gallery_sequence'] = state_value.get('gallery_interaction_sequence', 0)
            previous = json.loads(previous_view) if previous_view else {}
            if previous.get('workspace') == gallery_view['workspace']:
                # Selection does not change the files. Avoid Gallery.postprocess
                # and an audio render on every visual thumbnail click.
                if previous.get('video') == video:
                    updates[outputs.index(gallery)] = gr.update(selected_index=selected)
                if all(previous.get(key) == gallery_view[key] for key in ('audio', 'audio_offset', 'audio_selected')):
                    updates[4:7] = [gr.skip()] * 3
            return *updates, current, restoration, json.dumps(gallery_view)

    # The patched Gallery applies explicit indices with the value, including its
    # first population. A second response can restore an already obsolete index.
    gr.on([main.load, trigger.click], refresh, inputs=[state, revision, restored, view], outputs=[*outputs, revision, restored, view], queue=False, show_progress='hidden', trigger_mode='always_last').success(fn=None, inputs=[restored, revision], outputs=None, js='(payload, revision) => window.__wangpAssistantChatNS.galleryRestored?.(JSON.parse(payload), revision)')

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
