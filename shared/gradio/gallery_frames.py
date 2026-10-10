"""Save a gallery video's selected frame without changing gallery selection."""
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import time

import gradio as gr

from shared.deepy.video_tools import resolve_video_frame_no
from shared.utils.audio_video import save_image_metadata
from shared.utils.utils import get_video_frame


def add_gallery_frame(service, payload):
    seconds = float(payload['time'])
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid video time.')
    with service._mutation_lock:
        if payload['workspace'] != service.workspace_id:
            raise ValueError('The workspace changed. Please try again.')
        gen = service._state['gen']
        source = payload['path']
        if source not in gen['file_list'] or service.gallery._detect_media_type(source) != 'video':
            raise ValueError('The source video is no longer in this gallery.')
        index = gen['file_list'].index(source)
        frame_no = resolve_video_frame_no(source, time_seconds=seconds)
        image = get_video_frame(source, frame_no, return_PIL=True)
        metadata = deepcopy(gen['file_settings_list'][index] or {})
        for key in ('gallery_media_ids', 'deepy_session_id', 'deepy_media_id', 'deepy_media_fingerprint', 'client_id'):
            metadata.pop(key, None)
        now = time.time()
        metadata.update(image_mode=1, video_length=1, resolution=f'{image.width}x{image.height}', source_video=source, source_frame_no=frame_no, source_time_seconds=seconds, creation_timestamp=int(now), creation_date=datetime.fromtimestamp(now).isoformat(timespec='seconds'), comments=f'Frame {frame_no} (zero-based) at {seconds:.3f}s from "{Path(source).name}"')
        deps = service._deps.controller._deps
        name = f'{Path(source).stem}_frame{frame_no}.png'
        # Reuse the configured image output directory and WanGP's naming policy;
        # exclusive creation also protects against another writer winning the race.
        while True:
            output = Path(deps.get_output_filepath(name, True, False))
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                writer = output.open('xb')
                break
            except FileExistsError:
                continue
        try:
            with writer:
                image.save(writer, format='PNG')
            if not save_image_metadata(str(output), metadata):
                raise OSError('Could not save frame metadata.')
        except Exception:
            output.unlink(missing_ok=True)
            raise
        finally:
            image.close()
        deps.record_file_metadata(str(output), metadata, True, False, gen, notify_generation=False, write_metadata=False, record_notification=False)
        # Appending must not activate the shared "follow the latest output" mode.
        gen['last_selected'] = False
        service.save_workspace()
        service.host_changed()
        return output


def bind_gallery_frames(service):
    request = gr.Textbox(visible=False, elem_id='wangp-gallery-add-frame')

    def add(payload):
        try:
            output = add_gallery_frame(service, json.loads(payload))
        except (ValueError, OSError) as error:
            raise gr.Error(str(error), print_exception=False) from error
        gr.Info(f'Added {output.name} to the Image Gallery.')

    request.input(add, inputs=[request], outputs=None, queue=True, show_progress='hidden', trigger_mode='multiple', api_name=False)
