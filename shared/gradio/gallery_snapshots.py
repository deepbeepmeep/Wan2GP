# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Add a snapshot of the 3D viewer to the image gallery, so that a 3D asset view can feed image and video models."""
import base64
import binascii
import io
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from shared.utils.audio_video import save_image_metadata
from shared.utils.gallery_sources import MESH

_PNG_PREFIX = "data:image/png;base64,"
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


def add_mesh_snapshot(service, payload):
    data = payload['image']
    if not isinstance(data, str) or not data.startswith(_PNG_PREFIX) or len(data) > _MAX_SNAPSHOT_BYTES:
        raise ValueError('Invalid snapshot image.')
    try:
        image = Image.open(io.BytesIO(base64.b64decode(data[len(_PNG_PREFIX):], validate=True)))
        image.load()
    except (binascii.Error, OSError) as error:
        raise ValueError('Invalid snapshot image.') from error
    with service._mutation_lock:
        if payload['workspace'] != service.workspace_id:
            raise ValueError('The workspace changed. Please try again.')
        gen = service._state['gen']
        source = payload['path']
        if source not in gen[MESH.files_key]:
            raise ValueError('The 3D asset is no longer in this gallery.')
        # A rendered view, not a generation: the settings of the 3D asset do not apply to it, only a comment names its source
        # (the WanGP type keeps this metadata when the gallery reads it back from the file)
        metadata = {'type': 'WanGP DeepBeepMeep - 3D Viewer Snapshot', 'comments': f'Snapshot of "{Path(source).name}"'}
        deps = service._deps.controller._deps
        name = f'{Path(source).stem}_snapshot.png'
        # Exclusive creation keeps WanGP's naming policy safe against another writer.
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
                raise OSError('Could not save snapshot metadata.')
        except Exception:
            output.unlink(missing_ok=True)
            raise
        finally:
            image.close()
        deps.record_file_metadata(str(output), metadata, True, False, gen, notify_generation=False, write_metadata=False, record_notification=False)
        gen['last_selected'] = False
        service.save_workspace()
        service.host_changed()
        return output


def bind_gallery_snapshots(service):
    request = gr.Textbox(visible=False, elem_id='wangp-gallery-add-snapshot')

    def add(payload):
        try:
            output = add_mesh_snapshot(service, json.loads(payload))
        except (ValueError, OSError) as error:
            raise gr.Error(str(error), print_exception=False) from error
        gr.Info(f'Added {output.name} to the Image Gallery.')

    request.input(add, inputs=[request], outputs=None, queue=True, show_progress='hidden', trigger_mode='multiple', api_name=False)
