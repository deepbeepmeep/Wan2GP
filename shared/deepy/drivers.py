"""Adapters between Deepy's commands and its host interfaces."""
from __future__ import annotations

import time

from shared.deepy.gallery import Gallery
from shared.deepy.video_tools import get_video_thumbnail_data_url
from shared.utils.downloads import register_file_download


def gradio_chat_updates(commands, refresh_id):
    import gradio as gr

    first_publication = True
    for command, data in commands:
        outputs = [gr.update() for _ in range(5)]
        if command == "chat_output":
            outputs[0] = data if data is not None else gr.update()
            if first_publication:
                outputs[2] = gr.update(value="")
                first_publication = False
        elif command == "request_accepted":
            outputs[2] = gr.update(value="")
        elif command in {"load_queue_trigger", "refresh_gallery"}:
            outputs[1 if command == "load_queue_trigger" else 3] = str(time.time()) + "_" + str(refresh_id())
        elif command == "abort_client_id":
            outputs[4] = str(data or "")
        yield tuple(outputs)


class AppGallery(Gallery):
    """The API operates on the same media data as the CLI and Deepy tools."""

    def media_info(self, media_id):
        for audio, index, record, path in self._iter_records():
            if record["media_id"] == media_id:
                _, settings = self._resolve_lists(audio)
                return self._deps.callbacks.emit_first("format_media_info", path, settings[index])
        raise ValueError("Media not found.")

    def snapshot(self):
        gen = self._gen()
        items = []
        for audio, index, record, path in self._iter_records():
            download = register_file_download(path)
            items.append({"id": record["media_id"], "name": record["label"], "kind": record["media_type"], "url": download["url"], "selected": index == gen["audio_selected" if audio else "selected"], "active": gen["current_gallery_source"] == ("audio" if audio else "video")})
            if record["media_type"] == "video":
                items[-1]["poster"] = get_video_thumbnail_data_url(path)
        return items


class GradioGallery(AppGallery):
    """An in-process driver for a live WebUI state; rendering stays in Gradio.

    The refresh callback requests a view update, never starts generation. This
    allows the same API to attach to the WebUI when hybrid mode is enabled.
    """

    def __init__(self, deps, state, refresh):
        super().__init__(deps, state)
        self._refresh = refresh

    def add_path(self, raw_path, preferred_type="any"):
        result = super().add_path(raw_path, preferred_type)
        self._refresh()
        return result

    def select(self, reference, media_type="any"):
        result = super().select(reference, media_type)
        self._refresh()
        return result

    def sync_refresh_path(self, payload):
        super().sync_refresh_path(payload)
        self._refresh()

    def sync_latest_generated(self, before_counts):
        super().sync_latest_generated(before_counts)
        self._refresh()
