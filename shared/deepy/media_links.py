"""Rebind saved chat media URLs to the current HTTP host's file registry."""
from __future__ import annotations

import html
import os
import re
import urllib.parse

from shared.utils.downloads import register_file_download, register_gallery_download
from shared.utils.gallery_media import gallery_media_ids


def restore_media_links(transcript, replay_commands, media_records):
    paths, replacements, thumbnails = {}, {}, {}

    def register(path):
        if not path or not os.path.isfile(path):
            return None
        key = os.path.normcase(os.path.abspath(path))
        url = register_file_download(path)["url"]
        paths[key] = url
        return url

    for record in media_records:
        if register(record["path"]):
            gallery = "audio" if record["media_type"] == "audio" else "visual"
            for media_id in gallery_media_ids(record["path"], gallery, record.get("settings")):
                register_gallery_download(media_id, record["path"])

    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for old, path in value.get("download_links", {}).items():
                url = register(path)
                if url:
                    replacements[old] = url
            if "href" in value and "path_key" in value:
                path = value.get("path") or value["path_key"]
                url = register(path)
                if url:
                    replacements[value["href"]] = url
                    if value.get("kind") in {"image", "video"} and value.get("thumb_url"):
                        thumbnails[value['thumb_url']] = url + '/thumbnail'
                    elif value.get("kind") in {"audio", "archive"} and value.get("thumb_url"):
                        from shared.deepy.chat import _AUDIO_THUMBNAIL_PATH, _ARCHIVE_THUMBNAIL_PATH
                        thumbnail = register(_AUDIO_THUMBNAIL_PATH if value["kind"] == "audio" else _ARCHIVE_THUMBNAIL_PATH)
                        if thumbnail:
                            replacements[value["thumb_url"]] = thumbnail
            for item in value.values():
                collect(item)

    collect(transcript)

    def legacy_url(match):
        path = urllib.parse.unquote(html.unescape(match.group(1)))
        return paths.get(os.path.normcase(os.path.abspath(path)), match.group(0))

    legacy = re.compile(r'/gradio_api/file(?:=|%3[Dd])([^\s\"\'<>)]+)')
    # Rewrite all links in one pass. Repeated str.replace for every attachment
    # made restoring a long, media-rich transcript quadratic in its size.
    replacements = {old: new for old, new in replacements.items() if old and old != new}
    links = re.compile('|'.join(re.escape(old) for old in sorted(replacements, key=len, reverse=True))) if replacements else None
    image_src = re.compile(r'(<img\b[^>]*?\bsrc=[\"\'])([^\"\']+)([\"\'])', re.IGNORECASE)

    def rewrite(value):
        if isinstance(value, dict):
            return {replacements.get(key, key): thumbnails[item] if key == 'thumb_url' and item in thumbnails else rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            # Old image cards used the very same URL for href and src. Only
            # the preview becomes a thumbnail; opening keeps the original.
            value = image_src.sub(lambda match: match[1] + thumbnails.get(html.unescape(match[2]), match[2]) + match[3], value)
            if links is not None:
                value = links.sub(lambda match: replacements[match[0]], value)
            return legacy.sub(legacy_url, value)
        return value

    transcript[:] = rewrite(transcript)
    replay_commands[:] = rewrite(replay_commands)
