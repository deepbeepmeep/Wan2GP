"""Paged native workspace management, independent of Gradio component state."""
import hashlib
import json
import os
import zipfile
from urllib.parse import urlencode
from collections import Counter
from pathlib import Path

from shared.utils.downloads import register_download, register_file_download, stream_writer
from shared.utils.gallery_view import gallery_offset
from shared.utils.config_store import update_config
from shared.utils.workspaces import ARCHIVE_AFTER_DAYS_KEY, ARCHIVE_PERIODS


def retain_entries(gallery, prefix, indices, *, inplace=False):
    paths = gallery[prefix + 'file_list']
    selected = gallery[prefix + 'selected']
    retained = [paths[i] for i in indices]
    if inplace:
        paths[:] = retained
    else:
        gallery[prefix + 'file_list'] = retained
    settings_key = prefix + 'file_settings_list'
    if settings_key in gallery:
        # Reordering/ejecting entries must not read their media metadata.
        retained_settings = type(gallery[settings_key])()
        retained_settings.extend(list.__getitem__(gallery[settings_key], i) for i in indices)
        if inplace:
            gallery[settings_key][:] = list.__iter__(retained_settings)
        else:
            gallery[settings_key] = retained_settings
    gallery[prefix + 'selected'] = indices.index(selected) if selected in indices else min(selected, len(indices) - 1)
    gallery[prefix + 'last_selected'] = gallery[prefix + 'selected'] == len(indices) - 1


class WorkspaceViewer:
    page_size = 60

    def __init__(self, service):
        self.service = service
        self._cache = {}

    def retention(self, days=None):
        with self.service._mutation_lock:
            config = self.service._deps.get_server_config()
            if days is not None:
                if days not in {value for _, value in ARCHIVE_PERIODS}:
                    raise ValueError('Choose a supported workspace archive period.')
                filename = self.service._deps.controller._deps.get_server_config_filename()
                update_config(config, filename, {ARCHIVE_AFTER_DAYS_KEY: days})
            return {'days': config.get(ARCHIVE_AFTER_DAYS_KEY, 0), 'choices': ARCHIVE_PERIODS}

    def protect(self, workspace, protected):
        with self.service._mutation_lock:
            if workspace != self.service.workspace_id:
                raise ValueError('The active workspace changed. Please try again.')
            self.service.workspaces.protect(workspace, protected)
            self.service.publish_workspaces()
            return {'archive_protected': protected}

    def _catalog(self, source):
        if source not in ('video', 'audio'):
            raise ValueError('Choose the image/video or audio gallery.')
        prefix = 'audio_' if source == 'audio' else ''
        gen = self.service._state['gen']
        paths = gen[prefix + 'file_list']
        signature = (self.service.workspace_id, tuple(paths), self.service._deps.get_server_config()['clear_file_list'])
        cached = self._cache.get(source)
        if cached is None or cached['signature'] != signature:
            occurrences = Counter()
            entries = []
            for index, path in enumerate(paths):
                occurrences[path] += 1
                key = hashlib.sha256((path + '\0' + str(occurrences[path])).encode()).hexdigest()[:24]
                entries.append({'key': key, 'index': index, 'path': path})
            cached = {'signature': signature, 'revision': hashlib.sha256(repr(signature).encode()).hexdigest(), 'entries': entries, 'by_key': {entry['key']: entry for entry in entries}, 'prefix': prefix}
            self._cache[source] = cached
        return cached

    def page(self, workspace, source='video', page=0, selected=(), initial=False):
        with self.service._mutation_lock, self.service.gallery_lock:
            if workspace != self.service.workspace_id:
                raise ValueError('The active workspace changed. Reopen the workspace viewer.')
            gen = self.service._state['gen']
            if initial:
                source = gen['current_gallery_source']
            catalog = self._catalog(source)
            count = len(catalog['entries'])
            if initial:
                index = gen[catalog['prefix'] + 'selected']
                selected = [catalog['entries'][index]['key']] if 0 <= index < count else []
                page = index // self.page_size if selected else 0
            pages = max(1, (count + self.page_size - 1) // self.page_size)
            page = max(0, min(page, pages - 1))
            offset = gallery_offset(count, catalog['signature'][2])
            items = []
            for entry in catalog['entries'][page * self.page_size:(page + 1) * self.page_size]:
                path = entry['path']
                kind = self.service.gallery._detect_media_type(path)
                exists = os.path.isfile(path)
                download = register_file_download(path) if exists else None
                item = {'key': entry['key'], 'index': entry['index'], 'name': Path(path).name, 'kind': kind, 'visible': entry['index'] >= offset, 'url': download['url'] if exists else None, 'size': download['size_bytes'] if exists else None}
                if exists and kind in ('image', 'video'):
                    item['thumbnail'] = 'deepy_api/workspace_viewer/thumbnail?' + urlencode({'workspace': workspace, 'source': source, 'key': entry['key'], 'v': os.stat(path).st_mtime_ns})
                items.append(item)
            record = self.service.workspaces.get(workspace)
            return {'workspace': workspace, 'name': record['name'], 'last_activity': record['last_activity'], 'archive_protected': record['archive_protected'], 'source': source, 'revision': catalog['revision'], 'page': page, 'pages': pages, 'total': count, 'visible': count - offset, 'items': items, 'selected': [key for key in selected if key in catalog['by_key']]}

    def _entry(self, workspace, source, key):
        if workspace != self.service.workspace_id:
            raise ValueError('The active workspace changed. Reopen the workspace viewer.')
        entry = self._catalog(source)['by_key'].get(key)
        if entry is None:
            raise ValueError('This media is no longer in the workspace.')
        return entry

    def info(self, workspace, source, key):
        with self.service._mutation_lock:
            entry = self._entry(workspace, source, key)
            prefix = 'audio_' if source == 'audio' else ''
            settings = self.service._state['gen'][prefix + 'file_settings_list'][entry['index']]
            return {'html': self.service._deps.callbacks.emit_first('format_media_info', entry['path'], settings)}

    def thumbnail_path(self, workspace, source, key):
        with self.service._mutation_lock:
            return self._entry(workspace, source, key)['path']

    def action(self, action, payload):
        service = self.service
        with service._mutation_lock, service.gallery_lock:
            workspace, source = payload['workspace'], payload['source']
            if workspace != service.workspace_id:
                raise ValueError('The active workspace changed. Reopen the workspace viewer.')
            catalog = self._catalog(source)
            if payload['revision'] != catalog['revision']:
                raise ValueError('The gallery changed on another page. Refresh your selection and try again.')
            keys = payload['keys']
            if not isinstance(keys, list) or not keys or any(not isinstance(key, str) or key not in catalog['by_key'] for key in keys):
                raise ValueError('Select media from this workspace first.')
            chosen = set(keys)
            entries = [entry for entry in catalog['entries'] if entry['key'] in chosen]
            prefix, gen = catalog['prefix'], service._state['gen']
            if action == 'archive':
                name = payload.get('name', '').strip()
                if not name or len(name) > 120 or any(char in name for char in '/\\:*?"<>|') or name in ('.', '..'):
                    raise ValueError('Enter a valid ZIP filename, without a folder path.')
                paths = [entry['path'] for entry in entries]
                def write_zip(output):
                    names = set()
                    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
                        for path in paths:
                            basename = Path(path).name
                            stem, suffix, index = Path(basename).stem, Path(basename).suffix, 1
                            while basename.casefold() in names:
                                index += 1
                                basename = f'{stem} ({index}){suffix}'
                            names.add(basename.casefold())
                            archive.write(path, basename)
                return json.loads(register_download(name if name.lower().endswith('.zip') else name + '.zip', 'application/zip', lambda: stream_writer(write_zip)))
            if action in ('copy', 'move'):
                target = payload.get('target')
                if target == workspace:
                    raise ValueError('Choose another workspace.')
                with service.workspaces.lock:
                    record = service.workspaces.get(target)
                    saved = dict(record['gallery'])
                    saved[prefix + 'file_list'] = list(saved.get(prefix + 'file_list', [])) + [os.path.abspath(entry['path']) for entry in entries]
                    saved[prefix + 'selected'] = len(saved[prefix + 'file_list']) - 1
                    saved[prefix + 'last_selected'] = True
                    saved['current_gallery_source'] = source
                    service.workspaces.save(target, saved)
                service.publish_workspaces()
                if action == 'copy':
                    return {'count': len(entries)}
            if action not in ('reorder', 'eject', 'delete', 'move'):
                raise ValueError('Unknown workspace media action.')
            if action == 'delete':
                service.require_idle()
                if payload.get('confirmed') is not True:
                    raise ValueError('Confirm permanent file deletion first.')
                # A file may be referenced more than once or by several workspaces.
                removed, failures = set(), []
                for entry in entries:
                    path = os.path.abspath(entry['path'])
                    try:
                        Path(path).unlink(missing_ok=True)
                        removed.add(os.path.normcase(path))
                    except OSError as exc:
                        failures.append(f'{Path(path).name}: {exc.strerror}')
                for group in ('', 'audio_'):
                    retain_entries(gen, group, [i for i, path in enumerate(gen[group + 'file_list']) if os.path.normcase(os.path.abspath(path)) not in removed], inplace=True)
                with service.workspaces.lock:
                    for target in service.workspaces.catalog():
                        if target['id'] == workspace:
                            continue
                        saved = dict(service.workspaces.get(target['id'])['gallery'])
                        for group in ('', 'audio_'):
                            paths = saved.get(group + 'file_list', [])
                            indices = [i for i, path in enumerate(paths) if os.path.normcase(os.path.abspath(path)) not in removed]
                            if len(indices) != len(paths):
                                retain_entries(saved, group, indices)
                        service.workspaces.save(target['id'], saved)
            else:
                indices = [entry['index'] for entry in catalog['entries'] if entry['key'] not in chosen]
                if action == 'reorder':
                    before = payload.get('before')
                    if before is not None and before not in catalog['by_key']:
                        raise ValueError('The drop target no longer exists.')
                    if before in chosen:
                        return {'count': 0}
                    position = 0 if payload.get('edge') == 'start' else len(indices) if before is None else indices.index(catalog['by_key'][before]['index'])
                    indices[position:position] = [entry['index'] for entry in entries]
                retain_entries(gen, prefix, indices, inplace=True)
                failures = []
            gen['selected_video_time'] = None
            service.publish('gallery', service.gallery_snapshot())
            service.host_changed()
            return {'count': len(entries), 'errors': failures}
