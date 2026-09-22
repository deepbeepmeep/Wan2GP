"""Persistent gallery collections. Media stay at their original locations."""
import os
import threading
import time
import uuid
from pathlib import Path

from shared.utils.config_store import read_config, write_config
from shared.utils.media_settings import MediaSettings


ARCHIVE_AFTER_DAYS_KEY = 'workspace_archive_after_days'
ARCHIVE_PERIODS = [
    ('Disabled', 0), ('1 week', 7), ('2 weeks', 14), ('3 weeks', 21),
    *[(f'{months} month' + ('s' if months > 1 else ''), months * 30) for months in range(1, 12)], ('1 year', 365),
]


class WorkspaceStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.records = {}
        self.session_workspaces = {}
        for path in self.root.glob('*.json'):
            record = read_config(path)
            if not isinstance(record, dict) or record.get('id') != path.stem or not isinstance(record.get('name'), str) or not isinstance(record.get('gallery'), dict):
                raise ValueError(f'Invalid workspace: {path.name}')
            # Older manifests have no content-activity timestamp. Seed it from
            # their last saved date without rewriting every file at startup.
            record.setdefault('last_activity', path.stat().st_mtime)
            record.setdefault('archive_protected', False)
            if not isinstance(record['last_activity'], (int, float)) or not isinstance(record['archive_protected'], bool):
                raise ValueError(f'Invalid workspace activity or protection: {path.name}')
            self.records[record['id']] = record
            owner = record.get('deepy_session_id')
            if owner is not None:
                if not isinstance(owner, str) or not owner or owner in self.session_workspaces:
                    raise ValueError(f'Invalid Deepy workspace ownership: {path.name}')
                self.session_workspaces[owner] = record['id']
        if not self.records:
            self.create('Default')

    def catalog(self):
        with self.lock:
            records = sorted(self.records.values(), key=lambda record: (-record['last_activity'], record['name'].casefold()))
            return [{key: record[key] for key in ('id', 'name', 'deepy_session_id') if key in record} for record in records]

    def _name(self, name, excluding=None, owner=None):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            raise ValueError('Enter a workspace name between 1 and 120 characters.')
        name = name.strip()
        if not owner and any(record['id'] != excluding and not record.get('deepy_session_id') and record['name'].casefold() == name.casefold() for record in self.records.values()):
            raise ValueError('A workspace with this name already exists.')
        return name

    def get(self, workspace_id):
        if not isinstance(workspace_id, str) or workspace_id not in self.records:
            raise ValueError('This workspace no longer exists.')
        return self.records[workspace_id]

    def _write(self, record):
        write_config(record, self.root / (record['id'] + '.json'))
        self.records[record['id']] = record

    def for_session(self, session_id):
        with self.lock:
            workspace_id = self.session_workspaces.get(session_id)
            return self.records[workspace_id] if workspace_id else None

    def create(self, name, *, deepy_session_id=None):
        with self.lock:
            if deepy_session_id and deepy_session_id in self.session_workspaces:
                raise ValueError('This Deepy session already has a dedicated workspace.')
            record = {'id': uuid.uuid4().hex, 'name': self._name(name, owner=deepy_session_id), 'gallery': {}, 'last_activity': time.time(), 'archive_protected': False}
            if deepy_session_id:
                record['deepy_session_id'] = deepy_session_id
            self._write(record)
            if deepy_session_id:
                self.session_workspaces[deepy_session_id] = record['id']
            return record['id']

    def rename(self, workspace_id, name, *, deepy_session_id=None):
        with self.lock:
            record = self.get(workspace_id)
            if record.get('deepy_session_id') and record['deepy_session_id'] != deepy_session_id:
                raise ValueError('Rename this workspace through its Deepy session.')
            self._write({**record, 'name': self._name(name, workspace_id, record.get('deepy_session_id'))})

    def delete(self, workspace_id, *, deepy_session_id=None):
        with self.lock:
            record = self.get(workspace_id)
            owner = record.get('deepy_session_id')
            if owner and owner != deepy_session_id:
                raise ValueError('Delete this workspace through its Deepy session.')
            os.unlink(self.root / (workspace_id + '.json'))
            del self.records[workspace_id]
            if owner:
                del self.session_workspaces[owner]
            if not self.records:
                self.create('Default')

    def save(self, workspace_id, gallery):
        with self.lock:
            record = self.get(workspace_id)
            if record['gallery'] != gallery:
                changed = any(record['gallery'].get(key, []) != gallery.get(key, []) for key in ('file_list', 'audio_file_list'))
                self._write({**record, 'gallery': gallery, 'last_activity': time.time() if changed else record['last_activity']})
                return changed

    def protect(self, workspace_id, protected):
        with self.lock:
            record = self.get(workspace_id)
            if record['archive_protected'] != protected:
                self._write({**record, 'archive_protected': protected})

    def expired(self, days):
        if type(days) is not int or days not in {value for _, value in ARCHIVE_PERIODS}:
            raise ValueError('Choose a supported workspace archive period.')
        if not days:
            return []
        cutoff = time.time() - days * 86400
        with self.lock:
            return [record['id'] for record in self.records.values() if not record['archive_protected'] and record['last_activity'] <= cutoff]

    def archive(self, workspace_id):
        with self.lock:
            record = self.get(workspace_id)
            destination = self.root / 'archives' / (workspace_id + '.json')
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                raise ValueError(f'An archive already exists for workspace {record["name"]}.')
            (self.root / (workspace_id + '.json')).rename(destination)
            del self.records[workspace_id]
            if record.get('deepy_session_id'):
                del self.session_workspaces[record['deepy_session_id']]


def capture_gallery(gen):
    return {key: [os.path.abspath(path) for path in gen[key]] for key in ('file_list', 'audio_file_list')} | {key: gen[key] for key in ('selected', 'audio_selected', 'last_selected', 'audio_last_selected', 'current_gallery_source', 'selected_video_time')}


def load_gallery(record, read_settings):
    saved = record['gallery']
    result = {'current_gallery_source': saved.get('current_gallery_source', 'video'), 'selected_video_time': saved.get('selected_video_time')}
    if result['current_gallery_source'] not in ('video', 'audio'):
        raise ValueError('Invalid workspace gallery tab.')
    for prefix in ('', 'audio_'):
        paths = saved.get(prefix + 'file_list', [])
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError('Invalid workspace media paths.')
        selected = saved.get(prefix + 'selected', -1)
        if not isinstance(selected, int):
            raise ValueError('Invalid workspace media selection.')
        selected_path = paths[selected] if 0 <= selected < len(paths) else None
        existing = [path for path in paths if os.path.isfile(path)]
        result[prefix + 'file_list'] = existing
        result[prefix + 'file_settings_list'] = MediaSettings(existing, read_settings)
        result[prefix + 'selected'] = existing.index(selected_path) if selected_path in existing else len(existing) - 1
        result[prefix + 'last_selected'] = bool(saved.get(prefix + 'last_selected', True))
    return result
