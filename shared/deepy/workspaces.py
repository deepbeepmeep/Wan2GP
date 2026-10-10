"""Workspace lifecycle shared by native Gradio and standalone Web access."""
import logging
from contextlib import nullcontext

from shared.deepy import chat, media_registry, session_store
from shared.deepy.errors import DeepyBusy
from shared.utils.workspaces import ARCHIVE_AFTER_DAYS_KEY, WorkspaceStore, capture_gallery, load_gallery
from shared.utils.media_settings import MediaSettingsCache


class WorkspaceSupport:
    workspaces = None

    def configure_workspaces(self, root):
        with self._mutation_lock:
            self.workspaces = WorkspaceStore(root)
            self._archive_old_workspaces()
            self._deps.controller.workspace_host = self
            self._workspace_settings = MediaSettingsCache(lambda path: self._deps.get_settings_from_file(self._state, path, False, False, False)[0])
            self._web_gallery_cache = None
            self._empty_gallery = load_gallery({'gallery': {}}, self._workspace_settings)
            self.workspace_revision = 0
            saved_id = self._deps.get_server_config().get('last_workspace_id')
            self.workspace_id = saved_id if saved_id in self.workspaces.records else self.workspaces.catalog()[0]['id']
            self._apply_workspace(self.workspace_id)
            self._deps.callbacks.emit('save_workspace_choice', self.workspace_id)

    def _archive_old_workspaces(self):
        # Startup only: manifests already loaded by WorkspaceStore are enough.
        # Never inspect thumbnails, media metadata, or conversation context here.
        logger = logging.getLogger('wangp.workspaces')
        try:
            expired = self.workspaces.expired(self._deps.get_server_config().get(ARCHIVE_AFTER_DAYS_KEY, 0))
        except ValueError as error:
            logger.warning('Workspace auto-archive skipped: %s', error)
            return
        for workspace_id in expired:
            record = self.workspaces.get(workspace_id)
            owner = record.get('deepy_session_id')
            try:
                with session_store.archiving_session(owner) if owner else nullcontext():
                    self.workspaces.archive(workspace_id)
            except Exception as error:
                logger.warning('Could not archive workspace %r: %s', record['name'], error)
        if not self.workspaces.records:
            self.workspaces.create('Default')

    def workspace_snapshot(self):
        if self.workspaces is None:
            return None
        with self._mutation_lock:
            owned = self.workspaces.for_session(self._session.storage_session_id)
            locked = self._deepy_workspace_locked()
            return {'revision': self.workspace_revision, 'selected': self.workspace_id, 'items': self.workspaces.catalog(), 'session_workspace_id': owned['id'] if owned else None, 'deepy_locked': locked, 'deepy_selected': (owned['id'] if owned else None) if locked else self.workspace_id}

    def gallery_snapshot(self):
        with self._mutation_lock:
            if self.workspaces is None or not self._deepy_workspace_locked():
                return self.gallery.snapshot()
            owned = self.workspaces.for_session(self._session.storage_session_id)
            if owned is None:
                return self.gallery.snapshot(self._empty_gallery)
            if owned['id'] == self.workspace_id:
                return self.gallery.snapshot()
            # Main Gradio browsing does not redirect Deepy's gallery. Reuse the
            # same lazy adapter and reload this view only when its manifest changes.
            if self._web_gallery_cache is None or self._web_gallery_cache[0] is not owned:
                self._web_gallery_cache = (owned, load_gallery(owned, self._workspace_settings))
            return self.gallery.snapshot(self._web_gallery_cache[1])

    def prepare_media_workspace(self):
        if self.workspaces is None or not self._deepy_workspace_locked():
            return
        controller, session = self._deps.controller, self._session
        owned = self.workspaces.for_session(session.storage_session_id)
        if owned is not None and owned['id'] == self.workspace_id:
            return
        self.require_idle()
        if owned is None and not controller.dedicated_workspace_enabled():
            raise DeepyBusy('Restart WanGP to activate dedicated workspaces before importing into Deepy.')
        if not session.storage_session_id:
            # Import is meaningful content: persist a blank session and give its
            # workspace a provisional title, without starting the engine.
            session_store.ensure_session(session, '', controller.get_deepy_type(), session.gallery_media_mode, media_import=True)
            self.publish('chat', chat.build_sync_event(session))
        self.ensure_session_workspace()

    def _deepy_workspace_locked(self):
        # Follow the running mode; saved preferences may await Save or restart.
        return bool(self.workspaces.for_session(self._session.storage_session_id)) or self._deps.controller.dedicated_workspace_enabled()

    def publish_workspaces(self):
        if self.workspaces is not None:
            with self._mutation_lock:
                self.workspace_revision += 1
                self.publish('workspaces', self.workspace_snapshot())

    def save_workspace(self):
        if self.workspaces is not None:
            with self._mutation_lock:
                if self.workspaces.save(self.workspace_id, capture_gallery(self._state['gen'])):
                    self.publish_workspaces()

    def _apply_workspace(self, workspace_id):
        gallery = load_gallery(self.workspaces.get(workspace_id), self._workspace_settings)
        self._state['gen'].update(gallery)
        self._state['gen']['refresh_tab'] = False
        self.workspace_id = workspace_id
        owned = self.workspaces.for_session(self._session.storage_session_id)
        self._session.gallery_workspace_id = owned['id'] if owned else workspace_id
        media_registry.sync_tool_call_gallery_media(self._session, self._state['gen'])

    def validate_session_workspace(self, workspace_id):
        if workspace_id is not None and not isinstance(workspace_id, str):
            raise ValueError('Invalid workspace identifier in the saved Deepy session.')
        # Dedicated mode resolves by session ownership, never by an imported
        # session's old gallery reference (which may come from another machine).
        if workspace_id and not self._deps.controller.dedicated_workspace_enabled() and workspace_id not in self.workspaces.records:
            raise ValueError('The workspace saved with this Deepy session no longer exists. Restore that workspace before resuming the session.')

    def restore_session_workspace(self, workspace_id):
        if self._deps.controller.dedicated_workspace_enabled() or self.workspaces.for_session(self._session.storage_session_id):
            self.ensure_session_workspace()
            return
        # Older sessions have no association and keep the current workspace.
        if workspace_id and workspace_id != self.workspace_id:
            self.save_workspace()
            self._apply_workspace(workspace_id)
            self._deps.callbacks.emit('save_workspace_choice', workspace_id)
            self.publish_workspaces()
        self._session.gallery_workspace_id = self.workspace_id

    def ensure_session_workspace(self):
        session = self._session
        if not session.storage_session_id:
            return
        with self._mutation_lock:
            owned = self.workspaces.for_session(session.storage_session_id)
            if owned is None:
                if not self._deps.controller.dedicated_workspace_enabled():
                    return
                # A new conversation gets an empty collection; never adopt the
                # previous session's gallery, even when restoring an old session.
                workspace_id = self.workspaces.create(session.storage_title, deepy_session_id=session.storage_session_id)
            else:
                workspace_id = owned['id']
                if owned['name'] != session.storage_title:
                    self.workspaces.rename(workspace_id, session.storage_title, deepy_session_id=session.storage_session_id)
                    self.publish_workspaces()
            if workspace_id == self.workspace_id and session.gallery_workspace_id == workspace_id:
                return
            if workspace_id != self.workspace_id:
                self.save_workspace()
                self._apply_workspace(workspace_id)
                self._deps.callbacks.emit('save_workspace_choice', workspace_id)
                self.publish('gallery', self.gallery_snapshot())
                self.publish_workspaces()
            session.gallery_workspace_id = workspace_id
            session_store.schedule_autosave(session)

    def rename_session_workspace(self, session_id, title):
        with self._mutation_lock:
            owned = self.workspaces.for_session(session_id)
            if owned is not None:
                self.workspaces.rename(owned['id'], title, deepy_session_id=session_id)
                self.publish_workspaces()

    def delete_session_workspace(self, session_id):
        with self._mutation_lock:
            owned = self.workspaces.for_session(session_id)
            if owned is None:
                return
            self.workspaces.delete(owned['id'], deepy_session_id=session_id)
            if self.workspace_id == owned['id']:
                self._apply_workspace(self.workspaces.catalog()[0]['id'])
                self._deps.callbacks.emit('save_workspace_choice', self.workspace_id)
                self.publish('gallery', self.gallery_snapshot())
            self.publish_workspaces()

    def workspace_control(self, action, payload):
        if self.workspaces is None:
            raise ValueError('Workspaces are not configured.')
        with self._mutation_lock:
            workspace_id = payload.get('id')
            if payload.get('context') == 'deepy' and self._deepy_workspace_locked():
                raise DeepyBusy('Workspaces are assigned by Deepy session in dedicated mode. Change the Deepy session to use another workspace.')
            if action in ('select', 'create', 'delete'):
                if self.generation_running or self._threads or self._session.worker_active or self._session.queued_job_count:
                    raise DeepyBusy('Wait for Deepy and the current generation to finish before changing workspace.')
                self.save_workspace()
            if action == 'create':
                workspace_id = self.workspaces.create(payload.get('name'))
            elif action == 'rename':
                self.workspaces.rename(workspace_id, payload.get('name'))
            elif action == 'delete':
                if workspace_id != self.workspace_id:
                    raise ValueError('The selected workspace has changed. Please try again.')
                self.workspaces.delete(workspace_id)
                workspace_id = self.workspaces.catalog()[0]['id']
            elif action != 'select':
                raise ValueError('Unknown workspace action.')
            if action != 'rename':
                if workspace_id != self.workspace_id and not self.workspaces.for_session(self._session.storage_session_id):
                    with self._session.turn_lock:
                        self._session.pending_chat_media.clear()
                self._apply_workspace(workspace_id)
                self._deps.callbacks.emit('save_workspace_choice', self.workspace_id)
                self.publish('gallery', self.gallery_snapshot())
                session_store.schedule_autosave(self._session)
            self.publish_workspaces()
            return self.workspace_snapshot()
