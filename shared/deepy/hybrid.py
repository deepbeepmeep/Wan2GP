"""One persistent Deepy service shared by Gradio and the standalone browser client."""
from __future__ import annotations

import json
import threading
import traceback
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace

from shared.deepy.drivers import GradioGallery
from shared.deepy.runtime import GenerationRuntime
from shared.deepy.service import DeepyService


class SharedState(dict):
    """Browser-local generation forms with shared queue, galleries and Deepy."""
    service = None
    shared_keys = {'gen', 'assistant_session', 'last_model_per_family', 'last_model_per_type', 'last_model_per_output_filter', 'last_resolution_per_group'}

    def __deepcopy__(self, memo):
        copied = type(self)()
        memo[id(self)] = copied
        copied.service = self.service
        for key, value in self.items():
            copied[key] = value if key in self.shared_keys else deepcopy(value, memo)
        return copied


def service_for(state):
    return state.service if isinstance(state, SharedState) else None


class HybridService(DeepyService):
    def __init__(self, deps, state, *, process_queue, finalize_queue, unload, gallery_lock=None):
        # Gradio historically initializes these lists on the first UI interaction.
        defaults = GenerationRuntime._build_state(SimpleNamespace(_deps=deps))['gen']
        defaults.pop('in_progress')
        for key, value in defaults.items():
            state['gen'].setdefault(key, value)
        self._host_dirty = threading.Event()
        self._host_signature = None
        self._gallery_signature = None
        self.gallery_revision = 0
        self.settings_revision = 0
        self.preview_revision = 0
        self._preview_image = None
        self._settings_signature = None
        self._catalog_revision = 0
        self._model_forms = {}
        self._model_forms_lock = threading.Lock()
        self._queue_worker = None
        self._queue_updates = (None, None, None)
        self._queue_revision = 0
        self.autoloaded_queue = False
        self._process_queue = process_queue
        self._finalize_queue = finalize_queue
        self._unload = unload
        super().__init__(deps, state=state, gallery_factory=lambda deps, state: GradioGallery(deps, state, self.host_changed))
        state.service = self
        from shared.deepy.workspace_viewer import WorkspaceViewer
        self.gallery_lock = threading.RLock() if gallery_lock is None else gallery_lock
        self.workspace_viewer = WorkspaceViewer(self)
        self._host_worker = threading.Thread(target=self._publish_host_changes, name='Deepy UI synchronization', daemon=True)
        self._host_worker.start()
        self.host_changed()

    def record_model_form(self, model_type, settings):
        snapshot = deepcopy(settings)
        with self._model_forms_lock:
            self._model_forms[model_type] = snapshot

    def load_model_form(self, model_type):
        with self._model_forms_lock:
            snapshot = self._model_forms.get(model_type)
        return deepcopy(snapshot) if snapshot is not None else None

    def host_changed(self):
        self._host_dirty.set()

    def select_gallery_view(self, payload):
        with self._mutation_lock:
            # A click sent before another page switched workspace is obsolete.
            if payload['workspace'] != self.workspace_id:
                return
            source = payload['source']
            if source not in ('video', 'audio'):
                raise ValueError('Invalid gallery source.')
            gen = self._state['gen']
            path = payload['path']
            if path is None:
                gen['current_gallery_source'] = source
                if source == 'audio':
                    gen['selected_video_time'] = None
            else:
                paths, _ = self.gallery._resolve_lists(source == 'audio')
                index = payload['index']
                if not isinstance(index, int) or index < 0 or index >= len(paths) or paths[index] != path:
                    return
                self.gallery._select_index(index, source == 'audio')
            self.host_changed()

    def publish_workspaces(self):
        super().publish_workspaces()
        self.host_changed()

    def _publish_host_changes(self):
        while True:
            self._host_dirty.wait()
            self._host_dirty.clear()
            if self._closing:
                return
            try:
                with self._mutation_lock:
                    gen = self._state['gen']
                    if gen['last_selected']:
                        gen['selected'] = len(gen['file_list']) - 1
                    if gen['audio_last_selected']:
                        gen['audio_selected'] = len(gen['audio_file_list']) - 1
                    if gen.pop('refresh_tab', False):
                        gen['current_gallery_source'] = 'audio' if gen['last_was_audio'] else 'video'
                    signature = (tuple(gen['file_list']), tuple(gen['audio_file_list']), gen['selected'], gen['audio_selected'], gen['current_gallery_source'], tuple(map(id, list.__iter__(gen['file_settings_list']))), tuple(map(id, list.__iter__(gen['audio_file_settings_list']))), self._deps.get_server_config().get('clear_file_list', 5))
                    gallery_view = (signature, self.generation_running, self._queue_revision)
                    settings = (json.dumps(self._session.tool_ui_settings, sort_keys=True), self._session.storage_session_id, self._session.storage_title, self._catalog_revision)
                    changed = False
                    preview = gen.get('preview')
                    if preview is not self._preview_image:
                        self._preview_image = preview
                        self.preview_revision += 1
                        changed = True
                    if signature != self._gallery_signature:
                        media_changed = self._gallery_signature is None or signature[:2] != self._gallery_signature[:2]
                        self._gallery_signature = signature
                        self.publish('gallery', self.gallery_snapshot())
                        if media_changed:
                            self.save_session()
                    if gallery_view != self._host_signature:
                        self._host_signature = gallery_view
                        self.gallery_revision += 1
                        changed = True
                    if settings != self._settings_signature:
                        self._settings_signature = settings
                        self.settings_revision += 1
                        changed = True
                    if changed:
                        self.publish('host_view', {'gallery': self.gallery_revision, 'settings': self.settings_revision, 'preview': self.preview_revision})
            except Exception:
                traceback.print_exc()
                self.publish_error('Could not synchronize the Gradio view.')

    def snapshot(self):
        result = super().snapshot()
        result['hybrid'] = True
        result['host_view'] = {'gallery': self.gallery_revision, 'settings': self.settings_revision, 'preview': self.preview_revision}
        result['forms'] = self.forms.revisions()
        return result

    def command(self, command, data=None):
        super().command(command, data)
        if command in {'refresh_gallery', 'load_queue_trigger'}:
            self.host_changed()

    def _run_worker(self, task, acknowledged_ids):
        try:
            super()._run_worker(task, acknowledged_ids)
        finally:
            self.host_changed()

    def publish_sessions(self):
        self._catalog_revision += 1
        super().publish_sessions()
        self.host_changed()

    def update_settings(self, values, **kwargs):
        result = super().update_settings(values, **kwargs)
        self.host_changed()
        return result

    def control(self, action, payload):
        result = super().control(action, payload)
        self.host_changed()
        return result

    @property
    def generation_running(self):
        return self._queue_worker is not None

    def start_generation(self):
        with self._mutation_lock:
            if self._queue_worker is None and self._state['gen']['queue']:
                self._queue_updates = (None, None, None)
                self._queue_worker = threading.Thread(target=self._run_generation, name='WanGP generation service', daemon=True)
                self._queue_worker.start()
            return self._queue_worker

    def _run_generation(self):
        before = self.gallery.counts()
        self._state["gen"]["status_display"] = True
        self._state["gen"]["last_progress_args"] = None
        self._generation_event("status", "Preparing generation…")
        try:
            for updates in self._process_queue(self._state):
                with self._condition:
                    self._queue_updates = updates
                    self._queue_revision += 1
                    self._condition.notify_all()
                self.host_changed()
        except Exception as error:
            traceback.print_exc()
            # The service broadcasts the error once to every connected client.
            # Observers must not raise it again through their own Gradio requests.
            self.publish_error(error)
        finally:
            try:
                self._finalize_queue(self._state)
                self.gallery.sync_latest_generated(before)
                self._unload(self._state)
            finally:
                with self._condition:
                    self._queue_worker = None
                    self._active_generation_client_id = ''
                    self._condition.notify_all()
                self._generation_event('exit', None)
                self.host_changed()

    def generation_event(self, command, data):
        queue = self._state['gen']['queue']
        if queue:
            self._active_generation_client_id = str(queue[0]['params'].get('client_id', ''))
        if command in {'progress', 'status'}:
            self._generation_event(command, data)
        elif command == 'output':
            self.host_changed()

    def _process_inline_queue(self, payload=None):
        self._deps.load_queue_action(None, self._state, SimpleNamespace(target=1))
        worker = self.start_generation()
        if worker is not None:
            worker.join()

    def gradio_generation(self):
        """Observe the server worker; cancelling a browser request cannot stop it."""
        import gradio as gr
        worker = self.start_generation()
        revision = -1
        while worker is not None:
            with self._condition:
                self._condition.wait_for(lambda: self._queue_revision != revision or self._queue_worker is not worker, timeout=1)
                changed = self._queue_revision != revision
                revision = self._queue_revision
                updates = self._queue_updates
                finished = self._queue_worker is not worker
            if changed:
                yield tuple(gr.update() if value is None else value for value in updates)
            if finished:
                break

    def finalized_updates(self):
        import gradio as gr
        # The revisioned view refresh publishes current state to every browser.
        # A late Gradio callback must not replay an earlier generation's selection.
        return (gr.update(),) * 12

    def close(self):
        super().close()
        self._host_dirty.set()
        if threading.current_thread() is not self._host_worker:
            self._host_worker.join(5)

    @asynccontextmanager
    async def lifespan(self, app, *, auth=None, voice_language=None, https_port=None):
        from shared.deepy.server import create_app
        from starlette.concurrency import run_in_threadpool
        app.mount('/deepy', create_app(self, auth=auth, voice_language=voice_language, https_port=https_port))
        try:
            await run_in_threadpool(self._deps.load_queue_action, None, self._state, SimpleNamespace(target=None))
            self.start_generation()
            yield
        finally:
            self.close()


def launch_gradio(demo, service, args, **kwargs):
    """Use Gradio's supported lifespan hook to mount the shared Web application."""
    from shared.deepy.server import server_options
    host, port, cert, key, https_port, auth = server_options(args)
    print("WanGP web authentication disabled." if auth.gate is None else "WanGP web authentication enabled.")
    scheme = 'https' if cert and https_port is None else 'http'
    print(f"Deepy Web app: {scheme}://{host}:{port}/deepy/")

    @asynccontextmanager
    async def lifespan(app):
        demo.run_startup_events()
        await demo.run_extra_startup_events()
        app.startup_events_triggered = True
        async with service.lifespan(app, auth=auth, voice_language=args.deepy_voice_language, https_port=https_port):
            yield

    from starlette.middleware import Middleware
    from shared.authentication.web import GradioStartupMiddleware, WebAuthMiddleware
    from shared.authentication.tls import HTTPSRedirect
    middleware = [Middleware(WebAuthMiddleware, auth=auth)]
    if https_port is not None:
        middleware.insert(0, Middleware(HTTPSRedirect, port=https_port))
    middleware.insert(0, Middleware(GradioStartupMiddleware))
    demo.launch(**kwargs, app_kwargs={"lifespan": lifespan, "middleware": middleware}, ssl_certfile=cert if https_port is None else None, ssl_keyfile=key if https_port is None else None, ssl_verify=False if cert else True, prevent_thread_lock=https_port is not None)
    if https_port is not None:
        import uvicorn
        print(f"Deepy HTTPS app: https://{host}:{https_port}/deepy/")
        try:
            uvicorn.run(demo.app, host=host, port=https_port, ssl_certfile=cert, ssl_keyfile=key, lifespan='off', timeout_graceful_shutdown=5)
        finally:
            demo.close()
