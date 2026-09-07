"""Persistent Deepy execution. HTTP connections are subscribers, not workers."""
from __future__ import annotations

import json
import threading
import traceback
from collections import OrderedDict, deque

from shared.deepy import chat, session_store, ui_settings
from shared.deepy.drivers import AppGallery
from shared.deepy.engine import get_or_create_assistant_session
from shared.deepy.runtime import GenerationRuntime


class DeepyService(GenerationRuntime):
    def __init__(self, deps, *, state=None, gallery_factory=AppGallery):
        self._deps = deps
        self._state = self._build_state() if state is None else state
        self._session = get_or_create_assistant_session(self._state)
        self.gallery = self._gallery = gallery_factory(deps, self._state)
        self._active_generation_client_id = ""
        self._condition = threading.Condition(threading.RLock())
        self._events = deque(maxlen=256)
        self._revision = 0
        self._submissions = OrderedDict()
        self._threads = set()
        self._generation_lock = threading.Lock()
        self._progress = None
        self._generation_aborting = False
        self._closing = False

    def _print(self, text=""):
        print(text)

    def publish(self, kind, data):
        with self._condition:
            self._revision += 1
            self._events.append({"id": self._revision, "type": kind, "data": data})
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {"cursor": self._revision, "chat": chat.build_sync_event(self._session), "gallery": self.gallery.snapshot(), "progress": self._progress, "busy": bool(self._threads), "sessions": self._deps.controller.list_saved_sessions(), "active_session_id": self._session.storage_session_id, "multi_session": self._deps.controller.multi_session_enabled(), "deepy_type": self._deps.controller.get_deepy_type()}

    def events_after(self, cursor):
        with self._condition:
            if cursor == self._revision and not self._closing:
                self._condition.wait(timeout=15)
            if cursor > self._revision or (self._events and cursor < self._events[0]["id"] - 1):
                return [{"id": self._revision, "type": "snapshot", "data": self.snapshot()}]
            return [event for event in self._events if event["id"] > cursor]

    def settings(self):
        current = ui_settings.get_persisted_assistant_tool_ui_settings(self._deps.get_server_config())
        current.update(self._session.tool_ui_settings)
        return ui_settings.get_simplified_settings_form(current)

    def update_settings(self, values):
        with self._condition:
            ui_settings.validate_simplified_settings(values, self.settings())
            current = ui_settings.get_persisted_assistant_tool_ui_settings(self._deps.get_server_config())
            current.update(self._session.tool_ui_settings)
            current.update(values)
            self._deps.controller.update_tool_ui_settings(self._state, **current, persist=True)
            return self.settings()

    def submit(self, text, submission_id, *, steering=False):
        with self._condition:
            if self._closing:
                raise ValueError("Deepy is shutting down.")
            if submission_id in self._submissions:
                if self._submissions[submission_id] != (text, steering):
                    raise ValueError("This submission id already belongs to another request.")
                return
            self._submissions[submission_id] = (text, steering)
            if len(self._submissions) > 4096:
                self._submissions.popitem(last=False)
            self._start_worker(lambda: self._run_request(text, submission_id, steering), [submission_id])

    def _start_worker(self, task, acknowledged_ids=()):
        worker = threading.Thread(target=self._run_worker, args=(task, acknowledged_ids), name="Deepy web request", daemon=True)
        self._threads.add(worker)
        worker.start()

    def _run_worker(self, task, acknowledged_ids):
        try:
            task()
        except Exception as exc:
            traceback.print_exc()
            self.publish("error", str(exc))
        finally:
            with self._condition:
                self._threads.discard(threading.current_thread())
                self.publish("chat", chat.build_sync_event(self._session, acknowledged_submission_ids=acknowledged_ids))
                self.publish("gallery", self.gallery.snapshot())

    def _run_request(self, text, submission_id, steering):
        for command, data in self._deps.controller.iter_commands(self._state, text, submission_id, steering):
            self.command(command, data)

    def _resume_session(self):
        try:
            self._deps.controller.prefill_restored_session_context(self._state)
        finally:
            self.publish("chat", chat.build_status_event(None, visible=False, session=self._session))
        for _ in self._deps.controller.resume_restored_action(self._state, command_callback=self.command):
            pass

    def command(self, command, data=None):
        if command == "chat_output" and data is not None:
            self.publish("chat", data)
        elif command == "load_queue_trigger":
            with self._generation_lock:
                self._process_inline_queue(data)
            self.publish("gallery", self.gallery.snapshot())
        elif command == "refresh_gallery":
            self.gallery.sync_refresh_path(data)
            self.publish("gallery", self.gallery.snapshot())
        elif command == "abort_client_id":
            self._deps.callbacks.emit("abort_generation", self._state, str(data or ""))
        elif command == "error":
            self.publish("error", str(data))

    def _generation_event(self, command, data):
        with self._condition:
            if command in {"progress", "status"}:
                self._progress = {"type": command, "data": data, "aborting": self._generation_aborting}
                self.publish("progress", self._progress)
            elif command == "exit":
                self._generation_aborting = False
                self._progress = None
                self.publish("progress", None)

    def control(self, action, payload):
        controller = self._deps.controller
        if action == "abort":
            with self._condition:
                if self._active_generation_client_id and self._progress is not None and not self._generation_aborting:
                    self._generation_aborting = True
                    self._generation_event("status", "Aborting generation…")
                    self.command("abort_client_id", self._active_generation_client_id)
            return self.snapshot()
        elif action == "stop":
            result = controller.stop_ai(self._state)
            if self._active_generation_client_id:
                self.command("abort_client_id", self._active_generation_client_id)
        elif action == "pause":
            result = controller._toggle_pause_ai(self._state)
        elif action == "queued":
            result = controller.stop_ai(self._state, json.dumps(payload))
        elif action in {"reset", "resume"}:
            with self._condition:
                if self._active_generation_client_id or self._progress is not None:
                    raise ValueError("A generation is in progress. Wait for it to finish before changing conversation.")
                if self._threads or self._session.worker_active or self._session.queued_job_count:
                    raise ValueError("Deepy is active in this conversation. Wait for it to finish before changing conversation.")
                if action == "reset":
                    result = controller.reset_ai(self._state)
                else:
                    resumed = controller.resume_saved_session(self._state, payload["id"], defer_context_prefill=True)
                    result = (resumed["event"],)
                    self.publish("chat", chat.build_status_event(f"Loading Session {self._session.storage_title}", kind="session_loading", session=self._session))
                    self._start_worker(self._resume_session)
        else:
            raise ValueError("Unknown Deepy action.")
        for value in result:
            if isinstance(value, str):
                self.publish("chat", value)
        return self.snapshot()

    def import_media(self, path):
        with self._condition:
            result = self.gallery.add_path(str(path))
            self.publish("gallery", self.gallery.snapshot())
            return result["record"]["media_id"]

    def select_media(self, media_id):
        with self._condition:
            if self.gallery.select(media_id) is None:
                raise ValueError("Media not found.")
            self.publish("gallery", self.gallery.snapshot())

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        session_store.flush_session(self._session)
