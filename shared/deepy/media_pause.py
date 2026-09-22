"""Forward chat pause to the media worker's existing cooperative checkpoint."""
from uuid import uuid4
import time
import threading

from shared.utils.process_locks import gen_lock


def postprocessing_pause_checkpoint(gen):
    """Pause between progress callbacks, retaining processor-owned GPU state."""
    while True:
        with gen_lock:
            status = gen.get('process_status', '') or ''
            requested = status == 'request:pause' or status.startswith('request:deepy_pause_')
            paused = status == 'process:pause' or status.startswith('process:deepy_pause_')
            if not requested and not paused:
                return
            if gen.get('abort'):
                gen['process_status'] = 'process:main'
                return
            if requested:
                gen['process_status'] = status.replace('request:', 'process:', 1)
        time.sleep(.1)


def generation_pause_controls(state, previous):
    import gradio as gr
    status = state.get('gen', {}).get('process_status', '') or ''
    pending = status == 'request:pause' or status.startswith('request:deepy_pause_')
    paused = status == 'process:pause' or status.startswith('process:deepy_pause_')
    current = (pending, paused)
    if current == previous:
        return gr.skip(), gr.skip(), current
    return gr.update(visible=not paused, interactive=not pending), gr.update(visible=paused), current


class MediaToolPause:
    def __init__(self, session, gen, client_ids, send_cmd):
        self.session = session
        self.gen = gen
        self.client_ids = set(client_ids)
        self.send_cmd = send_cmd
        self.owner = 'deepy_pause_' + uuid4().hex
        self.request = 'request:' + self.owner
        self.process = 'process:' + self.owner
        self.manual_pause = False
        self.announced_paused = False
        self.closed = threading.Event()
        self.watcher = None

    def _owns_active_task(self):
        task = self.gen.get('api_active_queue_task')
        if task is None:
            queue = self.gen.get('queue') or []
            task = queue[0] if queue else {}
        params = task.get('params', task.get('settings', task))
        return params.get('client_id') in self.client_ids

    def __enter__(self):
        if self.session is not None:
            self.session.media_tool_active = True
            def watch():
                while not self.closed.wait(.1):
                    self.poll()
            self.watcher = threading.Thread(target=watch, name='Deepy media pause', daemon=True)
            self.watcher.start()
        return self

    def _release_locked(self):
        if self.manual_pause:
            self.gen['resume'] = True
        if self.gen.get('process_status') in (self.request, self.process):
            self.gen['process_status'] = 'process:main' if self.gen.get('main_process_running') else None
        self.gen.get('process_names', {}).pop(self.owner, None)

    def poll(self):
        from shared.deepy.engine import mark_assistant_paused, request_assistant_pause
        from shared.deepy.chat import build_status_event

        session = self.session
        if session is None:
            return
        acknowledged = False
        resuming = False
        with session.turn_lock:
            with gen_lock:
                status = self.gen.get('process_status')
                if status not in ('request:pause', 'process:pause'):
                    self.manual_pause = False
                if status in ('request:pause', 'process:pause') and self._owns_active_task() and not self.manual_pause:
                    self.manual_pause = True
                    request_assistant_pause(session)
                if not session.pause_requested or session.interrupt_requested or session.drop_state_requested or self.gen.get('abort'):
                    self._release_locked()
                    if self.announced_paused and not session.interrupt_requested and not session.drop_state_requested:
                        self.announced_paused = False
                        resuming = True
                elif status == self.process or (status == 'process:pause' and self.manual_pause):
                    acknowledged = not session.paused
                elif status == 'process:main' and self.gen.get('main_process_running'):
                    # Only suspend this tool's active task, never another chat's
                    # generation or an unrelated GPU extension.
                    if self._owns_active_task():
                        self.gen.setdefault('process_names', {})[self.owner] = 'Deepy Pause'
                        self.gen['pause_msg'] = 'Media Processing Paused - Resume in Deepy'
                        self.gen['process_status'] = self.request
            if resuming:
                self.send_cmd('chat_output', build_status_event('Resuming media processing...', kind='tool', session=session))
            if acknowledged and mark_assistant_paused(session):
                self.announced_paused = True
                self.send_cmd('chat_output', build_status_event('Deepy is paused.', kind='paused', session=session))

    def __exit__(self, *exc):
        self.closed.set()
        if self.watcher is not None:
            self.watcher.join()
        with gen_lock:
            self._release_locked()
        if self.session is not None:
            with self.session.turn_lock:
                self.session.media_tool_active = False
                # If a tool ends while paused (completion/error), hand the
                # pending request back to the assistant's normal pause gate.
                if self.session.pause_requested:
                    self.session.paused = False
