"""Wake Gradio's existing scheduler on enqueue/completion instead of polling."""
import asyncio
from asyncio import TimeoutError
from functools import wraps
from queue import Queue
from types import FunctionType, SimpleNamespace

from gradio import queueing, routes
from shared.utils.fastapi_routes import iter_app_routes


def _with_sleep(fn, sleep):
    return FunctionType(fn.__code__, {**fn.__globals__, 'asyncio': SimpleNamespace(sleep=sleep, CancelledError=asyncio.CancelledError)}, fn.__name__, fn.__defaults__, fn.__closure__)


class MessageQueue(Queue):
    def __init__(self):
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self._available = asyncio.Event()

    def put(self, *args, **kwargs):
        super().put(*args, **kwargs)
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._available.set)

    async def wait(self, delay):
        self._available.clear()
        if not self.empty():
            return
        try:
            await asyncio.wait_for(self._available.wait(), timeout=delay)
        except TimeoutError:
            # Keep Gradio's existing disconnect/heartbeat checks at their normal
            # interval; arriving messages interrupt the wait immediately.
            pass


class WakeupQueue(queueing.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._work_ready = asyncio.Event()
        self._work_loop = None

    async def _wait_for_work(self, delay):
        await self._work_ready.wait()
        self._work_ready.clear()

    async def start_processing(self):
        self._work_loop = asyncio.get_running_loop()
        self._work_ready = asyncio.Event()
        original = super().start_processing
        # Keep upstream dispatch, batching, concurrency and cleanup verbatim.
        # Only this method's asyncio.sleep calls wait for a work notification;
        # asyncio itself and generation/progress timers are untouched.
        scheduler = _with_sleep(original, self._wait_for_work)
        await scheduler(self)

    def _notify_work(self):
        if self._work_loop is None:
            self._work_ready.set()
        elif not self._work_loop.is_closed():
            self._work_loop.call_soon_threadsafe(self._work_ready.set)

    async def push(self, *args, **kwargs):
        result = await super().push(*args, **kwargs)
        if result[0]:
            self._notify_work()
        return result

    async def process_events(self, *args, **kwargs):
        try:
            return await super().process_events(*args, **kwargs)
        finally:
            self._notify_work()

    def close(self):
        super().close()
        self._notify_work()


def _patch_message_delivery(app):
    # Both /queue/data and /call/... share this closure. Leave the complete SSE
    # protocol (errors, heartbeats, cancellation, close messages) in upstream code.
    for route in iter_app_routes(app):
        endpoint = getattr(route, 'endpoint', None)
        if not isinstance(endpoint, FunctionType):
            continue
        for name, cell in zip(endpoint.__code__.co_freevars, endpoint.__closure__ or ()):
            if name != 'queue_data_helper':
                continue
            original = cell.cell_contents
            if getattr(original, '_wangp_message_wakeup', False):
                continue

            @wraps(original)
            async def queue_data_helper(request, session_hash, process_msg, _original=original):
                async def wait_for_message(delay):
                    messages = app.get_blocks()._queue.pending_messages_per_session[session_hash]
                    await messages.wait(delay)
                return await _with_sleep(_original, wait_for_message)(request, session_hash, process_msg)

            queue_data_helper._wangp_message_wakeup = True
            cell.cell_contents = queue_data_helper


def install():
    if queueing.Queue is WakeupQueue:
        return
    queueing.Queue = WakeupQueue
    queueing.ThreadQueue = MessageQueue
    original_create_app = routes.App.create_app

    @wraps(original_create_app)
    def create_app(*args, **kwargs):
        app = original_create_app(*args, **kwargs)
        _patch_message_delivery(app)
        return app

    routes.App.create_app = staticmethod(create_app)
