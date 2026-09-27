"""Cooperative media pause/abort at MMGP block boundaries and preparation checkpoints."""

from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import inspect
import time

from shared.utils.process_locks import gen_lock


class MediaProcessingAborted(Exception):
    """Normal cancellation of an edit or generation worker."""


_control = ContextVar("media_control", default=None)


def current_control():
    return _control.get()


def pause_checkpoint(gen, send_cmd=None, managers=()):
    """Acknowledge pause only after unloading; restore in-flight MMGP state on resume."""
    status = gen.get("process_status") or ""
    if gen.get("abort") or not status.startswith("request:"):
        return False
    suspended = []
    with gen_lock:
        status = gen.get("process_status") or ""
        if gen.get("abort") or not status.startswith("request:"):
            return False
        for manager in managers:
            active = [(model_id, manager.loaded_blocks[model_id]) for model_id in manager.active_models_ids]
            manager.unload_all()
            suspended.append((manager, active))
        gen["process_status"] = status.replace("request:", "process:", 1)
    if send_cmd is not None:
        send_cmd("progress", [0, gen.get("pause_msg", "Media Processing Paused - Click Resume to Continue")])
    while True:
        with gen_lock:
            status = gen.get("process_status") or ""
            manual_pause = status == "process:pause" or status.startswith("process:deepy_pause_")
            if gen.get("abort") and manual_pause:
                gen["resume"] = True
                gen["process_status"] = "process:main"
                break
            if status in ("", "process:main"):
                break
            if status.startswith("request:"):
                gen["process_status"] = status.replace("request:", "process:", 1)
        time.sleep(0.1)
    if not gen.get("abort"):
        for manager, active in suspended:
            for model_id, block in active:
                manager.ensure_model_loaded(model_id)
                if block is not None:
                    manager.gpu_load_blocks(model_id, block)
    return True


class MediaControl:
    def __init__(self, gen, send_cmd, get_offloadobj=lambda: None):
        self.gen = gen
        self.publish = send_cmd
        self.progress = None
        self.get_offloadobj = get_offloadobj
        self.loading = False
        self.loading_callback = None

    def __enter__(self):
        self.token = _control.set(self)
        return self

    def __exit__(self, *exc):
        try:
            if self.gen.get("abort"):
                for manager in self.live_managers():
                    manager.unload_all()
        finally:
            _control.reset(self.token)

    def live_managers(self):
        from shared.utils import offload_registry
        managers = list(offload_registry.query_offloadobjs().values())
        main = self.get_offloadobj()
        if main is not None:
            managers.append(main)
        return list(dict.fromkeys(managers))

    def send_cmd(self, command, data=None):
        if command == "progress":
            self.progress = data
        elif command == "status":
            self.progress = [0, data]
        self.publish(command, data)
        if command == "progress":
            self.checkpoint()

    def checkpoint(self, *, unload=True):
        if not (self.gen.get("process_status") or "").startswith("request:"):
            return
        if pause_checkpoint(self.gen, self.publish, self.live_managers() if unload and not self.loading else ()):
            if not self.gen.get("abort") and self.progress is not None:
                self.publish("progress", self.progress)

def media_abort_requested(gen):
    control = current_control()
    if control is not None and control.gen is gen:
        control.checkpoint()
    return bool(gen.get("abort"))


def controlled_model_loading(fn):
    """Pause loading through its callback, without unloading partially built models."""
    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapped(*args, **kwargs):
        control = current_control()
        if control is None:
            return fn(*args, **kwargs)
        from mmgp import offload
        bound = signature.bind(*args, **kwargs)
        original = bound.arguments.get("loading_callback")
        if "gen" in signature.parameters:
            bound.arguments.setdefault("gen", control.gen)

        def abort_requested():
            control.checkpoint(unload=False)
            return bool(control.gen.get("abort") or (original is not None and original.abort_requested is not None and original.abort_requested()))

        def progress(phase, completed, total, model_id):
            if original is not None and original.progress is not None:
                original.progress(phase, completed, total, model_id)
            else:
                component = model_id.replace("_", " ").title()
                control.send_cmd("progress", [(completed, total), f"Loading - {phase} {component}".rstrip(), total])

        callback = offload.LoadingCallback(abort_requested, progress)
        if "loading_callback" in signature.parameters:
            bound.arguments["loading_callback"] = callback
        previous = control.loading, control.loading_callback
        control.loading, control.loading_callback = True, callback
        try:
            with offload.loading_context(callback):
                callback.check_abort()
                result = fn(*bound.args, **bound.kwargs)
                control.checkpoint(unload=False)
                return result
        except offload.LoadingCancelled as error:
            control.gen["abort"] = True
            raise MediaProcessingAborted() from error
        finally:
            control.loading, control.loading_callback = previous

    return wrapped


def loading_callback():
    control = current_control()
    return None if control is None else control.loading_callback


def inference_checkpoint():
    """State-only callback for model loops that do not emit step progress."""
    control = current_control()
    if control is not None:
        control.checkpoint()
        if control.gen.get("abort"):
            raise MediaProcessingAborted()


@contextmanager
def checkpoint_modules(modules):
    """Use scoped callbacks for third-party encoders without an interrupt argument."""
    if current_control() is None:
        yield
        return
    handles = []
    try:
        for module in modules:
            handles.append(module.register_forward_pre_hook(lambda *_: inference_checkpoint()))
        yield
        inference_checkpoint()
    finally:
        for handle in handles:
            handle.remove()
