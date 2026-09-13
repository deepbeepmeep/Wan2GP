"""Layer/tile progress scoped to one generation, with throttled UI and abort polling."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from time import monotonic

from tqdm import tqdm


class GenerationAborted(Exception):
    """Normal cancellation, caught at the generation boundary."""


_generation = ContextVar("phase_progress_generation", default=None)


class _GenerationProgress:
    def __init__(self, pipeline, callback, set_status):
        self.pipeline = pipeline
        self.callback = callback
        self.set_status = set_status
        self.next_poll = 0.0
        self.next_update = 0.0
        self.prompt_no = 0
        self.prompt_total = 1
        self.text_phase_active = False
        self.prompt_scope_active = False
        self.vae_encoding_enabled = False

    def check_abort(self):
        now = monotonic()
        if now >= self.next_poll:
            self.next_poll = now + 1 / 3
            if self.pipeline._interrupt:
                raise GenerationAborted


def generation_progress(method):
    method_signature = signature(method)

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if _generation.get() is not None:
            return method(self, *args, **kwargs)
        arguments = method_signature.bind(self, *args, **kwargs).arguments
        state = _GenerationProgress(self, arguments.get("callback"), arguments.get("set_progress_status"))
        token = _generation.set(state)
        try:
            state.check_abort()
            result = method(self, *args, **kwargs)
            return None if self._interrupt else result
        except GenerationAborted:
            return None
        finally:
            _generation.reset(token)

    return wrapped


def check_abort():
    state = _generation.get()
    if state is not None:
        state.check_abort()


def set_phase_status(title):
    state = _generation.get()
    if state is not None:
        if state.pipeline._interrupt:
            raise GenerationAborted
        if state.set_status is not None:
            state.set_status(title)


class PhaseProgress:
    def __init__(self, total, title="VAE Decoding", unit="tiles", next_status=None):
        self.state = _generation.get()
        self.total = total
        self.title = title
        self.unit = unit
        self.next_status = next_status
        self.completed = 0
        self.started = False

    def _report(self):
        if self.state is None:
            return
        self.state.check_abort()
        now = monotonic()
        if now >= self.state.next_update:
            self.state.next_update = now + 1 / 3
            if self.state.callback is not None:
                self.state.callback(self.completed - 1, None, not self.started, override_num_inference_steps=self.total, progress_unit=self.unit, progress_title=self.title)
                self.started = True

    def advance(self, count=1):
        self.completed += count
        self._report()

    def __enter__(self):
        self._report()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self._report()
            # A final-layer/tile abort must not start the next (possibly expensive) phase.
            if self.state is not None and self.state.pipeline._interrupt:
                raise GenerationAborted
            if self.next_status is not None and self.state is not None and self.state.set_status is not None:
                self.state.set_status(self.next_status)

    @contextmanager
    def track(self, module, count=1):
        """Count completed module calls without changing its forward/offload contract."""
        if self.state is None:
            yield
            return
        handle = module.register_forward_hook(lambda *_: self.advance(count))
        try:
            yield
        finally:
            handle.remove()


@contextmanager
def text_encoding_prompts(total, inherit=False):
    state = _generation.get()
    if state is None or (inherit and state.prompt_scope_active):
        yield
        return
    previous = state.prompt_no, state.prompt_total, state.prompt_scope_active
    state.prompt_no, state.prompt_total = 0, total
    state.prompt_scope_active = True
    try:
        yield
    finally:
        state.prompt_no, state.prompt_total, state.prompt_scope_active = previous


@contextmanager
def text_encoding_progress(layers, next_status="Preparing Conditioning", prompt_count=1):
    state = _generation.get()
    if state is None or state.text_phase_active:
        yield
        return
    state.prompt_no += prompt_count
    total = max(state.prompt_total, prompt_count)
    title = "Encoding Text Prompt"
    if total > 1:
        title += f" {state.prompt_no}/{total}"
    handles = []
    with PhaseProgress(len(layers), title, "layers", next_status) as progress, tqdm(total=len(layers), desc=title, mininterval=1 / 3, leave=False) as bar:
        def completed(*_):
            bar.update()
            progress.advance()

        try:
            state.text_phase_active = True
            for layer in layers:
                handles.append(layer.register_forward_hook(completed))
            yield
        finally:
            state.text_phase_active = False
            for handle in handles:
                handle.remove()


@contextmanager
def vae_decoding_progress(total, decoder, count=1, cleanup=None, title="VAE Decoding", next_status=None):
    """Track actual decoder calls and allow cancellation within an untiled decode."""
    if _generation.get() is None:
        yield
        return
    handles = []
    try:
        with PhaseProgress(total, title=title, next_status=next_status) as progress, tqdm(total=total, desc=title, unit="tiles", mininterval=1 / 3, leave=False) as bar:
            def completed(*_):
                bar.update(count)
                progress.advance(count)

            handles.append(decoder.register_forward_hook(completed))
            for module in decoder.modules():
                if not module._modules:
                    handles.append(module.register_forward_pre_hook(lambda *_: check_abort()))
            yield
    finally:
        for handle in handles:
            handle.remove()
        if cleanup is not None:
            cleanup()


@contextmanager
def vae_encoding_progress(total, encoder, count=1, cleanup=None, enabled=True):
    """Count encoder tiles with the same throttling and cancellation as decoding."""
    state = _generation.get()
    if state is None or not (state.vae_encoding_enabled and enabled):
        yield
        return
    with vae_decoding_progress(total, encoder, count, cleanup, title="VAE Encoding", next_status="Preparing Conditioning"):
        yield


@contextmanager
def control_video_encoding(enabled=True):
    """Enable VAE tile progress only while encoding a control video."""
    state = _generation.get()
    if state is None:
        yield
        return
    previous = state.vae_encoding_enabled
    state.vae_encoding_enabled = enabled
    try:
        yield
    finally:
        state.vae_encoding_enabled = previous
