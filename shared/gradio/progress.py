"""WanGP progress values and rendering, shared by Gradio and the web app."""

import inspect
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
from html import escape
from pathlib import Path


_tracked_progress = ContextVar("wangp_tqdm_progress", default=None)
_duration = r"(?:(?:\d+h )?(?:\d+m )?\d+(?:\.\d+)?s|\d+:\d{2}(?::\d{2})?)"
_timing = re.compile(rf"{_duration}(?: / {_duration})?")


def _format_bytes(value):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024


class WangpProgress:
    def __init__(self, track_tqdm=False):
        self.track_tqdm = track_tqdm
        self.description = ""
        self.position = None
        self.download = None
        self._bars = []
        self._completion = None
        self._download_finished_at = None

    def __call__(self, progress, desc=None, total=None, unit="steps", _tqdm=None):
        if desc is not None:
            self.description = desc
        if progress is None:
            self.position = None
            self.download = None
            self._completion = None
        elif isinstance(progress, tuple):
            index, total = progress
            self.position = (index, total, unit, None)
        else:
            self.position = (None, total, unit, progress)
        if self.download is None and self.position is not None:
            index, count, unit, fraction = self.position
            if fraction == 1 or (count and index == count):
                self._completion = (self.position, None, self.description, None)

    def status(self, description):
        self.description = description
        self.position = (None, None, "", None)

    def set_download(self, data):
        self.download = data
        if data is not None and data.get("finished_at") is not None and data["finished_at"] != self._download_finished_at:
            self._download_finished_at = data["finished_at"]
            self._completion = (self.position, data, self.description, None)

    @property
    def completion_delay(self):
        if self._completion is None:
            return 0
        return .2 if self._completion[3] is None else max(0, self._completion[3] - time.monotonic())

    def render(self, *, compact=False, aborting=False, active=True, hold_complete=False, bar_text=None):
        # A standalone worker may publish its next update while the UI renders.
        position, download, title = self.position, self.download, self.description
        if hold_complete and not aborting and self.completion_delay:
            completion = self._completion
            position, download, title, deadline = completion
            if deadline is None and self._completion is completion:
                self._completion = (position, download, title, time.monotonic() + .2)
            active = True
        if position is None and download is None:
            return ""
        if download is not None and not title.lower().startswith("downloading"):
            if title.lower().startswith("loading "):
                title = "Downloading" + title[len("Loading"):]
            else:
                title = f"Downloading files — {title}" if title else "Downloading files"
        if title.lower().startswith(("downloading", "loading model")):
            title = title.removesuffix("...").removesuffix("…").rstrip()
        if aborting:
            title = f"Stopping… {title}"
        filename = counter = amount = speed = timing = ""
        if download is not None and active:
            data = download
            completed, total = data["completed"], data["total"]
            ratio = 1 if data.get("finished_at") is not None else completed / total if total else None
            amount = _format_bytes(completed) + (f" / {_format_bytes(total)}" if total is not None else "")
            speed = f'{_format_bytes(data["speed"])}/s'
            filename = data["filename"] or ""
            if data["file_count"] > 1:
                counter = f'File {data["file_index"]}/{data["file_count"]}'
        elif position is not None and active:
            index, total, unit, ratio = position
            if index is not None:
                ratio = index / total if total else None
                amount = f"{index}" + (f" / {total}" if total is not None else "") + f" {unit}"
            elif total is None and ratio == 0:
                ratio = None
            phase, separator, suffix = title.rpartition(" | ")
            if separator and _timing.fullmatch(suffix):
                title, timing = phase, suffix
        else:
            ratio = None
        percentage = max(0, min(100, ratio * 100)) if ratio is not None else None
        classes = "wangp-progress" + (" compact" if compact else "")
        if not active:
            classes += " finished status-only"
        elif download is None and position == (None, None, "", None):
            classes += " status-only"
        text = filename if bar_text is None else bar_text
        text_html = f'<span class="progress-filename">{escape(text)}</span>' if text else ""
        width = f"{percentage:.2f}%" if percentage is not None else "0%"
        value = f' aria-valuenow="{percentage:.1f}"' if percentage is not None else ""
        track = "progress-track" + (" indeterminate" if percentage is None and active and "status-only" not in classes else "")
        percent = f"{percentage:.0f}%" if percentage is not None else ""
        complete = percentage == 100 and (download is None or download.get("finished_at") is not None)
        tooltip = escape(title + ("\n" + text if text else ""))
        details = escape(" · ".join(item for item in (amount, speed, counter, timing) if item))
        metadata = ' · '.join(f'<span class="progress-{name}">{escape(value)}</span>' for name, value in (("amount", amount), ("counter", counter), ("speed", speed)) if value)
        return f'''<div class="{classes}" data-complete="{str(complete).lower()}" style="--progress:{width}">
            <div class="progress-heading" title="{tooltip}"><div class="progress-caption"><span class="progress-title">{escape(title)}</span><span class="progress-percent">{' - ' + percent if percent else ''}</span></div><span class="progress-timing" title="{escape(timing)}">{escape(timing)}</span></div>
            <div class="{track}" role="progressbar" aria-label="{escape(title or 'Progress')}" aria-valuemin="0" aria-valuemax="100"{value} title="{tooltip}">
                <div class="progress-fill"></div><div class="progress-label">{text_html}</div><div class="progress-label progress-covered" aria-hidden="true">{text_html}</div>
            </div>
            <div class="progress-details" title="{details}">{metadata}</div>
        </div>'''

    @staticmethod
    def css():
        return Path(__file__).with_suffix(".css").read_text(encoding="utf-8")

    @staticmethod
    def theme_css(theme):
        """Give the standalone component the same tokens as a Gradio theme."""
        values = theme.to_dict()["theme"]

        def declarations(dark=False):
            rules = []
            for key in ("color_accent", "color_accent_soft", "body_text_color", "block_background_fill", "border_color_primary", "block_border_width", "block_border_color", "block_radius", "block_shadow", "text_md", "text_sm"):
                value = values.get(key + "_dark", values[key]) if dark else values[key]
                while value.startswith("*"):
                    value = values[value[1:]]
                rules.append(f"--{key.replace('_', '-')}: {value};")
            return " ".join(rules)

        return f'.wangp-progress {{ {declarations()} }} @media (prefers-color-scheme: dark) {{ .wangp-progress {{ {declarations(True)} }} }} .dark .wangp-progress {{ {declarations(True)} }}'

    @contextmanager
    def track(self):
        """Track this operation's tqdm bars without suppressing their console output."""
        if self.track_tqdm:
            _install_tqdm_tracking()
        token = _tracked_progress.set(self if self.track_tqdm else None)
        try:
            yield self
        finally:
            try:
                for bar in self._bars[:]:
                    bar.close()
            finally:
                _tracked_progress.reset(token)

    def tqdm(self, iterable=None, desc=None, total=None, unit="steps", _tqdm=None):
        from tqdm.auto import tqdm

        _install_tqdm_tracking()
        token = _tracked_progress.set(self)
        try:
            return tqdm(iterable, desc=desc, total=total, unit=unit)
        finally:
            _tracked_progress.reset(token)

    def update(self, n=1):
        self._bars[-1].update(n)

    def close(self, _tqdm):
        _tqdm.close()

    @staticmethod
    def component(*, visible=False, **kwargs):
        import gradio as gr

        initial = WangpProgress()
        initial.status("Ready")
        return gr.HTML(value=initial.render(active=False) if visible else "", visible=visible, container=True, padding=False, elem_classes=["wangp-progress-container"], **kwargs)

    @classmethod
    def bind(cls, event, fn, *, inputs, outputs, component, hide=(), **kwargs):
        """Bind an operation, optionally swapping controls with its progress in one update."""
        import gradio as gr

        signature = inspect.signature(fn)
        track_tqdm = signature.parameters["progress"].default.track_tqdm

        @wraps(fn)
        def run(*args, **kw):
            progress = cls(track_tqdm=track_tqdm)
            progress.status("Preparing…")

            def work():
                with progress.track():
                    return fn(*args, **kw, progress=progress)

            unchanged = [gr.update() for _ in outputs]
            keep_hidden = [gr.update() for _ in hide]
            restore = [gr.update(visible=True) for _ in hide]
            if hide:
                yield *unchanged, *(gr.update(visible=False) for _ in hide), gr.update(value=progress.render(), visible=True)
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="wangp-progress") as worker:
                future = worker.submit(copy_context().run, work)
                previous = None
                while not future.done():
                    html = progress.render(hold_complete=True)
                    if html != previous:
                        yield *unchanged, *keep_hidden, gr.update(value=html, visible=bool(html))
                        previous = html
                    # A short bounded wait also delivers completion without another polling delay.
                    wait([future], timeout=0.1)
                try:
                    result = future.result()
                except Exception:
                    yield *unchanged, *restore, gr.update(value="", visible=False)
                    raise
            values = [result] if len(outputs) == 1 else result
            if progress.completion_delay:
                yield *values, *keep_hidden, gr.update(value=progress.render(hold_complete=True), visible=True)
                time.sleep(progress.completion_delay)
            yield *values, *restore, gr.update(value="", visible=False)

        # Gradio must not replace our per-invocation tracker with gr.Progress.
        run.__signature__ = signature.replace(parameters=[param for name, param in signature.parameters.items() if name != "progress"])
        kwargs.setdefault("concurrency_id", f"wangp-progress-{component._id}")
        return event(fn=run, inputs=inputs, outputs=[*outputs, *hide, component], show_progress="hidden", **kwargs)


def _install_tqdm_tracking():
    from tqdm import tqdm

    if getattr(tqdm.__init__, "_wangp_progress", False):
        return
    original_init, original_update, original_close = tqdm.__init__, tqdm.update, tqdm.close

    @wraps(original_init)
    def init(bar, *args, **kwargs):
        original_init(bar, *args, **kwargs)
        bar._wangp_progress = None if bar.disable else _tracked_progress.get()
        if bar._wangp_progress is not None:
            bar._wangp_progress._bars.append(bar)
            bar._wangp_progress((bar.n, bar.total), desc=(bar.desc or None) if bar._wangp_progress.download is None else None, unit=bar.unit)

    @wraps(original_update)
    def update(bar, n=1):
        result = original_update(bar, n)
        progress = getattr(bar, "_wangp_progress", None)
        if progress is not None:
            progress((bar.n, bar.total), desc=(bar.desc or None) if progress.download is None else None, unit=bar.unit)
        return result

    @wraps(original_close)
    def close(bar):
        progress = getattr(bar, "_wangp_progress", None)
        if progress is not None and bar in progress._bars:
            progress._bars.remove(bar)
            current = progress._bars[-1] if progress._bars else bar
            progress((current.n, current.total), desc=(current.desc or None) if progress.download is None else None, unit=current.unit)
        return original_close(bar)

    init._wangp_progress = True
    tqdm.__init__, tqdm.update, tqdm.close = init, update, close
