"""Lightweight in-process API wrapper around WanGP generation."""

from __future__ import annotations

import contextlib
import copy
import importlib
import io
import json
import numpy as np
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image

from shared.utils.process_locks import set_main_generation_running
from shared.utils.virtual_media import parse_virtual_media_path, replace_virtual_media_source

_RUNTIME_LOCK = threading.RLock()
_GENERATION_LOCK = threading.RLock()
_RUNTIME: "_WanGPRuntime | None" = None
_BANNER_PRINTED = False
_STATUS_STEP_PREFIX_RE = re.compile(r"^(?:prompt|sample|sliding window|window|chunk|task|step|phase|pass)\s+\d+\s*/\s*\d+\s*(?:,\s*)?", re.IGNORECASE)
_STATUS_INDEX_RE = re.compile(r"^\[\s*\d+\s*/\s*\d+\s*\]\s*")
_STATUS_TIME_ONLY_RE = re.compile(r"^[\d:.]+\s*[smh]?$", re.IGNORECASE)


def extract_status_phase_label(text: str | None) -> str:
    raw_text = str(text or "").strip()
    if len(raw_text) == 0:
        return ""
    parts = [part.strip() for part in raw_text.split("|") if len(part.strip()) > 0] or [raw_text]
    stripped_wrapper = False
    for part in parts:
        phase_text = part.rsplit(" - ", 1)[-1].strip()
        while True:
            cleaned = _STATUS_INDEX_RE.sub("", phase_text)
            cleaned = _STATUS_STEP_PREFIX_RE.sub("", cleaned)
            cleaned = cleaned.lstrip(" -:,")
            if cleaned == phase_text:
                break
            stripped_wrapper = True
            phase_text = cleaned.strip()
        if len(phase_text) > 0 and not _STATUS_TIME_ONLY_RE.fullmatch(phase_text):
            return phase_text
    return "" if stripped_wrapper else raw_text


@dataclass(frozen=True)
class StreamMessage:
    stream: str
    text: str


@dataclass(frozen=True)
class ProgressUpdate:
    phase: str
    status: str
    progress: int
    current_step: int | None
    total_steps: int | None
    raw_phase: str | None = None
    unit: str | None = None


@dataclass(frozen=True)
class PreviewUpdate:
    image: Image.Image | None
    phase: str
    status: str
    progress: int
    current_step: int | None
    total_steps: int | None


@dataclass(frozen=True)
class SessionEvent:
    kind: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GeneratedArtifact:
    path: str | None
    media_type: str
    client_id: str = ""
    video_tensor_uint8: Any = None
    audio_tensor: Any = None
    audio_sampling_rate: int | None = None
    fps: float | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, default_client_id: str = "") -> "GeneratedArtifact | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            path=str(payload.get("path") or "") or None,
            media_type=str(payload.get("media_type") or "video"),
            client_id=str(payload.get("client_id") or default_client_id or "").strip(),
            video_tensor_uint8=payload.get("video_tensor_uint8"),
            audio_tensor=payload.get("audio_tensor"),
            audio_sampling_rate=payload.get("audio_sampling_rate"),
            fps=payload.get("fps"),
        )


@dataclass(frozen=True)
class GenerationResult:
    success: bool
    generated_files: list[str]
    errors: list["GenerationError"]
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    artifacts: tuple[GeneratedArtifact, ...] = ()

    @property
    def cancelled(self) -> bool:
        return len(self.errors) > 0 and all(error.cancelled for error in self.errors)


@dataclass(frozen=True)
class GenerationError:
    message: str
    task_index: int | None = None
    task_id: Any = None
    stage: str | None = None

    def __str__(self) -> str:
        return self.message

    @property
    def cancelled(self) -> bool:
        stage = str(self.stage or "").strip().lower()
        if stage == "cancelled":
            return True
        return str(self.message or "").strip().lower() == "generation was cancelled"


def get_api_output_options(plugin_data: Any) -> tuple[bool, bool]:
    api_options = {} if not isinstance(plugin_data, dict) else plugin_data.get("api", {})
    if not isinstance(api_options, dict):
        return False, False
    return bool(api_options.get("return_video_uint8") or api_options.get("return_media")), bool(api_options.get("return_audio") or api_options.get("return_media"))


def _coerce_api_video_tensor_uint8(output_video_frames: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(output_video_frames):
        return output_video_frames
    if isinstance(output_video_frames, list) and len(output_video_frames) == 1 and torch is not None and torch.is_tensor(output_video_frames[0]):
        return output_video_frames[0]
    return None


def _coerce_api_audio_tensor(output_audio_data: Any) -> Any:
    return None if output_audio_data is None else np.asarray(output_audio_data, dtype=np.float32)


def build_api_output_artifact_payload(client_id: str, video_path: Any, media_type: str, output_video_frames: Any, output_audio_data: Any, output_audio_sampling_rate: Any, output_fps: Any) -> dict[str, Any] | None:
    client_id = str(client_id or "").strip()
    if len(client_id) == 0:
        return None
    output_path = str(video_path[0]) if isinstance(video_path, list) and len(video_path) > 0 else str(video_path or "")
    return {
        "client_id": client_id,
        "path": output_path,
        "media_type": str(media_type or "video"),
        "video_tensor_uint8": _coerce_api_video_tensor_uint8(output_video_frames),
        "audio_tensor": _coerce_api_audio_tensor(output_audio_data),
        "audio_sampling_rate": int(output_audio_sampling_rate) if output_audio_sampling_rate else None,
        "fps": float(output_fps) if output_fps else None,
    }


def store_api_output_artifact(gen: dict[str, Any], client_id: str, video_path: Any, media_type: str, output_video_frames: Any, output_audio_data: Any, output_audio_sampling_rate: Any, output_fps: Any) -> bool:
    payload = build_api_output_artifact_payload(client_id, video_path, media_type, output_video_frames, output_audio_data, output_audio_sampling_rate, output_fps)
    if payload is None:
        return False
    gen.setdefault("api_output_artifacts", {})[payload["client_id"]] = payload
    return True


class SessionStream:
    def __init__(self) -> None:
        self._queue: queue.Queue[SessionEvent | object] = queue.Queue()
        self._closed = threading.Event()
        self._sentinel = object()

    def put(self, kind: str, data: Any = None) -> None:
        if self._closed.is_set():
            return
        self._queue.put(SessionEvent(kind=kind, data=data))

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(self._sentinel)

    def get(self, timeout: float | None = None) -> SessionEvent | None:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is self._sentinel:
            return None
        return item

    def iter(self, timeout: float | None = None) -> Iterator[SessionEvent]:
        while True:
            event = self.get(timeout=timeout)
            if event is None:
                if self._closed.is_set():
                    break
                continue
            yield event

    @property
    def closed(self) -> bool:
        return self._closed.is_set()


class _OutputCapture(io.TextIOBase):
    def __init__(
        self,
        stream_name: str,
        emit_line,
        console: io.TextIOBase | None = None,
        *,
        console_isatty: bool = True,
    ) -> None:
        self._stream_name = stream_name
        self._emit_line = emit_line
        self._console = console
        self._console_isatty = bool(console_isatty)
        self._buffer = ""

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return str(getattr(self._console, "encoding", "utf-8"))

    def isatty(self) -> bool:
        return self._console_isatty

    def write(self, text: str) -> int:
        if not text:
            return 0
        if self._console is not None:
            self._console.write(text)
        self._buffer += text
        self._drain(False)
        return len(text)

    def flush(self) -> None:
        if self._console is not None:
            self._console.flush()
        self._drain(True)

    def _drain(self, flush_all: bool) -> None:
        while True:
            split_at = -1
            for delimiter in ("\r", "\n"):
                index = self._buffer.find(delimiter)
                if index >= 0 and (split_at < 0 or index < split_at):
                    split_at = index
            if split_at < 0:
                break
            line = self._buffer[:split_at]
            self._buffer = self._buffer[split_at + 1 :]
            if line:
                self._emit_line(self._stream_name, line)
        if flush_all and self._buffer:
            self._emit_line(self._stream_name, self._buffer)
            self._buffer = ""


@dataclass(frozen=True)
class _WanGPRuntime:
    module: Any
    root: Path
    config_path: Path
    cli_args: tuple[str, ...]


class SessionJob:
    def __init__(self, session: "WanGPSession") -> None:
        self._session = session
        self._callbacks: object | None = None
        self.events = SessionStream()
        self._done = threading.Event()
        self._cancel_requested = threading.Event()
        self._webui_submission_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: GenerationResult | None = None
        self._webui_manifest: list[dict[str, Any]] = []
        self._webui_client_ids: tuple[str, ...] = ()
        self._webui_load_queue_token = ""
        self._webui_owner_call_id = ""

    def _bind_thread(self, thread: threading.Thread) -> None:
        self._thread = thread

    def _bind_callbacks(self, callbacks: object | None) -> None:
        self._callbacks = callbacks

    def _set_result(self, result: GenerationResult) -> None:
        self._result = result
        self._done.set()

    def _set_webui_bridge(self, *, manifest: Sequence[dict[str, Any]], client_ids: Sequence[str], load_queue_token: str) -> None:
        self._webui_manifest = copy.deepcopy(list(manifest))
        self._webui_client_ids = tuple(str(client_id or "").strip() for client_id in client_ids if str(client_id or "").strip())
        self._webui_load_queue_token = str(load_queue_token or "").strip()

    def _mark_webui_submission_ready(self) -> None:
        self._webui_submission_ready.set()

    def _bind_webui_owner_call(self, call_id: str) -> None:
        self._webui_owner_call_id = str(call_id or "").strip()

    def cancel(self) -> None:
        self._cancel_requested.set()
        owner = getattr(self._session, "_gradio_session_proxy", None)
        capture = getattr(owner, "_capture_cancelled_job", None)
        if callable(capture):
            capture(self)

    def result(self, timeout: float | None = None) -> GenerationResult:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError("WanGP session job timed out")
        return self._result or GenerationResult(
            success=False,
            generated_files=[],
            errors=[],
            total_tasks=0,
            successful_tasks=0,
            failed_tasks=0,
            artifacts=(),
        )

    def join(self, timeout: float | None = None) -> GenerationResult:
        return self.result(timeout=timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    @property
    def webui_manifest(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._webui_manifest)

    @property
    def webui_client_ids(self) -> tuple[str, ...]:
        return self._webui_client_ids

    @property
    def primary_client_id(self) -> str:
        return "" if not self._webui_client_ids else self._webui_client_ids[0]

    @property
    def webui_load_queue_token(self) -> str:
        return self._webui_load_queue_token

    @property
    def webui_submission_ready(self) -> bool:
        return self._webui_submission_ready.is_set()

    @property
    def webui_owner_call_id(self) -> str:
        return self._webui_owner_call_id


class WanGPSession:
    def __init__(
        self,
        *,
        root: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        callbacks: object | None = None,
        cli_args: Sequence[str] = (),
        console_output: bool = True,
        console_isatty: bool = True,
        webui_state: dict[str, Any] | None = None,
    ) -> None:
        self._root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        self._config_path = Path(config_path).resolve() if config_path is not None else (self._root / "wgp_config.json").resolve()
        self._output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self._callbacks = callbacks
        self._cli_args = tuple(str(arg) for arg in cli_args)
        self._console_output = bool(console_output)
        self._console_isatty = bool(console_isatty)
        self._use_webui_queue = isinstance(webui_state, dict)
        self._state = webui_state if isinstance(webui_state, dict) else self._create_headless_state()
        self._active_job: SessionJob | None = None
        self._job_lock = threading.Lock()
        self._attachment_keys: tuple[str, ...] | None = None

    def ensure_ready(self) -> "WanGPSession":
        self._ensure_runtime()
        return self

    def submit(self, source: str | os.PathLike[str] | dict[str, Any] | list[dict[str, Any]], callbacks: object | None = None) -> SessionJob:
        tasks = self._normalize_source(source, caller_base_path=self._get_caller_base_path())
        return self._submit_tasks(tasks, callbacks=callbacks)

    def submit_task(self, settings: dict[str, Any], callbacks: object | None = None) -> SessionJob:
        caller_base_path = self._get_caller_base_path()
        task = self._normalize_task(settings, task_index=1)
        return self._submit_tasks([self._absolutize_task_paths(task, caller_base_path)], callbacks=callbacks)

    def submit_manifest(self, settings_list: list[dict[str, Any]], callbacks: object | None = None) -> SessionJob:
        caller_base_path = self._get_caller_base_path()
        tasks = [
            self._absolutize_task_paths(self._normalize_task(settings, task_index=index + 1), caller_base_path)
            for index, settings in enumerate(settings_list)
        ]
        return self._submit_tasks(tasks, callbacks=callbacks)

    def run(self, source: str | os.PathLike[str] | dict[str, Any] | list[dict[str, Any]], callbacks: object | None = None) -> GenerationResult:
        return self.submit(source, callbacks=callbacks).result()

    def run_task(self, settings: dict[str, Any], callbacks: object | None = None) -> GenerationResult:
        return self.submit_task(settings, callbacks=callbacks).result()

    def run_manifest(self, settings_list: list[dict[str, Any]], callbacks: object | None = None) -> GenerationResult:
        return self.submit_manifest(settings_list, callbacks=callbacks).result()

    def close(self) -> None:
        if self._use_webui_queue:
            return
        runtime = self._ensure_runtime()
        with _GENERATION_LOCK, _pushd(runtime.root):
            runtime.module.release_model()

    def cancel(self) -> None:
        with self._job_lock:
            job = self._active_job
        if job is not None:
            job.cancel()

    @staticmethod
    def _create_headless_state() -> dict[str, Any]:
        return {
            "gen": {
                "queue": [],
                "in_progress": False,
                "file_list": [],
                "file_settings_list": [],
                "audio_file_list": [],
                "audio_file_settings_list": [],
                "selected": 0,
                "audio_selected": 0,
                "prompt_no": 0,
                "prompts_max": 0,
                "repeat_no": 0,
                "total_generation": 1,
                "window_no": 0,
                "total_windows": 0,
                "progress_status": "",
                "process_status": "process:main",
                "api_output_artifacts": {},
            },
            "loras": [],
        }

    def _submit_tasks(self, tasks: list[dict[str, Any]], callbacks: object | None = None) -> SessionJob:
        with self._job_lock:
            if self._active_job is not None and not self._active_job.done:
                raise RuntimeError("WanGP session already has a generation in progress")
            job = SessionJob(self)
            self._bind_callbacks_to_job(job, callbacks)
            prepared_tasks = copy.deepcopy(tasks)
            client_ids = self._ensure_task_client_ids(prepared_tasks, priority=self._use_webui_queue)
            if self._use_webui_queue:
                prepared_tasks, manifest, load_queue_token = self._prepare_webui_bridge(prepared_tasks)
                job._set_webui_bridge(manifest=manifest, client_ids=client_ids, load_queue_token=load_queue_token)
            thread = threading.Thread(
                target=self._run_job,
                args=(job, prepared_tasks),
                daemon=True,
                name="wangp-session-job",
            )
            job._bind_thread(thread)
            self._active_job = job
            thread.start()
            return job

    def _bind_callbacks_to_job(self, job: SessionJob, callbacks: object | None = None) -> None:
        callback = self._callbacks if callbacks is None else callbacks
        job._bind_callbacks(callback)
        if callback is None:
            return
        binder = getattr(callback, "bind_job", None)
        if not callable(binder):
            return
        try:
            binder(session=self, job=job)
        except TypeError:
            binder(job)

    @staticmethod
    def _ensure_task_client_ids(tasks: list[dict[str, Any]], *, priority: bool = False) -> tuple[str, ...]:
        client_seed = time.time_ns()
        client_ids: list[str] = []
        for index, task in enumerate(tasks, start=1):
            params = copy.deepcopy(WanGPSession._get_task_settings(task))
            client_id = str(params.get("client_id", "") or "").strip()
            if len(client_id) == 0:
                client_id = f"api_{client_seed}_{index}"
            params["client_id"] = client_id
            if priority:
                params["priority"] = True
            elif "priority" in params and not params["priority"]:
                params.pop("priority", None)
            task["params"] = params
            client_ids.append(client_id)
        return tuple(client_ids)

    def _prepare_webui_bridge(self, tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        manifest = []
        for index, task in enumerate(tasks, start=1):
            params = copy.deepcopy(self._get_task_settings(task))
            params["priority"] = True
            task["params"] = params
            manifest.append({
                "id": task.get("id", index),
                "params": copy.deepcopy(params),
                "plugin_data": copy.deepcopy(task.get("plugin_data", {})),
            })
        return tasks, manifest, str(time.time_ns())

    def _run_job(self, job: SessionJob, tasks: list[dict[str, Any]]) -> None:
        if self._use_webui_queue:
            self._run_webui_job(job, tasks)
            return
        from shared.api_cli import run_cli_job

        run_cli_job(self, job, tasks)

    def _run_webui_job(self, job: SessionJob, tasks: list[dict[str, Any]]) -> None:
        from shared.api_webui import run_webui_job

        run_webui_job(self, job, tasks)

    def _build_progress_update(self, data: Any, *, include_state_fallback: bool = True) -> ProgressUpdate:
        current_step: int | None = None
        total_steps: int | None = None
        status = ""
        unit: str | None = None

        if isinstance(data, list) and data:
            head = data[0]
            if isinstance(head, tuple) and len(head) == 2:
                current_step = int(head[0])
                total_steps = int(head[1])
                status = str(data[1] if len(data) > 1 else "")
                if len(data) > 3:
                    unit = str(data[3])
            else:
                status = str(data[1] if len(data) > 1 else head)
        else:
            status = str(data or "")

        raw_phase = None
        if include_state_fallback:
            progress_phase = self._state["gen"].get("progress_phase")
            if isinstance(progress_phase, tuple) and progress_phase:
                raw_phase = extract_status_phase_label(progress_phase[0])
                if current_step is None and len(progress_phase) > 1 and "denoising" in raw_phase.lower():
                    try:
                        progress_step = int(progress_phase[1])
                    except (TypeError, ValueError):
                        progress_step = -1
                    try:
                        inference_steps = int(self._state["gen"].get("num_inference_steps") or 0)
                    except (TypeError, ValueError):
                        inference_steps = 0
                    if progress_step >= 0 and inference_steps > 0:
                        current_step = progress_step
                        total_steps = inference_steps
            if len(status) == 0:
                status = str(self._state["gen"].get("progress_status", "") or raw_phase or "")
        status_phase_label = extract_status_phase_label(status)
        if len(status_phase_label) > 0 and len(str(raw_phase or "").strip()) > 0 and current_step is None:
            normalized_status_phase = self._normalize_phase(status_phase_label)
            normalized_raw_phase = self._normalize_phase(raw_phase)
            if normalized_status_phase != normalized_raw_phase:
                raw_phase = None
        display_phase = raw_phase or status_phase_label
        phase = self._normalize_phase(display_phase or status)
        if not self._phase_supports_progress(phase):
            current_step = None
            total_steps = None
        progress = self._estimate_progress(phase, current_step, total_steps)
        return ProgressUpdate(
            phase=phase,
            status=status,
            progress=progress,
            current_step=current_step,
            total_steps=total_steps,
            raw_phase=display_phase or None,
            unit=unit,
        )

    def _build_preview_update(self, wgp, tasks: list[dict[str, Any]], payload: Any) -> PreviewUpdate | None:
        progress = self._build_progress_update([0, self._state["gen"].get("progress_status", "")])
        model_type = ""
        queue_tasks = self._state["gen"].get("queue") or tasks
        if queue_tasks:
            model_type = str(self._get_task_settings(queue_tasks[0]).get("model_type", ""))
        image = wgp.generate_preview(model_type, payload) if model_type else None
        return PreviewUpdate(
            image=image,
            phase=progress.phase,
            status=progress.status,
            progress=progress.progress,
            current_step=progress.current_step,
            total_steps=progress.total_steps,
        )

    def _emit_stream(self, job: SessionJob, stream_name: str, line: str) -> None:
        message = StreamMessage(stream=stream_name, text=line)
        job.events.put("stream", message)
        self._emit_callback("on_stream", message, job=job)

    def _emit_callback(self, method_name: str, payload: Any, *, job: SessionJob | None = None) -> None:
        callback = self._callbacks if job is None or job._callbacks is None else job._callbacks
        if callback is None:
            return
        method = getattr(callback, method_name, None)
        if callable(method):
            method(payload)
        on_event = getattr(callback, "on_event", None)
        if callable(on_event):
            on_event(SessionEvent(kind=method_name.removeprefix("on_"), data=payload))

    def _configure_runtime(self, runtime: _WanGPRuntime) -> None:
        runtime.module.server_config["notification_sound_enabled"] = 0
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            runtime.module.server_config["save_path"] = str(self._output_dir)
            runtime.module.server_config["image_save_path"] = str(self._output_dir)
            runtime.module.server_config["audio_save_path"] = str(self._output_dir)
            runtime.module.save_path = str(self._output_dir)
            runtime.module.image_save_path = str(self._output_dir)
            runtime.module.audio_save_path = str(self._output_dir)
        for output_path in (
            runtime.module.save_path,
            runtime.module.image_save_path,
            runtime.module.audio_save_path,
        ):
            Path(output_path).mkdir(parents=True, exist_ok=True)

    def _prepare_state_for_run(self, tasks: list[dict[str, Any]]) -> None:
        gen = self._state["gen"]
        gen["queue"] = tasks
        set_main_generation_running(self._state, True)
        gen["process_status"] = "process:main"
        gen["progress_status"] = ""
        gen["progress_phase"] = ("", -1)
        gen["abort"] = False
        gen["early_stop"] = False
        gen["early_stop_forwarded"] = False
        gen["preview"] = None
        gen["status"] = "Generating..."
        gen["in_progress"] = True
        gen.setdefault("api_output_artifacts", {})
        self._ensure_runtime().module.gen_in_progress = True

    def _reset_state_after_run(self) -> None:
        gen = self._state["gen"]
        gen["queue"] = []
        set_main_generation_running(self._state, False)
        gen["process_status"] = "process:main"
        gen["progress_status"] = ""
        gen["progress_phase"] = ("", -1)
        gen["abort"] = False
        gen["early_stop"] = False
        gen["early_stop_forwarded"] = False
        gen.pop("in_progress", None)
        self._ensure_runtime().module.gen_in_progress = False

    def _collect_outputs(self, base_file_count: int, base_audio_count: int) -> list[str]:
        gen = self._state["gen"]
        files = gen["file_list"][base_file_count:]
        audio_files = gen["audio_file_list"][base_audio_count:]
        return [str(Path(path).resolve()) for path in [*files, *audio_files]]

    def _consume_output_artifact(self, client_id: str) -> GeneratedArtifact | None:
        gen = self._state["gen"]
        artifacts = gen.get("api_output_artifacts")
        if not isinstance(artifacts, dict):
            return None
        payload = artifacts.pop(str(client_id or "").strip(), None)
        return GeneratedArtifact.from_payload(payload, default_client_id=str(client_id or "").strip())

    def _peek_output_artifact(self, client_id: str) -> GeneratedArtifact | None:
        gen = self._state["gen"]
        artifacts = gen.get("api_output_artifacts")
        if not isinstance(artifacts, dict):
            return None
        payload = artifacts.get(str(client_id or "").strip(), None)
        return GeneratedArtifact.from_payload(payload, default_client_id=str(client_id or "").strip())

    def _consume_output_artifacts(self, tasks: Sequence[dict[str, Any]]) -> tuple[GeneratedArtifact, ...]:
        artifacts: list[GeneratedArtifact] = []
        for task in tasks:
            client_id = str(self._get_task_settings(task).get("client_id", "") or "").strip()
            if len(client_id) == 0:
                continue
            artifact = self._consume_output_artifact(client_id)
            if artifact is not None:
                artifacts.append(artifact)
        return tuple(artifacts)

    def _request_cancel_unlocked(self, wgp) -> None:
        gen = self._state["gen"]
        gen["resume"] = True
        gen["abort"] = True
        if wgp.wan_model is not None:
            wgp.wan_model._interrupt = True

    def _normalize_source(
        self,
        source: str | os.PathLike[str] | dict[str, Any] | list[dict[str, Any]],
        *,
        caller_base_path: Path,
    ) -> list[dict[str, Any]]:
        if isinstance(source, (str, os.PathLike)):
            return self._load_tasks_from_path(self._resolve_source_path(Path(source), caller_base_path), caller_base_path)
        if isinstance(source, list):
            return [
                self._absolutize_task_paths(self._normalize_task(task, task_index=index + 1), caller_base_path)
                for index, task in enumerate(source)
            ]
        if isinstance(source, dict):
            if isinstance(source.get("tasks"), list):
                tasks = source["tasks"]
                return [
                    self._absolutize_task_paths(self._normalize_task(task, task_index=index + 1), caller_base_path)
                    for index, task in enumerate(tasks)
                ]
            return [self._absolutize_task_paths(self._normalize_task(source, task_index=1), caller_base_path)]
        raise TypeError("WanGP session source must be a path, a settings dict, or a manifest list")

    def _normalize_task(self, task: dict[str, Any], *, task_index: int) -> dict[str, Any]:
        if not isinstance(task, dict):
            raise TypeError(f"Task {task_index} must be a dictionary")
        normalized = copy.deepcopy(task)
        if "settings" in normalized and "params" not in normalized:
            normalized["params"] = normalized.pop("settings")
        if "params" not in normalized:
            normalized = {"id": task_index, "params": normalized, "plugin_data": {}}
        normalized.setdefault("id", task_index)
        normalized.setdefault("plugin_data", {})
        normalized.setdefault("params", {})
        if not isinstance(normalized["plugin_data"], dict):
            normalized["plugin_data"] = {}
        settings = normalized["params"]
        if isinstance(settings, dict):
            api_options = settings.pop("_api", None)
            if isinstance(api_options, dict):
                normalized["plugin_data"]["api"] = copy.deepcopy(api_options)
            runtime_settings_version = getattr(self._ensure_runtime().module, "settings_version", None)
            if runtime_settings_version is not None:
                settings.setdefault("settings_version", runtime_settings_version)
            self._normalize_settings_values(settings)
            normalized.setdefault("prompt", settings.get("prompt", ""))
            normalized.setdefault("length", settings.get("video_length"))
            normalized.setdefault("steps", settings.get("num_inference_steps"))
            normalized.setdefault("repeats", settings.get("repeat_generation", 1))
        return normalized

    @staticmethod
    def _normalize_settings_values(settings: dict[str, Any]) -> None:
        force_fps = settings.get("force_fps")
        if isinstance(force_fps, (int, float)) and not isinstance(force_fps, bool):
            if isinstance(force_fps, float) and not force_fps.is_integer():
                settings["force_fps"] = str(force_fps)
            else:
                settings["force_fps"] = str(int(force_fps))

    @staticmethod
    def _get_task_settings(task: dict[str, Any]) -> dict[str, Any]:
        settings = task.get("params")
        if isinstance(settings, dict):
            return settings
        settings = task.get("settings")
        if isinstance(settings, dict):
            return settings
        return {}

    def _load_tasks_from_path(self, path: Path, caller_base_path: Path) -> list[dict[str, Any]]:
        runtime = self._ensure_runtime()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".json":
            return self._load_settings_json(path, caller_base_path)
        with _pushd(runtime.root):
            tasks, error = runtime.module._parse_queue_zip(str(path), self._state)
        if error:
            raise RuntimeError(error)
        return [self._normalize_task(task, task_index=index + 1) for index, task in enumerate(tasks)]

    def _load_settings_json(self, path: Path, caller_base_path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            raw_tasks = payload
        elif isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
            raw_tasks = payload["tasks"]
        elif isinstance(payload, dict):
            raw_tasks = [payload]
        else:
            raise RuntimeError("Settings file must contain a JSON object or a list of tasks")

        tasks = [self._normalize_task(task, task_index=index + 1) for index, task in enumerate(raw_tasks)]
        return [self._absolutize_task_paths(task, caller_base_path) for task in tasks]

    @staticmethod
    def _get_caller_base_path() -> Path:
        return Path.cwd().resolve()

    @staticmethod
    def _resolve_source_path(path: Path, caller_base_path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (caller_base_path / path).resolve()

    def _absolutize_task_paths(self, task: dict[str, Any], caller_base_path: Path) -> dict[str, Any]:
        normalized = copy.deepcopy(task)
        settings = normalized.get("params")
        if not isinstance(settings, dict):
            return normalized
        for key in self._get_attachment_keys():
            if key not in settings:
                continue
            settings[key] = self._absolutize_setting_path(settings[key], caller_base_path)
        return normalized

    def _get_attachment_keys(self) -> tuple[str, ...]:
        if self._attachment_keys is None:
            runtime = self._ensure_runtime()
            keys = getattr(runtime.module, "ATTACHMENT_KEYS", ())
            self._attachment_keys = tuple(str(key) for key in keys)
        return self._attachment_keys

    def _absolutize_setting_path(self, value: Any, caller_base_path: Path) -> Any:
        if isinstance(value, list):
            return [self._absolutize_setting_path(item, caller_base_path) for item in value]
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if not isinstance(value, str) or not value.strip():
            return value
        spec = parse_virtual_media_path(value)
        path = Path(spec.source_path if spec is not None else value)
        if path.is_absolute():
            resolved = str(path.resolve())
        else:
            resolved = str((caller_base_path / path).resolve())
        return replace_virtual_media_source(value, resolved) if spec is not None else resolved

    @staticmethod
    def _make_generation_error(
        error: Any,
        *,
        task_index: int | None = None,
        task_id: Any = None,
        stage: str | None = None,
    ) -> GenerationError:
        if isinstance(error, GenerationError):
            return error
        if isinstance(error, BaseException):
            message = str(error) or error.__class__.__name__
        else:
            message = str(error)
        return GenerationError(message=message, task_index=task_index, task_id=task_id, stage=stage)

    def _ensure_runtime(self) -> _WanGPRuntime:
        global _RUNTIME
        with _RUNTIME_LOCK:
            if _RUNTIME is not None:
                if _RUNTIME.root != self._root or _RUNTIME.config_path != self._config_path or _RUNTIME.cli_args != self._cli_args:
                    raise RuntimeError("WanGP runtime already loaded with different root/config/cli args")
                return _RUNTIME

            argv = ["wgp.py", *self._cli_args]
            default_config_path = (self._root / "wgp_config.json").resolve()
            if self._config_path.name != "wgp_config.json":
                raise ValueError("config_path must point to a file named 'wgp_config.json'")
            if self._config_path != default_config_path:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                if "--config" not in argv:
                    argv.extend(["--config", str(self._config_path.parent)])

            if str(self._root) not in sys.path:
                sys.path.insert(0, str(self._root))

            with _pushd(self._root), _temporary_argv(argv):
                module = importlib.import_module("wgp")
                module_root = Path(module.__file__).resolve().parent
                if module_root != self._root:
                    raise RuntimeError(f"WanGP module already loaded from {module_root}, expected {self._root}")
                if not hasattr(module, "app"):
                    module.app = module.WAN2GPApplication()
                module.download_ffmpeg()

            _RUNTIME = _WanGPRuntime(
                module=module,
                root=self._root,
                config_path=self._config_path,
                cli_args=self._cli_args,
            )
            _print_banner_once(module, enabled=not self._use_webui_queue)
            return _RUNTIME

    @staticmethod
    def _normalize_phase(text: str | None) -> str:
        lowered = extract_status_phase_label(text).lower()
        if "denoising first pass" in lowered or "denoising 1st pass" in lowered:
            return "inference_stage_1"
        if "denoising second pass" in lowered or "denoising 2nd pass" in lowered:
            return "inference_stage_2"
        if "denoising third pass" in lowered or "denoising 3rd pass" in lowered:
            return "inference_stage_3"
        if "loading model" in lowered or lowered.startswith("loading"):
            return "loading_model"
        if "enhancing prompt" in lowered or "encoding prompt" in lowered or "encoding" in lowered:
            return "encoding_text"
        if "vae decoding" in lowered or "decoding" in lowered:
            return "decoding"
        if "saved" in lowered or "completed" in lowered or "output" in lowered:
            return "downloading_output"
        if "cancel" in lowered or "abort" in lowered:
            return "cancelled"
        return "inference"

    @staticmethod
    def _phase_supports_progress(phase: str | None) -> bool:
        return str(phase or "") in {"inference", "inference_stage_1", "inference_stage_2", "inference_stage_3"}

    @staticmethod
    def _estimate_progress(phase: str, current_step: int | None, total_steps: int | None) -> int:
        if total_steps is None or total_steps <= 0 or current_step is None:
            if phase == "loading_model":
                return 10
            if phase == "encoding_text":
                return 18
            if phase == "inference_stage_1":
                return 25
            if phase == "inference_stage_2":
                return 70
            if phase == "inference_stage_3":
                return 80
            if phase == "decoding":
                return 90
            if phase == "downloading_output":
                return 95
            if phase == "cancelled":
                return 0
            return 15
        ratio = max(0.0, min(1.0, current_step / total_steps))
        if phase == "loading_model":
            return min(15, 5 + int(ratio * 10))
        if phase == "encoding_text":
            return min(22, 12 + int(ratio * 10))
        if phase == "inference_stage_1":
            return min(68, 20 + int(ratio * 48))
        if phase == "inference_stage_2":
            return min(88, 68 + int(ratio * 20))
        if phase == "inference_stage_3":
            return min(89, 80 + int(ratio * 9))
        if phase == "decoding":
            return min(95, 85 + int(ratio * 10))
        if phase == "downloading_output":
            return min(98, 92 + int(ratio * 6))
        if phase == "cancelled":
            return 0
        return min(90, 20 + int(ratio * 65))


def init(
    *,
    root: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    callbacks: object | None = None,
    cli_args: Sequence[str] = (),
    console_output: bool = True,
    console_isatty: bool = True,
    webui_state: dict[str, Any] | None = None,
) -> WanGPSession:
    """Create and eagerly initialize a reusable WanGP session."""

    return WanGPSession(
        root=root,
        config_path=config_path,
        output_dir=output_dir,
        callbacks=callbacks,
        cli_args=cli_args,
        console_output=console_output,
        console_isatty=console_isatty,
        webui_state=webui_state,
    ).ensure_ready()

def create_gradio_webui_session(plugin) -> Any:
    from shared.api_webui import create_gradio_webui_session as _create_gradio_webui_session

    return _create_gradio_webui_session(plugin, init_fn=init)


def create_gradio_progress_callbacks(progress) -> Any:
    from shared.api_webui import create_gradio_progress_callbacks as _create_gradio_progress_callbacks

    return _create_gradio_progress_callbacks(progress)


@contextlib.contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterator[None]:
    previous = list(sys.argv)
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = previous


def _print_banner_once(module, *, enabled: bool = True) -> None:
    global _BANNER_PRINTED
    if not enabled:
        return
    if _BANNER_PRINTED:
        return
    _BANNER_PRINTED = True
    banner = f"Powered by WanGP v{module.WanGP_version} - a DeepBeepMeep Production\n"
    console = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    if console is not None:
        console.write(banner)
        console.flush()
