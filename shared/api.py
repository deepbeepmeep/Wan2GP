"""Lightweight in-process API wrapper around WanGP generation."""

from __future__ import annotations

import contextlib
import copy
import importlib
import inspect
import io
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image

from shared.utils.process_locks import set_main_generation_running
from shared.utils.thread_utils import AsyncStream

_RUNTIME_LOCK = threading.RLock()
_GENERATION_LOCK = threading.RLock()
_RUNTIME: "_WanGPRuntime | None" = None
_BANNER_PRINTED = False


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
class GenerationResult:
    success: bool
    generated_files: list[str]
    errors: list["GenerationError"]
    total_tasks: int
    successful_tasks: int
    failed_tasks: int


@dataclass(frozen=True)
class GenerationError:
    message: str
    task_index: int | None = None
    task_id: Any = None
    stage: str | None = None

    def __str__(self) -> str:
        return self.message


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
        self.events = SessionStream()
        self._done = threading.Event()
        self._cancel_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: GenerationResult | None = None

    def _bind_thread(self, thread: threading.Thread) -> None:
        self._thread = thread

    def _set_result(self, result: GenerationResult) -> None:
        self._result = result
        self._done.set()

    def cancel(self) -> None:
        self._cancel_requested.set()

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
        )

    def join(self, timeout: float | None = None) -> GenerationResult:
        return self.result(timeout=timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()


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
    ) -> None:
        self._root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        self._config_path = Path(config_path).resolve() if config_path is not None else (self._root / "wgp_config.json").resolve()
        self._output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self._callbacks = callbacks
        self._cli_args = tuple(str(arg) for arg in cli_args)
        self._console_output = bool(console_output)
        self._console_isatty = bool(console_isatty)
        self._state = self._create_headless_state()
        self._active_job: SessionJob | None = None
        self._job_lock = threading.Lock()
        self._attachment_keys: tuple[str, ...] | None = None

    def ensure_ready(self) -> "WanGPSession":
        self._ensure_runtime()
        return self

    def submit(self, source: str | os.PathLike[str] | dict[str, Any] | list[dict[str, Any]]) -> SessionJob:
        tasks = self._normalize_source(source, caller_base_path=self._get_caller_base_path())
        return self._submit_tasks(tasks)

    def submit_task(self, settings: dict[str, Any]) -> SessionJob:
        caller_base_path = self._get_caller_base_path()
        task = self._normalize_task(settings, task_index=1)
        return self._submit_tasks([self._absolutize_task_paths(task, caller_base_path)])

    def submit_manifest(self, settings_list: list[dict[str, Any]]) -> SessionJob:
        caller_base_path = self._get_caller_base_path()
        tasks = [
            self._absolutize_task_paths(self._normalize_task(settings, task_index=index + 1), caller_base_path)
            for index, settings in enumerate(settings_list)
        ]
        return self._submit_tasks(tasks)

    def run(self, source: str | os.PathLike[str] | dict[str, Any] | list[dict[str, Any]]) -> GenerationResult:
        return self.submit(source).result()

    def run_task(self, settings: dict[str, Any]) -> GenerationResult:
        return self.submit_task(settings).result()

    def run_manifest(self, settings_list: list[dict[str, Any]]) -> GenerationResult:
        return self.submit_manifest(settings_list).result()

    def close(self) -> None:
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
            },
            "loras": [],
        }

    def _submit_tasks(self, tasks: list[dict[str, Any]]) -> SessionJob:
        with self._job_lock:
            if self._active_job is not None and not self._active_job.done:
                raise RuntimeError("WanGP session already has a generation in progress")
            job = SessionJob(self)
            thread = threading.Thread(
                target=self._run_job,
                args=(job, copy.deepcopy(tasks)),
                daemon=True,
                name="wangp-session-job",
            )
            job._bind_thread(thread)
            self._active_job = job
            thread.start()
            return job

    def _run_job(self, job: SessionJob, tasks: list[dict[str, Any]]) -> None:
        stream = AsyncStream()
        gen = self._state["gen"]
        worker_done = threading.Event()
        base_file_count = len(gen["file_list"])
        base_audio_count = len(gen["audio_file_list"])
        total_tasks = len(tasks)
        runtime: _WanGPRuntime | None = None
        task_summary: dict[str, Any] = {
            "errors": [],
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_tasks": total_tasks,
        }

        try:
            runtime = self._ensure_runtime()
            with _GENERATION_LOCK, _pushd(runtime.root):
                self._configure_runtime(runtime)
                self._prepare_state_for_run(tasks)
                job.events.put("started", {"tasks": len(tasks)})

                def worker() -> None:
                    stdout_capture = _OutputCapture(
                        "stdout",
                        lambda stream_name, line: self._emit_stream(job, stream_name, line),
                        console=sys.__stdout__ if self._console_output else None,
                        console_isatty=self._console_isatty,
                    )
                    stderr_capture = _OutputCapture(
                        "stderr",
                        lambda stream_name, line: self._emit_stream(job, stream_name, line),
                        console=sys.__stderr__ if self._console_output else None,
                        console_isatty=self._console_isatty,
                    )
                    try:
                        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                            self._run_tasks_worker(runtime.module, tasks, stream, job, task_summary)
                    except BaseException as exc:
                        failure = self._make_generation_error(
                            exc,
                            task_index=None,
                            task_id=None,
                            stage="runtime",
                        )
                        task_summary["errors"].append(failure)
                        stream.output_queue.push("error", failure)
                    finally:
                        stdout_capture.flush()
                        stderr_capture.flush()
                        stream.output_queue.push("worker_exit", None)
                        worker_done.set()

                worker_thread = threading.Thread(target=worker, daemon=True, name="wangp-session-worker")
                worker_thread.start()

                while True:
                    if job.cancel_requested:
                        self._request_cancel_unlocked(runtime.module)
                    item = stream.output_queue.pop()
                    if item is None:
                        if worker_done.is_set() and not worker_thread.is_alive():
                            break
                        time.sleep(0.01)
                        continue
                    command, data = item
                    if command == "worker_exit":
                        break
                    self._handle_command(job, runtime.module, tasks, command, data)

                worker_thread.join(timeout=0.1)
                outputs = self._collect_outputs(base_file_count, base_audio_count)
                if job.cancel_requested and not task_summary["errors"]:
                    task_summary["errors"].append(
                        GenerationError(message="Generation was cancelled", stage="cancelled")
                    )
                    task_summary["failed_tasks"] = max(task_summary["failed_tasks"], 1)
                result = GenerationResult(
                    success=not task_summary["errors"],
                    generated_files=outputs,
                    errors=list(task_summary["errors"]),
                    total_tasks=task_summary["total_tasks"],
                    successful_tasks=task_summary["successful_tasks"],
                    failed_tasks=task_summary["failed_tasks"],
                )
                job.events.put("completed", result)
                self._emit_callback("on_complete", result)
                job._set_result(result)
        except BaseException as exc:
            failure = self._make_generation_error(exc, task_index=None, task_id=None, stage="runtime")
            result = GenerationResult(
                success=False,
                generated_files=[],
                errors=[failure],
                total_tasks=total_tasks,
                successful_tasks=task_summary["successful_tasks"],
                failed_tasks=max(task_summary["failed_tasks"], 1 if total_tasks > 0 else 0),
            )
            job.events.put("error", failure)
            self._emit_callback("on_error", failure)
            job.events.put("completed", result)
            self._emit_callback("on_complete", result)
            job._set_result(result)
        finally:
            job.events.close()
            if runtime is not None:
                self._reset_state_after_run()
            with self._job_lock:
                if self._active_job is job:
                    self._active_job = None

    def _run_tasks_worker(
        self,
        wgp,
        tasks: list[dict[str, Any]],
        stream: AsyncStream,
        job: SessionJob,
        task_summary: dict[str, Any],
    ) -> None:
        expected_args = set(inspect.signature(wgp.generate_video).parameters.keys())
        total_tasks = len(tasks)

        for task_index, task in enumerate(tasks, start=1):
            if job.cancel_requested:
                break

            self._state["gen"]["prompt_no"] = task_index
            self._state["gen"]["prompts_max"] = total_tasks
            self._state["gen"]["queue"] = tasks
            task_id = task.get("id")
            task_errors: list[GenerationError] = []

            def send_cmd(command: str, data: Any = None) -> None:
                if command == "error":
                    failure = self._make_generation_error(
                        data,
                        task_index=task_index,
                        task_id=task_id,
                        stage="generation",
                    )
                    task_errors.append(failure)
                    stream.output_queue.push("error", failure)
                    return
                stream.output_queue.push(command, data)

            validated_settings, validation_error = wgp.validate_task(task, self._state)
            if validated_settings is None:
                failure = GenerationError(
                    message=validation_error or f"Task {task_index} failed validation",
                    task_index=task_index,
                    task_id=task_id,
                    stage="validation",
                )
                task_summary["errors"].append(failure)
                task_summary["failed_tasks"] += 1
                stream.output_queue.push("error", failure)
                continue

            task_settings = validated_settings.copy()
            task_settings["state"] = self._state
            filtered_params = {key: value for key, value in task_settings.items() if key in expected_args}
            plugin_data = task.get("plugin_data", {})
            try:
                success = wgp.generate_video(task, send_cmd, plugin_data=plugin_data, **filtered_params)
            except BaseException as exc:
                if not task_errors:
                    task_errors.append(
                        self._make_generation_error(
                            exc,
                            task_index=task_index,
                            task_id=task_id,
                            stage="generation",
                        )
                    )
                    stream.output_queue.push("error", task_errors[-1])
                success = False

            if self._state["gen"].get("abort", False) or job.cancel_requested:
                task_errors.append(
                    GenerationError(
                        message="Generation was cancelled",
                        task_index=task_index,
                        task_id=task_id,
                        stage="cancelled",
                    )
                )
                stream.output_queue.push("error", task_errors[-1])
                task_summary["errors"].extend(task_errors)
                task_summary["failed_tasks"] += 1
                break

            if task_errors:
                task_summary["errors"].extend(task_errors)
                task_summary["failed_tasks"] += 1
                continue

            if not success:
                failure = GenerationError(
                    message=f"Task {task_index} did not complete successfully",
                    task_index=task_index,
                    task_id=task_id,
                    stage="generation",
                )
                task_summary["errors"].append(failure)
                task_summary["failed_tasks"] += 1
                stream.output_queue.push("error", failure)
                continue

            task_summary["successful_tasks"] += 1

    def _handle_command(self, job: SessionJob, wgp, tasks: list[dict[str, Any]], command: str, data: Any) -> None:
        if command == "progress":
            progress = self._build_progress_update(data)
            job.events.put("progress", progress)
            self._emit_callback("on_progress", progress)
            return
        if command == "preview":
            preview = self._build_preview_update(wgp, tasks, data)
            if preview is not None:
                job.events.put("preview", preview)
                self._emit_callback("on_preview", preview)
            return
        if command == "status":
            text = str(data or "")
            job.events.put("status", text)
            self._emit_callback("on_status", text)
            return
        if command == "info":
            text = str(data or "")
            job.events.put("info", text)
            self._emit_callback("on_info", text)
            return
        if command == "output":
            job.events.put("output", data)
            self._emit_callback("on_output", data)
            return
        if command == "refresh_models":
            job.events.put("refresh_models", data)
            return
        if command == "error":
            error = data if isinstance(data, GenerationError) else self._make_generation_error(data)
            job.events.put("error", error)
            self._emit_callback("on_error", error)
            return
        job.events.put(command, data)

    def _build_progress_update(self, data: Any) -> ProgressUpdate:
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
        progress_phase = self._state["gen"].get("progress_phase")
        if isinstance(progress_phase, tuple) and progress_phase:
            raw_phase = str(progress_phase[0] or "")
        phase = self._normalize_phase(raw_phase or status)
        progress = self._estimate_progress(phase, current_step, total_steps)
        return ProgressUpdate(
            phase=phase,
            status=status,
            progress=progress,
            current_step=current_step,
            total_steps=total_steps,
            raw_phase=raw_phase,
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
        self._emit_callback("on_stream", message)

    def _emit_callback(self, method_name: str, payload: Any) -> None:
        callback = self._callbacks
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
        settings = normalized["params"]
        if isinstance(settings, dict):
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
        path = Path(value)
        if path.is_absolute():
            return str(path.resolve())
        return str((caller_base_path / path).resolve())

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
            _print_banner_once(module)
            return _RUNTIME

    @staticmethod
    def _normalize_phase(text: str | None) -> str:
        lowered = str(text or "").lower()
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
) -> WanGPSession:
    """Create and eagerly initialize a reusable WanGP session."""

    return WanGPSession(
        root=root,
        config_path=config_path,
        output_dir=output_dir,
        callbacks=callbacks,
        cli_args=cli_args,
        console_output=console_output,
    ).ensure_ready()


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


def _print_banner_once(module) -> None:
    global _BANNER_PRINTED
    if _BANNER_PRINTED:
        return
    _BANNER_PRINTED = True
    banner = f"Powered by WanGP v{module.WanGP_version} - a DeepBeepMeep Production\n"
    console = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    if console is not None:
        console.write(banner)
        console.flush()
