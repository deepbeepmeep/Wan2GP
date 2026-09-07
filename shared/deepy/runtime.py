"""Browser-independent state and generation execution shared by CLI and web."""
from __future__ import annotations

import inspect
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from shared.utils.thread_utils import AsyncStream, async_run_in


@dataclass(slots=True)
class RuntimeCallbacks:
    handlers: dict[str, Any] = field(default_factory=dict)

    def _iter_handlers(self, name: str) -> tuple[Callable[..., Any], ...]:
        registered = self.handlers.get(str(name or "").strip(), ())
        if callable(registered):
            return (registered,)
        if isinstance(registered, (list, tuple)):
            return tuple(handler for handler in registered if callable(handler))
        return ()

    def emit(self, name: str, *args, **kwargs) -> list[Any]:
        return [handler(*args, **kwargs) for handler in self._iter_handlers(name)]

    def emit_first(self, name: str, *args, **kwargs) -> Any:
        for handler in self._iter_handlers(name):
            return handler(*args, **kwargs)
        return None


@dataclass(slots=True)
class RuntimeDeps:
    controller: Any
    get_server_config: Callable[[], dict[str, Any]]
    get_gen_info: Callable[[dict[str, Any]], dict[str, Any]]
    get_settings_from_file: Callable[[dict[str, Any], str, bool, bool, bool], tuple[Any, bool, bool]]
    load_queue_action: Callable[[Any, dict[str, Any], Any], Any]
    validate_task: Callable[[dict[str, Any], dict[str, Any]], tuple[dict[str, Any] | None, str]]
    generate_media: Callable[..., Any]
    default_model_type: str
    callbacks: RuntimeCallbacks = field(default_factory=RuntimeCallbacks)


class _CliEvent:
    target = 1


class GenerationRuntime:
    def _generation_event(self, cmd, data):
        """Optional publication hook; execution never depends on a subscriber."""

    def _build_state(self) -> dict[str, Any]:
        server_config = self._deps.get_server_config()
        return {
            "active_form": "add",
            "model_type": self._deps.default_model_type,
            "gen": {
                "queue": [],
                "queue_errors": {},
                "in_progress": False,
                "file_list": [],
                "file_settings_list": [],
                "audio_file_list": [],
                "audio_file_settings_list": [],
                "selected": -1,
                "audio_selected": -1,
                "last_selected": True,
                "audio_last_selected": True,
                "last_was_audio": False,
                "current_gallery_source": "video",
                "selected_video_time": None,
                "prompt_no": 0,
                "prompts_max": 0,
                "repeat_no": 0,
                "total_generation": 1,
                "window_no": 0,
                "total_windows": 0,
                "progress_status": "",
            },
            "loras": [],
            "last_model_per_family": dict(server_config.get("last_model_per_family", {}) or {}),
            "last_model_per_type": dict(server_config.get("last_model_per_type", {}) or {}),
            "last_resolution_per_group": dict(server_config.get("last_resolution_per_group", {}) or {}),
        }

    def _record_queue_error(self, queue: list[dict[str, Any]], error_text: str) -> None:
        queue_errors = self._state["gen"].setdefault("queue_errors", {})
        for task in list(queue or []):
            params = task.get("params", {}) if isinstance(task, dict) else {}
            client_id = str(params.get("client_id", "") or "").strip()
            if len(client_id) > 0:
                queue_errors[client_id] = (str(error_text), False, False)

    def _run_loaded_queue(self, queue: list[dict[str, Any]]) -> tuple[bool, str]:
        for task in list(queue or []):
            validated_params, validation_error = self._deps.validate_task(task, self._state)
            if validated_params is None:
                return False, validation_error or "Task failed validation."
            params = dict(validated_params or {})
            task_stream = AsyncStream()
            task_error = ""

            def worker():
                try:
                    expected_args = set(inspect.signature(self._deps.generate_media).parameters.keys())
                    filtered_params = {key: value for key, value in params.items() if key in expected_args}
                    filtered_params.setdefault("client_id", "")
                    plugin_data = task.get("plugin_data", {}) if isinstance(task, dict) else {}
                    self._deps.generate_media(task, task_stream.output_queue.push, plugin_data=plugin_data, **filtered_params)
                except Exception as exc:
                    traceback.print_exc()
                    task_stream.output_queue.push("error", str(exc))
                finally:
                    task_stream.output_queue.push("exit", None)

            self._generation_event("status", "Preparing generation…")
            async_run_in("generation", worker)
            last_msg_len = 0
            in_status_line = False
            while True:
                cmd, data = task_stream.output_queue.next()
                self._generation_event(cmd, data)
                if cmd == "exit":
                    if in_status_line:
                        self._print()
                    break
                if cmd == "error":
                    task_error = str(data or "Generation failed.")
                    self._print(f"[ERROR] {task_error}")
                    in_status_line = False
                    continue
                if cmd == "progress" and isinstance(data, list) and len(data) >= 2:
                    if isinstance(data[0], tuple):
                        step, total = data[0]
                        msg = data[1] if len(data) > 1 else ""
                    else:
                        step, total = 0, 1
                        msg = data[1] if len(data) > 1 else str(data[0])
                    status_line = f"\r[{step}/{total}] {msg}"
                    print(status_line.ljust(max(last_msg_len, len(status_line))), end="", flush=True)
                    last_msg_len = len(status_line)
                    in_status_line = True
                    continue
                if cmd == "status":
                    text = str(data or "")
                    if "Loading" in text:
                        if in_status_line:
                            self._print()
                            in_status_line = False
                            last_msg_len = 0
                        self._print(text)
                    else:
                        status_line = f"\r{text}"
                        print(status_line.ljust(max(last_msg_len, len(status_line))), end="", flush=True)
                        last_msg_len = len(status_line)
                        in_status_line = True
                    continue
                if cmd == "info":
                    if in_status_line:
                        self._print()
                        in_status_line = False
                        last_msg_len = 0
                    self._print(str(data or ""))
            if len(task_error) > 0:
                return False, task_error
        return True, ""

    def _process_inline_queue(self, payload: Any = None) -> None:
        before_counts = self._gallery.counts()
        self._deps.load_queue_action(None, self._state, _CliEvent())
        queue = list(self._state["gen"].get("queue", []) or [])
        if len(queue) == 0:
            return
        requested_client_id = ""
        if isinstance(payload, dict):
            requested_client_id = str(payload.get("client_id", "") or "").strip()
        self._active_generation_client_id = requested_client_id or str(queue[0].get("params", {}).get("client_id", "") or "").strip()
        try:
            success, error_text = self._run_loaded_queue(queue)
            if not success and len(error_text) > 0:
                self._record_queue_error(queue, error_text)
        finally:
            self._active_generation_client_id = ""
            self._state["gen"]["queue"].clear()
            self._gallery.sync_latest_generated(before_counts)

