from __future__ import annotations

import inspect
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
except Exception:  # pragma: no cover
    PromptSession = None
    KeyBindings = None

from shared.deepy import media_registry, tool_settings as deepy_tool_settings, ui_settings as deepy_ui_settings
from shared.deepy.engine import begin_assistant_turn, get_or_create_assistant_session
from shared.gradio import assistant_chat
from shared.utils.thread_utils import AsyncStream, async_run_in
from shared.utils.utils import get_video_info


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
_VIDEO_EXTENSIONS = {".mp4", ".mkv"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".flac", ".m4a", ".ogg", ".wma"}
_DEEPY_LOGO = (
    "    ____                       ",
    "   / __ \\___  ___  ____  __  __",
    "  / / / / _ \\/ _ \\/ __ \\/ / / /",
    " / /_/ /  __/  __/ /_/ / /_/ / ",
    "/_____/\\___/\\___/ .___/\\__, /  ",
    "               /_/    /____/   ",
)
_TOOL_ALIASES = {
    "gen_image": "gen_image",
    "image": "gen_image",
    "image_gen": "gen_image",
    "generate_image": "gen_image",
    "edit_image": "edit_image",
    "edit": "edit_image",
    "image_edit": "edit_image",
    "gen_video": "gen_video",
    "video": "gen_video",
    "video_gen": "gen_video",
    "generate_video": "gen_video",
}


def _reconfigure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


@dataclass(slots=True)
class DeepyCliDeps:
    controller: Any
    get_server_config: Callable[[], dict[str, Any]]
    get_gen_info: Callable[[dict[str, Any]], dict[str, Any]]
    get_settings_from_file: Callable[[dict[str, Any], str, bool, bool, bool], tuple[Any, bool, bool]]
    load_queue_action: Callable[[Any, dict[str, Any], Any], Any]
    validate_task: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None]
    generate_video: Callable[..., Any]
    default_model_type: str


class _CliEvent:
    target = 1


class _VirtualGallery:
    def __init__(self, deps: DeepyCliDeps, state: dict[str, Any]):
        self._deps = deps
        self._state = state
        self._session = get_or_create_assistant_session(state)

    def _gen(self) -> dict[str, Any]:
        return self._deps.get_gen_info(self._state)

    def _detect_media_type(self, path: str) -> str:
        ext = os.path.splitext(str(path or "").strip())[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            return "image"
        if ext in _VIDEO_EXTENSIONS:
            return "video"
        if ext in _AUDIO_EXTENSIONS:
            return "audio"
        return "any"

    def _resolve_lists(self, audio_only: bool) -> tuple[list[Any], list[Any]]:
        gen = self._gen()
        if audio_only:
            return gen["audio_file_list"], gen["audio_file_settings_list"]
        return gen["file_list"], gen["file_settings_list"]

    def _select_index(self, index: int, audio_only: bool) -> None:
        gen = self._gen()
        file_list, _file_settings_list = self._resolve_lists(audio_only)
        if len(file_list) == 0:
            index = -1
        else:
            index = max(0, min(int(index), len(file_list) - 1))
        if audio_only:
            gen["audio_selected"] = index
            gen["audio_last_selected"] = (index + 1) >= len(file_list)
            gen["current_gallery_source"] = "audio"
            gen["selected_video_time"] = None
        else:
            gen["selected"] = index
            gen["last_selected"] = (index + 1) >= len(file_list)
            gen["current_gallery_source"] = "video"
            selected_path = str(file_list[index] or "").strip() if 0 <= index < len(file_list) else ""
            gen["selected_video_time"] = 0.0 if self._detect_media_type(selected_path) == "video" else None

    def _register(self, path: str, settings: dict[str, Any] | None = None) -> dict[str, Any] | None:
        settings = settings if isinstance(settings, dict) else None
        client_id = "" if settings is None else str(settings.get("client_id", "") or "").strip()
        source = "deepy" if client_id.startswith("ai_") else "wangp"
        return media_registry.register_media(self._session, path, settings=settings, source=source, client_id=client_id)

    def add_path(self, raw_path: str, preferred_type: str = "any") -> dict[str, Any]:
        path = os.path.abspath(os.path.normpath(str(raw_path or "").strip().strip('"')))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        detected_type = self._detect_media_type(path)
        if detected_type == "any":
            raise ValueError("Unsupported media type. Use an image, video, or audio file.")
        preferred_type = str(preferred_type or "any").strip().lower() or "any"
        if preferred_type != "any" and detected_type != preferred_type:
            raise ValueError(f"Expected a {preferred_type} file, got a {detected_type} file.")
        configs, _any_image_or_video, any_audio = self._deps.get_settings_from_file(self._state, path, False, False, False)
        audio_only = detected_type == "audio" or bool(any_audio)
        file_list, file_settings_list = self._resolve_lists(audio_only)
        normalized_path = os.path.normcase(path)
        for index, existing_path in enumerate(file_list):
            if os.path.normcase(str(existing_path or "")) != normalized_path:
                continue
            self._select_index(index, audio_only)
            record = self._register(path, file_settings_list[index] if index < len(file_settings_list) else configs)
            return {"record": record, "added": False}
        file_list.append(path)
        file_settings_list.append(configs if isinstance(configs, dict) else None)
        self._select_index(len(file_list) - 1, audio_only)
        record = self._register(path, configs)
        return {"record": record, "added": True}

    def counts(self) -> tuple[int, int]:
        gen = self._gen()
        return len(gen["file_list"]), len(gen["audio_file_list"])

    def sync_latest_generated(self, before_counts: tuple[int, int]) -> None:
        gen = self._gen()
        before_files, before_audio = before_counts
        if len(gen["audio_file_list"]) > before_audio:
            self._select_index(len(gen["audio_file_list"]) - 1, True)
            self._register(str(gen["audio_file_list"][-1] or ""), gen["audio_file_settings_list"][-1] if gen["audio_file_settings_list"] else None)
            return
        if len(gen["file_list"]) > before_files:
            self._select_index(len(gen["file_list"]) - 1, False)
            self._register(str(gen["file_list"][-1] or ""), gen["file_settings_list"][-1] if gen["file_settings_list"] else None)

    def sync_refresh_path(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        path = str(payload.get("path", "") or "").strip()
        if len(path) == 0:
            return
        detected_type = self._detect_media_type(path)
        if detected_type == "audio":
            file_list = self._gen()["audio_file_list"]
            if path in file_list:
                self._select_index(file_list.index(path), True)
        elif detected_type in {"image", "video"}:
            file_list = self._gen()["file_list"]
            if path in file_list:
                self._select_index(file_list.index(path), False)

    def _iter_records(self, audio_only: bool | None = None) -> list[tuple[bool, int, dict[str, Any], str]]:
        gen = self._gen()
        groups = []
        if audio_only is None or not audio_only:
            file_list, file_settings_list = gen["file_list"], gen["file_settings_list"]
            for index, path in enumerate(file_list):
                record = self._register(str(path or ""), file_settings_list[index] if index < len(file_settings_list) else None)
                if record is not None:
                    groups.append((False, index, record, str(path or "")))
        if audio_only is None or audio_only:
            file_list, file_settings_list = gen["audio_file_list"], gen["audio_file_settings_list"]
            for index, path in enumerate(file_list):
                record = self._register(str(path or ""), file_settings_list[index] if index < len(file_settings_list) else None)
                if record is not None:
                    groups.append((True, index, record, str(path or "")))
        return groups

    def list_lines(self, media_scope: str = "all") -> list[str]:
        scope = str(media_scope or "all").strip().lower()
        if scope not in {"all", "media", "image", "video", "audio"}:
            scope = "all"
        gen = self._gen()
        lines = []
        media_items = self._iter_records(False)
        audio_items = self._iter_records(True)
        groups = []
        if scope == "all":
            groups.append(("Media", media_items))
            groups.append(("Audio", audio_items))
        elif scope == "media":
            groups.append(("Media", media_items))
        elif scope == "image":
            groups.append(("Images", [item for item in media_items if item[2].get("media_type") == "image"]))
        elif scope == "video":
            groups.append(("Videos", [item for item in media_items if item[2].get("media_type") == "video"]))
        elif scope == "audio":
            groups.append(("Audio", audio_items))
        for header, items in groups:
            if len(items) == 0:
                continue
            lines.append(f"{header}:")
            for audio_only, index, record, path in items:
                is_selected = (audio_only and gen["current_gallery_source"] == "audio" and gen["audio_selected"] == index) or (
                    (not audio_only) and gen["current_gallery_source"] == "video" and gen["selected"] == index
                )
                selected_suffix = ""
                if is_selected and record.get("media_type") == "video":
                    selected_suffix = f" @ {float(gen.get('selected_video_time', 0.0) or 0.0):.3f}s"
                marker = "*" if is_selected else " "
                label = str(record.get("label", "") or os.path.basename(path)).strip()
                description = str(record.get("prompt_summary", "") or "").strip()
                if len(description) > 0 and description.casefold() != label.casefold():
                    label = f"{label} - {description}"
                lines.append(f"{marker} {record.get('media_id', '?')} [{index + 1}] {label}{selected_suffix}")
        if len(lines) == 0:
            return ["No media available."]
        return lines

    def _resolve_by_reference(self, reference: str, media_type: str = "any") -> tuple[bool, int] | None:
        ref = str(reference or "").strip()
        if len(ref) == 0:
            return None
        normalized_media_type = media_registry.normalize_media_type(media_type)
        candidates = [
            (audio_only, index, record, path)
            for audio_only, index, record, path in self._iter_records()
            if normalized_media_type == "any" or str(record.get("media_type", "")).strip() == normalized_media_type
        ]
        if ref.isdigit():
            choice = int(ref) - 1
            if 0 <= choice < len(candidates):
                return candidates[choice][0], candidates[choice][1]
            return None
        record = media_registry.get_media_record(self._session, ref)
        if record is not None and (normalized_media_type == "any" or str(record.get("media_type", "")).strip() == normalized_media_type):
            for audio_only, index, candidate, _path in candidates:
                if candidate.get("media_id") == record.get("media_id"):
                    return audio_only, index
        normalized_ref = ref.lower()
        matches = []
        for audio_only, index, record, path in candidates:
            haystack = " ".join(
                [
                    str(record.get("media_id", "") or ""),
                    str(record.get("label", "") or ""),
                    str(record.get("prompt_summary", "") or ""),
                    os.path.basename(path),
                ]
            ).lower()
            if normalized_ref in haystack:
                matches.append((audio_only, index))
        if len(matches) == 1:
            return matches[0]
        return None

    def select(self, reference: str, media_type: str = "any") -> dict[str, Any] | None:
        resolved = self._resolve_by_reference(reference, media_type=media_type)
        if resolved is None:
            return None
        audio_only, index = resolved
        self._select_index(index, audio_only)
        file_list, file_settings_list = self._resolve_lists(audio_only)
        settings = file_settings_list[index] if index < len(file_settings_list) else None
        return self._register(str(file_list[index] or ""), settings)

    def selected_summary(self) -> str:
        gen = self._gen()
        audio_only = gen.get("current_gallery_source", "video") == "audio"
        file_list, file_settings_list = self._resolve_lists(audio_only)
        choice = gen["audio_selected"] if audio_only else gen["selected"]
        if choice is None or choice < 0 or choice >= len(file_list):
            return "No media is currently selected."
        record = self._register(str(file_list[choice] or ""), file_settings_list[choice] if choice < len(file_settings_list) else None)
        if record is None:
            return "No media is currently selected."
        summary = f"Selected {record.get('media_id')}: {record.get('label', os.path.basename(str(file_list[choice] or '')))}"
        if record.get("media_type") == "video":
            frame_text = self.selected_frame_summary()
            summary += f" @ {float(gen.get('selected_video_time', 0.0) or 0.0):.3f}s"
            if frame_text.startswith("Selected frame: "):
                summary += f" ({frame_text[len('Selected frame: '):]})"
        return summary

    def set_selected_time(self, seconds: float) -> str:
        gen = self._gen()
        file_list = gen["file_list"]
        choice = gen["selected"]
        if gen.get("current_gallery_source", "video") != "video" or choice is None or choice < 0 or choice >= len(file_list):
            return "No video is currently selected."
        path = str(file_list[choice] or "").strip()
        if self._detect_media_type(path) != "video":
            return "The selected media is not a video."
        gen["selected_video_time"] = max(0.0, float(seconds))
        return f"Selected video time set to {gen['selected_video_time']:.3f}s."

    def _get_selected_video_state(self) -> tuple[dict[str, Any] | None, str, int, str]:
        gen = self._gen()
        file_list = gen["file_list"]
        file_settings_list = gen["file_settings_list"]
        choice = gen["selected"]
        if gen.get("current_gallery_source", "video") != "video" or choice is None or choice < 0 or choice >= len(file_list):
            return None, "", 0, "No video is currently selected."
        path = str(file_list[choice] or "").strip()
        if self._detect_media_type(path) != "video":
            return None, "", 0, "The selected media is not a video."
        settings = file_settings_list[choice] if choice < len(file_settings_list) else None
        record = self._register(path, settings)
        if record is None:
            return None, "", 0, "No video is currently selected."
        return record, path, choice, ""

    def selected_video_summary(self) -> str:
        record, _path, _choice, error = self._get_selected_video_state()
        if len(error) > 0:
            return error
        return f"Selected video: {record.get('media_id')} ({record.get('label', '')})"

    def selected_frame_summary(self) -> str:
        record, path, _choice, error = self._get_selected_video_state()
        if len(error) > 0:
            return error
        fps, _width, _height, frame_count = get_video_info(path)
        if fps <= 0 or frame_count <= 0:
            return f"Unable to read frame info for {record.get('media_id')}."
        current_time = float(self._gen().get("selected_video_time", 0.0) or 0.0)
        frame_no = max(0, min(frame_count - 1, int(round(current_time * fps))))
        return f"Selected frame: {frame_no} on {record.get('media_id')} @ {current_time:.3f}s ({fps} fps, {frame_count} frames)"

    def set_selected_frame(self, frame_no: int) -> str:
        record, path, _choice, error = self._get_selected_video_state()
        if len(error) > 0:
            return error
        fps, _width, _height, frame_count = get_video_info(path)
        if fps <= 0 or frame_count <= 0:
            return f"Unable to read frame info for {record.get('media_id')}."
        frame_no = int(frame_no)
        if frame_no < 0 or frame_no >= frame_count:
            return f"Frame must be between 0 and {frame_count - 1}."
        seconds = frame_no / float(fps)
        self._gen()["selected_video_time"] = seconds
        return f"Selected frame set to {frame_no} on {record.get('media_id')} @ {seconds:.3f}s."

    def clear_media(self) -> None:
        gen = self._gen()
        gen["file_list"].clear()
        gen["file_settings_list"].clear()
        gen["audio_file_list"].clear()
        gen["audio_file_settings_list"].clear()
        gen["selected"] = -1
        gen["audio_selected"] = -1
        gen["current_gallery_source"] = "video"
        gen["selected_video_time"] = None
        self._session.media_registry.clear()


class DeepyCliSession:
    def __init__(self, deps: DeepyCliDeps):
        self._deps = deps
        self._state = self._build_state()
        self._session = get_or_create_assistant_session(self._state)
        self._gallery = _VirtualGallery(deps, self._state)
        self._last_status_text = ""
        self._assistant_live_print_state: dict[str, dict[str, str]] = {}
        self._interactive = False
        self._prompt_session = None

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

    def _print(self, text: str = "") -> None:
        print(text, flush=True)

    def _supports_terminal_formatting(self) -> bool:
        stream = getattr(sys, "stdout", None)
        return bool(getattr(stream, "isatty", lambda: False)())

    def _style_terminal_text(self, text: str, code: str) -> str:
        rendered = str(text or "")
        if not self._supports_terminal_formatting() or len(rendered) == 0:
            return rendered
        return f"\033[{code}m{rendered}\033[0m"

    def _italicize_terminal_block(self, text: str) -> str:
        lines = []
        for line in str(text or "").split("\n"):
            lines.append(self._style_terminal_text(line, "3") if len(line) > 0 else "")
        return "\n".join(lines)

    def _render_terminal_markdown(self, text: str) -> str:
        rendered_lines = []
        in_code_block = False
        for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            stripped = raw_line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                rendered_lines.append(self._style_terminal_text(raw_line, "96"))
                continue
            line = raw_line
            heading_match = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
            if heading_match is not None:
                heading_text = heading_match.group(2).strip()
                rendered_lines.append(self._style_terminal_text(heading_text, "1"))
                continue
            line = re.sub(r"\*\*([^*\n]+)\*\*", lambda match: self._style_terminal_text(match.group(1), "1"), line)
            line = re.sub(r"`([^`\n]+)`", lambda match: self._style_terminal_text(match.group(1), "96"), line)
            rendered_lines.append(line)
        return "\n".join(rendered_lines)

    def _print_status(self, text: str | None) -> None:
        text = str(text or "").strip()
        if len(text) == 0 or text == self._last_status_text:
            return
        self._last_status_text = text
        self._print(f"[Deepy] {text}")

    def _reset_status(self) -> None:
        self._last_status_text = ""

    def _assistant_print_state(self, message_id: str) -> dict[str, str]:
        return self._assistant_live_print_state.setdefault(str(message_id or "").strip(), {"reasoning": "", "content": ""})

    def _print_assistant_block(self, text: str, *, italic: bool = False, show_prefix: bool = True) -> None:
        rendered = self._render_terminal_markdown(text)
        if italic:
            rendered = self._italicize_terminal_block(rendered)
        if not show_prefix:
            self._print(rendered)
        elif "\n" not in rendered:
            self._print(f"Deepy> {rendered}")
        else:
            self._print("Deepy>")
            self._print(rendered)

    def _print_assistant_delta(self, message_id: str, text: str, *, field: str, italic: bool = False, show_prefix: bool = True) -> bool:
        normalized_text = str(text or "").strip()
        if len(normalized_text) == 0:
            return False
        state = self._assistant_print_state(message_id)
        previous = str(state.get(field, "") or "")
        if normalized_text == previous:
            return False
        if len(previous) > 0 and normalized_text.startswith(previous):
            delta = normalized_text[len(previous):].lstrip("\n")
        else:
            delta = normalized_text
        state[field] = normalized_text
        if len(delta.strip()) == 0:
            return False
        self._print_assistant_block(delta, italic=italic, show_prefix=show_prefix)
        return True

    def _flush_live_assistant_reasoning(self) -> None:
        if int(getattr(self._deps.controller, "get_verbose_level", lambda: 0)() or 0) > 1:
            return
        for record in self._session.chat_transcript:
            if str(record.get("role", "")).strip() != "assistant":
                continue
            message_id = str(record.get("id", "") or "").strip()
            reasoning_text = assistant_chat.get_message_reasoning_content(self._session, message_id).strip()
            self._print_assistant_delta(message_id, reasoning_text, field="reasoning", italic=True, show_prefix=False)

    def _consume_chat_payload(self, payload: str) -> None:
        payload_text = str(payload or "").strip()
        if len(payload_text) == 0:
            return
        try:
            envelope = json.loads(payload_text)
        except Exception:
            return
        batch = envelope.get("batch", None)
        if isinstance(batch, list):
            for item in batch:
                self._consume_chat_payload(json.dumps(item, ensure_ascii=False))
            return
        event = envelope.get("event", envelope)
        if not isinstance(event, dict):
            return
        if event.get("type") == "status":
            status = event.get("status", None)
            if isinstance(status, dict) and status.get("visible", False) and len(str(status.get("text", "")).strip()) > 0:
                status_text = str(status.get("text", "")).strip()
                if status_text.startswith("Using "):
                    self._flush_live_assistant_reasoning()
                self._print_status(status.get("text", ""))
            else:
                self._reset_status()

    def _record_queue_error(self, queue: list[dict[str, Any]], error_text: str) -> None:
        queue_errors = self._state["gen"].setdefault("queue_errors", {})
        for task in list(queue or []):
            params = task.get("params", {}) if isinstance(task, dict) else {}
            client_id = str(params.get("client_id", "") or "").strip()
            if len(client_id) > 0:
                queue_errors[client_id] = (str(error_text), False, False)

    def _run_loaded_queue(self, queue: list[dict[str, Any]]) -> tuple[bool, str]:
        for task in list(queue or []):
            validated_params = self._deps.validate_task(task, self._state)
            if validated_params is None:
                return False, "Task failed validation."
            params = dict(validated_params or {})
            task_stream = AsyncStream()
            task_error = ""

            def worker():
                try:
                    expected_args = set(inspect.signature(self._deps.generate_video).parameters.keys())
                    filtered_params = {key: value for key, value in params.items() if key in expected_args}
                    filtered_params.setdefault("client_id", "")
                    plugin_data = task.get("plugin_data", {}) if isinstance(task, dict) else {}
                    self._deps.generate_video(task, task_stream.output_queue.push, plugin_data=plugin_data, **filtered_params)
                except Exception as exc:
                    traceback.print_exc()
                    task_stream.output_queue.push("error", str(exc))
                finally:
                    task_stream.output_queue.push("exit", None)

            async_run_in("generation", worker)
            last_msg_len = 0
            in_status_line = False
            while True:
                cmd, data = task_stream.output_queue.next()
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

    def _process_inline_queue(self) -> None:
        before_counts = self._gallery.counts()
        self._deps.load_queue_action(None, self._state, _CliEvent())
        queue = list(self._state["gen"].get("queue", []) or [])
        if len(queue) == 0:
            return
        success, error_text = self._run_loaded_queue(queue)
        if not success and len(error_text) > 0:
            self._record_queue_error(queue, error_text)
        self._state["gen"]["queue"].clear()
        self._gallery.sync_latest_generated(before_counts)

    def _send_cmd(self, cmd: str, data: Any = None) -> None:
        if cmd == "chat_output":
            self._consume_chat_payload(str(data or ""))
        elif cmd == "load_queue_trigger":
            self._process_inline_queue()
        elif cmd == "refresh_gallery":
            self._gallery.sync_refresh_path(data)
        elif cmd == "error":
            self._print(f"[ERROR] {data}")

    def _reset_conversation(self) -> None:
        self._session.messages.clear()
        self._session.rendered_token_ids.clear()
        self._session.runtime_snapshot = None
        self._session.chat_html = ""
        self._session.chat_transcript_counter = 0
        self._assistant_live_print_state.clear()
        self._session.current_turn = None
        self._session.interrupt_requested = False
        self._session.drop_state_requested = False
        self._session.interruption_notice = ""
        self._session.runtime_status_note = ""
        self._session.runtime_status_signature = ""
        self._session.rendered_system_prompt_signature = ""
        self._session.rendered_context_window_tokens = 0
        assistant_chat.reset_session_chat(self._session)

    def _print_new_assistant_messages(self, transcript_start: int) -> None:
        printed_any = False
        show_reasoning = int(getattr(self._deps.controller, "get_verbose_level", lambda: 0)() or 0) <= 1
        for record in self._session.chat_transcript[transcript_start:]:
            if str(record.get("role", "")).strip() != "assistant":
                continue
            message_id = str(record.get("id", "") or "").strip()
            reasoning_text = assistant_chat.get_message_reasoning_content(self._session, message_id).strip() if show_reasoning else ""
            text = assistant_chat.get_message_content(self._session, message_id).strip()
            if self._print_assistant_delta(message_id, reasoning_text, field="reasoning", italic=True, show_prefix=False):
                printed_any = True
                if len(text) > 0:
                    self._print()
            if self._print_assistant_delta(message_id, text, field="content", italic=False):
                printed_any = True
        if not printed_any:
            self._print("Deepy> (no final response)")

    def _get_tool_ui_settings(self) -> dict[str, Any]:
        if isinstance(self._session.tool_ui_settings, dict) and len(self._session.tool_ui_settings) > 0:
            return deepy_ui_settings.normalize_assistant_tool_ui_settings(**self._session.tool_ui_settings)
        return deepy_ui_settings.normalize_assistant_tool_ui_settings()

    def _update_tool_ui_settings(self, **changes) -> dict[str, Any]:
        settings = self._get_tool_ui_settings()
        settings.update(changes)
        self._deps.controller.update_tool_ui_settings(self._state, **settings)
        return self._get_tool_ui_settings()

    def _resolve_tool_name(self, value: str) -> str | None:
        return _TOOL_ALIASES.get(str(value or "").strip().lower())

    def _parse_dimensions(self, text: str) -> tuple[int, int] | None:
        cleaned = str(text or "").strip().lower().replace(",", " ").replace("x", " ")
        parts = [part for part in cleaned.split() if len(part) > 0]
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except Exception:
            return None

    def _format_template_value(self, value: str) -> str:
        resolved = str(value or "").strip()
        if len(resolved) == 0:
            return "(unset)"
        if os.path.isfile(resolved):
            return os.path.abspath(resolved)
        return resolved

    def _tool_settings_lines(self) -> list[str]:
        settings = self._get_tool_ui_settings()
        return [
            f"Template properties: {'on' if settings['use_template_properties'] else 'off'}",
            f"Default size: {settings['width']}x{settings['height']}",
            f"Video frames: {settings['num_frames']}",
            f"gen_image template: {self._format_template_value(settings['image_generator_variant'])}",
            f"edit_image template: {self._format_template_value(settings['image_editor_variant'])}",
            f"gen_video template: {self._format_template_value(settings['video_generator_variant'])}",
        ]

    def _print_tool_settings(self) -> None:
        for line in self._tool_settings_lines():
            self._print(line)

    def _build_prompt_session(self, input=None, output=None):
        if PromptSession is None or KeyBindings is None:
            return None
        bindings = KeyBindings()

        @bindings.add("enter")
        def _submit(event):
            event.current_buffer.validate_and_handle()

        @bindings.add("c-j")
        def _newline_ctrl_j(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("escape", "c-j")
        def _newline_ctrl_enter_windows(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("escape", "enter")
        def _newline_escape_enter(event):
            event.current_buffer.insert_text("\n")

        return PromptSession(
            input=input,
            output=output,
            multiline=True,
            key_bindings=bindings,
            prompt_continuation=lambda width, _line_no, _wrap_count: "... ".rjust(width),
        )

    def _read_line(self) -> str:
        if not self._interactive:
            return input("")
        if self._prompt_session is None:
            self._prompt_session = self._build_prompt_session()
        if self._prompt_session is None:
            return input("deepy> ")
        return self._prompt_session.prompt("deepy> ")

    def ask(self, text: str) -> None:
        prompt = str(text or "").strip()
        if len(prompt) == 0:
            return
        if not self._deps.controller.is_available():
            self._print(self._deps.controller.requirement_error_text())
            return
        transcript_start = len(self._session.chat_transcript)
        user_message_id, _event = assistant_chat.add_user_message(self._session, prompt, queued=False)
        begin_assistant_turn(self._session, user_message_id, prompt)
        self._reset_status()
        tools = self._deps.controller.create_tools(self._state, self._send_cmd, session=self._session)
        try:
            self._deps.controller.run_assistant_prompt_turn(self._state, None, "AK", [prompt], 0, override_profile=3.5, send_cmd=self._send_cmd, tools=tools)
        except Exception as exc:
            self._print(f"[ERROR] {exc}")
            return
        self._reset_status()
        self._print_new_assistant_messages(transcript_start)

    def _handle_add_command(self, value: str, preferred_type: str = "any") -> None:
        try:
            result = self._gallery.add_path(value, preferred_type=preferred_type)
        except Exception as exc:
            self._print(f"[ERROR] {exc}")
            return
        record = result.get("record", None)
        action = "Added" if result.get("added", False) else "Selected"
        if isinstance(record, dict):
            self._print(f"{action} {record.get('media_id')}: {record.get('label', '')}")
        else:
            self._print(f"{action} file.")

    def _handle_command(self, line: str) -> bool:
        command, _, rest = str(line or "").strip().partition(" ")
        argument = rest.strip()
        command = command.lower()
        if command in {"/quit", "/exit"}:
            return False
        if command == "/help":
            self._print("Commands:")
            self._print("  /add <path>     Add and select an image, video, or audio file")
            self._print("  /image <path>   Add and select an image file")
            self._print("  /video <path>   Add and select a video file")
            self._print("  /audio <path>   Add and select an audio file")
            self._print("  /list [scope]   List known media; scope: all, media, image, video, audio")
            self._print("  /media [scope]  Alias for /list")
            self._print("  /select <ref>   Select media by id, index, or name fragment")
            self._print("  /select-video <media_id>  Select a video by media id")
            self._print("  /time <secs>    Set the selected video playback time")
            self._print("  /frame [index]  Show or set the selected video frame (0-based)")
            self._print("  /selected       Show the selected media")
            self._print("  /selected-video Show the selected video media id")
            self._print("  /settings       Show current CLI generation settings")
            self._print("  /size [WxH]     Show or set default generation size and disable template properties")
            self._print("  /frames [count] Show or set default gen_video frame count and disable template properties")
            self._print("  /template <tool> <variant-or-json-path>  Set the preset for gen_image, edit_image, or gen_video")
            self._print("  /templates [tool]  List available preset variants")
            self._print("  /template-props [on|off]  Show or toggle template resolution/frame properties")
            self._print("  /reset          Clear the Deepy conversation but keep media")
            self._print("  /clear-media    Remove all virtual gallery media")
            self._print("  /quit           Exit the session")
            if self._interactive:
                self._print("Prompt entry:")
                self._print("  Enter           Send the current prompt")
                self._print("  Ctrl+Enter      Insert a newline on Windows terminals that expose it")
                self._print("  Ctrl+J          Insert a newline fallback")
                self._print("  Alt+Enter       Insert a newline")
                self._print("  Shift+Enter     Not available here; the console reports it as plain Enter")
            return True
        if command == "/add":
            self._handle_add_command(argument, "any")
            return True
        if command == "/image":
            self._handle_add_command(argument, "image")
            return True
        if command == "/video":
            self._handle_add_command(argument, "video")
            return True
        if command == "/audio":
            self._handle_add_command(argument, "audio")
            return True
        if command in {"/list", "/media"}:
            for one_line in self._gallery.list_lines(argument or "all"):
                self._print(one_line)
            return True
        if command == "/select":
            record = self._gallery.select(argument)
            if record is None:
                self._print("Unable to resolve media selection.")
            else:
                self._print(f"Selected {record.get('media_id')}: {record.get('label', '')}")
            return True
        if command == "/select-video":
            record = self._gallery.select(argument, media_type="video")
            if record is None:
                self._print("Unable to resolve video selection.")
            else:
                self._print(f"Selected video {record.get('media_id')}: {record.get('label', '')}")
            return True
        if command == "/time":
            try:
                seconds = float(argument)
            except Exception:
                self._print("Provide a numeric time in seconds.")
            else:
                self._print(self._gallery.set_selected_time(seconds))
            return True
        if command == "/frame":
            if len(argument) == 0:
                self._print(self._gallery.selected_frame_summary())
                return True
            try:
                frame_no = int(argument)
            except Exception:
                self._print("Provide a numeric frame index.")
            else:
                self._print(self._gallery.set_selected_frame(frame_no))
            return True
        if command == "/selected":
            self._print(self._gallery.selected_summary())
            return True
        if command == "/selected-video":
            self._print(self._gallery.selected_video_summary())
            return True
        if command == "/settings":
            self._print_tool_settings()
            return True
        if command in {"/size", "/resolution"}:
            if len(argument) == 0:
                settings = self._get_tool_ui_settings()
                self._print(f"Default size: {settings['width']}x{settings['height']} (template properties {'on' if settings['use_template_properties'] else 'off'})")
                return True
            dimensions = self._parse_dimensions(argument)
            if dimensions is None:
                self._print("Use /size <width>x<height>.")
                return True
            try:
                settings = self._update_tool_ui_settings(width=dimensions[0], height=dimensions[1], use_template_properties=False)
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"Default size set to {settings['width']}x{settings['height']}. Template properties disabled.")
            return True
        if command == "/frames":
            if len(argument) == 0:
                settings = self._get_tool_ui_settings()
                self._print(f"Video frames: {settings['num_frames']} (template properties {'on' if settings['use_template_properties'] else 'off'})")
                return True
            try:
                frame_count = int(argument)
            except Exception:
                self._print("Use /frames <count>.")
                return True
            try:
                settings = self._update_tool_ui_settings(num_frames=frame_count, use_template_properties=False)
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"Default gen_video frame count set to {settings['num_frames']}. Template properties disabled.")
            return True
        if command in {"/template", "/preset"}:
            tool_name, _, tool_value = argument.partition(" ")
            resolved_tool = self._resolve_tool_name(tool_name)
            if resolved_tool is None or len(tool_value.strip()) == 0:
                self._print("Use /template <gen_image|edit_image|gen_video> <variant-or-json-path>.")
                return True
            try:
                deepy_tool_settings.refresh_tool_presets()
                if resolved_tool == "gen_image":
                    settings = self._update_tool_ui_settings(image_generator_variant=tool_value.strip())
                    value = settings["image_generator_variant"]
                elif resolved_tool == "edit_image":
                    settings = self._update_tool_ui_settings(image_editor_variant=tool_value.strip())
                    value = settings["image_editor_variant"]
                else:
                    settings = self._update_tool_ui_settings(video_generator_variant=tool_value.strip())
                    value = settings["video_generator_variant"]
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"{resolved_tool} template set to {self._format_template_value(value)}")
            return True
        if command == "/templates":
            if len(argument) == 0:
                for tool_name in ("gen_image", "edit_image", "gen_video"):
                    variants = deepy_tool_settings.list_tool_variants(tool_name)
                    suffix = ", ".join(variants) if variants else "(none)"
                    self._print(f"{tool_name}: {suffix}")
                return True
            resolved_tool = self._resolve_tool_name(argument)
            if resolved_tool is None:
                self._print("Use /templates [gen_image|edit_image|gen_video].")
                return True
            variants = deepy_tool_settings.list_tool_variants(resolved_tool)
            self._print(f"{resolved_tool}: {', '.join(variants) if variants else '(none)'}")
            return True
        if command == "/template-props":
            if len(argument) == 0:
                settings = self._get_tool_ui_settings()
                self._print(f"Template properties are {'on' if settings['use_template_properties'] else 'off'}.")
                return True
            normalized = argument.strip().lower()
            if normalized not in {"on", "off"}:
                self._print("Use /template-props on|off.")
                return True
            try:
                settings = self._update_tool_ui_settings(use_template_properties=normalized == "on")
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"Template properties {'enabled' if settings['use_template_properties'] else 'disabled'}.")
            return True
        if command == "/reset":
            self._reset_conversation()
            self._print("Deepy conversation reset.")
            return True
        if command == "/clear-media":
            self._gallery.clear_media()
            self._print("Virtual galleries cleared.")
            return True
        self._print(f"Unknown command: {command}. Use /help.")
        return True

    def run(self) -> int:
        if not self._deps.controller.is_available():
            self._print(self._deps.controller.requirement_error_text())
            return 1
        self._interactive = bool(getattr(__import__("sys").stdin, "isatty", lambda: False)())
        for line in _DEEPY_LOGO:
            self._print(line)
        preload_runtime = getattr(self._deps.controller, "preload_cli_runtime", None)
        if callable(preload_runtime):
            self._print("[Deepy] Preloading runtime...")
            try:
                preload_result = preload_runtime(self._state, override_profile=3.5)
            except Exception as exc:
                self._print(f"[ERROR] Deepy preload failed: {exc}")
                return 1
            warmed_vllm = bool((preload_result or {}).get("warmed_vllm", False)) if isinstance(preload_result, dict) else False
            self._print("[Deepy] Prompt enhancer and vLLM are ready." if warmed_vllm else "[Deepy] Prompt enhancer is ready.")
        self._print("Deepy CLI session. Use /help for commands.")
        if self._interactive and PromptSession is not None:
            self._print("Multiline input: Enter sends, Ctrl+Enter or Alt+Enter inserts a newline, Ctrl+J remains available as fallback.")
        while True:
            try:
                line = self._read_line()
            except EOFError:
                self._print()
                return 0
            except KeyboardInterrupt:
                self._print()
                return 130
            line = str(line or "").strip()
            if len(line) == 0:
                continue
            if line.startswith("/"):
                if not self._handle_command(line):
                    return 0
                continue
            self.ask(line)


def run_deepy_cli_session(deps: DeepyCliDeps) -> int:
    _reconfigure_stdio()
    return DeepyCliSession(deps).run()


__all__ = ["DeepyCliDeps", "DeepyCliSession", "run_deepy_cli_session"]
