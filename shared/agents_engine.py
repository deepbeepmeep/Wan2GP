from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image

from shared.assistant_config import (
    DEEPY_VRAM_ALWAYS,
    DEEPY_VRAM_UNLOAD,
    DEEPY_VRAM_UNLOAD_ON_REQUEST,
    normalize_deepy_vram_mode,
)
from shared.deepy import DEFAULT_SYSTEM_PROMPT as ASSISTANT_SYSTEM_PROMPT
from shared.deepy import media_registry, tool_settings as deepy_tool_settings, vision as deepy_vision
from shared.gradio import assistant_chat
from shared.prompt_enhancer import qwen35_text
from shared.prompt_enhancer.qwen35_assistant_runtime import (
    Qwen35AssistantRuntime,
    extract_tool_calls,
    render_assistant_messages,
    render_text_user_turn_suffix,
    strip_inline_tool_call_text,
    strip_tool_blocks,
    strip_trailing_stop_markup,
)


ASSISTANT_DEBUG = True
MAX_TOOL_CALLS_PER_TURN = 4
MAX_MODEL_PASSES_PER_TURN = 6

_TOOL_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}
_AI_GEN_NO = 0


def _json_type_from_annotation(annotation) -> str:
    annotation_name = getattr(annotation, "__name__", str(annotation))
    return _TOOL_TYPE_MAP.get(annotation_name, "string")


def assistant_tool(
    name: str | None = None,
    description: str = "",
    parameters: dict[str, dict[str, Any]] | None = None,
    display_name: str | None = None,
    pause_runtime: bool = True,
    pause_reason: str = "tool",
):
    def decorator(func):
        func._assistant_tool = {
            "name": str(name or func.__name__).strip(),
            "display_name": str(display_name or name or func.__name__).strip(),
            "description": str(description or "").strip(),
            "parameters": dict(parameters or {}),
            "pause_runtime": bool(pause_runtime),
            "pause_reason": str(pause_reason or "tool").strip() or "tool",
        }
        return func

    return decorator


@dataclass(slots=True)
class AssistantSessionState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    rendered_token_ids: list[int] = field(default_factory=list)
    runtime_snapshot: dict[str, Any] | None = None
    media_registry: list[dict[str, Any]] = field(default_factory=list)
    media_registry_counter: int = 0
    chat_html: str = ""
    chat_transcript: list[dict[str, Any]] = field(default_factory=list)
    chat_transcript_counter: int = 0
    interrupt_requested: bool = False
    drop_state_requested: bool = False
    worker_active: bool = False
    control_queue: Any | None = None
    queued_job_count: int = 0
    chat_epoch: int = 0
    release_vram_callback: Callable[[], None] | None = None
    force_loading_status_once: bool = False


@dataclass(slots=True)
class AssistantRuntimeHooks:
    acquire_gpu: Callable[[], None]
    release_gpu: Callable[..., None]
    register_gpu_resident: Callable[[Callable[[], None] | None, bool], None]
    clear_gpu_resident: Callable[[], None]
    ensure_loaded: Callable[[], tuple[Any, Any]]
    unload_runtime: Callable[[], None]
    unload_weights: Callable[[], None]
    ensure_vision_loaded: Callable[[], tuple[Any, Any]] | None = None


def get_or_create_assistant_session(state) -> AssistantSessionState:
    session = state.get("assistant_session", None)
    if isinstance(session, AssistantSessionState):
        return session
    session = AssistantSessionState()
    state["assistant_session"] = session
    return session


def clear_assistant_session(session: AssistantSessionState) -> None:
    session.messages.clear()
    session.rendered_token_ids.clear()
    session.runtime_snapshot = None
    session.media_registry.clear()
    session.media_registry_counter = 0
    session.chat_html = ""
    session.queued_job_count = 0
    session.release_vram_callback = None
    session.force_loading_status_once = False
    assistant_chat.reset_session_chat(session)


def request_assistant_reset(session: AssistantSessionState) -> None:
    session.interrupt_requested = True
    session.drop_state_requested = True
    session.chat_epoch += 1
    session.queued_job_count = 0
    control_queue = session.control_queue
    if control_queue is not None:
        try:
            control_queue.push("exit", None)
        except Exception:
            pass


def _next_ai_client_id() -> str:
    global _AI_GEN_NO
    _AI_GEN_NO += 1
    return f"ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_AI_GEN_NO}"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


class tools:
    def __init__(self, gen, get_processed_queue, send_cmd, session: AssistantSessionState | None = None):
        self.gen = gen
        self.get_processed_queue = get_processed_queue
        self.send_cmd = send_cmd
        self.session = session
        self._vision_query_callback: Callable[[dict[str, Any], str], dict[str, Any]] | None = None

    def _log(self, message: str) -> None:
        if ASSISTANT_DEBUG:
            print(f"[AssistantTool] {message}")

    def _is_interrupted(self) -> bool:
        return self.session is not None and self.session.interrupt_requested

    def _interrupted_result(self, client_id: str, task: dict[str, Any]) -> dict[str, Any]:
        self._log(f"Generation interrupted for {client_id}")
        return {
            "status": "interrupted",
            "client_id": client_id,
            "output_file": "",
            "prompt": task["prompt"],
            "resolution": task["resolution"],
            "error": "Interrupted by reset request.",
        }

    def _set_status(self, text: str | None, kind: str = "working") -> None:
        self.send_cmd("chat_output", assistant_chat.build_status_event(text, kind=kind, visible=text is not None and len(str(text).strip()) > 0))

    def bind_runtime_tools(self, vision_query_callback: Callable[[dict[str, Any], str], dict[str, Any]] | None = None) -> None:
        self._vision_query_callback = vision_query_callback

    def _sync_recent_media(self, max_items: int = 5) -> None:
        if self.session is None:
            return
        file_list, file_settings_list, audio_file_list, audio_file_settings_list = self.get_processed_queue(self.gen)
        media_registry.sync_recent_generated_media(self.session, file_list, file_settings_list, max_items=max_items)
        media_registry.sync_recent_generated_media(self.session, audio_file_list, audio_file_settings_list, max_items=max_items)

    def _register_tool_media(self, path: str, settings: dict[str, Any], label: str | None = None) -> dict[str, Any] | None:
        if self.session is None:
            return None
        return media_registry.register_media(
            self.session,
            path,
            settings=settings,
            source="deepy",
            client_id=str(settings.get("client_id", "") or "").strip(),
            label=label,
        )

    def _resolve_image_media(self, media_id: str, parameter_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        media_id = str(media_id or "").strip()
        if len(media_id) == 0:
            return None, None
        if self.session is None:
            return None, {"status": "error", parameter_name: media_id, "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return None, {"status": "error", parameter_name: media_id, "error": f"Unknown media id for {parameter_name}."}
        if media_record.get("media_type") != "image":
            return None, {
                "status": "error",
                parameter_name: media_record.get("media_id", ""),
                "media_type": media_record.get("media_type", ""),
                "error": f"{parameter_name} must reference an image.",
            }
        return media_record, None

    def _queue_generation_task(self, task: dict[str, Any], *, activity_label: str, output_label: str | None = None) -> dict[str, Any]:
        if not isinstance(self.gen, dict):
            raise RuntimeError("WanGP generation queue is not available.")
        client_id = str(task.get("client_id", "") or "").strip()
        prompt = str(task.get("prompt", "") or "").strip()
        resolution = str(task.get("resolution", "") or "").strip()
        gen = self.gen
        self.get_processed_queue(gen)
        self._set_status(f"Queueing {activity_label}...", kind="tool")
        gen["inline_queue"] = task
        self.send_cmd("load_queue_trigger", {"client_id": client_id})
        self._log(f"Queued {activity_label} for {client_id}")

        queue_detect_deadline = time.time() + 5
        while time.time() < queue_detect_deadline:
            if self._is_interrupted():
                return self._interrupted_result(client_id, task)
            queue_errors = gen.get("queue_errors", None) or {}
            if client_id in queue_errors:
                error_text = str(queue_errors[client_id][0])
                self._log(f"Queue error detected for {client_id}: {error_text}")
                self._set_status(f"{activity_label.capitalize()} failed: {error_text}", kind="error")
                return {
                    "status": "error",
                    "client_id": client_id,
                    "output_file": "",
                    "prompt": prompt,
                    "resolution": resolution,
                    "error": error_text,
                }
            queue = gen.get("queue", []) or []
            if any(isinstance(item, dict) and isinstance(item.get("params"), dict) and item["params"].get("client_id") == client_id for item in queue):
                self._set_status(f"{activity_label.capitalize()} started...", kind="tool")
                break
            time.sleep(0.25)

        while True:
            if self._is_interrupted():
                return self._interrupted_result(client_id, task)
            queue_errors = gen.get("queue_errors", None) or {}
            if client_id in queue_errors:
                error_text = str(queue_errors[client_id][0])
                self._log(f"Generation error detected for {client_id}: {error_text}")
                self._set_status(f"{activity_label.capitalize()} failed: {error_text}", kind="error")
                return {
                    "status": "error",
                    "client_id": client_id,
                    "output_file": "",
                    "prompt": prompt,
                    "resolution": resolution,
                    "error": error_text,
                }
            file_list, file_settings_list, _audio_file_list, _audio_file_settings_list = self.get_processed_queue(gen)
            for file_path, file_settings in zip(file_list, file_settings_list):
                if not isinstance(file_settings, dict):
                    continue
                if file_settings.get("client_id", "") != client_id:
                    continue
                media_record = self._register_tool_media(str(file_path), file_settings, label=output_label)
                result = {
                    "status": "done",
                    "client_id": client_id,
                    "output_file": str(file_path),
                    "media_id": "" if media_record is None else media_record.get("media_id", ""),
                    "prompt": prompt,
                    "resolution": resolution,
                    "error": "",
                }
                self._log(f"{activity_label.capitalize()} completed for {client_id}: {file_path}")
                self._set_status(f"{activity_label.capitalize()} finished.", kind="tool")
                return result
            time.sleep(0.5)

    @assistant_tool(
        display_name="Generate Image",
        description="Queue and generate an image from a text prompt inside WanGP, then wait until the output image is available.",
        parameters={
            "prompt": {
                "type": "string",
                "description": "The image generation prompt to send to WanGP.",
            }
        },
    )
    def gen_image(self, prompt: str) -> dict[str, Any]:
        client_id = _next_ai_client_id()
        generator_variant = deepy_tool_settings.get_default_image_generator_variant()
        task = deepy_tool_settings.build_generation_task("gen_image", generator_variant, prompt=prompt, client_id=client_id)
        if len(task["prompt"]) == 0:
            self._set_status("Image generation failed: prompt is empty.", kind="error")
            return {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": "",
                "resolution": task["resolution"],
                "error": "Prompt is empty.",
            }
        result = self._queue_generation_task(task, activity_label="image generation", output_label="Generated image")
        result["generator_variant"] = generator_variant
        return result

    @assistant_tool(
        display_name="Generate Video",
        description="Queue and generate a video from a text prompt inside WanGP, optionally using a start image and an end image, then wait until the output video is available.",
        parameters={
            "prompt": {
                "type": "string",
                "description": "The video generation prompt to send to WanGP.",
            },
            "image_start": {
                "type": "string",
                "description": "Optional media id of the start image returned by Resolve Media.",
                "required": False,
            },
            "image_end": {
                "type": "string",
                "description": "Optional media id of the end image returned by Resolve Media.",
                "required": False,
            },
        },
    )
    def gen_video(self, prompt: str, image_start: str | None = None, image_end: str | None = None) -> dict[str, Any]:
        self._sync_recent_media()
        start_media, error_result = self._resolve_image_media(image_start or "", "image_start")
        if error_result is not None:
            error_result.update({"prompt": str(prompt or "").strip(), "output_file": ""})
            return error_result
        end_media, error_result = self._resolve_image_media(image_end or "", "image_end")
        if error_result is not None:
            error_result.update({"prompt": str(prompt or "").strip(), "output_file": ""})
            return error_result
        client_id = _next_ai_client_id()
        generator_variant = deepy_tool_settings.get_default_video_generator_variant()
        task = deepy_tool_settings.build_generation_task(
            "gen_video",
            generator_variant,
            prompt=prompt,
            client_id=client_id,
            image_start=None if start_media is None else str(start_media.get("path", "")).strip(),
            image_end=None if end_media is None else str(end_media.get("path", "")).strip(),
        )
        if len(task["prompt"]) == 0:
            self._set_status("Video generation failed: prompt is empty.", kind="error")
            return {
                "status": "error",
                "client_id": client_id,
                "output_file": "",
                "prompt": "",
                "resolution": task.get("resolution", ""),
                "error": "Prompt is empty.",
            }
        result = self._queue_generation_task(task, activity_label="video generation", output_label="Generated video")
        result["generator_variant"] = generator_variant
        if start_media is not None:
            result["source_start_media_id"] = start_media.get("media_id", "")
        if end_media is not None:
            result["source_end_media_id"] = end_media.get("media_id", "")
        return result

    @assistant_tool(
        display_name="Edit Image",
        description="Edit a previously resolved image using an instruction prompt inside WanGP and wait until the edited image is available.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id returned by Resolve Media.",
            },
            "prompt": {
                "type": "string",
                "description": "The instruction prompt describing how to modify the image.",
            },
        },
    )
    def edit_image(self, media_id: str, prompt: str) -> dict[str, Any]:
        self._sync_recent_media()
        if self.session is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "prompt": str(prompt or "").strip(), "output_file": "", "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "prompt": str(prompt or "").strip(), "output_file": "", "error": "Unknown media id."}
        if media_record.get("media_type") != "image":
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "media_type": media_record.get("media_type", ""),
                "prompt": str(prompt or "").strip(),
                "output_file": "",
                "error": "Edit Image currently supports images only.",
            }
        editor_variant = deepy_tool_settings.get_default_image_editor_variant()
        client_id = _next_ai_client_id()
        task = deepy_tool_settings.build_generation_task(
            "edit_image",
            editor_variant,
            prompt=prompt,
            client_id=client_id,
            image_refs=[str(media_record.get("path", "")).strip()],
        )
        if len(task["prompt"]) == 0:
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "prompt": "",
                "output_file": "",
                "error": "Prompt is empty.",
            }
        result = self._queue_generation_task(task, activity_label="image editing", output_label="Edited image")
        result["editor_variant"] = editor_variant
        result["source_media_id"] = media_record.get("media_id", "")
        return result

    @assistant_tool(
        display_name="Resolve Media",
        description="Look up previously generated WanGP media by natural reference such as last image, previous image, or a short description.",
        parameters={
            "reference": {
                "type": "string",
                "description": "The user's natural-language reference to previously generated media, such as 'last image' or 'robot on the moon'.",
            },
            "media_type": {
                "type": "string",
                "description": "The desired media type: image, video, audio, or any.",
            },
        },
        pause_runtime=False,
    )
    def resolve_media_reference(self, reference: str, media_type: str) -> dict[str, Any]:
        self._sync_recent_media()
        if self.session is None:
            return {"status": "error", "reference": str(reference or "").strip(), "media_type": str(media_type or "any").strip() or "any", "matches": [], "error": "Assistant session is not available."}
        result = media_registry.resolve_media_reference(self.session, reference, media_type)
        result.setdefault("error", "")
        return result

    @assistant_tool(
        display_name="Inspect Image",
        description="Ask Deepy to inspect a previously resolved image and answer a visual question about it.",
        parameters={
            "media_id": {
                "type": "string",
                "description": "The media id returned by Resolve Media.",
            },
            "question": {
                "type": "string",
                "description": "The visual question to answer about that image.",
            },
        },
        pause_runtime=True,
        pause_reason="vision",
    )
    def inspect_media(self, media_id: str, question: str) -> dict[str, Any]:
        self._sync_recent_media()
        if self.session is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "question": str(question or "").strip(), "answer": "", "error": "Assistant session is not available."}
        media_record = media_registry.get_media_record(self.session, media_id)
        if media_record is None:
            return {"status": "error", "media_id": str(media_id or "").strip(), "question": str(question or "").strip(), "answer": "", "error": "Unknown media id."}
        if media_record.get("media_type") != "image":
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "media_type": media_record.get("media_type", ""),
                "question": str(question or "").strip(),
                "answer": "",
                "error": "Visual inspection currently supports images only.",
            }
        if self._vision_query_callback is None:
            return {
                "status": "error",
                "media_id": media_record.get("media_id", ""),
                "media_type": media_record.get("media_type", ""),
                "question": str(question or "").strip(),
                "answer": "",
                "error": "Deepy vision inspection is not available.",
            }
        return self._vision_query_callback(media_record, question)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name)
            metadata = getattr(method, "_assistant_tool", None)
            if metadata is None:
                continue
            properties = {}
            required = []
            annotations = getattr(method, "__annotations__", {})
            for param_name, param_meta in metadata["parameters"].items():
                properties[param_name] = {
                    "type": param_meta.get("type") or _json_type_from_annotation(annotations.get(param_name, str)),
                    "description": str(param_meta.get("description", "")).strip(),
                }
                if bool(param_meta.get("required", True)):
                    required.append(param_name)
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": metadata["name"],
                        "description": metadata["description"],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return schemas

    def get_tool_display_name(self, tool_name: str) -> str:
        lookup_name = str(tool_name or "").strip()
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name)
            metadata = getattr(method, "_assistant_tool", None)
            if metadata is None or metadata["name"] != lookup_name:
                continue
            return str(metadata.get("display_name", lookup_name)).strip() or lookup_name
        return lookup_name.replace("_", " ").replace("-", " ").strip().title() or "Tool"

    def get_tool_policy(self, tool_name: str) -> dict[str, Any]:
        lookup_name = str(tool_name or "").strip()
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name)
            metadata = getattr(method, "_assistant_tool", None)
            if metadata is None or metadata["name"] != lookup_name:
                continue
            return {
                "pause_runtime": bool(metadata.get("pause_runtime", True)),
                "pause_reason": str(metadata.get("pause_reason", "tool") or "tool"),
            }
        return {"pause_runtime": True, "pause_reason": "tool"}

    def infer_tool_calls(self, raw_text: str) -> list[dict[str, Any]]:
        candidate_texts = []
        thinking_text, answer_text = qwen35_text._split_generated_text(raw_text)
        for candidate in (raw_text, answer_text, thinking_text):
            candidate = str(candidate or "").strip()
            if len(candidate) > 0:
                candidate_texts.append(candidate)

        by_name = {}
        sole_tool_name = None
        sole_tool_params = set()
        for schema in self.get_tool_schemas():
            function_spec = schema.get("function", {})
            tool_name = str(function_spec.get("name", "")).strip()
            if len(tool_name) == 0:
                continue
            by_name[tool_name] = set(function_spec.get("parameters", {}).get("properties", {}).keys())
        if len(by_name) == 1:
            sole_tool_name = next(iter(by_name))
            sole_tool_params = by_name[sole_tool_name]

        for candidate in candidate_texts:
            pseudo_match = re.search(r"Tool call:\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", candidate, flags=re.DOTALL)
            if pseudo_match is not None:
                tool_name = pseudo_match.group(1).strip()
                raw_args = pseudo_match.group(2).strip()
                arguments = {}
                for arg_name, quoted_value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"', raw_args):
                    arguments[arg_name] = quoted_value
                for arg_name, quoted_value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'([^']*)'", raw_args):
                    arguments[arg_name] = quoted_value
                if tool_name in by_name:
                    return [{"name": tool_name, "arguments": arguments}]

            fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
            json_candidate = fenced_match.group(1).strip() if fenced_match is not None else strip_trailing_stop_markup(candidate).strip()
            try:
                parsed = json.loads(json_candidate)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            if "name" in parsed and "arguments" in parsed:
                tool_name = str(parsed.get("name", "")).strip()
                arguments = parsed.get("arguments", {})
                if isinstance(arguments, dict) and tool_name in by_name:
                    return [{"name": tool_name, "arguments": arguments}]
            if sole_tool_name is not None and set(parsed.keys()).issubset(sole_tool_params):
                return [{"name": sole_tool_name, "arguments": parsed}]
        return []

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name)
            metadata = getattr(method, "_assistant_tool", None)
            if metadata is None:
                continue
            if metadata["name"] != tool_name:
                continue
            return method(**dict(arguments or {}))
        raise KeyError(f"Unknown assistant tool: {tool_name}")


class AssistantEngine:
    def __init__(self, session: AssistantSessionState, runtime_hooks: AssistantRuntimeHooks, tool_box: tools, send_cmd, debug_enabled: bool | None = None, thinking_enabled: bool = True, vram_mode: str = DEEPY_VRAM_UNLOAD):
        self.session = session
        self.runtime_hooks = runtime_hooks
        self.tool_box = tool_box
        self.send_cmd = send_cmd
        self.debug_enabled = ASSISTANT_DEBUG if debug_enabled is None else bool(debug_enabled)
        self.thinking_enabled = bool(thinking_enabled)
        self.vram_mode = normalize_deepy_vram_mode(vram_mode)
        self.runtime: Qwen35AssistantRuntime | None = None
        self._gpu_acquired = False
        self._skip_pause_snapshot = False
        self._active_turn_id = ""
        bind_runtime_tools = getattr(self.tool_box, "bind_runtime_tools", None)
        if callable(bind_runtime_tools):
            bind_runtime_tools(vision_query_callback=self._run_visual_query)

    def _log(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[Assistant] {message}")

    def _emit_chat_event(self, payload: str | None) -> None:
        if payload is None or len(str(payload).strip()) == 0:
            return
        self.send_cmd("chat_output", payload)

    def _set_status(self, text: str | None, kind: str = "thinking") -> None:
        self._emit_chat_event(assistant_chat.build_status_event(text, kind=kind, visible=text is not None and len(str(text).strip()) > 0))

    def _hide_status(self) -> None:
        self._emit_chat_event(assistant_chat.build_status_event(None, visible=False))

    def _send_chat(self, text: str) -> None:
        text = str(text or "").strip()
        if len(text) == 0:
            return
        self._emit_chat_event(assistant_chat.set_assistant_content(self.session, self._ensure_active_turn(), text))

    def _ensure_active_turn(self) -> str:
        if len(self._active_turn_id) == 0:
            self._active_turn_id = assistant_chat.create_assistant_turn(self.session)
        return self._active_turn_id

    def _split_for_display(self, raw_text: str) -> tuple[str, str]:
        thinking_text, answer_text = qwen35_text._split_generated_text(raw_text)
        if self.debug_enabled and len(thinking_text) > 0:
            print("[Assistant][Thinking]")
            print(thinking_text)
        if len(thinking_text) > 0:
            self._emit_chat_event(assistant_chat.append_reasoning(self.session, self._ensure_active_turn(), thinking_text))
        return thinking_text, answer_text

    def _acquire_runtime(self) -> Qwen35AssistantRuntime:
        acquired_here = False
        if not self._gpu_acquired:
            self.runtime_hooks.clear_gpu_resident()
            self.session.release_vram_callback = None
            self.runtime_hooks.acquire_gpu()
            self._gpu_acquired = True
            acquired_here = True
        try:
            model, _tokenizer = self.runtime_hooks.ensure_loaded()
            if self.runtime is None or self.runtime.model is not model:
                self.runtime = Qwen35AssistantRuntime(model, debug_enabled=self.debug_enabled)
            return self.runtime
        except Exception:
            if acquired_here:
                self._gpu_acquired = False
                self.runtime_hooks.release_gpu()
            raise

    def _ensure_vision_loaded(self) -> tuple[Any, Any]:
        ensure_vision_loaded = self.runtime_hooks.ensure_vision_loaded
        if not callable(ensure_vision_loaded):
            raise RuntimeError("Deepy vision runtime is not available.")
        caption_model, caption_processor = ensure_vision_loaded()
        if caption_model is None or caption_processor is None:
            raise RuntimeError("Deepy vision runtime is not available.")
        return caption_model, caption_processor

    def _run_visual_query(self, media_record: dict[str, Any], question: str) -> dict[str, Any]:
        if not self._gpu_acquired:
            self.runtime_hooks.clear_gpu_resident()
            self.session.release_vram_callback = None
            self.runtime_hooks.acquire_gpu()
            self._gpu_acquired = True
        media_path = str(media_record.get("path", "")).strip()
        if len(media_path) == 0 or not os.path.isfile(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")
        caption_model, caption_processor = self._ensure_vision_loaded()
        with Image.open(media_path) as image_handle:
            image = image_handle.convert("RGB")
        prompt_token_ids, prompt_embeds, prompt_position_ids, position_offset = deepy_vision.build_image_question_prompt(
            caption_model,
            caption_processor,
            image,
            question,
        )
        runtime = self._acquire_runtime()
        answer = runtime.generate_embedded_answer(
            prompt_token_ids,
            prompt_embeds,
            prompt_position_ids,
            position_offset,
            max_new_tokens=192,
            seed=0,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        return {
            "status": "done",
            "media_id": media_record.get("media_id", ""),
            "media_type": media_record.get("media_type", ""),
            "label": media_record.get("label", ""),
            "question": str(question or "").strip(),
            "answer": answer,
            "error": "",
        }

    def _force_release_vram(self) -> None:
        self.runtime_hooks.clear_gpu_resident()
        try:
            if self.runtime is not None and self.session.runtime_snapshot is None and len(self.session.rendered_token_ids) > 0:
                self.session.runtime_snapshot = self.runtime.snapshot_context()
        except Exception as exc:
            self._log(f"Resident snapshot before VRAM release failed: {exc}")
        try:
            self.runtime_hooks.unload_runtime()
        finally:
            self.runtime_hooks.unload_weights()
            self.runtime = None
            self.session.release_vram_callback = None

    def _pause_runtime(self, pause_reason: str = "idle") -> None:
        keep_loaded = self.vram_mode in (DEEPY_VRAM_ALWAYS, DEEPY_VRAM_UNLOAD_ON_REQUEST)
        if pause_reason == "vision":
            keep_loaded = False
        if pause_reason == "tool" and self.vram_mode != DEEPY_VRAM_ALWAYS:
            keep_loaded = False
        allow_force_release = keep_loaded and self.vram_mode == DEEPY_VRAM_UNLOAD_ON_REQUEST and pause_reason != "tool"
        release_callback = self._force_release_vram if keep_loaded else None
        if keep_loaded:
            self.session.release_vram_callback = release_callback
        else:
            self.session.release_vram_callback = None

        if not self._gpu_acquired:
            if self.session.drop_state_requested:
                if callable(self.session.release_vram_callback):
                    self.session.release_vram_callback()
                clear_assistant_session(self.session)
                self.session.drop_state_requested = False
            return
        try:
            if self.runtime is not None and not self.session.drop_state_requested and not self._skip_pause_snapshot:
                self.session.runtime_snapshot = self.runtime.snapshot_context()
            else:
                self.session.runtime_snapshot = None
        finally:
            try:
                if not keep_loaded:
                    self.runtime_hooks.unload_runtime()
            finally:
                try:
                    if not keep_loaded:
                        self.runtime_hooks.unload_weights()
                        self.runtime = None
                finally:
                    self.runtime_hooks.release_gpu(
                        keep_resident=allow_force_release,
                        release_vram_callback=release_callback,
                        force_release_on_acquire=allow_force_release,
                    )
                    self._gpu_acquired = False
                    self._skip_pause_snapshot = False
                    if self.session.drop_state_requested:
                        if keep_loaded and callable(self.session.release_vram_callback):
                            self.session.release_vram_callback()
                        clear_assistant_session(self.session)
                        self.session.drop_state_requested = False

    def _render_messages(self, add_generation_prompt: bool) -> list[int]:
        if self.runtime is None:
            raise RuntimeError("Assistant runtime is not available for prompt rendering.")
        messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}] + self.session.messages
        thinking_enabled = qwen35_text._prompt_enhancer_thinking_enabled(self.runtime.model, thinking_enabled=self.thinking_enabled)
        return render_assistant_messages(
            self.runtime.tokenizer,
            messages,
            self.tool_box.get_tool_schemas(),
            add_generation_prompt=add_generation_prompt,
            thinking_enabled=thinking_enabled,
        )

    def _restore_or_replay_session(self) -> str:
        if self.runtime is None:
            raise RuntimeError("Assistant runtime is not available for restore.")
        runtime = self.runtime
        fallback_tokens = self.session.rendered_token_ids
        if len(fallback_tokens) == 0:
            return "empty"
        try:
            live_seq = runtime._get_active_sequence()
        except Exception:
            live_seq = None
        if live_seq is not None:
            live_token_ids = [int(token_id) for token_id in live_seq.token_ids]
            snapshot_seq = None if self.session.runtime_snapshot is None else self.session.runtime_snapshot.get("sequence", {})
            snapshot_token_ids = [] if not isinstance(snapshot_seq, dict) else [int(token_id) for token_id in snapshot_seq.get("token_ids", []) or []]
            if len(snapshot_token_ids) > 0 and snapshot_token_ids == live_token_ids:
                self._log("Session context reused live runtime.")
                self.session.runtime_snapshot = None
                return "reused"
            if fallback_tokens[: len(live_token_ids)] == live_token_ids:
                self._log("Session context reused live runtime.")
                self.session.runtime_snapshot = None
                return "reused"
        mode = runtime.restore_or_replay(self.session.runtime_snapshot, fallback_tokens)
        self._log(f"Session context {mode}.")
        self.session.runtime_snapshot = None
        return mode

    def _sync_generation_context(self, pending_user_text: str | None = None) -> None:
        runtime = self._acquire_runtime()
        if len(self.session.rendered_token_ids) > 0:
            restore_mode = self._restore_or_replay_session()
            if pending_user_text is not None and restore_mode in ("reused", "restored"):
                thinking_enabled = qwen35_text._prompt_enhancer_thinking_enabled(self.runtime.model, thinking_enabled=self.thinking_enabled)
                suffix_tokens = render_text_user_turn_suffix(runtime.tokenizer, pending_user_text, thinking_enabled=thinking_enabled)
                if len(suffix_tokens) > 0:
                    mode = runtime.append_suffix(suffix_tokens)
                    self._log(f"Generation context {mode} from live runtime.")
                    return
        target_tokens = self._render_messages(add_generation_prompt=True)
        if len(self.session.rendered_token_ids) > 0:
            mode = runtime.extend_context(target_tokens)
            self._log(f"Generation context {mode}.")
            return
        runtime.prime_context(target_tokens)
        self._log("Generation context primed.")

    def _canonicalize_context(self, sync_runtime: bool | str = True) -> str:
        if self.runtime is None:
            raise RuntimeError("Assistant runtime is not available for canonicalization.")
        target_tokens = self._render_messages(add_generation_prompt=False)
        if not sync_runtime or sync_runtime == "record_only":
            self.session.rendered_token_ids = list(target_tokens)
            self.session.runtime_snapshot = None
            self._skip_pause_snapshot = True
            self._log("Canonical context recorded without runtime sync.")
            return "recorded"
        if sync_runtime == "record_preserve_live":
            self.session.rendered_token_ids = list(target_tokens)
            self.session.runtime_snapshot = None
            self._skip_pause_snapshot = False
            self._log("Canonical context recorded while preserving live runtime.")
            return "recorded"
        current_seq = self.runtime._get_active_sequence()
        if sync_runtime == "if_cheap":
            if current_seq is None or len(current_seq.token_ids) == 0:
                self.session.rendered_token_ids = list(target_tokens)
                self.session.runtime_snapshot = None
                self._skip_pause_snapshot = True
                self._log("Canonical context recorded without runtime sync because no active sequence was available.")
                return "recorded"
            current_token_ids = [int(token_id) for token_id in current_seq.token_ids]
            if target_tokens[: len(current_token_ids)] != current_token_ids:
                self.session.rendered_token_ids = list(target_tokens)
                self.session.runtime_snapshot = None
                self._skip_pause_snapshot = True
                self._log("Canonical context recorded without runtime sync because it would require replay.")
                return "recorded"
        self._skip_pause_snapshot = False
        if current_seq is None or len(current_seq.token_ids) == 0:
            self.runtime.prime_context(target_tokens)
            self._log("Canonical context rebuilt from scratch.")
            mode = "replayed"
        else:
            mode = self.runtime.extend_context(target_tokens)
            self._log(f"Canonical context {mode}.")
        self.session.rendered_token_ids = list(target_tokens)
        return mode

    def _build_tool_error(self, tool_name: str, arguments: dict[str, Any], error_text: str) -> dict[str, Any]:
        return {
            "status": "error",
            "tool": tool_name,
            "arguments": dict(arguments or {}),
            "error": str(error_text),
        }

    def _execute_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(tool_call.get("name", "")).strip()
        tool_label = self.tool_box.get_tool_display_name(tool_name)
        tool_policy = self.tool_box.get_tool_policy(tool_name)
        arguments = dict(tool_call.get("arguments", {}) or {})
        self._log(f"Tool call: {tool_name} {arguments}")
        tool_id, tool_event = assistant_chat.add_tool_call(self.session, self._ensure_active_turn(), tool_name, arguments, tool_label=tool_label)
        self._emit_chat_event(tool_event)
        self._set_status(f"Using {tool_label}...", kind="tool")
        if tool_policy.get("pause_runtime", True):
            self._pause_runtime(pause_reason=tool_policy.get("pause_reason", "tool"))
        try:
            result = self.tool_box.call(tool_name, arguments)
        except Exception as exc:
            result = self._build_tool_error(tool_name, arguments, str(exc))
            self._log(f"Tool error: {exc}")
        self._log(f"Tool result: {_json_dumps(result)}")
        self._emit_chat_event(assistant_chat.complete_tool_call(self.session, self._ensure_active_turn(), tool_id, result))
        return result

    def _append_assistant_message(self, raw_text: str, tool_calls: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        cleaned_text = strip_tool_blocks(raw_text)
        if tool_calls:
            cleaned_text = strip_inline_tool_call_text(cleaned_text)
        message = {"role": "assistant"}
        stripped_text = strip_trailing_stop_markup(cleaned_text)
        thinking_text, answer_text = qwen35_text._split_generated_text(stripped_text)
        thinking_enabled = self.runtime is not None and qwen35_text._prompt_enhancer_thinking_enabled(self.runtime.model, thinking_enabled=self.thinking_enabled)
        if thinking_enabled and ("<think>" in stripped_text.lower() or "</think>" in stripped_text.lower() or len(thinking_text) > 0):
            content = "<think>\n"
            if len(thinking_text) > 0:
                content += f"{thinking_text}\n"
            content += "</think>"
            if len(answer_text) > 0:
                content += f"\n\n{answer_text}"
        else:
            content = answer_text if len(answer_text) > 0 else stripped_text
        if len(content) > 0:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{int(time.time() * 1000)}_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": dict(tool_call["arguments"]),
                    },
                }
                for idx, tool_call in enumerate(tool_calls)
            ]
        self.session.messages.append(message)
        return message.get("tool_calls", [])

    def _append_tool_message(self, payload: dict[str, Any], tool_call_id: str | None = None) -> None:
        message = {"role": "tool", "content": _json_dumps(payload)}
        if tool_call_id:
            message["tool_call_id"] = str(tool_call_id)
        self.session.messages.append(message)

    def run_turn(self, user_text: str, max_new_tokens: int = 1024, seed: int | None = 0, do_sample: bool = True, temperature: float | None = 0.6, top_p: float | None = 0.9, top_k: int | None = None) -> None:
        user_text = str(user_text or "").strip()
        if len(user_text) == 0:
            self._send_chat("Please enter a request.")
            return

        self.session.interrupt_requested = False
        self._active_turn_id = ""
        self.session.messages.append({"role": "user", "content": user_text})
        tool_calls_used = 0
        model_passes = 0
        final_user_text = ""
        pending_user_text = user_text
        try:
            while True:
                if self.session.interrupt_requested:
                    break
                if model_passes >= MAX_MODEL_PASSES_PER_TURN:
                    self._send_chat("Assistant stopped because the model-pass limit was reached.")
                    break
                show_loading_status = model_passes == 0 and (
                    self.session.force_loading_status_once
                    or (len(self.session.rendered_token_ids) == 0 and self.session.runtime_snapshot is None)
                )
                self._set_status("Loading Deepy..." if show_loading_status else "Thinking...", kind="loading" if show_loading_status else "thinking")
                self._sync_generation_context(pending_user_text=pending_user_text)
                pending_user_text = None
                if self.session.interrupt_requested:
                    break
                if show_loading_status:
                    self.session.force_loading_status_once = False
                    self._set_status("Thinking...", kind="thinking")
                result = self.runtime.generate_segment(
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    thinking_enabled=self.thinking_enabled,
                )
                model_passes += 1
                raw_text = result.raw_text
                _thinking, answer_text = self._split_for_display(raw_text)
                tool_calls = extract_tool_calls(raw_text)
                if len(tool_calls) == 0:
                    tool_calls = self.tool_box.infer_tool_calls(raw_text)
                if self.debug_enabled:
                    self._log(f"Model stop reason: {result.stop_reason}")
                    print("[Assistant][Raw]")
                    print(raw_text)
                if tool_calls:
                    tool_calls_used += len(tool_calls)
                    if tool_calls_used > MAX_TOOL_CALLS_PER_TURN:
                        self._send_chat("Assistant stopped because the tool-call limit was reached.")
                        break
                    stored_tool_calls = self._append_assistant_message(raw_text, tool_calls=tool_calls)
                    self._canonicalize_context(sync_runtime="if_cheap")
                    for tool_call, stored_tool_call in zip(tool_calls, stored_tool_calls):
                        if self.session.interrupt_requested:
                            break
                        tool_result = self._execute_tool(tool_call)
                        self._append_tool_message(tool_result, stored_tool_call.get("id"))
                    if self.session.interrupt_requested:
                        break
                    continue

                self._append_assistant_message(raw_text)
                self._canonicalize_context(sync_runtime="record_preserve_live")
                final_user_text = answer_text or qwen35_text._clean_generated_text(raw_text)
                break
        finally:
            self._hide_status()
            try:
                self._pause_runtime(pause_reason="idle")
            except Exception as exc:
                self._log(f"Pause-after-turn failed: {exc}")
        if not self.session.interrupt_requested and len(final_user_text.strip()) > 0:
            self._send_chat(final_user_text)
