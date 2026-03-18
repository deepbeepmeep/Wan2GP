from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from shared.assistant_config import (
    DEEPY_VRAM_ALWAYS,
    DEEPY_VRAM_UNLOAD,
    DEEPY_VRAM_UNLOAD_ON_REQUEST,
    normalize_deepy_vram_mode,
)
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
ASSISTANT_SYSTEM_PROMPT = (
    "You are Deepy, the WanGP assistant. Help the user create images and videos. "
    "Use tools when they are needed. When a tool is the best next action, emit a tool call in Qwen tool-calling format. "
    "Be concise in user-facing chat output. Do not expose raw tool traces to the user."
)
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


def assistant_tool(name: str | None = None, description: str = "", parameters: dict[str, dict[str, Any]] | None = None, display_name: str | None = None):
    def decorator(func):
        func._assistant_tool = {
            "name": str(name or func.__name__).strip(),
            "display_name": str(display_name or name or func.__name__).strip(),
            "description": str(description or "").strip(),
            "parameters": dict(parameters or {}),
        }
        return func

    return decorator


@dataclass(slots=True)
class AssistantSessionState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    rendered_token_ids: list[int] = field(default_factory=list)
    runtime_snapshot: dict[str, Any] | None = None
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
    return f"ai{_AI_GEN_NO}"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


class tools:
    def __init__(self, gen, get_processed_queue, send_cmd, session: AssistantSessionState | None = None):
        self.gen = gen
        self.get_processed_queue = get_processed_queue
        self.send_cmd = send_cmd
        self.session = session

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
        task = {
            "model_type": "z_image",
            "client_id": client_id,
            "image_mode": 1,
            "resolution": "1280x720",   
            "num_inference_steps": 8,
            "prompt": str(prompt or "").strip(),
        }
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

        gen = self.gen
        file_list, file_settings_list, _audio_file_list, _audio_file_settings_list = self.get_processed_queue(gen)
        start_file_count = len(file_list)
        self._set_status("Queueing image generation...", kind="tool")
        gen["inline_queue"] = task
        self.send_cmd("load_queue_trigger", {"client_id": client_id})
        self._log(f"Queued image generation for {client_id}")

        queue_detect_deadline = time.time() + 5 # 30.0
        while time.time() < queue_detect_deadline:
            if self._is_interrupted():
                return self._interrupted_result(client_id, task)
            queue_errors = gen.get("queue_errors", None) or {}
            if client_id in queue_errors:
                error_text = str(queue_errors[client_id][0])
                self._log(f"Queue error detected for {client_id}: {error_text}")
                self._set_status(f"Image generation failed: {error_text}", kind="error")
                return {
                    "status": "error",
                    "client_id": client_id,
                    "output_file": "",
                    "prompt": task["prompt"],
                    "resolution": task["resolution"],
                    "error": error_text,
                }
            queue = gen.get("queue", []) or []
            if any(item.get("params", {}).get("client_id") == client_id for item in queue):
                self._set_status("Image generation started...", kind="tool")
                break
            time.sleep(0.25)

        while True:
            if self._is_interrupted():
                return self._interrupted_result(client_id, task)
            queue_errors = gen.get("queue_errors", None) or {}
            if client_id in queue_errors:
                error_text = str(queue_errors[client_id][0])
                self._log(f"Generation error detected for {client_id}: {error_text}")
                self._set_status(f"Image generation failed: {error_text}", kind="error")
                return {
                    "status": "error",
                    "client_id": client_id,
                    "output_file": "",
                    "prompt": task["prompt"],
                    "resolution": task["resolution"],
                    "error": error_text,
                }
            file_list, file_settings_list, _audio_file_list, _audio_file_settings_list = self.get_processed_queue(gen)
            for file_path, file_settings in zip(file_list, file_settings_list):
                if file_settings.get("client_id", "") == client_id:
                    result = {
                        "status": "done",
                        "client_id": client_id,
                        "output_file": str(file_path),
                        "prompt": task["prompt"],
                        "resolution": task["resolution"],
                        "error": "",
                    }
                    self._log(f"Generation completed for {client_id}: {file_path}")
                    self._set_status("Image generation finished.", kind="tool")
                    return result
            time.sleep(0.5)

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
        self.runtime_hooks.clear_gpu_resident()
        self.session.release_vram_callback = None
        self.runtime_hooks.acquire_gpu()
        self._gpu_acquired = True
        try:
            model, _tokenizer = self.runtime_hooks.ensure_loaded()
            if self.runtime is None or self.runtime.model is not model:
                self.runtime = Qwen35AssistantRuntime(model, debug_enabled=self.debug_enabled)
            return self.runtime
        except Exception:
            self._gpu_acquired = False
            self.runtime_hooks.release_gpu()
            raise

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
        arguments = dict(tool_call.get("arguments", {}) or {})
        self._log(f"Tool call: {tool_name} {arguments}")
        tool_id, tool_event = assistant_chat.add_tool_call(self.session, self._ensure_active_turn(), tool_name, arguments, tool_label=tool_label)
        self._emit_chat_event(tool_event)
        self._set_status(f"Using {tool_label}...", kind="tool")
        self._pause_runtime(pause_reason="tool")
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
