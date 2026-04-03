from __future__ import annotations

import json
import secrets
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr

from shared.deepy.config import (
    DEEPY_ENABLED_KEY,
    DEEPY_VRAM_MODE_KEY,
    DEEPY_VRAM_MODE_UNLOAD,
    deepy_available,
    deepy_requirement_met,
    normalize_deepy_enabled,
    normalize_deepy_vram_mode,
    set_deepy_runtime_config,
)
from shared.deepy import ui_settings as deepy_ui_settings
from shared.deepy.engine import (
    AssistantEngine,
    AssistantRuntimeHooks,
    begin_assistant_turn,
    build_interruption_notice,
    clear_assistant_session,
    get_or_create_assistant_session,
    mark_assistant_turn_message,
    record_interruption_history,
    request_assistant_interrupt,
    request_assistant_reset,
    set_assistant_debug,
    set_assistant_tool_ui_settings,
    tools as AssistantTools,
)
from shared.gradio import assistant_chat
from shared.utils.thread_utils import AsyncStream, async_run_in


_DEEPY_GPU_PROCESS_ID = "deepy"
_DEEPY_REQUIREMENT_TEXT = "Deepy requires Prompt Enhancer to be set to Qwen3.5VL Abliterated 4B or 9B."
_DEEPY_DISABLED_TEXT = "Deepy is disabled in Configuration > Deepy."


@dataclass(slots=True)
class DeepyDeps:
    get_server_config: Callable[[], dict[str, Any]]
    get_server_config_filename: Callable[[], str]
    get_verbose_level: Callable[[], int]
    resolve_prompt_enhancer_settings: Callable[..., tuple[Any, int]]
    get_state_model_type: Callable[[Any], str]
    get_model_def: Callable[[str], Any]
    ensure_prompt_enhancer_loaded: Callable[..., tuple[Any, Any]]
    unload_prompt_enhancer_runtime: Callable[[], None]
    get_image_caption_model: Callable[[], Any]
    get_image_caption_processor: Callable[[], Any]
    get_enhancer_offloadobj: Callable[[], Any]
    acquire_gpu: Callable[[Any], None]
    release_gpu: Callable[..., None]
    register_gpu_resident: Callable[..., None]
    clear_gpu_resident: Callable[[Any], None]
    get_new_refresh_id: Callable[[], Any]
    get_gen_info: Callable[[Any], dict[str, Any]]
    get_processed_queue: Callable[[dict[str, Any]], tuple[list[Any], list[Any], list[Any], list[Any]]]
    get_output_filepath: Callable[[str, bool, bool], str]
    record_file_metadata: Callable[..., Any]
    exec_prompt_enhancer_engine: Callable[..., Any]
    clear_queue_action: Callable[[Any], Any]


def _unload_prompt_enhancer_runtime(prompt_enhancer_image_caption_model, prompt_enhancer_llm_model) -> None:
    from shared.prompt_enhancer import unload_prompt_enhancer_models

    unload_prompt_enhancer_models(prompt_enhancer_image_caption_model, prompt_enhancer_llm_model)


class DeepyController:
    def __init__(self, deps: DeepyDeps):
        self._deps = deps
        self._active_assistant_session: Any | None = None

    def get_verbose_level(self) -> int:
        try:
            return int(self._deps.get_verbose_level() or 0)
        except Exception:
            return 0

    def _sync_debug_enabled(self) -> bool:
        try:
            debug_enabled = int(self._deps.get_verbose_level() or 0) >= 2
        except Exception:
            debug_enabled = False
        set_assistant_debug(debug_enabled)
        return debug_enabled

    def _server_config(self) -> dict[str, Any]:
        return self._deps.get_server_config() or {}

    def _persist_tool_ui_settings(self, normalized: dict[str, Any]) -> None:
        server_config = self._server_config()
        server_config_filename = str(self._deps.get_server_config_filename() or "").strip()
        deepy_ui_settings.store_assistant_tool_ui_settings(server_config, normalized)
        set_deepy_runtime_config(server_config, server_config_filename)
        if len(server_config_filename) > 0:
            with open(server_config_filename, "w", encoding="utf-8") as writer:
                writer.write(json.dumps(server_config, indent=4))
        gr.Info("New Deepy Setting Saved")

    def _reset_foreign_active_session(self, session) -> bool:
        active_session = self._active_assistant_session
        if active_session is None or active_session is session:
            return False
        request_assistant_reset(active_session)
        assistant_chat.reset_session_chat(active_session)
        active_session.chat_html = ""
        return True

    @staticmethod
    def _find_next_queued_user_message_id(session) -> str:
        for record in list(session.chat_transcript or []):
            if not isinstance(record, dict):
                continue
            if str(record.get("role", "")).strip().lower() != "user":
                continue
            if str(record.get("badge", "")).strip() != "Queued":
                continue
            message_id = str(record.get("id", "") or "").strip()
            if len(message_id) > 0:
                return message_id
        return ""

    def _cancel_next_queued_request(self, session) -> bool:
        if int(session.queued_job_count or 0) <= 0:
            return False
        message_id = self._find_next_queued_user_message_id(session)
        if len(message_id) == 0:
            return False
        user_text = assistant_chat.get_message_content(session, message_id)
        interruption_notice = build_interruption_notice(user_text)
        session.queued_job_count = max(0, int(session.queued_job_count or 0) - 1)
        session.queued_cancel_count = max(0, int(session.queued_cancel_count or 0)) + 1
        assistant_chat.set_message_badge(session, message_id, "Interrupted")
        assistant_chat.add_assistant_note(session, interruption_notice, badge="Interrupted", author="System")
        if len(str(user_text or "").strip()) > 0:
            session.messages.append({"role": "user", "content": str(user_text).strip()})
        session.messages.append({"role": "assistant", "content": interruption_notice})
        session.interruption_notice = interruption_notice
        record_interruption_history(session, user_text, interruption_notice)
        return True

    def is_available(self) -> bool:
        return deepy_available(self._server_config())

    def requirement_error_text(self) -> str:
        server_config = self._server_config()
        if not deepy_requirement_met(server_config):
            return _DEEPY_REQUIREMENT_TEXT
        if not normalize_deepy_enabled(server_config.get(DEEPY_ENABLED_KEY, 0)):
            return _DEEPY_DISABLED_TEXT
        return ""

    def get_vram_mode(self) -> str:
        server_config = self._server_config()
        return normalize_deepy_vram_mode(server_config.get(DEEPY_VRAM_MODE_KEY, DEEPY_VRAM_MODE_UNLOAD))

    def _ensure_vision_loaded(self, override_profile=None):
        self._deps.ensure_prompt_enhancer_loaded(override_profile=override_profile)
        image_caption_model = self._deps.get_image_caption_model()
        image_caption_processor = self._deps.get_image_caption_processor()
        if image_caption_model is None or image_caption_processor is None:
            raise gr.Error("Prompt enhancer vision runtime is not available.")
        return image_caption_model, image_caption_processor

    def _unload_weights(self) -> None:
        enhancer_offloadobj = self._deps.get_enhancer_offloadobj()
        if enhancer_offloadobj is not None:
            enhancer_offloadobj.unload_all()

    def _build_preload_release_callback(self) -> Callable[[], None]:
        def _release_preloaded_runtime() -> None:
            try:
                self._deps.unload_prompt_enhancer_runtime()
            finally:
                self._unload_weights()

        return _release_preloaded_runtime

    def release_vram(self, state, clear_session_state = False, discard_runtime_snapshot = False):
        session = get_or_create_assistant_session(state)
        release_callback = session.release_vram_callback
        session.release_vram_callback = None
        session.discard_runtime_snapshot_on_release = bool(discard_runtime_snapshot)
        self._deps.clear_gpu_resident(state)
        try:
            if callable(release_callback):
                release_callback()
        finally:
            if discard_runtime_snapshot:
                session.runtime_snapshot = None
                if len(session.rendered_token_ids) == 0:
                    session.pending_replay_reason = ""
            session.discard_runtime_snapshot_on_release = False
        if clear_session_state:
            clear_assistant_session(session)

    def preload_cli_runtime(self, state, override_profile=None) -> dict[str, Any]:
        self._sync_debug_enabled()
        self._deps.clear_gpu_resident(state)
        self._deps.acquire_gpu(state)
        keep_resident = False
        warmed_vllm = False
        try:
            model, _tokenizer = self._deps.ensure_prompt_enhancer_loaded(override_profile=override_profile)
            from shared.prompt_enhancer import qwen35_text

            if qwen35_text._use_vllm_prompt_enhancer(model):
                engine = qwen35_text._get_or_create_vllm_engine(model, usage_mode="assistant")
                engine.reserve_runtime(prompt_len=64, max_tokens=1, cfg_scale=1.0)
                engine._ensure_llm()
                llm = getattr(engine, "_llm", None)
                if llm is None:
                    raise RuntimeError("Assistant NanoVLLM runtime is not available.")
                llm.model_runner.ensure_runtime_ready()
                engine.release_runtime_allocations()
                warmed_vllm = True
            keep_resident = True
            return {"status": "ready", "warmed_vllm": warmed_vllm}
        finally:
            self._deps.release_gpu(
                state,
                keep_resident=keep_resident,
                release_vram_callback=self._build_preload_release_callback() if keep_resident else None,
                force_release_on_acquire=True,
            )

    def update_tool_ui_settings(self, state, *, auto_cancel_queue_tasks=None, separate_requests_with_empty_line=None, use_template_properties=None, width=None, height=None, num_frames=None, seed=None, video_with_speech_variant=None, image_generator_variant=None, image_editor_variant=None, video_generator_variant=None, speech_from_description_variant=None, speech_from_sample_variant=None, persist=False):
        session = get_or_create_assistant_session(state)
        normalized = set_assistant_tool_ui_settings(
            session,
            auto_cancel_queue_tasks=auto_cancel_queue_tasks,
            separate_requests_with_empty_line=separate_requests_with_empty_line,
            use_template_properties=use_template_properties,
            width=width,
            height=height,
            num_frames=num_frames,
            seed=seed,
            video_with_speech_variant=video_with_speech_variant,
            image_generator_variant=image_generator_variant,
            image_editor_variant=image_editor_variant,
            video_generator_variant=video_generator_variant,
            speech_from_description_variant=speech_from_description_variant,
            speech_from_sample_variant=speech_from_sample_variant,
        )
        if persist:
            self._persist_tool_ui_settings(normalized)
        return normalized

    @staticmethod
    def _split_request_blocks(text: str) -> list[str]:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(normalized) == 0:
            return []
        requests = []
        current_lines = []
        for raw_line in normalized.split("\n"):
            if not raw_line.strip():
                if current_lines:
                    requests.append("\n".join(current_lines).strip())
                    current_lines = []
                continue
            current_lines.append(raw_line.rstrip())
        if current_lines:
            requests.append("\n".join(current_lines).strip())
        return [request for request in requests if len(request) > 0]

    def _expand_assistant_requests(self, session, ask_request: Any) -> list[str]:
        normalized_request = str(ask_request or "").strip()
        if len(normalized_request) == 0:
            return []
        tool_ui_settings = deepy_ui_settings.normalize_assistant_tool_ui_settings(**session.tool_ui_settings) if isinstance(session.tool_ui_settings, dict) and len(session.tool_ui_settings) > 0 else deepy_ui_settings.get_persisted_assistant_tool_ui_settings(self._server_config())
        if not tool_ui_settings["separate_requests_with_empty_line"]:
            return [normalized_request]
        return self._split_request_blocks(normalized_request) or [normalized_request]

    def _queue_assistant_request(self, state, session, output_queue, ask_request: str, queued_epoch: int, *, queued: bool, precreate_assistant_turn: bool = False) -> None:
        raw_send_cmd = output_queue.push
        assistant_turn_id = ""

        def send_cmd(cmd, data=None):
            if queued_epoch != session.chat_epoch and cmd in {"chat_output", "load_queue_trigger", "refresh_gallery", "error"}:
                return
            raw_send_cmd(cmd, data)

        session.queued_job_count += 1
        user_message_id, _user_event = assistant_chat.add_user_message(session, ask_request, queued=queued)
        if precreate_assistant_turn:
            assistant_turn_id = assistant_chat.create_assistant_turn(session)

        def queue_worker_func():
            started_turn = False
            if queued_epoch != session.chat_epoch:
                if session.control_queue is output_queue:
                    session.control_queue = None
                raw_send_cmd("exit", None)
                return
            if int(session.queued_cancel_count or 0) > 0:
                session.queued_cancel_count = max(0, int(session.queued_cancel_count or 0) - 1)
                assistant_chat.set_message_badge(session, user_message_id, "Interrupted")
                if session.control_queue is output_queue and session.queued_job_count <= 0:
                    session.control_queue = None
                raw_send_cmd("chat_output", assistant_chat.build_sync_event(session))
                if session.queued_job_count > 0:
                    raw_send_cmd("chat_output", assistant_chat.build_status_event("Queued behind the current assistant task.", kind="queued"))
                else:
                    raw_send_cmd("chat_output", assistant_chat.build_status_event(None, visible=False))
                    raw_send_cmd("exit", None)
                return
            session.queued_job_count = max(0, session.queued_job_count - 1)
            session.interrupt_requested = False
            session.control_queue = output_queue
            session.worker_active = True
            self._active_assistant_session = session
            begin_assistant_turn(session, user_message_id, ask_request)
            started_turn = True
            assistant_chat.set_message_badge(session, user_message_id, None)
            active_turn_id = assistant_turn_id or assistant_chat.create_assistant_turn(session)
            mark_assistant_turn_message(session, active_turn_id)
            send_cmd("chat_output", assistant_chat.build_sync_event(session))
            my_tools = self.create_tools(state, send_cmd, session=session)
            try:
                self._deps.exec_prompt_enhancer_engine(state, None, "AK", [ask_request], None, None, False, False, 0, None, 3.5, send_cmd, my_tools)
            except Exception as e:
                traceback.print_exc()
                error_turn_id = assistant_turn_id or assistant_chat.create_assistant_turn(session)
                error_event = assistant_chat.set_assistant_content(session, error_turn_id, f"Assistant crashed: {e}")
                if error_event is not None:
                    send_cmd("chat_output", error_event)
                send_cmd("chat_output", assistant_chat.build_status_event(None, visible=False))
            finally:
                if self._active_assistant_session is session:
                    self._active_assistant_session = None
                session.worker_active = False
                stale_turn = queued_epoch != session.chat_epoch
                has_more_work = not stale_turn and session.queued_job_count > 0
                if not has_more_work and session.control_queue is output_queue:
                    session.control_queue = None
                if stale_turn:
                    if started_turn:
                        raw_send_cmd("chat_output", assistant_chat.build_reset_event())
                else:
                    raw_send_cmd("chat_output", assistant_chat.build_sync_event(session))
                    if has_more_work:
                        raw_send_cmd("chat_output", assistant_chat.build_status_event("Queued behind the current assistant task.", kind="queued"))
                session.interrupt_requested = False
                if not has_more_work:
                    raw_send_cmd("exit", None)

        async_run_in("assistant", queue_worker_func)

    def store_selected_video_time(self, state, current_time):
        gen = self._deps.get_gen_info(state)
        try:
            value = float(current_time)
        except Exception:
            value = None
        gen["selected_video_time"] = None if value is None or value < 0 else value

    def create_tools(self, state, send_cmd, session = None):
        active_session = get_or_create_assistant_session(state) if session is None else session
        gen = self._deps.get_gen_info(state)
        return AssistantTools(
            gen,
            self._deps.get_processed_queue,
            send_cmd,
            session=active_session,
            get_output_filepath=self._deps.get_output_filepath,
            record_file_metadata=self._deps.record_file_metadata,
            get_server_config=self._server_config,
        )

    def run_assistant_prompt_turn(self, state, model_def, prompt_enhancer_modes, original_prompts, seed, override_profile=None, send_cmd=None, tools=None) -> None:
        debug_enabled = self._sync_debug_enabled()
        server_config = self._server_config()
        if not normalize_deepy_enabled(server_config.get(DEEPY_ENABLED_KEY, 0)):
            raise gr.Error(_DEEPY_DISABLED_TEXT)
        if not deepy_requirement_met(server_config):
            raise gr.Error(_DEEPY_REQUIREMENT_TEXT)
        if send_cmd is None or tools is None:
            raise gr.Error("Assistant mode requires a command stream and a tool registry.")
        enhancer_temperature = server_config.get("prompt_enhancer_temperature", 0.6)
        enhancer_top_p = server_config.get("prompt_enhancer_top_p", 0.9)
        randomize_seed = server_config.get("prompt_enhancer_randomize_seed", True)
        assistant_seed = secrets.randbits(32) if randomize_seed else (seed if seed is not None and seed >= 0 else 0)
        session = get_or_create_assistant_session(state)
        assistant_model_def = model_def
        _assistant_instructions, assistant_max_new_tokens = self._deps.resolve_prompt_enhancer_settings(assistant_model_def, prompt_enhancer_modes, is_image=False, text_encoder_max_tokens=1024)
        assistant = AssistantEngine(
            session,
            AssistantRuntimeHooks(
                acquire_gpu=lambda: self._deps.acquire_gpu(state),
                release_gpu=lambda keep_resident = False, release_vram_callback = None, force_release_on_acquire = True: self._deps.release_gpu(state, keep_resident=keep_resident, release_vram_callback=release_vram_callback, force_release_on_acquire=force_release_on_acquire),
                register_gpu_resident=lambda release_vram_callback = None, force_release_on_acquire = True: self._deps.register_gpu_resident(state, release_vram_callback=release_vram_callback, force_release_on_acquire=force_release_on_acquire),
                clear_gpu_resident=lambda: self._deps.clear_gpu_resident(state),
                ensure_loaded=lambda: self._deps.ensure_prompt_enhancer_loaded(override_profile=override_profile),
                unload_runtime=self._deps.unload_prompt_enhancer_runtime,
                unload_weights=self._unload_weights,
                ensure_vision_loaded=lambda: self._ensure_vision_loaded(override_profile=override_profile),
            ),
            tools,
            send_cmd,
            debug_enabled=debug_enabled,
            thinking_enabled="K" in prompt_enhancer_modes,
            vram_mode=self.get_vram_mode(),
        )
        assistant.run_turn(
            original_prompts[0] if len(original_prompts) > 0 else "",
            max_new_tokens=max(1024, int(assistant_max_new_tokens)),
            seed=assistant_seed,
            do_sample=True,
            temperature=enhancer_temperature,
            top_p=enhancer_top_p,
        )

    def ask_ai(self, state, ask_request):
        self._sync_debug_enabled()

        def get_refresh_id():
            return str(time.time()) + "_" + str(self._deps.get_new_refresh_id())

        def drain_chat_output_batch(first_payload):
            payloads = [first_payload]
            while True:
                next_item = com_stream.output_queue.top()
                if not isinstance(next_item, tuple) or len(next_item) < 1 or next_item[0] != "chat_output":
                    break
                _cmd, next_payload = com_stream.output_queue.pop()
                payloads.append(next_payload)
            return assistant_chat.build_event_batch(payloads)

        session = get_or_create_assistant_session(state)
        foreign_session_reset = self._reset_foreign_active_session(session)
        request_blocks = self._expand_assistant_requests(session, ask_request)
        if len(request_blocks) == 0:
            yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            return
        if session.drop_state_requested:
            yield assistant_chat.build_status_event("Resetting after the current work stops...", kind="queued"), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        if not self.is_available():
            error_turn_id = assistant_chat.create_assistant_turn(session)
            error_event = assistant_chat.set_assistant_content(session, error_turn_id, self.requirement_error_text())
            yield error_event if error_event is not None else gr.update(), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        com_stream = AsyncStream()
        output_queue = com_stream.output_queue
        queued = foreign_session_reset or session.worker_active or session.queued_job_count > 0
        queued_epoch = session.chat_epoch
        for index, request_block in enumerate(request_blocks):
            self._queue_assistant_request(state, session, output_queue, request_block, queued_epoch, queued=queued or index > 0, precreate_assistant_turn=not queued and index == 0 and len(request_blocks) > 1)
        yield assistant_chat.build_sync_event(session), gr.update(), gr.update(value=""), gr.update(), gr.update()
        if queued or len(request_blocks) > 1:
            yield assistant_chat.build_status_event("Queued behind the current assistant task.", kind="queued"), gr.update(), gr.update(), gr.update(), gr.update()
        while True:
            cmd, data = com_stream.output_queue.next()
            if cmd == "console_output":
                print(data)
            elif cmd == "chat_output":
                yield drain_chat_output_batch(data), gr.update(), gr.update(), gr.update(), gr.update()
            elif cmd == "load_queue_trigger":
                yield gr.update(), str(get_refresh_id()), gr.update(), gr.update(), gr.update()
            elif cmd == "abort_client_id":
                yield gr.update(), gr.update(), gr.update(), gr.update(), str(data or "")
            elif cmd == "refresh_gallery":
                yield gr.update(), gr.update(), gr.update(), str(get_refresh_id()), gr.update()
            elif cmd == "error":
                error_turn_id = assistant_chat.create_assistant_turn(session)
                error_event = assistant_chat.set_assistant_content(session, error_turn_id, str(data or "Assistant error."))
                yield error_event if error_event is not None else gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            elif cmd == "exit":
                break

    def enqueue_ai_while_busy(self, state, ask_request):
        self._sync_debug_enabled()
        session = get_or_create_assistant_session(state)
        request_blocks = self._expand_assistant_requests(session, ask_request)
        if len(request_blocks) == 0:
            return gr.update(), gr.update(value="")
        if session.drop_state_requested:
            return assistant_chat.build_status_event("Resetting after the current work stops...", kind="queued"), gr.update(value="")
        if not self.is_available():
            if session.control_queue is not None:
                error_turn_id = assistant_chat.create_assistant_turn(session)
                error_event = assistant_chat.set_assistant_content(session, error_turn_id, self.requirement_error_text())
                if error_event is not None:
                    session.control_queue.push("chat_output", error_event)
            return gr.update(), gr.update(value="")
        output_queue = session.control_queue
        if output_queue is None or (not session.worker_active and session.queued_job_count <= 0):
            return gr.update(), gr.update(value="")
        queued_epoch = session.chat_epoch
        for request_block in request_blocks:
            self._queue_assistant_request(state, session, output_queue, request_block, queued_epoch, queued=True)
        output_queue.push("chat_output", assistant_chat.build_sync_event(session))
        output_queue.push("chat_output", assistant_chat.build_status_event("Queued behind the current assistant task.", kind="queued"))
        return gr.update(), gr.update(value="")

    def stop_ai(self, state):
        session = get_or_create_assistant_session(state)
        if session.worker_active:
            request_assistant_interrupt(session)
            return assistant_chat.build_status_event("Interrupting the current assistant task...", kind="queued"), gr.update(), gr.update(), gr.update()
        if not session.worker_active and self._cancel_next_queued_request(session):
            chat_event = assistant_chat.build_sync_event(session)
            return chat_event, gr.update(), gr.update(), gr.update()
        return gr.update(), gr.update(), gr.update(), gr.update()

    def reset_ai(self, state):
        session = get_or_create_assistant_session(state)
        if session.worker_active:
            print("[Assistant] Reset requested during an active turn; the chat will reset after the current work stops.")
            request_assistant_reset(session)
            session.chat_html = ""
            return assistant_chat.build_status_event("Resetting after the current work stops...", kind="queued"), gr.update(), gr.update(value=""), gr.update()
        else:
            reset_to_base_callback = session.reset_to_base_callback
            reset_applied = False
            if callable(reset_to_base_callback):
                try:
                    reset_applied = bool(reset_to_base_callback())
                except Exception as exc:
                    print(f"[Assistant] Idle reset-base reuse failed: {exc}")
                    reset_applied = False
            if reset_applied:
                print("[Assistant] Idle Reset: reused preserved header snapshot. [no prefill redone]")
            if not reset_applied:
                print("[Assistant] Idle Reset: fallback to full clear.")
                self.release_vram(state, True)
        session.chat_html = ""
        return assistant_chat.build_reset_event(), gr.update(), gr.update(value=""), gr.update()

    def browser_session_started(self, state):
        session = get_or_create_assistant_session(state)
        if self._reset_foreign_active_session(session):
            return assistant_chat.build_reset_event(), gr.update(), gr.update(value=""), gr.update()
        if not session.worker_active and session.queued_job_count <= 0:
            return gr.update(), gr.update(), gr.update(), gr.update()
        request_assistant_reset(session)
        session.chat_html = ""
        return assistant_chat.build_status_event("Resetting after the current work stops...", kind="queued"), gr.update(), gr.update(value=""), gr.update()


def create_controller(**deps_kwargs) -> DeepyController:
    return DeepyController(DeepyDeps(**deps_kwargs))
