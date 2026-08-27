from __future__ import annotations

import json
import re
import secrets
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr

from shared.deepy.config import (
    DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_ENABLED_KEY,
    DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_TYPE_KEY,
    DEEPY_TYPE_PRIME,
    DEEPY_VRAM_MODE_KEY,
    DEEPY_VRAM_MODE_UNLOAD,
    deepy_available,
    deepy_requirement_error,
    deepy_requirement_met,
    normalize_deepy_enabled,
    normalize_deepy_type,
    normalize_deepy_vram_mode,
    set_deepy_runtime_config,
)
from shared.deepy import PRIME_SYSTEM_PROMPT, ZERO_SYSTEM_PROMPT
from shared.deepy import ui_settings as deepy_ui_settings
from shared.deepy.debug_bootstrap import deepy_log_scope
from shared.deepy.engine import (
    AssistantEngine,
    AssistantRuntimeHooks,
    begin_assistant_turn,
    build_interruption_notice,
    clear_assistant_steering,
    clear_assistant_session,
    get_or_create_assistant_session,
    mark_assistant_turn_message,
    record_interruption_history,
    request_assistant_interrupt,
    request_assistant_steering,
    request_assistant_reset,
    set_assistant_debug,
    set_assistant_tool_ui_settings,
    DeepyZeroTools,
)
from shared.gradio import assistant_chat
from shared.utils.thread_utils import AsyncStream, async_run_in
from shared.remote_llm.config import is_remote_engine, resolve_role_engine


_DEEPY_GPU_PROCESS_ID = "deepy"
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
        self._queue_state_lock = threading.RLock()

    def get_verbose_level(self) -> int:
        try:
            return int(self._deps.get_verbose_level() or 0)
        except Exception:
            return 0

    def _debug_log(self, message: str) -> None:
        if self.get_verbose_level() >= 2:
            with deepy_log_scope(start_if_needed=True):
                print(f"[AssistantController] {message}")

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
            if not bool(record.get("queued", False)):
                continue
            message_id = str(record.get("id", "") or "").strip()
            if len(message_id) > 0:
                return message_id
        return ""

    def _cancel_next_queued_request(self, session) -> str:
        with self._queue_state_lock:
            if int(session.queued_job_count or 0) <= 0:
                return ""
            message_id = self._find_next_queued_user_message_id(session)
            if len(message_id) == 0:
                return ""
            user_text = assistant_chat.get_message_content(session, message_id)
            interruption_notice = build_interruption_notice(user_text)
            session.queued_job_count = max(0, int(session.queued_job_count or 0) - 1)
            session.queued_cancel_count = max(0, int(session.queued_cancel_count or 0)) + 1
            assistant_chat._find_message(session, message_id)["queued"] = False
            assistant_chat.set_message_badge(session, message_id, "Interrupted")
            record_interruption_history(session, user_text, interruption_notice)
            return user_text

    def _apply_queued_request_action(self, state, action_payload):
        session = get_or_create_assistant_session(state)
        try:
            payload = json.loads(str(action_payload or ""))
        except (TypeError, ValueError):
            return gr.update(), gr.update(), gr.update(), gr.update()
        if not isinstance(payload, dict):
            return gr.update(), gr.update(), gr.update(), gr.update()
        action = str(payload.get("action", "") or "").strip().lower()
        message_id = str(payload.get("message_id", "") or "").strip()
        text = str(payload.get("text", "") or "").strip()
        if action not in {"edit", "remove"} or len(message_id) == 0 or (action == "edit" and len(text) == 0):
            return gr.update(), gr.update(), gr.update(), gr.update()
        with self._queue_state_lock:
            record = assistant_chat._find_message(session, message_id)
            if record is None or str(record.get("role", "")).strip() != "user" or str(record.get("badge", "")).strip() != "Queued":
                return gr.update(), gr.update(), gr.update(), gr.update()
            if action == "edit":
                chat_event = assistant_chat.set_user_message_content(session, message_id, text)
            else:
                session.cancelled_queued_message_ids.add(message_id)
                session.queued_job_count = max(0, int(session.queued_job_count or 0) - 1)
                chat_event = assistant_chat.remove_message(session, message_id)
        self._debug_log(f"Queued request {action} user_message_id={message_id} queued_jobs={int(session.queued_job_count or 0)}")
        return chat_event if chat_event is not None else gr.update(), gr.update(), gr.update(), gr.update()

    def _cancel_active_prime_job(self, session, action: str) -> str:
        cancel_active_job = getattr(session.prime_toolbox, "cancel_active_job", None)
        if not callable(cancel_active_job):
            return ""
        try:
            return str(cancel_active_job() or "").strip()
        except Exception as exc:
            self._debug_log(f"Immediate MCP job cancellation failed after {action}: {exc}")
            return ""

    def is_available(self) -> bool:
        return deepy_available(self._server_config())

    def requirement_error_text(self) -> str:
        server_config = self._server_config()
        if not deepy_requirement_met(server_config):
            return deepy_requirement_error(server_config)
        if not normalize_deepy_enabled(server_config.get(DEEPY_ENABLED_KEY, 0)):
            return _DEEPY_DISABLED_TEXT
        return ""

    def get_vram_mode(self) -> str:
        server_config = self._server_config()
        return normalize_deepy_vram_mode(server_config.get(DEEPY_VRAM_MODE_KEY, DEEPY_VRAM_MODE_UNLOAD))

    def get_deepy_type(self) -> str:
        return normalize_deepy_type(self._server_config().get(DEEPY_TYPE_KEY, ""))

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

    def update_tool_ui_settings(self, state, *, auto_cancel_queue_tasks=None, separate_requests_with_empty_line=None, use_template_properties=None, width=None, height=None, num_frames=None, audio_duration=None, seed=None, video_with_speech_variant=None, image_generator_variant=None, image_editor_variant=None, video_generator_variant=None, song_variant=None, speech_from_description_variant=None, speech_from_sample_variant=None, persist=False):
        session = get_or_create_assistant_session(state)
        normalized = set_assistant_tool_ui_settings(
            session,
            auto_cancel_queue_tasks=auto_cancel_queue_tasks,
            separate_requests_with_empty_line=separate_requests_with_empty_line,
            use_template_properties=use_template_properties,
            width=width,
            height=height,
            num_frames=num_frames,
            audio_duration=audio_duration,
            seed=seed,
            video_with_speech_variant=video_with_speech_variant,
            image_generator_variant=image_generator_variant,
            image_editor_variant=image_editor_variant,
            video_generator_variant=video_generator_variant,
            song_variant=song_variant,
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
            stripped_line = raw_line.strip()
            if not stripped_line or re.fullmatch(r"[-=_]{3,}", stripped_line):
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

    def _queue_assistant_request(self, state, session, output_queue, ask_request: str, queued_epoch: int, *, queued: bool, client_submission_id: str = "", assistant_badge: str = "") -> None:
        raw_send_cmd = output_queue.push
        assistant_badge = str(assistant_badge or "").strip()

        def send_cmd(cmd, data=None):
            if queued_epoch != session.chat_epoch and cmd in {"chat_output", "load_queue_trigger", "refresh_gallery", "error"}:
                return
            raw_send_cmd(cmd, data)

        with self._queue_state_lock:
            session.queued_job_count += 1
            user_message_id, _user_event = assistant_chat.add_user_message(session, ask_request, queued=queued, client_submission_id=client_submission_id)
            if assistant_badge:
                user_record = assistant_chat._find_message(session, user_message_id)
                user_record["assistant_badge"] = assistant_badge
                user_record["badge"] = ""
            self._debug_log(f"Request enqueued user_message_id={user_message_id} queued={bool(queued)} queued_jobs={int(session.queued_job_count or 0)}")

        def queue_worker_func():
            with deepy_log_scope(start_if_needed=True):
                started_turn = False
                active_request = ask_request
                with self._queue_state_lock:
                    stale_request = queued_epoch != session.chat_epoch
                    targeted_cancelled = not stale_request and user_message_id in session.cancelled_queued_message_ids
                    cancelled_request = not stale_request and (targeted_cancelled or int(session.queued_cancel_count or 0) > 0)
                    if stale_request:
                        if session.control_queue is output_queue:
                            session.control_queue = None
                    elif cancelled_request:
                        if targeted_cancelled:
                            session.cancelled_queued_message_ids.discard(user_message_id)
                        else:
                            session.queued_cancel_count = max(0, int(session.queued_cancel_count or 0) - 1)
                            assistant_chat.set_message_badge(session, user_message_id, "Interrupted")
                        cancelled_sync = assistant_chat.build_sync_event(session)
                        cancelled_has_more_work = session.queued_job_count > 0
                    else:
                        active_request = assistant_chat.get_message_content(session, user_message_id)
                        session.queued_job_count = max(0, session.queued_job_count - 1)
                        session.interrupt_requested = False
                        clear_assistant_steering(session)
                        session.control_queue = output_queue
                        session.worker_active = True
                        self._active_assistant_session = session
                        begin_assistant_turn(session, user_message_id, active_request, assistant_badge=assistant_badge)
                        started_turn = True
                        assistant_chat._find_message(session, user_message_id)["queued"] = False
                        assistant_chat.set_message_badge(session, user_message_id, None)
                        starting_status = {"visible": True, "kind": "queued" if assistant_badge else "loading", "text": "Steering accepted. Deepy is applying the new instructions..." if assistant_badge else "Starting Deepy..."}
                        starting_sync = assistant_chat.build_sync_event(session, status=starting_status)
                if stale_request:
                    self._debug_log(f"Worker skipped stale request user_message_id={user_message_id} queued_epoch={queued_epoch} chat_epoch={session.chat_epoch}")
                    raw_send_cmd("exit", None)
                    return
                if cancelled_request:
                    self._debug_log(f"Worker cancelled queued request user_message_id={user_message_id}")
                    raw_send_cmd("chat_output", cancelled_sync)
                    if cancelled_has_more_work:
                        raw_send_cmd("chat_output", assistant_chat.build_status_event("Queued behind the current assistant task.", kind="queued"))
                    else:
                        raw_send_cmd("chat_output", assistant_chat.build_status_event(None, visible=False))
                        raw_send_cmd("exit", None)
                    return
                self._debug_log(f"Worker starting user_message_id={user_message_id} queued_jobs={int(session.queued_job_count or 0)}")
                send_cmd("chat_output", starting_sync)
                my_tools = self.create_tools(state, send_cmd, session=session)
                try:
                    self._debug_log(f"Prompt enhancer dispatch starting user_message_id={user_message_id}")
                    self._deps.exec_prompt_enhancer_engine(state, "", None, "AK", [active_request], None, None, False, False, 0, None, 3.5, send_cmd, my_tools)
                except Exception as e:
                    user_action_required = bool(getattr(e, "user_action_required", False))
                    if not user_action_required:
                        traceback.print_exc()
                    error_turn_id = assistant_chat.create_assistant_turn(session)
                    if assistant_badge:
                        assistant_chat.set_message_badge(session, error_turn_id, assistant_badge)
                    mark_assistant_turn_message(session, error_turn_id)
                    error_event = assistant_chat.set_assistant_content(session, error_turn_id, str(e) if user_action_required else f"Assistant crashed: {e}")
                    if error_event is not None:
                        send_cmd("chat_output", error_event)
                    send_cmd("chat_output", assistant_chat.build_status_event(None, visible=False))
                finally:
                    with self._queue_state_lock:
                        if self._active_assistant_session is session:
                            self._active_assistant_session = None
                        session.worker_active = False
                        stale_turn = queued_epoch != session.chat_epoch
                        has_more_work = not stale_turn and session.queued_job_count > 0
                        if not has_more_work and session.control_queue is output_queue:
                            session.control_queue = None
                        pending_steering = has_more_work and any(str(record.get("role", "")).strip() == "user" and bool(record.get("queued", False)) and str(record.get("assistant_badge", "")).strip() == "Steered" for record in session.chat_transcript)
                        final_status = None if not has_more_work else {"visible": True, "kind": "queued", "text": "Steering accepted. Deepy is applying the new instructions..." if pending_steering else "Queued behind the current assistant task."}
                        final_sync = None if stale_turn else assistant_chat.build_sync_event(session, status=final_status)
                        session.interrupt_requested = False
                    if stale_turn:
                        if started_turn:
                            raw_send_cmd("chat_output", assistant_chat.build_reset_event(session))
                    else:
                        raw_send_cmd("chat_output", final_sync)
                    self._debug_log(f"Worker finished user_message_id={user_message_id} stale={bool(stale_turn)} has_more_work={bool(has_more_work)} queued_jobs={int(session.queued_job_count or 0)}")
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
        if self.get_deepy_type() == DEEPY_TYPE_PRIME:
            from shared.deepy.prime_tools import DeepyPrimeTools

            if active_session.prime_toolbox is None:
                gen = self._deps.get_gen_info(state)
                zero_tools = DeepyZeroTools(gen, self._deps.get_processed_queue, send_cmd, session=active_session, get_output_filepath=self._deps.get_output_filepath, record_file_metadata=self._deps.record_file_metadata, get_server_config=self._server_config)
                active_session.prime_toolbox = DeepyPrimeTools(state, send_cmd, active_session, zero_tools=zero_tools)
            else:
                active_session.prime_toolbox.bind_turn(state, send_cmd)
            return active_session.prime_toolbox
        if active_session.prime_toolbox is not None:
            active_session.prime_toolbox.close()
            active_session.prime_toolbox = None
        gen = self._deps.get_gen_info(state)
        return DeepyZeroTools(
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
            raise gr.Error(deepy_requirement_error(server_config))
        if send_cmd is None or tools is None:
            raise gr.Error("Assistant mode requires a command stream and a tool registry.")
        enhancer_temperature = server_config.get("prompt_enhancer_temperature", 0.6)
        enhancer_top_p = server_config.get("prompt_enhancer_top_p", 0.9)
        randomize_seed = server_config.get("prompt_enhancer_randomize_seed", True)
        assistant_seed = secrets.randbits(32) if randomize_seed else (seed if seed is not None and seed >= 0 else 0)
        session = get_or_create_assistant_session(state)
        assistant_model_def = model_def
        deepy_type = self.get_deepy_type()
        from shared.deepy.filesystem import build_file_access_policy

        file_access_policy = build_file_access_policy(server_config)
        session.file_access_policy = file_access_policy
        system_prompt = ZERO_SYSTEM_PROMPT
        custom_system_prompt_key = DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY
        if deepy_type == DEEPY_TYPE_PRIME:
            server_instructions = tools.get_system_instructions()
            system_prompt = f"{PRIME_SYSTEM_PROMPT}\n\n{server_instructions}".strip() if server_instructions else PRIME_SYSTEM_PROMPT
            custom_system_prompt_key = DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY
        if file_access_policy.read_enabled:
            access = "read/write" if file_access_policy.write_enabled else "read-only"
            scope = "Reading is allowed everywhere; writing remains limited to output and selected folders." if file_access_policy.read_everywhere else "Use @alias/path from wangp_io roots; plain paths use @outputs."
            file_access_instructions = f"Filesystem {access} access is enabled. {scope} Use wangp_io for file discovery, text access, ZIP creation or extraction, and downloads. When a directory listing has_more, repeat the same filters with offset=next_offset."
        else:
            file_access_instructions = "Filesystem access is disabled. Use Gallery/media ids rather than direct paths; wangp_io can only inspect or download Gallery media."
        system_prompt = f"{system_prompt}\n\n{file_access_instructions}"
        remote_engine = resolve_role_engine(server_config, "deepy")
        if is_remote_engine(remote_engine):
            from shared.deepy.config import normalize_deepy_custom_system_prompt
            from shared.remote_llm.deepy_runner import run_remote_deepy_turn

            custom_prompt = normalize_deepy_custom_system_prompt(server_config.get(custom_system_prompt_key, ""))
            system_context_getter = getattr(tools, "get_system_context", None)
            system_context = str(system_context_getter() or "").strip() if callable(system_context_getter) else ""
            remote_system_prompt = "\n\n".join(part for part in (system_prompt, custom_prompt, system_context) if part).strip()
            return run_remote_deepy_turn(server_config, session, original_prompts[0] if original_prompts else "", remote_system_prompt, tools, send_cmd)
        _assistant_instructions, assistant_max_new_tokens = self._deps.resolve_prompt_enhancer_settings("", assistant_model_def, prompt_enhancer_modes, is_image=False, text_encoder_max_tokens=1024)
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
            system_prompt=system_prompt,
            custom_system_prompt_key=custom_system_prompt_key,
        )
        with deepy_log_scope(start_if_needed=debug_enabled):
            assistant.run_turn(
                original_prompts[0] if len(original_prompts) > 0 else "",
                max_new_tokens=max(1024, int(assistant_max_new_tokens)),
                seed=assistant_seed,
                do_sample=True,
                temperature=enhancer_temperature,
                top_p=enhancer_top_p,
            )

    def ask_ai(self, state, ask_request, client_submission_id: str = "", steering: bool = False):
        debug_enabled = self._sync_debug_enabled()
        submission_id = str(client_submission_id or "").strip()[:128]
        acknowledged_submission_ids = [submission_id] if submission_id else []

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
            if debug_enabled:
                self._debug_log("Request ignored because it was empty after normalization.")
            yield assistant_chat.build_sync_event(session, acknowledged_submission_ids=acknowledged_submission_ids), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        if debug_enabled:
            self._debug_log(f"Request received blocks={len(request_blocks)} worker_active={bool(session.worker_active)} queued_jobs={int(session.queued_job_count or 0)} foreign_session_reset={bool(foreign_session_reset)}")
        if session.drop_state_requested:
            if debug_enabled:
                self._debug_log("Request held because a Deepy reset is pending.")
            status = {"visible": True, "kind": "queued", "text": "Resetting after the current work stops..."}
            yield assistant_chat.build_sync_event(session, status=status, acknowledged_submission_ids=acknowledged_submission_ids), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        if not self.is_available():
            if debug_enabled:
                self._debug_log(f"Request rejected: {self.requirement_error_text()}")
            error_turn_id = assistant_chat.create_assistant_turn(session)
            assistant_chat.set_assistant_content(session, error_turn_id, self.requirement_error_text())
            yield assistant_chat.build_sync_event(session, acknowledged_submission_ids=acknowledged_submission_ids), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        if steering:
            with self._queue_state_lock:
                steering_active = bool(session.worker_active and session.control_queue is not None)
                if steering_active:
                    output_queue = session.control_queue
                    queued_epoch = session.chat_epoch
                    with session.turn_lock:
                        checkpoint = session.current_turn
                        steered_current_turn = isinstance(checkpoint, dict)
                        if steered_current_turn:
                            request_assistant_steering(session)
                    for index, request_block in enumerate(request_blocks):
                        self._queue_assistant_request(state, session, output_queue, request_block, queued_epoch, queued=True, client_submission_id=submission_id, assistant_badge="Steered" if steered_current_turn and index == 0 else "")
                    if steered_current_turn:
                        if session.assistant_action_active:
                            steering_text = "Steering accepted. Steering will apply once the current tool action is done."
                        elif session.assistant_thought_active:
                            steering_text = "Steering accepted. Waiting for the current thought to finish..."
                        else:
                            steering_text = "Steering accepted. Applying the new instructions at the current boundary..."
                    else:
                        steering_text = "Queued behind the current assistant task."
                    steering_status = {"visible": True, "kind": "queued", "text": steering_text}
                    steering_sync = assistant_chat.build_sync_event(session, status=steering_status, acknowledged_submission_ids=acknowledged_submission_ids)
            if steering_active:
                self._debug_log(f"Steering requested worker_active=True active_turn={steered_current_turn} thought_active={bool(session.assistant_thought_active)} action_active={bool(session.assistant_action_active)} queued_jobs={int(session.queued_job_count or 0)}")
                yield steering_sync, gr.update(), gr.update(value=""), gr.update(), gr.update()
                return
        com_stream = AsyncStream()
        output_queue = com_stream.output_queue
        with self._queue_state_lock:
            queued = foreign_session_reset or session.worker_active or session.queued_job_count > 0
            queued_epoch = session.chat_epoch
            for index, request_block in enumerate(request_blocks):
                self._queue_assistant_request(state, session, output_queue, request_block, queued_epoch, queued=queued or index > 0, client_submission_id=submission_id)
            accepted_status = None if queued else {"visible": True, "kind": "loading", "text": "Starting Deepy..."}
            accepted_sync = assistant_chat.build_sync_event(session, status=accepted_status, acknowledged_submission_ids=acknowledged_submission_ids)
        yield accepted_sync, gr.update(), gr.update(value=""), gr.update(), gr.update()
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

    def enqueue_ai_while_busy(self, state, ask_request, client_submission_id: str = ""):
        self._sync_debug_enabled()
        session = get_or_create_assistant_session(state)
        submission_id = str(client_submission_id or "").strip()[:128]
        acknowledged_submission_ids = [submission_id] if submission_id else []
        request_blocks = self._expand_assistant_requests(session, ask_request)
        if len(request_blocks) == 0:
            yield assistant_chat.build_sync_event(session, acknowledged_submission_ids=acknowledged_submission_ids), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        if session.drop_state_requested:
            status = {"visible": True, "kind": "queued", "text": "Resetting after the current work stops..."}
            yield assistant_chat.build_sync_event(session, status=status, acknowledged_submission_ids=acknowledged_submission_ids), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        if not self.is_available():
            error_turn_id = assistant_chat.create_assistant_turn(session)
            assistant_chat.set_assistant_content(session, error_turn_id, self.requirement_error_text())
            yield assistant_chat.build_sync_event(session, acknowledged_submission_ids=acknowledged_submission_ids), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        with self._queue_state_lock:
            output_queue = session.control_queue
            enqueue_active = output_queue is not None and (session.worker_active or session.queued_job_count > 0)
            if enqueue_active:
                queued_epoch = session.chat_epoch
                for request_block in request_blocks:
                    self._queue_assistant_request(state, session, output_queue, request_block, queued_epoch, queued=True, client_submission_id=submission_id)
                status = {"visible": True, "kind": "queued", "text": "Queued behind the current assistant task."}
                queued_sync = assistant_chat.build_sync_event(session, status=status, acknowledged_submission_ids=acknowledged_submission_ids)
        if enqueue_active:
            yield queued_sync, gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        yield from self.ask_ai(state, ask_request, client_submission_id=submission_id)

    def stop_ai(self, state, queued_action=""):
        if str(queued_action or "").strip():
            return self._apply_queued_request_action(state, queued_action)
        session = get_or_create_assistant_session(state)
        with self._queue_state_lock:
            worker_active = session.worker_active
            interrupt_requested = session.interrupt_requested
            if worker_active and not interrupt_requested:
                with session.turn_lock:
                    request_assistant_interrupt(session)
        if worker_active:
            cancelled_user_text = self._cancel_next_queued_request(session) if interrupt_requested else ""
            if cancelled_user_text:
                return assistant_chat.build_sync_event(session), gr.update(), gr.update(), gr.update()
            cancelled_job_id = self._cancel_active_prime_job(session, "Stop")
            self._debug_log(f"Stop requested worker_active=True active_prime_mcp_job={cancelled_job_id or 'none'}")
            status_text = "Stopping generation..." if cancelled_job_id else "Interrupting the current assistant task..."
            status = {"visible": True, "kind": "queued", "text": status_text}
            return assistant_chat.build_sync_event(session, status=status), gr.update(), gr.update(), gr.update()
        cancelled_user_text = self._cancel_next_queued_request(session)
        if cancelled_user_text:
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
        return assistant_chat.build_reset_event(session), gr.update(), gr.update(value=""), gr.update()

    def browser_session_started(self, state):
        session = get_or_create_assistant_session(state)
        if self._reset_foreign_active_session(session):
            return assistant_chat.build_reset_event(session), gr.update(), gr.update(value=""), gr.update()
        if not session.worker_active and session.queued_job_count <= 0:
            return gr.update(), gr.update(), gr.update(), gr.update()
        request_assistant_reset(session)
        session.chat_html = ""
        return assistant_chat.build_status_event("Resetting after the current work stops...", kind="queued"), gr.update(), gr.update(value=""), gr.update()


def create_controller(**deps_kwargs) -> DeepyController:
    return DeepyController(DeepyDeps(**deps_kwargs))
