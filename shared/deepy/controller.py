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
    DEEPY_VRAM_UNLOAD,
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
    clear_assistant_session,
    get_or_create_assistant_session,
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
        return normalize_deepy_vram_mode(server_config.get(DEEPY_VRAM_MODE_KEY, DEEPY_VRAM_UNLOAD))

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

    def release_vram(self, state, clear_session_state = False):
        session = get_or_create_assistant_session(state)
        release_callback = session.release_vram_callback
        session.release_vram_callback = None
        self._deps.clear_gpu_resident(state)
        if callable(release_callback):
            release_callback()
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

    def update_tool_ui_settings(self, state, *, auto_cancel_queue_tasks=None, use_template_properties=None, width=None, height=None, num_frames=None, video_with_speech_variant=None, image_generator_variant=None, image_editor_variant=None, video_generator_variant=None, speech_from_description_variant=None, speech_from_sample_variant=None, persist=False):
        session = get_or_create_assistant_session(state)
        normalized = set_assistant_tool_ui_settings(
            session,
            auto_cancel_queue_tasks=auto_cancel_queue_tasks,
            use_template_properties=use_template_properties,
            width=width,
            height=height,
            num_frames=num_frames,
            video_with_speech_variant=video_with_speech_variant,
            image_generator_variant=image_generator_variant,
            image_editor_variant=image_editor_variant,
            video_generator_variant=video_generator_variant,
            speech_from_description_variant=speech_from_description_variant,
            speech_from_sample_variant=speech_from_sample_variant,
        )
        if persist:
            server_config = self._server_config()
            server_config_filename = str(self._deps.get_server_config_filename() or "").strip()
            if deepy_ui_settings.store_assistant_tool_ui_settings(server_config, normalized):
                set_deepy_runtime_config(server_config, server_config_filename)
                if len(server_config_filename) > 0:
                    with open(server_config_filename, "w", encoding="utf-8") as writer:
                        writer.write(json.dumps(server_config, indent=4))
        return normalized

    def persist_auto_cancel_queue_tasks(self, state, auto_cancel_queue_tasks):
        session = get_or_create_assistant_session(state)
        current = dict(session.tool_ui_settings or deepy_ui_settings.normalize_assistant_tool_ui_settings())
        current["auto_cancel_queue_tasks"] = auto_cancel_queue_tasks
        normalized = deepy_ui_settings.normalize_assistant_tool_ui_settings(**current)
        session.tool_ui_settings = dict(normalized)
        server_config = self._server_config()
        server_config_filename = str(self._deps.get_server_config_filename() or "").strip()
        if deepy_ui_settings.store_assistant_tool_ui_settings(server_config, normalized):
            set_deepy_runtime_config(server_config, server_config_filename)
            if len(server_config_filename) > 0:
                with open(server_config_filename, "w", encoding="utf-8") as writer:
                    writer.write(json.dumps(server_config, indent=4))
        return normalized["auto_cancel_queue_tasks"]

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
            get_current_model_def=lambda: self._deps.get_model_def(self._deps.get_state_model_type(state)),
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
        assistant_model_def = model_def if model_def is not None else self._deps.get_model_def(self._deps.get_state_model_type(state))
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
        ask_request = str(ask_request or "").strip()
        if len(ask_request) == 0:
            yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            return
        if not self.is_available():
            error_turn_id = assistant_chat.create_assistant_turn(session)
            error_event = assistant_chat.set_assistant_content(session, error_turn_id, self.requirement_error_text())
            yield error_event if error_event is not None else gr.update(), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        gen = self._deps.get_gen_info(state)
        com_stream = AsyncStream()
        send_cmd = com_stream.output_queue.push
        queued = session.worker_active or session.queued_job_count > 0
        queued_epoch = session.chat_epoch
        session.queued_job_count += 1
        user_message_id, _user_event = assistant_chat.add_user_message(session, ask_request, queued=queued)
        yield assistant_chat.build_sync_event(session), gr.update(), gr.update(value=""), gr.update(), gr.update()
        if queued:
            yield assistant_chat.build_status_event("Queued behind the current assistant task.", kind="queued"), gr.update(), gr.update(), gr.update(), gr.update()

        def queue_worker_func():
            session.queued_job_count = max(0, session.queued_job_count - 1)
            if queued_epoch != session.chat_epoch:
                send_cmd("exit", None)
                return
            session.interrupt_requested = False
            session.control_queue = com_stream.output_queue
            session.worker_active = True
            begin_assistant_turn(session, user_message_id, ask_request)
            send_cmd("chat_output", assistant_chat.build_sync_event(session))
            queued_badge_event = assistant_chat.set_message_badge(session, user_message_id, None)
            if queued_badge_event is not None:
                send_cmd("chat_output", queued_badge_event)
            my_tools = self.create_tools(state, send_cmd, session=session)
            try:
                self._deps.exec_prompt_enhancer_engine(state, None, "AK", [ask_request], None, None, False, False, 0, None, 3.5, send_cmd, my_tools)
            except Exception as e:
                traceback.print_exc()
                error_turn_id = assistant_chat.create_assistant_turn(session)
                error_event = assistant_chat.set_assistant_content(session, error_turn_id, f"Assistant crashed: {e}")
                if error_event is not None:
                    send_cmd("chat_output", error_event)
                send_cmd("chat_output", assistant_chat.build_status_event(None, visible=False))
            finally:
                session.worker_active = False
                if session.control_queue is com_stream.output_queue:
                    session.control_queue = None
                if queued_epoch == session.chat_epoch:
                    send_cmd("chat_output", assistant_chat.build_sync_event(session))
                session.interrupt_requested = False
                send_cmd("exit", None)

        async_run_in("assistant", queue_worker_func)
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

    def stop_ai(self, state):
        session = get_or_create_assistant_session(state)
        if not session.worker_active:
            return gr.update(), gr.update(), gr.update(), gr.update()
        request_assistant_interrupt(session)
        return assistant_chat.build_status_event(None, visible=False), gr.update(), gr.update(), gr.update()

    def reset_ai(self, state):
        session = get_or_create_assistant_session(state)
        if session.worker_active:
            request_assistant_reset(session)
            assistant_chat.reset_session_chat(session)
        else:
            self.release_vram(state, True)
        session.chat_html = ""
        return assistant_chat.build_reset_event(), gr.update(), gr.update(value=""), gr.update()


def create_controller(**deps_kwargs) -> DeepyController:
    return DeepyController(DeepyDeps(**deps_kwargs))
