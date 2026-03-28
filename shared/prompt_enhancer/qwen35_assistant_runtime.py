from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from shared.llm_engines.nanovllm import SamplingParams
from shared.llm_engines.nanovllm.engine.block_manager import BlockManager
from shared.llm_engines.nanovllm.engine.sequence import Sequence, SequenceStatus
from shared.prompt_enhancer import qwen35_text
from shared.prompt_enhancer.streaming import ThrottledStreamEmitter


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL | re.IGNORECASE)
_TOOL_BLOCK_RE = re.compile(r"\s*<tool_call>\s*.*?\s*</tool_call>\s*", flags=re.DOTALL | re.IGNORECASE)
_TRAILING_STOP_RE = re.compile(r"(?:<\|im_end\|>\s*|</s>\s*)+$", flags=re.IGNORECASE)
_FUNCTION_TAG_RE = re.compile(r"<function(?:=|\s+name=)([^\s>]+)[^>]*>(.*?)</function>", flags=re.DOTALL | re.IGNORECASE)
_FUNCTION_START_RE = re.compile(r"<function(?:=|\s+name=)([^\s>]+)[^>]*>", flags=re.IGNORECASE)
_PARAM_TAG_RE = re.compile(r"<parameter(?:=|\s+name=)([^\s>]+)[^>]*>(.*?)</parameter>", flags=re.DOTALL | re.IGNORECASE)
_GENERIC_PARAM_TAG_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</(?:parameter|\1)>", flags=re.DOTALL | re.IGNORECASE)
_MIN_PREFILL_REBUILD_SUFFIX_TOKENS = 256
_PREFILL_REBUILD_SUFFIX_RATIO_DIVISOR = 2


@dataclass(slots=True)
class AssistantDecodeResult:
    raw_text: str
    stop_reason: str
    token_count: int
    stop_token_id: int | None = None


def render_assistant_messages(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, add_generation_prompt: bool, thinking_enabled: bool) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=bool(add_generation_prompt),
        tokenize=True,
        enable_thinking=bool(thinking_enabled),
    )
    if torch.is_tensor(rendered):
        rendered = rendered.tolist()
    return [int(token_id) for token_id in rendered]


def render_text_user_turn_suffix(tokenizer, user_content: str, thinking_enabled: bool) -> list[int]:
    user_content = str(user_content or "").strip()
    if len(user_content) == 0:
        return []
    suffix = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    suffix += "<think>\n" if bool(thinking_enabled) else "<think>\n\n</think>\n\n"
    token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if torch.is_tensor(token_ids):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def render_tool_turn_suffix(tokenizer, tool_contents: list[str], thinking_enabled: bool) -> list[int]:
    normalized_contents = [str(content or "").strip() for content in list(tool_contents or []) if len(str(content or "").strip()) > 0]
    if len(normalized_contents) == 0:
        return []
    suffix = "<|im_end|>\n<|im_start|>user"
    for tool_content in normalized_contents:
        suffix += f"\n<tool_response>\n{tool_content}\n</tool_response>"
    suffix += "<|im_end|>\n<|im_start|>assistant\n"
    suffix += "<think>\n" if bool(thinking_enabled) else "<think>\n\n</think>\n\n"
    token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if torch.is_tensor(token_ids):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def strip_tool_blocks(raw_text: str) -> str:
    return _TOOL_BLOCK_RE.sub("\n", str(raw_text or "")).strip()


def strip_trailing_stop_markup(raw_text: str) -> str:
    return _TRAILING_STOP_RE.sub("", str(raw_text or "")).rstrip()


def _clean_tag_name(name: str) -> str:
    name = str(name or "").strip()
    if len(name) >= 2 and name[0] in ("'", '"') and name[-1] == name[0]:
        return name[1:-1].strip()
    return name


def _parse_tagged_tool_call(payload: str) -> dict[str, Any] | None:
    function_match = _FUNCTION_TAG_RE.search(str(payload or ""))
    function_body = ""
    if function_match is not None:
        name = _clean_tag_name(function_match.group(1))
        function_body = function_match.group(2)
    else:
        function_start_match = _FUNCTION_START_RE.search(str(payload or ""))
        if function_start_match is None:
            return None
        name = _clean_tag_name(function_start_match.group(1))
        function_body = str(payload or "")[function_start_match.end():]
    if len(name) == 0:
        return None
    arguments = {}
    for param_name, param_value in _PARAM_TAG_RE.findall(function_body):
        clean_name = _clean_tag_name(param_name)
        clean_value = str(param_value or "").strip()
        if len(clean_name) == 0:
            continue
        try:
            arguments[clean_name] = json.loads(clean_value)
        except Exception:
            arguments[clean_name] = clean_value
    for param_name, param_value in _GENERIC_PARAM_TAG_RE.findall(function_body):
        clean_name = _clean_tag_name(param_name)
        clean_value = str(param_value or "").strip()
        if len(clean_name) == 0 or clean_name.lower() in {"function", "parameter"} or clean_name in arguments:
            continue
        try:
            arguments[clean_name] = json.loads(clean_value)
        except Exception:
            arguments[clean_name] = clean_value
    return {"name": name, "arguments": arguments}


def _normalize_tool_call_dict(parsed: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    name = str(parsed.get("name", "")).strip()
    arguments = parsed.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    if len(name) == 0:
        return None
    return {"name": name, "arguments": arguments}


def _extract_bare_json_tool_call(text: str) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
    decoder = json.JSONDecoder()
    source_text = str(text or "")
    for start_idx, ch in enumerate(source_text):
        if ch not in "{[":
            continue
        try:
            parsed, end_idx = decoder.raw_decode(source_text[start_idx:])
        except Exception:
            continue
        tool_call = _normalize_tool_call_dict(parsed)
        if tool_call is None:
            continue
        return tool_call, (start_idx, start_idx + end_idx)
    return None, None


def _extract_inline_tool_call(text: str) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
    candidate = strip_trailing_stop_markup(str(text or "")).strip()
    if len(candidate) == 0:
        return None, None
    tagged_tool_call = _parse_tagged_tool_call(candidate)
    if tagged_tool_call is not None:
        return tagged_tool_call, (0, len(candidate))
    return _extract_bare_json_tool_call(candidate)


def extract_tool_calls(raw_text: str) -> list[dict[str, Any]]:
    tool_calls = []
    source_text = str(raw_text or "")
    for match in _TOOL_CALL_RE.finditer(source_text):
        payload = match.group(1).strip()
        if len(payload) == 0:
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = _parse_tagged_tool_call(payload)
        tool_call = _normalize_tool_call_dict(parsed)
        if tool_call is None:
            continue
        tool_calls.append(tool_call)
    if len(tool_calls) > 0:
        return tool_calls
    _thinking_text, answer_text = qwen35_text._split_generated_text(source_text)
    inline_tool_call, _inline_span = _extract_inline_tool_call(answer_text)
    if inline_tool_call is not None:
        tool_calls.append(inline_tool_call)
    return tool_calls


def strip_inline_tool_call_text(raw_text: str) -> str:
    text = strip_trailing_stop_markup(str(raw_text or ""))
    inline_tool_call, inline_span = _extract_inline_tool_call(text)
    if inline_tool_call is None or inline_span is None:
        return text
    start_idx, end_idx = inline_span
    stripped_text = (text[:start_idx] + text[end_idx:]).strip()
    return stripped_text


def has_complete_tool_call(raw_text: str) -> bool:
    text = str(raw_text or "")
    if len(_TOOL_CALL_RE.findall(text)) > 0:
        return text.count("<tool_call>") == text.count("</tool_call>")
    _thinking_text, answer_text = qwen35_text._split_generated_text(text)
    inline_tool_call, _inline_span = _extract_inline_tool_call(answer_text)
    return inline_tool_call is not None


class Qwen35AssistantRuntime:
    def __init__(self, model, debug_enabled: bool = False):
        self.model = model
        self.tokenizer = getattr(model, "_prompt_enhancer_tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("Prompt enhancer tokenizer is missing for assistant runtime.")
        self.debug_enabled = bool(debug_enabled)
        self._runtime_extra_tokens = getattr(model, "_prompt_enhancer_thinking_extra_tokens", 0)

    def _log(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[AssistantRuntime] {message}")

    @staticmethod
    def _format_preview(text: str, limit: int = 120) -> str:
        normalized = str(text or "").replace("\r", "\\r").replace("\n", "\\n")
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit]}..."

    @staticmethod
    def _describe_tensor(value: Any) -> str:
        if value is None:
            return "None"
        if not torch.is_tensor(value):
            return type(value).__name__
        try:
            ptr = int(value.data_ptr())
        except Exception:
            ptr = None
        return f"shape={tuple(int(x) for x in value.shape)} dtype={str(value.dtype).replace('torch.', '')} device={value.device} ptr={ptr}"

    def _describe_engine_state(self, engine) -> str:
        llm = None if engine is None else getattr(engine, "_llm", None)
        runner = None if llm is None else getattr(llm, "model_runner", None)
        live_model_len = 0 if llm is None else int(getattr(llm.config, "max_model_len", 0) or 0)
        runtime_ready = None if runner is None else getattr(runner, "_runtime_ready", None)
        runtime_signature = None if runner is None else getattr(runner, "_runtime_signature", None)
        kv_cache = None if runner is None else getattr(runner, "kv_cache", None)
        return (
            f"engine_id={id(engine) if engine is not None else None} "
            f"llm_id={id(llm) if llm is not None else None} "
            f"hints=(model_len={getattr(engine, '_max_model_len_hint', None)}, seqs={getattr(engine, '_max_num_seqs_hint', None)}, batched={getattr(engine, '_max_num_batched_tokens_hint', None)}) "
            f"live_model_len={live_model_len} runtime_ready={runtime_ready} runtime_signature={runtime_signature} "
            f"kv_cache={self._describe_tensor(kv_cache)}"
        )

    def _get_engine(self, max_context_tokens: int, max_new_tokens: int):
        engine = qwen35_text._get_or_create_vllm_engine(self.model, usage_mode="assistant")
        desired_model_len, desired_num_seqs, desired_num_batched_tokens = engine._compute_runtime_hints(prompt_len=max_context_tokens, max_tokens=max_new_tokens, cfg_scale=1.0)
        desired_model_len = max(desired_model_len, engine._get_min_model_len_hint())
        desired_num_batched_tokens = max(desired_num_batched_tokens, desired_model_len * desired_num_seqs)
        live_llm = getattr(engine, "_llm", None)
        self._log(
            "Requesting assistant engine "
            f"context={int(max_context_tokens)} max_new={int(max_new_tokens)} desired=(model_len={desired_model_len}, seqs={desired_num_seqs}, batched={desired_num_batched_tokens}) "
            f"live_before={self._describe_engine_state(engine)}"
        )
        if live_llm is not None and (
            int(getattr(live_llm.config, "max_model_len", 0) or 0) != desired_model_len
            or int(getattr(engine, "_max_num_seqs_hint", 0) or 0) != desired_num_seqs
            or int(getattr(engine, "_max_num_batched_tokens_hint", 0) or 0) != desired_num_batched_tokens
        ):
            self._log("Closing assistant engine before reserve because live runtime hints do not match the requested embedded decode.")
            engine.close()
            engine._max_model_len_hint = None
            engine._max_num_seqs_hint = None
            engine._max_num_batched_tokens_hint = None
        engine.reserve_runtime(prompt_len=max_context_tokens, max_tokens=max_new_tokens, cfg_scale=1.0)
        engine._ensure_llm()
        if engine._llm is None:
            raise RuntimeError("Assistant NanoVLLM runtime is not available.")
        self._log(f"Assistant engine ready after reserve: {self._describe_engine_state(engine)}")
        return engine

    def _get_linear_state_modules(self):
        llm = self._get_live_llm()
        modules = []
        for module in llm.model_runner.model.modules():
            if getattr(module, "layer_type", None) == "linear_attention" and hasattr(module, "conv_state_buffer") and hasattr(module, "recurrent_state_buffer"):
                modules.append(module)
        return modules

    def _get_live_llm(self):
        engine = getattr(self.model, "_prompt_enhancer_vllm_engine", None)
        llm = None if engine is None else getattr(engine, "_llm", None)
        if llm is None:
            raise RuntimeError("Assistant runtime is not initialized.")
        return llm

    def get_max_model_len(self) -> int:
        return int(getattr(self._get_live_llm().config, "max_model_len", 0) or 0)

    def _ensure_clean_runtime(self, max_context_tokens: int, max_new_tokens: int, seed: int | None = None):
        engine = self._get_engine(max_context_tokens=max_context_tokens, max_new_tokens=max_new_tokens)
        llm = engine._llm
        llm.reset()
        llm.model_runner.ensure_runtime_ready()
        if llm.config.num_kvcache_blocks > 0 and len(llm.scheduler.block_manager.blocks) != llm.config.num_kvcache_blocks:
            llm.scheduler.block_manager = BlockManager(llm.config.num_kvcache_blocks, llm.config.kvcache_block_size)
        llm.model_runner.reset_generation_state()
        llm.model_runner.call("set_sampling_seed", None if seed is None else int(seed))
        llm.scheduler.waiting.clear()
        llm.scheduler.running.clear()
        return engine, llm

    def _build_sampling_params(self, max_new_tokens: int, seed: int | None, do_sample: bool, temperature: float | None, top_p: float | None, top_k: int | None, thinking_enabled: bool):
        logits_bias = qwen35_text._build_suppressed_token_logits_bias(self.model, thinking_enabled=thinking_enabled)
        logits_processor, logits_processor_update_state = qwen35_text._build_prompt_logits_processor(self.model, thinking_enabled=thinking_enabled)
        temp, normalized_top_p, normalized_top_k = qwen35_text._normalize_vllm_sampling(
            do_sample=bool(do_sample),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        runtime_extra = qwen35_text._resolve_prompt_runtime_extra_tokens(self.model, thinking_enabled=thinking_enabled)
        return SamplingParams(
            temperature=temp,
            max_tokens=int(max_new_tokens) + int(runtime_extra),
            cfg_scale=1.0,
            top_k=normalized_top_k,
            top_p=normalized_top_p,
            min_p=qwen35_text._resolve_prompt_min_p(self.model),
            ignore_eos=True,
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            logits_bias=logits_bias,
            seed=None if seed is None else int(seed),
        )

    def _get_active_sequence(self):
        try:
            llm = self._get_live_llm()
        except Exception:
            return None
        if llm.scheduler.running:
            return llm.scheduler.running[0]
        if llm.scheduler.waiting:
            return llm.scheduler.waiting[0]
        return None

    def _seal_sequence(self, seq: Sequence) -> None:
        seq.num_prompt_tokens = seq.num_tokens
        seq.logits_processor = None
        seq.logits_processor_update_state = None
        seq.ignore_eos = True

    def _should_rebuild_with_prefill(self, current_len: int, suffix_len: int) -> bool:
        suffix_len = int(suffix_len)
        if suffix_len <= 0:
            return False
        current_len = max(0, int(current_len))
        rebuild_threshold = max(_MIN_PREFILL_REBUILD_SUFFIX_TOKENS, current_len // _PREFILL_REBUILD_SUFFIX_RATIO_DIVISOR)
        return suffix_len >= rebuild_threshold

    def _prefill_context(self, token_ids: list[int], seed: int | None = None) -> Sequence:
        normalized_token_ids = [int(token_id) for token_id in token_ids]
        if len(normalized_token_ids) == 0:
            raise ValueError("Cannot prefill assistant context with an empty token sequence.")
        _engine, llm = self._ensure_clean_runtime(max_context_tokens=len(normalized_token_ids), max_new_tokens=1, seed=seed)
        seq = Sequence(normalized_token_ids, SamplingParams(max_tokens=1, ignore_eos=True))
        llm.scheduler.add(seq)
        scheduled, is_prefill = llm.scheduler.schedule()
        if not scheduled or not is_prefill:
            raise RuntimeError("Assistant context prefill did not schedule a prefill batch.")
        llm.model_runner.call("run", scheduled, is_prefill)
        seq = scheduled[0]
        self._seal_sequence(seq)
        self._log(f"Primed assistant context with {len(normalized_token_ids)} tokens.")
        return seq

    def _force_append_tokens(self, seq: Sequence, token_ids: list[int]) -> Sequence:
        if len(token_ids) == 0:
            return seq
        llm = self._get_live_llm()
        original_processor = seq.logits_processor
        original_update = seq.logits_processor_update_state
        original_max_tokens = seq.max_tokens
        seq.logits_processor = None
        seq.logits_processor_update_state = None
        seq.ignore_eos = True
        seq.max_tokens = max(int(original_max_tokens), seq.num_completion_tokens + len(token_ids) + 8)
        try:
            for forced_token in token_ids:
                scheduled, is_prefill = llm.scheduler.schedule()
                llm.model_runner.call("run", scheduled, is_prefill)
                llm.scheduler.postprocess(scheduled, [int(forced_token)])
                seq = scheduled[0]
        finally:
            seq.logits_processor = original_processor
            seq.logits_processor_update_state = original_update
            seq.max_tokens = original_max_tokens
        return seq

    def prime_context(self, token_ids: list[int], seed: int | None = None) -> Sequence:
        return self._prefill_context(token_ids, seed=seed)

    def restore_or_replay(self, snapshot: dict[str, Any] | None, fallback_tokens: list[int], seed: int | None = None) -> tuple[str, str]:
        if snapshot:
            try:
                self.restore_snapshot(snapshot)
                return "restored", "exact KV snapshot restored"
            except Exception as exc:
                self._log(f"Exact restore failed, falling back to replay: {exc}")
                self.prime_context(fallback_tokens, seed=seed)
                return "replayed", f"exact KV restore failed: {exc}"
        self.prime_context(fallback_tokens, seed=seed)
        return "replayed", "no exact runtime snapshot was available"

    def extend_context(self, target_token_ids: list[int]) -> str:
        seq = self._get_active_sequence()
        if seq is None:
            self.prime_context(target_token_ids)
            return "replayed"
        current_token_ids = [int(token_id) for token_id in seq.token_ids]
        if target_token_ids[: len(current_token_ids)] != current_token_ids:
            self.prime_context(target_token_ids)
            return "replayed"
        suffix = [int(token_id) for token_id in target_token_ids[len(current_token_ids) :]]
        if suffix:
            if self._should_rebuild_with_prefill(len(current_token_ids), len(suffix)):
                self.prime_context(target_token_ids)
                return "prefilled"
            seq = self._force_append_tokens(seq, suffix)
        self._seal_sequence(seq)
        return "extended"

    def append_suffix(self, suffix_token_ids: list[int]) -> str:
        seq = self._get_active_sequence()
        if seq is None:
            raise RuntimeError("Assistant context is not initialized.")
        suffix = [int(token_id) for token_id in suffix_token_ids]
        if suffix:
            if self._should_rebuild_with_prefill(len(seq.token_ids), len(suffix)):
                self.prime_context([int(token_id) for token_id in seq.token_ids] + suffix)
                return "prefilled"
            seq = self._force_append_tokens(seq, suffix)
        self._seal_sequence(seq)
        return "extended"

    def generate_embedded_answer(
        self,
        prompt_token_ids: list[int],
        prompt_embeds,
        prompt_position_ids,
        position_offset: int,
        *,
        max_new_tokens: int,
        seed: int | None,
        do_sample: bool,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
    ) -> str:
        self._log(
            "Embedded decode request "
            f"prompt_tokens={len(prompt_token_ids)} prompt_embeds={self._describe_tensor(prompt_embeds)} "
            f"prompt_position_ids={self._describe_tensor(prompt_position_ids)} position_offset={int(position_offset or 0)} "
            f"max_new={int(max_new_tokens)} seed={seed} do_sample={bool(do_sample)}"
        )
        snapshot = self.snapshot_context()
        engine = self._get_engine(max_context_tokens=len(prompt_token_ids), max_new_tokens=max_new_tokens)
        try:
            temp, normalized_top_p, normalized_top_k = qwen35_text._normalize_vllm_sampling(
                do_sample=bool(do_sample),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            response = engine.generate_embedded(
                prompt_token_ids=[int(token_id) for token_id in prompt_token_ids],
                prompt_embeds=prompt_embeds,
                prompt_position_ids=prompt_position_ids,
                max_tokens=int(max_new_tokens),
                temperature=temp,
                top_p=normalized_top_p,
                top_k=normalized_top_k,
                cfg_scale=1.0,
                seed=seed,
                use_tqdm=True,
                release_vram_after=False,
                ignore_eos=False,
                position_offset=int(position_offset or 0),
            )
            cleaned = qwen35_text._clean_generated_text("" if response is None else response.get("text", ""))
            self._log(f"Embedded decode response preview={self._format_preview(cleaned)}")
            return cleaned
        finally:
            if snapshot is not None:
                self._log("Embedded decode finished; restoring assistant snapshot.")
                self.restore_snapshot(snapshot)
            else:
                self._log("Embedded decode finished without an active assistant snapshot; releasing runtime allocations.")
                engine.release_runtime_allocations()

    def start_generation_segment(self, max_new_tokens: int, seed: int | None, do_sample: bool, temperature: float | None, top_p: float | None, top_k: int | None, thinking_enabled: bool) -> Sequence:
        seq = self._get_active_sequence()
        if seq is None:
            raise RuntimeError("Assistant context is not initialized.")
        sampling_params = self._build_sampling_params(max_new_tokens=max_new_tokens, seed=seed, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, thinking_enabled=thinking_enabled)
        seq.num_prompt_tokens = seq.num_tokens
        seq.max_tokens = sampling_params.max_tokens
        seq.temperature = sampling_params.temperature
        seq.ignore_eos = True
        seq.top_k = sampling_params.top_k
        seq.top_p = sampling_params.top_p
        seq.min_p = sampling_params.min_p
        seq.cfg_scale = sampling_params.cfg_scale
        seq.logits_processor = sampling_params.logits_processor
        seq.logits_processor_update_state = sampling_params.logits_processor_update_state
        seq.logits_bias = sampling_params.logits_bias
        llm = self._get_live_llm()
        llm.model_runner.call("set_sampling_seed", sampling_params.seed)
        return seq

    def generate_segment(self, max_new_tokens: int, seed: int | None, do_sample: bool, temperature: float | None, top_p: float | None, top_k: int | None, thinking_enabled: bool, stop_requested=None, stream_callback=None, stream_interval_seconds: float = 1.0) -> AssistantDecodeResult:
        seq = self.start_generation_segment(max_new_tokens=max_new_tokens, seed=seed, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, thinking_enabled=thinking_enabled)
        stop_token_ids = {int(token_id) for token_id in getattr(self.model, "_prompt_enhancer_stop_token_ids", []) or [] if int(token_id) >= 0}
        stream_emitter = ThrottledStreamEmitter(stream_interval_seconds) if callable(stream_callback) else None
        raw_text = ""
        for step_no in range(int(seq.max_tokens)):
            if callable(stop_requested) and stop_requested():
                if stream_emitter is not None:
                    stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=step_no, stop_reason="interrupted", is_final=True, force=True)
                return AssistantDecodeResult(raw_text=raw_text, stop_reason="interrupted", token_count=step_no)
            llm = self._get_live_llm()
            if len(seq.token_ids) >= int(llm.config.max_model_len):
                if stream_emitter is not None:
                    stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=step_no, stop_reason="context_limit", is_final=True, force=True)
                return AssistantDecodeResult(raw_text=raw_text, stop_reason="context_limit", token_count=step_no)
            try:
                scheduled, is_prefill = llm.scheduler.schedule()
            except AssertionError:
                if len(seq.token_ids) >= int(llm.config.max_model_len):
                    if stream_emitter is not None:
                        stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=step_no, stop_reason="context_limit", is_final=True, force=True)
                    return AssistantDecodeResult(raw_text=raw_text, stop_reason="context_limit", token_count=step_no)
                raise
            sampled_token_ids = llm.model_runner.call("run", scheduled, is_prefill)
            llm.scheduler.postprocess(scheduled, sampled_token_ids)
            seq = scheduled[0]
            raw_text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            last_token_id = int(sampled_token_ids[0])
            if stream_emitter is not None:
                stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=step_no + 1, stop_reason=None, is_final=False)
            if has_complete_tool_call(raw_text):
                if stream_emitter is not None:
                    stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=step_no + 1, stop_reason="tool_call", is_final=True, force=True)
                return AssistantDecodeResult(raw_text=raw_text, stop_reason="tool_call", token_count=step_no + 1, stop_token_id=last_token_id)
            if last_token_id in stop_token_ids:
                if stream_emitter is not None:
                    stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=step_no + 1, stop_reason="stop_token", is_final=True, force=True)
                return AssistantDecodeResult(raw_text=raw_text, stop_reason="stop_token", token_count=step_no + 1, stop_token_id=last_token_id)
        raw_text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if stream_emitter is not None:
            stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=int(seq.max_tokens), stop_reason="max_tokens", is_final=True, force=True)
        return AssistantDecodeResult(raw_text=raw_text, stop_reason="max_tokens", token_count=int(seq.max_tokens))

    def snapshot_context(self) -> dict[str, Any] | None:
        seq = self._get_active_sequence()
        if seq is None:
            self._log("Snapshot requested but no active assistant sequence is available.")
            return None
        llm = self._get_live_llm()
        runner = llm.model_runner
        torch.cuda.synchronize()
        linear_modules = self._get_linear_state_modules()
        snapshot = {
            "max_model_len_hint": getattr(getattr(self.model, "_prompt_enhancer_vllm_engine", None), "_max_model_len_hint", None),
            "max_num_seqs_hint": getattr(getattr(self.model, "_prompt_enhancer_vllm_engine", None), "_max_num_seqs_hint", None),
            "max_num_batched_tokens_hint": getattr(getattr(self.model, "_prompt_enhancer_vllm_engine", None), "_max_num_batched_tokens_hint", None),
            "runner_max_model_len": int(llm.config.max_model_len),
            "sequence": {
                "token_ids": [int(token_id) for token_id in seq.token_ids],
                "num_prompt_tokens": int(seq.num_prompt_tokens),
                "num_cached_tokens": int(seq.num_cached_tokens),
                "block_table": [int(block_id) for block_id in seq.block_table],
                "status": seq.status.name,
                "max_tokens": int(seq.max_tokens),
                "temperature": float(seq.temperature),
                "ignore_eos": bool(seq.ignore_eos),
                "top_k": None if seq.top_k is None else int(seq.top_k),
                "top_p": None if seq.top_p is None else float(seq.top_p),
                "min_p": None if seq.min_p is None else float(seq.min_p),
            },
            "block_manager": {
                "block_size": int(llm.scheduler.block_manager.block_size),
                "blocks": [
                    {
                        "ref_count": int(block.ref_count),
                        "hash": int(block.hash),
                        "token_ids": [int(token_id) for token_id in block.token_ids],
                    }
                    for block in llm.scheduler.block_manager.blocks
                ],
                "hash_to_block_id": {int(hash_key): int(block_id) for hash_key, block_id in llm.scheduler.block_manager.hash_to_block_id.items()},
                "free_block_ids": [int(block_id) for block_id in llm.scheduler.block_manager.free_block_ids],
                "used_block_ids": [int(block_id) for block_id in llm.scheduler.block_manager.used_block_ids],
            },
            "kv_cache": None if not hasattr(runner, "kv_cache") else runner.kv_cache.detach().to("cpu").clone(),
            "linear_states": [
                {
                    "conv": module.conv_state_buffer.detach().to("cpu").clone(),
                    "recurrent": module.recurrent_state_buffer.detach().to("cpu").clone(),
                }
                for module in linear_modules
            ],
        }
        self._log(
            f"Snapshotted assistant context with {len(seq.token_ids)} tokens. "
            f"saved_hints=(model_len={snapshot['max_model_len_hint']}, seqs={snapshot['max_num_seqs_hint']}, batched={snapshot['max_num_batched_tokens_hint']}) "
            f"runner_state={self._describe_engine_state(getattr(self.model, '_prompt_enhancer_vllm_engine', None))}"
        )
        return snapshot

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        engine = qwen35_text._get_or_create_vllm_engine(self.model, usage_mode="assistant")
        saved_model_len = int(snapshot.get("max_model_len_hint", 0) or snapshot.get("runner_max_model_len", 0) or 0)
        saved_num_seqs = int(snapshot.get("max_num_seqs_hint", 0) or 1)
        saved_num_batched_tokens = int(snapshot.get("max_num_batched_tokens_hint", 0) or 0)
        saved_model_len = max(saved_model_len, engine._get_min_model_len_hint())
        saved_num_seqs = max(1, saved_num_seqs)
        saved_num_batched_tokens = max(saved_num_batched_tokens, saved_model_len * saved_num_seqs)
        live_llm = getattr(engine, "_llm", None)
        self._log(
            "Restoring assistant snapshot "
            f"saved=(model_len={saved_model_len}, seqs={saved_num_seqs}, batched={saved_num_batched_tokens}) "
            f"live_before={self._describe_engine_state(engine)}"
        )
        if live_llm is not None and (
            int(getattr(live_llm.config, "max_model_len", 0) or 0) != saved_model_len
            or int(getattr(engine, "_max_num_seqs_hint", 0) or 0) != saved_num_seqs
            or int(getattr(engine, "_max_num_batched_tokens_hint", 0) or 0) != saved_num_batched_tokens
        ):
            self._log("Closing assistant engine before restore because live runtime hints do not match the saved snapshot.")
            engine.close()
        engine._max_model_len_hint = saved_model_len
        engine._max_num_seqs_hint = saved_num_seqs
        engine._max_num_batched_tokens_hint = saved_num_batched_tokens
        engine._ensure_llm()
        llm = engine._llm
        llm.reset()
        llm.model_runner.ensure_runtime_ready()
        runner = llm.model_runner
        if not hasattr(runner, "kv_cache"):
            raise RuntimeError("Assistant runtime has no KV cache to restore into.")
        kv_cache = snapshot.get("kv_cache")
        if kv_cache is None or tuple(kv_cache.shape) != tuple(runner.kv_cache.shape):
            saved_shape = None if kv_cache is None else tuple(int(x) for x in kv_cache.shape)
            live_shape = tuple(int(x) for x in runner.kv_cache.shape)
            self._log(f"Assistant KV cache snapshot mismatch saved_shape={saved_shape} live_shape={live_shape}")
            raise RuntimeError("Assistant KV cache snapshot shape does not match current runtime.")
        with torch.inference_mode():
            runner.kv_cache.copy_(kv_cache.to(device=runner.kv_cache.device, dtype=runner.kv_cache.dtype))
        linear_modules = self._get_linear_state_modules()
        linear_states = snapshot.get("linear_states", [])
        if len(linear_modules) != len(linear_states):
            raise RuntimeError("Assistant linear-state snapshot does not match runtime layer count.")
        with torch.inference_mode():
            for module, saved_state in zip(linear_modules, linear_states):
                saved_conv = saved_state["conv"].to(device=module.conv_state_buffer.device, dtype=module.conv_state_buffer.dtype)
                saved_recurrent = saved_state["recurrent"].to(device=module.recurrent_state_buffer.device, dtype=module.recurrent_state_buffer.dtype)
                if tuple(saved_conv.shape) != tuple(module.conv_state_buffer.shape) or tuple(saved_recurrent.shape) != tuple(module.recurrent_state_buffer.shape):
                    raise RuntimeError("Assistant linear-state snapshot tensor shape mismatch.")
                module.conv_state_buffer.copy_(saved_conv)
                module.recurrent_state_buffer.copy_(saved_recurrent)
        saved_block_manager = snapshot["block_manager"]
        llm.scheduler.block_manager = BlockManager(len(saved_block_manager["blocks"]), int(saved_block_manager["block_size"]))
        for block, saved_block in zip(llm.scheduler.block_manager.blocks, saved_block_manager["blocks"]):
            block.ref_count = int(saved_block["ref_count"])
            block.hash = int(saved_block["hash"])
            block.token_ids = [int(token_id) for token_id in saved_block["token_ids"]]
        llm.scheduler.block_manager.hash_to_block_id = {int(hash_key): int(block_id) for hash_key, block_id in saved_block_manager["hash_to_block_id"].items()}
        llm.scheduler.block_manager.free_block_ids = deque(int(block_id) for block_id in saved_block_manager["free_block_ids"])
        llm.scheduler.block_manager.used_block_ids = set(int(block_id) for block_id in saved_block_manager["used_block_ids"])
        saved_seq = snapshot["sequence"]
        restored_seq = Sequence([int(token_id) for token_id in saved_seq["token_ids"]], SamplingParams(max_tokens=int(saved_seq["max_tokens"]), ignore_eos=bool(saved_seq["ignore_eos"])))
        restored_seq.num_prompt_tokens = int(saved_seq["num_prompt_tokens"])
        restored_seq.num_cached_tokens = int(saved_seq["num_cached_tokens"])
        restored_seq.block_table = [int(block_id) for block_id in saved_seq["block_table"]]
        restored_seq.status = SequenceStatus[saved_seq["status"]]
        restored_seq.max_tokens = int(saved_seq["max_tokens"])
        restored_seq.temperature = float(saved_seq["temperature"])
        restored_seq.ignore_eos = bool(saved_seq["ignore_eos"])
        restored_seq.top_k = None if saved_seq["top_k"] is None else int(saved_seq["top_k"])
        restored_seq.top_p = None if saved_seq["top_p"] is None else float(saved_seq["top_p"])
        restored_seq.min_p = None if saved_seq["min_p"] is None else float(saved_seq["min_p"])
        restored_seq.logits_processor = None
        restored_seq.logits_processor_update_state = None
        llm.scheduler.waiting.clear()
        llm.scheduler.running.clear()
        if restored_seq.status == SequenceStatus.WAITING:
            llm.scheduler.waiting.append(restored_seq)
        else:
            restored_seq.status = SequenceStatus.RUNNING
            llm.scheduler.running.append(restored_seq)
        self._log(
            f"Restored assistant context with {len(restored_seq.token_ids)} tokens. "
            f"runner_state={self._describe_engine_state(engine)}"
        )
