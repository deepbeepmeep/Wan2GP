from __future__ import annotations

import gc
import json
import os
import re
import sys
import types
from collections import OrderedDict

import torch
from mmgp import offload
from tqdm.auto import tqdm
from shared.utils import files_locator as fl
from shared.llm_engines.nanovllm import SamplingParams
from shared.llm_engines.nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM, clear_qwen35_runtime_caches
from shared.llm_engines.nanovllm.utils.context import reset_context
from shared.llm_engines.nanovllm.vllm_support import (
    NanoVllmTextEngine,
    resolve_lm_decoder_engine,
)
from shared.qtypes.gguf import GGUFWeightTensor, materialize_module_source_tensors

from .qwen3_5 import register_qwen35_config
from .qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from .qwen35_vl import (
    enhancer_quantization_GGUF,
    enhancer_quantization_QUANTO_INT8,
    _collect_suppressed_token_ids,
    _collect_stop_token_ids,
    _resolve_qwen35_checkpoint_file,
    _load_qwen35_tokenizer,
    _resolve_qwen35_asset_file,
    _resolve_qwen35_assets_dir,
    get_qwen35_variant_spec,
    get_qwen35_text_gguf_path,
)


QWEN35_TEXT_VLLM_SWITCH_ENV = "WGP_QWEN35_PROMPT_ENHANCER_VLLM"
QWEN35_TEXT_VLLM_CUDAGRAPH_ENV = "WGP_QWEN35_PROMPT_ENHANCER_VLLM_CUDAGRAPH"
QWEN35_GGUF_LLAMACPP_ENV = "WGP_GGUF_LLAMACPP_CUDA"
QWEN35_PROMPT_MIN_NEW_TOKENS = 4
QWEN35_PROMPT_DEFAULT_TOP_K = 20
QWEN35_PROMPT_DEFAULT_MIN_P_GGUF = 0.05
QWEN35_PROMPT_ENABLE_PRESENCE_PENALTY = True
QWEN35_PROMPT_PRESENCE_PENALTY = 1.5
QWEN35_PROMPT_SUPPRESS_LOGITS_BIAS = -1e4
QWEN35_PROMPT_ENABLE_THINKING = False
QWEN35_PROMPT_THINKING_EXTRA_TOKENS = 3000
QWEN35_PROMPT_THINKING_MAX_TOKENS = 2000


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _resolve_gguf_model_path(model_path: str | None, assets_dir: str, variant: str | None = None) -> str:
    if model_path is not None:
        resolved = fl.locate_file(model_path, error_if_none=False) or model_path
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Missing Qwen3.5 GGUF checkpoint: {resolved}")
        return resolved

    exact_path = get_qwen35_text_gguf_path(assets_dir, variant=variant)
    if os.path.isfile(exact_path):
        return exact_path
    raise FileNotFoundError(f"Missing expected Qwen3.5 GGUF checkpoint: {exact_path}")


def get_qwen35_text_assets_dir(assets_dir: str, variant: str | None = None) -> str:
    return _resolve_qwen35_assets_dir(assets_dir, variant=variant)


def get_qwen35_text_quanto_int8_path(assets_dir: str, variant: str | None = None) -> str:
    filename = get_qwen35_variant_spec(variant)["text_int8_filename"]
    return _resolve_qwen35_checkpoint_file(assets_dir, filename, variant=variant, error_if_none=False)


def _resolve_gguf_linear_attention_layout_from_filename(model_path: str) -> tuple[bool, bool, bool]:
    filename = os.path.basename(str(model_path or "")).strip().lower().replace("_", "-")
    if filename == "qwen3.5-9b-abliterated-q4-k-m-bis.gguf":
        return True, True, False
    if filename in {
        "qwen3.5-9b-abliterated-text-q4-k-m.gguf",
        "qwen3.5-4b-abliterated-text-q4-k-m.gguf",
    }:
        return False, True, False
    return False, False, True


def _load_text_config(config_path: str) -> Qwen3_5TextConfig:
    with open(config_path, "r", encoding="utf-8") as reader:
        config = json.load(reader)
    text_config = config.get("text_config", config)
    return Qwen3_5TextConfig(**text_config)

def _ensure_tied_output_weight(new_sd, tied_map):
    if "token_embd.weight" not in new_sd:
        return tied_map
    new_sd.pop("output.weight", None)
    tied_map = dict(tied_map or {})
    tied_list = list(tied_map.get("token_embd.weight", []))
    if "output.weight" not in tied_list:
        tied_list.append("output.weight")
    tied_map["token_embd.weight"] = tied_list
    return tied_map


def _build_qwen35_gguf_preprocess_sd(tie_output_to_embeddings: bool = False):
    def preprocess_sd(sd, quant_map=None, tied_map=None):
        new_sd = OrderedDict()
        for name, tensor in sd.items():
            if name.startswith("mtp.") or name.startswith("v."):
                continue
            if name.endswith(".ssm_dt.bias"):
                name = name[:-5]
            if name.endswith(".ssm_conv1d.weight"):
                tensor = tensor.unsqueeze(1)
            new_sd[name] = tensor
        if tie_output_to_embeddings:
            tied_map = _ensure_tied_output_weight(new_sd, tied_map)
        return new_sd, quant_map, tied_map

    return preprocess_sd


def _normalize_generated_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _clean_answer_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^(?:\s*</think>\s*)+", "", text, flags=re.IGNORECASE)
    text = text.replace("<think>", "\n").replace("</think>", "\n")
    text = re.sub(r"^\s*assistant\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.split(
        r"(?:<\|im_start\|>\s*(?:assistant|user|system|tool)\b|<tool_call>\s*|<tool_response>\s*|<tools>\s*|\n\s*(?:assistant|user|system|tool)\s*:)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) == 0:
            continue
        if stripped.lower() == "code interpreter":
            break
        cleaned_lines.append(line)
    return _normalize_generated_text("\n".join(cleaned_lines))


def _split_generated_text(text: str) -> tuple[str, str]:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    think_chunks = [match.group(1) for match in re.finditer(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)]
    answer_text = re.sub(r"<think>.*?</think>", "\n", text, flags=re.DOTALL | re.IGNORECASE)
    if len(think_chunks) == 0:
        forced_open_match = re.search(r"</think>", text, flags=re.IGNORECASE)
        if forced_open_match is not None:
            forced_reasoning = text[:forced_open_match.start()]
            forced_reasoning = forced_reasoning.replace("<think>", "\n")
            forced_reasoning = _normalize_generated_text(forced_reasoning)
            if len(forced_reasoning) > 0:
                think_chunks.append(forced_reasoning)
            answer_text = text[forced_open_match.end():]
    if len(think_chunks) == 0:
        timeline_match = re.search(r"(?mi)^\(at\s+[0-9]+(?:\.[0-9]+)?\s+seconds?\s*:", text)
        if timeline_match is not None:
            leading_text = _normalize_generated_text(text[:timeline_match.start()])
            if leading_text.lower().startswith("thinking process"):
                think_chunks.append(leading_text)
                answer_text = text[timeline_match.start():]
    answer_text = re.sub(r"<think>.*$", "\n", answer_text, flags=re.DOTALL | re.IGNORECASE)
    return _normalize_generated_text("\n\n".join(chunk for chunk in think_chunks if chunk.strip())), _clean_answer_text(answer_text)


def _clean_generated_text(text: str) -> str:
    _thinking_text, answer_text = _split_generated_text(text)
    return answer_text


def _prompt_enhancer_thinking_enabled(model, thinking_enabled: bool | None = None) -> bool:
    if thinking_enabled is not None:
        return bool(thinking_enabled)
    return bool(getattr(model, "_prompt_enhancer_enable_thinking", QWEN35_PROMPT_ENABLE_THINKING))


def set_qwen35_prompt_enhancer_thinking(model, enabled: bool) -> None:
    model._prompt_enhancer_enable_thinking = bool(enabled)
    tokenizer = getattr(model, "_prompt_enhancer_tokenizer", None)
    if tokenizer is None:
        return
    base_suppress_token_ids = tuple(_collect_suppressed_token_ids(tokenizer))
    model._prompt_enhancer_base_suppress_token_ids = base_suppress_token_ids
    model._prompt_enhancer_suppress_token_ids = [] if model._prompt_enhancer_enable_thinking else list(base_suppress_token_ids)
    model._prompt_enhancer_suppress_logits_bias = None


def _resolve_prompt_runtime_extra_tokens(model, thinking_enabled: bool | None = None) -> int:
    if not _prompt_enhancer_thinking_enabled(model, thinking_enabled=thinking_enabled):
        return 0
    extra_tokens = getattr(model, "_prompt_enhancer_thinking_extra_tokens", QWEN35_PROMPT_THINKING_EXTRA_TOKENS)
    try:
        return max(0, int(extra_tokens))
    except Exception:
        return 0


def _print_thinking_process(message_index: int, total_messages: int, thinking_text: str) -> None:
    if len(thinking_text) == 0:
        return
    print(f"[Prompt Enhancer][Thinking {message_index + 1}/{total_messages}]")
    try:
        print(thinking_text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = thinking_text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe_text + "\n")
        sys.stdout.flush()


class _PresencePenaltyState:
    def __init__(self, presence_penalty: float | None):
        if presence_penalty is None or float(presence_penalty) <= 0:
            self.penalty = None
        else:
            self.penalty = float(presence_penalty)
        self._seen_token_ids = set()
        self._bias_cache = {}

    def enabled(self) -> bool:
        return self.penalty is not None

    def update(self, token_id: int) -> None:
        if self.penalty is None:
            return
        token_id = int(token_id)
        if token_id < 0 or token_id in self._seen_token_ids:
            return
        self._seen_token_ids.add(token_id)
        for (vocab_size, _, _), bias in self._bias_cache.items():
            if token_id < vocab_size:
                bias[token_id] = bias.new_tensor(-self.penalty)

    def apply_(self, logits: torch.Tensor) -> torch.Tensor:
        if self.penalty is None or len(self._seen_token_ids) == 0:
            return logits
        cache_key = (logits.shape[-1], logits.device, logits.dtype)
        bias = self._bias_cache.get(cache_key)
        if bias is None:
            bias = logits.new_zeros(logits.shape[-1])
            valid_token_ids = [token_id for token_id in self._seen_token_ids if token_id < logits.shape[-1]]
            if len(valid_token_ids) > 0:
                bias.index_fill_(0, torch.tensor(valid_token_ids, device=logits.device, dtype=torch.long), bias.new_tensor(-self.penalty))
            self._bias_cache[cache_key] = bias
        logits.add_(bias)
        return logits


class _ThinkingBudgetState:
    def __init__(self, close_think_token_id: int | None, max_thinking_tokens: int | None):
        try:
            close_think_token_id = int(close_think_token_id)
        except (TypeError, ValueError):
            close_think_token_id = -1
        try:
            max_thinking_tokens = int(max_thinking_tokens)
        except (TypeError, ValueError):
            max_thinking_tokens = 0
        self.close_think_token_id = close_think_token_id
        self.max_thinking_tokens = max_thinking_tokens
        self.in_thinking = close_think_token_id >= 0 and max_thinking_tokens > 0
        self.generated_thinking_tokens = 0

    def enabled(self) -> bool:
        return self.in_thinking

    def update(self, token_id: int) -> None:
        if not self.in_thinking:
            return
        token_id = int(token_id)
        if token_id == self.close_think_token_id:
            self.in_thinking = False
            return
        self.generated_thinking_tokens += 1

    def apply_(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.in_thinking or self.generated_thinking_tokens < self.max_thinking_tokens:
            return logits
        logits.fill_(float("-inf"))
        logits[..., self.close_think_token_id] = 0
        return logits




def _build_chat_prompt(tokenizer, message, enable_thinking: bool = False):
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )
    return text.rstrip() + "\n"


def _resolve_prompt_presence_penalty(model) -> float | None:
    if not bool(getattr(model, "_prompt_enhancer_enable_presence_penalty", QWEN35_PROMPT_ENABLE_PRESENCE_PENALTY)):
        return None
    presence_penalty = getattr(model, "_prompt_enhancer_presence_penalty", QWEN35_PROMPT_PRESENCE_PENALTY)
    if presence_penalty is None:
        return None
    presence_penalty = float(presence_penalty)
    return presence_penalty if presence_penalty > 0 else None


def _build_presence_penalty_logits_processor(presence_penalty: float | None):
    state = _PresencePenaltyState(presence_penalty)
    if not state.enabled():
        return None, None

    def logits_processor(_input_ids, logits):
        state.apply_(logits)
        return logits

    def update_state(token_id: int):
        state.update(token_id)

    return logits_processor, update_state


def _build_prompt_logits_processor(model, thinking_enabled: bool | None = None):
    processors = []
    update_callbacks = []

    presence_processor, presence_update_state = _build_presence_penalty_logits_processor(_resolve_prompt_presence_penalty(model))
    if presence_processor is not None:
        processors.append(presence_processor)
    if presence_update_state is not None:
        update_callbacks.append(presence_update_state)

    if _prompt_enhancer_thinking_enabled(model, thinking_enabled=thinking_enabled):
        thinking_state = _ThinkingBudgetState(
            getattr(model, "_prompt_enhancer_close_think_token_id", None),
            getattr(model, "_prompt_enhancer_thinking_max_tokens", QWEN35_PROMPT_THINKING_MAX_TOKENS),
        )
        if thinking_state.enabled():
            def thinking_logits_processor(_input_ids, logits):
                return thinking_state.apply_(logits)

            processors.append(thinking_logits_processor)
            update_callbacks.append(thinking_state.update)

    if not processors:
        return None, None

    if len(processors) == 1:
        logits_processor = processors[0]
    else:
        def logits_processor(input_ids, logits):
            for processor in processors:
                logits = processor(input_ids, logits)
            return logits

    if len(update_callbacks) == 1:
        update_state = update_callbacks[0]
    else:
        def update_state(token_id: int):
            for callback in update_callbacks:
                callback(token_id)

    return logits_processor, update_state


def _build_suppressed_token_logits_bias(model, thinking_enabled: bool | None = None):
    if _prompt_enhancer_thinking_enabled(model, thinking_enabled=thinking_enabled):
        return None
    suppress_token_ids = tuple(int(token_id) for token_id in tuple(getattr(model, "_prompt_enhancer_base_suppress_token_ids", ()) or ()) if int(token_id) >= 0)
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if vocab_size <= 0 or len(suppress_token_ids) == 0:
        return None
    valid_token_ids = tuple(token_id for token_id in suppress_token_ids if token_id < vocab_size)
    if len(valid_token_ids) == 0:
        return None
    cache = getattr(model, "_prompt_enhancer_suppress_logits_bias_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        model._prompt_enhancer_suppress_logits_bias_cache = cache
    cache_key = (vocab_size, valid_token_ids)
    cached_bias = cache.get(cache_key)
    if torch.is_tensor(cached_bias):
        return cached_bias
    bias = torch.zeros(vocab_size, dtype=torch.float32)
    bias[torch.tensor(valid_token_ids, dtype=torch.long)] = float(QWEN35_PROMPT_SUPPRESS_LOGITS_BIAS)
    cache[cache_key] = bias
    return bias


def _normalize_vllm_sampling(do_sample, temperature, top_p, top_k):
    if not do_sample:
        return 1.0, None, 1
    if temperature is None or float(temperature) <= 0:
        return 1.0, None, 1
    normalized_top_p = top_p if top_p is not None and 0.0 < float(top_p) < 1.0 else None
    normalized_top_k = int(top_k) if top_k is not None and int(top_k) > 0 else None
    return float(temperature), normalized_top_p, normalized_top_k


def _resolve_prompt_top_k(model, top_k):
    if top_k is not None:
        resolved_top_k = int(top_k)
        return resolved_top_k if resolved_top_k > 0 else None
    default_top_k = getattr(model, "_prompt_enhancer_default_top_k", QWEN35_PROMPT_DEFAULT_TOP_K)
    if default_top_k is None:
        return None
    default_top_k = int(default_top_k)
    return default_top_k if default_top_k > 0 else None


def _resolve_prompt_min_p(model):
    min_p = getattr(model, "_prompt_enhancer_default_min_p", None)
    if min_p is None:
        return None
    min_p = float(min_p)
    return min_p if min_p > 0.0 else None


def _resolve_prompt_enhancer_engine(backend: str, requested_lm_engine: str, runtime_model_path: str | None):
    del backend
    if runtime_model_path is None:
        return "legacy", "vllm runtime path is not configured", False, False
    if not _env_enabled(QWEN35_TEXT_VLLM_SWITCH_ENV, default=True):
        return "legacy", f"disabled by {QWEN35_TEXT_VLLM_SWITCH_ENV}", False, False
    requested_lm_engine = str(requested_lm_engine or "").strip().lower()
    requested_label = requested_lm_engine or "auto"
    resolved_engine = resolve_lm_decoder_engine(requested_lm_engine, ["cg", "vllm"])
    enable_cudagraph = _env_enabled(QWEN35_TEXT_VLLM_CUDAGRAPH_ENV, default=True)

    if resolved_engine == "legacy":
        detail = f"lm_decoder_engine={requested_label}"
        if requested_lm_engine in ("", "cg", "vllm"):
            detail = f"lm_decoder_engine={requested_label} -> legacy"
        return "legacy", detail, False, False

    if resolved_engine == "cg":
        detail = "cuda graph only" if enable_cudagraph else f"eager only; disabled by {QWEN35_TEXT_VLLM_CUDAGRAPH_ENV}"
        if requested_lm_engine != "cg":
            detail = f"lm_decoder_engine={requested_label} -> cg" #; {detail}"
        return "cg", detail, enable_cudagraph, False

    detail = "cuda graph + vllm kernels" if enable_cudagraph else f"eager + vllm kernels; disabled by {QWEN35_TEXT_VLLM_CUDAGRAPH_ENV}"
    if requested_lm_engine == "":
        detail = f"lm_decoder_engine=auto -> vllm" #; {detail}"
    return "vllm", detail, enable_cudagraph, True


def _use_vllm_prompt_enhancer(model) -> bool:
    if not bool(getattr(model, "_prompt_enhancer_use_vllm", False)):
        return False
    if not _env_enabled(QWEN35_TEXT_VLLM_SWITCH_ENV, default=True):
        return False
    if not torch.cuda.is_available():
        return False
    return True


def _use_legacy_cuda_runner_prompt_enhancer(model) -> bool:
    return bool(getattr(model, "_prompt_enhancer_use_legacy_cuda_runner", False)) and torch.cuda.is_available()


def _get_or_create_vllm_engine(model, usage_mode: str | None = None):
    register_qwen35_config()
    engine = getattr(model, "_prompt_enhancer_vllm_engine", None)
    active_mode = getattr(model, "_prompt_enhancer_vllm_mode", None)
    if engine is not None and usage_mode is not None and active_mode not in (None, usage_mode):
        try:
            engine.close()
        except Exception:
            pass
        engine = None
        model._prompt_enhancer_vllm_engine = None
    if engine is not None:
        if usage_mode is not None:
            model._prompt_enhancer_vllm_mode = usage_mode
        return engine

    runtime_model_path = getattr(model, "_prompt_enhancer_vllm_model_path", None)
    tokenizer = getattr(model, "_prompt_enhancer_tokenizer", None)
    if not runtime_model_path:
        raise RuntimeError("Qwen3.5 prompt enhancer vLLM runtime path is not configured.")
    if tokenizer is None:
        raise RuntimeError("Qwen3.5 prompt enhancer tokenizer is not configured.")
    enable_cudagraph = bool(getattr(model, "_prompt_enhancer_enable_cudagraph", False))

    engine = NanoVllmTextEngine(
        model=model,
        model_path=runtime_model_path,
        tokenizer=tokenizer,
        enforce_eager=not enable_cudagraph,
    )
    model._prompt_enhancer_vllm_engine = engine
    model._prompt_enhancer_vllm_mode = usage_mode
    return engine


def _reset_vllm_sequence_state(model):
    for module in model.modules():
        reset_sequence_state = getattr(module, "reset_sequence_state", None)
        if callable(reset_sequence_state):
            reset_sequence_state()
            continue
        if hasattr(module, "conv_state"):
            module.conv_state = None
        if hasattr(module, "recurrent_state"):
            module.recurrent_state = None


def _generate_messages_vllm(
    self,
    messages,
    max_new_tokens,
    do_sample=True,
    temperature=None,
    top_p=None,
    top_k=None,
    seed=None,
    thinking_enabled: bool | None = None,
):
    reset_context()
    tokenizer = self._prompt_enhancer_tokenizer
    if len(messages) == 0:
        return []

    engine = _get_or_create_vllm_engine(self, usage_mode="text")
    outputs = []
    thinking_enabled = _prompt_enhancer_thinking_enabled(self, thinking_enabled=thinking_enabled)
    runtime_extra_tokens = _resolve_prompt_runtime_extra_tokens(self, thinking_enabled=thinking_enabled)
    progress_desc = (
        "Qwen3.5 prompt enhancement (legacy)"
        if _use_legacy_cuda_runner_prompt_enhancer(self)
        else f"Qwen3.5 prompt enhancement ({getattr(self, '_prompt_enhancer_engine_name', 'vllm')})"
    )
    for idx, message in enumerate(tqdm(messages, total=len(messages), desc=progress_desc, dynamic_ncols=True, leave=False)):
        prompt = _build_chat_prompt(tokenizer, message, enable_thinking=thinking_enabled)
        try:
            prompt_len = len(tokenizer.encode(prompt))
        except Exception:
            prompt_len = 0
        generation_max_tokens = int(max_new_tokens) + runtime_extra_tokens
        engine.reserve_runtime(prompt_len=prompt_len, max_tokens=generation_max_tokens, cfg_scale=1.0)
        engine._ensure_llm()
        if engine._llm is None:
            raise RuntimeError("Qwen3.5 prompt enhancer vLLM runtime is not available.")
        temp, normalized_top_p, normalized_top_k = _normalize_vllm_sampling(
            do_sample=bool(do_sample),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        logits_bias = _build_suppressed_token_logits_bias(self, thinking_enabled=thinking_enabled)
        sample_seed = None if seed is None else int(seed) + idx
        logits_processor, logits_processor_update_state = _build_prompt_logits_processor(self, thinking_enabled=thinking_enabled)
        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=generation_max_tokens,
            cfg_scale=1.0,
            top_k=normalized_top_k,
            top_p=normalized_top_p,
            min_p=_resolve_prompt_min_p(self),
            ignore_eos=False,
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            logits_bias=logits_bias,
            seed=sample_seed,
        )

        try:
            batch_outputs = engine._llm.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                use_tqdm=True,
                unconditional_prompts=None,
            )
            engine._last_failure_reason = ""
        except Exception as exc:
            engine._last_failure_reason = str(exc)
            raise
        finally:
            reset_context()

        text, token_ids = engine._extract_text_and_tokens(batch_outputs[0] if batch_outputs else None)
        raw_text = text
        if token_ids:
            try:
                raw_text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except Exception:
                raw_text = text
        thinking_text, answer_text = _split_generated_text(raw_text)
        if thinking_enabled:
            _print_thinking_process(idx, len(messages), thinking_text)
        outputs.append(answer_text)
    return outputs


def _generate_messages(
    self,
    messages,
    max_new_tokens,
    do_sample=True,
    temperature=None,
    top_p=None,
    top_k=None,
    seed=None,
    thinking_enabled: bool | None = None,
):
    top_k = _resolve_prompt_top_k(self, top_k)
    if _use_vllm_prompt_enhancer(self) or _use_legacy_cuda_runner_prompt_enhancer(self):
        return _generate_messages_vllm(
            self,
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            thinking_enabled=thinking_enabled,
        )
    raise RuntimeError("Qwen3.5 prompt enhancer text runtime is not configured with an available decode engine.")


def _unload_prompt_enhancer_text_runtime(self):
    engine = getattr(self, "_prompt_enhancer_vllm_engine", None)
    if engine is not None:
        try:
            engine.close()
        finally:
            self._prompt_enhancer_vllm_engine = None
            self._prompt_enhancer_vllm_mode = None
    try:
        clear_qwen35_runtime_caches()
    except Exception:
        pass
    reset_context()
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _load_local_text_model(
    model_path: str,
    config_path: str,
    preprocess_sd=None,
    default_dtype: torch.dtype = torch.float16,
    safe_legacy_mode: bool = False,
    materialize_source_tensors: bool = True,
):
    config = _load_text_config(config_path)
    config._prompt_enhancer_safe_legacy = bool(safe_legacy_mode)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)

    offload.load_model_data(
        model,
        model_path,
        preprocess_sd=preprocess_sd,
        writable_tensors=False,
        default_dtype=default_dtype,
    )
    if materialize_source_tensors:
        materialize_module_source_tensors(model)
    return model


def _resolve_legacy_text_execution_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 legacy prompt enhancement now requires CUDA.")
    return torch.device("cuda", torch.cuda.current_device())


def _configure_qwen35_gguf_text_model(
    model,
    model_dtype: torch.dtype,
    *,
    v_head_reordered: bool = False,
    ssm_param_reordered: bool = False,
    interleave_ssm_ab: bool = False,
):
    for module in model.modules():
        if getattr(module, "layer_type", None) == "linear_attention" and hasattr(module, "_gguf_interleave_ssm_ab"):
            module._gguf_interleave_ssm_ab = bool(interleave_ssm_ab)
        if getattr(module, "layer_type", None) == "linear_attention" and hasattr(module, "_gguf_v_head_reordered"):
            module._gguf_v_head_reordered = bool(v_head_reordered)
        if getattr(module, "layer_type", None) == "linear_attention" and hasattr(module, "_gguf_ssm_param_reordered"):
            module._gguf_ssm_param_reordered = bool(ssm_param_reordered)
    model.config.dtype = model_dtype
    model.config.torch_dtype = model_dtype


def _concat_linear_weights(weights):
    if len(weights) == 0:
        raise ValueError("At least one linear weight is required to build a fused projection.")
    first = weights[0]
    if isinstance(first, GGUFWeightTensor):
        tensor_type = getattr(first, "_tensor_type", None)
        dtype = getattr(first, "dtype", torch.float16)
        device = getattr(first, "device", torch.device("cpu"))
        if all(
            isinstance(weight, GGUFWeightTensor)
            and getattr(weight, "_tensor_type", None) == tensor_type
            and int(weight.shape[1]) == int(first.shape[1])
            for weight in weights
        ):
            raw = torch.cat([weight._data for weight in weights], dim=0)
            out_features = sum(int(weight.shape[0]) for weight in weights)
            in_features = int(first.shape[1])
            return GGUFWeightTensor.create(
                raw_tensor=raw,
                size=(out_features, in_features),
                stride=(in_features, 1),
                dtype=dtype,
                device=device,
                requires_grad=False,
                tensor_type=tensor_type,
                tensor_shape=(out_features, in_features),
            )
    return torch.cat([weight.detach() for weight in weights], dim=0)


def _concat_linear_biases(biases):
    if len(biases) == 0 or any(bias is None for bias in biases):
        return None
    return torch.cat([bias.detach() for bias in biases], dim=0)


def _build_fused_column_linear(modules):
    modules = [module for module in modules if module is not None]
    if len(modules) <= 1:
        return modules[0] if modules else None
    template = modules[0]
    out_features = sum(int(module.weight.shape[0]) for module in modules)
    fused_kwargs = {}
    for attr_name, kwarg_name in (
        ("weight_qtype", "weights"),
        ("activation_qtype", "activations"),
        ("optimizer", "optimizer"),
        ("quantize_input", "quantize_input"),
    ):
        if hasattr(template, attr_name):
            fused_kwargs[kwarg_name] = getattr(template, attr_name)
    if hasattr(template, "_router_default_dtype"):
        fused_kwargs["dtype"] = template._router_default_dtype
    if hasattr(template, "weight") and torch.is_tensor(template.weight):
        fused_kwargs.setdefault("device", template.weight.device)
        fused_kwargs.setdefault("dtype", template.weight.dtype)
    try:
        fused = template.__class__(int(template.in_features), out_features, bias=all(module.bias is not None for module in modules), **fused_kwargs)
    except TypeError:
        fused = template.__class__(int(template.in_features), out_features, bias=all(module.bias is not None for module in modules))
    fused_weight = _concat_linear_weights([module.weight for module in modules])
    fused._parameters["weight"] = fused_weight
    fused._parameters["bias"] = _concat_linear_biases([module.bias for module in modules])
    if hasattr(fused, "qweight"):
        try:
            fused.qweight = fused_weight
        except Exception:
            pass
    for attr_name in ("weight_qtype", "activation_qtype", "optimizer", "input_scale", "output_scale", "_router_default_dtype", "_router_forward_impl"):
        if hasattr(template, attr_name):
            setattr(fused, attr_name, getattr(template, attr_name))
    if hasattr(template, "_gguf_default_dtype"):
        fused._gguf_default_dtype = template._gguf_default_dtype
    return fused


def _apply_qwen35_gguf_projection_fusions(model) -> None:
    for block in getattr(model, "blk", ()):
        if getattr(block, "layer_type", None) != "linear_attention":
            continue
        # Real prompt-enhancer benchmarks only kept this fusion: it removes two extra
        # GGUF matmul launches from the hottest decode block without changing the decode path.
        if getattr(block, "attn_gate_ab", None) is None and all(
            getattr(block, name, None) is not None for name in ("attn_gate", "ssm_alpha", "ssm_beta")
        ):
            block.attn_gate_ab = _build_fused_column_linear([block.attn_gate, block.ssm_alpha, block.ssm_beta])
            block.attn_gate = None
            block.ssm_alpha = None
            block.ssm_beta = None

def load_qwen35_text_prompt_enhancer(
    model_path: str | None = None,
    assets_dir: str | None = None,
    default_dtype: torch.dtype = torch.float16,
    backend: str = enhancer_quantization_QUANTO_INT8,
    attn_implementation: str = "sdpa",
    requested_lm_engine: str = "",
    variant: str | None = None,
):
    del attn_implementation
    if assets_dir is None:
        raise ValueError("A local Qwen3.5 assets directory is required.")

    assets_dir = _resolve_qwen35_assets_dir(assets_dir, variant=variant)
    spec = get_qwen35_variant_spec(variant)
    text_assets_dir = assets_dir
    tokenizer_json = _resolve_qwen35_asset_file(assets_dir, "tokenizer.json", variant=variant, error_if_none=False) or os.path.join(assets_dir, "tokenizer.json")
    tokenizer_config = _resolve_qwen35_asset_file(assets_dir, "tokenizer_config.json", variant=variant, error_if_none=False) or os.path.join(assets_dir, "tokenizer_config.json")
    text_config_path = _resolve_qwen35_asset_file(text_assets_dir, "config.json", error_if_none=False) or os.path.join(text_assets_dir, "config.json")
    for required_file in (tokenizer_json, tokenizer_config, text_config_path):
        if not os.path.isfile(required_file):
            raise FileNotFoundError(f"Missing Qwen3.5 text prompt enhancer asset: {required_file}")

    tokenizer = _load_qwen35_tokenizer(assets_dir)
    chat_template_path = _resolve_qwen35_asset_file(text_assets_dir, "chat_template.jinja", error_if_none=False) or os.path.join(text_assets_dir, "chat_template.jinja")
    if os.path.isfile(chat_template_path):
        with open(chat_template_path, "r", encoding="utf-8") as reader:
            tokenizer.chat_template = reader.read()

    if backend == enhancer_quantization_GGUF:
        model_path = _resolve_gguf_model_path(model_path, assets_dir, variant=variant)
        gguf_v_head_reordered, gguf_ssm_param_reordered, gguf_interleave_ssm_ab = _resolve_gguf_linear_attention_layout_from_filename(model_path)
        preprocess_sd = _build_qwen35_gguf_preprocess_sd(
            tie_output_to_embeddings=bool(spec.get("tie_word_embeddings", False)),
        )
        runtime_model_path = text_assets_dir
    else:
        gguf_v_head_reordered = False
        gguf_ssm_param_reordered = False
        gguf_interleave_ssm_ab = False
        if model_path is None:
            model_path = get_qwen35_text_quanto_int8_path(assets_dir, variant=variant)
        preprocess_sd = None
        runtime_model_path = text_assets_dir

    engine_name, _engine_detail, enable_cudagraph, allow_vllm_kernels = _resolve_prompt_enhancer_engine(
        backend=backend,
        requested_lm_engine=requested_lm_engine,
        runtime_model_path=runtime_model_path,
    )
    safe_legacy_mode = not allow_vllm_kernels

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Qwen3.5 text checkpoint not found: {model_path}")
    print(f"[Prompt Enhancer][{spec['display_name']}][{backend}] Loading text checkpoint: {model_path}")

    model = _load_local_text_model(
        model_path,
        text_config_path,
        preprocess_sd=preprocess_sd,
        default_dtype=default_dtype,
        safe_legacy_mode=safe_legacy_mode,
        materialize_source_tensors=backend != enhancer_quantization_GGUF,
    )
    if backend == enhancer_quantization_GGUF:
        _configure_qwen35_gguf_text_model(
            model,
            default_dtype,
            v_head_reordered=gguf_v_head_reordered,
            ssm_param_reordered=gguf_ssm_param_reordered,
            interleave_ssm_ab=gguf_interleave_ssm_ab,
        )
        _apply_qwen35_gguf_projection_fusions(model)

    model._prompt_enhancer_tokenizer = tokenizer
    model._prompt_enhancer_gguf_v_head_reordered = bool(gguf_v_head_reordered)
    model._prompt_enhancer_gguf_ssm_param_reordered = bool(gguf_ssm_param_reordered)
    model._prompt_enhancer_thinking_extra_tokens = QWEN35_PROMPT_THINKING_EXTRA_TOKENS
    model._prompt_enhancer_thinking_max_tokens = QWEN35_PROMPT_THINKING_MAX_TOKENS
    model._prompt_enhancer_close_think_token_id = tokenizer.convert_tokens_to_ids("</think>")
    model._prompt_enhancer_stop_token_ids = _collect_stop_token_ids(tokenizer, model.config)
    set_qwen35_prompt_enhancer_thinking(model, QWEN35_PROMPT_ENABLE_THINKING)
    model._prompt_enhancer_suppress_logits_bias = None
    model._prompt_enhancer_suppress_logits_bias_cache = {}
    model._prompt_enhancer_default_top_k = QWEN35_PROMPT_DEFAULT_TOP_K
    model._prompt_enhancer_default_min_p = QWEN35_PROMPT_DEFAULT_MIN_P_GGUF if backend == enhancer_quantization_GGUF else None
    model._prompt_enhancer_enable_presence_penalty = backend != enhancer_quantization_GGUF and QWEN35_PROMPT_ENABLE_PRESENCE_PENALTY
    model._prompt_enhancer_presence_penalty = QWEN35_PROMPT_PRESENCE_PENALTY
    model._prompt_enhancer_min_new_tokens = (
        QWEN35_PROMPT_MIN_NEW_TOKENS
        if backend == enhancer_quantization_GGUF and _env_enabled(QWEN35_GGUF_LLAMACPP_ENV, default=True)
        else 0
    )
    model._prompt_enhancer_use_vllm = False
    model._prompt_enhancer_use_legacy_cuda_runner = False
    model._prompt_enhancer_engine_name = engine_name
    model._prompt_enhancer_enable_cudagraph = bool(enable_cudagraph and engine_name in ("cg", "vllm"))
    model._prompt_enhancer_allow_vllm_kernels = bool(allow_vllm_kernels)
    model._prompt_enhancer_vllm_model_path = runtime_model_path
    model._prompt_enhancer_vllm_engine = None
    model._prompt_enhancer_vllm_mode = None
    model._prompt_enhancer_safe_legacy = safe_legacy_mode
    model._prompt_enhancer_use_vllm = engine_name in ("cg", "vllm")
    model._prompt_enhancer_use_legacy_cuda_runner = engine_name == "legacy"
    if model._prompt_enhancer_use_vllm or model._prompt_enhancer_use_legacy_cuda_runner:
        model._budget = 0
    print(f"[Prompt Enhancer][{spec['display_name']}] Text generation engine: {engine_name}")
    model.generate_messages = types.MethodType(_generate_messages, model)
    model.unload = types.MethodType(_unload_prompt_enhancer_text_runtime, model)
    model._offload_hooks = ["forward"]
    model.eval()
    return model


load_qwen35_prompt_enhancer = load_qwen35_text_prompt_enhancer


__all__ = [
    "get_qwen35_text_quanto_int8_path",
    "load_qwen35_prompt_enhancer",
    "load_qwen35_text_prompt_enhancer",
    "set_qwen35_prompt_enhancer_thinking",
]
