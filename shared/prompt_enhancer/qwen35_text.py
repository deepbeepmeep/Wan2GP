from __future__ import annotations

import gc
import importlib
import json
import os
import re
import types
from collections import OrderedDict

import torch
from mmgp import offload
from tqdm.auto import tqdm
from shared.utils import files_locator as fl
from shared.llm_engines.nanovllm import SamplingParams
from shared.llm_engines.nanovllm.models.qwen3_5 import Qwen3_5DynamicCache, Qwen3_5ForCausalLM, clear_qwen35_runtime_caches
from shared.llm_engines.nanovllm.utils.context import reset_context
from shared.llm_engines.nanovllm.vllm_support import (
    NanoVllmTextEngine,
    probe_vllm_runtime,
    resolve_lm_decoder_engine,
)
from shared.qtypes.gguf import materialize_module_source_tensors

from .qwen3_5 import register_qwen35_config
from .qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from .qwen35_vl import (
    enhancer_quantization_GGUF,
    enhancer_quantization_QUANTO_INT8,
    _collect_suppressed_token_ids,
    _collect_stop_token_ids,
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
QWEN35_PROMPT_ENABLE_PRESENCE_PENALTY = True
QWEN35_PROMPT_PRESENCE_PENALTY = 1.5
QWEN35_PROMPT_SUPPRESS_LOGITS_BIAS = -1e4


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _resolve_gguf_model_path(model_path: str | None, assets_dir: str, variant: str | None = None) -> str:
    if model_path is not None:
        resolved = fl.locate_file(model_path, error_if_none=False) or model_path
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Missing Qwen3.5 GGUF checkpoint: {resolved}")
        return resolved

    assets_dir = _resolve_qwen35_assets_dir(assets_dir, variant=variant)
    exact_path = get_qwen35_text_gguf_path(assets_dir, variant=variant)
    if os.path.isfile(exact_path):
        return exact_path
    raise FileNotFoundError(f"Missing expected Qwen3.5 GGUF checkpoint: {exact_path}")


def _configure_qwen35_text_safe_legacy_kernels(enabled: bool) -> None:
    importlib.import_module("shared.llm_engines.nanovllm.models.qwen3_5").configure_qwen35_safe_legacy_kernels(bool(enabled))


def get_qwen35_text_assets_dir(assets_dir: str, variant: str | None = None) -> str:
    return _resolve_qwen35_assets_dir(assets_dir, variant=variant)


def get_qwen35_text_quanto_int8_path(assets_dir: str, variant: str | None = None) -> str:
    assets_dir = _resolve_qwen35_assets_dir(assets_dir, variant=variant)
    filename = get_qwen35_variant_spec(variant)["text_int8_filename"]
    return _resolve_qwen35_asset_file(assets_dir, filename, variant=variant, error_if_none=False) or os.path.join(assets_dir, filename)

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


def _interleave_qwen35_ssm_heads(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0 or tensor.shape[0] % 2 != 0:
        return tensor
    half = tensor.shape[0] // 2
    if tensor.ndim == 1:
        output = torch.empty_like(tensor)
        output[0::2] = tensor[:half]
        output[1::2] = tensor[half:]
        return output
    first_half, second_half = tensor[:half], tensor[half:]
    return torch.stack((first_half, second_half), dim=1).flatten(0, 1)


def _build_qwen35_gguf_preprocess_sd(tie_output_to_embeddings: bool = False):
    def preprocess_sd(sd, quant_map=None, tied_map=None):
        new_sd = OrderedDict()
        for name, tensor in sd.items():
            if name.startswith("mtp.") or name.startswith("v."):
                continue
            if name.endswith(".ssm_dt.bias"):
                name = name[:-5]
            if name.endswith(".ssm_dt") or name.endswith(".ssm_a"):
                tensor = _interleave_qwen35_ssm_heads(tensor)
            if name.endswith(".ssm_conv1d.weight"):
                tensor = tensor.unsqueeze(1)
            new_sd[name] = tensor
        if tie_output_to_embeddings:
            tied_map = _ensure_tied_output_weight(new_sd, tied_map)
        return new_sd, quant_map, tied_map

    return preprocess_sd


def _clean_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    logits = logits.clone()

    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, int(top_k), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > float(top_p)
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return logits


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
                bias[token_id] = -self.penalty

    def apply_(self, logits: torch.Tensor) -> torch.Tensor:
        if self.penalty is None or len(self._seen_token_ids) == 0:
            return logits
        cache_key = (logits.shape[-1], logits.device, logits.dtype)
        bias = self._bias_cache.get(cache_key)
        if bias is None:
            bias = logits.new_zeros(logits.shape[-1])
            valid_token_ids = [token_id for token_id in self._seen_token_ids if token_id < logits.shape[-1]]
            if len(valid_token_ids) > 0:
                bias[torch.tensor(valid_token_ids, device=logits.device, dtype=torch.long)] = -self.penalty
            self._bias_cache[cache_key] = bias
        logits.add_(bias)
        return logits


def _prepare_sampling_logits(logits: torch.Tensor, suppress_token_ids: set[int] | None = None, presence_penalty_state: _PresencePenaltyState | None = None) -> torch.Tensor:
    if (presence_penalty_state is None or not presence_penalty_state.enabled() or len(presence_penalty_state._seen_token_ids) == 0) and not suppress_token_ids:
        return logits
    logits = logits.clone()
    if presence_penalty_state is not None:
        presence_penalty_state.apply_(logits)
    if suppress_token_ids:
        blocked = [token_id for token_id in suppress_token_ids if 0 <= token_id < logits.shape[-1]]
        if blocked:
            logits[..., blocked] = float("-inf")
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    generator: torch.Generator | None,
    suppress_token_ids: set[int] | None = None,
    presence_penalty_state: _PresencePenaltyState | None = None,
) -> torch.Tensor:
    logits = _prepare_sampling_logits(logits, suppress_token_ids=suppress_token_ids, presence_penalty_state=presence_penalty_state)
    if not do_sample or temperature is None or float(temperature) <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / float(temperature)
    logits = _apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _last_step_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        return logits[:, -1, :]
    return logits


def _make_sampling_generator(device: torch.device, seed_value: int | None) -> torch.Generator | None:
    if seed_value is None:
        return None
    if device.type == "cuda":
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator()
    generator.manual_seed(int(seed_value))
    return generator


def _build_chat_prompt(tokenizer, message):
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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


def _build_prompt_presence_penalty_state(model) -> _PresencePenaltyState | None:
    state = _PresencePenaltyState(_resolve_prompt_presence_penalty(model))
    return state if state.enabled() else None


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


def _build_suppressed_token_logits_bias(model):
    cached_bias = getattr(model, "_prompt_enhancer_suppress_logits_bias", None)
    if torch.is_tensor(cached_bias):
        return cached_bias
    suppress_token_ids = list(getattr(model, "_prompt_enhancer_suppress_token_ids", []) or [])
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if vocab_size <= 0 or len(suppress_token_ids) == 0:
        return None
    valid_token_ids = [int(token_id) for token_id in suppress_token_ids if 0 <= int(token_id) < vocab_size]
    if len(valid_token_ids) == 0:
        return None
    bias = torch.zeros(vocab_size, dtype=torch.float32)
    bias[torch.tensor(valid_token_ids, dtype=torch.long)] = float(QWEN35_PROMPT_SUPPRESS_LOGITS_BIAS)
    model._prompt_enhancer_suppress_logits_bias = bias
    return bias


def _generate_messages_legacy(
    self,
    messages,
    max_new_tokens,
    do_sample=True,
    temperature=None,
    top_p=None,
    top_k=None,
    seed=None,
):
    reset_context()
    outputs = []
    tokenizer = self._prompt_enhancer_tokenizer
    stop_token_ids = getattr(self, "_prompt_enhancer_stop_token_ids", None)
    if stop_token_ids is None:
        stop_token_ids = _collect_stop_token_ids(tokenizer, self.config)
        self._prompt_enhancer_stop_token_ids = stop_token_ids
    stop_token_ids = set(stop_token_ids)
    suppress_token_ids = set(getattr(self, "_prompt_enhancer_suppress_token_ids", []) or [])
    presence_penalty_state = _build_prompt_presence_penalty_state(self)

    for idx, message in enumerate(tqdm(messages, total=len(messages), desc="Qwen3.5 prompt enhancement", dynamic_ncols=True, leave=False)):
        text = _build_chat_prompt(tokenizer, message)
        model_inputs = tokenizer(text, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        position_ids = torch.arange(input_ids.shape[1], device=self.device, dtype=torch.long).unsqueeze(0)
        past_key_values = Qwen3_5DynamicCache(self.config)
        generated_tokens = []
        with torch.inference_mode():
            hidden_states = self(input_ids, positions=position_ids, past_key_values=past_key_values)
            logits = _last_step_logits(self.compute_logits(hidden_states))
            generator = _make_sampling_generator(logits.device, int(seed) + idx if seed is not None else None)

            min_new_tokens = int(getattr(self, "_prompt_enhancer_min_new_tokens", 0) or 0)
            step_iter = tqdm(
                range(int(max_new_tokens)),
                total=int(max_new_tokens),
                desc="Qwen3.5 prompt enhancement tokens",
                unit="tok",
                dynamic_ncols=True,
                leave=False,
            )
            for step in step_iter:
                step_suppress = suppress_token_ids | stop_token_ids if step < min_new_tokens else suppress_token_ids
                next_token = _sample_next_token(
                    logits,
                    do_sample=bool(do_sample),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generator=generator,
                    suppress_token_ids=step_suppress,
                    presence_penalty_state=presence_penalty_state,
                )
                token_id = int(next_token.item())
                generated_tokens.append(token_id)
                if presence_penalty_state is not None:
                    presence_penalty_state.update(token_id)
                if token_id in stop_token_ids:
                    break
                next_position = torch.tensor([[input_ids.shape[1] + len(generated_tokens) - 1]], device=self.device, dtype=torch.long)
                hidden_states = self(next_token, positions=next_position, past_key_values=past_key_values)
                logits = _last_step_logits(self.compute_logits(hidden_states))

        outputs.append(
            _clean_generated_text(
                tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        )

    return outputs


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


def _format_vllm_probe_failure(probe_result) -> str:
    checks = probe_result.get("checks", {}) if isinstance(probe_result, dict) else {}
    reasons = []
    if isinstance(checks, dict):
        for check_name, check_data in checks.items():
            if not isinstance(check_data, dict) or bool(check_data.get("ok", False)):
                continue
            msg = str(check_data.get("message", "failed")).replace("\n", " ").strip()
            if len(msg) > 160:
                msg = msg[:160] + "..."
            reasons.append(f"{check_name}={msg}")
    return "; ".join(reasons) if len(reasons) > 0 else "runtime probe failed"


def _resolve_prompt_enhancer_engine(backend: str, requested_lm_engine: str, runtime_model_path: str | None):
    if runtime_model_path is None:
        return "legacy", "vllm runtime path is not configured"
    if not _env_enabled(QWEN35_TEXT_VLLM_SWITCH_ENV, default=True):
        return "legacy", f"disabled by {QWEN35_TEXT_VLLM_SWITCH_ENV}"
    requested_lm_engine = str(requested_lm_engine or "").strip().lower()
    resolved_engine = resolve_lm_decoder_engine(requested_lm_engine, ["vllm"])
    if resolved_engine != "vllm":
        requested_label = requested_lm_engine or "auto"
        if requested_lm_engine in ("", "vllm"):
            return "legacy", f"lm_decoder_engine={requested_label}; {_format_vllm_probe_failure(probe_vllm_runtime())}"
        return "legacy", f"lm_decoder_engine={requested_label}"
    probe_result = probe_vllm_runtime()
    if bool(probe_result.get("supported", False)):
        if _env_enabled(QWEN35_TEXT_VLLM_CUDAGRAPH_ENV, default=True):
            return "vllm", "cuda graph"
        return "vllm", "eager"
    return "legacy", _format_vllm_probe_failure(probe_result)


def _use_vllm_prompt_enhancer(model) -> bool:
    if not bool(getattr(model, "_prompt_enhancer_use_vllm", False)):
        return False
    if not _env_enabled(QWEN35_TEXT_VLLM_SWITCH_ENV, default=True):
        return False
    if not torch.cuda.is_available():
        return False
    probe_result = probe_vllm_runtime()
    return bool(probe_result.get("supported", False))


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
    force_eager = bool(getattr(model, "_prompt_enhancer_vllm_force_eager", False))
    enable_cudagraph = (not force_eager) and _env_enabled(QWEN35_TEXT_VLLM_CUDAGRAPH_ENV, default=True)

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
):
    reset_context()
    tokenizer = self._prompt_enhancer_tokenizer
    if len(messages) == 0:
        return []

    engine = _get_or_create_vllm_engine(self, usage_mode="text")
    outputs = []
    for idx, message in enumerate(tqdm(messages, total=len(messages), desc="Qwen3.5 prompt enhancement (vllm)", dynamic_ncols=True, leave=False)):
        prompt = _build_chat_prompt(tokenizer, message)
        try:
            prompt_len = len(tokenizer.encode(prompt))
        except Exception:
            prompt_len = 0
        engine.reserve_runtime(prompt_len=prompt_len, max_tokens=int(max_new_tokens), cfg_scale=1.0)
        engine._ensure_llm()
        if engine._llm is None:
            raise RuntimeError("Qwen3.5 prompt enhancer vLLM runtime is not available.")
        # The shared Qwen text model may have been used by captioning right before prompt
        # enhancement. Rebuild the vLLM runtime so CUDA graphs are always captured against
        # clean sequence/KV state for the current MMGP-loaded weights.
        engine.release_runtime_allocations()
        _reset_vllm_sequence_state(self)
        engine._llm.model_runner.ensure_runtime_ready()
        try:
            engine._llm.reset()
        except Exception:
            pass

        temp, normalized_top_p, normalized_top_k = _normalize_vllm_sampling(
            do_sample=bool(do_sample),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        sample_seed = None if seed is None else int(seed) + idx
        logits_bias = _build_suppressed_token_logits_bias(self)
        logits_processor, logits_processor_update_state = _build_presence_penalty_logits_processor(_resolve_prompt_presence_penalty(self))
        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=int(max_new_tokens),
            cfg_scale=1.0,
            top_k=normalized_top_k,
            top_p=normalized_top_p,
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

        text, _ = engine._extract_text_and_tokens(batch_outputs[0] if batch_outputs else None)
        outputs.append(_clean_generated_text(text))
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
):
    top_k = _resolve_prompt_top_k(self, top_k)
    if _use_vllm_prompt_enhancer(self):
        return _generate_messages_vllm(
            self,
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
    return _generate_messages_legacy(
        self,
        messages,
        max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )


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
):
    config = _load_text_config(config_path)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)

    offload.load_model_data(
        model,
        model_path,
        preprocess_sd=preprocess_sd,
        writable_tensors=False,
        default_dtype=default_dtype,
    )
    materialize_module_source_tensors(model)
    return model


def _configure_qwen35_gguf_text_model(model, model_dtype: torch.dtype):
    for module in model.modules():
        if getattr(module, "layer_type", None) == "linear_attention" and hasattr(module, "_gguf_interleave_ssm_ab"):
            module._gguf_interleave_ssm_ab = True
    model.config.dtype = model_dtype
    model.config.torch_dtype = model_dtype

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
    text_assets_dir = get_qwen35_text_assets_dir(assets_dir, variant=variant)
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
        preprocess_sd = _build_qwen35_gguf_preprocess_sd(
            tie_output_to_embeddings=bool(spec.get("tie_word_embeddings", False)),
        )
        runtime_model_path = text_assets_dir
    else:
        if model_path is None:
            model_path = get_qwen35_text_quanto_int8_path(assets_dir, variant=variant)
        preprocess_sd = None
        runtime_model_path = text_assets_dir

    engine_name, engine_detail = _resolve_prompt_enhancer_engine(
        backend=backend,
        requested_lm_engine=requested_lm_engine,
        runtime_model_path=runtime_model_path,
    )
    safe_legacy_mode = engine_name == "legacy"
    _configure_qwen35_text_safe_legacy_kernels(safe_legacy_mode)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Qwen3.5 text checkpoint not found: {model_path}")

    model = _load_local_text_model(
        model_path,
        text_config_path,
        preprocess_sd=preprocess_sd,
        default_dtype=default_dtype,
    )
    if backend == enhancer_quantization_GGUF:
        _configure_qwen35_gguf_text_model(model, default_dtype)

    model._prompt_enhancer_tokenizer = tokenizer
    model._prompt_enhancer_stop_token_ids = _collect_stop_token_ids(tokenizer, model.config)
    model._prompt_enhancer_suppress_token_ids = _collect_suppressed_token_ids(tokenizer)
    model._prompt_enhancer_suppress_logits_bias = None
    model._prompt_enhancer_default_top_k = QWEN35_PROMPT_DEFAULT_TOP_K
    model._prompt_enhancer_enable_presence_penalty = QWEN35_PROMPT_ENABLE_PRESENCE_PENALTY
    model._prompt_enhancer_presence_penalty = QWEN35_PROMPT_PRESENCE_PENALTY
    model._prompt_enhancer_min_new_tokens = (
        QWEN35_PROMPT_MIN_NEW_TOKENS
        if backend == enhancer_quantization_GGUF and _env_enabled(QWEN35_GGUF_LLAMACPP_ENV, default=True)
        else 0
    )
    model._prompt_enhancer_use_vllm = False
    model._prompt_enhancer_vllm_model_path = runtime_model_path
    model._prompt_enhancer_vllm_engine = None
    model._prompt_enhancer_vllm_mode = None
    model._prompt_enhancer_safe_legacy = safe_legacy_mode
    model._prompt_enhancer_use_vllm = engine_name == "vllm"
    model._prompt_enhancer_vllm_force_eager = engine_name == "vllm" and engine_detail != "cuda graph"
    if engine_name == "vllm":
        model._budget = 0
    print(f"[Prompt Enhancer][{spec['display_name']}] Text generation engine: {engine_name} ({engine_detail})")
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
]
