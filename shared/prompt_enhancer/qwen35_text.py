from __future__ import annotations

import gc
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
from shared.llm_engines.nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM, clear_qwen35_runtime_caches
from shared.llm_engines.nanovllm.utils.context import reset_context
from shared.llm_engines.nanovllm.vllm_support import (
    NanoVllmTextEngine,
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
):
    reset_context()
    tokenizer = self._prompt_enhancer_tokenizer
    if len(messages) == 0:
        return []

    engine = _get_or_create_vllm_engine(self, usage_mode="text")
    outputs = []
    progress_desc = (
        "Qwen3.5 prompt enhancement (legacy)"
        if _use_legacy_cuda_runner_prompt_enhancer(self)
        else f"Qwen3.5 prompt enhancement ({getattr(self, '_prompt_enhancer_engine_name', 'vllm')})"
    )
    for idx, message in enumerate(tqdm(messages, total=len(messages), desc=progress_desc, dynamic_ncols=True, leave=False)):
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
    materialize_module_source_tensors(model)
    return model


def _resolve_legacy_text_execution_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 legacy prompt enhancement now requires CUDA.")
    return torch.device("cuda", torch.cuda.current_device())


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

    engine_name, _engine_detail, enable_cudagraph, allow_vllm_kernels = _resolve_prompt_enhancer_engine(
        backend=backend,
        requested_lm_engine=requested_lm_engine,
        runtime_model_path=runtime_model_path,
    )
    safe_legacy_mode = not allow_vllm_kernels

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Qwen3.5 text checkpoint not found: {model_path}")

    model = _load_local_text_model(
        model_path,
        text_config_path,
        preprocess_sd=preprocess_sd,
        default_dtype=default_dtype,
        safe_legacy_mode=safe_legacy_mode,
    )
    if engine_name == "legacy":
        model = model.to(_resolve_legacy_text_execution_device())
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
]
