"""Qwen3 LM audio-code decoding engines (legacy/pt/vllm)."""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Callable, Optional

import torch

try:
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
except Exception:  # pragma: no cover
    class LogitsProcessor:  # type: ignore
        pass
    LogitsProcessorList = list  # type: ignore


_AUDIO_CODE_RE = re.compile(r"<\|audio_code_(\d+)\|>")


def _raise_if_non_finite_logits(logits: torch.Tensor, where: str) -> None:
    non_finite = ~torch.isfinite(logits)
    if not bool(non_finite.any()):
        return
    idx = torch.nonzero(non_finite, as_tuple=False)
    first = idx[0].tolist() if idx.numel() > 0 else []
    nan_count = int(torch.isnan(logits).sum().item())
    posinf_count = int(torch.isposinf(logits).sum().item())
    neginf_count = int(torch.isneginf(logits).sum().item())
    raise RuntimeError(
        f"[ace_step15] Non-finite logits detected at {where}: "
        f"nan={nan_count} +inf={posinf_count} -inf={neginf_count} first_index={first}"
    )


def _validate_masked_logits_for_sampling(logits: torch.Tensor, where: str) -> None:
    nan_count = int(torch.isnan(logits).sum().item())
    posinf_count = int(torch.isposinf(logits).sum().item())
    neginf_count = int(torch.isneginf(logits).sum().item())
    if nan_count > 0 or posinf_count > 0:
        bad_mask = torch.isnan(logits) | torch.isposinf(logits)
        idx = torch.nonzero(bad_mask, as_tuple=False)
        first = idx[0].tolist() if idx.numel() > 0 else []
        raise RuntimeError(
            f"[ace_step15] Non-finite logits detected at {where}: "
            f"nan={nan_count} +inf={posinf_count} -inf={neginf_count} first_index={first}"
        )
    check_logits = logits.unsqueeze(0) if logits.dim() == 1 else logits
    finite_per_row = torch.isfinite(check_logits).any(dim=-1)
    if not bool(finite_per_row.all()):
        bad_rows = torch.nonzero(~finite_per_row, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"[ace_step15] Decoding error at {where}: all candidates are -inf for row(s) {bad_rows}."
        )


def _token_id_to_audio_code(token_id: int, token_map: dict[int, int], tokenizer) -> Optional[int]:
    if token_map and token_id in token_map:
        return token_map[token_id]
    try:
        token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return None
    match = _AUDIO_CODE_RE.search(token_text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _postprocess_audio_codes(codes, min_tokens: int, max_tokens: int):
    if codes is None:
        return None
    if len(codes) == 0:
        return []
    if len(codes) < min_tokens:
        pad_val = codes[-1]
        codes.extend([pad_val] * (min_tokens - len(codes)))
    return codes[:max_tokens]


def _guess_lm_model_size(model_path: str) -> str | None:
    name = (model_path or "").lower()
    if "0.6b" in name:
        return "0.6B"
    if "1.7b" in name:
        return "1.7B"
    if "4b" in name:
        return "4B"
    return None


def _compute_vllm_gpu_utilization(model_path: str) -> float:
    try:
        if not torch.cuda.is_available():
            return 0.9
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return 0.9
    if total_gb <= 0:
        return 0.9
    target_map = {
        "0.6B": 3.0,
        "1.7B": 8.0,
        "4B": 12.0,
    }
    model_size = _guess_lm_model_size(model_path)
    target_gb = target_map.get(model_size, 3.0)
    ratio = target_gb / total_gb
    if total_gb >= 24:
        return min(0.9, max(0.2, ratio))
    return min(0.9, max(0.1, ratio))


class AudioCodeMaskProcessor(LogitsProcessor):
    def __init__(self, audio_code_mask: Optional[torch.Tensor]):
        self._mask_cpu = audio_code_mask
        self._allowed_idx_cpu = None
        self._allowed_idx_cache = {}
        if audio_code_mask is not None:
            try:
                allowed = torch.isfinite(audio_code_mask) & (audio_code_mask >= 0)
                self._allowed_idx_cpu = torch.nonzero(allowed, as_tuple=False).flatten().to("cpu", dtype=torch.long)
            except Exception:
                self._allowed_idx_cpu = None

    def _get_allowed_idx(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        if self._allowed_idx_cpu is None:
            return None
        key = logits.device
        cached = self._allowed_idx_cache.get(key)
        if cached is not None:
            return cached
        allowed_idx = self._allowed_idx_cpu.to(device=logits.device, dtype=torch.long)
        self._allowed_idx_cache[key] = allowed_idx
        return allowed_idx

    def __call__(self, input_ids, scores):
        allowed_idx = self._get_allowed_idx(scores)
        if allowed_idx is None:
            return scores
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        _raise_if_non_finite_logits(scores, "AudioCodeMaskProcessor")
        masked_scores = torch.full_like(scores, float("-inf"))
        masked_scores[:, allowed_idx] = scores[:, allowed_idx]
        if squeeze_back:
            masked_scores = masked_scores.squeeze(0)
        return masked_scores


def _prepare_cfg_inputs(tokenizer, prompt: str, prompt_negative: str):
    if prompt_negative is None:
        prompt_negative = prompt
    pos_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    neg_ids = tokenizer(prompt_negative, add_special_tokens=False)["input_ids"]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    max_len = max(len(pos_ids), len(neg_ids))
    pos_ids = ([pad_id] * (max_len - len(pos_ids))) + pos_ids
    neg_ids = ([pad_id] * (max_len - len(neg_ids))) + neg_ids
    input_ids = torch.tensor([pos_ids, neg_ids])
    return input_ids, pad_id, pos_ids


def _apply_top_k_top_p(cfg_logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        top_k_vals, _ = torch.topk(cfg_logits, top_k)
        min_val = top_k_vals[..., -1, None]
        cfg_logits = cfg_logits.clone()
        cfg_logits[cfg_logits < min_val] = float("-inf")

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(cfg_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        cfg_logits = cfg_logits.clone()
        cfg_logits[indices_to_remove] = float("-inf")
    return cfg_logits


def _sample_token_from_logits(
    cfg_logits: torch.Tensor,
    temperature: Optional[float],
    generator: Optional[torch.Generator],
) -> int:
    check_logits = cfg_logits.unsqueeze(0) if cfg_logits.dim() == 1 else cfg_logits
    nan_count = int(torch.isnan(check_logits).sum().item())
    posinf_count = int(torch.isposinf(check_logits).sum().item())
    if nan_count > 0 or posinf_count > 0:
        raise RuntimeError(
            f"[ace_step15] Decoding error before sampling: nan={nan_count} +inf={posinf_count}"
        )
    finite_per_row = torch.isfinite(check_logits).any(dim=-1)
    if not bool(finite_per_row.all()):
        bad_rows = torch.nonzero(~finite_per_row, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"[ace_step15] Decoding error before sampling: all candidates are -inf for row(s) {bad_rows}."
        )

    if temperature is not None and temperature > 0:
        scaled = cfg_logits / float(temperature)
        next_token = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1, generator=generator).squeeze(1)
    else:
        next_token = torch.argmax(cfg_logits, dim=-1)
    return int(next_token.item())


def _generate_token_ids_legacy(
    model,
    tokenizer,
    device,
    prompt: str,
    prompt_negative: str,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    cfg_scale: float,
    seed: Optional[int],
    callback=None,
    abort_fn: Optional[Callable[[], bool]] = None,
    logits_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    logits_processor_update_state: Optional[Callable[[int], None]] = None,
    stop_checker: Optional[Callable[[list[int], int], bool]] = None,
    progress_label: str = "LM tokens",
    ignore_eos: bool = False,
):
    input_ids, pad_id, pos_ids = _prepare_cfg_inputs(tokenizer, prompt, prompt_negative)
    input_ids = input_ids.to(device)
    attention_mask = (input_ids != pad_id).to(torch.long)
    cond_token_ids = list(pos_ids)

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1]

    token_ids = []
    eos_token_id = tokenizer.eos_token_id

    if callback is not None:
        callback(
            step_idx=-1,
            override_num_inference_steps=max_tokens,
            denoising_extra=f"{progress_label} 0/{max_tokens}",
            progress_unit="tokens",
        )

    for step in range(max_tokens):
        if abort_fn is not None and abort_fn():
            return None

        cond_logits = next_logits[0:1]
        uncond_logits = next_logits[1:2]
        cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
        _raise_if_non_finite_logits(cfg_logits, "legacy_cfg_logits_pre_mask")

        if logits_processor is not None:
            seq_input_ids = torch.tensor([cond_token_ids], device=device)
            cfg_logits = logits_processor(seq_input_ids, cfg_logits)
            _validate_masked_logits_for_sampling(cfg_logits, "legacy_cfg_logits_post_processor")

        cfg_logits = _apply_top_k_top_p(cfg_logits, top_k, top_p)
        token_id = _sample_token_from_logits(cfg_logits, temperature, generator)

        if eos_token_id is not None and token_id == int(eos_token_id) and not ignore_eos:
            if callback is not None:
                callback(
                    step_idx=int(step),
                    override_num_inference_steps=max_tokens,
                    denoising_extra=f"{progress_label} {step+1}/{max_tokens}",
                    progress_unit="tokens",
                )
            break

        token_ids.append(token_id)
        cond_token_ids.append(token_id)

        if logits_processor_update_state is not None:
            logits_processor_update_state(token_id)

        if callback is not None:
            callback(
                step_idx=int(step),
                override_num_inference_steps=max_tokens,
                denoising_extra=f"{progress_label} {step+1}/{max_tokens}",
                progress_unit="tokens",
            )

        if stop_checker is not None and stop_checker(token_ids, token_id):
            break

        next_input = torch.tensor([[token_id], [token_id]], device=device)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((2, 1), device=device, dtype=attention_mask.dtype)],
            dim=1,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1]

    return token_ids


def generate_text_legacy(
    model,
    tokenizer,
    device,
    prompt: str,
    prompt_negative: str,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    cfg_scale: float,
    seed: Optional[int],
    callback=None,
    abort_fn: Optional[Callable[[], bool]] = None,
    logits_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    logits_processor_update_state: Optional[Callable[[int], None]] = None,
    stop_checker: Optional[Callable[[list[int], int], bool]] = None,
    progress_label: str = "LM text",
    ignore_eos: bool = False,
):
    token_ids = _generate_token_ids_legacy(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        prompt_negative=prompt_negative,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        cfg_scale=cfg_scale,
        seed=seed,
        callback=callback,
        abort_fn=abort_fn,
        logits_processor=logits_processor,
        logits_processor_update_state=logits_processor_update_state,
        stop_checker=stop_checker,
        progress_label=progress_label,
        ignore_eos=ignore_eos,
    )
    if token_ids is None:
        return None
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return {"token_ids": token_ids, "text": text}


def generate_audio_codes_legacy(
    model,
    tokenizer,
    device,
    prompt: str,
    prompt_negative: str,
    audio_code_mask: Optional[torch.Tensor],
    audio_code_token_map: dict[int, int],
    min_tokens: int,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    cfg_scale: float,
    seed: Optional[int],
    callback=None,
    abort_fn: Optional[Callable[[], bool]] = None,
):
    mask_processor = AudioCodeMaskProcessor(audio_code_mask)
    text_out = generate_text_legacy(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        prompt_negative=prompt_negative,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        cfg_scale=cfg_scale,
        seed=seed,
        callback=callback,
        abort_fn=abort_fn,
        logits_processor=mask_processor,
        progress_label="Compute Audio Codes",
        ignore_eos=True,
    )
    if text_out is None:
        return None

    token_ids = text_out.get("token_ids", [])
    audio_codes = []
    for token_id in token_ids:
        code_val = _token_id_to_audio_code(token_id, audio_code_token_map, tokenizer)
        if code_val is not None:
            audio_codes.append(code_val)
            if len(audio_codes) >= max_tokens:
                break

    return _postprocess_audio_codes(audio_codes, min_tokens, max_tokens)


class Qwen3PtEngine:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        audio_code_mask: Optional[torch.Tensor],
        audio_code_token_map: dict[int, int],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.audio_code_mask = audio_code_mask
        self.audio_code_token_map = audio_code_token_map
        self._mask_processor = AudioCodeMaskProcessor(audio_code_mask)

    def generate_audio_codes(
        self,
        prompt: str,
        prompt_negative: str,
        min_tokens: int,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        callback=None,
        abort_fn: Optional[Callable[[], bool]] = None,
    ):
        if cfg_scale > 1.0:
            return generate_audio_codes_legacy(
                self.model,
                self.tokenizer,
                self.device,
                prompt,
                prompt_negative,
                self.audio_code_mask,
                self.audio_code_token_map,
                min_tokens,
                max_tokens,
                temperature,
                top_p,
                top_k,
                cfg_scale,
                seed,
                callback=callback,
                abort_fn=abort_fn,
            )

        if abort_fn is not None and abort_fn():
            return None

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        logits_processors = LogitsProcessorList([self._mask_processor])

        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=max_tokens,
                denoising_extra=f"Compute Audio Codes 0/{max_tokens}",
                progress_unit="tokens",
            )

        do_sample = temperature is not None and temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k if top_k is not None and top_k > 0 else None,
                top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
                logits_processor=logits_processors,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                generator=generator,
            )

        generated_ids = outputs[0][input_ids.shape[1]:]
        if generated_ids.is_cuda:
            generated_ids = generated_ids.cpu()

        audio_codes = []
        for token_id in generated_ids.tolist():
            code_val = _token_id_to_audio_code(int(token_id), self.audio_code_token_map, self.tokenizer)
            if code_val is not None:
                audio_codes.append(code_val)
                if len(audio_codes) >= max_tokens:
                    break

        if callback is not None:
            callback(
                step_idx=max(0, max_tokens - 1),
                override_num_inference_steps=max_tokens,
                denoising_extra=f"Compute Audio Codes {max_tokens}/{max_tokens}",
                progress_unit="tokens",
            )

        return _postprocess_audio_codes(audio_codes, min_tokens, max_tokens)

    def generate_text(
        self,
        prompt: str,
        prompt_negative: str,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        callback=None,
        abort_fn: Optional[Callable[[], bool]] = None,
        logits_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        logits_processor_update_state: Optional[Callable[[int], None]] = None,
        stop_checker: Optional[Callable[[list[int], int], bool]] = None,
        progress_label: str = "LM text",
    ):
        return generate_text_legacy(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt=prompt,
            prompt_negative=prompt_negative,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cfg_scale=cfg_scale,
            seed=seed,
            callback=callback,
            abort_fn=abort_fn,
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            stop_checker=stop_checker,
            progress_label=progress_label,
            ignore_eos=False,
        )


class Qwen3VllmEngine:
    def __init__(
        self,
        lm_weights_path: str,
        tokenizer,
        audio_code_mask: Optional[torch.Tensor],
        audio_code_token_map: dict[int, int],
        weight_load_mode: str = "lazy",
    ):
        self.lm_weights_path = lm_weights_path
        self.tokenizer = tokenizer
        self.audio_code_mask = audio_code_mask
        self.audio_code_token_map = audio_code_token_map
        self.weight_load_mode = (weight_load_mode or "lazy").lower()
        self._llm = None
        self._sampling_params_cls = None
        self._model_dir = None
        self._max_model_len_hint = None
        self._max_num_seqs_hint = None
        self._max_num_batched_tokens_hint = None
        self._last_zero_code_diagnostic = None

    @staticmethod
    def _compute_runtime_hints(prompt_len: int, max_tokens: int, cfg_scale: float):
        max_model_len = max(8, int(prompt_len) + int(max_tokens))
        max_num_seqs = 2 if cfg_scale and cfg_scale > 1.0 else 1
        max_num_batched_tokens = max_model_len * max_num_seqs
        return max_model_len, max_num_seqs, max_num_batched_tokens

    def _ensure_runtime_capacity(self, max_model_len: int, max_num_seqs: int, max_num_batched_tokens: int):
        if self._max_model_len_hint is None:
            self._max_model_len_hint = max_model_len
            self._max_num_seqs_hint = max_num_seqs
            self._max_num_batched_tokens_hint = max_num_batched_tokens
            return

        need_grow = (
            max_model_len > int(self._max_model_len_hint)
            or max_num_seqs > int(self._max_num_seqs_hint)
            or max_num_batched_tokens > int(self._max_num_batched_tokens_hint)
        )
        if not need_grow:
            return

        self._max_model_len_hint = max_model_len
        self._max_num_seqs_hint = max_num_seqs
        self._max_num_batched_tokens_hint = max_num_batched_tokens
        # Existing LLM was built with smaller runtime limits; rebuild safely.
        self.close()

    def _ensure_nanovllm_on_path(self):
        llm_engines_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shared", "llm_engines"))
        if llm_engines_root not in sys.path:
            sys.path.insert(0, llm_engines_root)

    def _prepare_model_dir(self):
        if self._model_dir is not None:
            return self._model_dir

        if not self.lm_weights_path:
            raise RuntimeError("vllm engine requires lm_weights_path")

        # For vLLM/nano-vllm, we can point directly at the weight file.
        # nanovllm.config will derive model_dir from the file path to find config/tokenizer.
        self._model_dir = self.lm_weights_path
        return self._model_dir

    def _ensure_llm(self):
        if self._llm is not None:
            return
        # Keep online-softmax path enabled for compiled LM sampling kernels.
        # Without this, Inductor may split reductions and emit a warning while
        # falling back to a slower softmax lowering.
        try:
            import torch._inductor.config as inductor_config

            if bool(getattr(inductor_config, "split_reductions", False)):
                inductor_config.split_reductions = False
        except Exception:
            pass
        self._ensure_nanovllm_on_path()
        try:
            from nanovllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"nano-vllm is not available for vllm engine: {exc}") from exc

        model_dir = self._prepare_model_dir()
        gpu_memory_utilization = _compute_vllm_gpu_utilization(self.lm_weights_path)
        max_model_len = self._max_model_len_hint or 4096
        max_num_seqs = self._max_num_seqs_hint or 1
        max_num_batched_tokens = self._max_num_batched_tokens_hint or (max_model_len * max_num_seqs)
        self._llm = LLM(
            model=model_dir,
            enforce_eager=False,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            tokenizer=self.tokenizer,
            weight_load_mode=self.weight_load_mode,
        )
        self._sampling_params_cls = SamplingParams

    def close(self):
        if self._llm is None:
            return
        try:
            self._llm.unload_weights()
        except Exception:
            pass
        try:
            self._llm.clear_graph_cache()
        except Exception:
            pass
        self._llm = None

    def get_last_failure_reason(self) -> str:
        diag = self._last_zero_code_diagnostic
        if not diag:
            return ""
        return (
            f"attempt={diag.get('attempt')} token_ids={diag.get('token_ids')} "
            f"unmapped={diag.get('unmapped')} out_of_vocab={diag.get('out_of_vocab')} "
            f"regex_hits={diag.get('regex_hits')} first_token_ids={diag.get('first_token_ids')} "
            f"text_preview='{diag.get('text_preview')}'"
        )

    def __del__(self):
        self.close()

    @staticmethod
    def _extract_text_and_tokens(output_obj) -> tuple[str, list[int]]:
        if output_obj is None:
            return "", []
        if isinstance(output_obj, dict):
            text = str(output_obj.get("text", "") or "")
            token_ids = output_obj.get("token_ids", []) or []
            return text, [int(x) for x in token_ids]
        if hasattr(output_obj, "outputs"):
            outputs = getattr(output_obj, "outputs", None)
            if outputs and len(outputs) > 0:
                text = str(getattr(outputs[0], "text", "") or "")
                token_ids = getattr(outputs[0], "token_ids", None)
                if token_ids is None:
                    token_ids = getattr(outputs[0], "token_ids_list", []) or []
                return text, [int(x) for x in token_ids] if token_ids else []
        text = str(getattr(output_obj, "text", "") or "")
        token_ids = getattr(output_obj, "token_ids", None)
        if token_ids is None:
            token_ids = []
        return text, [int(x) for x in token_ids]

    def generate_audio_codes(
        self,
        prompt: str,
        prompt_negative: str,
        min_tokens: int,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        callback=None,
        abort_fn: Optional[Callable[[], bool]] = None,
    ):
        if abort_fn is not None and abort_fn():
            return None

        try:
            prompt_len = len(self.tokenizer.encode(prompt))
        except Exception:
            prompt_len = 0
        # CFG runs conditional + unconditional sequences; runtime capacity must fit
        # the longest prompt between both or decode can enter invalid states.
        if cfg_scale > 1.0 and prompt_negative:
            try:
                prompt_len = max(prompt_len, len(self.tokenizer.encode(prompt_negative)))
            except Exception:
                pass
        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
        )
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)
        self._ensure_llm()
        if self._llm is None:
            return None
        try:
            self._llm.reset()
        except Exception:
            pass

        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=max_tokens,
                denoising_extra=f"Compute Audio Codes 0/{max_tokens}",
                progress_unit="tokens",
            )

        seed_value = None
        if seed is not None:
            try:
                seed_value = int(seed)
            except Exception:
                seed_value = None
            if seed_value is not None and seed_value < 0:
                seed_value = None

        temp = temperature if temperature is not None and temperature > 0 else 1e-5
        sampling_params = self._sampling_params_cls(
            temperature=temp,
            max_tokens=max_tokens,
            cfg_scale=max(cfg_scale, 1.0),
            top_k=top_k if top_k is not None and top_k > 0 else None,
            top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
            ignore_eos=True,
            logits_processor=AudioCodeMaskProcessor(self.audio_code_mask),
            seed=seed_value,
        )

        audio_codes = []
        self._last_zero_code_diagnostic = None
        for attempt in range(2):
            outputs = self._llm.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                use_tqdm=True,
                unconditional_prompts=[prompt_negative] if cfg_scale > 1.0 else None,
            )

            token_ids = outputs[0].get("token_ids", []) if outputs else []
            parsed_codes = []
            unmapped_count = 0
            out_of_vocab_count = 0
            vocab_size = None
            try:
                vocab_size = len(self.tokenizer.get_vocab())
            except Exception:
                vocab_size = None
            for token_id in token_ids:
                code_val = _token_id_to_audio_code(int(token_id), self.audio_code_token_map, self.tokenizer)
                if code_val is not None:
                    parsed_codes.append(code_val)
                    if len(parsed_codes) >= max_tokens:
                        break
                else:
                    unmapped_count += 1
                    if vocab_size is not None and (token_id < 0 or token_id >= vocab_size):
                        out_of_vocab_count += 1

            # Fallback parser: extract <|audio_code_x|> from decoded text if token_ids couldn't be mapped.
            if len(parsed_codes) == 0 and outputs:
                try:
                    decoded_text = str(outputs[0].get("text", ""))
                    parsed_codes = [int(x) for x in _AUDIO_CODE_RE.findall(decoded_text)]
                    if len(parsed_codes) > max_tokens:
                        parsed_codes = parsed_codes[:max_tokens]
                except Exception:
                    parsed_codes = []

            if parsed_codes:
                audio_codes = parsed_codes
                self._last_zero_code_diagnostic = None
                break

            decoded_text = ""
            regex_hits = 0
            if outputs:
                try:
                    decoded_text = str(outputs[0].get("text", ""))
                except Exception:
                    decoded_text = ""
            if decoded_text:
                try:
                    regex_hits = len(_AUDIO_CODE_RE.findall(decoded_text))
                except Exception:
                    regex_hits = 0
            text_preview = decoded_text[:280].replace("\n", "\\n")
            self._last_zero_code_diagnostic = {
                "attempt": attempt + 1,
                "token_ids": len(token_ids),
                "unmapped": unmapped_count,
                "out_of_vocab": out_of_vocab_count,
                "regex_hits": regex_hits,
                "first_token_ids": token_ids[:24],
                "text_preview": text_preview,
            }
            print(
                f"[ace_step15][vllm][diagnostic] attempt={attempt+1} "
                f"token_ids={len(token_ids)} parsed_codes=0 "
                f"unmapped={unmapped_count} out_of_vocab={out_of_vocab_count} "
                f"first_token_ids={token_ids[:24]} text_preview='{text_preview}'"
            )

            if attempt == 0:
                print("[ace_step15] Warning: vllm LM returned 0 audio codes; rebuilding LM runtime and retrying once.")
                self.close()
                self._ensure_llm()
                if self._llm is None:
                    break

        if callback is not None:
            callback(
                step_idx=max(0, max_tokens - 1),
                override_num_inference_steps=max_tokens,
                denoising_extra=f"Compute Audio Codes {max_tokens}/{max_tokens}",
                progress_unit="tokens",
            )

        if self.weight_load_mode in ("lazy", "pinned"):
            try:
                self._llm.unload_weights()
            except Exception:
                pass

        return _postprocess_audio_codes(audio_codes, min_tokens, max_tokens)

    def generate_text(
        self,
        prompt: str,
        prompt_negative: str,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        callback=None,
        abort_fn: Optional[Callable[[], bool]] = None,
        logits_processor: Optional[Any] = None,
        logits_processor_update_state: Optional[Callable[[int], None]] = None,
        stop_checker: Optional[Callable[[list[int], int], bool]] = None,
        progress_label: str = "LM text",
    ):
        del stop_checker  # nanovllm does not support external stop callbacks yet.
        if abort_fn is not None and abort_fn():
            return None

        try:
            prompt_len = len(self.tokenizer.encode(prompt))
        except Exception:
            prompt_len = 0
        if cfg_scale > 1.0 and prompt_negative:
            try:
                prompt_len = max(prompt_len, len(self.tokenizer.encode(prompt_negative)))
            except Exception:
                pass

        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
        )
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)
        self._ensure_llm()
        if self._llm is None:
            return None
        try:
            self._llm.reset()
        except Exception:
            pass

        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=max_tokens,
                denoising_extra=f"{progress_label} 0/{max_tokens}",
                progress_unit="tokens",
            )

        seed_value = None
        if seed is not None:
            try:
                seed_value = int(seed)
            except Exception:
                seed_value = None
            if seed_value is not None and seed_value < 0:
                seed_value = None

        temp = temperature if temperature is not None and temperature > 0 else 1e-5
        sampling_params = self._sampling_params_cls(
            temperature=temp,
            max_tokens=max_tokens,
            cfg_scale=max(cfg_scale, 1.0),
            top_k=top_k if top_k is not None and top_k > 0 else None,
            top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
            ignore_eos=False,
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            seed=seed_value,
        )

        outputs = self._llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
            use_tqdm=True,
            unconditional_prompts=[prompt_negative] if cfg_scale > 1.0 else None,
        )
        text = ""
        token_ids: list[int] = []
        if outputs:
            text, token_ids = self._extract_text_and_tokens(outputs[0])
        if (not text) and token_ids:
            try:
                text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception:
                text = ""

        if callback is not None:
            callback(
                step_idx=max(0, max_tokens - 1),
                override_num_inference_steps=max_tokens,
                denoising_extra=f"{progress_label} {max_tokens}/{max_tokens}",
                progress_unit="tokens",
            )

        if self.weight_load_mode in ("lazy", "pinned"):
            try:
                self._llm.unload_weights()
            except Exception:
                pass

        return {"token_ids": token_ids, "text": text}


def create_qwen3_lm_engine(
    engine_name: str,
    model,
    tokenizer,
    device,
    lm_weights_path: str,
    audio_code_mask: Optional[torch.Tensor],
    audio_code_token_map: dict[int, int],
    weight_load_mode: str = "lazy",
):
    engine = (engine_name or "legacy").strip().lower()
    if engine == "vllm":
        if not lm_weights_path or tokenizer is None:
            return None
        return Qwen3VllmEngine(
            lm_weights_path=lm_weights_path,
            tokenizer=tokenizer,
            audio_code_mask=audio_code_mask,
            audio_code_token_map=audio_code_token_map,
            weight_load_mode=weight_load_mode,
        )
    if engine == "pt":
        if model is None or tokenizer is None:
            return None
        return Qwen3PtEngine(
            model=model,
            tokenizer=tokenizer,
            device=device,
            audio_code_mask=audio_code_mask,
            audio_code_token_map=audio_code_token_map,
        )
    return None
