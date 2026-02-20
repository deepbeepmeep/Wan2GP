"""Qwen3 LM audio-code sampling helpers (engine-neutral)."""

from __future__ import annotations

import re
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


def generate_text_sampling(
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


def generate_audio_codes_with_engine_sampling(
    engine,
    tokenizer,
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
    release_vram_after: bool = True,
):
    mask_processor = AudioCodeMaskProcessor(audio_code_mask)
    text_out = engine.generate_text(
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
        logits_processor_update_state=None,
        stop_checker=None,
        progress_label="Compute Audio Codes",
        release_vram_after=release_vram_after,
        ignore_eos=True,
    )
    if text_out is None:
        return None, ""
    if abort_fn is not None and abort_fn():
        return None, ""

    if isinstance(text_out, dict):
        token_ids = text_out.get("token_ids", []) or []
        decoded_text = str(text_out.get("text", "") or "")
    else:
        token_ids = []
        decoded_text = str(text_out or "")
    token_ids = [int(x) for x in token_ids]

    parsed_codes = []
    unmapped_count = 0
    out_of_vocab_count = 0
    vocab_size = None
    try:
        vocab_size = len(tokenizer.get_vocab())
    except Exception:
        vocab_size = None

    for token_id in token_ids:
        code_val = _token_id_to_audio_code(int(token_id), audio_code_token_map, tokenizer)
        if code_val is not None:
            parsed_codes.append(code_val)
            if len(parsed_codes) >= max_tokens:
                break
        else:
            unmapped_count += 1
            if vocab_size is not None and (token_id < 0 or token_id >= vocab_size):
                out_of_vocab_count += 1

    regex_hits = 0
    if len(parsed_codes) == 0 and decoded_text:
        try:
            parsed_codes = [int(x) for x in _AUDIO_CODE_RE.findall(decoded_text)]
            regex_hits = len(parsed_codes)
            if len(parsed_codes) > max_tokens:
                parsed_codes = parsed_codes[:max_tokens]
        except Exception:
            parsed_codes = []
            regex_hits = 0
    if abort_fn is not None and abort_fn():
        return None, ""

    failure_reason = ""
    if len(parsed_codes) == 0:
        text_preview = decoded_text[:280].replace("\n", "\\n")
        failure_reason = (
            f"token_ids={len(token_ids)} unmapped={unmapped_count} out_of_vocab={out_of_vocab_count} "
            f"regex_hits={regex_hits} first_token_ids={token_ids[:24]} text_preview='{text_preview}'"
        )

    return _postprocess_audio_codes(parsed_codes, min_tokens, max_tokens), failure_reason
