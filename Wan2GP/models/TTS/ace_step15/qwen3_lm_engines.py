"""Qwen3 LM engine adapters."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

from shared.llm_engines.nanovllm.vllm_support import NanoVllmTextEngine

from .qwen3_audio_codes import generate_text_sampling


@runtime_checkable
class Qwen3LmEngine(Protocol):
    keep_loaded_for_phase2: bool

    def reserve_runtime(self, prompt_len: int, max_tokens: int, cfg_scale: float):
        ...

    def release_runtime_allocations(self):
        ...

    def get_last_failure_reason(self) -> str:
        ...

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
        logits_processor: Optional[Callable[[Any, Any], Any]] = None,
        logits_processor_update_state: Optional[Callable[[int], None]] = None,
        stop_checker: Optional[Callable[[list[int], int], bool]] = None,
        progress_label: str = "LM text",
        release_vram_after: bool = True,
        ignore_eos: bool = False,
    ):
        ...


class Qwen3LegacyEngine:
    keep_loaded_for_phase2 = False

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def reserve_runtime(self, prompt_len: int, max_tokens: int, cfg_scale: float):
        del prompt_len, max_tokens, cfg_scale

    def release_runtime_allocations(self):
        return

    def get_last_failure_reason(self) -> str:
        return ""

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
        logits_processor: Optional[Callable[[Any, Any], Any]] = None,
        logits_processor_update_state: Optional[Callable[[int], None]] = None,
        stop_checker: Optional[Callable[[list[int], int], bool]] = None,
        progress_label: str = "LM text",
        release_vram_after: bool = True,
        ignore_eos: bool = False,
    ):
        del release_vram_after
        return generate_text_sampling(
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
            ignore_eos=ignore_eos,
        )


class Qwen3VllmEngine(NanoVllmTextEngine):
    keep_loaded_for_phase2 = True

    def __init__(self, model, tokenizer, device, lm_weights_path: str):
        del device
        super().__init__(model=model, model_path=lm_weights_path, tokenizer=tokenizer)

    def reserve_runtime(self, prompt_len: int, max_tokens: int, cfg_scale: float):
        return super().reserve_runtime(prompt_len=prompt_len, max_tokens=max_tokens, cfg_scale=cfg_scale)

    def release_runtime_allocations(self):
        return super().release_runtime_allocations()

    def get_last_failure_reason(self) -> str:
        return super().get_last_failure_reason()

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
        logits_processor: Optional[Callable[[Any, Any], Any]] = None,
        logits_processor_update_state: Optional[Callable[[int], None]] = None,
        stop_checker: Optional[Callable[[list[int], int], bool]] = None,
        progress_label: str = "LM text",
        release_vram_after: bool = True,
        ignore_eos: bool = False,
    ):
        return super().generate_text(
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
            release_vram_after=release_vram_after,
            ignore_eos=ignore_eos,
        )
