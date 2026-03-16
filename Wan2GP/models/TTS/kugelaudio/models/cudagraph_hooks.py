from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers.cache_utils import StaticCache

from shared.llm_engines.cudagraph_kit import AutoRegressiveCudaGraphKit

if TYPE_CHECKING:
    from .kugelaudio_inference import KugelAudioForConditionalGenerationInference


class KugelAudioCudaGraphHooks:
    def __init__(self, model: "KugelAudioForConditionalGenerationInference") -> None:
        self.model = model
        self._enabled = False
        self._kit: AutoRegressiveCudaGraphKit | None = None
        self._cache: StaticCache | None = None
        self._max_batch_size = 0
        self._max_cache_tokens = 0
        self._last_prepare_reused = False
        self._decode_cache_position: torch.Tensor | None = None

    def set_lm_decoder_engine(self, lm_decoder_engine: str | None) -> None:
        engine = str(lm_decoder_engine or "legacy").strip().lower()
        self._enabled = engine == "cg"
        if not self._enabled:
            self.release()

    def prepare(self, *, max_batch_size: int, max_cache_tokens: int) -> None:
        self._last_prepare_reused = False
        if not self._enabled:
            return
        if max_batch_size <= 0 or max_cache_tokens <= 0:
            raise ValueError("KugelAudio CUDA graph capacities must be > 0.")
        model_device = next(self.model.parameters()).device
        if model_device.type != "cuda":
            return
        if (
            self._cache is not None
            and self._kit is not None
            and self._max_batch_size >= int(max_batch_size)
            and self._max_cache_tokens >= int(max_cache_tokens)
        ):
            self._last_prepare_reused = True
            self._cache.reset()
            if self._decode_cache_position is None or self._decode_cache_position.device != model_device:
                self._decode_cache_position = torch.zeros(1, device=model_device, dtype=torch.long)
            return
        self.release()
        model_dtype = next(self.model.parameters()).dtype
        self._cache = StaticCache(
            config=self.model.config.decoder_config,
            max_batch_size=int(max_batch_size),
            max_cache_len=int(max_cache_tokens),
            device=model_device,
            dtype=model_dtype,
        )
        self._decode_cache_position = torch.zeros(1, device=model_device, dtype=torch.long)
        self._kit = AutoRegressiveCudaGraphKit("kugelaudio")
        self._max_batch_size = int(max_batch_size)
        self._max_cache_tokens = int(max_cache_tokens)

    def release(self) -> None:
        if self._kit is not None:
            self._kit.release()
            self._kit = None
        self._cache = None
        self._decode_cache_position = None
        self._max_batch_size = 0
        self._max_cache_tokens = 0
        self._last_prepare_reused = False

    def is_active(self) -> bool:
        return self._kit is not None and self._cache is not None and self._max_cache_tokens > 0

    def is_enabled(self) -> bool:
        return bool(self._enabled)

    def last_prepare_reused(self) -> bool:
        return bool(self._last_prepare_reused)

    def has_capacity(self, token_position: int) -> bool:
        return int(token_position) < self._max_cache_tokens

    def _check_batch_size(self, batch_size: int) -> None:
        if int(batch_size) > self._max_batch_size:
            raise RuntimeError(
                f"KugelAudio CUDA graph batch size {int(batch_size)} exceeds prepared capacity {self._max_batch_size}."
            )

    def prefill(self, inputs_embeds: torch.Tensor):
        if not self.is_active():
            raise RuntimeError("KugelAudio CUDA graph prefill called while CUDA graph is inactive.")
        self._check_batch_size(int(inputs_embeds.shape[0]))
        self._cache.reset()
        prompt_len = int(inputs_embeds.shape[1])
        if prompt_len <= 0:
            raise RuntimeError("KugelAudio CUDA graph prefill requires at least one prompt token.")
        if prompt_len > self._max_cache_tokens:
            raise RuntimeError(
                f"KugelAudio prompt length {prompt_len} exceeds CUDA graph cache capacity {self._max_cache_tokens}."
            )
        cache_position = torch.arange(prompt_len, device=inputs_embeds.device, dtype=torch.long)
        try:
            return self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=self._cache,
                use_cache=True,
                return_dict=True,
                cache_position=cache_position,
            )
        except Exception as exc:
            self.release()
            raise RuntimeError(f"KugelAudio CUDA graph prefill failed: {exc}") from exc

    def _decode_step(self, token_embeds: torch.Tensor, cache_position: torch.Tensor):
        outputs = self.model(
            inputs_embeds=token_embeds,
            past_key_values=self._cache,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )
        return outputs.logits, outputs.last_hidden_state

    def run_decode(self, token_embeds: torch.Tensor, token_position: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_active():
            raise RuntimeError("KugelAudio CUDA graph decode called while CUDA graph is inactive.")
        self._check_batch_size(int(token_embeds.shape[0]))
        if int(token_embeds.shape[1]) != 1:
            raise RuntimeError(
                f"KugelAudio CUDA graph decode expects a single token, got seq_len={int(token_embeds.shape[1])}."
            )
        absolute_pos = int(token_position)
        if absolute_pos < 0 or absolute_pos >= self._max_cache_tokens:
            raise RuntimeError(
                f"KugelAudio decode position {absolute_pos} exceeds CUDA graph cache capacity {self._max_cache_tokens}."
            )
        if self._decode_cache_position is None or self._decode_cache_position.device != token_embeds.device:
            self._decode_cache_position = torch.empty(1, device=token_embeds.device, dtype=torch.long)
        cache_position = self._decode_cache_position.fill_(absolute_pos)
        try:
            return self._kit.run("lm_decode", self._decode_step, token_embeds, cache_position)
        except Exception as exc:
            self.release()
            raise RuntimeError(f"KugelAudio CUDA graph decode failed: {exc}") from exc
