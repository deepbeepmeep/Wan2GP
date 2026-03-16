from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from transformers.cache_utils import StaticCache

from shared.llm_engines.cudagraph_kit import AutoRegressiveCudaGraphKit

if TYPE_CHECKING:
    from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig
    from .modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration


class Qwen3TTSCudaGraphHooks:
    def __init__(self, talker: "Qwen3TTSTalkerForConditionalGeneration") -> None:
        self.talker = talker
        self._enabled = False
        self._kit: AutoRegressiveCudaGraphKit | None = None
        self._talker_cache: StaticCache | None = None
        self._subtalker_cache: StaticCache | None = None
        self._max_batch_size = 0
        self._max_talker_tokens = 0
        self._max_subtalker_tokens = 0
        self._last_prepare_reused = False

        self._talker_decode_cache_position: torch.Tensor | None = None
        self._subtalker_decode_cache_position: torch.Tensor | None = None
        self._talker_decode_attention_mask: torch.Tensor | None = None
        self._subtalker_decode_attention_mask: torch.Tensor | None = None
        self._talker_decode_position_ids: torch.Tensor | None = None
        self._subtalker_decode_position_ids: torch.Tensor | None = None

    def set_lm_decoder_engine(self, lm_decoder_engine: str | None) -> None:
        engine = str(lm_decoder_engine or "legacy").strip().lower()
        self._enabled = engine in ("cg", "cudagraph")
        if not self._enabled:
            self.release()

    def is_enabled(self) -> bool:
        return bool(self._enabled)

    def is_active(self) -> bool:
        return (
            self._kit is not None
            and self._talker_cache is not None
            and self._subtalker_cache is not None
            and self._max_talker_tokens > 0
            and self._max_subtalker_tokens > 0
        )

    def last_prepare_reused(self) -> bool:
        return bool(self._last_prepare_reused)

    def prepare(
        self,
        *,
        max_batch_size: int,
        max_talker_tokens: int,
        max_subtalker_tokens: int,
        talker_config: "Qwen3TTSTalkerConfig",
        subtalker_config: "Qwen3TTSTalkerCodePredictorConfig",
    ) -> None:
        self._last_prepare_reused = False
        if not self._enabled:
            return
        if max_batch_size <= 0 or max_talker_tokens <= 0 or max_subtalker_tokens <= 0:
            raise ValueError("Qwen3 CUDA graph capacities must be > 0.")

        first_param = next(self.talker.parameters(), None)
        if first_param is None or first_param.device.type != "cuda":
            return

        if (
            self._talker_cache is not None
            and self._subtalker_cache is not None
            and self._kit is not None
            and self._max_batch_size >= int(max_batch_size)
            and self._max_talker_tokens >= int(max_talker_tokens)
            and self._max_subtalker_tokens >= int(max_subtalker_tokens)
        ):
            self._last_prepare_reused = True
            self.reset_talker_cache()
            self.reset_subtalker_cache()
            self._ensure_runtime_tensors(first_param.device)
            return

        self.release()
        self._kit = AutoRegressiveCudaGraphKit("qwen3_tts")
        self._max_batch_size = int(max_batch_size)
        self._max_talker_tokens = int(max_talker_tokens)
        self._max_subtalker_tokens = int(max_subtalker_tokens)
        self._talker_cache = StaticCache(
            config=talker_config,
            max_batch_size=self._max_batch_size,
            max_cache_len=self._max_talker_tokens,
            device=first_param.device,
            dtype=first_param.dtype,
        )
        self._subtalker_cache = StaticCache(
            config=subtalker_config,
            max_batch_size=self._max_batch_size,
            max_cache_len=self._max_subtalker_tokens,
            device=first_param.device,
            dtype=first_param.dtype,
        )
        self._ensure_runtime_tensors(first_param.device)

    def release(self) -> None:
        if self._kit is not None:
            self._kit.release()
            self._kit = None
        self._talker_cache = None
        self._subtalker_cache = None
        self._max_batch_size = 0
        self._max_talker_tokens = 0
        self._max_subtalker_tokens = 0
        self._last_prepare_reused = False
        self._talker_decode_cache_position = None
        self._subtalker_decode_cache_position = None
        self._talker_decode_attention_mask = None
        self._subtalker_decode_attention_mask = None
        self._talker_decode_position_ids = None
        self._subtalker_decode_position_ids = None

    def _ensure_runtime_tensors(self, device: torch.device) -> None:
        if self._max_batch_size <= 0:
            return
        if self._talker_decode_cache_position is None or self._talker_decode_cache_position.device != device:
            self._talker_decode_cache_position = torch.zeros(1, dtype=torch.long, device=device)
        if self._subtalker_decode_cache_position is None or self._subtalker_decode_cache_position.device != device:
            self._subtalker_decode_cache_position = torch.zeros(1, dtype=torch.long, device=device)
        if (
            self._talker_decode_attention_mask is None
            or self._talker_decode_attention_mask.device != device
            or self._talker_decode_attention_mask.shape != (self._max_batch_size, self._max_talker_tokens)
        ):
            self._talker_decode_attention_mask = torch.zeros(
                (self._max_batch_size, self._max_talker_tokens), dtype=torch.long, device=device
            )
        if (
            self._subtalker_decode_attention_mask is None
            or self._subtalker_decode_attention_mask.device != device
            or self._subtalker_decode_attention_mask.shape != (self._max_batch_size, self._max_subtalker_tokens)
        ):
            self._subtalker_decode_attention_mask = torch.zeros(
                (self._max_batch_size, self._max_subtalker_tokens), dtype=torch.long, device=device
            )
        if (
            self._talker_decode_position_ids is None
            or self._talker_decode_position_ids.device != device
            or self._talker_decode_position_ids.shape != (3, self._max_batch_size, 1)
        ):
            self._talker_decode_position_ids = torch.zeros((3, self._max_batch_size, 1), dtype=torch.long, device=device)
        if (
            self._subtalker_decode_position_ids is None
            or self._subtalker_decode_position_ids.device != device
            or self._subtalker_decode_position_ids.shape != (self._max_batch_size, 1)
        ):
            self._subtalker_decode_position_ids = torch.zeros((self._max_batch_size, 1), dtype=torch.long, device=device)

    def _check_batch_size(self, batch_size: int) -> None:
        if int(batch_size) > self._max_batch_size:
            raise RuntimeError(
                f"Qwen3 CUDA graph batch size {int(batch_size)} exceeds prepared capacity {self._max_batch_size}."
            )

    def reset_talker_cache(self) -> None:
        if self._talker_cache is not None:
            self._talker_cache.reset()

    def reset_subtalker_cache(self) -> None:
        if self._subtalker_cache is not None:
            self._subtalker_cache.reset()

    def get_talker_cache(self) -> StaticCache | None:
        return self._talker_cache

    def get_subtalker_cache(self) -> StaticCache | None:
        return self._subtalker_cache

    def is_talker_cache(self, cache_obj) -> bool:
        return cache_obj is self._talker_cache and cache_obj is not None

    def is_subtalker_cache(self, cache_obj) -> bool:
        return cache_obj is self._subtalker_cache and cache_obj is not None

    def _copy_attention_mask(
        self,
        source_mask: torch.Tensor | None,
        dest_mask: torch.Tensor,
        *,
        cache_position: int,
        max_tokens: int,
        batch_size: int,
    ) -> torch.Tensor:
        if source_mask is None:
            current_len = min(max_tokens, cache_position + 1)
            dest_mask.zero_()
            dest_mask[:batch_size, :current_len] = 1
            return dest_mask[:batch_size]
        if source_mask.dim() != 2:
            raise RuntimeError(f"Expected 2D attention mask for CUDA graph decode, got shape={tuple(source_mask.shape)}.")
        if int(source_mask.shape[0]) != batch_size:
            raise RuntimeError(
                f"Attention mask batch {int(source_mask.shape[0])} does not match decode batch {batch_size}."
            )
        current_len = int(source_mask.shape[1])
        if current_len > max_tokens:
            raise RuntimeError(
                f"Attention mask length {current_len} exceeds CUDA graph cache capacity {max_tokens}."
            )
        dest_mask.zero_()
        mask = source_mask.to(device=dest_mask.device, dtype=dest_mask.dtype)
        dest_mask[:batch_size, :current_len].copy_(mask)
        return dest_mask[:batch_size]

    def run_talker_decode(
        self,
        decode_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], object],
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ):
        if not self.is_active():
            raise RuntimeError("Qwen3 talker CUDA graph decode called while CUDA graph is inactive.")
        if int(inputs_embeds.shape[1]) != 1:
            raise RuntimeError(f"Qwen3 talker CUDA graph decode expects seq_len=1, got {int(inputs_embeds.shape[1])}.")
        if int(position_ids.shape[-1]) != 1:
            raise RuntimeError(
                f"Qwen3 talker CUDA graph decode expects position_ids with last dim=1, got {tuple(position_ids.shape)}."
            )
        absolute_pos = int(cache_position.reshape(-1)[0].item())
        if absolute_pos < 0 or absolute_pos >= self._max_talker_tokens:
            raise RuntimeError(
                f"Qwen3 talker decode position {absolute_pos} exceeds CUDA graph cache capacity {self._max_talker_tokens}."
            )
        batch_size = int(inputs_embeds.shape[0])
        self._check_batch_size(batch_size)
        self._ensure_runtime_tensors(inputs_embeds.device)
        assert self._talker_decode_cache_position is not None
        assert self._talker_decode_attention_mask is not None
        assert self._talker_decode_position_ids is not None
        static_cache_position = self._talker_decode_cache_position.fill_(absolute_pos)
        static_attention_mask = self._copy_attention_mask(
            attention_mask,
            self._talker_decode_attention_mask,
            cache_position=absolute_pos,
            max_tokens=self._max_talker_tokens,
            batch_size=batch_size,
        )
        static_position_ids = self._talker_decode_position_ids[:3, :batch_size]
        static_position_ids.copy_(position_ids.to(device=static_position_ids.device, dtype=static_position_ids.dtype))
        try:
            return self._kit.run("talker_decode", decode_fn, inputs_embeds, static_attention_mask, static_position_ids, static_cache_position)
        except Exception as exc:
            self.release()
            raise RuntimeError(f"Qwen3 talker CUDA graph decode failed: {exc}") from exc

    def run_subtalker_decode(
        self,
        decode_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], object],
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        cache_position: torch.Tensor,
        generation_step: int,
    ):
        if not self.is_active():
            raise RuntimeError("Qwen3 subtalker CUDA graph decode called while CUDA graph is inactive.")
        if int(inputs_embeds.shape[1]) != 1:
            raise RuntimeError(
                f"Qwen3 subtalker CUDA graph decode expects seq_len=1, got {int(inputs_embeds.shape[1])}."
            )
        absolute_pos = int(cache_position.reshape(-1)[0].item())
        if absolute_pos < 0 or absolute_pos >= self._max_subtalker_tokens:
            raise RuntimeError(
                f"Qwen3 subtalker decode position {absolute_pos} exceeds CUDA graph cache capacity {self._max_subtalker_tokens}."
            )
        batch_size = int(inputs_embeds.shape[0])
        self._check_batch_size(batch_size)
        self._ensure_runtime_tensors(inputs_embeds.device)
        assert self._subtalker_decode_cache_position is not None
        assert self._subtalker_decode_attention_mask is not None
        assert self._subtalker_decode_position_ids is not None
        static_cache_position = self._subtalker_decode_cache_position.fill_(absolute_pos)
        static_attention_mask = self._copy_attention_mask(
            attention_mask,
            self._subtalker_decode_attention_mask,
            cache_position=absolute_pos,
            max_tokens=self._max_subtalker_tokens,
            batch_size=batch_size,
        )
        static_position_ids = self._subtalker_decode_position_ids[:batch_size]
        if position_ids is None:
            static_position_ids.fill_(absolute_pos)
        else:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(-1)
            if position_ids.dim() != 2 or int(position_ids.shape[-1]) != 1:
                raise RuntimeError(
                    f"Qwen3 subtalker CUDA graph decode expects position_ids shape [batch, 1], got {tuple(position_ids.shape)}."
                )
            static_position_ids.copy_(position_ids.to(device=static_position_ids.device, dtype=static_position_ids.dtype))
        runner_name = "subtalker_decode"
        try:
            return self._kit.run(
                runner_name,
                decode_fn,
                inputs_embeds,
                static_attention_mask,
                static_position_ids,
                static_cache_position,
            )
        except Exception as exc:
            self.release()
            raise RuntimeError(f"Qwen3 subtalker CUDA graph decode failed: {exc}") from exc
