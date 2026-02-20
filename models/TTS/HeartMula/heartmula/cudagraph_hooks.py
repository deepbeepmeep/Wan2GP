from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from shared.llm_engines.cudagraph_kit import AutoRegressiveCudaGraphKit

if TYPE_CHECKING:
    from .llama_blocks import TransformerDecoder
    from .modeling_heartmula import HeartMuLa


class HeartMuLaCudaGraphHooks:
    def __init__(self, model: "HeartMuLa") -> None:
        self.model = model
        self._enabled = False
        self._kit: AutoRegressiveCudaGraphKit | None = None
        self._backbone_tokens = 0
        self._decoder_tokens = 0
        self._backbone_flash_graph = False

    def set_lm_decoder_engine(self, lm_decoder_engine: str | None) -> None:
        engine = str(lm_decoder_engine or "legacy").strip().lower()
        self._enabled = engine == "cg"
        if not self._enabled:
            self.release()

    @staticmethod
    def _collect_attention_modules(transformer: "TransformerDecoder"):
        return [layer.attn for layer in transformer.layers if hasattr(layer, "attn")]

    @staticmethod
    def _can_use_flash_graph(attn_modules: list[object]) -> bool:
        if not attn_modules:
            return False
        for attn in attn_modules:
            if not bool(getattr(attn, "allow_flash2_kvcache", False)):
                return False
            if not bool(getattr(attn, "_flash_kv_ready", False)):
                return False
        return True

    def prepare(self, *, max_backbone_tokens: int, max_decoder_tokens: int) -> None:
        self.release()
        if not self._enabled:
            return
        model_device = next(self.model.parameters()).device
        if model_device.type != "cuda":
            return
        if not self.model.backbone.caches_are_enabled() or not self.model.decoder[0].caches_are_enabled():
            raise RuntimeError("HeartMuLa caches must be setup before CUDA graph preparation.")
        if max_backbone_tokens <= 0 or max_decoder_tokens <= 0:
            raise ValueError("CUDA graph token limits must be > 0.")
        kit = AutoRegressiveCudaGraphKit("heartmula")
        backbone_attn = self._collect_attention_modules(self.model.backbone)
        decoder_attn = self._collect_attention_modules(self.model.decoder[0])
        self._backbone_flash_graph = self._can_use_flash_graph(backbone_attn)
        backbone_graph_mode = "flash_kvcache" if self._backbone_flash_graph else "index"
        self._backbone_tokens = int(
            kit.attach_attention_modules(backbone_attn, max_backbone_tokens, graph_mode=backbone_graph_mode)
        )
        self._decoder_tokens = int(
            kit.attach_attention_modules(decoder_attn, max_decoder_tokens, graph_mode="index")
        )
        self._kit = kit
        self._prime_decode_kernels()

    def release(self) -> None:
        if self._kit is not None:
            self._kit.release()
            self._kit = None
        self._backbone_tokens = 0
        self._decoder_tokens = 0
        self._backbone_flash_graph = False

    def _active(self) -> bool:
        return self._kit is not None and self._backbone_tokens > 0 and self._decoder_tokens > 0

    def is_active(self) -> bool:
        return self._active()

    def _backbone_decode_step(self, hidden_states: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        if self._backbone_flash_graph:
            return self.model.backbone(hidden_states, input_pos=input_pos, mask=None)
        mask = self.model.backbone_causal_mask[input_pos, : self._backbone_tokens]
        return self.model.backbone(hidden_states, input_pos=input_pos, mask=mask)

    def _decoder_decode_step(self, hidden_states: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        mask = self.model.decoder_causal_mask[input_pos, : self._decoder_tokens]
        return self.model.decoder[0](self.model.projection(hidden_states), input_pos=input_pos, mask=mask)

    def _prime_decode_kernels(self) -> None:
        if not self._active():
            return
        kv_cache = self.model.backbone.layers[0].attn.kv_cache
        if kv_cache is None:
            return
        batch_size = int(kv_cache.k_cache.shape[0])
        hidden_dim = int(self.model.projection.in_features)
        model_device = kv_cache.k_cache.device
        model_dtype = next(self.model.parameters()).dtype
        hidden = torch.zeros((batch_size, 1, hidden_dim), device=model_device, dtype=model_dtype)
        pos = torch.zeros((batch_size, 1), device=model_device, dtype=torch.long)
        with torch.no_grad():
            _ = self._backbone_decode_step(hidden, pos)
            self.model.backbone.reset_caches()
            _ = self._decoder_decode_step(hidden, pos)
            self.model.decoder[0].reset_caches()

    def run_backbone_decode(self, hidden_states: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        if not self._active():
            raise RuntimeError("Backbone CUDA graph decode called while CUDA graph is inactive.")
        token_pos = int(input_pos.reshape(-1)[0].item())
        if token_pos < 0 or token_pos >= self._backbone_tokens:
            raise RuntimeError(
                f"Backbone decode position {token_pos} exceeds CUDA graph cache capacity {self._backbone_tokens}."
            )
        try:
            return self._kit.run("backbone_decode", self._backbone_decode_step, hidden_states, input_pos)
        except Exception as exc:
            self.release()
            raise RuntimeError(f"HeartMuLa backbone CUDA graph decode failed: {exc}") from exc

    def run_decoder(self, hidden_states: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        if not self._active():
            raise RuntimeError("Decoder CUDA graph decode called while CUDA graph is inactive.")
        last_output = None
        try:
            for token_idx in range(hidden_states.shape[1]):
                token_pos = input_pos[:, token_idx : token_idx + 1]
                absolute_pos = int(token_pos.reshape(-1)[0].item())
                if absolute_pos < 0 or absolute_pos >= self._decoder_tokens:
                    raise RuntimeError(
                        f"Decoder decode position {absolute_pos} exceeds CUDA graph cache capacity {self._decoder_tokens}."
                    )
                token_hidden = hidden_states[:, token_idx : token_idx + 1]
                last_output = self._kit.run("decoder_decode", self._decoder_decode_step, token_hidden, token_pos)
        except Exception as exc:
            self.release()
            raise RuntimeError(f"HeartMuLa decoder CUDA graph decode failed: {exc}") from exc
        return last_output
