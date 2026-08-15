"""Qwen3 decode model adapted from WanGP's nano-vLLM implementation."""

import torch
import torch.nn as nn

from shared.llm_engines.nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from shared.llm_engines.nanovllm.layers.layernorm import RMSNorm
from shared.llm_engines.nanovllm.models.qwen3 import Qwen3DecoderLayer


class MiniMaxMusic3Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor, inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MiniMaxMusic3Qwen3ForCausalLM(nn.Module):
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MiniMaxMusic3Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    @property
    def dtype(self):
        return self.model.embed_tokens.weight.dtype

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor, inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
