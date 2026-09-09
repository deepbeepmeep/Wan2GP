# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""WanGP-local port of Diffusers' MiniMax Music 3 RVQ depth decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm

from shared.attention import pay_attention
from shared.llm_engines.cudagraph_kit import CUDAGraphRunner


class MiniMaxMusic3DepthAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.k_cache = self.v_cache = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        key = self.to_k(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        value = self.to_v(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        hidden_states = pay_attention([query, key, value], causal=True, force_attention="sdpa", recycle_q=True)
        return self.to_out(hidden_states.flatten(2, 3).to(query.dtype))

    def decode(self, hidden_states: torch.Tensor, position: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states).view(batch_size, 1, self.heads, self.head_dim)
        key = self.to_k(hidden_states).view(batch_size, 1, self.heads, self.head_dim)
        value = self.to_v(hidden_states).view(batch_size, 1, self.heads, self.head_dim)
        self.k_cache.index_copy_(1, position, key)
        self.v_cache.index_copy_(1, position, value)
        attention_mask = causal_mask.index_select(0, position).view(1, 1, 1, -1)
        hidden_states = F.scaled_dot_product_attention(query.transpose(1, 2), self.k_cache.transpose(1, 2), self.v_cache.transpose(1, 2), attn_mask=attention_mask, dropout_p=0.0)
        return self.to_out(hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype))


class MiniMaxMusic3DepthDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = MiniMaxMusic3DepthAttention(dim, heads)
        self.post_attention_layernorm = RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.input_layernorm(hidden_states))
        norm_states = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.down_proj(F.silu(self.gate_proj(norm_states)) * self.up_proj(norm_states))

    def decode(self, hidden_states: torch.Tensor, position: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn.decode(self.input_layernorm(hidden_states), position, causal_mask)
        norm_states = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.down_proj(F.silu(self.gate_proj(norm_states)) * self.up_proj(norm_states))


class MiniMaxMusic3RVQDepthDecoder(ModelMixin, ConfigMixin):
    """Local language model that predicts residual RVQ codebooks within each audio frame."""

    _no_split_modules = ["MiniMaxMusic3DepthDecoderBlock"]

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        audio_vocab_size: int = 1024,
        num_codebooks: int = 8,
        max_position_embeddings: int = 16,
    ):
        super().__init__()
        self.audio_embeddings = nn.Embedding(audio_vocab_size * (num_codebooks - 1), hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList([
            MiniMaxMusic3DepthDecoderBlock(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.audio_heads = nn.ModuleList([
            nn.Linear(hidden_size, audio_vocab_size, bias=False) for _ in range(num_codebooks - 1)
        ])
        self._interrupt_check = None
        self._decode_engine = "legacy"
        self._decode_runner = None
        self._decode_causal_mask = None
        self._decode_positions = None

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if self._interrupt_check is not None and self._interrupt_check():
                raise InterruptedError("MiniMax Music 3 generation interrupted")
        return self.norm(hidden_states)

    def set_decode_engine(self, engine: str):
        self._decode_engine = engine

    def _decode_step(self, hidden_states: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        hidden_states = self.projection(hidden_states).unsqueeze(1) + self.pos_embedding(position).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer.decode(hidden_states, position, self._decode_causal_mask)
        return self.norm(hidden_states)[:, -1]

    def prepare_decode_graph(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        if self._decode_engine == "legacy":
            return
        self.release_decode_graph()
        for layer in self.layers:
            shape = (batch_size, self.config.max_position_embeddings, layer.attn.heads, layer.attn.head_dim)
            layer.attn.k_cache = torch.zeros(shape, device=device, dtype=dtype)
            layer.attn.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self._decode_positions = torch.arange(self.config.max_position_embeddings, device=device)
        self._decode_causal_mask = self._decode_positions.unsqueeze(0) <= self._decode_positions.unsqueeze(1)
        self._decode_runner = CUDAGraphRunner("minimax_music3:rvq_decode", self._decode_step)
        self._decode_step(torch.zeros((batch_size, self.config.hidden_size), device=device, dtype=dtype), self._decode_positions[:1])

    def decode_embedding(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        if self._decode_runner is None:
            raise RuntimeError("MiniMax Music 3 RVQ CUDA graph is not prepared")
        return self._decode_runner.run(hidden_states, self._decode_positions[position:position + 1])

    def release_decode_graph(self):
        if self._decode_runner is not None:
            self._decode_runner.release()
        self._decode_runner = None
        self._decode_causal_mask = None
        self._decode_positions = None
        for layer in self.layers:
            layer.attn.k_cache = layer.attn.v_cache = None
