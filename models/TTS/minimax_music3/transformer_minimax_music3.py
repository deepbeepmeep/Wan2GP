# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WanGP-local port of Diffusers' MiniMax Music 3 flow transformer."""

import math
from functools import lru_cache

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from shared.attention import pay_attention


class MiniMaxMusic3FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_dim // 2, 1))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * timestep.unsqueeze(-1) @ self.weight.T
        return torch.cat((angles.cos(), angles.sin()), dim=-1)


class MiniMaxMusic3RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, theta: float = 10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.theta = theta

    @lru_cache(maxsize=32)
    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, device=device).float() / self.rotary_dim))
        freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos().contiguous(), freqs.sin().contiguous()


def _apply_partial_rotary_emb(hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rotary_emb
    rotary_dim = cos.shape[-1]
    cos, sin = cos[:, None, :].to(hidden_states.dtype), sin[:, None, :].to(hidden_states.dtype)
    rotated = hidden_states[..., :rotary_dim]
    first, second = rotated.chunk(2, dim=-1)
    half = first.shape[-1]
    first_out = first * cos[..., :half] - second * sin[..., :half]
    second_out = second * cos[..., half:] + first * sin[..., half:]
    first.copy_(first_out)
    second.copy_(second_out)
    return hidden_states


class MiniMaxMusic3Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])

    def forward(self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        key = self.to_k(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        value = self.to_v(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        query = _apply_partial_rotary_emb(query, rotary_emb)
        key = _apply_partial_rotary_emb(key, rotary_emb)
        hidden_states = pay_attention([query, key, value], causal=False, recycle_q=True)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return self.to_out[1](self.to_out[0](hidden_states))


class MiniMaxMusic3TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MiniMaxMusic3Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2)
        self.ff_out = nn.Linear(ff_inner_dim, dim)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_emb)
        gate_states, gate = self.ff_in(self.norm2(hidden_states)).chunk(2, dim=-1)
        return hidden_states + self.ff_out(gate_states * torch.nn.functional.silu(gate))


class MiniMaxMusic3Transformer1DModel(ModelMixin, ConfigMixin):
    """Flow-matching transformer that turns Music 3 frame conditioning into audio latents."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxMusic3TransformerBlock"]
    _repeated_blocks = ["MiniMaxMusic3TransformerBlock"]
    _skip_layerwise_casting_patterns = ["time_proj", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        condition_dim: int = 2048,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        ff_inner_dim: int = 8192,
        rotary_dim: int = 32,
        fourier_embedding_dim: int = 256,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        concat_channels = 2 * in_channels + condition_dim
        self.time_proj = MiniMaxMusic3FourierEmbedding(fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(fourier_embedding_dim, inner_dim)
        self.preprocess_conv = nn.Conv1d(concat_channels, concat_channels, 1, bias=False)
        self.proj_in = nn.Linear(concat_channels, inner_dim, bias=False)
        self.rotary_emb = MiniMaxMusic3RotaryEmbedding(rotary_dim)
        self.transformer_blocks = nn.ModuleList([
            MiniMaxMusic3TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, ff_inner_dim)
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.gradient_checkpointing = False
        self._interrupt_check = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        residual = torch.cat((hidden_states, torch.zeros_like(hidden_states), encoder_hidden_states.transpose(1, 2)), dim=1)
        hidden_states = self.preprocess_conv(residual) + residual
        del residual
        hidden_states = hidden_states.transpose(1, 2)
        temb = self.time_embed(self.time_proj(timestep))
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat((temb.unsqueeze(1), hidden_states), dim=1)
        rotary_emb = self.rotary_emb(hidden_states.shape[1], hidden_states.device)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)
            if self._interrupt_check is not None and self._interrupt_check():
                raise InterruptedError("MiniMax Music 3 generation interrupted")
        hidden_states = self.proj_out(hidden_states[:, 1:]).transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states
        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)
