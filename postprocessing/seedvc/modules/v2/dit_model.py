# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import time

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.linear = nn.Linear(d_model, 6 * d_model)
        self.act = nn.SiLU()
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, x: Tensor, emb: Tensor) -> Tuple[Tensor]:
        emb = self.linear(self.act(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=-1)

        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaptiveLayerNormFinal(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNormFinal, self).__init__()
        self.linear = nn.Linear(d_model, 2 * d_model)
        self.act = nn.SiLU()
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, x: Tensor, emb: Tensor) -> Tuple[Tensor]:
        emb = self.linear(self.act(emb))
        scale, shift = torch.chunk(emb, 2, dim=-1)

        x = self.norm(x) * (1 + scale) + shift
        return x

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    uvit_skip_connection: bool = False
    time_as_token: bool = False
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        # self.head_dim = self.dim // self.n_head

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = AdaptiveLayerNormFinal(config.dim, RMSNorm(config.dim, eps=config.norm_eps))

        self.max_batch_size = -1
        self.max_seq_length = config.block_size

        self.uvit_skip_connection = self.config.uvit_skip_connection
        if self.uvit_skip_connection:
            self.layers_emit_skip = [i for i in range(self.config.n_layer) if i < self.config.n_layer // 2]
            self.layers_receive_skip = [i for i in range(self.config.n_layer) if i > self.config.n_layer // 2]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []
        freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.head_dim,
                                              self.config.rope_base)
        self.register_buffer("freqs_cis", freqs_cis)

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self,
                x: Tensor,
                c: Tensor,
                input_pos: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                ) -> Tensor:
        mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        for i, layer in enumerate(self.layers):
            x = layer(x, c, freqs_cis, mask)
        x = self.norm(x, c)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
    def forward(self,
                x: Tensor,
                c: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                ) -> Tensor:
        normed_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attention_norm(x, emb=c)
        # attention
        attn_output = self.attention(normed_x, freqs_cis, mask)
        x = x + gate_msa * attn_output
        normed_x = self.ffn_norm(x) * (1 + scale_mlp) + shift_mlp
        ff_output = self.feed_forward(normed_x)
        x = x + gate_mlp * ff_output
        return x


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, is_cross_attention: bool = False):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        if is_cross_attention:
            self.wq = nn.Linear(config.dim, config.n_head * config.head_dim, bias=False)
            self.wkv = nn.Linear(config.context_dim, 2 * config.n_local_heads * config.head_dim, bias=False)
        else:
            self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attn_dropout_rate = config.attn_dropout_rate

    def forward(self,
                x: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                context: Optional[Tensor] = None,
                context_freqs_cis: Optional[Tensor] = None,
                ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
        context_seqlen = seqlen

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
        seq_len: int, n_elem: int, base: int = 10000,
        dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

