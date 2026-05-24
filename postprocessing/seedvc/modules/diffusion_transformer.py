import torch
from torch import nn
import math

# from .torchscript_modules.gpt_fast_model import ModelArgs, Transformer
from .wavenet import WN
from .commons import sequence_mask

from torch.nn.utils import weight_norm

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if embedding is None:
            return self.norm(input)
        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


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
    has_cross_attention: bool = False
    context_dim: int = 0
    uvit_skip_connection: bool = False
    time_as_token: bool = False

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
        self.norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length, use_kv_cache=False):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.head_dim,
                                              self.config.rope_base, dtype).to(device)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).to(device)
        self.use_kv_cache = use_kv_cache
        self.uvit_skip_connection = self.config.uvit_skip_connection
        if self.uvit_skip_connection:
            self.layers_emit_skip = [i for i in range(self.config.n_layer) if i < self.config.n_layer // 2]
            self.layers_receive_skip = [i for i in range(self.config.n_layer) if i > self.config.n_layer // 2]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []

    def forward(self,
                x: Tensor,
                c: Tensor,
                input_pos: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                context: Optional[Tensor] = None,
                context_input_pos: Optional[Tensor] = None,
                cross_attention_mask: Optional[Tensor] = None,
                ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if mask is None: # in case of non-causal model
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        if context is not None:
            context_freqs_cis = self.freqs_cis[context_input_pos]
        else:
            context_freqs_cis = None
        skip_in_x_list = []
        for i, layer in enumerate(self.layers):
            if self.uvit_skip_connection and i in self.layers_receive_skip:
                skip_in_x = skip_in_x_list.pop(-1)
            else:
                skip_in_x = None
            x = layer(x, c, input_pos, freqs_cis, mask, context, context_freqs_cis, cross_attention_mask, skip_in_x)
            if self.uvit_skip_connection and i in self.layers_emit_skip:
                skip_in_x_list.append(x)
        x = self.norm(x, c)
        return x

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.attention_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))

        if config.has_cross_attention:
            self.has_cross_attention = True
            self.cross_attention = Attention(config, is_cross_attention=True)
            self.cross_attention_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        else:
            self.has_cross_attention = False

        if config.uvit_skip_connection:
            self.skip_in_linear = nn.Linear(config.dim * 2, config.dim)
            self.uvit_skip_connection = True
        else:
            self.uvit_skip_connection = False

        self.time_as_token = config.time_as_token

    def forward(self,
                x: Tensor,
                c: Tensor,
                input_pos: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                context: Optional[Tensor] = None,
                context_freqs_cis: Optional[Tensor] = None,
                cross_attention_mask: Optional[Tensor] = None,
                skip_in_x: Optional[Tensor] = None,
                ) -> Tensor:
        c = None if self.time_as_token else c
        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask, input_pos)
        if self.has_cross_attention:
            h = h + self.cross_attention(self.cross_attention_norm(h, c), freqs_cis, cross_attention_mask, input_pos, context, context_freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h, c))
        return out


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
        # self._register_load_state_dict_pre_hook(self.load_hook)

    # def load_hook(self, state_dict, prefix, *args):
    #     if prefix + "wq.weight" in state_dict:
    #         wq = state_dict.pop(prefix + "wq.weight")
    #         wk = state_dict.pop(prefix + "wk.weight")
    #         wv = state_dict.pop(prefix + "wv.weight")
    #         state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self,
                x: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                input_pos: Optional[Tensor] = None,
                context: Optional[Tensor] = None,
                context_freqs_cis: Optional[Tensor] = None,
                ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        if context is None:
            q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
            context_seqlen = seqlen
        else:
            q = self.wq(x)
            k, v = self.wkv(context).split([kv_size, kv_size], dim=-1)
            context_seqlen = context.shape[1]

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, context_freqs_cis if context_freqs_cis is not None else freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

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

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.scale = 1000

        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        args = self.scale * t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class StyleEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, input_size, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(int(use_cfg_embedding), hidden_size)
        self.style_in = weight_norm(nn.Linear(input_size, hidden_size, bias=True))
        self.input_size = input_size
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        else:
            labels = self.style_in(labels)
        embeddings = labels
        return embeddings

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(torch.nn.Module):
    def __init__(
        self,
        args
    ):
        super(DiT, self).__init__()
        self.time_as_token = args.DiT.time_as_token if hasattr(args.DiT, 'time_as_token') else False
        self.style_as_token = args.DiT.style_as_token if hasattr(args.DiT, 'style_as_token') else False
        self.uvit_skip_connection = args.DiT.uvit_skip_connection if hasattr(args.DiT, 'uvit_skip_connection') else False
        model_args = ModelArgs(
            block_size=16384,#args.DiT.block_size,
            n_layer=args.DiT.depth,
            n_head=args.DiT.num_heads,
            dim=args.DiT.hidden_dim,
            head_dim=args.DiT.hidden_dim // args.DiT.num_heads,
            vocab_size=1024,
            uvit_skip_connection=self.uvit_skip_connection,
            time_as_token=self.time_as_token,
        )
        self.transformer = Transformer(model_args)
        self.in_channels = args.DiT.in_channels
        self.out_channels = args.DiT.in_channels
        self.num_heads = args.DiT.num_heads

        self.x_embedder = weight_norm(nn.Linear(args.DiT.in_channels, args.DiT.hidden_dim, bias=True))

        self.content_type = args.DiT.content_type  # 'discrete' or 'continuous'
        self.content_codebook_size = args.DiT.content_codebook_size # for discrete content
        self.content_dim = args.DiT.content_dim # for continuous content
        self.cond_embedder = nn.Embedding(args.DiT.content_codebook_size, args.DiT.hidden_dim)  # discrete content
        self.cond_projection = nn.Linear(args.DiT.content_dim, args.DiT.hidden_dim, bias=True) # continuous content

        self.is_causal = args.DiT.is_causal

        self.t_embedder = TimestepEmbedder(args.DiT.hidden_dim)

        input_pos = torch.arange(16384)
        self.register_buffer("input_pos", input_pos)

        self.final_layer_type = args.DiT.final_layer_type  # mlp or wavenet
        if self.final_layer_type == 'wavenet':
            self.t_embedder2 = TimestepEmbedder(args.wavenet.hidden_dim)
            self.conv1 = nn.Linear(args.DiT.hidden_dim, args.wavenet.hidden_dim)
            self.conv2 = nn.Conv1d(args.wavenet.hidden_dim, args.DiT.in_channels, 1)
            self.wavenet = WN(hidden_channels=args.wavenet.hidden_dim,
                              kernel_size=args.wavenet.kernel_size,
                              dilation_rate=args.wavenet.dilation_rate,
                              n_layers=args.wavenet.num_layers,
                              gin_channels=args.wavenet.hidden_dim,
                              p_dropout=args.wavenet.p_dropout,
                              causal=False)
            self.final_layer = FinalLayer(args.wavenet.hidden_dim, 1, args.wavenet.hidden_dim)
            self.res_projection = nn.Linear(args.DiT.hidden_dim,
                                            args.wavenet.hidden_dim)  # residual connection from tranformer output to final output
            self.wavenet_style_condition = args.wavenet.style_condition
            assert args.DiT.style_condition == args.wavenet.style_condition
        else:
            self.final_mlp = nn.Sequential(
                    nn.Linear(args.DiT.hidden_dim, args.DiT.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(args.DiT.hidden_dim, args.DiT.in_channels),
            )
        self.transformer_style_condition = args.DiT.style_condition


        self.class_dropout_prob = args.DiT.class_dropout_prob
        self.content_mask_embedder = nn.Embedding(1, args.DiT.hidden_dim)

        self.long_skip_connection = args.DiT.long_skip_connection
        self.skip_linear = nn.Linear(args.DiT.hidden_dim + args.DiT.in_channels, args.DiT.hidden_dim)

        self.cond_x_merge_linear = nn.Linear(args.DiT.hidden_dim + args.DiT.in_channels * 2 +
                                             args.style_encoder.dim * self.transformer_style_condition * (not self.style_as_token),
                                             args.DiT.hidden_dim)
        if self.style_as_token:
            self.style_in = nn.Linear(args.style_encoder.dim, args.DiT.hidden_dim)

    def setup_caches(self, max_batch_size, max_seq_length):
        self.transformer.setup_caches(max_batch_size, max_seq_length, use_kv_cache=False)
    def forward(self, x, prompt_x, x_lens, t, style, cond, mask_content=False):
        class_dropout = False
        if self.training and torch.rand(1) < self.class_dropout_prob:
            class_dropout = True
        if not self.training and mask_content:
            class_dropout = True
        # cond_in_module = self.cond_embedder if self.content_type == 'discrete' else self.cond_projection
        cond_in_module = self.cond_projection

        B, _, T = x.size()


        t1 = self.t_embedder(t)  # (N, D)

        cond = cond_in_module(cond)

        x = x.transpose(1, 2)
        prompt_x = prompt_x.transpose(1, 2)

        x_in = torch.cat([x, prompt_x, cond], dim=-1)
        if self.transformer_style_condition and not self.style_as_token:
            x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], dim=-1)
        if class_dropout:
            x_in[..., self.in_channels:] = x_in[..., self.in_channels:] * 0
        x_in = self.cond_x_merge_linear(x_in)  # (N, T, D)

        if self.style_as_token:
            style = self.style_in(style)
            style = torch.zeros_like(style) if class_dropout else style
            x_in = torch.cat([style.unsqueeze(1), x_in], dim=1)
        if self.time_as_token:
            x_in = torch.cat([t1.unsqueeze(1), x_in], dim=1)
        x_mask = sequence_mask(x_lens + self.style_as_token + self.time_as_token).to(x.device).unsqueeze(1)
        input_pos = self.input_pos[:x_in.size(1)]  # (T,)
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1) if not self.is_causal else None
        x_res = self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)
        x_res = x_res[:, 1:] if self.time_as_token else x_res
        x_res = x_res[:, 1:] if self.style_as_token else x_res
        if self.long_skip_connection:
            x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))
        if self.final_layer_type == 'wavenet':
            x = self.conv1(x_res)
            x = x.transpose(1, 2)
            t2 = self.t_embedder2(t)
            x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + self.res_projection(
                x_res)  # long residual connection
            x = self.final_layer(x, t1).transpose(1, 2)
            x = self.conv2(x)
        else:
            x = self.final_mlp(x_res)
            x = x.transpose(1, 2)
        return x