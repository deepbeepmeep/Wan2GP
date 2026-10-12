# Transformer layers shared by the TripoSplat flow model and Gaussian decoder, ported from VAST-AI-Research/TripoSplat model.py (MIT
# License, Copyright (c) 2026 VAST, see LICENSE in this folder). Modified by DeepBeepMeep for WanGP: fused q/k/v projections split at load
# for the WanGP attention kit, in-place q/k norms and RoPE by chunks of tokens, in-place residual updates. The arithmetic follows upstream
# (FP16 checkpoint, FP32 layer norms, q/k RMS norms and RoPE products rounded to the activation dtype).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import attention_kit


def layer_norm(x, weight=None, bias=None, eps=1e-6):
    """Upstream LayerNorm32: computed in FP32, rounded to the activation dtype."""
    return F.layer_norm(x.float(), x.shape[-1:], None if weight is None else weight.float(), None if bias is None else bias.float(), eps).to(x.dtype)


def sinusoidal_embedding(t, dim, max_period, level):
    """Upstream TimestepEmbedder (level False) / LevelEmbedder (level True: angles times 2 pi) embeddings of t (batch,). The frequencies are
    computed on the CPU, as upstream builds them without a device, then moved to the device of t."""
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device="cpu") / half).to(device=t.device)
    args = t[:, None].float() * freqs[None] * 2 * torch.pi if level else t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class EmbedderMLP(nn.Module):
    """TimestepEmbedder (max_period 10000) and LevelEmbedder (max_period 1024, level True)."""

    def __init__(self, hidden_size, max_period=10000, level=False, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.max_period, self.level, self.frequency_embedding_size = max_period, level, frequency_embedding_size

    def forward(self, t):
        emb = sinusoidal_embedding(t, self.frequency_embedding_size, self.max_period, self.level)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class FeedForward(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, int(channels * mlp_ratio)), nn.GELU(approximate="tanh"), nn.Linear(int(channels * mlp_ratio), channels))

    def forward(self, x):
        return self.mlp(x)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))


def rms_norm_rows(part, gamma, scale):
    """Upstream MultiHeadRMSNorm of rows (batch, tokens, heads, head_dim): F.normalize * gamma * sqrt(head_dim) in FP32, rounded."""
    return (F.normalize(part.float(), dim=-1) * gamma * scale).to(part.dtype)


def rope_rows(part, phases):
    """Upstream apply_rotary_emb: consecutive channel pairs times complex64 phases (batch, tokens, heads, head_dim / 2), rounded."""
    rotated = torch.view_as_complex(part.float().reshape(*part.shape[:-1], -1, 2)) * phases
    return torch.view_as_real(rotated).flatten(-2).to(part.dtype)


def norm_rows_(tensor, norm, heads, phases=None):
    """In place on tensor (batch, tokens, heads of the range, head_dim) by chunks of tokens: RoPE (when phases are given) then the RMS norm,
    upstream's order."""
    gamma = norm.gamma[heads.start:heads.stop].float()
    if phases is None:
        attention_kit.rows_(tensor, lambda part, first, last: rms_norm_rows(part, gamma, norm.scale))
    else:
        phases = phases[:, :, heads.start:heads.stop]
        attention_kit.rows_(tensor, lambda part, first, last: rms_norm_rows(rope_rows(part, phases[:, first:last]), gamma, norm.scale))


class SelfAttention(nn.Module):
    """Upstream RopeMultiHeadAttention / MultiHeadAttention (self) with q/k RMS norms, its fused qkv projection split at load."""

    def __init__(self, channels, heads, names=("q", "k", "v", "q_norm", "k_norm", "out")):
        super().__init__()
        self.heads, self.head_dim, self.names = heads, channels // heads, names
        for name in names[:3] + names[5:]:
            setattr(self, name, nn.Linear(channels, channels))
        for name in names[3:5]:
            setattr(self, name, MultiHeadRMSNorm(self.head_dim, heads))

    def forward(self, x_list, phases=None):
        q, k, v, q_norm, k_norm, out = (getattr(self, name) for name in self.names)

        def norm_rope(query, key, group):
            if query is not None:
                norm_rows_(query, q_norm, group.q, phases)
            if key is not None:
                norm_rows_(key, k_norm, group.kv, phases)

        return out(attention_kit.qkv_attention(x_list, q, k, v, self.heads, self.head_dim, norm_rope).flatten(2))


class CrossAttention(nn.Module):
    """Upstream MultiHeadAttention (cross) with q/k RMS norms, its fused kv projection split at load."""

    def __init__(self, channels, ctx_channels, heads):
        super().__init__()
        self.heads, self.head_dim = heads, channels // heads
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(ctx_channels, channels)
        self.to_v = nn.Linear(ctx_channels, channels)
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x_list, context):
        all_heads = range(self.heads)
        query = self.to_q(x_list.pop()).unflatten(-1, (self.heads, self.head_dim))
        norm_rows_(query, self.q_rms_norm, all_heads)
        key = self.to_k(context).unflatten(-1, (self.heads, self.head_dim))
        norm_rows_(key, self.k_rms_norm, all_heads)
        qkv_list = [query, key, self.to_v(context).unflatten(-1, (self.heads, self.head_dim))]
        del query, key
        return self.to_out(attention_kit.kv_first_attention(qkv_list).flatten(2))
