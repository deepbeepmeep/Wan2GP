# TRELLIS.2 flow transformer, inference port of microsoft/TRELLIS.2 (trellis2/models/sparse_structure_flow.py, structured_latent_flow.py,
# trellis2/modules/{transformer,sparse/transformer,attention,sparse/attention}, MIT License, see LICENSE in this folder).
# One class serves the five flow models: the sparse-structure flow attends over its dense 16^3 grid flattened to 4096 tokens, the shape
# and texture flows over the active voxels of one sample (batch 1, so the sparse attention is a plain attention over all of them).
# Precision follows upstream: the blocks run in BF16 with FP32 norms, q/k RMS norms and RoPE; the timestep embedder, the shared
# AdaLN projection and the input/output layers compute in FP32 (upstream keeps them FP32 and loads the BF16 checkpoint into them).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import attention_kit
from shared.utils.phase_progress import check_abort
from .sparse import row_chunks

_FFN_ELEMENTS = 1 << 26  # bound of the FFN intermediate of one chunk of tokens


def fused_split_map(channels):
    """mmgp split map of the fused upstream projections: self-attention to_qkv and cross-attention to_kv become separate q/k/v linears."""
    return {"to_qkv": {"mapped_modules": ["to_q", "to_k", "to_v"], "split_sizes": [channels] * 3},
            "to_kv": {"mapped_modules": ["to_k", "to_v"], "split_sizes": [channels] * 2}}


def rope_phases(coords, head_dim, rope_freq=(1.0, 10000.0)):
    """Upstream RotaryPositionEmbedder phases (tokens, head_dim / 2) complex64 of integer voxel coords (tokens, 3), computed on the device
    of coords: per axis head_dim // 6 frequencies, the remaining pair left unrotated."""
    freq_dim = head_dim // 2 // 3
    freqs = torch.arange(freq_dim, dtype=torch.float32, device="cpu") / freq_dim
    freqs = (rope_freq[0] / (rope_freq[1] ** freqs)).to(coords.device)
    phases = torch.outer(coords.reshape(-1), freqs)
    phases = torch.polar(torch.ones_like(phases), phases).reshape(coords.shape[0], -1)
    pad = head_dim // 2 - phases.shape[-1]
    if pad:
        shape = (coords.shape[0], pad)
        phases = torch.cat([phases, torch.polar(torch.ones(shape, device=coords.device), torch.zeros(shape, device=coords.device))], dim=-1)
    return phases


def _norm_rows(part, gamma, scale, phases=None):
    """Upstream MultiHeadRMSNorm (F.normalize * gamma * sqrt(head_dim) in FP32, rounded) then RoPE (complex product in FP32, rounded)
    of rows (batch, tokens, heads, head_dim)."""
    y = (F.normalize(part.float(), dim=-1) * gamma * scale).to(part.dtype)
    if phases is None:
        return y
    y = torch.view_as_complex(y.float().reshape(*y.shape[:-1], -1, 2)) * phases.unsqueeze(-2)
    return torch.view_as_real(y).flatten(-2).to(part.dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def apply_(self, tensor, heads=None, phases=None):
        """In place on tensor (batch, tokens, heads, head_dim) by chunks of tokens; heads: the head range held by tensor."""
        gamma = self.gamma if heads is None else self.gamma[heads.start:heads.stop]
        attention_kit.rows_(tensor, lambda part, first, last: _norm_rows(part, gamma, self.scale, None if phases is None else phases[first:last]))


class SelfAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads, self.head_dim = heads, channels // heads
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x_list, phases):
        def norm_rope(query, key, group):
            if query is not None:
                self.q_rms_norm.apply_(query, group.q, phases)
            if key is not None:
                self.k_rms_norm.apply_(key, group.kv, phases)

        out = attention_kit.qkv_attention(x_list, self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, norm_rope)
        return self.to_out(out.flatten(2))


class CrossAttention(nn.Module):
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
        query = self.to_q(x_list.pop()).unflatten(-1, (self.heads, self.head_dim))
        self.q_rms_norm.apply_(query)
        key = self.to_k(context).unflatten(-1, (self.heads, self.head_dim))
        self.k_rms_norm.apply_(key)
        qkv_list = [query, key, self.to_v(context).unflatten(-1, (self.heads, self.head_dim))]
        del query, key
        return self.to_out(attention_kit.kv_first_attention(qkv_list).flatten(2))


class FeedForward(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, channels))


def _layer_norm(x, weight=None, bias=None, eps=1e-6, scale1=None, shift=None):
    """Upstream LayerNorm32 (FP32, rounded to the activation dtype), then h * (1 + scale) + shift as separate BF16 ops when given, into a
    new tensor by chunks of tokens (bounded FP32 scratch)."""
    out = torch.empty_like(x)
    weight, bias = (None, None) if weight is None else (weight.float(), bias.float())
    for start, stop in row_chunks(x.shape[1], x.shape[-1]):
        part = F.layer_norm(x[:, start:stop].float(), x.shape[-1:], weight, bias, eps).to(x.dtype)
        out[:, start:stop] = part if scale1 is None else part.mul_(scale1).add_(shift)
    return out


class TransformerBlock(nn.Module):
    """ModulatedTransformerCrossBlock / ModulatedSparseTransformerCrossBlock with a shared AdaLN modulation and a per-block offset."""

    def __init__(self, channels, ctx_channels, heads, mlp_ratio):
        super().__init__()
        self.hidden = int(channels * mlp_ratio)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        self.self_attn = SelfAttention(channels, heads)
        self.cross_attn = CrossAttention(channels, ctx_channels, heads)
        self.mlp = FeedForward(channels, self.hidden)
        self.modulation = nn.Parameter(torch.zeros(6 * channels))

    def forward(self, x, mod, context, phases):
        """x (1, tokens, channels) BF16, updated in place; mod (1, 6 * channels) BF16."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation.float() + mod).to(mod.dtype).chunk(6, dim=1)
        scale_msa, scale_mlp = 1 + scale_msa, 1 + scale_mlp
        # each attention input is handed off in its list: the attention releases it after its projections
        x.add_(self.self_attn([_layer_norm(x, scale1=scale_msa, shift=shift_msa)], phases).mul_(gate_msa))
        x.add_(self.cross_attn([_layer_norm(x, self.norm2.weight, self.norm2.bias, self.norm2.eps)], context))
        # the FFN by chunks of tokens: norm, modulation, MLP and gate are row-wise, so one chunk of the intermediate exists at a time
        for start, stop in row_chunks(x.shape[1], self.hidden, _FFN_ELEMENTS):
            part = x[:, start:stop]
            part.add_(self.mlp.mlp(_layer_norm(part, scale1=scale_mlp, shift=shift_mlp)).mul_(gate_mlp))
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device="cpu") / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class TrellisFlowDiT(nn.Module):
    """SparseStructureFlowModel / SLatFlowModel (share_mod, QK RMS norms, 3D RoPE) over the tokens of one sample."""

    def __init__(self, in_channels, out_channels, model_channels=1536, cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
                 rope_freq=(1.0, 10000.0), resolution=None):
        super().__init__()
        self.in_channels, self.out_channels, self.resolution = in_channels, out_channels, resolution
        self.head_dim, self.rope_freq = model_channels // num_heads, tuple(rope_freq)
        self.t_embedder = TimestepEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList([TransformerBlock(model_channels, cond_channels, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.out_layer = nn.Linear(model_channels, out_channels)

    def phases(self, coords):
        return rope_phases(coords, self.head_dim, self.rope_freq)

    def forward(self, x, t, contexts, phases):
        """x (1, tokens, in_channels) FP32, t (1,) FP32 timestep in [0, 1000], contexts: list of (1, cond tokens, cond_channels) BF16
        conditionings (CFG branches, joint pass: each block runs every branch while its weights are loaded), phases (tokens, head_dim / 2)
        complex64. Returns one (1, tokens, out_channels) FP32 prediction per context."""
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            h = self.input_layer(x).to(torch.bfloat16)
            mod = self.adaLN_modulation(self.t_embedder(t)).to(torch.bfloat16)
        hidden = [h] + [h.clone() for _ in contexts[1:]]
        del h
        for block in self.blocks:
            for h, context in zip(hidden, contexts):
                block(h, mod, context, phases)
            check_abort()
        outputs = []
        while hidden:
            h = hidden.pop(0)
            out = h.new_empty((*h.shape[:-1], self.out_channels), dtype=torch.float32)
            with torch.autocast(device_type=x.device.type, dtype=torch.float32):
                for start, stop in row_chunks(h.shape[1], h.shape[-1]):
                    part = h[:, start:stop].float()
                    out[:, start:stop] = self.out_layer(F.layer_norm(part, part.shape[-1:]))
            outputs.append(out)
            del h
        return outputs
