"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from .rotary import apply_rotary_pos_emb
from shared.attention import pay_attention


def join_tokens(parts):
    """Consume separate token streams without retaining them in the caller."""
    shape = list(parts[0].shape)
    shape[1] = sum(part.shape[1] for part in parts)
    output = parts[0].new_empty(shape)
    offset = 0
    while parts:
        part = parts.pop(0)
        output[:, offset:offset + part.shape[1]].copy_(part)
        offset += part.shape[1]
        del part
    return output


def feed_forward_residual(x, norm, ff, scale, shift, gate):
    normalized = norm(x).mul_(1 + scale[:, None]).add_(shift[:, None])
    inputs = [normalized]
    normalized = None
    return x.add_(ff(inputs).mul_(gate[:, None]))


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B 1 N]
        x = x.permute(0, 2, 1)  # [B D N]

        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)  # [B N D]

        return x


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x).mul_(1 + scale_msa[:, None]).add_(shift_msa[:, None])
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNorm for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x).mul_((1 + scale)[:, None, :]).add_(shift[:, None, :])
        return x


# FeedForward (SwiGLU)


class SwiGLU(nn.Module):
    """
    Flux 2 uses a SwiGLU-style activation in the transformer feedforward sub-blocks, but with the linear projection
    layer fused into the first linear layer of the FF sub-block. Thus, this module has no trainable parameters.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1, inplace=True).mul_(x2)
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # SwiGLU will reduce the dimension by half
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.act_fn = SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=False)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x = inputs.pop()
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
    ):
        super().__init__()

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim

        self.to_qkv = nn.Linear(dim, 3 * self.inner_dim)

        self.q_norm = torch.nn.RMSNorm(dim_head, elementwise_affine=True)
        self.k_norm = torch.nn.RMSNorm(dim_head, elementwise_affine=True)

        if self.context_dim is not None:
            self.to_qkv_c = nn.Linear(context_dim, 3 * self.inner_dim)
            self.c_q_norm = torch.nn.RMSNorm(dim_head, elementwise_affine=True)
            self.c_k_norm = torch.nn.RMSNorm(dim_head, elementwise_affine=True)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))

        if self.context_dim is not None:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: list[torch.Tensor],  # consumed normalized audio
        c: list[torch.Tensor] = None,  # consumed normalized context
        mask: torch.Tensor | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
        c_mask: torch.Tensor | None = None,  # text mask
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope, c_mask=c_mask)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention processor

class AttnProcessor:
    def __init__(
        self,
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ):

        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Attention,
        x: list[torch.Tensor],  # consumed normalized input
        mask: torch.Tensor | None = None,
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        x = x.pop()
        batch_size = x.shape[0]

        # `sample` projections
        query, key, value = attn.to_qkv(x).chunk(3, dim=-1)
        del x

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        query = attn.q_norm(query)
        key = attn.k_norm(key)
        value = value.contiguous()  # Release the fused projection before FP32 rotary scratch.

        # apply rotary position embedding
        if rope is not None:
            query = apply_rotary_pos_emb(query, rope)
            key = apply_rotary_pos_emb(key, rope)

        attention_mask = mask[:, None, None, :] if self.attn_mask_enabled and mask is not None else None
        qkv_list = [query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)]
        query = key = value = None
        x = pay_attention(qkv_list, attention_mask=attention_mask, recycle_q=True)
        x = x.reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        x = attn.to_out[0](x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(
        self,
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ):

        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Attention,
        x: list[torch.Tensor],  # consumed normalized audio
        c: list[torch.Tensor] = None,  # consumed normalized text
        mask: torch.Tensor | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
        c_mask: torch.Tensor | None = None,  # text mask
    ) -> torch.FloatTensor:
        x, c = x.pop(), c.pop()
        audio_length, context_length = x.shape[1], c.shape[1]
        audio_mask = mask

        batch_size = c.shape[0]

        # `sample` projections
        query, key, value = attn.to_qkv(x).chunk(3, dim=-1)

        # `context` projections
        c_query, c_key, c_value = attn.to_qkv_c(c).chunk(3, dim=-1)
        del x, c

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        query = attn.q_norm(query)
        key = attn.k_norm(key)
        c_query = attn.c_q_norm(c_query)
        c_key = attn.c_k_norm(c_key)
        value, c_value = value.contiguous(), c_value.contiguous()

        # apply rope for context and noised input independently
        if rope is not None:
            query = apply_rotary_pos_emb(query, rope)
            key = apply_rotary_pos_emb(key, rope)
        if c_rope is not None:
            c_query = apply_rotary_pos_emb(c_query, c_rope)
            c_key = apply_rotary_pos_emb(c_key, c_rope)

        # joint attention
        query_parts = [query.transpose(1, 2), c_query.transpose(1, 2)]
        key_parts = [key.transpose(1, 2), c_key.transpose(1, 2)]
        value_parts = [value.transpose(1, 2), c_value.transpose(1, 2)]
        query = key = value = c_query = c_key = c_value = None
        qkv_list = [join_tokens(query_parts), join_tokens(key_parts), join_tokens(value_parts)]

        # build combined mask for joint attention: audio mask + text mask
        if self.attn_mask_enabled and mask is not None:
            if c_mask is not None:
                mask = torch.cat([mask, c_mask], dim=1)
            else:
                mask = F.pad(mask, (0, context_length), value=True)

        attention_mask = mask[:, None, None, :] if self.attn_mask_enabled and mask is not None else None
        x = pay_attention(qkv_list, attention_mask=attention_mask, recycle_q=True)
        x = x.reshape(batch_size, -1, attn.heads * head_dim)

        # Split the attention outputs.
        x, c = (
            x[:, :audio_length],
            x[:, audio_length:],
        )

        # linear proj
        x = attn.to_out[0](x)
        c = attn.to_out_c(c)

        if audio_mask is not None:
            x = x.masked_fill(~audio_mask.unsqueeze(-1), 0.0)
        if c_mask is not None:
            c = c.masked_fill(~c_mask.unsqueeze(-1), 0.0)

        return x, c


# DiT Block


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        attn_backend="torch",  # "torch" or "flash_attn"
        attn_mask_enabled=True,
    ):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        if isinstance(x, list):
            return [self.forward(branch, t, mask, rope) for branch in x]
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attention_inputs = [norm]
        norm = None
        attn_output = self.attn(x=attention_inputs, mask=mask, rope=rope)

        # process attention output for input x
        x.add_(attn_output.mul_(gate_msa.unsqueeze(1)))
        del attn_output
        return feed_forward_residual(x, self.ff_norm, self.ff, scale_mlp, shift_mlp, gate_mlp)


# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    """

    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        context_dim=None,
        attn_backend="torch",
        attn_mask_enabled=False,
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim

        self.attn_norm_c = AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
        )

        self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
        self.ff_c = SwiGLUFeedForward(dim=context_dim, mult=ff_mult)
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None, c_mask=None):  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        if isinstance(x, list):
            outputs = [self.forward(branch, context, t, mask, rope, c_rope, c_mask) for branch, context in zip(x, c)]
            return [one[0] for one in outputs], [one[1] for one in outputs]
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        audio_inputs, context_inputs = [norm_x], [norm_c]
        norm_x = norm_c = None
        x_attn_output, c_attn_output = self.attn(x=audio_inputs, c=context_inputs, mask=mask, rope=rope, c_rope=c_rope, c_mask=c_mask)

        # process attention output for context c
        c.add_(c_attn_output.mul_(c_gate_msa.unsqueeze(1)))
        del c_attn_output
        c = feed_forward_residual(c, self.ff_norm_c, self.ff_c, c_scale_mlp, c_shift_mlp, c_gate_mlp)

        # process attention output for input x
        x.add_(x_attn_output.mul_(x_gate_msa.unsqueeze(1)))
        del x_attn_output
        x = feed_forward_residual(x, self.ff_norm_x, self.ff_x, x_scale_mlp, x_shift_mlp, x_gate_mlp)

        return c, x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time
