"""Local inference-only SD3-Medium MMDiT for Chain-of-Zoom.

Replaces the diffusers SD3Transformer2DModel: bf16 weights stay on CPU (MMGP owns
residency), attention is routed through the shared ``pay_attention`` wrapper with
q/k/v list-handoff, modulation/residuals are applied in place and the fixed 2D
sincos positional table is generated at runtime instead of being stored in the
checkpoint. State dict key names match the diffusers layout so the merged
Chain-of-Zoom checkpoint loads unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.attention import pay_attention


SD3_MEDIUM_CONFIG = {
    "attention_head_dim": 64,
    "caption_projection_dim": 1536,
    "in_channels": 16,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 24,
    "out_channels": 16,
    "patch_size": 2,
    "pooled_projection_dim": 2048,
    "pos_embed_max_size": 192,
    "sample_size": 128,
}


def _timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    # diffusers Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    half_dim = dim // 2
    exponent = torch.arange(half_dim, dtype=torch.float32, device=timestep.device).mul_(-math.log(10000) / half_dim)
    emb = timestep.float()[:, None] * exponent.exp()[None, :]
    return torch.cat([emb.cos(), emb.sin()], dim=-1)


def _sincos_pos_embed_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0))
    out = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _sincos_pos_embed_2d(embed_dim: int, grid_size: int, base_size: int) -> np.ndarray:
    # matches the table stored in the original SD3-Medium checkpoint (SAI convention:
    # centered grid scaled by 1/4, i.e. coords = arange(192)/4 - 16), not the diffusers
    # get_2d_sincos_pos_embed default which uses arange(192)/(192/base_size)
    coords = np.arange(grid_size, dtype=np.float64) / 4.0 - base_size / 4.0
    grid = np.meshgrid(coords, coords)  # [w, h]
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _sincos_pos_embed_1d(embed_dim // 2, grid[0])
    emb_w = _sincos_pos_embed_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, pos_embed_max_size: int, base_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.base_size = base_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos_embed", torch.empty(1, pos_embed_max_size * pos_embed_max_size, embed_dim), persistent=False)

    def init_runtime_buffers(self) -> None:
        embed = _sincos_pos_embed_2d(self.proj.out_channels, self.pos_embed_max_size, self.base_size)
        self.pos_embed = torch.from_numpy(embed).float().unsqueeze(0).to(torch.bfloat16)

    def cropped_pos_embed(self, height: int, width: int) -> torch.Tensor:
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        table = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        return table[:, top:top + height, left:left + width].reshape(1, height * width, -1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        height = latent.shape[-2] // self.patch_size
        width = latent.shape[-1] // self.patch_size
        latent = self.proj(latent).flatten(2).transpose(1, 2)
        latent.add_(self.cropped_pos_embed(height, width).to(device=latent.device, dtype=latent.dtype))
        return latent


class TimestepTextEmbed(nn.Module):
    def __init__(self, embed_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.timestep_embedder = nn.Sequential()
        self.timestep_embedder.linear_1 = nn.Linear(256, embed_dim)
        self.timestep_embedder.linear_2 = nn.Linear(embed_dim, embed_dim)
        self.text_embedder = nn.Sequential()
        self.text_embedder.linear_1 = nn.Linear(pooled_projection_dim, embed_dim)
        self.text_embedder.linear_2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, timestep: torch.Tensor, pooled_projections: torch.Tensor) -> torch.Tensor:
        dtype = pooled_projections.dtype
        temb = self.timestep_embedder.linear_2(F.silu(self.timestep_embedder.linear_1(_timestep_embedding(timestep, 256).to(dtype))))
        pooled = self.text_embedder.linear_2(F.silu(self.text_embedder.linear_1(pooled_projections)))
        return temb + pooled


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear(F.silu(temb)).chunk(6, dim=1)
        x = self.norm(x)
        x.mul_(1 + scale_msa[:, None]).add_(shift_msa[:, None])
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=1)
        x = self.norm(x)
        x.mul_(1 + scale[:, None]).add_(shift[:, None])
        return x


class FeedForward(nn.Module):
    # diffusers FeedForward(activation_fn="gelu-approximate"): net.0 = GELU(proj), net.1 = Dropout, net.2 = Linear
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.net = nn.Sequential()
        gelu = nn.Sequential()
        gelu.proj = nn.Linear(dim, inner_dim)
        self.net.append(gelu)
        self.net.append(nn.Identity())
        self.net.append(nn.Linear(inner_dim, dim))

    def forward(self, x_list: list) -> torch.Tensor:
        x = x_list[0]
        x_list.clear()
        x = F.gelu(self.net[0].proj(x), approximate="tanh")
        return self.net[2](x)


class JointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, context_pre_only: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.context_pre_only = context_pre_only
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.Linear(dim, dim))
        if not context_pre_only:
            self.to_add_out = nn.Linear(dim, dim)

    def forward(self, hidden_list: list, encoder_list: list):
        hidden_states = hidden_list[0]
        encoder_states = encoder_list[0]
        hidden_list.clear()
        encoder_list.clear()
        batch, image_tokens = hidden_states.shape[:2]
        shape = (batch, -1, self.num_heads, self.head_dim)
        k = torch.cat([self.to_k(hidden_states), self.add_k_proj(encoder_states)], dim=1).view(shape)
        v = torch.cat([self.to_v(hidden_states), self.add_v_proj(encoder_states)], dim=1).view(shape)
        if self.context_pre_only:
            # the context attention output is discarded downstream, only image queries are needed
            q = self.to_q(hidden_states).view(shape)
        else:
            q = torch.cat([self.to_q(hidden_states), self.add_q_proj(encoder_states)], dim=1).view(shape)
        hidden_states = encoder_states = None
        qkv_list = [q, k, v]
        q = k = v = None
        attn = pay_attention(qkv_list).flatten(2)
        if self.context_pre_only:
            return self.to_out[0](attn), None
        return self.to_out[0](attn[:, :image_tokens]), self.to_add_out(attn[:, image_tokens:])


class JointTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, context_pre_only: bool):
        super().__init__()
        self.context_pre_only = context_pre_only
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormContinuous(dim) if context_pre_only else AdaLayerNormZero(dim)
        self.attn = JointAttention(dim, num_heads, context_pre_only)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, 4 * dim)
        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim, 4 * dim)

    def forward(self, hidden_states: torch.Tensor, encoder_states: torch.Tensor, temb: torch.Tensor):
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        if self.context_pre_only:
            norm_encoder = self.norm1_context(encoder_states, temb)
        else:
            norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_states, temb)
        norm_hidden_list, norm_encoder_list = [norm_hidden], [norm_encoder]
        norm_hidden = norm_encoder = None
        attn_output, context_attn_output = self.attn(norm_hidden_list, norm_encoder_list)

        hidden_states.addcmul_(gate_msa[:, None], attn_output)
        del attn_output
        norm_hidden = self.norm2(hidden_states)
        norm_hidden.mul_(1 + scale_mlp[:, None]).add_(shift_mlp[:, None])
        norm_hidden_list = [norm_hidden]
        norm_hidden = None
        hidden_states.addcmul_(gate_mlp[:, None], self.ff(norm_hidden_list))

        if self.context_pre_only:
            return hidden_states, None
        encoder_states.addcmul_(c_gate_msa[:, None], context_attn_output)
        del context_attn_output
        norm_encoder = self.norm2_context(encoder_states)
        norm_encoder.mul_(1 + c_scale_mlp[:, None]).add_(c_shift_mlp[:, None])
        norm_encoder_list = [norm_encoder]
        norm_encoder = None
        encoder_states.addcmul_(c_gate_mlp[:, None], self.ff_context(norm_encoder_list))
        return hidden_states, encoder_states


class SD3Transformer(nn.Module):
    def __init__(self, attention_head_dim=64, caption_projection_dim=1536, in_channels=16, joint_attention_dim=4096, num_attention_heads=24, num_layers=24, out_channels=16, patch_size=2, pooled_projection_dim=2048, pos_embed_max_size=192, sample_size=128, **kwargs):
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.pos_embed = PatchEmbed(in_channels, dim, patch_size, pos_embed_max_size, sample_size // patch_size)
        self.time_text_embed = TimestepTextEmbed(dim, pooled_projection_dim)
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)
        self.transformer_blocks = nn.ModuleList([JointTransformerBlock(dim, num_attention_heads, context_pre_only=index == num_layers - 1) for index in range(num_layers)])
        self.norm_out = AdaLayerNormContinuous(dim)
        self.proj_out = nn.Linear(dim, patch_size * patch_size * out_channels)

    def init_runtime_buffers(self) -> None:
        self.pos_embed.init_runtime_buffers()

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, pooled_projections: torch.Tensor) -> torch.Tensor:
        height, width = hidden_states.shape[-2:]
        height //= self.patch_size
        width //= self.patch_size
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_states = self.context_embedder(encoder_hidden_states)
        for block in self.transformer_blocks:
            hidden_states, encoder_states = block(hidden_states, encoder_states, temb)
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
