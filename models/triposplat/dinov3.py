# DINOv3 ViT-H+/16 image encoder of TripoSplat, ported from VAST-AI-Research/TripoSplat model.py (DinoV3ViT; MIT License, Copyright (c)
# 2026 VAST, see LICENSE in this folder), which follows Hugging Face Transformers' DINOv3 ViT. The weights are Meta's DINOv3 ViT-H+/16
# LVD-1689M, covered by the DINOv3 License (LICENSE_DINOv3.md beside the checkpoint).
# Modified by DeepBeepMeep for WanGP: Transformers checkpoint module names (no key remapping), attention through the WanGP attention kit
# with in-place RoPE by chunks of tokens, abort checks. The arithmetic follows upstream (BF16 checkpoint, BF16 RoPE tables).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import attention_kit
from shared.utils.phase_progress import check_abort


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_tables(height, width, head_dim, theta, dtype, device):
    """Upstream DinoV3RotaryEmbedding2D: patch centres in [-1, 1], (y, x) angles with head_dim / 4 frequencies each, tiled twice. The
    frequencies are rounded to the compute dtype as upstream does (its inverse frequency buffer follows the model's .to(dtype))."""
    inv_freq = (1.0 / (theta ** torch.arange(0, 1, 4.0 / head_dim, dtype=torch.float32, device="cpu"))).to(dtype).to(device)
    coords_h = torch.arange(0.5, height, dtype=torch.float32, device=device) / height
    coords_w = torch.arange(0.5, width, dtype=torch.float32, device=device) / width
    coords = (2.0 * torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1) - 1.0).flatten(0, 1)
    angles = (2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)
    return angles.cos().to(dtype), angles.sin().to(dtype)


class DINOv3Embeddings(nn.Module):
    def __init__(self, hidden_size, patch_size, num_register_tokens):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))  # masked image modelling only, kept for the checkpoint layout
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, hidden_size))
        self.patch_embeddings = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)


class DINOv3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.heads, self.head_dim = num_heads, hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_list, cos, sin, prefix):
        """x handed off in x_list (batch, tokens, hidden); the patch tokens after the prefix (class and register tokens) are rotated."""
        def rope(part, first, last):
            c, s = cos[first - prefix:last - prefix, None], sin[first - prefix:last - prefix, None]
            return part * c + _rotate_half(part) * s

        def norm_rope(query, key, group):
            for tensor in (query, key):
                if tensor is not None:
                    attention_kit.rows_(tensor, rope, start=prefix)

        return self.o_proj(attention_kit.qkv_attention(x_list, self.q_proj, self.k_proj, self.v_proj, self.heads, self.head_dim, norm_rope).flatten(2))


class DINOv3LayerScale(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))


class DINOv3GatedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DINOv3Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attention = DINOv3Attention(hidden_size, num_heads)
        self.layer_scale1 = DINOv3LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = DINOv3GatedMLP(hidden_size, intermediate_size)
        self.layer_scale2 = DINOv3LayerScale(hidden_size)

    def forward(self, x, cos, sin, prefix):
        x = x + self.layer_scale1.lambda1 * self.attention([self.norm1(x)], cos, sin, prefix)
        return x + self.layer_scale2.lambda1 * self.mlp(self.norm2(x))


class DINOv3ViT(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, patch_size, num_register_tokens, intermediate_size, rope_theta, layer_norm_eps):
        super().__init__()
        self.patch_size, self.rope_theta, self.head_dim = patch_size, rope_theta, hidden_size // num_heads
        self.prefix = 1 + num_register_tokens
        self.embeddings = DINOv3Embeddings(hidden_size, patch_size, num_register_tokens)
        self.layer = nn.ModuleList([DINOv3Layer(hidden_size, num_heads, intermediate_size, layer_norm_eps) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, pixel_values):
        """pixel_values (batch, 3, H, W) ImageNet-normalized FP32. Returns the TripoSplat image features (batch, 1 + registers + patches,
        hidden) FP32: upstream encode_image's parameter-free layer norm of the model output (after its final norm)."""
        embeddings = self.embeddings
        x = embeddings.patch_embeddings(pixel_values.to(embeddings.patch_embeddings.weight.dtype)).flatten(2).transpose(1, 2)
        batch, _, height, width = pixel_values.shape
        cos, sin = rope_tables(height // self.patch_size, width // self.patch_size, self.head_dim, self.rope_theta, x.dtype, x.device)
        x = torch.cat([embeddings.cls_token.expand(batch, -1, -1), embeddings.register_tokens.expand(batch, -1, -1), x], dim=1)
        for layer in self.layer:
            x = layer(x, cos, sin, self.prefix)
            check_abort()
        x = self.norm(x)
        return F.layer_norm(x.float(), x.shape[-1:])
