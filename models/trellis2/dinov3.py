# DINOv3 ViT image encoder, local inference port of Hugging Face Transformers 4.56 models/dinov3_vit/modeling_dinov3_vit.py
# (Copyright 2025 Meta AI and The HuggingFace Inc. team, Apache License 2.0), used as the TRELLIS.2 conditioning encoder
# (trellis2/modules/image_feature_extractor.py DinoV3FeatureExtractor, MIT License). The weights are covered by the DINOv3 License.
# Module names match the Transformers checkpoints; everything runs in FP32 as upstream (FP32 checkpoint and SDPA attention).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from shared.attention import pay_attention
from shared.utils.phase_progress import check_abort

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


class DINOv3Embeddings(nn.Module):
    def __init__(self, hidden_size, patch_size, num_register_tokens, num_channels=3):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, hidden_size))
        self.patch_embeddings = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        patches = self.patch_embeddings(pixel_values.to(self.patch_embeddings.weight.dtype)).flatten(2).transpose(1, 2)
        batch = pixel_values.shape[0]
        return torch.cat([self.cls_token.expand(batch, -1, -1), self.register_tokens.expand(batch, -1, -1), patches], dim=1)


def rope_cos_sin(height, width, patch_size, head_dim, theta, device):
    """DINOv3ViTRopePositionEmbedding: patch centers in [-1, 1], (y, x) angles with head_dim / 4 frequencies each, tiled twice."""
    inv_freq = 1 / theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32, device="cpu")
    patches_h, patches_w = height // patch_size, width // patch_size
    coords_h = torch.arange(0.5, patches_h, dtype=torch.float32, device=device) / patches_h
    coords_w = torch.arange(0.5, patches_w, dtype=torch.float32, device=device) / patches_w
    coords = 2.0 * torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1) - 1.0
    angles = (2 * math.pi * coords[:, :, None] * inv_freq.to(device)[None, None, :]).flatten(1, 2).tile(2)
    return torch.cos(angles), torch.sin(angles)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class DINOv3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, query_bias=True, key_bias=False, value_bias=True, proj_bias=True):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=query_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=key_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=value_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=proj_bias)

    def forward(self, x, cos, sin):
        batch, tokens, _ = x.shape
        q, k, v = (proj(x).view(batch, tokens, self.num_heads, self.head_dim) for proj in (self.q_proj, self.k_proj, self.v_proj))
        prefix = tokens - cos.shape[0]  # the class and register tokens are not rotated
        cos, sin = cos[:, None], sin[:, None]
        for tensor in (q, k):
            patches = tensor[:, prefix:]
            patches.copy_(patches * cos + _rotate_half(patches) * sin)
        qkv_list = [q, k, v]
        del q, k, v
        return self.o_proj(pay_attention(qkv_list, force_attention="sdpa", recycle_q=True).reshape(batch, tokens, -1))


class DINOv3LayerScale(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))


class DINOv3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=True):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class DINOv3Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attention = DINOv3Attention(hidden_size, num_heads)
        self.layer_scale1 = DINOv3LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = DINOv3MLP(hidden_size, intermediate_size)
        self.layer_scale2 = DINOv3LayerScale(hidden_size)

    def forward(self, x, cos, sin):
        x = self.attention(self.norm1(x), cos, sin) * self.layer_scale1.lambda1 + x
        return self.mlp(self.norm2(x)) * self.layer_scale2.lambda1 + x


class DINOv3ViT(nn.Module):
    """DINOv3ViTModel without its final norm: TRELLIS.2 conditions on a parameter-free layer norm of the last block output."""

    def __init__(self, hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16, patch_size=16,
                 num_register_tokens=4, rope_theta=100.0, layer_norm_eps=1e-5, **kwargs):
        super().__init__()
        self.patch_size, self.rope_theta, self.head_dim = patch_size, rope_theta, hidden_size // num_attention_heads
        self.embeddings = DINOv3Embeddings(hidden_size, patch_size, num_register_tokens)
        self.layer = nn.ModuleList([DINOv3Layer(hidden_size, num_attention_heads, intermediate_size, layer_norm_eps) for _ in range(num_hidden_layers)])

    def forward(self, pixel_values):
        """pixel_values (batch, 3, H, W) normalized. Returns the TRELLIS.2 features (batch, 1 + registers + patches, hidden) FP32:
        F.layer_norm of the last layer output, without the model's final norm parameters."""
        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        x = self.embeddings(pixel_values)
        cos, sin = rope_cos_sin(pixel_values.shape[2], pixel_values.shape[3], self.patch_size, self.head_dim, self.rope_theta, pixel_values.device)
        for layer in self.layer:
            x = layer(x, cos, sin)
            check_abort()
        return F.layer_norm(x, x.shape[-1:])


def image_to_pixels(image, size, device):
    """Upstream DinoV3FeatureExtractor preprocessing: Lanczos resize to size x size, [0, 1] RGB, ImageNet normalization on the device."""
    pixels = np.array(image.resize((size, size), Image.LANCZOS).convert("RGB")).astype(np.float32) / 255
    pixels = torch.from_numpy(pixels).permute(2, 0, 1).float()[None].to(device)
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32, device=device).view(-1, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32, device=device).view(-1, 1, 1)
    return pixels.sub_(mean).div_(std)
