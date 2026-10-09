# DINOv2 vision transformer (Oquab et al., facebookresearch/dinov2, Apache-2.0, Copyright Meta Platforms, Inc. and affiliates; vendored by
# microsoft/MoGe, see LICENSE_MoGe) in its torch hub parameter layout, reimplemented for WanGP (inference only, modified): self-attention
# goes through shared.attention_kit, the fused qkv projection being split into q/k/v at load (QKV_SPLIT_MAP). MoGe-2 uses ViT-L/14,
# MapAnything the first 24 blocks of ViT-g/14; MapAnything's alternating-attention transformer (UniCeption, BSD-3-Clause, see
# LICENSE_UniCeption) has the same block layout and reuses ViTBlock.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn.functional as F
from torch import nn

from shared import attention_kit

QKV_SPLIT_MAP = {"qkv": {"mapped_modules": ["q", "k", "v"]}}


def _no_norm_rope(query, key, group):
    pass


class ViTBlock(nn.Module):
    """x + ls1 * attn(norm1(x)), then x + ls2 * mlp(norm2(x)), the MLP being GELU (fc1/fc2) or fused SwiGLU (w12/w3). The LayerScale
    products are not computed in place: under autocast they promote the 16-bit branch to the FP32 residual stream as upstream."""

    def __init__(self, dim, heads, swiglu=False, eps=1e-6):
        super().__init__()
        self.heads, self.head_dim, self.swiglu = heads, dim // heads, swiglu
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = nn.Module()
        for name in ("q", "k", "v", "proj"):
            setattr(self.attn, name, nn.Linear(dim, dim))
        self.ls1, self.ls2 = nn.Module(), nn.Module()
        self.ls1.gamma, self.ls2.gamma = nn.Parameter(torch.empty(dim)), nn.Parameter(torch.empty(dim))
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = nn.Module()
        if swiglu:
            hidden = (int(dim * 4 * 2 / 3) + 7) // 8 * 8
            self.mlp.w12, self.mlp.w3 = nn.Linear(dim, 2 * hidden), nn.Linear(hidden, dim)
        else:
            self.mlp.fc1, self.mlp.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    @staticmethod
    def _linear_input(h):
        """Norm output in the autocast dtype, as autocast casts the input of a linear: same result for floating point weights, and the
        quantized (INT8) linears, which autocast does not cast, also compute in 16 bits."""
        return h.to(torch.get_autocast_dtype(h.device.type)) if torch.is_autocast_enabled(h.device.type) else h

    def forward(self, x):
        attn = self.attn
        out = attention_kit.qkv_attention([self._linear_input(self.norm1(x))], attn.q, attn.k, attn.v, self.heads, self.head_dim, _no_norm_rope)
        x = x + attn.proj(out.flatten(2)) * self.ls1.gamma
        del out
        h = self._linear_input(self.norm2(x))
        if self.swiglu:
            h1, h2 = self.mlp.w12(h).chunk(2, dim=-1)
            h = self.mlp.w3(F.silu(h1) * h2)
        else:
            h = self.mlp.fc2(F.gelu(self.mlp.fc1(h)))
        return x + h * self.ls2.gamma


class DinoV2(nn.Module):
    """Patch embedding, class token, 518 px position table (37 x 37 patches) and blocks; the final norm only when the model uses it."""

    def __init__(self, dim, depth, heads, swiglu=False, norm=True, patch_size=14, grid=37):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.empty(1, grid * grid + 1, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, swiglu) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6) if norm else None

    def position_table(self, rows, cols, dtype):
        """Upstream interpolate_pos_encoding: the table as stored for a square 37 x 37 grid, otherwise resized in FP32 (bicubic, scale
        factors with the 0.1 offset, no antialiasing) and cast to the token dtype."""
        side = math.isqrt(self.pos_embed.shape[1] - 1)
        if rows == cols == side:
            return self.pos_embed
        table = self.pos_embed.float()
        patch = table[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, scale_factor=((rows + 0.1) / side, (cols + 0.1) / side), mode="bicubic", antialias=False)
        return torch.cat([table[:, :1], patch.permute(0, 2, 3, 1).flatten(1, 2)], dim=1).to(dtype)

    def forward(self, pixels, take, progress=None):
        """Outputs of the blocks listed in take (B, 1 + rows * cols, dim), class token first; progress() after each block."""
        batch, _, height, width = pixels.shape
        x = self.patch_embed.proj(pixels).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(batch, -1, -1), x], dim=1)
        x = x + self.position_table(height // self.patch_size, width // self.patch_size, x.dtype)
        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in take:
                outputs.append(x)
            if progress is not None:
                progress()
        return outputs
