# DINOv2 image encoder (Oquab et al., facebookresearch/dinov2), the architecture and parameter names of Hugging Face transformers'
# Dinov2Model (Apache-2.0, Copyright Meta Platforms and The HuggingFace Inc. team), reimplemented for WanGP: self-attention through
# shared.attention_kit, inference only. Hunyuan3D 2.1 conditions its shape DiT on DINOv2-L (518 px) and its paint UNet on DINOv2-giant
# (224 px); both read last_hidden_state (the final LayerNorm output, class token first).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn.functional as F
from torch import nn

from shared import attention_kit


class Dinov2Embeddings(nn.Module):
    def __init__(self, hidden_size, patch_size, num_positions):
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.empty(1, num_positions + 1, hidden_size))
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.projection = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)

    def position_table(self, height, width):
        """transformers' interpolate_pos_encoding: the 37x37 table resized (bicubic, FP32) when the patch grid differs."""
        positions = self.position_embeddings
        grid = positions.shape[1] - 1
        new_height, new_width = height // self.patch_size, width // self.patch_size
        if new_height * new_width == grid and height == width:
            return positions
        side = int(grid ** 0.5)
        patch = positions[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch.to(torch.float32), size=(new_height, new_width), mode="bicubic", align_corners=False).to(positions.dtype)
        return torch.cat((positions[:, :1], patch.permute(0, 2, 3, 1).reshape(1, -1, positions.shape[-1])), dim=1)

    def forward(self, pixel_values):
        projection = self.patch_embeddings.projection
        embeddings = projection(pixel_values.to(projection.weight.dtype)).flatten(2).transpose(1, 2)
        embeddings = torch.cat((self.cls_token.expand(embeddings.shape[0], -1, -1), embeddings), dim=1)
        return embeddings + self.position_table(pixel_values.shape[-2], pixel_values.shape[-1])


class Dinov2Layer(nn.Module):
    def __init__(self, hidden_size, heads, swiglu, eps):
        super().__init__()
        self.heads, self.head_dim = heads, hidden_size // heads
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attention = nn.Module()
        self.attention.attention = nn.Module()
        for name in ("query", "key", "value"):
            setattr(self.attention.attention, name, nn.Linear(hidden_size, hidden_size))
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_scale1 = nn.Module()
        self.layer_scale1.lambda1 = nn.Parameter(torch.empty(hidden_size))
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = nn.Module()
        if swiglu:
            hidden = (int(hidden_size * 4 * 2 / 3) + 7) // 8 * 8
            self.mlp.weights_in = nn.Linear(hidden_size, 2 * hidden)
            self.mlp.weights_out = nn.Linear(hidden, hidden_size)
        else:
            self.mlp.fc1 = nn.Linear(hidden_size, hidden_size * 4)
            self.mlp.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.swiglu = swiglu
        self.layer_scale2 = nn.Module()
        self.layer_scale2.lambda1 = nn.Parameter(torch.empty(hidden_size))

    def forward(self, hidden_states):
        attention = self.attention.attention
        out = attention_kit.qkv_attention([self.norm1(hidden_states)], attention.query, attention.key, attention.value, self.heads, self.head_dim, lambda query, key, group: None)
        hidden_states = self.attention.output.dense(out.flatten(2)).mul_(self.layer_scale1.lambda1).add_(hidden_states)
        del out
        x = self.norm2(hidden_states)
        if self.swiglu:
            x1, x2 = self.mlp.weights_in(x).chunk(2, dim=-1)
            x = self.mlp.weights_out(F.silu(x1) * x2)
        else:
            x = self.mlp.fc2(F.gelu(self.mlp.fc1(x)))
        return x.mul_(self.layer_scale2.lambda1).add_(hidden_states)


class Dinov2Model(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads, use_swiglu_ffn=False, patch_size=14, image_size=518, layer_norm_eps=1e-6, **_):
        super().__init__()
        self.embeddings = Dinov2Embeddings(hidden_size, patch_size, (image_size // patch_size) ** 2)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([Dinov2Layer(hidden_size, num_attention_heads, use_swiglu_ffn, layer_norm_eps) for _ in range(num_hidden_layers)])
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder.layer:
            hidden_states = layer(hidden_states)
        return self.layernorm(hidden_states)
