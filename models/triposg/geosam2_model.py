# GeoSAM2 network: SAM 2.1 (Hiera-B+) with a second Hiera encoder for position maps, LoRA adapters and feature fusion, ported from
# VAST-AI-Research/GeoSAM2 sam2/modeling/{sam2_base_geosam2.py, backbones/hieradet.py, backbones/image_encoder.py, backbones/utils.py,
# sam/lora.py, sam/mask_decoder.py, sam/prompt_encoder.py, sam/transformer.py, memory_attention.py, memory_encoder.py,
# position_encoding.py, feature_fusion.py, sam2_utils.py} (Copyright (c) Meta Platforms, Inc. and affiliates; modifications copyright
# 2025 VAST-AI-Research and the GeoSAM2 authors; Apache License 2.0, see LICENSE_GeoSAM2 and NOTICE_GeoSAM2 in this folder).
# Modified by DeepBeepMeep for WanGP: inference only, built in Python from geosam2.json instead of Hydra; attention through pay_attention
# (SDPA, whose head sizes 56/16/256 the quantized backends do not take); tensors created on explicit devices; the training paths, the
# unused SAM2 options and the image/video predictor plumbing removed (the propagation logic lives in geosam2_main.py).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from shared.attention import pay_attention

NO_OBJ_SCORE = -1024.0


def _attention(q, k, v):
    """SDPA through pay_attention on (batch, heads, tokens, head_dim) tensors."""
    qkv_list = [q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)]
    q = k = v = None
    return pay_attention(qkv_list, force_attention="sdpa").transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU, sigmoid_output=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return torch.sigmoid(x) if self.sigmoid_output else x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ------------------------------------------------------------------------------------------------ position encodings

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats, temperature=10000):
        super().__init__()
        self.num_pos_feats, self.temperature, self.scale = num_pos_feats // 2, temperature, 2 * math.pi
        self.cache = {}

    @torch.no_grad()
    def forward(self, x):
        B, H, W, device = x.shape[0], x.shape[-2], x.shape[-1], x.device
        if (H, W) not in self.cache:
            y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).view(1, -1, 1).repeat(1, 1, W)
            x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).view(1, 1, -1).repeat(1, H, 1)
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale
            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
            pos_x, pos_y = x_embed[:, :, :, None] / dim_t, y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            self.cache[(H, W)] = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)[0]
        return self.cache[(H, W)].to(device)[None].repeat(B, 1, 1, 1)


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats=64):
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", torch.empty((2, num_pos_feats)))

    def _pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size
        grid = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device, dtype=torch.float32)
        y_embed = (grid.cumsum(dim=0) - 0.5) / h
        x_embed = (grid.cumsum(dim=1) - 0.5) / w
        return self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1)).permute(2, 0, 1)

    def forward_with_coords(self, coords_input, image_size):
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


def compute_axial_cis(dim, end_x, end_y, theta=10000.0, device=None):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim))
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x, t_y = (t % end_x).float(), torch.div(t, end_x, rounding_mode="floor").float()
    freqs_x, freqs_y = torch.outer(t_x, freqs_x), torch.outer(t_y, freqs_y)
    return torch.cat([torch.polar(torch.ones_like(freqs_x), freqs_x), torch.polar(torch.ones_like(freqs_y), freqs_y)], dim=-1)


def apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=False):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = freqs_cis.view(*[d if i >= xq_.ndim - 2 else 1 for i, d in enumerate(xq_.shape)])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq), xk
    if repeat_freqs_k:
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), xk_.shape[-2] // xq_.shape[-2], 1)
    return xq_out.type_as(xq), torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


# ------------------------------------------------------------------------------------------------ Hiera image encoders

def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h, pad_w = (window_size - H % window_size) % window_size, (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C), (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    (Hp, Wp), (H, W) = pad_hw, hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :] if Hp > H or Wp > W else x


def do_pool(x, pool):
    if pool is None:
        return x
    return pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class LoRAQKV(nn.Module):
    """GeoSAM2 _LoRA_qkv: the frozen qkv projection plus rank-4 updates of its q and v outputs."""

    def __init__(self, dim, dim_out, rank):
        super().__init__()
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.linear_a_q = nn.Linear(dim, rank, bias=False)
        self.linear_b_q = nn.Linear(rank, dim, bias=False)
        self.linear_a_v = nn.Linear(dim, rank, bias=False)
        self.linear_b_v = nn.Linear(rank, dim, bias=False)
        self.dim = dim

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, :, : self.dim] += self.linear_b_q(self.linear_a_q(x))
        qkv[:, :, :, -self.dim:] += self.linear_b_v(self.linear_a_v(x))
        return qkv


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_pool, rank):
        super().__init__()
        self.num_heads, self.q_pool = num_heads, q_pool
        self.qkv = LoRAQKV(dim, dim_out, rank)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        B, H, W, _ = x.shape
        q, k, v = torch.unbind(self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1), 2)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        qkv_list = [q, k, v]
        q = k = v = None
        x = pay_attention(qkv_list, force_attention="sdpa")
        return self.proj(x.reshape(B, H, W, -1))


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_stride, window_size, rank, mlp_ratio=4.0):
        super().__init__()
        self.dim, self.dim_out, self.window_size, self.q_stride = dim, dim_out, window_size, q_stride
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False) if q_stride else None
        self.attn = MultiScaleAttention(dim, dim_out, num_heads, self.pool, rank)
        self.norm2 = nn.LayerNorm(dim_out, eps=1e-6)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=nn.GELU)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)
        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_hw = (H + (window_size - H % window_size) % window_size, W + (window_size - W % window_size) % window_size)
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class Hiera(nn.Module):
    def __init__(self, embed_dim=112, num_heads=2, q_pool=3, q_stride=(2, 2), stages=(2, 3, 16, 3), dim_mul=2.0, head_mul=2.0,
                 window_pos_embed_bkg_spatial_size=(14, 14), window_spec=(8, 4, 14, 7), global_att_blocks=(12, 16, 20), rank=4):
        super().__init__()
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, kernel_size=7, stride=4, padding=3)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, window_spec[0], window_spec[0]))
        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(sum(stages)):
            dim_out = embed_dim
            window_size = 0 if i in global_att_blocks else window_spec[cur_stage - 1]
            if i - 1 in self.stage_ends:
                dim_out, num_heads = int(embed_dim * dim_mul), int(num_heads * head_mul)
                cur_stage += 1
            self.blocks.append(MultiScaleBlock(embed_dim, dim_out, num_heads, q_stride if i in q_pool_blocks else None, window_size, rank))
            embed_dim = dim_out
        self.channel_list = [self.blocks[i].dim_out for i in self.stage_ends[::-1]]

    def forward(self, x):
        x = self.patch_embed.proj(x).permute(0, 2, 3, 1)
        pos_embed = F.interpolate(self.pos_embed, size=x.shape[1:3], mode="bicubic")
        pos_embed = pos_embed + self.pos_embed_window.tile([a // b for a, b in zip(pos_embed.shape, self.pos_embed_window.shape)])
        x = x + pos_embed.permute(0, 2, 3, 1)
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.stage_ends:
                outputs.append(x.permute(0, 3, 1, 2))
        return outputs


class FpnNeck(nn.Module):
    def __init__(self, d_model=256, backbone_channel_list=(896, 448, 224, 112), fpn_top_down_levels=(2, 3)):
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(d_model)
        self.convs = nn.ModuleList()
        for dim in backbone_channel_list:
            conv = nn.Sequential()
            conv.add_module("conv", nn.Conv2d(dim, d_model, kernel_size=1))
            self.convs.append(conv)
        self.fpn_top_down_levels = list(fpn_top_down_levels)
        self.d_model = d_model

    def forward(self, xs):
        out, pos = [None] * len(self.convs), [None] * len(self.convs)
        prev_features, n = None, len(self.convs) - 1
        for i in range(n, -1, -1):
            lateral_features = self.convs[n - i](xs[i])
            if i in self.fpn_top_down_levels and prev_features is not None:
                prev_features = lateral_features + F.interpolate(prev_features.to(dtype=torch.float32), scale_factor=2.0, mode="nearest")
            else:
                prev_features = lateral_features
            out[i] = prev_features
            pos[i] = self.position_encoding(prev_features).to(prev_features.dtype)
        return out, pos


class ImageEncoder(nn.Module):
    """SAMLoraImgEncoder with scalp 1: the lowest-resolution level is discarded."""

    def __init__(self):
        super().__init__()
        self.trunk = Hiera()
        self.neck = FpnNeck()

    def forward(self, sample):
        features, pos = self.neck(self.trunk(sample))
        return features[:-1], pos[:-1]


class FeatureFusion(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.fusion_high = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.fusion_mid = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.fusion_low = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, features1, features2):
        convs = (self.fusion_high, self.fusion_mid, self.fusion_low)
        return [convs[min(i, 2)](torch.cat([feat1, feat2], dim=1)) + feat1 for i, (feat1, feat2) in enumerate(zip(features1, features2))]


# ------------------------------------------------------------------------------------------------ SAM heads

class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads, downsample_rate=1, kv_in_dim=None):
        super().__init__()
        kv_in_dim = kv_in_dim or embedding_dim
        self.internal_dim, self.num_heads = embedding_dim // downsample_rate, num_heads
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _heads(self, x):
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)

    def forward(self, q, k, v):
        out = _attention(self._heads(self.q_proj(q)), self._heads(self.k_proj(k)), self._heads(self.v_proj(v)))
        b, h, n, c = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, n, h * c))


class RoPEAttention(Attention):
    def __init__(self, *args, rope_theta=10000.0, rope_k_repeat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.rope_theta, self.rope_k_repeat = rope_theta, rope_k_repeat
        self.freqs_cis = None

    def forward(self, q, k, v, num_k_exclude_rope=0):
        q, k, v = self._heads(self.q_proj(q)), self._heads(self.k_proj(k)), self._heads(self.v_proj(v))
        side = int(math.sqrt(q.shape[-2]))
        if self.freqs_cis is None or self.freqs_cis.shape[0] != q.shape[-2] or self.freqs_cis.device != q.device:
            self.freqs_cis = compute_axial_cis(self.internal_dim // self.num_heads, side, side, self.rope_theta, device=q.device)
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(q, k[:, :, :num_k_rope], self.freqs_cis, repeat_freqs_k=self.rope_k_repeat)
        out = _attention(q, k, v)
        b, h, n, c = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, n, h * c))


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim, skip_first_layer_pe, attention_downsample_rate=2):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=nn.ReLU)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)
        q, k = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))
        queries = self.norm3(queries + self.mlp(queries))
        q, k = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048, attention_downsample_rate=2):
        super().__init__()
        self.layers = nn.ModuleList([TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim, i == 0, attention_downsample_rate) for i in range(depth)])
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, image_pe, point_embedding):
        keys = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        attn_out = self.final_attn_token_to_image(q=queries + point_embedding, k=keys + image_pe, v=keys)
        return self.norm_final_attn(queries + attn_out), keys


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16):
        super().__init__()
        self.embed_dim, self.input_image_size, self.image_embedding_size = embed_dim, input_image_size, image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, embed_dim) for _ in range(4)])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans // 4), nn.GELU(),
                                              nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans), nn.GELU(),
                                              nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1))
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points, labels):
        points = points + 0.5
        points = torch.cat([points, torch.zeros((points.shape[0], 1, 2), device=points.device)], dim=1)
        labels = torch.cat([labels, -torch.ones((labels.shape[0], 1), device=labels.device)], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding = torch.where((labels == -1).unsqueeze(-1), torch.zeros_like(point_embedding) + self.not_a_point_embed.weight, point_embedding)
        index = torch.zeros(1, dtype=torch.long, device=point_embedding.device)
        for label, embedding in enumerate(self.point_embeddings):  # called, not read: MMGP loads the modules of a ModuleList as they run
            point_embedding = torch.where((labels == label).unsqueeze(-1), point_embedding + embedding(index), point_embedding)
        return point_embedding

    def forward(self, points, masks):
        """points (coords, labels) of shape (B, N, 2) / (B, N); masks (B, 1, 256, 256) or None. Boxes are given as labels 2/3 points."""
        sparse_embeddings = self._embed_points(*points)
        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(points[0].shape[0], -1, *self.image_embedding_size)
        return sparse_embeddings, dense_embeddings


class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim=256, num_multimask_outputs=3, iou_head_depth=3, iou_head_hidden_dim=256):
        super().__init__()
        self.transformer = TwoWayTransformer()
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.output_upscaling = nn.Sequential(nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), LayerNorm2d(transformer_dim // 4),
                                              nn.GELU(), nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), nn.GELU())
        self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
        self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth, sigmoid_output=True)
        self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, repeat_image, high_res_features,
                dynamic_multimask=False):
        """dynamic_multimask: upstream's dynamic_multimask_via_stability (delta 0.05, threshold 0.98) of the video predictor."""
        output_tokens = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0)
        tokens = torch.cat((output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1), sparse_prompt_embeddings), dim=1)
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) if repeat_image else image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out, mask_tokens_out = hs[:, 1, :], hs[:, 2: 2 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled_embedding = act2(dc2(act1(ln1(dc1(src) + feat_s1))) + feat_s0)
        hyper_in = torch.stack([mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)], dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        if multimask_output:
            masks, iou_pred = masks[:, 1:, :, :], iou_pred[:, 1:]
        elif dynamic_multimask:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks, iou_pred = masks[:, 0:1, :, :], iou_pred[:, 0:1]
        sam_tokens_out = mask_tokens_out[:, 1:] if multimask_output else mask_tokens_out[:, 0:1]
        return masks, iou_pred, sam_tokens_out, object_score_logits

    @staticmethod
    def _dynamic_multimask_via_stability(all_mask_logits, all_iou_scores, delta=0.05, threshold=0.98):
        multimask_logits, multimask_iou_scores = all_mask_logits[:, 1:, :, :], all_iou_scores[:, 1:]
        best = torch.argmax(multimask_iou_scores, dim=-1)
        batch = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_logits, best_scores = multimask_logits[batch, best].unsqueeze(1), multimask_iou_scores[batch, best].unsqueeze(1)
        single_logits, single_scores = all_mask_logits[:, 0:1, :, :], all_iou_scores[:, 0:1]
        flat = single_logits.flatten(-2)
        area_i, area_u = torch.sum(flat > delta, dim=-1).float(), torch.sum(flat > -delta, dim=-1).float()
        is_stable = torch.where(area_u > 0, area_i / area_u, 1.0) >= threshold
        return (torch.where(is_stable[..., None, None].expand_as(single_logits), single_logits, best_logits),
                torch.where(is_stable.expand_as(single_scores), single_scores, best_scores))


# ------------------------------------------------------------------------------------------------ memory

class MemoryAttentionLayer(nn.Module):
    def __init__(self, d_model=256, dim_feedforward=2048):
        super().__init__()
        self.self_attn = RoPEAttention(d_model, 1)
        self.cross_attn_image = RoPEAttention(d_model, 1, rope_k_repeat=True, kv_in_dim=64)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, tgt, memory, pos, query_pos, num_k_exclude_rope=0):
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.self_attn(tgt2, tgt2, v=tgt2)
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.cross_attn_image(q=tgt2, k=memory + pos, v=memory, num_k_exclude_rope=num_k_exclude_rope)
        return tgt + self.linear2(F.relu(self.linear1(self.norm3(tgt))))


class MemoryAttention(nn.Module):
    def __init__(self, d_model=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([MemoryAttentionLayer(d_model) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens=0):
        output = (curr + 0.1 * curr_pos).transpose(0, 1)
        curr_pos, memory, memory_pos = curr_pos.transpose(0, 1), memory.transpose(0, 1), memory_pos.transpose(0, 1)
        for layer in self.layers:
            output = layer(output, memory, memory_pos, curr_pos, num_k_exclude_rope=num_obj_ptr_tokens)
        return self.norm(output).transpose(0, 1)


class CXBlock(nn.Module):
    def __init__(self, dim=256, kernel_size=7, padding=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = self.norm(self.dwconv(x)).permute(0, 2, 3, 1)
        y = self.gamma * self.pwconv2(F.gelu(self.pwconv1(y)))
        return x + y.permute(0, 3, 1, 2)


class MemoryEncoder(nn.Module):
    def __init__(self, out_dim=64, in_dim=256):
        super().__init__()
        self.mask_downsampler = nn.Module()
        layers, channels = [], 1
        for _ in range(4):  # kernel 3, stride 2, padding 1, total stride 16
            layers += [nn.Conv2d(channels, channels * 4, kernel_size=3, stride=2, padding=1), LayerNorm2d(channels * 4), nn.GELU()]
            channels *= 4
        self.mask_downsampler.encoder = nn.Sequential(*layers, nn.Conv2d(channels, in_dim, kernel_size=1))
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = nn.Module()
        self.fuser.layers = nn.ModuleList([CXBlock(in_dim), CXBlock(in_dim)])
        self.position_encoding = PositionEmbeddingSine(out_dim)
        self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, pix_feat, masks):
        x = self.pix_feat_proj(pix_feat) + self.mask_downsampler.encoder(masks)
        for layer in self.fuser.layers:
            x = layer(x)
        x = self.out_proj(x)
        return x, self.position_encoding(x).to(x.dtype)


# ------------------------------------------------------------------------------------------------ GeoSAM2

class GeoSAM2(nn.Module):
    """SAM2Base of GeoSAM2 with the configuration of sam2/configs/geosam2.yaml."""

    def __init__(self, image_size=1024, backbone_stride=16, num_maskmem=7, hidden_dim=256, mem_dim=64, max_obj_ptrs_in_encoder=16, **_):
        super().__init__()
        self.image_encoder, self.pos_map_encoder, self.feature_fusion = ImageEncoder(), ImageEncoder(), FeatureFusion(hidden_dim)
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.memory_attention, self.memory_encoder = MemoryAttention(hidden_dim), MemoryEncoder(mem_dim, hidden_dim)
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem, 1, 1, mem_dim))
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, hidden_dim))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, mem_dim))
        self.sam_prompt_encoder = PromptEncoder(hidden_dim, (image_size // backbone_stride,) * 2, (image_size, image_size))
        self.sam_mask_decoder = MaskDecoder(hidden_dim)
        self.obj_ptr_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.obj_ptr_tpos_proj = nn.Linear(hidden_dim, mem_dim)
        self.image_size, self.hidden_dim, self.mem_dim, self.num_maskmem = image_size, hidden_dim, mem_dim, num_maskmem
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder

    def forward_image(self, pos_map, norm_map):
        """Features of one view: the normal map through the image encoder, the position map through its own encoder, fused. Returns
        the feature maps (levels 0 and 1 already projected for the SAM decoder) and their position encodings."""
        features, pos = self.image_encoder(norm_map)
        pos_features, _ = self.pos_map_encoder(pos_map)
        fpn = self.feature_fusion(features, pos_features)
        fpn[0] = self.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.sam_mask_decoder.conv_s1(fpn[1])
        return fpn, pos

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False,
                           dynamic_multimask=True):
        B, device = backbone_features.size(0), backbone_features.device
        if point_inputs is not None:
            coords, labels = point_inputs["point_coords"], point_inputs["point_labels"]
        else:
            coords, labels = torch.zeros(B, 1, 2, device=device), -torch.ones(B, 1, dtype=torch.int32, device=device)
        if mask_inputs is not None and mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
            mask_inputs = F.interpolate(mask_inputs.float(), size=self.sam_prompt_encoder.mask_input_size, align_corners=False, mode="bilinear", antialias=True)
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder((coords, labels), mask_inputs)
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            backbone_features, self.sam_prompt_encoder.get_dense_pe(), sparse_embeddings, dense_embeddings, multimask_output, False, high_res_features,
            dynamic_multimask)
        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE).float()
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds, batch_inds = torch.argmax(ious, dim=-1), torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.float()
        obj_ptr = lambda_is_obj_appearing * obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        out_scale, out_bias = 20.0, -10.0
        high_res_masks = mask_inputs.float() * out_scale + out_bias
        low_res_masks = F.interpolate(high_res_masks, size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4), align_corners=False,
                                      mode="bilinear", antialias=True)
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(backbone_features, mask_inputs=self.mask_downsample(mask_inputs.float()),
                                                            high_res_features=high_res_features)
        lambda_is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)[..., None].float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return low_res_masks, high_res_masks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes,
                                             output_dict, num_frames):
        """Upstream SAM2Base._prepare_memory_conditioned_features (eval, forward tracking, memory stride 1, all conditioning frames), with
        GeoSAM2's circular frame indices."""
        B, C = current_vision_feats[-1].size(1), self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        if is_init_cond_frame:  # directly_add_no_mem_embed
            return (current_vision_feats[-1] + self.no_mem_embed).permute(1, 2, 0).view(B, C, H, W)
        to_cat_memory, to_cat_memory_pos_embed = [], []
        cond_outputs = output_dict["cond_frame_outputs"]
        t_pos_and_prevs = [(0, out) for out in cond_outputs.values()]
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            prev_frame_idx = frame_idx - t_rel if t_rel == 1 else (frame_idx - 2) - (t_rel - 2)
            t_pos_and_prevs.append((t_pos, output_dict["non_cond_frame_outputs"].get(prev_frame_idx % num_frames)))
        for t_pos, prev in t_pos_and_prevs:
            if prev is None:
                continue
            to_cat_memory.append(prev["maskmem_features"].to(device).flatten(2).permute(2, 0, 1))
            maskmem_enc = prev["maskmem_pos_enc"][-1].to(device).flatten(2).permute(2, 0, 1)
            to_cat_memory_pos_embed.append(maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1])
        max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
        pos_and_ptrs = [(frame_idx - t, out["obj_ptr"]) for t, out in cond_outputs.items() if t <= frame_idx]
        for t_diff in range(1, max_obj_ptrs_in_encoder):
            t = frame_idx - t_diff
            if t < 0 or t >= num_frames:
                break
            out = output_dict["non_cond_frame_outputs"].get(t)
            if out is not None:
                pos_and_ptrs.append((t_diff, out["obj_ptr"]))
        num_obj_ptr_tokens = 0
        if pos_and_ptrs:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=0)
            obj_pos = get_1d_sine_pe(torch.tensor(pos_list, device=device) / (max_obj_ptrs_in_encoder - 1), dim=C)
            obj_pos = self.obj_ptr_tpos_proj(obj_pos).unsqueeze(1).expand(-1, B, self.mem_dim)
            obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim).permute(0, 2, 1, 3).flatten(0, 1)
            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos_embed.append(obj_pos.repeat_interleave(C // self.mem_dim, dim=0))
            num_obj_ptr_tokens = obj_ptrs.shape[0]
        memory, memory_pos_embed = torch.cat(to_cat_memory, dim=0), torch.cat(to_cat_memory_pos_embed, dim=0)
        pix_feat_with_mem = self.memory_attention(current_vision_feats[-1], memory, current_vision_pos_embeds[-1], memory_pos_embed, num_obj_ptr_tokens)
        return pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

    def encode_new_memory(self, current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts):
        """binarize_mask_from_pts_for_mem_enc (video predictor), sigmoid scale 20 and bias -10."""
        B, C = current_vision_feats[-1].size(1), self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        mask_for_mem = (pred_masks_high_res > 0).float() if is_mask_from_pts else torch.sigmoid(pred_masks_high_res)
        mask_for_mem = mask_for_mem * 20.0 - 10.0
        maskmem_features, maskmem_pos_enc = self.memory_encoder(pix_feat, mask_for_mem)
        is_obj_appearing = (object_score_logits > 0).float()
        maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)
        return maskmem_features, [maskmem_pos_enc]

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict,
                   num_frames, run_mem_encoder=True, prev_sam_mask_logits=None):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        high_res_features = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])]
        if mask_inputs is not None:  # use_mask_input_as_output_without_sam
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            pix_feat = self._prepare_memory_conditioned_features(frame_idx, is_init_cond_frame, current_vision_feats[-1:], current_vision_pos_embeds[-1:],
                                                                 feat_sizes[-1:], output_dict, num_frames)
            if prev_sam_mask_logits is not None:
                mask_inputs = prev_sam_mask_logits
            num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
            sam_outputs = self._forward_sam_heads(pix_feat, point_inputs, mask_inputs, high_res_features, multimask_output=num_pts <= 1)
        _, _, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs
        current_out.update(pred_masks=low_res_masks, pred_masks_high_res=high_res_masks, obj_ptr=obj_ptr, ious=ious, object_score_logits=object_score_logits)
        if run_mem_encoder:
            current_out["maskmem_features"], current_out["maskmem_pos_enc"] = self.encode_new_memory(
                current_vision_feats, feat_sizes, high_res_masks, object_score_logits, point_inputs is not None)
        else:
            current_out["maskmem_features"] = current_out["maskmem_pos_enc"] = None
        return current_out
