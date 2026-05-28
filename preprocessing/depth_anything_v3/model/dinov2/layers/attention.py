# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger("dinov2")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def _apply_rope(self, qk_list: list[Tensor], pos) -> tuple[Tensor, Tensor]:
        q, k = qk_list
        qk_list.clear()
        return self.rope(q, pos), self.rope(k, pos)

    def forward(self, x: Tensor | list[Tensor], pos=None, attn_mask=None) -> Tensor:
        if isinstance(x, list):
            x_list = x
            x = x_list[0]
            x_list.clear()
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        del x

        q_list = [qkv[0]]
        k_list = [qkv[1]]
        v = qkv[2]
        del qkv
        q = self.q_norm(q_list[0])
        q_list.clear()
        k = self.k_norm(k_list[0])
        k_list.clear()
        if self.rope is not None and pos is not None:
            qk_list = [q, k]
            del q, k
            q, k = self._apply_rope(qk_list, pos)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=(
                    (attn_mask)[:, None].expand(-1, self.num_heads, -1, -1)
                    if attn_mask is not None
                    else None
                ),
            )
            del q, k, v
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            del q, k
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            del attn, v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
