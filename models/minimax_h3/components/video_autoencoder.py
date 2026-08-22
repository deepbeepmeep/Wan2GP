# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionMixin
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook

from shared.attention import pay_attention

from ..interrupt import GenerationInterrupted


logger = logging.get_logger(__name__)


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    r"""
    3D convolution used throughout the MiniMax-H3 video encoder.

    Spatial padding is symmetric and uses `spatial_padding_mode` (`"reflect"` in the released checkpoint); temporal
    padding is causal, i.e. `kernel_size_t - 1` zero frames are prepended and nothing is appended.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
    r"""
    Group normalization applied to each latent frame in isolation (`use_t_isolated_gn` in the original config): the
    temporal axis is folded into the batch axis so statistics never mix across frames.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.nin_shortcut = None
        if in_channels != out_channels:
            self.nin_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.nin_shortcut is not None:
            residual = self.nin_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    r"""
    Strided 3x3x3 downsampling convolution. A spatial stride of 2 is preceded by an asymmetric bottom/right pad of 1
    (the convolution itself carries no spatial padding), so the output is exactly `ceil(size / 2)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.downsample = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsample = MiniMaxH3VideoDownsample3d(
                out_channels,
                out_channels,
                temporal_stride=temporal_downsample_factor,
                spatial_stride=spatial_downsample_factor,
                spatial_padding_mode=spatial_padding_mode,
            )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.block:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
            else:
                hidden_states = resnet(hidden_states)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    r"""
    Causal 3D CNN encoder. `block_out_channels` gives the channel count of every level; the per-level
    `spatial_downsample_factors` / `temporal_downsample_factors` multiply out to the total compression ratios.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(len(block_out_channels))
            ]
        )

        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down:
            hidden_states = down_block(hidden_states)
            if getattr(self, "_interrupt", False):
                raise GenerationInterrupted
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    r"""
    3-axis rotary embedding for the ViT decoder. Coordinates are length-normalized to `[-1, 1)` per axis and scaled by
    `2 * pi`, and the resulting `(t, h, w)` angles are concatenated and then duplicated, so the first
    `rope_dim_ratio * attention_head_dim` channels of every head are rotated.
    """

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"`dim` {dim} must be divisible by `2 * num_axes` {2 * num_axes}.")
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


def _take(x_list: list[torch.Tensor]) -> torch.Tensor:
    hidden_states = x_list[0]
    x_list.clear()
    return hidden_states


def _split_interleaved_qkv(src, dim, split_sizes, context):
    info = context["info"]
    heads, head_dim = info["num_attention_heads"], info["attention_head_dim"]
    grouped = src.reshape(heads, 3, head_dim, *src.shape[1:])
    return [grouped[:, index].reshape(split_sizes[index], *src.shape[1:]).contiguous() for index in range(3)]


def get_linear_split_map(inner_dim: int = 2048, num_attention_heads: int = 32,
                         attention_head_dim: int = 64):
    return {
        "to_qkv": {
            "mapped_modules": ["to_q", "to_k", "to_v"],
            "split_sizes": [inner_dim, inner_dim, inner_dim],
            "num_attention_heads": num_attention_heads,
            "attention_head_dim": attention_head_dim,
            "split_handlers": {"weight": _split_interleaved_qkv, "bias": _split_interleaved_qkv},
        }
    }


class MiniMaxH3VideoAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MiniMaxH3VideoAttention",
        hidden_states: list[torch.Tensor],
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden_states = _take(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape
        if hasattr(attn, "to_q"):
            dtype = hidden_states.dtype
            query = attn.norm_q(attn.to_q(hidden_states).view(batch_size, seq_len, attn.heads, attn.dim_head).float()).to(dtype)
            key = attn.norm_k(attn.to_k(hidden_states).view(batch_size, seq_len, attn.heads, attn.dim_head).float()).to(dtype)
            value = attn.to_v(hidden_states).view(batch_size, seq_len, attn.heads, attn.dim_head)
        else:
            qkv = attn.to_qkv(hidden_states).view(batch_size, seq_len, attn.heads, 3, attn.dim_head)
            query, key, value = qkv.unbind(dim=3)
            query = attn.norm_q(query.float()).to(query.dtype)
            key = attn.norm_k(key.float()).to(key.dtype)
            value = value.clone()
            del qkv
        del hidden_states

        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos, sin = cos.to(query.dtype), sin.to(query.dtype)
            rotary_dim = cos.shape[-1]
            query_rotary = query[..., :rotary_dim]
            key_rotary = key[..., :rotary_dim]
            query_first, query_second = query_rotary.chunk(2, dim=-1)
            key_first, key_second = key_rotary.chunk(2, dim=-1)
            half = query_first.shape[-1]
            query_first_out = query_first * cos[..., :half] - query_second * sin[..., :half]
            query_second_out = query_second * cos[..., half:] + query_first * sin[..., half:]
            key_first_out = key_first * cos[..., :half] - key_second * sin[..., :half]
            key_second_out = key_second * cos[..., half:] + key_first * sin[..., half:]
            query_first.copy_(query_first_out)
            query_second.copy_(query_second_out)
            key_first.copy_(key_first_out)
            key_second.copy_(key_second_out)
            del query_first_out, query_second_out, key_first_out, key_second_out

        output_dtype = query.dtype
        if output_dtype == torch.float32:
            query, key, value = query.half(), key.half(), value.half()
        hidden_states = pay_attention([query, key, value], causal=False, recycle_q=True)
        return attn.to_out(hidden_states.flatten(2, 3).to(output_dtype))


class MiniMaxH3VideoAttention(nn.Module):
    _default_processor_cls = MiniMaxH3VideoAttnProcessor
    _available_processors = [MiniMaxH3VideoAttnProcessor]

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.use_bias = bias
        inner_dim = heads * dim_head

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=bias)

        self.processor = MiniMaxH3VideoAttnProcessor()

    def forward(
        self, hidden_states: list[torch.Tensor], rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb)


class MiniMaxH3VideoFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, bias: bool = True) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.w1 = nn.Linear(dim, 2 * inner_dim, bias=bias)
        self.w2 = nn.Linear(inner_dim, dim, bias=bias)

    def _project(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        hidden_states = _take(hidden_states)
        expanded = self.w1(hidden_states)
        del hidden_states
        gate, value = expanded.chunk(2, dim=-1)
        F.silu(gate, inplace=True).mul_(value)
        del expanded, value
        return self.w2(gate)

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        hidden_states = _take(hidden_states)
        chunk_size = max(1, hidden_states.shape[1] * hidden_states.shape[2] // self.w1.out_features)
        if hidden_states.shape[1] <= chunk_size:
            return self._project([hidden_states])
        for start in range(0, hidden_states.shape[1], chunk_size):
            output = self._project([hidden_states[:, start : start + chunk_size]])
            hidden_states[:, start : start + output.shape[1]].copy_(output)
            del output
        return hidden_states


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3VideoAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = MiniMaxH3VideoFeedForward(dim, ffn_mult, bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self, hidden_states: list[torch.Tensor], rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        hidden_states = _take(hidden_states)
        norm_hidden_states = self.norm1(hidden_states.to(self.norm1.weight.dtype)).to(hidden_states.dtype)
        branch = self.attn([norm_hidden_states], rotary_emb)
        branch.mul_(self.scale1).add_(hidden_states)
        del hidden_states
        hidden_states = branch
        norm_hidden_states = self.norm2(hidden_states.to(self.norm2.weight.dtype)).to(hidden_states.dtype)
        branch = self.ff([norm_hidden_states])
        branch.mul_(self.scale2).add_(hidden_states)
        del hidden_states
        return branch


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    r"""
    Non-causal ViT decoder. Every latent voxel becomes one token; `num_register_tokens` learned register tokens plus a
    single all-zero token are appended (all at position `0`), attended over with full self-attention, and dropped
    again before the patch projection expands each token into a `patch_size_t x patch_size x patch_size` pixel block.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.register_buffer("mask_token", torch.zeros(1, 1, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = self.x_embedder(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            if getattr(self, "_interrupt", False):
                raise GenerationInterrupted
            hidden_states = block([hidden_states], rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


class AutoencoderKLMiniMaxH3(ModelMixin, ConfigMixin, AttentionMixin, AutoencoderMixin):
    r"""
    A VAE model with a causal 3D CNN encoder and a non-causal ViT decoder, used in
    [MiniMax-H3](https://huggingface.co/MiniMaxAI).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Latents are normalized with per-channel `latents_mean` / `latents_std` rather than a `scaling_factor`; a pipeline
    encodes with `(latent - latents_mean) / latents_std` and decodes with `latent * latents_std + latents_mean`.

    The pixel convention is ImageNet-normalized RGB over a `[0, 1]` base range, not the usual `[-1, 1]`: `encode`
    expects `(pixel - imagenet_mean) / imagenet_std` and `decode` returns values in that same space, so a pipeline has
    to apply `sample * imagenet_std + imagenet_mean` (mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`) and
    clamp to `[0, 1]` before postprocessing.

    The temporal geometry is fixed by `clip_length` (17 pixel frames per encoder chunk) and `token_drop` (3 trailing
    latent frames dropped per encode): `17 * n + 5` pixel frames map to `5 * n + 2` latent frames.

    Unlike most autoencoders in the library, spatial tiling is **on by default**: MiniMax-H3 was released with tiling
    enabled for both encoding and decoding, and the released frames are the blended-tile ones, so disabling tiling
    changes the output. Use `enable_tiling` to change the tile geometry, `disable_tiling` to turn it off.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3VideoResnetBlock3d", "MiniMaxH3VideoTransformerBlock"]
    _repeated_blocks = ["MiniMaxH3VideoTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    # The released checkpoint is float32 and the verified decode recipe is float16 *autocast over float32 weights*
    # (see `decode`). A pipeline-level `torch_dtype=torch.bfloat16` must therefore not downcast the weights, so every
    # top-level module is pinned, mirroring the transformer's mixed-precision contract.
    _keep_in_fp32_modules = ["encoder", "decoder", "quant_conv", "post_quant_conv"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        decoder_num_layers: int = 36,
        decoder_num_attention_heads: int = 32,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.75,
        decoder_norm_eps: float = 1e-5,
        clip_length: int = 17,
        token_drop: int = 3,
        latents_mean: tuple[float, ...] = (0.0,) * 24,
        latents_std: tuple[float, ...] = (1.0,) * 24,
    ) -> None:
        super().__init__()

        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(temporal_downsample_factors)

        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=in_channels,
            out_channels=2 * latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            spatial_downsample_factors=spatial_downsample_factors,
            temporal_downsample_factors=temporal_downsample_factors,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
            patch_size_t=self.temporal_compression_ratio,
            num_layers=decoder_num_layers,
            num_attention_heads=decoder_num_attention_heads,
            attention_head_dim=decoder_attention_head_dim,
            num_register_tokens=decoder_num_register_tokens,
            ffn_mult=decoder_ffn_mult,
            rope_theta=decoder_rope_theta,
            rope_dim_ratio=decoder_rope_dim_ratio,
            norm_eps=decoder_norm_eps,
        )

        # Derived temporal-chunking geometry. `clip_length` pixel frames are encoded at a time; because
        # `clip_length` is not a multiple of `temporal_compression_ratio`, the decoder has to re-derive the
        # implicit leading pad (`frame_pre_padding`) and the overlap that `token_drop` leaves behind.
        self.frame_pre_padding = (-clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When encoding/decoding spatially large videos, the memory requirement is very high. By splitting the frames
        # into smaller tiles, running the encoder/decoder per tile and blending the overlaps, the memory requirement
        # can be lowered. MiniMax-H3 ships with tiling enabled.
        self.use_tiling = True

        # The tile size in pixel space, and the minimum overlap between two neighbouring tiles. The actual overlaps are
        # widened (in multiples of `spatial_compression_ratio`) so that the tiles cover the frame exactly.
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        r"""
        Enable tiled VAE encoding/decoding. When this option is enabled, the VAE splits the frames into tiles, encodes
        or decodes each tile separately and linearly blends the overlaps back together. This lowers the memory
        requirement and allows processing larger frames.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The tile height in pixel space. Frames taller than this are split along the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The tile width in pixel space. Frames wider than this are split along the width dimension.
            tile_sample_min_overlap_height (`int`, *optional*):
                The minimum overlap, in pixels, between two consecutive vertical tiles.
            tile_sample_min_overlap_width (`int`, *optional*):
                The minimum overlap, in pixels, between two consecutive horizontal tiles.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = tile_sample_min_overlap_width or self.tile_sample_min_overlap_width

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        r"""
        Lay `tile_size`-wide tiles over `length` pixels. The number of tiles is the smallest one whose union can cover
        `length` while keeping every overlap at least `min_overlap`; the slack is then distributed round-robin over the
        overlaps in whole `spatial_compression_ratio` steps so that every tile boundary stays latent-aligned.
        """
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1

        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio

        tile_start_indices = [0]
        for i in range(num_tiles - 1):
            tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[i])
        return tile_start_indices, [tile_size] * num_tiles, overlaps

    def _blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    @apply_forward_hook
    def _encode_clip(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Encode one temporal clip, spatially tiled when tiling is enabled.

        MiniMax-H3 encodes a keyframe or an image reference through this method rather than through [`~encode`],
        because a single frame must not go through the temporal chunking, so it carries the offload hook too.
        """
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x))

        height, width = x.shape[-2], x.shape[-1]
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)

        latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        r"""Decode one temporal clip, spatially tiled when tiling is enabled."""
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))

        # Tiles are laid out in pixel space and then mapped back onto the latent grid.
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        ratio = self.spatial_compression_ratio
        tiles = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            for j_pos, j_len in zip(x_indices, x_lengths):
                tiles.append(z[..., i_pos // ratio : i_pos // ratio + i_len // ratio,
                               j_pos // ratio : j_pos // ratio + j_len // ratio])
        batch_size = z.shape[0]
        tile_batch = torch.cat(tiles, dim=0)
        del tiles
        hidden_states = self.post_quant_conv(tile_batch)
        del tile_batch
        decoded = self.decoder(hidden_states)
        del hidden_states

        canvas = torch.empty(batch_size, *decoded.shape[1:-2], height, width,
                             dtype=decoded.dtype, device=decoded.device)
        row_tails = []
        out_y = 0
        tile_index = 0
        for i in range(len(y_indices)):
            new_tails = []
            left_tail = None
            out_x = 0
            for j in range(len(x_indices)):
                tile = decoded[tile_index * batch_size : (tile_index + 1) * batch_size]
                tile_index += 1
                if i > 0:
                    tile = self._blend(row_tails[j], tile, y_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(left_tail, tile, x_overlaps[j - 1], dim=-1)
                # Preserve both-axis contributions at corners and when three spatial tiles overlap.
                if i < len(y_indices) - 1:
                    new_tails.append(tile[..., -y_overlaps[i] :, :].clone())
                next_left_tail = tile[..., :, -x_overlaps[j] :].clone() if j < len(x_indices) - 1 else None
                left_tail = next_left_tail
                if i < len(y_indices) - 1:
                    tile = tile[..., : -y_overlaps[i], :]
                if j < len(x_indices) - 1:
                    tile = tile[..., :, : -x_overlaps[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
                canvas[..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        del decoded
        return canvas

    @apply_forward_hook
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Encode a video in `clip_length`-frame chunks and drop the `token_drop` trailing latent frames.

        MiniMax-H3 encodes a video reference through this method rather than through [`~encode`], because the
        posterior is sampled under a fixed generator rather than through the distribution object, so it carries the
        offload hook too.
        """
        clip_length = self.config.clip_length
        num_frames = x.shape[2]
        if num_frames % clip_length != 0:
            pad_frames = x[:, :, -1:].repeat(1, 1, (-num_frames) % clip_length, 1, 1)
            x = torch.cat([x, pad_frames], dim=2)

        moments = torch.cat(
            [
                self._encode_clip(x[:, :, i * clip_length : (i + 1) * clip_length])
                for i in range(x.shape[2] // clip_length)
            ],
            dim=2,
        )
        if self.config.token_drop > 0:
            moments = moments[:, :, : -self.config.token_drop]
        return moments

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Decode a latent video, mirroring the chunking that `_encode` applied.

        `token_drop` removed the tail of every encoded chunk, so consecutive decoded chunks overlap by
        `frame_overlap` pixel frames and are linearly cross-faded. Latent frames are repeated at the end when the
        length is not a whole number of chunks; the extra pixel frames are cut off again at the end.
        """
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        intra_tail = self.config.clip_length % temporal_ratio
        num_tokens_before_pad = z.shape[2] - pad_tokens
        pad_frames = sum(
            intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
            for k in range(pad_tokens)
        )
        output_frames = num_chunks * (chunk_num_frames - self.frame_pre_padding) + self.frame_overlap - pad_frames
        decoded = None
        write_position = 0
        overlap = None
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            clip = self._decode_clip(z[:, :, start : start + tokens_chunk_size + self.token_overlap])
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    if decoded is None:
                        decoded = torch.empty(*chunk.shape[:2], output_frames, *chunk.shape[3:],
                                              dtype=chunk.dtype, device=chunk.device)
                    copy_frames = min(chunk.shape[2], output_frames - write_position)
                    if copy_frames > 0:
                        decoded[:, :, write_position : write_position + copy_frames].copy_(chunk[:, :, :copy_frames])
                        write_position += copy_frames
                else:
                    overlap = chunk.contiguous()
            del clip
        if overlap is not None:
            copy_frames = min(overlap.shape[2], output_frames - write_position)
            if copy_frames > 0:
                decoded[:, :, write_position : write_position + copy_frames].copy_(overlap[:, :, :copy_frames])
                write_position += copy_frames
        if write_position != output_frames:
            raise RuntimeError(f"MiniMax H3 VAE decoded {write_position} frames, expected {output_frames}")
        return decoded

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple[torch.Tensor]:
        r"""
        Encode a batch of videos into latents.

        Args:
            x (`torch.Tensor`):
                Input batch of videos, shape `(batch_size, in_channels, num_frames, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.autoencoder_kl.AutoencoderKLOutput`] instead of a plain
                tuple.

        Returns:
            The latent distribution of the encoded videos. Note that MiniMax-H3 normalizes the sampled latents with
            `latents_mean` / `latents_std` afterwards.
        """
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self._encode(x_slice) for x_slice in x.split(1)])
        else:
            moments = self._encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Decode a batch of latent videos.

        Args:
            z (`torch.Tensor`):
                Input batch of latent videos, shape `(batch_size, latent_channels, num_latent_frames, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The decoded videos, shape `(batch_size, out_channels, num_frames, height, width)`.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)])
        else:
            decoded = self._decode(z)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Encode then decode a batch of videos.

        Args:
            sample (`torch.Tensor`):
                Input batch of videos, shape `(batch_size, in_channels, num_frames, height, width)`.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample the posterior instead of taking its mode.
            generator (`torch.Generator`, *optional*):
                Generator used when `sample_posterior=True`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The round-tripped videos, shape `(batch_size, out_channels, num_frames, height, width)`.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
