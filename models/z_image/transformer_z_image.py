# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from shared.attention import pay_attention


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ZSingleStreamAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_list: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            x_in = x_list[0]
            dtype = x_in.dtype 
            x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
            x_in = None
            x_list.clear()
            cos = freqs_cis[..., 0].unsqueeze(2)
            sin = freqs_cis[..., 1].unsqueeze(2)
            x0 = x[..., 0]
            x1 = x[..., 1]
            x0_orig = x0.clone()
            x0.mul_(cos)
            x0.addcmul_(x1, sin, value=-1)
            x1.mul_(cos)
            x1.addcmul_(x0_orig, sin)
            return x.flatten(3).to(dtype)

        dtype = query.dtype
        if freqs_cis is not None:
            x_list= [query]
            del query
            query = apply_rotary_emb(x_list, freqs_cis).to(dtype)
            x_list= [key]
            del key
            key = apply_rotary_emb(x_list, freqs_cis).to(dtype)


        # Compute joint attention
        qkv_list = [query, key, value]
        hidden_states = pay_attention(
            qkv_list,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))

    def forward_into(self, x, out):
        assert out.shape == x.shape
        x1 = self.w1(x)                 # (..., hidden_dim)
        x3 = self.w3(x)                 # (..., hidden_dim)
        F.silu(x1, inplace=True)        # x1 = silu(x1)
        x1.mul_(x3)                     # x1 = x1 * x3
        torch.matmul(
            x1.view(-1, self.hidden_dim),
            self.w2.weight.t(),
            out=out.view(-1, self.dim),
        )
        return out

@maybe_allow_in_graph
class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        block_id=None,  # For control hint injection
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.block_id = block_id  # For control hint injection

        # Refactored to use diffusers Attention with custom processor
        # Original Z-Image params: dim, n_heads, n_kv_heads, qk_norm
        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=ZSingleStreamAttnProcessor(),
        )
        self.ffn_mult =  8 / 3  
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim * self.ffn_mult))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def _apply_ffn(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        ffn_in = self.ffn_norm1(x)
        if scale is not None:
            ffn_in.mul_(scale)
        ffn_in = ffn_in.reshape(-1, dim)
        chunk_size = max(int(seq_len // self.ffn_mult), 1)
        for ffn_chunk in torch.split(ffn_in, chunk_size):
            ffn_chunk[...]= self.feed_forward(ffn_chunk)

        return self.ffn_norm2(ffn_in.view(batch, seq_len, dim))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        hints: Optional[List[torch.Tensor]] = None,
        context_scale: float = 1.0,
    ):
        # Process control hint just-in-time if this layer has control attached
        hint_skip = None
        if self.block_id is not None and hints is not None and hasattr(self, 'control'):
            kwargs = {
                "attn_mask": attn_mask,
                "freqs_cis": freqs_cis,
                "adaln_input": adaln_input,
            }
            hint_skip = self.control(hints, x if self.block_id == 0 else None, **kwargs)

        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa.tanh_()
            gate_mlp.tanh_()
            scale_msa.add_(1.0)
            scale_mlp.add_(1.0)

            attn_in = self.attention_norm1(x)
            attn_in.mul_(scale_msa)
            attn_out = self.attention(
                attn_in,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x.addcmul_(self.attention_norm2(attn_out), gate_msa)

            x.addcmul_(self._apply_ffn(x, scale_mlp), gate_mlp)
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x.add_(self.attention_norm2(attn_out))

            x.add_(self._apply_ffn(x))

        # Apply control hint skip connection
        if hint_skip is not None:
            if context_scale == 1.0:
                x.add_(hint_skip)
            else:
                x.add_(hint_skip, alpha=context_scale)

        return x


@maybe_allow_in_graph
class ZImageControlTransformerBlock(ZImageTransformerBlock):
    """Control block that processes control context and generates skip connection hints."""

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        block_id=0,
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation, block_id=None)
        self.control_block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, hints, x, **kwargs):
        # Get current control state from mutable list
        c = hints[0]
        hints[0] = None  # Free memory immediately

        # First block: project control and add to main signal
        if self.control_block_id == 0:
            c = self.before_proj(c) + x

        # Run through parent transformer block (without hint application)
        c = super().forward(c, **kwargs)

        # Create skip connection output
        c_skip = self.after_proj(c)
        hints[0] = c  # Put processed state back for next control layer
        return c_skip  # Return skip connection (used immediately)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=1)


class ZImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock", "ZImageControlTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        enable_control=False,  # Whether to create control layers
        control_layers_places=None,  # Explicit layer positions for control injection, e.g. [0, 5, 10, 15, 20, 25]
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        # Control layers - created if enable_control is True
        self.enable_control = enable_control

        if enable_control:
            # Use explicit control_layers_places from config (e.g. [0, 5, 10, 15, 20, 25])
            if control_layers_places is None:
                raise ValueError("control_layers_places must be specified when enable_control=True")
            self.control_layers_places = list(control_layers_places)
            assert 0 in self.control_layers_places, "Layer 0 must be in control_layers_places"
            self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers_places)}

            # Main layers with block_id set for control injection
            self.layers = nn.ModuleList(
                [
                    ZImageTransformerBlock(
                        layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm,
                        block_id=self.control_layers_mapping[layer_id] if layer_id in self.control_layers_places else None
                    )
                    for layer_id in range(n_layers)
                ]
            )

            # Control blocks that generate hints
            self.control_layers = nn.ModuleList(
                [
                    ZImageControlTransformerBlock(
                        layer_pos, dim, n_heads, n_kv_heads, norm_eps, qk_norm,
                        modulation=True, block_id=idx
                    )
                    for idx, layer_pos in enumerate(self.control_layers_places)
                ]
            )

            # Control input embedders (same structure as main embedders)
            control_all_x_embedder = {}
            for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
                control_x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
                control_all_x_embedder[f"{patch_size}-{f_patch_size}"] = control_x_embedder
            self.control_all_x_embedder = nn.ModuleDict(control_all_x_embedder)

            # Control noise refiner
            self.control_noise_refiner = nn.ModuleList(
                [
                    ZImageTransformerBlock(
                        1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm,
                        modulation=True,
                    )
                    for layer_id in range(n_refiner_layers)
                ]
            )
        else:
            # Non-control mode: standard layers without block_id
            self.layers = nn.ModuleList(
                [
                    ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                    for layer_id in range(n_layers)
                ]
            )

        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    @property
    def has_control(self) -> bool:
        """Returns True if the model has control layers enabled."""
        return self.enable_control

    def remap_state_dict(self, sd):
        """Remap state dict keys for control layers to be attached to main layers."""
        if not sd:
            return sd
        first = next(iter(sd.keys()))

        # Remap control_layers.X.* to layers.Y.control.* where Y is the target layer
        if first.startswith("control_layers."):
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("control_layers."):
                    parts = k.split(".")
                    control_idx = int(parts[1])
                    target_layer = self.control_layers_places[control_idx]
                    parts[0] = f"layers.{target_layer}"
                    parts[1] = "control"
                    k = ".".join(parts)
                new_sd[k] = v
            sd = new_sd

        return sd

    def adapt_control_model(self):
        """Move control blocks to be submodules of their corresponding main layers."""
        if not self.enable_control or not hasattr(self, 'control_layers'):
            return

        modules_dict = {k: m for k, m in self.named_modules()}
        for model_layer, control_idx in self.control_layers_mapping.items():
            control_module = modules_dict[f"control_layers.{control_idx}"]
            target = modules_dict[f"layers.{model_layer}"]
            setattr(target, "control", control_module)
        delattr(self, "control_layers")

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def patchify(
        self,
        all_image: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
        cap_len: int,
    ):
        """Patchify control images without caption processing (for control context)."""
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []

        for image in all_image:
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)

            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )

            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return all_image_out, all_image_size, all_image_pos_ids, all_image_pad_mask

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        callback=None,
        pipeline=None,
        control_latent: Optional[List[torch.Tensor]] = None,
        control_context_scale: float = 1.0,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        adaln_input = t

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        # x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        # for i, seq_len in enumerate(x_item_seqlens):
        #     x_attn_mask[i, :seq_len] = 1
        x_attn_mask = None

        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        # Prepare control hints if control is enabled and control_latent is provided
        hints = None
        if self.enable_control and control_latent is not None:
            # Patchify control context
            control_context, _, ctrl_pos_ids, ctrl_inner_pad_mask = self.patchify(
                control_latent, patch_size, f_patch_size, cap_item_seqlens[0]
            )

            # Embed control context
            ctrl_item_seqlens = [len(_) for _ in control_context]
            ctrl_max_item_seqlen = max(ctrl_item_seqlens)

            control_context = torch.cat(control_context, dim=0)
            control_context = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control_context)

            # Match adaln_input dtype to control_context
            adaln_input_ctrl = adaln_input[:1].type_as(control_context)
            control_context[torch.cat(ctrl_inner_pad_mask)] = self.x_pad_token
            control_context = list(control_context.split(ctrl_item_seqlens, dim=0))
            ctrl_freqs_cis = list(self.rope_embedder(torch.cat(ctrl_pos_ids, dim=0)).split(ctrl_item_seqlens, dim=0))

            control_context = pad_sequence(control_context, batch_first=True, padding_value=0.0)
            ctrl_freqs_cis = pad_sequence(ctrl_freqs_cis, batch_first=True, padding_value=0.0)
            ctrl_attn_mask = torch.zeros((1, ctrl_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(ctrl_item_seqlens):
                ctrl_attn_mask[i, :seq_len] = 1

            # Refine control context through control noise refiner
            for layer in self.control_noise_refiner:
                control_context = layer(control_context, ctrl_attn_mask, ctrl_freqs_cis, adaln_input_ctrl)

            # Unify control context with caption features
            control_context_unified = []
            for i in range(1):
                ctrl_len = ctrl_item_seqlens[i]
                cap_len = cap_item_seqlens[i]
                control_context_unified.append(torch.cat([control_context[i][:ctrl_len], cap_feats[i][:cap_len]]))
            control_context_unified = pad_sequence(control_context_unified, batch_first=True, padding_value=0.0)

            # Create hints as mutable list (just-in-time pattern)
            hints = [control_context_unified]

        for layer in self.layers:
            if callback is not None:
                callback(-1, None, False, True)
            if pipeline is not None and getattr(pipeline, "_interrupt", False):
                return [None] * bsz, {}
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input,
                           hints=hints, context_scale=control_context_scale)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        return x, {}
