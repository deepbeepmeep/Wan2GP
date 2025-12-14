# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
##### Enjoy this spagheti VRAM optimizations done by DeepBeepMeep !
# I am sure you are a nice person and as you copy this code, you will give me officially proper credits:
# Please link to https://github.com/deepbeepmeep/Wan2GP and @deepbeepmeep on twitter  

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.models.normalization import RMSNorm

from shared.attention import pay_attention

class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    def __call__(self):
        return self.module
        
def apply_rotary_emb_inplace(x_list: list, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_in = x_list.pop()
    dtype = x_in.dtype
    x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)  # [B, S, H, D//2, 2]
    x_in = None
    # freqs_cis shape: [B, S, rope_dim, 2] -> cos/sin: [B, S, 1, rope_dim] to broadcast over heads
    cos = freqs_cis[..., 0].unsqueeze(2)  # [B, S, 1, rope_dim]
    sin = freqs_cis[..., 1].unsqueeze(2)  # [B, S, 1, rope_dim]
    x0, x1 = x[..., 0], x[..., 1]  # [B, S, H, D//2]
    x0_orig = x0.clone()
    x0.mul_(cos).addcmul_(x1, sin, value=-1)
    x1.mul_(cos).addcmul_(x0_orig, sin)
    return x.flatten(3).to(dtype)


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
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


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


class Attention(nn.Module):
    """Simple attention module to match weight key structure."""
    def __init__(self, dim: int, n_heads: int, qk_norm: bool):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        # Sequential to match weight keys: to_out.0.weight
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False))

        self.norm_q = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else None

    def forward(self, h_list: list, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Compute self-attention with RoPE. h_list is cleared to free memory early."""
        h = h_list.pop()
        query = self.to_q(h)
        key = self.to_k(h)
        value = self.to_v(h); del h
        # Reshape to [batch, seq, heads, head_dim]
        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_heads, -1))
        value = value.unflatten(-1, (self.n_heads, -1))
        # Apply QK normalization
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)
        if freqs_cis is not None:
            q_list = [query]; del query
            query = apply_rotary_emb_inplace(q_list, freqs_cis)
            k_list = [key]; del key
            key = apply_rotary_emb_inplace(k_list, freqs_cis)
        # Compute attention
        dtype = query.dtype
        out = pay_attention([query, key, value]); del query, key, value
        return self.to_out(out.flatten(2, 3).to(dtype))


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,  # kept for API compatibility
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Attention module (named 'attention' to match weight keys)
        self.attention = Attention(dim, n_heads, qk_norm)

        # SwiGLU FFN: hidden_dim = dim * 8/3 ≈ 2.67x expansion
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.ffn_mult = 8 / 3  # For chunking: matches SwiGLU expansion factor
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

    def _apply_ffn_chunked(self, ffn_in: torch.Tensor) -> None:
        _, seq_len, dim = ffn_in.shape
        ffn_in_flat = ffn_in.reshape(-1, dim)
        chunk_size = max(int(seq_len // self.ffn_mult), 1)
        for ffn_chunk in torch.split(ffn_in_flat, chunk_size):
            ffn_chunk[...] = self.feed_forward(ffn_chunk)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            # In-place modulation for attention block
            scale_msa.add_(1.0)
            normed = self.attention_norm1(x)
            normed.mul_(scale_msa)
            attn_out = self.attention_norm2(self.attention([normed], freqs_cis))
            attn_out.mul_(gate_msa.tanh_())
            x.add_(attn_out); attn_out = None
            # In-place modulation for FFN block (chunked)
            scale_mlp.add_(1.0)
            normed = self.ffn_norm1(x)
            normed.mul_(scale_mlp)
            self._apply_ffn_chunked(normed)
            ffn_out = self.ffn_norm2(normed); normed = None
            ffn_out.mul_(gate_mlp.tanh_())
            x.add_(ffn_out); ffn_out = None
        else:
            x.add_(self.attention_norm2(self.attention([self.attention_norm1(x)], freqs_cis)))
            normed = self.ffn_norm1(x)
            self._apply_ffn_chunked(normed)
            x.add_(self.ffn_norm2(normed)); normed = None
        return x


class ZImageControlTransformerBlock(ZImageTransformerBlock):
    """Control block that processes control signals and produces skip connections."""
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        block_id=0
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, hints, x, **kwargs):
        # behold dbm magic !
        c = hints[0]
        hints[0] = None
        if self.block_id == 0:
            c = self.before_proj(c)
            bz = x.shape[0]
            if bz > c.shape[0]: c = c.repeat(bz, 1, 1 )
            c += x
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        hints[0] = c
        return c_skip
    


class BaseZImageTransformerBlock(ZImageTransformerBlock):
    """Base block that can optionally apply control hints."""
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        block_id=None
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
        self.block_id = block_id

    def forward(self, hidden_states, hints=None, hints_kwargs=None, context_scale=1.0, **kwargs):
        hints_processed = None
        if self.block_id is not None and hints is not None:
            hints_processed = self.control()(hints, **hints_kwargs)
        hidden_states = super().forward(hidden_states, **kwargs)
        if hints_processed is not None:
            hidden_states.add_(hints_processed, alpha=context_scale); hints_processed = None
        return hidden_states


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
        scale = self.adaLN_modulation(c)
        scale.add_(1.0)
        x = self.norm_final(x)
        x.mul_(scale.unsqueeze(1))
        return self.linear(x)


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
        """Precompute cos/sin frequencies for RoPE (real arithmetic, no complex)."""
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
                timestep = torch.arange(e, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                # Stack cos/sin in last dim: [end, dim//2, 2]
                freqs_cis.append(torch.stack([freqs.cos(), freqs.sin()], dim=-1))
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        # Cat on dim=-2 (D//2 dimension) since format is [S, D//2, 2]
        return torch.cat(result, dim=-2)


class ZImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        # Control-specific parameters (optional)
        control_layers_places=None,
        control_refiner_layers_places=None,
        control_in_dim=None,
        add_control_noise_refiner=False,
        enable_control=False,
        # Base model parameters
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

        # Control-specific attributes
        self.control_in_dim = control_in_dim if control_in_dim is not None else in_channels
        self.control_layers_places = [i for i in range(0, n_layers, 2)] if control_layers_places is None else control_layers_places
        self.control_refiner_layers_places = [i for i in range(0, n_refiner_layers)] if control_refiner_layers_places is None else control_refiner_layers_places
        self.add_control_noise_refiner = add_control_noise_refiner
        self.enable_control = enable_control

        assert 0 in self.control_layers_places
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers_places)}
        self.control_refiner_layers_mapping = {i: n for n, i in enumerate(self.control_refiner_layers_places)}

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

        # Noise refiner - use control version if enable_control and add_control_noise_refiner
        if enable_control and add_control_noise_refiner:
            self.noise_refiner = nn.ModuleList(
                [
                    BaseZImageTransformerBlock(
                        1000 + layer_id,
                        dim,
                        n_heads,
                        n_kv_heads,
                        norm_eps,
                        qk_norm,
                        modulation=True,
                        block_id=self.control_refiner_layers_mapping[layer_id] if layer_id in self.control_refiner_layers_places else None
                    )
                    for layer_id in range(n_refiner_layers)
                ]
            )
        else:
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

        # Main layers - use control version if enable_control
        if enable_control:
            self.layers = nn.ModuleList(
                [
                    BaseZImageTransformerBlock(
                        i,
                        dim,
                        n_heads,
                        n_kv_heads,
                        norm_eps,
                        qk_norm,
                        block_id=self.control_layers_mapping[i] if i in self.control_layers_places else None
                    )
                    for i in range(n_layers)
                ]
            )
        else:
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

        # Control-specific layers (only created when enable_control=True)
        if enable_control:
            # Control blocks
            self.control_layers = nn.ModuleList(
                [
                    ZImageControlTransformerBlock(
                        i,
                        dim,
                        n_heads,
                        n_kv_heads,
                        norm_eps,
                        qk_norm,
                        block_id=i
                    )
                    for i in self.control_layers_places
                ]
            )

            # Control patch embeddings
            control_all_x_embedder = {}
            for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
                x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * self.control_in_dim, dim, bias=True)
                control_all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder
            self.control_all_x_embedder = nn.ModuleDict(control_all_x_embedder)

            # Control noise refiner (for v2 control)
            if add_control_noise_refiner:
                self.control_noise_refiner = nn.ModuleList(
                    [
                        ZImageControlTransformerBlock(
                            1000 + layer_id,
                            dim,
                            n_heads,
                            n_kv_heads,
                            norm_eps,
                            qk_norm,
                            modulation=True,
                            block_id=layer_id
                        )
                        for layer_id in range(n_refiner_layers)
                    ]
                )
            else:
                # For v1 control
                self.control_noise_refiner = nn.ModuleList(
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

            self.adapt_control_model()

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        x_out_list = []
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x_out_list.append( 
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return torch.stack(x_out_list)

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify(
        self,
        all_image: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
        cap_padding_len: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []

        for i, image in enumerate(all_image):
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
                start=(cap_padding_len + 1, 0, 0),
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
            all_image_size,
            all_image_pos_ids,
            all_image_pad_mask,
        )

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

    @property
    def has_control(self) -> bool:
        """Returns True if the model has control layers enabled."""
        return self.enable_control

    def adapt_control_model(self):
        """Move control blocks to be submodules of their corresponding main layers."""
        if not self.enable_control or not hasattr(self, 'control_layers'):
            return

        modules_dict = {k: m for k, m in self.named_modules()}
        for model_layer, control_idx in self.control_layers_mapping.items():
            control_module = modules_dict[f"control_layers.{control_idx}"]
            target = modules_dict[f"layers.{model_layer}"]
            setattr(target, "control", ModuleWrapper(control_module))


        for model_layer, control_idx in self.control_refiner_layers_mapping.items():
            control_module = modules_dict[f"control_layers.{control_idx}"]
            target = modules_dict[f"noise_refiner.{model_layer}"]
            setattr(target, "control", ModuleWrapper(control_module))

    def prepare_forward_control_1_0(
        self,
        x,
        cap_feats,
        control_context,
        kwargs,
        t=None,
        patch_size=2,
        f_patch_size=1,
    ):
        """Control v1.0 processing (without noise refiner control)."""
        # embeddings
        bsz = len(control_context)
        device = control_context[0].device
        (
            control_context,
            x_size,
            x_pos_ids,
            x_inner_pad_mask,
        ) = self.patchify(control_context, patch_size, f_patch_size, cap_feats[0].size(0))

        # control_context embed & refine
        x_item_seqlens = [len(_) for _ in control_context]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        control_context = torch.cat(control_context, dim=0)
        control_context = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control_context)

        # Match t_embedder output dtype to control_context for layerwise casting compatibility
        adaln_input = t.type_as(control_context)
        control_context[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        control_context = list(control_context.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        control_context = pad_sequence(control_context, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        for layer in self.control_noise_refiner:
            control_context = layer(control_context, x_attn_mask, x_freqs_cis, adaln_input)

        # unified
        cap_item_seqlens = [len(_) for _ in cap_feats]
        control_context_unified = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            control_context_unified.append(torch.cat([control_context[i][:x_len], cap_feats[i][:cap_len]]))
        control_context_unified = pad_sequence(control_context_unified, batch_first=True, padding_value=0.0)
        c = control_context_unified

        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        return new_kwargs, c

    def prepare_forward_control_2_0_refiner(
        self,
        x,
        cap_feats,
        control_context,
        kwargs,
        t=None,
        patch_size=2,
        f_patch_size=1,
    ):
        """Control v2.0 refiner processing."""
        # embeddings
        bsz = len(control_context)
        device = control_context[0].device
        (
            control_context,
            control_context_size,
            control_context_pos_ids,
            control_context_inner_pad_mask,
        ) = self.patchify(control_context, patch_size, f_patch_size, cap_feats[0].size(0))

        # control_context embed & refine
        control_context_item_seqlens = [len(_) for _ in control_context]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in control_context_item_seqlens)
        control_context_max_item_seqlen = max(control_context_item_seqlens)

        control_context = torch.cat(control_context, dim=0)
        control_context = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control_context)

        # Match t_embedder output dtype to control_context for layerwise casting compatibility
        adaln_input = t.type_as(control_context)
        control_context[torch.cat(control_context_inner_pad_mask)] = self.x_pad_token
        control_context = list(control_context.split(control_context_item_seqlens, dim=0))
        control_context_freqs_cis = list(self.rope_embedder(torch.cat(control_context_pos_ids, dim=0)).split(control_context_item_seqlens, dim=0))

        control_context = pad_sequence(control_context, batch_first=True, padding_value=0.0)
        control_context_freqs_cis = pad_sequence(control_context_freqs_cis, batch_first=True, padding_value=0.0)
        control_context_attn_mask = torch.zeros((bsz, control_context_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(control_context_item_seqlens):
            control_context_attn_mask[i, :seq_len] = 1
        c = control_context

        new_kwargs = dict(
            x=x,
            attn_mask=control_context_attn_mask,
            freqs_cis=control_context_freqs_cis,
            adaln_input=adaln_input,
        )
        new_kwargs.update(kwargs)

        return new_kwargs, c, control_context_item_seqlens
    
    def prepare_forward_control_2_0_layers(
        self,
        x,
        cap_feats,
        control_context,
        control_context_item_seqlens,
        kwargs,
    ):
        """Control v2.0 layers processing."""
        control_context_len = control_context_item_seqlens
        cap_len = cap_feats.shape[1]
        control_context_unified= torch.cat([control_context[:control_context_len], cap_feats[:cap_len]], dim=1)

        c = pad_sequence(control_context_unified, batch_first=True, padding_value=0.0)

        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        return new_kwargs, c

    def forward(
        self,
        x_list: List[torch.Tensor],
        t,
        cap_feats_list: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        control_context_list=None,
        control_context_scale=1.0,
        callback=None,
        pipeline=None,
    ):
        """Forward pass with list-based processing (outer loop over layers, inner loop over samples)."""
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        num_noise_samples = len(x_list)
        assert len(cap_feats_list) == num_noise_samples, "cap_feats_list must match x_list length"

        device = x_list[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        # Patchify and embed each (x, cap_feats) pair
        x_embedder = self.all_x_embedder[f"{patch_size}-{f_patch_size}"]
        embedded_x_list = []
        cap_embedded_list = []
        x_seqlen = None
        cap_seqlen = None
        x_freqs_cis = None
        cap_freqs_cis = None

        for i,(x,cap_feats) in enumerate(zip(x_list, cap_feats_list)):
            bsz = x.shape[0]
            ( x_patches, cap_out, x_i_size, x_pos_ids, cap_pos_ids, x_inner_pad_mask, cap_inner_pad_mask) = self.patchify_and_embed(x, [cap_feats]* bsz, patch_size, f_patch_size)

            # Store x_size from first sample (all should be same shape)
            if i==0:
                x_seqlen = len(x_patches[0])
                cap_seqlen = len(cap_out[0])
                x_freqs_cis = self.rope_embedder(torch.cat(x_pos_ids[:1], dim=0))
                cap_freqs_cis = self.rope_embedder(torch.cat(cap_pos_ids[:1], dim=0))
                x_size = x_i_size
            # Embed x
            # x_patches = 
            x_embedded = x_embedder(torch.stack(x_patches))
            x_embedded[torch.stack(x_inner_pad_mask)] = self.x_pad_token
            embedded_x_list.append(x_embedded)
            # Embed cap_feats
            cap_embedded = self.cap_embedder(torch.stack(cap_out))
            cap_embedded[torch.stack(cap_inner_pad_mask)] = self.cap_pad_token
            cap_embedded_list.append(cap_embedded)
            x = cap_feats = None

        # Match t_embedder output dtype
        adaln_input = t.type_as(embedded_x_list[0])[:1]  # Same timestep for all samples

        # Prepare attention masks and freqs
        x_attn_mask = torch.ones((1, x_seqlen), dtype=torch.bool, device=device)
        x_freqs_i = x_freqs_cis.unsqueeze(0)

        # Control processing - compute hints with batch dimension [num_samples, seq, dim]
        refiner_hints_tuple_list = []  # List of tensors with batch dim, one per layer
        ctrl_ctx_tensor_list = []  # control context for v2.0 layers
        ctrl_seqlens_list = []
        refiner_hints_kwargs_list= []
        any_control = control_context_list is not None and self.enable_control
        control_V2 = any_control and self.add_control_noise_refiner
        if control_V2:
            # Control v2.0 refiner - process all samples together (no duplication)
            # Stack all embedded_x into batch: [num_samples, seq, dim]
            kwargs = dict(
                attn_mask=x_attn_mask,
                freqs_cis=x_freqs_i, 
                adaln_input=adaln_input,
            )                
            for i,(x_batch,cap_embedded_ref, control_ctx_input) in enumerate(zip(embedded_x_list, cap_embedded_list, control_context_list)):

                # Pass all samples at once - hints will have shape [num_samples, seq, dim] each #ctrl_ctx_tensor,
                refiner_hints_kwargs, refiner_hints_tuple, ctrl_seqlens = self.prepare_forward_control_2_0_refiner(
                    x_batch,
                    [cap_embedded_ref] ,
                    [control_ctx_input.squeeze(0)] ,
                    kwargs = kwargs,
                    t=adaln_input, patch_size=patch_size, f_patch_size=f_patch_size
                )
                refiner_hints_kwargs_list.append(refiner_hints_kwargs)
                refiner_hints_tuple_list.append([refiner_hints_tuple])
                # ctrl_ctx_tensor_list.append(ctrl_ctx_tensor)
                ctrl_seqlens_list.append(ctrl_seqlens[0])
                x_batch = cap_embedded_ref = control_ctx_input = None

        # Noise refiner
        for layer in self.noise_refiner:
            for i, x_i in enumerate(embedded_x_list):
                kwargs = dict(attn_mask=x_attn_mask, freqs_cis=x_freqs_i, adaln_input=adaln_input)
                if control_V2:  # v2 control
                    # kwargs["hints"] = refiner_hints_tuple_list[i]
                    kwargs["hints"] = refiner_hints_tuple_list[i]
                    kwargs["hints_kwargs"] = refiner_hints_kwargs_list[i]                        
                    kwargs["context_scale"] = control_context_scale
                embedded_x_list[i] = layer(x_i, **kwargs)
                x_i = None

        if control_V2:
            # finish processing v2 control hints
            for hints, hints_kwargs in zip(refiner_hints_tuple_list, refiner_hints_kwargs_list):
                for layer in self.control_layers[len(self.control_refiner_layers_places):]:
                    layer(hints, **hints_kwargs)
                ctrl_ctx_tensor_list.append(hints[0])
                hints = hints_kwargs = None

        # Context refiner
        cap_attn_mask = torch.ones((1, cap_seqlen), dtype=torch.bool, device=device)
        cap_freqs_i = cap_freqs_cis.unsqueeze(0)
        for layer in self.context_refiner:
            for i, cap_i in enumerate(cap_embedded_list):
                cap_embedded_list[i] = layer( cap_i, attn_mask=cap_attn_mask, freqs_cis=cap_freqs_i)
                cap_i = None

        # Create unified (x + cap) for each sample
        unified_list = []
        for i,(embedded_x, cap_embedded) in enumerate(zip(embedded_x_list, cap_embedded_list)):
            unified_list.append(torch.cat([embedded_x, cap_embedded], dim=1))
            embedded_x_list[i] = None
        unified_freqs_i = torch.cat([x_freqs_i, cap_freqs_i], dim=1)
        unified_seqlen = x_seqlen + cap_seqlen
        unified_attn_mask = torch.ones((1, unified_seqlen), dtype=torch.bool, device=device)
        hints_list = []
        hints_kwargs_list = []

        # Compute control hints for main layers
        if any_control:
            cap_embedded_ref = cap_embedded_list[0]
            control_kwargs = dict(
                attn_mask=unified_attn_mask,
                freqs_cis=unified_freqs_i,
                adaln_input=adaln_input.expand(num_noise_samples, -1),
            )
            if control_V2:
                # Control v2.0 
                for unified_batch, ctrl_ctx_tensor, ctrl_seqlens in zip(unified_list, ctrl_ctx_tensor_list, ctrl_seqlens_list):
                    hints_kwargs, hints_tuple = self.prepare_forward_control_2_0_layers( unified_batch, cap_embedded_ref , ctrl_ctx_tensor, ctrl_seqlens, control_kwargs )
                    hints_kwargs_list.append(hints_kwargs)
                    hints_list.append([hints_tuple])
                    unified_batch = ctrl_ctx_tensor = ctrl_seqlens = None
            else:
                # Control v1.0 
                for unified_batch, ctrl_ctx_tensor in zip(unified_list, control_context_list):
                    hints_kwargs, hints_tuple = self.prepare_forward_control_1_0( unified_batch, cap_embedded_ref, ctrl_ctx_tensor, control_kwargs, t=adaln_input, patch_size=patch_size, f_patch_size=f_patch_size)
                    hints_kwargs_list.append(hints_kwargs)
                    hints_list.append([hints_tuple])
                    unified_batch = ctrl_ctx_tensor = None

        if len(hints_list) == 0:
            hints_list = [None] * len(x_list)
            hints_kwargs_list = [None] * len(x_list)

        # Main layers
        for layer in self.layers:
            if callback is not None:
                callback(-1, None, False, True)
            if pipeline is not None and getattr(pipeline, "_interrupt", False):
                return None

            for i, (unified_i, hints, hints_kwargs) in enumerate(zip(unified_list, hints_list, hints_kwargs_list)):
                kwargs = {}
                if hints is not None:
                    kwargs.update(dict(hints= hints, hints_kwargs = hints_kwargs, context_scale = control_context_scale))
                unified_list[i] = layer(unified_i, attn_mask=unified_attn_mask, freqs_cis=unified_freqs_i, adaln_input=adaln_input, **kwargs)
                unified_i = hints = kwargs = hints_kwargs = None

        # Final layer and unpatchify
        output_list = []
        final_layer = self.all_final_layer[f"{patch_size}-{f_patch_size}"]
        for i in range(num_noise_samples):
            final_out = final_layer(unified_list[i], adaln_input)
            final_out = final_out[:, :x_seqlen]
            unpatchified = self.unpatchify(final_out, x_size, patch_size, f_patch_size)
            output_list.append(unpatchified)
            final_out = None

        return output_list
