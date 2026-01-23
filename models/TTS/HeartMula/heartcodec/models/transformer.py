import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self._cache = {}

    def get_sin_cos(self, seq_len: int, device, dtype):
        key = (seq_len, device, dtype)
        cached = self._cache.get(key, None)
        if cached is not None and cached[0].device == device:
            return cached
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim)
        )
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        sin = freqs.sin()
        cos = freqs.cos()
        self._cache[key] = (sin, cos)
        return sin, cos

    def apply_rotary(
        self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
    ) -> torch.Tensor:
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 : self.dim]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x[..., : self.dim])
        return (x[..., : self.dim] * cos.unsqueeze(-1)).reshape_as(
            x[..., : self.dim]
        ) + (x_rot * sin.unsqueeze(-1)).reshape_as(x[..., : self.dim])


class LlamaAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        bias: bool = False,
        dropout: float = 0.0,
        rope_dim: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim
        self.cross_attention_dim = cross_attention_dim
        self.q_proj = nn.Linear(dim, self.inner_dim, bias=bias)
        k_in = dim if cross_attention_dim is None else cross_attention_dim
        self.k_proj = nn.Linear(k_in, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(k_in, self.inner_dim, bias=bias)
        self.o_proj = nn.Linear(self.inner_dim, dim, bias=bias)
        self.dropout = dropout
        self.rope_dim = rope_dim if rope_dim is not None else head_dim
        self.rope = RotaryEmbedding(self.rope_dim)
        self.use_sdpa = use_sdpa
        self._has_sdpa = hasattr(F, "scaled_dot_product_attention")

    def _shape(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self._shape(self.q_proj(x), b, t)
        if encoder_hidden_states is None:
            k = self._shape(self.k_proj(x), b, t)
            v = self._shape(self.v_proj(x), b, t)
        else:
            _, tk, _ = encoder_hidden_states.shape
            k = self._shape(self.k_proj(encoder_hidden_states), b, tk)
            v = self._shape(self.v_proj(encoder_hidden_states), b, tk)

        rope_dim = min(self.rope_dim, self.head_dim)
        seq_len_for_rope = k.shape[-2]
        sin, cos = self.rope.get_sin_cos(
            seq_len_for_rope, device=x.device, dtype=x.dtype
        )

        def apply_rope_vec(tensor):
            head = tensor[..., :rope_dim]
            tail = tensor[..., rope_dim:]
            _, _, tt, _ = head.shape
            head = head.view(b, self.n_heads, tt, rope_dim // 2, 2)
            sin_ = sin.view(1, 1, tt, rope_dim // 2, 1)
            cos_ = cos.view(1, 1, tt, rope_dim // 2, 1)
            x1 = head[..., 0:1]
            x2 = head[..., 1:2]
            rot = torch.cat([x1 * cos_ - x2 * sin_, x1 * sin_ + x2 * cos_], dim=-1)
            rot = rot.view(b, self.n_heads, tt, rope_dim)
            return torch.cat([rot, tail], dim=-1)

        q = apply_rope_vec(q)
        k = apply_rope_vec(k)

        if self.use_sdpa and self._has_sdpa:
            s = k.shape[-2]
            attn_mask_sdpa = None
            if attention_mask is not None:
                m = attention_mask

                if m.dim() == 2 and m.shape == (b, s):
                    m = m[:, None, None, :]
                elif m.dim() == 3 and m.shape[-2] == 1:
                    m = m[:, None, :, :]
                elif m.dim() == 3 and m.shape[-2] == t:
                    m = m[:, None, :, :]
                elif m.dim() == 4 and m.shape[1] == 1:
                    pass
                attn_mask_sdpa = m

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask_sdpa,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            out = out.transpose(1, 2).contiguous().view(b, t, self.inner_dim)
            return self.o_proj(out)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn = attn_scores.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.inner_dim)
        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        multiple_of: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.up(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.down(x)


class LlamaTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        mlp_multiple_of: int = 256,
        dropout: float = 0.0,
        attention_bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        use_ada_layer_norm_single: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, 1e-6)
        self.attn = LlamaAttention(
            dim,
            n_heads,
            head_dim,
            bias=attention_bias,
            dropout=dropout,
            rope_dim=head_dim,
            cross_attention_dim=None,
        )
        self.cross_attn = None
        if cross_attention_dim is not None:
            self.cross_attn_norm = RMSNorm(dim, 1e-6)
            self.cross_attn = LlamaAttention(
                dim,
                n_heads,
                head_dim,
                bias=attention_bias,
                dropout=dropout,
                rope_dim=head_dim,
                cross_attention_dim=cross_attention_dim,
            )
        self.mlp_norm = RMSNorm(dim, 1e-6)
        self.mlp = LlamaMLP(dim, multiple_of=mlp_multiple_of, dropout=dropout)
        self.use_ada_layer_norm_single = use_ada_layer_norm_single
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_ada_layer_norm_single:
            batch_size = x.shape[0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)

            norm_hidden_states = self.attn_norm(x)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            h = self.attn(norm_hidden_states, attention_mask=attention_mask)
            h = gate_msa * h
            x = x + h

            norm_hidden_states = self.mlp_norm(x)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            h = self.mlp(norm_hidden_states)
            h = gate_mlp * h
            x = x + h
            return x

        h = self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + h
        h = self.mlp(self.mlp_norm(x))
        x = x + h
        return x


class ProjectLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, kernel_size=1, dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.ffn_1 = nn.Conv1d(
            hidden_size, filter_size, kernel_size, padding=kernel_size // 2
        )
        self.ffn_2 = nn.Linear(filter_size, filter_size)

    def forward(self, x):
        x = self.ffn_1(x.transpose(1, 2)).transpose(1, 2)
        x = x * self.kernel_size**-0.5
        x = self.ffn_2(x)
        return x


class LlamaTransformer(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        out_channels: int,
        num_layers: int = 12,
        num_layers_2: int = 2,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        inner_dim_2 = inner_dim * 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dim = inner_dim
        self.inner_dim_2 = inner_dim_2
        self.dropout = dropout

        self.proj_in = ProjectLayer(in_channels, inner_dim, kernel_size=3)

        use_ada_single = norm_type == "ada_norm_single"
        self.transformer_blocks = nn.ModuleList(
            [
                LlamaTransformerBlock(
                    dim=inner_dim,
                    n_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    dropout=dropout,
                    attention_bias=False,
                    cross_attention_dim=cross_attention_dim,
                    use_ada_layer_norm_single=use_ada_single,
                )
                for _ in range(num_layers)
            ]
        )

        self.transformer_blocks_2 = nn.ModuleList(
            [
                LlamaTransformerBlock(
                    dim=inner_dim_2,
                    n_heads=num_attention_heads,
                    head_dim=attention_head_dim * 2,
                    dropout=dropout,
                    attention_bias=False,
                    cross_attention_dim=cross_attention_dim,
                    use_ada_layer_norm_single=use_ada_single,
                )
                for _ in range(num_layers_2)
            ]
        )

        self.connection_proj = ProjectLayer(
            in_channels + inner_dim, inner_dim_2, kernel_size=3
        )
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.norm_out_2 = nn.LayerNorm(inner_dim_2, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim**0.5
        )
        self.scale_shift_table_2 = nn.Parameter(
            torch.randn(2, inner_dim_2) / inner_dim_2**0.5
        )
        self.proj_out = ProjectLayer(inner_dim_2, out_channels, kernel_size=3)
        self.adaln_single = AdaLayerNormSingleFlow(inner_dim)
        self.adaln_single_2 = AdaLayerNormSingleFlow(inner_dim_2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
    ):
        s = self.proj_in(hidden_states)

        embedded_timestep = None
        timestep_mod = None
        if self.adaln_single is not None and timestep is not None:
            timestep_mod, embedded_timestep = self.adaln_single(
                timestep, hidden_dtype=s.dtype
            )
        for blk in self.transformer_blocks:
            s = blk(s, timestep=timestep_mod)

        if embedded_timestep is None:
            embedded_timestep = torch.zeros(
                s.size(0), s.size(-1), device=s.device, dtype=s.dtype
            )

        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None]
        ).chunk(2, dim=1)
        s = self.norm_out(s)
        s = s * (1 + scale) + shift

        x = torch.cat([hidden_states, s], dim=-1)
        x = self.connection_proj(x)

        embedded_timestep_2 = None
        timestep_mod_2 = None
        if self.adaln_single_2 is not None and timestep is not None:
            timestep_mod_2, embedded_timestep_2 = self.adaln_single_2(
                timestep, hidden_dtype=x.dtype
            )
        for blk in self.transformer_blocks_2:
            x = blk(x, timestep=timestep_mod_2)

        if embedded_timestep_2 is None:
            embedded_timestep_2 = torch.zeros(
                x.size(0), x.size(-1), device=x.device, dtype=x.dtype
            )

        shift_2, scale_2 = (
            self.scale_shift_table_2[None] + embedded_timestep_2[:, None]
        ).chunk(2, dim=1)
        x = self.norm_out_2(x)
        x = x * (1 + scale_2) + shift_2

        out = self.proj_out(x)
        return out


class PixArtAlphaCombinedFlowEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, size_emb_dim: int):
        super().__init__()
        self.flow_t_size = 512
        self.outdim = size_emb_dim
        self.timestep_embedder = TimestepEmbedding(
            in_channels=self.flow_t_size, time_embed_dim=embedding_dim
        )

    def timestep_embedding(self, timesteps, max_period=10000, scale=1000):
        half = self.flow_t_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, device=timesteps.device)
            / half
        ).type(timesteps.type())
        args = timesteps[:, None] * freqs[None] * scale
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.flow_t_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, timestep, hidden_dtype):
        timesteps_proj = self.timestep_embedding(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        return timesteps_emb


class AdaLayerNormSingleFlow(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.emb = PixArtAlphaCombinedFlowEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = (
            -math.log(10000)
            * torch.arange(0, half_dim, device=timesteps.device)
            / (half_dim - self.downscale_freq_shift)
        )
        emb = torch.exp(exponent)[None, :] * timesteps[:, None]
        if self.flip_sin_to_cos:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.num_channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb
