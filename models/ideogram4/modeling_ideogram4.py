from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.attention import pay_attention

from .constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, QWEN3_VL_ACTIVATION_LAYERS


@dataclass
class Ideogram4Config:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512
    in_channels: int = 128
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)
    rope_theta: int = 5_000_000
    mrope_section: tuple[int, ...] = (24, 20, 20)
    norm_eps: float = 1e-5


def get_linear_split_map(hidden_size: int = Ideogram4Config.emb_dim) -> dict[str, dict[str, list[int] | list[str]]]:
    return {"qkv": {"mapped_modules": ["q", "k", "v"], "split_sizes": [hidden_size, hidden_size, hidden_size]}}


def _ffn_chunk_size(seq_len: int, hidden_size: int, intermediate_size: int) -> int:
    if seq_len <= 1024:
        return 0
    chunk_size = seq_len * hidden_size // max(intermediate_size, 1)
    return max(128, min(seq_len, chunk_size))


def _take_tensor(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(value, list):
        tensor = value[0]
        value.clear()
        return tensor
    return value


def _apply_rotary_pos_emb_(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    half = q.shape[-1] // 2
    scratch = torch.empty_like(q[..., :half])
    for x in (q, k):
        x1 = x[..., :half]
        x2 = x[..., half:]
        scratch.copy_(x1)
        x1.mul_(cos).addcmul_(x2, sin, value=-1)
        x2.mul_(cos).addcmul_(scratch, sin)
    del scratch


class Ideogram4MRoPE(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: int, mrope_section: tuple[int, ...]) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim
        self.base = base

    def reset_inv_freq(self) -> None:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.inv_freq = inv_freq

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = position_ids.shape[0]
        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)
        freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)

        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            freqs_t[..., offset:length:3] = freqs[axis][..., offset:length:3]

        return freqs_t.cos(), freqs_t.sin()


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        x = _take_tensor(x)
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor | list[torch.Tensor], segment_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = _take_tensor(x)
        batch_size, seq_len, _ = x.shape
        shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.q(x).view(*shape)
        k = self.k(x).view(*shape)
        v = self.v(x).view(*shape)
        x = None
        q = self.norm_q([q])
        k = self.norm_k([k])
        _apply_rotary_pos_emb_(q, k, cos, sin)
        attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(2)
        qkv_list = [q, k, v]
        q = k = v = None
        out = pay_attention(qkv_list, attention_mask=attn_mask, recycle_q=True)
        out = out.reshape(batch_size, seq_len, self.hidden_size)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        x = _take_tensor(x)
        seq_len = x.shape[-2]
        chunk_size = _ffn_chunk_size(seq_len, self.dim, self.hidden_dim)
        if chunk_size == 0:
            hidden = self.w1(x)
            F.silu(hidden, inplace=True)
            gate = self.w3(x)
            x = None
            hidden.mul_(gate)
            del gate
            out = self.w2(hidden)
            del hidden
            return out
        out = x.new_empty(*x.shape[:-1], self.dim)
        for start in range(0, seq_len, chunk_size):
            chunk = x.narrow(-2, start, min(chunk_size, seq_len - start))
            hidden = self.w1(chunk)
            F.silu(hidden, inplace=True)
            gate = self.w3(chunk)
            hidden.mul_(gate)
            del gate
            chunk_out = self.w2(hidden)
            out.narrow(-2, start, chunk_out.shape[-2]).copy_(chunk_out)
            del chunk, hidden, chunk_out
        x = None
        return out


class Ideogram4TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, norm_eps: float, adanln_dim: int) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)
        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, adaln_input: torch.Tensor) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        mod = None
        gate_msa.tanh_()
        gate_mlp.tanh_()
        scale_msa.add_(1.0)
        scale_mlp.add_(1.0)

        attn_input = self.attention_norm1(x)
        attn_input.mul_(scale_msa)
        del scale_msa
        attn_input_list = [attn_input]
        attn_input = None
        attn_out = self.attention(attn_input_list, segment_ids=segment_ids, cos=cos, sin=sin)
        attn_out = self.attention_norm2([attn_out])
        attn_out.mul_(gate_msa)
        del gate_msa
        x.add_(attn_out)
        del attn_out

        ffn_input = self.ffn_norm1(x)
        ffn_input.mul_(scale_mlp)
        del scale_mlp
        ffn_input_list = [ffn_input]
        ffn_input = None
        ffn_out = self.feed_forward(ffn_input_list)
        ffn_out_list = [ffn_out]
        ffn_out = None
        ffn_out = self.ffn_norm2(ffn_out_list)
        ffn_out.mul_(gate_mlp)
        del gate_mlp
        x.add_(ffn_out)
        del ffn_out
        return x


def _sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 == 1 else emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(getattr(self.mlp_in, "compute_dtype", None) or self.mlp_in.weight.dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = self.adaln_modulation(F.silu(c))
        scale.add_(1.0)
        x = self.norm_final(x)
        x.mul_(scale)
        del scale
        return self.linear(x)


class Ideogram4Transformer(nn.Module):
    def __init__(self, config: Ideogram4Config) -> None:
        super().__init__()
        self.config = config
        head_dim = config.emb_dim // config.num_heads
        self.input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(config.llm_features_dim, config.emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(config.emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)
        self.embed_image_indicator = nn.Embedding(2, config.emb_dim)
        self.rotary_emb = Ideogram4MRoPE(head_dim=head_dim, base=config.rope_theta, mrope_section=config.mrope_section)
        self.layers = nn.ModuleList([
            Ideogram4TransformerBlock(config.emb_dim, config.intermediate_size, config.num_heads, config.norm_eps, config.adanln_dim)
            for _ in range(config.num_layers)
        ])
        self.final_layer = Ideogram4FinalLayer(config.emb_dim, config.in_channels, config.adanln_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        *,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
    ) -> torch.Tensor | None:
        batch_size, seq_len, in_channels = x.shape
        if in_channels != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels} input channels, got {in_channels}")

        param_dtype = getattr(self.input_proj, "compute_dtype", None) or self.input_proj.weight.dtype
        x = x.to(param_dtype)
        t = t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)

        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)
        text_len = llm_features.shape[1]
        llm_token_mask_text = llm_token_mask[:, :text_len] if text_len > 0 else None
        if text_len > 0:
            llm_features.mul_(llm_token_mask_text)
        x.mul_(output_image_mask)

        x = self.input_proj(x)
        x.mul_(output_image_mask)
        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = self.adaln_proj(t_cond)
        F.silu(adaln_input, inplace=True)

        img_indicator_ids = (indicator == OUTPUT_IMAGE_INDICATOR).long()
        if text_len > 0:
            llm_features = self.llm_cond_norm([llm_features])
            llm_embed = self.llm_cond_proj(llm_features)
            del llm_features
            llm_embed.mul_(llm_token_mask_text)
            x[:, :text_len].add_(llm_embed)
            del llm_embed
        image_indicator = self.embed_image_indicator(img_indicator_ids)
        x.add_(image_indicator)
        del image_indicator
        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        for layer in self.layers:
            x = layer(x, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)
            if getattr(self, "_interrupt", False):
                return None

        out = self.final_layer(x, adaln_input)
        out = out.float()
        out.mul_(output_image_mask)
        return out
