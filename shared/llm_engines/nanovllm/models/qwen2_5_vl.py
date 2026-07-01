from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..layers.activation import SiluAndMul
from ..layers.attention import Attention
from ..layers.embed_head import VocabParallelEmbedding
from ..layers.layernorm import RMSNorm
from ..layers.linear import ColumnParallelLinear, RowParallelLinear
from ..utils.context import get_context

_MROPE_FREQ_CACHE: dict[tuple[str, int, int, float], torch.Tensor] = {}


def _get_tp_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _build_mrope_freq_cache(device: torch.device, max_position: int, rotary_dim: int, rope_theta: float) -> torch.Tensor:
    cache_key = (str(device), int(max_position), int(rotary_dim), float(rope_theta))
    cached = _MROPE_FREQ_CACHE.get(cache_key)
    if cached is not None:
        return cached
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    cached = torch.einsum("i,j->ij", positions, inv_freq)
    _MROPE_FREQ_CACHE[cache_key] = cached
    return cached


def clear_qwen25_vl_runtime_caches(device: torch.device | None = None) -> None:
    if device is None:
        _MROPE_FREQ_CACHE.clear()
        return
    device_key = str(device)
    for cache_key in [key for key in _MROPE_FREQ_CACHE if key[0] == device_key]:
        _MROPE_FREQ_CACHE.pop(cache_key, None)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1).to(dtype=x.dtype)


class Qwen2_5_VLTextRotaryEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self.mrope_section = list(rope_scaling.get("mrope_section", [16, 24, 24]))
        self.max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rope_theta = float(getattr(config, "rope_theta", 1000000.0))
        self.rotary_dim = head_dim

    @staticmethod
    def _normalize_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.ndim == 1:
            return position_ids.unsqueeze(0).expand(3, -1)
        if position_ids.ndim == 2:
            return position_ids if position_ids.shape[0] == 3 else position_ids.reshape(1, -1).expand(3, -1)
        if position_ids.ndim == 3:
            if position_ids.shape[0] == 4:
                position_ids = position_ids[1:]
            return position_ids.reshape(3, -1)
        raise RuntimeError(f"Unsupported Qwen2.5-VL position_ids shape: {tuple(position_ids.shape)}")

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = self._normalize_position_ids(position_ids).long()
        freq_cache = _build_mrope_freq_cache(
            position_ids.device,
            max_position=self.max_position_embeddings,
            rotary_dim=self.rotary_dim,
            rope_theta=self.rope_theta,
        )
        freqs = torch.stack([freq_cache[position_ids[axis]] for axis in range(3)], dim=0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        sections = [int(section) * 2 for section in self.mrope_section]
        cos = torch.cat([part[axis] for axis, part in enumerate(cos.split(sections, dim=-1))], dim=-1)
        sin = torch.cat([part[axis] for axis, part in enumerate(sin.split(sections, dim=-1))], dim=-1)
        return cos.unsqueeze(1), sin.unsqueeze(1)


class Qwen2_5_VLAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        tp_size = _get_tp_size()
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.scaling = self.head_dim**-0.5
        self.q_proj = ColumnParallelLinear(config.hidden_size, self.total_num_heads * self.head_dim, bias=True)
        self.k_proj = ColumnParallelLinear(config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=True)
        self.v_proj = ColumnParallelLinear(config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=True)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, config.hidden_size, bias=False)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        cos, sin = position_embeddings
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return self.o_proj(self.attn(q, k, v).flatten(1, -1))


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise RuntimeError(f"Unsupported Qwen2.5-VL activation for nano decoder: {config.hidden_act}")
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(torch.cat((self.gate_proj(hidden_states), self.up_proj(hidden_states)), dim=-1)))


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = Qwen2_5_VLAttention(config)
        self.mlp = Qwen2_5_VLMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn(self.input_layernorm(hidden_states), position_embeddings)
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen2_5_VLTiedLMHead(nn.Module):
    def __init__(self, embed_tokens: VocabParallelEmbedding) -> None:
        super().__init__()
        object.__setattr__(self, "_embed_tokens", embed_tokens)

    @property
    def weight(self):
        return object.__getattribute__(self, "_embed_tokens").weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None and hidden_states.ndim == 2:
            hidden_states = hidden_states[(context.cu_seqlens_q[1:] - 1).to(device=hidden_states.device, dtype=torch.long)]
        return F.linear(hidden_states, self.weight)


class Qwen2_5_VLForCausalLM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        config = getattr(config, "text_config", config)
        if bool(getattr(config, "use_sliding_window", False)) or any(layer_type != "full_attention" for layer_type in getattr(config, "layer_types", ["full_attention"])):
            raise RuntimeError("Qwen2.5-VL nano decoder only supports full-attention configs.")
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(int(config.vocab_size), int(config.hidden_size))
        self.rotary_emb = Qwen2_5_VLTextRotaryEmbedding(config)
        self.layers = nn.ModuleList([Qwen2_5_VLDecoderLayer(config) for _ in range(int(config.num_hidden_layers))])
        self.norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.lm_head = Qwen2_5_VLTiedLMHead(self.embed_tokens)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if position_ids is not None:
            positions = position_ids
        if positions is None:
            seq_len = inputs_embeds.shape[-2] if inputs_embeds is not None else input_ids.numel()
            positions = torch.arange(seq_len, device=(inputs_embeds.device if inputs_embeds is not None else input_ids.device), dtype=torch.long)
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        position_embeddings = self.rotary_emb(hidden_states, positions)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        return self.norm(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


__all__ = ["Qwen2_5_VLForCausalLM", "clear_qwen25_vl_runtime_caches"]
