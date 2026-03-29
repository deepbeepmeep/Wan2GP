from __future__ import annotations

import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

logging.getLogger("fla.utils").setLevel(logging.ERROR)

try:
    from flash_attn import flash_attn_varlen_func
except Exception:  # pragma: no cover
    flash_attn_varlen_func = None

try:
    # from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    causal_conv1d_fn = None
    causal_conv1d_update = None
except Exception:  # pragma: no cover
    causal_conv1d_fn = None
    causal_conv1d_update = None

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fast_chunk_gated_delta_rule
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fast_recurrent_gated_delta_rule
except Exception:  # pragma: no cover
    fast_chunk_gated_delta_rule = None
    fast_recurrent_gated_delta_rule = None

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
except Exception:  # pragma: no cover
    FusedRMSNormGated = None
    ShortConvolution = None

_DEFAULT_FLASH_ATTN_VARLEN_FUNC = flash_attn_varlen_func
_DEFAULT_FAST_CHUNK_GATED_DELTA_RULE = fast_chunk_gated_delta_rule
_DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE = fast_recurrent_gated_delta_rule
_DEFAULT_FUSED_RMSNORM_GATED = FusedRMSNormGated
_DEFAULT_SHORT_CONVOLUTION = ShortConvolution


def configure_qwen35_safe_legacy_kernels(enabled: bool) -> None:
    global flash_attn_varlen_func, fast_chunk_gated_delta_rule, fast_recurrent_gated_delta_rule
    global FusedRMSNormGated, ShortConvolution
    if enabled:
        flash_attn_varlen_func = None
        fast_chunk_gated_delta_rule = None
        fast_recurrent_gated_delta_rule = None
        FusedRMSNormGated = None
        ShortConvolution = None
        return
    flash_attn_varlen_func = _DEFAULT_FLASH_ATTN_VARLEN_FUNC
    fast_chunk_gated_delta_rule = _DEFAULT_FAST_CHUNK_GATED_DELTA_RULE
    fast_recurrent_gated_delta_rule = _DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE
    FusedRMSNormGated = _DEFAULT_FUSED_RMSNORM_GATED
    ShortConvolution = _DEFAULT_SHORT_CONVOLUTION

from ..layers.activation import SiluAndMul
from ..layers.attention import Attention
from ..layers.layernorm import RMSNorm
from ..layers.linear import ColumnParallelLinear, RowParallelLinear
from ..utils.context import get_context

_CU_SEQLENS_CACHE: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}
_MROPE_FREQ_CACHE: dict[tuple[str, int, int, float], torch.Tensor] = {}


def _safe_legacy_kernels_enabled(config) -> bool:
    return bool(getattr(config, "_prompt_enhancer_safe_legacy", False))


def _get_tp_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _build_cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    normalized_lengths = tuple(int(length) for length in lengths)
    cache_key = (normalized_lengths, str(device))
    cached = _CU_SEQLENS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    total = 0
    out = [0]
    for length in normalized_lengths:
        total += length
        out.append(total)
    cached = torch.tensor(out, dtype=torch.int32, device=device)
    _CU_SEQLENS_CACHE[cache_key] = cached
    return cached


def _build_mrope_freq_cache(
    device: torch.device,
    max_position: int,
    rotary_dim: int,
    rope_theta: float,
) -> torch.Tensor:
    cache_key = (str(device), int(max_position), int(rotary_dim), float(rope_theta))
    cached = _MROPE_FREQ_CACHE.get(cache_key)
    if cached is not None:
        return cached

    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    cached = torch.einsum("i,j->ij", positions, inv_freq)
    _MROPE_FREQ_CACHE[cache_key] = cached
    return cached


def _interleave_axis_halves(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or tensor.shape[dim] % 2 != 0:
        return tensor
    first_half, second_half = tensor.chunk(2, dim=dim)
    return torch.stack((first_half, second_half), dim=dim + 1).flatten(dim, dim + 1)


def _inverse_interleave_axis_halves(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or tensor.shape[dim] % 2 != 0:
        return tensor
    half = tensor.shape[dim] // 2
    tensor = tensor.reshape(*tensor.shape[:dim], half, 2, *tensor.shape[dim + 1 :])
    return torch.cat((tensor.select(dim + 1, 0), tensor.select(dim + 1, 1)), dim=dim)


def _reorder_v_heads_grouped_to_tiled(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
        return tensor
    if tensor.shape[dim] != num_v_heads * head_dim:
        return tensor
    num_v_per_k = num_v_heads // num_k_heads
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).reshape(*shape)


def _reorder_v_heads_tiled_to_grouped(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
        return tensor
    if tensor.shape[dim] != num_v_heads * head_dim:
        return tensor
    num_v_per_k = num_v_heads // num_k_heads
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).reshape(*shape)


def _reorder_v_head_axis_grouped_to_tiled(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_heads: int,
) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
        return tensor
    if tensor.shape[dim] != num_v_heads:
        return tensor
    num_v_per_k = num_v_heads // num_k_heads
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).reshape(*shape)


def _reorder_v_head_axis_tiled_to_grouped(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_heads: int,
) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
        return tensor
    if tensor.shape[dim] != num_v_heads:
        return tensor
    num_v_per_k = num_v_heads // num_k_heads
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).reshape(*shape)


def _maybe_reorder_gguf_ssm_param(
    tensor: torch.Tensor,
    *,
    interleave_halves: bool,
    tiled_to_grouped: bool,
    num_k_heads: int,
    num_v_heads: int,
) -> torch.Tensor:
    if tiled_to_grouped:
        return _reorder_v_heads_tiled_to_grouped(tensor, dim=0, num_k_heads=num_k_heads, num_v_heads=num_v_heads, head_dim=1)
    if interleave_halves:
        return _interleave_axis_halves(tensor, dim=0)
    return tensor


def clear_qwen35_runtime_caches(device: torch.device | None = None) -> None:
    if device is None:
        _CU_SEQLENS_CACHE.clear()
        _MROPE_FREQ_CACHE.clear()
        return

    device_key = str(device)
    for cache_key in [key for key in _CU_SEQLENS_CACHE if key[1] == device_key]:
        _CU_SEQLENS_CACHE.pop(cache_key, None)
    for cache_key in [key for key in _MROPE_FREQ_CACHE if key[0] == device_key]:
        _MROPE_FREQ_CACHE.pop(cache_key, None)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias=bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key)]
    value, beta = [x.transpose(1, 2).contiguous() for x in (value, beta)]
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size
    query = (query * (query.shape[-1] ** -0.5)).reshape(query.shape[0], query.shape[1], -1, chunk_size, query.shape[-1])
    key = key.reshape(key.shape[0], key.shape[1], -1, chunk_size, key.shape[-1])
    value = value.reshape(value.shape[0], value.shape[1], -1, chunk_size, value.shape[-1])
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(dim=-1)
    lower_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    upper_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    outputs = torch.empty(batch_size, num_heads, num_chunks, chunk_size, v_head_dim, device=value.device, dtype=initial_dtype)
    eye = torch.eye(chunk_size, dtype=torch.float32, device=query.device)

    for idx in range(num_chunks):
        q_i = query[:, :, idx]
        k_i = key[:, :, idx]
        value_i = value[:, :, idx].to(torch.float32)
        beta_i = beta[:, :, idx].to(torch.float32)
        g_i = g[:, :, idx]
        beta_i_unsqueezed = beta_i.unsqueeze(-1)
        g_exp_i = g_i.exp().unsqueeze(-1)
        k_beta_i = k_i * beta_i_unsqueezed
        decay_mask_i = ((g_i.unsqueeze(-1) - g_i.unsqueeze(-2)).tril().exp()).tril()
        lower = k_beta_i @ k_i.transpose(-1, -2)
        lower.mul_(decay_mask_i)
        lower.masked_fill_(lower_mask, 0)
        lower.add_(eye)
        value_i.mul_(beta_i_unsqueezed)
        value_i = torch.linalg.solve_triangular(lower, value_i, upper=False, left=True)
        k_beta_i.mul_(g_exp_i)
        k_cumdecay_i = torch.linalg.solve_triangular(lower, k_beta_i, upper=False, left=True)
        attn_i = q_i @ k_i.transpose(-1, -2)
        attn_i.mul_(decay_mask_i)
        attn_i.masked_fill_(upper_mask, 0)
        value_i.sub_(k_cumdecay_i @ recurrent_state)
        outputs[:, :, idx] = (((q_i * g_exp_i) @ recurrent_state) + attn_i @ value_i).to(initial_dtype)
        recurrent_state.mul_(g_i[:, :, -1, None, None].exp())
        recurrent_state.add_((k_i * (g_i[:, :, -1, None] - g_i).exp().unsqueeze(-1)).transpose(-1, -2) @ value_i)

    outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], -1, outputs.shape[-1])
    outputs = outputs[:, :, :sequence_length]
    if not output_final_state:
        recurrent_state = None
    return outputs.transpose(1, 2).contiguous(), recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key, value, beta = [x.transpose(1, 2).contiguous() for x in (query, key, value, beta)]
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = query.shape[-1] ** -0.5
    outputs = torch.empty(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=initial_dtype)
    recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32)
    )

    for idx in range(sequence_length):
        q_t = query[:, :, idx].to(torch.float32)
        k_t = key[:, :, idx].to(torch.float32)
        v_t = value[:, :, idx].to(torch.float32)
        g_t = g[:, :, idx]
        beta_t = beta[:, :, idx].to(torch.float32)
        q_t = q_t * scale
        g_t = g_t.exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta_t.unsqueeze(-1)
        recurrent_state = recurrent_state * g_t
        kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        recurrent_state = recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs[:, :, idx] = (recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2).to(initial_dtype)

    if not output_final_state:
        recurrent_state = None
    return outputs.transpose(1, 2).contiguous(), recurrent_state


class Qwen3_5DynamicCache:
    def __init__(self, config):
        self.layer_types = list(config.layer_types)
        self.transformer_layers = [idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"]
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")
        self.conv_states = [None for _ in range(int(config.num_hidden_layers))]
        self.recurrent_states = [None for _ in range(int(config.num_hidden_layers))]
        self.key_cache = [None for _ in range(int(config.num_hidden_layers))]
        self.value_cache = [None for _ in range(int(config.num_hidden_layers))]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if not self.transformer_layers:
            return 0
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        cache = self.key_cache[layer_idx]
        if cache is None:
            return 0
        return int(cache.shape[-2])

    @property
    def has_previous_state(self) -> bool:
        return self.conv_states[self.last_linear_layer] is not None


class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        rotary_dim = int(head_dim * float(rope_parameters.get("partial_rotary_factor", 1.0)))
        rope_theta = float(rope_parameters.get("rope_theta", 1000000))
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu") / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = list(rope_parameters.get("mrope_section", [11, 11, 10]))
        self.max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim

    def apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        max_position = self.max_position_embeddings
        if not (position_ids.is_cuda and torch.cuda.is_current_stream_capturing()):
            max_position = max(max_position, int(position_ids.max().item()) + 1)
        freq_cache = _build_mrope_freq_cache(
            position_ids.device,
            max_position=max_position,
            rotary_dim=self.rotary_dim,
            rope_theta=self.rope_theta,
        )
        freqs = torch.stack([freq_cache[position_ids[axis].long()] for axis in range(3)], dim=0)
        freqs = self.apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        hidden_states = hidden_states * self.weight.float()
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(dtype=input_dtype)


def _repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    return hidden_states.repeat_interleave(num_repeats, dim=1)


def _flash_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_lengths: list[int],
    key_lengths: list[int],
    scaling: float,
    flash_attention_fn,
) -> torch.Tensor:
    if flash_attention_fn is not None and query_states.is_cuda:
        q_chunks = [query_states[idx, : q_len] for idx, q_len in enumerate(query_lengths) if q_len > 0]
        k_chunks = [key_states[idx, : k_len] for idx, k_len in enumerate(key_lengths) if k_len > 0]
        v_chunks = [value_states[idx, : k_len] for idx, k_len in enumerate(key_lengths) if k_len > 0]
        q_flat = torch.cat(q_chunks, dim=0)
        k_flat = torch.cat(k_chunks, dim=0)
        v_flat = torch.cat(v_chunks, dim=0)
        cu_q = _build_cu_seqlens(query_lengths, query_states.device)
        cu_k = _build_cu_seqlens(key_lengths, query_states.device)
        out_flat = flash_attention_fn(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(query_lengths),
            max_seqlen_k=max(key_lengths),
            softmax_scale=scaling,
            causal=True,
        )
        out = torch.zeros_like(query_states)
        offset = 0
        for idx, q_len in enumerate(query_lengths):
            if q_len <= 0:
                continue
            out[idx, :q_len] = out_flat[offset : offset + q_len]
            offset += q_len
        return out

    outputs = []
    for idx, q_len in enumerate(query_lengths):
        q = query_states[idx, :q_len].transpose(0, 1)
        k = key_states[idx, : key_lengths[idx]].transpose(0, 1)
        v = value_states[idx, : key_lengths[idx]].transpose(0, 1)
        k = _repeat_kv(k.unsqueeze(0), q.shape[0] // k.shape[0]).squeeze(0)
        v = _repeat_kv(v.unsqueeze(0), q.shape[0] // v.shape[0]).squeeze(0)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scaling,
        ).squeeze(0).transpose(0, 1)
        outputs.append(out)
    padded = torch.zeros_like(query_states)
    for idx, out in enumerate(outputs):
        padded[idx, : out.shape[0]] = out
    return padded


class Qwen3_5Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        safe_legacy_kernels = _safe_legacy_kernels_enabled(config)
        self.layer_type = str(config.layer_types[layer_idx])
        self.attn_norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.post_attention_norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.attn_norm.use_triton_rmsnorm = not safe_legacy_kernels
        self.post_attention_norm.use_triton_rmsnorm = not safe_legacy_kernels
        self.ffn_gate = ColumnParallelLinear(int(config.hidden_size), int(config.intermediate_size), bias=False)
        self.ffn_up = ColumnParallelLinear(int(config.hidden_size), int(config.intermediate_size), bias=False)
        self.ffn_gate_up = None
        self.ffn_down = RowParallelLinear(int(config.intermediate_size), int(config.hidden_size), bias=False)
        self.mlp_act_fn = SiluAndMul()

        if self.layer_type == "full_attention":
            tp_size = _get_tp_size()
            hidden_size = int(config.hidden_size)
            total_num_heads = int(config.num_attention_heads)
            total_num_kv_heads = int(config.num_key_value_heads)
            assert total_num_heads % tp_size == 0
            assert total_num_kv_heads % tp_size == 0
            self.num_heads = total_num_heads // tp_size
            self.num_kv_heads = total_num_kv_heads // tp_size
            self.head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))
            self.num_key_value_groups = self.num_heads // self.num_kv_heads
            self.scaling = self.head_dim**-0.5
            self.attn_q = ColumnParallelLinear(
                hidden_size,
                total_num_heads * self.head_dim * 2,
                bias=bool(config.attention_bias),
            )
            self.attn_k = ColumnParallelLinear(
                hidden_size,
                total_num_kv_heads * self.head_dim,
                bias=bool(config.attention_bias),
            )
            self.attn_v = ColumnParallelLinear(
                hidden_size,
                total_num_kv_heads * self.head_dim,
                bias=bool(config.attention_bias),
            )
            self.attn_kv = None
            self.attn_output = RowParallelLinear(
                total_num_heads * self.head_dim,
                hidden_size,
                bias=bool(config.attention_bias),
            )
            self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
            if safe_legacy_kernels:
                self.attn.flash_attn_varlen_func = None
                self.attn.flash_attn_with_kvcache = None
                self.attn.use_triton_kv_cache = False
            self.attn_q_norm = RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
            self.attn_k_norm = RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
            self.attn_q_norm.use_triton_rmsnorm = not safe_legacy_kernels
            self.attn_k_norm.use_triton_rmsnorm = not safe_legacy_kernels
            self._flash_attn_varlen_func = None if safe_legacy_kernels else _DEFAULT_FLASH_ATTN_VARLEN_FUNC
        else:
            self._short_convolution_cls = None if safe_legacy_kernels else _DEFAULT_SHORT_CONVOLUTION
            self._fast_recurrent_gated_delta_rule = None if safe_legacy_kernels else _DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE
            self._fast_chunk_gated_delta_rule = None if safe_legacy_kernels else _DEFAULT_FAST_CHUNK_GATED_DELTA_RULE
            self.num_v_heads = int(config.linear_num_value_heads)
            self.num_k_heads = int(config.linear_num_key_heads)
            self.head_k_dim = int(config.linear_key_head_dim)
            self.head_v_dim = int(config.linear_value_head_dim)
            self.key_dim = self.head_k_dim * self.num_k_heads
            self.value_dim = self.head_v_dim * self.num_v_heads
            self.conv_kernel_size = int(config.linear_conv_kernel_dim)
            hidden_size = int(config.hidden_size)
            self.attn_qkv = ColumnParallelLinear(hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
            self.attn_gate = ColumnParallelLinear(hidden_size, self.value_dim, bias=False)
            self.ssm_alpha = ColumnParallelLinear(hidden_size, self.num_v_heads, bias=False)
            self.ssm_beta = ColumnParallelLinear(hidden_size, self.num_v_heads, bias=False)
            self.attn_gate_ab = None
            self.ssm_dt = nn.Parameter(torch.zeros(self.num_v_heads))
            self.ssm_a = nn.Parameter(-torch.ones(self.num_v_heads))
            if self._short_convolution_cls is not None:
                self.ssm_conv1d = self._short_convolution_cls(
                    hidden_size=self.key_dim * 2 + self.value_dim,
                    kernel_size=self.conv_kernel_size,
                    bias=False,
                    activation="silu",
                )
                self._use_short_convolution = True
            else:
                self.ssm_conv1d = nn.Conv1d(
                    in_channels=self.key_dim * 2 + self.value_dim,
                    out_channels=self.key_dim * 2 + self.value_dim,
                    kernel_size=self.conv_kernel_size,
                    bias=False,
                    groups=self.key_dim * 2 + self.value_dim,
                    padding=self.conv_kernel_size - 1,
                )
                self._use_short_convolution = False
            if not safe_legacy_kernels and _DEFAULT_FUSED_RMSNORM_GATED is not None:
                self.ssm_norm = _DEFAULT_FUSED_RMSNORM_GATED(self.head_v_dim, eps=float(config.rms_norm_eps))
                self._use_fused_rmsnorm_gated = True
            else:
                self.ssm_norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=float(config.rms_norm_eps))
                self._use_fused_rmsnorm_gated = False
            self.ssm_out = RowParallelLinear(self.value_dim, hidden_size, bias=False)
            self.conv_state_buffer = torch.empty(0)
            self.recurrent_state_buffer = torch.empty(0)
            self._gguf_interleave_ssm_ab = False
            self._gguf_v_head_reordered = False
            self._gguf_ssm_param_reordered = False

    def prepare_sequence_state(self, max_batch_size: int, device: torch.device, dtype: torch.dtype):
        if self.layer_type != "linear_attention":
            return
        conv_shape = (int(max_batch_size), self.key_dim * 2 + self.value_dim, self.conv_kernel_size)
        recurrent_shape = (int(max_batch_size), self.num_v_heads, self.head_k_dim, self.head_v_dim)
        with torch.inference_mode():
            if (
                self.conv_state_buffer.numel() == 0
                or tuple(self.conv_state_buffer.shape) != conv_shape
                or self.conv_state_buffer.device != device
                or self.conv_state_buffer.dtype != dtype
            ):
                self.conv_state_buffer = torch.zeros(conv_shape, device=device, dtype=dtype)
            else:
                self.conv_state_buffer.zero_()
            if (
                self.recurrent_state_buffer.numel() == 0
                or tuple(self.recurrent_state_buffer.shape) != recurrent_shape
                or self.recurrent_state_buffer.device != device
                or self.recurrent_state_buffer.dtype != dtype
            ):
                self.recurrent_state_buffer = torch.zeros(recurrent_shape, device=device, dtype=dtype)
            else:
                self.recurrent_state_buffer.zero_()

    def reset_sequence_state(self):
        if self.layer_type != "linear_attention":
            return
        with torch.inference_mode():
            if self.conv_state_buffer.numel() > 0:
                self.conv_state_buffer.zero_()
            if self.recurrent_state_buffer.numel() > 0:
                self.recurrent_state_buffer.zero_()

    def release_sequence_state(self):
        if self.layer_type != "linear_attention":
            return
        with torch.inference_mode():
            self.conv_state_buffer = torch.empty(0)
            self.recurrent_state_buffer = torch.empty(0)

    def _get_runtime_conv_state(self, batch_size: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.conv_state_buffer.numel() == 0
            or self.conv_state_buffer.shape[0] < int(batch_size)
            or self.conv_state_buffer.device != hidden_states.device
            or self.conv_state_buffer.dtype != hidden_states.dtype
        ):
            self.prepare_sequence_state(batch_size, hidden_states.device, hidden_states.dtype)
        return self.conv_state_buffer[:batch_size]

    def _get_runtime_recurrent_state(self, batch_size: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.recurrent_state_buffer.numel() == 0
            or self.recurrent_state_buffer.shape[0] < int(batch_size)
            or self.recurrent_state_buffer.device != hidden_states.device
            or self.recurrent_state_buffer.dtype != hidden_states.dtype
        ):
            self.prepare_sequence_state(batch_size, hidden_states.device, hidden_states.dtype)
        return self.recurrent_state_buffer[:batch_size]

    def _forward_full_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        past_key_values: Qwen3_5DynamicCache | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q_and_gate = self.attn_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)

        query_states = self.attn_q_norm(query_states)
        if self.attn_kv is None:
            key_states = self.attn_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            value_states = self.attn_v(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        else:
            key_value_states = self.attn_kv(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim * 2)
            key_states, value_states = torch.chunk(key_value_states, 2, dim=-1)
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        key_states = self.attn_k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                layer_idx,
            )
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attn_output = _flash_attention(
                query_states,
                key_states,
                value_states,
                query_lengths=[int(seq_len)] * batch_size,
                key_lengths=[int(key_states.shape[1])] * batch_size,
                scaling=self.scaling,
                flash_attention_fn=self._flash_attn_varlen_func,
            ).reshape(batch_size, seq_len, -1)
        else:
            attn_output = self.attn(
                query_states.reshape(-1, self.num_heads, self.head_dim),
                key_states.reshape(-1, self.num_kv_heads, self.head_dim),
                value_states.reshape(-1, self.num_kv_heads, self.head_dim),
            ).reshape(batch_size, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.attn_output(attn_output)

    def _forward_linear_attention(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        cache_params: Qwen3_5DynamicCache | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape
        context = get_context()
        if cache_params is not None:
            conv_state = cache_params.conv_states[layer_idx]
            recurrent_state = cache_params.recurrent_states[layer_idx]
            has_previous_state = cache_params.has_previous_state
        else:
            conv_state = self._get_runtime_conv_state(batch_size, hidden_states)
            recurrent_state = self._get_runtime_recurrent_state(batch_size, hidden_states)
            has_previous_state = not context.is_prefill
        use_precomputed_states = has_previous_state and seq_len == 1

        mixed_qkv_input = self.attn_qkv(hidden_states)
        if self.attn_gate_ab is None:
            z = self.attn_gate(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
            a = self.ssm_alpha(hidden_states)
            b = self.ssm_beta(hidden_states)
        else:
            gate_ab = self.attn_gate_ab(hidden_states)
            gate_proj, a, b = torch.split(gate_ab, [self.value_dim, self.num_v_heads, self.num_v_heads], dim=-1)
            z = gate_proj.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if self._gguf_v_head_reordered:
            z = _reorder_v_head_axis_tiled_to_grouped(
                z,
                dim=2,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
            )
        if self._gguf_v_head_reordered:
            b = _reorder_v_heads_tiled_to_grouped(
                b,
                dim=-1,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_dim=1,
            )
            a = _reorder_v_heads_tiled_to_grouped(
                a,
                dim=-1,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_dim=1,
            )
        elif self._gguf_interleave_ssm_ab:
            b = _interleave_axis_halves(b, dim=-1)
            a = _interleave_axis_halves(a, dim=-1)

        use_short_convolution = (
            self._use_short_convolution
            and self._short_convolution_cls is not None
            and isinstance(self.ssm_conv1d, self._short_convolution_cls)
        )

        if use_short_convolution:
            short_conv_cache = conv_state if has_previous_state and conv_state is not None else None
            mixed_qkv, last_conv_state = self.ssm_conv1d(
                mixed_qkv_input,
                cache=short_conv_cache,
                output_final_state=True,
            )
        else:
            mixed_qkv = mixed_qkv_input.transpose(1, 2)
            use_fast_causal_conv = causal_conv1d_fn is not None and causal_conv1d_update is not None
            if use_precomputed_states:
                if use_fast_causal_conv:
                    conv_kernel = self.ssm_conv1d.weight.squeeze(1)
                    mixed_qkv = causal_conv1d_update(
                        mixed_qkv,
                        conv_state,
                        conv_kernel,
                        self.ssm_conv1d.bias,
                        "silu",
                    )
                else:
                    conv_kernel = self.ssm_conv1d.weight.reshape(
                        self.ssm_conv1d.weight.shape[0],
                        self.ssm_conv1d.weight.shape[-1],
                    )
                    mixed_qkv = torch_causal_conv1d_update(
                        mixed_qkv,
                        conv_state,
                        conv_kernel,
                        self.ssm_conv1d.bias,
                    )
            else:
                if cache_params is not None:
                    if mixed_qkv.shape[-1] >= self.conv_kernel_size:
                        cache_params.conv_states[layer_idx] = mixed_qkv[:, :, -self.conv_kernel_size :].contiguous()
                    else:
                        cache_params.conv_states[layer_idx] = F.pad(
                            mixed_qkv,
                            (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
                        )
                else:
                    if mixed_qkv.shape[-1] >= self.conv_kernel_size:
                        conv_state.copy_(mixed_qkv[:, :, -self.conv_kernel_size :].contiguous())
                    else:
                        conv_state.copy_(
                            F.pad(
                                mixed_qkv,
                                (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
                            )
                        )
                conv_input = mixed_qkv.to(self.ssm_conv1d.weight.dtype)
                if use_fast_causal_conv:
                    mixed_qkv = causal_conv1d_fn(
                        x=conv_input,
                        weight=self.ssm_conv1d.weight.squeeze(1),
                        bias=self.ssm_conv1d.bias,
                        activation="silu",
                        seq_idx=None,
                    )
                else:
                    mixed_qkv = F.silu(self.ssm_conv1d(conv_input)[:, :, :seq_len])
                mixed_qkv = mixed_qkv.to(hidden_states.dtype)
            mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_short_convolution:
            if cache_params is not None:
                cache_params.conv_states[layer_idx] = last_conv_state
            elif last_conv_state is not conv_state:
                conv_state.copy_(last_conv_state)

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if self._gguf_v_head_reordered:
            value = _reorder_v_head_axis_tiled_to_grouped(
                value,
                dim=2,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
            )

        beta = b.sigmoid()
        ssm_a = _maybe_reorder_gguf_ssm_param(
            self.ssm_a,
            interleave_halves=self._gguf_interleave_ssm_ab,
            tiled_to_grouped=self._gguf_ssm_param_reordered,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
        )
        ssm_dt = _maybe_reorder_gguf_ssm_param(
            self.ssm_dt,
            interleave_halves=self._gguf_interleave_ssm_ab,
            tiled_to_grouped=self._gguf_ssm_param_reordered,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
        )
        g = ssm_a.float() * F.softplus(a.float() + ssm_dt)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        if use_precomputed_states:
            if self._fast_recurrent_gated_delta_rule is not None and hidden_states.is_cuda:
                core_attn_out, last_recurrent_state = self._fast_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                )
        else:
            if self._fast_chunk_gated_delta_rule is not None and hidden_states.is_cuda:
                core_attn_out, last_recurrent_state = self._fast_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=True,
                )

        if cache_params is not None:
            cache_params.recurrent_states[layer_idx] = last_recurrent_state
        else:
            recurrent_state.copy_(last_recurrent_state)

        core_attn_out = self.ssm_norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        if self._gguf_v_head_reordered:
            core_attn_out = _reorder_v_heads_grouped_to_tiled(
                core_attn_out,
                dim=-1,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_dim=self.head_v_dim,
            )
        return self.ssm_out(core_attn_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        residual: torch.Tensor | None,
        layer_idx: int,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3_5DynamicCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.attn_norm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.attn_norm(hidden_states, residual)

        if self.layer_type == "full_attention":
            hidden_states = self._forward_full_attention(hidden_states, position_embeddings, layer_idx, past_key_values)
        else:
            hidden_states = self._forward_linear_attention(hidden_states, layer_idx, past_key_values, attention_mask)

        hidden_states, residual = self.post_attention_norm(hidden_states, residual)
        gate_up = self.ffn_gate_up(hidden_states) if self.ffn_gate_up is not None else torch.cat((self.ffn_gate(hidden_states), self.ffn_up(hidden_states)), dim=-1)
        hidden_states = self.ffn_down(self.mlp_act_fn(gate_up))
        return hidden_states, residual


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        safe_legacy_kernels = _safe_legacy_kernels_enabled(config)
        self.token_embd = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.blk = nn.ModuleList([Qwen3_5Block(config, idx) for idx in range(int(config.num_hidden_layers))])
        self.output_norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.output_norm.use_triton_rmsnorm = not safe_legacy_kernels
        self.output = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def embed_tokens(self):
        return self.token_embd

    @property
    def layers(self):
        return self.blk

    @property
    def norm(self):
        return self.output_norm

    @property
    def lm_head(self):
        return self.output

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3_5DynamicCache | None = None,
        use_cache: bool | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        del use_cache
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if position_ids is not None:
            positions = position_ids

        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions is None:
            seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            positions = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len,
                device=(inputs_embeds.device if inputs_embeds is not None else input_ids.device),
                dtype=torch.long,
            ).unsqueeze(0)
        elif positions.ndim == 1:
            positions = positions.unsqueeze(0)
        elif positions.ndim == 3 and positions.shape[0] == 4:
            positions = positions[1:]

        hidden_states = self.token_embd(input_ids) if inputs_embeds is None else inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, positions)
        residual = None
        for layer_idx, block in enumerate(self.blk):
            hidden_states, residual = block(
                hidden_states,
                position_embeddings,
                residual,
                layer_idx=layer_idx,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        hidden_states, _ = self.output_norm(hidden_states, residual)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if hidden_states.ndim == 3:
            if context.is_prefill and context.cu_seqlens_q is not None and context.cu_seqlens_q.numel() > 1:
                num_seqs = int(context.cu_seqlens_q.numel() - 1)
                if hidden_states.shape[0] == num_seqs:
                    lengths = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                    row_indices = torch.arange(num_seqs, device=hidden_states.device)
                    token_indices = lengths.to(device=hidden_states.device, dtype=torch.long) - 1
                    hidden_states = hidden_states[row_indices, token_indices]
                elif hidden_states.shape[0] == 1 and num_seqs == 1:
                    last_index = int(context.cu_seqlens_q[-1].item()) - 1
                    hidden_states = hidden_states[:, last_index, :]
            elif hidden_states.shape[-2] == 1:
                hidden_states = hidden_states[:, 0, :]
        return self.output(hidden_states)


__all__ = ["Qwen3_5DynamicCache", "Qwen3_5ForCausalLM", "clear_qwen35_runtime_caches"]
