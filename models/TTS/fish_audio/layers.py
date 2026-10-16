# Adapted from fishaudio/fish-speech, revision 1ff3c3ed2f80d3106a83909dd25b83129c3d816f.
# Copyright Fish Audio contributors. See LICENSE_FISH_AUDIO for the upstream license.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.types import Device

from .config import FishOmniTextConfig
from . import kernels
from shared import attention_kit
from shared.llm_engines.nanovllm.layerwise import cached_attention, delta_rule

GDN_CHUNK_SIZE = 64


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm scaling by 1 + weight, computed in fp32."""

    def __init__(
        self,
        dim: int,
        eps: float,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device))
        self.fused = False  # the vLLM engine's fused kernel (kernels.py)

    def forward(self, x: Tensor) -> Tensor:
        if self.fused and kernels.available(x):
            return kernels.zero_centered_rms_norm(x, self.weight, self.eps)
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """Gated DeltaNet output norm: weight * rmsnorm(x) * silu(gate)."""

    def __init__(
        self,
        dim: int,
        eps: float,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
        self.fused = False  # the vLLM engine's fused kernel (kernels.py)

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        if self.fused and kernels.available(x) and gate.is_contiguous():
            return kernels.gated_rms_norm(x, gate, self.weight, self.eps)
        dtype = x.dtype
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.weight * out.to(dtype)
        return (out * F.silu(gate.float())).to(dtype)


def apply_neox_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the first rotary_dim channels of x [B, T, heads, head_dim] (half-split)."""
    rotary_dim = cos.shape[-1]
    rotary, passthrough = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    out = rotary * cos + rotated * sin
    return torch.cat((out.to(x.dtype), passthrough), dim=-1)


@dataclass(kw_only=True)
class LayerContext:
    """Per-forward inputs shared by the layers of one device."""

    input_pos: Tensor  # [T] contiguous positions of the rows
    cos: Tensor
    sin: Tensor
    kv_len: int  # attended cache prefix (last position + 1)
    continue_state: bool  # the rows continue a cached sequence
    use_cache: bool


class FullAttention(nn.Module):
    """Gated softmax attention: GQA, q/k norm, partial RoPE, sigmoid output gate."""

    def __init__(
        self,
        config: FishOmniTextConfig,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        linear = dict(bias=False, dtype=dtype, device=device)
        # Output rows: each head's [query, gate] pair.
        self.query_proj = nn.Linear(hidden, self.num_heads * self.head_dim, **linear)
        self.gate_proj = nn.Linear(hidden, self.num_heads * self.head_dim, **linear)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, **linear)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, **linear)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, **linear)
        norm = dict(dtype=dtype, device=device)
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps, **norm)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps, **norm)

    def forward(self, x: Tensor, context: LayerContext, cache) -> Tensor:
        batch, length, _ = x.shape
        gate = self.gate_proj(x)

        def normalize(query, key, group):
            if query is not None:
                attention_kit.rows_(query, lambda part, first, last: apply_neox_rope(self.q_norm(part), context.cos[first:last], context.sin[first:last]))
            if key is not None:
                attention_kit.rows_(key, lambda part, first, last: apply_neox_rope(self.k_norm(part), context.cos[first:last], context.sin[first:last]))

        attended = attention_kit.qkv_attention([x], self.query_proj, self.k_proj, self.v_proj, self.num_heads, self.head_dim,
            normalize, kv_heads=self.num_kv_heads,
            attention_fn=lambda qkv, group: cached_attention(qkv, group, cache, context.input_pos, context.kv_len, self.engine))
        return self.o_proj(attended.reshape(batch, length, -1) * gate.sigmoid())


def l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    chunk_size: int = GDN_CHUNK_SIZE,
) -> tuple[Tensor, Tensor]:
    """Gated delta rule over whole chunks, in fp32.

    query / key [B, T, H, dk] (not yet l2-normalized), value [B, T, H, dv], decay
    (log space, <= 0) and beta [B, T, H], initial_state [B, H, dk, dv]. Returns the
    output [B, T, H, dv] in the query dtype and the final fp32 state.
    """
    output_dtype = query.dtype
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query, key, value, beta, decay = (
        t.transpose(1, 2).to(torch.float32).contiguous()
        for t in (query, key, value, beta, decay)
    )
    query = l2norm(query) * key_dim**-0.5
    key = l2norm(key)

    pad = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta, decay = (F.pad(t, (0, pad)) for t in (beta, decay))
    chunk_count = (length + pad) // chunk_size

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, key_beta, value_beta = (
        t.reshape(batch, heads, chunk_count, chunk_size, t.shape[-1])
        for t in (query, key, key_beta, value_beta)
    )
    decay = decay.reshape(batch, heads, chunk_count, chunk_size)
    upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)
    cumulative = decay.cumsum(dim=-1)
    pairwise = (
        (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2))
        .masked_fill(upper, float("-inf"))
        .exp()
    )

    system = (key_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_key_beta = key_beta * cumulative.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(
        system, value_beta, upper=False, unitriangular=True
    )
    key_cumulative_decay = torch.linalg.solve_triangular(
        system, decayed_key_beta, upper=False, unitriangular=True
    )

    if initial_state is None:
        state = query.new_zeros(batch, heads, key_dim, value_dim)
    else:
        state = initial_state.to(torch.float32)
    output = torch.empty_like(new_values)
    query = query * cumulative.exp().unsqueeze(-1)
    key = key * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp()[..., None, None]
    for index in range(chunk_count):
        corrected = new_values[:, :, index] - key_cumulative_decay[:, :, index] @ state
        output[:, :, index] = (
            query[:, :, index] @ state + intra[:, :, index] @ corrected
        )
        state = (
            state * chunk_decay[:, :, index]
            + key[:, :, index].transpose(-1, -2) @ corrected
        )

    output = output.reshape(batch, heads, -1, value_dim)[:, :, :length]
    return output.transpose(1, 2).contiguous().to(output_dtype), state


def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Token-by-token gated delta rule; same contract as chunk_gated_delta_rule."""
    output_dtype = query.dtype
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query, key, value, beta, decay = (
        t.transpose(1, 2).to(torch.float32).contiguous()
        for t in (query, key, value, beta, decay)
    )
    query = l2norm(query) * key_dim**-0.5
    key = l2norm(key)
    if initial_state is None:
        state = value.new_zeros(batch, heads, key_dim, value_dim)
    else:
        state = initial_state.to(torch.float32)
    output = torch.empty_like(value)
    for index in range(length):
        state = state * decay[:, :, index].exp()[..., None, None]
        k_t = key[:, :, index]
        remembered = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, index] - remembered) * beta[:, :, index].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = (state * query[:, :, index].unsqueeze(-1)).sum(dim=-2)
    return output.transpose(1, 2).contiguous().to(output_dtype), state


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: FishOmniTextConfig,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        linear = dict(bias=False, dtype=dtype, device=device)
        # Output rows: query, key, value.
        self.in_proj_qkv = nn.Linear(hidden, self.conv_dim, **linear)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, **linear)
        self.in_proj_b = nn.Linear(hidden, self.num_v_heads, **linear)
        self.in_proj_a = nn.Linear(hidden, self.num_v_heads, **linear)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads, dtype=dtype, device=device)
        )
        self.A_log = nn.Parameter(
            torch.zeros(self.num_v_heads, dtype=dtype, device=device)
        )
        self.norm = GatedRMSNorm(
            self.head_v_dim, config.rms_norm_eps, dtype=dtype, device=device
        )
        self.out_proj = nn.Linear(self.value_dim, hidden, **linear)

    def forward(self, x: Tensor, context: LayerContext, cache) -> Tensor:
        conv_state, recurrent_state = cache
        batch, length, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)  # [B, C, T]
        history = self.conv_kernel_size - 1
        if context.continue_state:
            assert conv_state is not None
            previous = conv_state[:batch].to(mixed.dtype)
        else:
            previous = mixed.new_zeros(batch, self.conv_dim, history)
        padded = torch.cat((previous, mixed), dim=-1)
        if context.use_cache:
            assert conv_state is not None
            conv_state[:batch].copy_(padded[..., -history:])
        mixed = F.silu(F.conv1d(padded, self.conv1d.weight, groups=self.conv_dim))
        query, key, value = mixed.transpose(1, 2).split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        z = self.in_proj_z(x).reshape(batch, length, self.num_v_heads, self.head_v_dim)
        beta = torch.sigmoid(self.in_proj_b(x).float())
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias.float()
        )
        groups = self.num_v_heads // self.num_k_heads
        if groups > 1:
            query = query.repeat_interleave(groups, dim=2)
            key = key.repeat_interleave(groups, dim=2)

        initial_state = None
        if context.continue_state:
            assert recurrent_state is not None
            initial_state = recurrent_state[:batch]
        rule = (
            recurrent_gated_delta_rule
            if context.continue_state and length == 1
            else chunk_gated_delta_rule
        )
        core, final_state = delta_rule(query, key, value, decay, beta, initial_state, engine=self.engine, recurrent=context.continue_state and length == 1, torch_impl=rule)
        if context.use_cache:
            assert recurrent_state is not None
            recurrent_state[:batch].copy_(final_state)

        core = self.norm(
            core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        ).reshape(batch, length, self.value_dim)
        return self.out_proj(core)
