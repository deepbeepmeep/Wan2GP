from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_with_kvcache,
        flash_attn_func,
    )
except Exception:
    flash_attn_with_kvcache = None
    flash_attn_func = None

#disabled flash to test sdpa
# flash_attn_with_kvcache = None
# flash_attn_func = None

_FLASH_ATTN_NOTICE_SHOWN = False
_FLASH_ATTN_FULL_NOTICE_SHOWN = False


def scale_hidden_dim_for_mlp(dim: int, multiple_of: int = 256) -> int:
    hidden_dim = 4 * int(2 * dim / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        down_proj: nn.Module,
        up_proj: Optional[nn.Module] = None,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.w1 = gate_proj
        self.w2 = down_proj
        self.w3 = up_proj
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.w1(x))
        if self.w3 is not None:
            h = h * self.w3(x)
        h = self.w2(h)
        return h


class Llama3ScaledRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        scale_factor: int = 8,
        low_freq_factor: int = 1,
        high_freq_factor: int = 4,
        old_context_len: int = 8192,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.is_cache_built = False
        self.rope_init()

    def rope_init(self) -> None:
        freqs = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        if freqs.is_meta:
            return

        theta = self.apply_scaling(
            freqs,
            self.scale_factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.old_context_len,
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def ensure_device(self, device: torch.device) -> None:
        if not hasattr(self, "theta"):
            return
        if getattr(self.theta, "is_meta", False):
            return
        if self.theta.device == device:
            return
        non_blocking = device.type == "cuda"
        self.theta = self.theta.to(device, non_blocking=non_blocking)
        if hasattr(self, "cache"):
            self.cache = self.cache.to(device, non_blocking=non_blocking)

    def apply_scaling(
        self,
        freqs: torch.Tensor,
        scale_factor: int,
        low_freq_factor: int,
        high_freq_factor: int,
        old_context_len: int,
    ) -> torch.Tensor:
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.is_cache_built:
            raise RuntimeError("RoPE cache is not built.")

        seq_len = x.size(1)
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class ExpandableKVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        *,
        expand_step: int = 200,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if expand_step <= 0:
            raise ValueError("expand_step must be > 0")
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expand_step = expand_step
        self.limit = max_seq_len
        self._pos = 0

        init_len = expand_step
        if max_seq_len is not None:
            init_len = min(max_seq_len, expand_step)
        self.capacity = init_len

        k_cache = torch.empty(
            (batch_size, self.capacity, num_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        v_cache = torch.empty_like(k_cache)
        self.register_buffer("k_cache", k_cache, persistent=False)
        self.register_buffer("v_cache", v_cache, persistent=False)

    @property
    def size(self) -> int:
        return self._pos

    def reset(self) -> None:
        self._pos = 0

    def _replace_buffer(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers.pop(name)
        self.register_buffer(name, tensor, persistent=False)

    def ensure_capacity(self, required_len: int) -> None:
        if required_len <= self.capacity:
            return
        if self.limit is not None and required_len > self.limit:
            raise RuntimeError(
                f"KV cache length {required_len} exceeds limit {self.limit}"
            )
        new_capacity = self.capacity
        while new_capacity < required_len:
            new_capacity += self.expand_step
        if self.limit is not None:
            new_capacity = min(new_capacity, self.limit)
        self._resize_cache(new_capacity)

    def _resize_cache(self, new_capacity: int) -> None:
        if new_capacity <= self.capacity:
            return
        device = self.k_cache.device
        dtype = self.k_cache.dtype
        bsz = self.k_cache.shape[0]

        if device.type == "cuda":
            k_cpu = self.k_cache[:, : self._pos].detach().cpu()
            v_cpu = self.v_cache[:, : self._pos].detach().cpu()
            self._buffers.pop("k_cache", None)
            self._buffers.pop("v_cache", None)
            k_new = torch.empty(
                (bsz, new_capacity, self.num_heads, self.head_dim),
                device=device,
                dtype=dtype,
            )
            v_new = torch.empty_like(k_new)
            if self._pos:
                k_new[:, : self._pos] = k_cpu.to(device)
                v_new[:, : self._pos] = v_cpu.to(device)
            k_cpu = None
            v_cpu = None
        else:
            k_new = torch.empty(
                (bsz, new_capacity, self.num_heads, self.head_dim),
                device=device,
                dtype=dtype,
            )
            v_new = torch.empty_like(k_new)
            if self._pos:
                k_new[:, : self._pos] = self.k_cache[:, : self._pos]
                v_new[:, : self._pos] = self.v_cache[:, : self._pos]

        self._replace_buffer("k_cache", k_new)
        self._replace_buffer("v_cache", v_new)
        self.capacity = new_capacity

    def ensure_device(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> None:
        target_dtype = self.k_cache.dtype if dtype is None else dtype
        if self.k_cache.device == device and self.k_cache.dtype == target_dtype:
            return
        bsz = self.k_cache.shape[0]
        if device.type == "cuda":
            k_cpu = self.k_cache[:, : self._pos].detach().cpu()
            v_cpu = self.v_cache[:, : self._pos].detach().cpu()
            self._buffers.pop("k_cache", None)
            self._buffers.pop("v_cache", None)
            k_new = torch.empty(
                (bsz, self.capacity, self.num_heads, self.head_dim),
                device=device,
                dtype=target_dtype,
            )
            v_new = torch.empty_like(k_new)
            if self._pos:
                k_new[:, : self._pos] = k_cpu.to(device=device, dtype=target_dtype)
                v_new[:, : self._pos] = v_cpu.to(device=device, dtype=target_dtype)
            k_cpu = None
            v_cpu = None
        else:
            k_new = torch.empty(
                (bsz, self.capacity, self.num_heads, self.head_dim),
                device=device,
                dtype=target_dtype,
            )
            v_new = torch.empty_like(k_new)
            if self._pos:
                k_new[:, : self._pos] = self.k_cache[:, : self._pos].to(
                    device=device, dtype=target_dtype
                )
                v_new[:, : self._pos] = self.v_cache[:, : self._pos].to(
                    device=device, dtype=target_dtype
                )

        self._replace_buffer("k_cache", k_new)
        self._replace_buffer("v_cache", v_new)

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"KV cache batch size {self.k_cache.shape[0]} is smaller than {bsz}"
            )
        required = self._pos + seq_len
        self.ensure_capacity(required)
        start = self._pos
        end = start + seq_len
        self.k_cache[:bsz, start:end] = k_val
        self.v_cache[:bsz, start:end] = v_val
        self._pos = end
        return self.k_cache, self.v_cache

    def advance(self, seq_len: int) -> None:
        self.ensure_capacity(self._pos + seq_len)
        self._pos += seq_len


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
        cache_expand_step: int = 200,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError("attn_dropout must be between 0.0 and 1.0")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        self.kv_cache: Optional[ExpandableKVCache] = None
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.pos_embeddings = pos_embeddings
        self.cache_enabled = False
        self.cache_expand_step = cache_expand_step
        self._flash_attn = flash_attn_with_kvcache
        self._flash_attn_func = flash_attn_func
        self._flash_kv_ready = False
        self._flash_full_ready = False
        self._flash_device = None
        self._flash_dtype = None

    def setup_cache(
        self, batch_size: int, dtype: torch.dtype, max_seq_len: int
    ) -> None:
        if self.kv_cache is not None:
            return
        device = self.q_proj.weight.device
        self.kv_cache = ExpandableKVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=dtype,
            expand_step=self.cache_expand_step,
            device=device,
        )
        self.cache_enabled = True

    def reset_cache(self) -> None:
        if self.kv_cache is None:
            raise RuntimeError("Key value caches are not setup.")
        self.kv_cache.reset()

    def _apply_rope(
        self, qk_list: list[torch.Tensor], input_pos: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q,k  = qk_list
        qk_list.clear()
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)
            k = self.pos_embeddings(k, input_pos=input_pos)
        return q, k

    def _expand_kv(
        self, kv_list: list[torch.Tensor], q_per_kv: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k,v = kv_list
        kv_list.clear()
        if self.num_heads != self.num_kv_heads:
            bsz, seq_len, _, _ = k.shape
            k = (
                k.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)
                .expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
                .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            )
            v = (
                v.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)
                .expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
                .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            )
        return k, v

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if mask is not None:
            if mask.device != q.device:
                mask = mask.to(q.device)
            attn_mask = mask
            if attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]
            if attn_mask.shape[-1] != k.shape[-2]:
                attn_mask = attn_mask[..., : k.shape[-2]]

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        return out.transpose(1, 2).contiguous()

    def _flash_full(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        global _FLASH_ATTN_FULL_NOTICE_SHOWN
        if not _FLASH_ATTN_FULL_NOTICE_SHOWN:
            print("\nHeartMuLa attention: using flash_attn_func.")
            _FLASH_ATTN_FULL_NOTICE_SHOWN = True
        return self._flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            causal=is_causal,
        )

    def configure_flash(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._flash_device == device and self._flash_dtype == dtype:
            return
        self._flash_device = device
        self._flash_dtype = dtype
        flash_ok = (
            device.type == "cuda"
            and dtype in (torch.float16, torch.bfloat16)
            and self.head_dim <= 256
        ) and False #disabled flash
        kv_ready = self.kv_cache is not None and self.cache_enabled
        self._flash_kv_ready = (
            flash_ok and self._flash_attn is not None and kv_ready
        )
        self._flash_full_ready = flash_ok and self._flash_attn_func is not None

    def reset_flash(self) -> None:
        self._flash_kv_ready = False
        self._flash_full_ready = False
        self._flash_device = None
        self._flash_dtype = None

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, s_x, _ = x.shape
        q = self.q_proj(x).view(bsz, s_x, self.num_heads, self.head_dim)
        if self.kv_cache is not None and self.cache_enabled:
            self.kv_cache.ensure_device(q.device, q.dtype)
            if mask is not None and s_x == 1:
                mask = None
        y = x if y is None else y
        s_y = y.shape[1]
        k = self.k_proj(y).view(bsz, s_y, self.num_kv_heads, self.head_dim)
        v = self.v_proj(y).view(bsz, s_y, self.num_kv_heads, self.head_dim)
        qk_list = [q, k]
        q= k = None
        q, k = self._apply_rope(qk_list, input_pos)
        kv_list = [k, v]
        k= v = None
        k, v = self._expand_kv(kv_list, self.num_heads // self.num_kv_heads)

        if s_x == 1:
            use_flash_kv = self._flash_kv_ready and mask is None
            if use_flash_kv:
                global _FLASH_ATTN_NOTICE_SHOWN
                if not _FLASH_ATTN_NOTICE_SHOWN:
                    print("\nHeartMuLa attention: using flash_attn_with_kvcache.")
                    _FLASH_ATTN_NOTICE_SHOWN = True
                self.kv_cache.ensure_capacity(self.kv_cache.size + s_y)
                cache_seqlens = torch.full(
                    (bsz,),
                    self.kv_cache.size,
                    device=q.device,
                    dtype=torch.int32,
                )
                attn_out = self._flash_attn(
                    q=q,
                    k_cache=self.kv_cache.k_cache,
                    v_cache=self.kv_cache.v_cache,
                    k=k,
                    v=v,
                    cache_seqlens=cache_seqlens,
                    cache_leftpad=None,
                    causal=True,
                )
                self.kv_cache.advance(s_y)
            else:
                if self.kv_cache is not None and self.cache_enabled:
                    k_cache, v_cache = self.kv_cache.update(k, v)
                    k = k_cache[:, : self.kv_cache.size]
                    v = v_cache[:, : self.kv_cache.size]
                attn_out = self._sdpa(
                    q,
                    k,
                    v,
                    mask=mask,
                    is_causal=False,
                )
        else:
            use_flash_full = self._flash_full_ready and mask is None
            if use_flash_full:
                if self.kv_cache is not None and self.cache_enabled:
                    self.kv_cache.update(k, v)
                attn_out = self._flash_full(
                    q,
                    k,
                    v,
                    is_causal=self.is_causal,
                )
            else:
                if self.kv_cache is not None and self.cache_enabled:
                    k_cache, v_cache = self.kv_cache.update(k, v)
                    k = k_cache[:, : self.kv_cache.size]
                    v = v_cache[:, : self.kv_cache.size]
                attn_out = self._sdpa(
                    q,
                    k,
                    v,
                    mask=mask,
                    is_causal=self.is_causal if mask is None else False,
                )

        q = None
        k = None
        v = None
        mask = None
        attn_out = attn_out.reshape(bsz, s_x, -1)
        out = self.output_proj(attn_out)
        return out


class TransformerSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
        cache_expand_step: int = 200,
    ) -> None:
        self.attn.cache_expand_step = cache_expand_step
        self.attn.setup_cache(batch_size, dtype, max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        return self.attn.cache_enabled

    def reset_cache(self) -> None:
        self.attn.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = self.sa_norm(x)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)
        x = x.add_(attn_out)
        attn_out = None
        h = self.mlp_norm(x)
        mlp_out = self.mlp(h)
        x = x.add_(mlp_out)
        mlp_out = None
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        tok_embeddings: nn.Module,
        layers: nn.ModuleList,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: nn.Module,
    ) -> None:
        super().__init__()
        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decoder_max_cache_seq_len = None

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
        cache_expand_step: int = 200,
    ) -> None:
        if decoder_max_seq_len is not None:
            self.decoder_max_cache_seq_len = decoder_max_seq_len
        else:
            self.decoder_max_cache_seq_len = self.max_seq_len

        for layer in self.layers:
            layer.setup_caches(
                batch_size,
                dtype,
                encoder_max_seq_len=encoder_max_seq_len or self.max_seq_len,
                decoder_max_seq_len=self.decoder_max_cache_seq_len,
                cache_expand_step=cache_expand_step,
            )

    def caches_are_setup(self) -> bool:
        return self.layers[0].caches_are_setup()

    def caches_are_enabled(self) -> bool:
        return self.layers[0].caches_are_enabled()

    def reset_caches(self) -> None:
        if not self.caches_are_enabled():
            raise RuntimeError("Key value caches are not setup.")
        for layer in self.layers:
            layer.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, input_pos=input_pos)
        x = self.norm(x)
        return x


def build_llama_decoder(
    *,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    scale_factor: int = 32,
) -> TransformerDecoder:
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = Llama3ScaledRoPE(
        dim=head_dim,
        max_seq_len=max_seq_len,
        base=rope_base,
        scale_factor=scale_factor,
    )
    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        hidden_dim = (
            intermediate_dim
            if intermediate_dim is not None
            else scale_hidden_dim_for_mlp(embed_dim)
        )
        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, hidden_dim, bias=False),
            down_proj=nn.Linear(hidden_dim, embed_dim, bias=False),
            up_proj=nn.Linear(embed_dim, hidden_dim, bias=False),
        )
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)

    tok_embeddings = nn.Identity()
    output_proj = nn.Identity()
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )
