from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except Exception:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

_SDPA_FALLBACK_NOTICE_SHOWN = False
_ALLOW_FLASH_ATTN2 = True
_ALLOW_TRITON_KVCACHE = True


@dataclass
class ForwardContext:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


_FORWARD_CONTEXT = ForwardContext()


def configure_attention_kernels(allow_flash2=True, allow_triton=True):
    global _ALLOW_FLASH_ATTN2, _ALLOW_TRITON_KVCACHE, _SDPA_FALLBACK_NOTICE_SHOWN
    _ALLOW_FLASH_ATTN2 = bool(allow_flash2)
    _ALLOW_TRITON_KVCACHE = bool(allow_triton)
    _SDPA_FALLBACK_NOTICE_SHOWN = False


def triton_available():
    return bool(_ALLOW_TRITON_KVCACHE and triton is not None)


def flash_attn2_available():
    return bool(_ALLOW_FLASH_ATTN2 and flash_attn_varlen_func is not None and flash_attn_with_kvcache is not None)


def get_forward_context():
    return _FORWARD_CONTEXT


def set_forward_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    global _FORWARD_CONTEXT
    _FORWARD_CONTEXT = ForwardContext(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_forward_context():
    global _FORWARD_CONTEXT
    _FORWARD_CONTEXT = ForwardContext()


if triton is not None:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        BLOCK_SIZE: tl.constexpr = 2048
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        d_offset = 0
        while d_offset < D:
            cur_block_size = min(BLOCK_SIZE, D - d_offset)
            key_offsets = idx * key_stride + d_offset + tl.arange(0, BLOCK_SIZE)
            value_offsets = idx * value_stride + d_offset + tl.arange(0, BLOCK_SIZE)
            cache_offsets = slot * D + d_offset + tl.arange(0, BLOCK_SIZE)

            mask = tl.arange(0, BLOCK_SIZE) < cur_block_size
            key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
            value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
            tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
            tl.store(v_cache_ptr + cache_offsets, value, mask=mask)

            d_offset += BLOCK_SIZE


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert slot_mapping.numel() == N
    if not triton_available():
        flat_slots = slot_mapping.to(device=key.device, dtype=torch.long)
        safe_slots = torch.clamp(flat_slots, min=0)
        valid = (flat_slots >= 0).to(dtype=key.dtype).view(N, 1, 1)
        flat_k_cache = k_cache.view(-1, num_heads, head_dim)
        flat_v_cache = v_cache.view(-1, num_heads, head_dim)
        src_k = key * valid
        src_v = value * valid
        upd_k = torch.zeros_like(flat_k_cache)
        upd_v = torch.zeros_like(flat_v_cache)
        upd_m = torch.zeros((flat_k_cache.shape[0], 1, 1), device=key.device, dtype=key.dtype)
        upd_k.index_add_(0, safe_slots, src_k)
        upd_v.index_add_(0, safe_slots, src_v)
        upd_m.index_add_(0, safe_slots, valid)
        has_update = upd_m > 0
        flat_k_cache.copy_(torch.where(has_update, upd_k, flat_k_cache))
        flat_v_cache.copy_(torch.where(has_update, upd_v, flat_v_cache))
        return
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self._flash_attn2_available = flash_attn2_available()

    @staticmethod
    def _to_int(value):
        return int(value.item()) if torch.is_tensor(value) else int(value)

    @staticmethod
    def _gather_cache_tokens(cache: torch.Tensor, block_table: torch.Tensor, token_count: int):
        if token_count <= 0:
            return cache.new_empty((0, cache.shape[2], cache.shape[3]))
        block_size = int(cache.shape[1])
        token_idx = torch.arange(token_count, device=cache.device, dtype=torch.long)
        block_idx = torch.div(token_idx, block_size, rounding_mode="floor")
        token_offset = torch.remainder(token_idx, block_size)
        block_ids = block_table.index_select(0, block_idx).to(torch.long)
        return cache[block_ids, token_offset]

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, prefix_len: int = 0, causal: bool = True):
        if q.numel() == 0:
            return q
        q_ = q.transpose(0, 1).unsqueeze(0)
        k_ = k.transpose(0, 1).unsqueeze(0)
        v_ = v.transpose(0, 1).unsqueeze(0)
        q_len, k_len = int(q.shape[0]), int(k.shape[0])
        use_causal = bool(causal and prefix_len == 0)
        attn_mask = None
        if causal and prefix_len > 0:
            q_idx = torch.arange(q_len, device=q.device, dtype=torch.long) + int(prefix_len)
            k_idx = torch.arange(k_len, device=q.device, dtype=torch.long)
            attn_mask = (k_idx.unsqueeze(0) <= q_idx.unsqueeze(1)).view(1, 1, q_len, k_len)
            use_causal = False
        out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, dropout_p=0.0, is_causal=use_causal)
        return out.squeeze(0).transpose(0, 1).contiguous()

    def _sdpa_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: ForwardContext, k_cache: torch.Tensor, v_cache: torch.Tensor):
        if context.cu_seqlens_q is None or context.cu_seqlens_k is None:
            return self._sdpa(q, k, v, prefix_len=max(0, int(k.shape[0]) - int(q.shape[0])), causal=True)
        num_seqs = int(context.cu_seqlens_q.numel()) - 1
        outs = []
        for i in range(num_seqs):
            q_start = self._to_int(context.cu_seqlens_q[i])
            q_end = self._to_int(context.cu_seqlens_q[i + 1])
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            q_i = q[q_start:q_end]
            if context.block_tables is None:
                k_start = self._to_int(context.cu_seqlens_k[i])
                k_end = self._to_int(context.cu_seqlens_k[i + 1])
                k_i = k[k_start:k_end]
                v_i = v[k_start:k_end]
                k_len = k_end - k_start
            else:
                k_len = self._to_int(context.cu_seqlens_k[i + 1]) - self._to_int(context.cu_seqlens_k[i])
                block_table = context.block_tables[i]
                k_i = self._gather_cache_tokens(k_cache, block_table, k_len)
                v_i = self._gather_cache_tokens(v_cache, block_table, k_len)
            prefix_len = max(0, k_len - q_len)
            outs.append(self._sdpa(q_i, k_i, v_i, prefix_len=prefix_len, causal=True))
        if not outs:
            return q.new_empty((0, self.num_heads, self.head_dim))
        return torch.cat(outs, dim=0)

    def _sdpa_decode(self, q: torch.Tensor, context: ForwardContext, k_cache: torch.Tensor, v_cache: torch.Tensor):
        if context.context_lens is None or context.block_tables is None:
            return q
        batch = int(context.context_lens.shape[0])
        if batch <= 0:
            return q
        q_total = int(q.shape[0])
        if q_total % batch != 0:
            raise RuntimeError(f"Invalid decode shape: q_tokens={q_total}, batch={batch}")
        q_per_seq = q_total // batch
        num_table_blocks = int(context.block_tables.shape[1])
        if num_table_blocks <= 0:
            return torch.zeros_like(q)
        block_size = int(k_cache.shape[1])
        max_tokens = num_table_blocks * block_size
        device = q.device
        token_idx = torch.arange(max_tokens, device=device, dtype=torch.long)
        block_idx = torch.div(token_idx, block_size, rounding_mode="floor")
        token_offset = torch.remainder(token_idx, block_size)
        block_ids = context.block_tables.index_select(1, block_idx)
        valid_blocks = block_ids >= 0
        safe_block_ids = torch.clamp(block_ids, min=0).to(torch.long)
        k_all = k_cache[safe_block_ids, token_offset]
        v_all = v_cache[safe_block_ids, token_offset]
        q_all = q.view(batch, q_per_seq, self.num_heads, self.head_dim)
        k_all = k_all.transpose(1, 2).contiguous()
        v_all = v_all.transpose(1, 2).contiguous()
        q_all = q_all.transpose(1, 2).contiguous()
        valid_len = token_idx.unsqueeze(0) < context.context_lens.to(device=device, dtype=torch.long).unsqueeze(1)
        base_mask = valid_blocks & valid_len
        q_steps = torch.arange(q_per_seq, device=device, dtype=torch.long).view(1, q_per_seq, 1)
        q_pos = context.context_lens.to(device=device, dtype=torch.long).view(batch, 1, 1) - q_per_seq + q_steps
        causal_mask = (token_idx.view(1, 1, max_tokens) <= q_pos).unsqueeze(1)
        attn_mask = (base_mask.view(batch, 1, 1, max_tokens) & causal_mask).expand(batch, self.num_heads, q_per_seq, max_tokens)
        out = F.scaled_dot_product_attention(q_all, k_all, v_all, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(q_total, self.num_heads, self.head_dim)
        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_forward_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if self._flash_attn2_available:
            if context.is_prefill:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
            return o

        if context.is_prefill:
            return self._sdpa_prefill(q, k, v, context, k_cache, v_cache)
        return self._sdpa_decode(q, context, k_cache, v_cache)
