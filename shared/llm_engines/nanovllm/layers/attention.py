import torch
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except Exception:  # pragma: no cover
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
from ..utils.context import get_context

_DEFAULT_FLASH_ATTN_VARLEN_FUNC = flash_attn_varlen_func
_DEFAULT_FLASH_ATTN_WITH_KVCACHE = flash_attn_with_kvcache
_USE_TRITON_KV_CACHE = triton is not None and tl is not None
_DEFAULT_USE_TRITON_KV_CACHE = _USE_TRITON_KV_CACHE


def configure_attention_safe_legacy_kernels(enabled: bool) -> None:
    global flash_attn_varlen_func, flash_attn_with_kvcache, _USE_TRITON_KV_CACHE
    if enabled:
        flash_attn_varlen_func = None
        flash_attn_with_kvcache = None
        _USE_TRITON_KV_CACHE = False
        return
    flash_attn_varlen_func = _DEFAULT_FLASH_ATTN_VARLEN_FUNC
    flash_attn_with_kvcache = _DEFAULT_FLASH_ATTN_WITH_KVCACHE
    _USE_TRITON_KV_CACHE = triton is not None and tl is not None


if triton is not None and tl is not None:
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
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def _repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    return hidden_states.repeat_interleave(num_repeats, dim=1)


def _sdpa_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    attention_bias: torch.Tensor | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    # Avoid enable_gqa here: on the local Torch 2.10 + CUDA 13 build it can pick
    # a much higher-workspace backend than the explicit-repeat SDPA path.
    key_states = _repeat_kv(key_states, num_key_value_groups)
    value_states = _repeat_kv(value_states, num_key_value_groups)
    if attention_bias is not None and not torch.is_floating_point(attention_bias):
        attention_bias = attention_bias.to(dtype=query_states.dtype)
    elif attention_bias is not None and attention_bias.dtype != query_states.dtype:
        attention_bias = attention_bias.to(dtype=query_states.dtype)
    return F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_bias,
        dropout_p=0.0,
        is_causal=bool(is_causal),
        scale=scaling,
    )


def _build_causal_bias(query_len: int, key_len: int, device: torch.device, query_offset: int = 0) -> torch.Tensor:
    q_pos = torch.arange(query_len, device=device, dtype=torch.long) + int(query_offset)
    k_pos = torch.arange(key_len, device=device, dtype=torch.long)
    valid = q_pos[:, None] >= k_pos[None, :]
    bias = torch.zeros((query_len, key_len), dtype=torch.float32, device=device)
    bias.masked_fill_(~valid, float("-inf"))
    return bias

def _gather_cache_tokens(cache: torch.Tensor, block_table: torch.Tensor, seq_len: int) -> torch.Tensor:
    if seq_len <= 0:
        return cache.new_empty((0, cache.shape[-2], cache.shape[-1]))
    block_size = int(cache.shape[1])
    num_blocks = (int(seq_len) + block_size - 1) // block_size
    valid_block_ids = [int(block_id) for block_id in block_table[:num_blocks].tolist() if int(block_id) >= 0]
    gathered = cache[torch.tensor(valid_block_ids, device=cache.device, dtype=torch.long)]
    return gathered.reshape(-1, cache.shape[-2], cache.shape[-1])[:seq_len]


def _flash_attention_fallback_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    num_key_value_groups: int,
    context,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    use_prefix_cache = context.block_tables is not None
    for idx in range(int(context.cu_seqlens_q.numel()) - 1):
        q_start = int(context.cu_seqlens_q[idx].item())
        q_end = int(context.cu_seqlens_q[idx + 1].item())
        k_start = int(context.cu_seqlens_k[idx].item())
        k_end = int(context.cu_seqlens_k[idx + 1].item())
        q_len = q_end - q_start
        k_len = k_end - k_start
        if q_len <= 0:
            continue
        q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        if use_prefix_cache:
            k_i = _gather_cache_tokens(k_cache, context.block_tables[idx], k_len).transpose(0, 1).unsqueeze(0)
            v_i = _gather_cache_tokens(v_cache, context.block_tables[idx], k_len).transpose(0, 1).unsqueeze(0)
            query_offset = k_len - q_len
            bias = _build_causal_bias(q_len, k_len, q.device, query_offset=query_offset).view(1, 1, q_len, k_len)
            output = _sdpa_attention(q_i, k_i, v_i, scale, num_key_value_groups, attention_bias=bias, is_causal=False)
        else:
            k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
            v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
            output = _sdpa_attention(q_i, k_i, v_i, scale, num_key_value_groups)
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)


def _flash_attention_fallback_decode(
    q: torch.Tensor,
    scale: float,
    num_key_value_groups: int,
    context,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    if q.is_cuda and context.block_tables is not None and context.context_lens is not None:
        batch_size = int(q.shape[0])
        max_num_blocks = int(context.block_tables.shape[1])
        block_size = int(k_cache.shape[1])
        total_cache_tokens = max_num_blocks * block_size
        flat_block_ids = context.block_tables.clamp_min(0).reshape(-1).long()
        k_tokens = k_cache.index_select(0, flat_block_ids).reshape(batch_size, max_num_blocks, block_size, k_cache.shape[-2], k_cache.shape[-1]).reshape(batch_size, total_cache_tokens, k_cache.shape[-2], k_cache.shape[-1])
        v_tokens = v_cache.index_select(0, flat_block_ids).reshape(batch_size, max_num_blocks, block_size, v_cache.shape[-2], v_cache.shape[-1]).reshape(batch_size, total_cache_tokens, v_cache.shape[-2], v_cache.shape[-1])
        block_mask = context.block_tables.ge(0).unsqueeze(-1).expand(-1, -1, block_size).reshape(batch_size, total_cache_tokens)
        token_positions = torch.arange(total_cache_tokens, device=q.device, dtype=context.context_lens.dtype).unsqueeze(0)
        valid_tokens = block_mask & token_positions.lt(context.context_lens.unsqueeze(1))
        has_tokens = valid_tokens.any(dim=1, keepdim=True)
        safe_valid_tokens = valid_tokens | (~has_tokens & token_positions.eq(0))
        kv_valid_mask = safe_valid_tokens.unsqueeze(-1).unsqueeze(-1)
        # The last active block may be only partially written. Zero invalid slots by replacement
        # so unwritten `torch.empty` cache contents cannot leak NaNs into SDPA.
        k_tokens = torch.where(kv_valid_mask, k_tokens, torch.zeros((), dtype=k_tokens.dtype, device=k_tokens.device))
        v_tokens = torch.where(kv_valid_mask, v_tokens, torch.zeros((), dtype=v_tokens.dtype, device=v_tokens.device))
        attention_bias = safe_valid_tokens.unsqueeze(1).unsqueeze(1)
        attn_output = _sdpa_attention(
            q.unsqueeze(2),
            k_tokens.permute(0, 2, 1, 3),
            v_tokens.permute(0, 2, 1, 3),
            scale,
            num_key_value_groups,
            attention_bias=attention_bias,
            is_causal=False,
        ).transpose(1, 2)
        return attn_output * has_tokens.view(batch_size, 1, 1, 1).to(attn_output.dtype)

    outputs = []
    for idx in range(int(q.shape[0])):
        seq_len = int(context.context_lens[idx].item())
        q_i = q[idx:idx + 1].transpose(0, 1).unsqueeze(0)
        k_i = _gather_cache_tokens(k_cache, context.block_tables[idx], seq_len).transpose(0, 1).unsqueeze(0)
        v_i = _gather_cache_tokens(v_cache, context.block_tables[idx], seq_len).transpose(0, 1).unsqueeze(0)
        outputs.append(_sdpa_attention(q_i, k_i, v_i, scale, num_key_value_groups, is_causal=False).transpose(1, 2))
    return torch.cat(outputs, dim=0)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    use_triton_kv_cache: bool | None = None,
):
    assert slot_mapping.numel() == key.shape[0]
    if use_triton_kv_cache is None:
        use_triton_kv_cache = _USE_TRITON_KV_CACHE
    if use_triton_kv_cache:
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
        return

    if slot_mapping.is_cuda and torch.cuda.is_current_stream_capturing():
        flat_k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
        flat_v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
        slot_ids = slot_mapping.long()
        flat_k_cache.index_copy_(0, slot_ids, key)
        flat_v_cache.index_copy_(0, slot_ids, value)
        return

    valid_mask = slot_mapping >= 0
    if not torch.any(valid_mask):
        return
    flat_k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
    flat_v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
    slot_ids = slot_mapping[valid_mask].long()
    flat_k_cache[slot_ids] = key[valid_mask]
    flat_v_cache[slot_ids] = value[valid_mask]


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.flash_attn_varlen_func = _DEFAULT_FLASH_ATTN_VARLEN_FUNC
        self.flash_attn_with_kvcache = _DEFAULT_FLASH_ATTN_WITH_KVCACHE
        self.use_triton_kv_cache = _DEFAULT_USE_TRITON_KV_CACHE
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, use_triton_kv_cache=self.use_triton_kv_cache)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            if self.flash_attn_varlen_func is not None:
                o = self.flash_attn_varlen_func(q, k, v,
                                                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                                softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = _flash_attention_fallback_prefill(
                    q,
                    k,
                    v,
                    self.scale,
                    self.num_heads // self.num_kv_heads,
                    context,
                    k_cache,
                    v_cache,
                )
        else:    # decode
            if self.flash_attn_with_kvcache is not None:
                o = self.flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                                 cache_seqlens=context.context_lens, block_table=context.block_tables,
                                                 softmax_scale=self.scale, causal=True)
            else:
                o = _flash_attention_fallback_decode(
                    q,
                    self.scale,
                    self.num_heads // self.num_kv_heads,
                    context,
                    k_cache,
                    v_cache,
                )
        return o
