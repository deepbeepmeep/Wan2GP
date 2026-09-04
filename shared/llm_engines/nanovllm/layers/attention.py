import torch
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as tl_libdevice
except Exception:  # pragma: no cover
    triton = None
    tl = None
    tl_libdevice = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except Exception:  # pragma: no cover
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
try:
    import llamacpp_gguf_cuda
except Exception:  # pragma: no cover
    llamacpp_gguf_cuda = None
from ..utils.context import get_context

_DEFAULT_FLASH_ATTN_VARLEN_FUNC = flash_attn_varlen_func
_DEFAULT_FLASH_ATTN_WITH_KVCACHE = flash_attn_with_kvcache
_USE_TRITON_KV_CACHE = triton is not None and tl is not None
_DEFAULT_USE_TRITON_KV_CACHE = _USE_TRITON_KV_CACHE
_Q8_KV_BLOCK_SIZE = 32
_LOGGED_KV_ATTENTION_BACKENDS = set()


def _log_kv_attention_backend_once(backend: str, message: str) -> None:
    if backend in _LOGGED_KV_ATTENTION_BACKENDS:
        return
    _LOGGED_KV_ATTENTION_BACKENDS.add(backend)
    print(message)


def reset_attention_backend_logs() -> None:
    _LOGGED_KV_ATTENTION_BACKENDS.clear()


_Q8_PAGED_ATTENTION = llamacpp_gguf_cuda.q8_paged_attention if llamacpp_gguf_cuda is not None and getattr(llamacpp_gguf_cuda, "q8_paged_attention_format", lambda: "")() == "q8_0_fp16_scales_v1" else None


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


    @triton.jit
    def store_int8_kvcache_kernel(
        key_ptr,
        key_stride_token,
        key_stride_head,
        value_ptr,
        value_stride_token,
        value_stride_head,
        k_cache_ptr,
        v_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,
        slot_mapping_ptr,
        H: tl.constexpr,
        D: tl.constexpr,
        QB: tl.constexpr,
    ):
        idx = tl.program_id(0)
        token_idx = idx // (H * QB)
        head_block_idx = idx % (H * QB)
        head_idx = head_block_idx // QB
        quant_block_idx = head_block_idx % QB
        slot = tl.load(slot_mapping_ptr + token_idx)
        if slot == -1: return
        offsets = quant_block_idx * 32 + tl.arange(0, 32)
        key = tl.load(key_ptr + token_idx * key_stride_token + head_idx * key_stride_head + offsets).to(tl.float32)
        value = tl.load(value_ptr + token_idx * value_stride_token + head_idx * value_stride_head + offsets).to(tl.float32)
        key_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / 127.0, 1e-8)
        value_scale = tl.maximum(tl.max(tl.abs(value), axis=0) / 127.0, 1e-8)
        key_int8 = tl_libdevice.rint(key / key_scale).to(tl.int8)
        value_int8 = tl_libdevice.rint(value / value_scale).to(tl.int8)
        cache_offsets = (slot * H + head_idx) * D + offsets
        scale_offset = (slot * H + head_idx) * QB + quant_block_idx
        tl.store(k_cache_ptr + cache_offsets, key_int8)
        tl.store(v_cache_ptr + cache_offsets, value_int8)
        tl.store(k_scale_ptr + scale_offset, key_scale)
        tl.store(v_scale_ptr + scale_offset, value_scale)


    @triton.jit
    def q8_paged_prefill_kernel(
        q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, block_tables_ptr, cu_q_ptr, cu_k_ptr, out_ptr,
        q_stride_t: tl.constexpr, q_stride_h: tl.constexpr, cache_stride_block: tl.constexpr,
        cache_stride_token: tl.constexpr, cache_stride_head: tl.constexpr, scale_stride_block: tl.constexpr,
        scale_stride_token: tl.constexpr, scale_stride_head: tl.constexpr, bt_stride: tl.constexpr,
        out_stride_t: tl.constexpr, out_stride_h: tl.constexpr, softmax_scale: tl.constexpr,
        H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        sequence = tl.program_id(0)
        query_head = tl.program_id(1)
        query_block = tl.program_id(2)
        q_start = tl.load(cu_q_ptr + sequence)
        q_end = tl.load(cu_q_ptr + sequence + 1)
        k_end = tl.load(cu_k_ptr + sequence + 1) - tl.load(cu_k_ptr + sequence)
        q_len = q_end - q_start
        prefix_len = k_end - q_len
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, D)
        row_mask = rows < q_len
        q = tl.load(q_ptr + (q_start + rows[:, None]) * q_stride_t + query_head * q_stride_h + dims[None, :], mask=row_mask[:, None]).to(tl.float32)
        maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        denominator = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, D), tl.float32)
        kv_head = query_head // (H_Q // H_KV)

        for key_start in tl.range(0, k_end, BLOCK_N):
            columns = key_start + tl.arange(0, BLOCK_N)
            column_mask = columns < k_end
            physical_block = tl.load(block_tables_ptr + sequence * bt_stride + columns // PAGE, mask=column_mask, other=0)
            physical_token = columns % PAGE
            cache_base = physical_block * cache_stride_block + physical_token * cache_stride_token + kv_head * cache_stride_head
            scale_base = physical_block * scale_stride_block + physical_token * scale_stride_token + kv_head * scale_stride_head
            k = tl.load(k_ptr + cache_base[None, :] + dims[:, None], mask=column_mask[None, :], other=0.0).to(tl.float32)
            k *= tl.load(ks_ptr + scale_base[None, :] + (dims[:, None] // 32), mask=column_mask[None, :], other=0.0).to(tl.float32)
            scores = tl.dot(q, k, input_precision="ieee") * softmax_scale
            causal = columns[None, :] <= prefix_len + rows[:, None]
            scores = tl.where(row_mask[:, None] & column_mask[None, :] & causal, scores, -float("inf"))
            tile_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            correction = tl.exp2((maximum - tile_maximum) * 1.4426950408889634)
            probabilities = tl.exp2((scores - tile_maximum[:, None]) * 1.4426950408889634)
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
            v = tl.load(v_ptr + cache_base[:, None] + dims[None, :], mask=column_mask[:, None], other=0.0).to(tl.float32)
            v *= tl.load(vs_ptr + scale_base[:, None] + (dims[None, :] // 32), mask=column_mask[:, None], other=0.0).to(tl.float32)
            accumulator = accumulator * correction[:, None] + tl.dot(probabilities, v, input_precision="ieee")
            maximum = tile_maximum
        output = accumulator / denominator[:, None]
        tl.store(out_ptr + (q_start + rows[:, None]) * out_stride_t + query_head * out_stride_h + dims[None, :], output, mask=row_mask[:, None])


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


def _flash_attention_fallback_speculative(
    q: torch.Tensor,
    scale: float,
    num_key_value_groups: int,
    context,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    if q.is_cuda:
        query_length = int(q.shape[0])
        block_size = int(k_cache.shape[1])
        max_num_blocks = int(context.block_tables.shape[1])
        total_cache_tokens = max_num_blocks * block_size
        block_ids = context.block_tables[0].clamp_min(0).long()
        keys = k_cache.index_select(0, block_ids).reshape(total_cache_tokens, k_cache.shape[-2], k_cache.shape[-1])
        values = v_cache.index_select(0, block_ids).reshape(total_cache_tokens, v_cache.shape[-2], v_cache.shape[-1])
        token_positions = torch.arange(total_cache_tokens, device=q.device, dtype=context.cu_seqlens_k.dtype)
        sequence_length = context.cu_seqlens_k[-1]
        cache_valid = context.block_tables[0].ge(0).unsqueeze(-1).expand(-1, block_size).reshape(total_cache_tokens) & token_positions.lt(sequence_length)
        kv_valid_mask = cache_valid.unsqueeze(-1).unsqueeze(-1)
        keys = torch.where(kv_valid_mask, keys, torch.zeros((), dtype=keys.dtype, device=keys.device))
        values = torch.where(kv_valid_mask, values, torch.zeros((), dtype=values.dtype, device=values.device))
        query_positions = sequence_length - query_length + torch.arange(query_length, device=q.device, dtype=sequence_length.dtype)
        attention_bias = (cache_valid.unsqueeze(0) & token_positions.unsqueeze(0).le(query_positions.unsqueeze(1))).unsqueeze(0).unsqueeze(0)
        return _sdpa_attention(q.transpose(0, 1).unsqueeze(0), keys.transpose(0, 1).unsqueeze(0), values.transpose(0, 1).unsqueeze(0), scale, num_key_value_groups, attention_bias=attention_bias, is_causal=False).squeeze(0).transpose(0, 1)

    sequence_length = int(context.cu_seqlens_k[-1].item())
    query_length = int(q.shape[0])
    keys = _gather_cache_tokens(k_cache, context.block_tables[0], sequence_length).transpose(0, 1).unsqueeze(0)
    values = _gather_cache_tokens(v_cache, context.block_tables[0], sequence_length).transpose(0, 1).unsqueeze(0)
    bias = _build_causal_bias(query_length, sequence_length, q.device, query_offset=sequence_length - query_length).view(1, 1, query_length, sequence_length)
    return _sdpa_attention(q.transpose(0, 1).unsqueeze(0), keys, values, scale, num_key_value_groups, attention_bias=bias, is_causal=False).squeeze(0).transpose(0, 1)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    use_triton_kv_cache: bool | None = None,
):
    assert slot_mapping.numel() == key.shape[0]
    quantized = k_cache.dtype == torch.int8
    if quantized and (k_scale is None or v_scale is None):
        raise RuntimeError("INT8 KV cache requires key and value scales.")
    if use_triton_kv_cache is None:
        use_triton_kv_cache = _USE_TRITON_KV_CACHE
    if use_triton_kv_cache:
        N, num_heads, head_dim = key.shape
        if quantized:
            if head_dim % _Q8_KV_BLOCK_SIZE:
                raise RuntimeError(f"INT8 KV cache head dimension must be divisible by {_Q8_KV_BLOCK_SIZE}; got {head_dim}.")
            quant_blocks = head_dim // _Q8_KV_BLOCK_SIZE
            store_int8_kvcache_kernel[(N * num_heads * quant_blocks,)](
                key, key.stride(0), key.stride(1), value, value.stride(0), value.stride(1),
                k_cache, v_cache, k_scale, v_scale, slot_mapping, num_heads, head_dim, quant_blocks,
            )
            return
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
        return

    if quantized:
        quant_blocks = key.shape[-1] // _Q8_KV_BLOCK_SIZE
        key_blocks = key.reshape(*key.shape[:-1], quant_blocks, _Q8_KV_BLOCK_SIZE)
        value_blocks = value.reshape(*value.shape[:-1], quant_blocks, _Q8_KV_BLOCK_SIZE)
        key_scale = key_blocks.float().abs().amax(dim=-1).div_(127).clamp_min_(1e-8).to(k_scale.dtype)
        value_scale = value_blocks.float().abs().amax(dim=-1).div_(127).clamp_min_(1e-8).to(v_scale.dtype)
        key = key_blocks.div(key_scale.unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).reshape_as(key)
        value = value_blocks.div(value_scale.unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).reshape_as(value)

    if slot_mapping.is_cuda and torch.cuda.is_current_stream_capturing():
        flat_k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
        flat_v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
        slot_ids = slot_mapping.long()
        flat_k_cache.index_copy_(0, slot_ids, key)
        flat_v_cache.index_copy_(0, slot_ids, value)
        if quantized:
            k_scale.reshape(-1, k_scale.shape[-2], k_scale.shape[-1]).index_copy_(0, slot_ids, key_scale)
            v_scale.reshape(-1, v_scale.shape[-2], v_scale.shape[-1]).index_copy_(0, slot_ids, value_scale)
        return

    valid_mask = slot_mapping >= 0
    if not torch.any(valid_mask):
        return
    flat_k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
    flat_v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
    slot_ids = slot_mapping[valid_mask].long()
    flat_k_cache[slot_ids] = key[valid_mask]
    flat_v_cache[slot_ids] = value[valid_mask]
    if quantized:
        k_scale.reshape(-1, k_scale.shape[-2], k_scale.shape[-1])[slot_ids] = key_scale[valid_mask]
        v_scale.reshape(-1, v_scale.shape[-2], v_scale.shape[-1])[slot_ids] = value_scale[valid_mask]


def _dequantize_kvcache(cache: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return cache.reshape(*cache.shape[:-1], scale.shape[-1], _Q8_KV_BLOCK_SIZE).to(dtype).mul_(scale.to(dtype).unsqueeze(-1)).reshape_as(cache)


def _q8_paged_prefill(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor, context, softmax_scale: float) -> torch.Tensor:
    output = torch.empty_like(q)
    query_lengths = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
    grid = (query_lengths.numel(), q.shape[1], triton.cdiv(q.shape[0], 32))
    q8_paged_prefill_kernel[grid](
        q, k_cache, v_cache, k_scale, v_scale, context.block_tables, context.cu_seqlens_q, context.cu_seqlens_k, output,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), context.block_tables.stride(0), output.stride(0), output.stride(1), softmax_scale,
        q.shape[1], k_cache.shape[2], q.shape[2], k_cache.shape[1], BLOCK_M=32, BLOCK_N=16, num_warps=8, num_stages=1,
    )
    return output


def _q8_paged_speculative(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    return _Q8_PAGED_ATTENTION(q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, softmax_scale).squeeze(1)


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
        self.k_scale = self.v_scale = torch.tensor([])
        self._q8_speculative_metadata = {}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.k_scale, self.v_scale, use_triton_kv_cache=self.use_triton_kv_cache)
        quantized_cache = k_cache.dtype == torch.int8
        reads_cache = context.speculative_verify or not context.is_prefill or context.block_tables is not None
        q8_attention = _Q8_PAGED_ATTENTION if quantized_cache and not context.speculative_verify and not context.is_prefill else None
        q8_prefill = quantized_cache and context.is_prefill and context.block_tables is not None and triton is not None
        q8_speculative = quantized_cache and context.speculative_verify and _Q8_PAGED_ATTENTION is not None
        if (q8_attention is not None or q8_prefill or q8_speculative) and q.is_cuda and q.is_contiguous() and q.dtype in (torch.float16, torch.bfloat16) and self.head_dim <= 256 and self.head_dim % _Q8_KV_BLOCK_SIZE == 0:
            dtype_name = "BF16" if q.dtype == torch.bfloat16 else "FP16"
            if context.speculative_verify:
                _log_kv_attention_backend_once("speculative_llamacpp_q8", f"[Deepy][Speculative] verification backend=batched llama.cpp fattn-vec Q8 paged attention ({dtype_name} I/O, FP32 accumulation, no dense KV materialization).")
                query_count = q.shape[0]
                metadata = self._q8_speculative_metadata.get(query_count)
                if metadata is None:
                    metadata = (
                        torch.arange(1, query_count + 1, dtype=torch.int32, device=q.device),
                        torch.empty(query_count, dtype=torch.int32, device=q.device),
                        torch.empty(query_count, context.block_tables.shape[1], dtype=torch.int32, device=q.device),
                    )
                    self._q8_speculative_metadata[query_count] = metadata
                query_offsets, context_lens, block_tables = metadata
                context_lens.copy_(query_offsets).add_(context.cu_seqlens_k[1] - query_count)
                block_tables.copy_(context.block_tables.expand(query_count, -1))
                return _q8_paged_speculative(q, k_cache, v_cache, self.k_scale, self.v_scale, context_lens, block_tables, self.scale)
            elif context.is_prefill:
                _log_kv_attention_backend_once("prefill_triton_q8", f"[Deepy][KV cache] prefix prefill backend=tiled Triton Q8 paged attention ({dtype_name} I/O, FP32 accumulation, no dense KV materialization).")
                return _q8_paged_prefill(q, k_cache, v_cache, self.k_scale, self.v_scale, context, self.scale)
            else:
                _log_kv_attention_backend_once("llamacpp_q8", f"[Deepy][KV cache] decode backend=llama.cpp fattn-vec Q8 paged adapter ({dtype_name} I/O, FP32 accumulation).")
            context_lens = context.cu_seqlens_k[1:] if context.speculative_verify else context.context_lens
            output = q8_attention(q, k_cache, v_cache, self.k_scale, self.v_scale, context.block_tables, context_lens, self.scale)
            return output.squeeze(1) if context.speculative_verify else output
        attention_k_cache = _dequantize_kvcache(k_cache, self.k_scale, q.dtype) if quantized_cache and reads_cache else k_cache
        attention_v_cache = _dequantize_kvcache(v_cache, self.v_scale, q.dtype) if quantized_cache and reads_cache else v_cache
        if context.speculative_verify:
            if self.flash_attn_with_kvcache is None:
                detail = " with on-the-fly Q8 cache dequantization" if quantized_cache else ""
                _log_kv_attention_backend_once("speculative_sdpa", f"[Deepy][Speculative] verification backend=PyTorch SDPA{detail}.")
                return _flash_attention_fallback_speculative(q, self.scale, self.num_heads // self.num_kv_heads, context, attention_k_cache, attention_v_cache)
            detail = " with on-the-fly Q8 cache dequantization" if quantized_cache else ""
            _log_kv_attention_backend_once("speculative_flash", f"[Deepy][Speculative] verification backend=FlashAttention KV cache{detail}.")
            # The q=2 verification queries have already been written to the paged
            # cache. FlashAttention's KV-cache API replays this layout correctly;
            # its varlen prefill API does not preserve the first query on replay.
            return self.flash_attn_with_kvcache(
                q.unsqueeze(0),
                attention_k_cache,
                attention_v_cache,
                cache_seqlens=context.cu_seqlens_k[1:],
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            ).squeeze(0)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = attention_k_cache, attention_v_cache
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
                    attention_k_cache,
                    attention_v_cache,
                )
        else:    # decode
            if self.flash_attn_with_kvcache is not None:
                if quantized_cache:
                    _log_kv_attention_backend_once("flash_q8_fallback", "[Deepy][KV cache] llama.cpp Q8 decode unavailable; backend=FlashAttention with on-the-fly cache dequantization.")
                o = self.flash_attn_with_kvcache(q.unsqueeze(1), attention_k_cache, attention_v_cache,
                                                 cache_seqlens=context.context_lens, block_table=context.block_tables,
                                                 softmax_scale=self.scale, causal=True)
            else:
                if quantized_cache:
                    _log_kv_attention_backend_once("sdpa_q8_fallback", "[Deepy][KV cache] llama.cpp Q8 decode and FlashAttention unavailable; backend=PyTorch SDPA with on-the-fly cache dequantization.")
                o = _flash_attention_fallback_decode(
                    q,
                    self.scale,
                    self.num_heads // self.num_kv_heads,
                    context,
                    attention_k_cache,
                    attention_v_cache,
                )
        return o

    def forward_list(self, qkv_list: list[torch.Tensor]):
        q, k, v = qkv_list
        qkv_list.clear()
        return self.forward(q, k, v)
