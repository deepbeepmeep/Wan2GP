# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""GPU segments of hybrid CPU/GPU autoregressive decoders.

The sequence loop stays in Python for CPU experts. MMGP graphs each GPU segment,
and owns all weight placement. Reuse the same attention/DeltaNet kernels as the
ordinary nano-vLLM runner; legacy/cg use PyTorch without Triton or FlashAttention.
"""
import torch
from shared.attention import pay_attention


def cached_attention(qkv, group, cache, positions, kv_len, engine):
    query, key, value = qkv
    qkv.clear()
    key_cache, value_cache = cache
    start, stop = group.kv.start, group.kv.stop
    heads = stop - start
    groups = query.shape[2] // heads
    key, value = key[:, :, ::groups], value[:, :, ::groups]
    key_cache[:, :, start:stop].index_copy_(1, positions, key)
    value_cache[:, :, start:stop].index_copy_(1, positions, value)
    if engine == "vllm":
        from .layers.attention import _DEFAULT_FLASH_ATTN_WITH_KVCACHE
        if _DEFAULT_FLASH_ATTN_WITH_KVCACHE is None:
            raise RuntimeError("The vLLM layerwise decoder requires FlashAttention 2")
        lengths = (positions[-1:] + 1).to(dtype=torch.int32).expand(query.shape[0]).contiguous()
        # Several query rows (prompt processing) fill the GPU without splitting the keys, whose float32 partial outputs would grow with
        # the rows; a decoding step keeps FlashAttention's choice.
        return _DEFAULT_FLASH_ATTN_WITH_KVCACHE(query, key_cache[:, :, start:stop], value_cache[:, :, start:stop],
                                              cache_seqlens=lengths, causal=True, num_splits=1 if query.shape[1] > 1 else 0)
    key, value = key_cache[:, :kv_len, start:stop], value_cache[:, :kv_len, start:stop]
    if groups > 1:
        key, value = key.repeat_interleave(groups, dim=2), value.repeat_interleave(groups, dim=2)
    mask = torch.arange(kv_len, device=query.device)[None, :] <= positions[:, None]
    return pay_attention([query, key, value], attention_mask=mask[None, :, None, :], force_attention="sdpa")


def delta_rule(query, key, value, decay, beta, initial_state, *, engine, recurrent, torch_impl):
    if engine == "vllm":
        from .models.qwen3_5 import (_DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE, _DEFAULT_FAST_CHUNK_GATED_DELTA_RULE,
                                    configure_qwen35_fla_prefill_autotune)
        configure_qwen35_fla_prefill_autotune(query.device)
        fn = _DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE if recurrent else _DEFAULT_FAST_CHUNK_GATED_DELTA_RULE
        if fn is None:
            raise RuntimeError("The vLLM layerwise decoder requires the shared FLA DeltaNet kernels")
        return fn(query.contiguous(), key.contiguous(), value.contiguous(), g=decay.contiguous(), beta=beta.contiguous(),
                  initial_state=initial_state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    return torch_impl(query, key, value, decay, beta, initial_state)
