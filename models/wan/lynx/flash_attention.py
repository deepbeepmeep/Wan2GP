# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 

import torch
try:
    from flash_attn_interface import flash_attn_varlen_func # flash attn 3
except:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func # flash attn 2

def flash_attention(query, key, value, q_lens, kv_lens, causal=False):
    """
    Args:
        query, key, value: [B, H, T, D_h]
        q_lens: list[int] per sequence query length
        kv_lens: list[int] per sequence key/value length
        causal: whether to use causal mask

    Returns:
        output: [B, H, T_q, D_h]
    """
    B, H, T_q, D_h = query.shape
    T_k = key.shape[2]
    device = query.device

    # Flatten: [B, H, T, D] -> [total_tokens, H, D]
    q = query.permute(0, 2, 1, 3).reshape(B * T_q, H, D_h)
    k = key.permute(0, 2, 1, 3).reshape(B * T_k, H, D_h)
    v = value.permute(0, 2, 1, 3).reshape(B * T_k, H, D_h)

    # Prepare cu_seqlens: prefix sum
    q_lens_tensor = torch.tensor(q_lens, device=device, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, device=device, dtype=torch.int32)

    cu_seqlens_q = torch.zeros(len(q_lens_tensor) + 1, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(len(kv_lens_tensor) + 1, device=device, dtype=torch.int32)

    cu_seqlens_q[1:] = torch.cumsum(q_lens_tensor, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(kv_lens_tensor, dim=0)

    max_seqlen_q = int(q_lens_tensor.max().item())
    max_seqlen_k = int(kv_lens_tensor.max().item())

    # Call FlashAttention varlen kernel
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal
    )
    if not torch.is_tensor(out): # flash attn 3
        out = out[0]

    # Restore shape: [total_q, H, D_h] -> [B, H, T_q, D_h]
    out = out.view(B, T_q, H, D_h).permute(0, 2, 1, 3).contiguous()

    return out