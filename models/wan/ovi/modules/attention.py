# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch



def attention_with_weights(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    average_for_q=False,
    total_video_latent_frames = 21
):
    """
    Compute attention with explicit attention weights for visualization.
    Returns both output and attention weights.
    """
    out_dtype = q.dtype
    
    # Handle sequence lengths
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    
    if q_lens is None:
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        # Ensure q_lens is on the same device as q
        q_lens = q_lens.to(q.device)
        
    if k_lens is None:
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        # Ensure k_lens is on the same device as k
        k_lens = k_lens.to(k.device)
    
    # Apply q_scale if provided
    if q_scale is not None:
        q = q * q_scale
    
    # Compute attention weights manually
    # q: [B, Lq, Nq, C], k: [B, Lk, Nk, C]
    scale = softmax_scale if softmax_scale is not None else (q.size(-1) ** -0.5)
    
    # Compute scores: [B, Nq, Lq, Lk]
    scores = torch.einsum('blhd,bshd->bhls', q, k) * scale
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones(lq, lk, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Mask for k_lens (columns)
    k_mask = torch.arange(lk, device=k.device).unsqueeze(0) >= k_lens.unsqueeze(1)  # [B, Lk]
    scores.masked_fill_(k_mask.unsqueeze(1).unsqueeze(2), float('-inf'))  # [B, 1, 1, Lk]
    
    # Mask for q_lens (rows) 
    q_mask = torch.arange(lq, device=q.device).unsqueeze(0) >= q_lens.unsqueeze(1)  # [B, Lq]
    scores.masked_fill_(q_mask.unsqueeze(1).unsqueeze(3), float('-inf'))  # [B, 1, Lq, 1]
    
    # Compute attention weights
    attn_weights = torch.softmax(scores, dim=-1)  # [B, Nq, Lq, Lk]
    assert attn_weights.shape[0] == 1, "Batch size > 1 not supported for attention visualization."
    
    # Average attention weights to reduce memory usage before returning
    # Average across batch dimension (should be 1) and query heads and query sequence length
    # This gives us attention weight per video token: [Lk]
    if average_for_q:
        #avg_attn_weights = torch.mean(attn_weights, dim=(0, 1, 3))  # [Lq]
        avg_attn_weights = torch.max(attn_weights, dim=3)[0].mean(dim=(0, 1))  # [Lq]
    else:
        if 0:
            avg_attn_weights = torch.mean(attn_weights, dim=(0, 1, 2))  # [Lk]
        elif 1:
            B, H, Lq, Lk = attn_weights.shape  # [1, H, Lq, Lk]
            per_frame_seq_len = Lk // total_video_latent_frames
            per_frame_aud_len = Lq // total_video_latent_frames

            avg_attn_weights = torch.zeros((Lk,), device=attn_weights.device, dtype=attn_weights.dtype)

            eps = 1e-8  # numerical stability
            for i in range(total_video_latent_frames):
                start_idx_v = i * per_frame_seq_len
                end_idx_v   = (i + 1) * per_frame_seq_len

                start_idx_a = i * per_frame_aud_len
                end_idx_a   = (i + 1) * per_frame_aud_len

                # attn_chunk: [H, La, Lv]
                attn_chunk = attn_weights[0, :, start_idx_a:end_idx_a, start_idx_v:end_idx_v]

                # ---- Head informativeness via (low) entropy over Lv ----
                # Normalize within the Lv slice per (head, query) to make a proper distribution
                p = attn_chunk / (attn_chunk.sum(dim=-1, keepdim=True) + eps)          # [H, La, Lv]
                entropy = -(p * (p + eps).log()).sum(dim=-1).mean(dim=1)               # [H]

                # Convert to positive head weights (lower entropy -> larger weight)
                saliency = 1.0 / (entropy + 1e-6)                                      # [H]
                head_w = saliency / (saliency.sum() + eps)                             # [H], sum=1

                # Reduce across audio queries first (pick strong responses), then weight heads
                per_head = torch.amax(attn_chunk, dim=1)                               # [H, Lv]
                weighted = (per_head * head_w[:, None]).sum(dim=0)                     # [Lv]

                avg_attn_weights[start_idx_v:end_idx_v] = weighted
        else:
            avg_attn_weights = torch.mean(attn_weights, dim=(0, 2)).max(dim=(0))[0]  # [Lk]
    
    # Compute output: [B, Lq, Nq, C]
    out = torch.einsum('bhls,bshd->blhd', attn_weights, v)
    
    return out.to(out_dtype), avg_attn_weights.to(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
