import torch
from torch import nn
from typing import Optional
import os


_SAMPLER_NUMERIC_GUARD = os.environ.get("WAN2GP_NANOVLLM_SAMPLER_NUMERIC_GUARD", "0") == "1"


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits (vLLM style).
    
    The logits tensor is updated in-place.
    """
    squeezed = False
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeezed = True
    if k is not None:
        k = k.reshape(-1)
    if p is not None:
        p = p.reshape(-1)

    if p is None:
        if k is None:
            return logits.squeeze(0) if squeezed else logits
        # Avoid sorting vocab for top-k only case
        logits = apply_top_k_only(logits, k)
        return logits.squeeze(0) if squeezed else logits

    # Need to sort for top-p
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k first
        vocab_size = logits_sort.size(1)
        # Clamp k to valid range
        k_clamped = k.clamp(1, vocab_size).long()
        top_k_mask_idx = vocab_size - k_clamped  # shape: [B]
        # Get the threshold value for each batch
        top_k_thresh = logits_sort.gather(1, top_k_mask_idx.unsqueeze(1))
        top_k_mask = logits_sort < top_k_thresh
        logits_sort.masked_fill_(top_k_mask, float('-inf'))

    # Apply top-p
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)  # reuse buffer
    top_p_mask = probs_sum <= (1.0 - p.unsqueeze(1))
    # Ensure at least one token is kept
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, float('-inf'))

    # Re-sort back to original positions
    logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
    return logits.squeeze(0) if squeezed else logits


def apply_min_p(
    logits: torch.Tensor,
    min_p: Optional[torch.Tensor],
) -> torch.Tensor:
    if min_p is None:
        return logits
    squeezed = False
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeezed = True
    min_p = min_p.reshape(-1)
    active_mask = min_p > 0
    if not torch.any(active_mask):
        return logits.squeeze(0) if squeezed else logits
    probs = logits.softmax(dim=-1)
    max_probs, max_idx = probs.max(dim=-1, keepdim=True)
    threshold = max_probs * min_p.unsqueeze(1)
    mask = probs < threshold
    mask.scatter_(1, max_idx, False)
    mask &= active_mask.unsqueeze(1)
    logits.masked_fill_(mask, float("-inf"))
    return logits.squeeze(0) if squeezed else logits


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Apply top-k mask without sorting the entire vocab (vLLM style).
    
    This is much faster than sorting for top-k only cases.
    The logits tensor is updated in-place.
    """
    squeezed = False
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeezed = True
    k = k.reshape(-1)

    vocab_size = logits.shape[1]
    # Handle cases where k >= vocab_size (no filtering needed)
    no_top_k_mask = (k <= 0) | (k >= vocab_size)
    # Set invalid k to 1 so we can still gather
    k_safe = k.masked_fill(no_top_k_mask, 1).long()
    # NOTE: This int() causes CPU-GPU sync, but torch.topk requires Python int
    max_top_k = int(k_safe.max().clamp(max=vocab_size))
    
    # Get top-k values for all batches
    # topk.values has shape [batch_size, max_top_k]
    topk_values = logits.topk(max_top_k, dim=1).values
    
    # Convert k to 0-based index: we want the k-th largest value (index k-1)
    # Clamp to valid range for gather
    k_index = (k_safe - 1).clamp(0, max_top_k - 1).unsqueeze(1)  # shape: [B, 1]
    # Gather the threshold value (the k-th largest)
    top_k_thresh = topk_values.gather(1, k_index)
    
    # For rows with no top-k filtering, set threshold to -inf so nothing gets masked
    top_k_thresh.masked_fill_(no_top_k_mask.unsqueeze(1), float('-inf'))
    
    # Mask all values below the threshold
    logits.masked_fill_(logits < top_k_thresh, float('-inf'))
    return logits.squeeze(0) if squeezed else logits


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self, 
        logits: torch.Tensor, 
        temperatures: torch.Tensor,
        top_ks: Optional[torch.Tensor] = None,
        top_ps: Optional[torch.Tensor] = None,
        min_ps: Optional[torch.Tensor] = None,
        repetition_penalties: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Sample tokens from logits with optional top-k and top-p filtering.
        
        Condition checking is done OUTSIDE the compiled function to avoid
        graph breaks from .any() calls.
        """
        # Apply temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        logits = apply_top_k_top_p(
            logits,
            top_ks,
            top_ps,
        )
        logits = apply_min_p(logits, min_ps)
        if _SAMPLER_NUMERIC_GUARD:
            logits = torch.nan_to_num(logits, nan=float("-inf"))
            invalid_rows = ~torch.isfinite(logits).any(dim=-1)
            if invalid_rows.any():
                logits[invalid_rows, 0] = 0.0
        probs = torch.softmax(logits, dim=-1)
        noise = torch.empty_like(probs).exponential_(1, generator=generator).clamp_min_(1e-10)
        sample_tokens = probs.div_(noise).argmax(dim=-1).reshape(-1)
        return sample_tokens
