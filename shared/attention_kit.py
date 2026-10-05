"""VRAM savings shared by model attentions: a model calls qkv_attention() instead of projecting q/k/v and calling pay_attention.

- SageAttention 2 (sm89/sm120): v, k and q are quantized as soon as each is projected (shared.sage2_core staged functions), so at most
  one of them exists in 16 bits; same results as pay_attention.
- The three projections share one input prepared by mmgp (offload.prepare_linear_input) when their weight format supports it: the
  input is quantized once instead of once per projection, and the 16-bit input is released before the projections.
- Attention Head Split (Performance tab, off by default): long sequences are attended one group of heads at a time into one output.
  q/k/v rows, per-head norms, RoPE and attention are per head; norms over all heads get per token statistics computed beforehand.
"""
from collections import namedtuple

import torch
from mmgp import offload

from shared.attention import pay_attention, sage2_staged_settings

HEAD_SPLIT_CHOICES = [("Off", 0), ("Low: saves some VRAM", 1), ("Medium (good balance): saves more VRAM", 2), ("High: saves the most VRAM", 3)]
_TARGET_GROUPS = (1, 4, 8, 16)
MIN_SPLIT_TOKENS = 8192  # shorter sequences are never split: little VRAM to save
head_split = 0
_unsplit_notice = False

# Head ranges of a group for q and for k/v; mean_squares: per token mean of the squares of q and of k over all heads, (batch, tokens, 1)
# in float32, when a norm spans all heads and the heads are split, otherwise None.
HeadGroup = namedtuple("HeadGroup", "q kv mean_squares")


def configure(level):
    global head_split
    head_split = int(level)


@torch.compiler.disable()
def kv_first_attention(qkv_list, force_attention=None):
    """pay_attention(qkv_list, force_attention=force_attention, recycle_q=True) when nothing else holds k and v: with SageAttention 2
    (sm89/sm120), k is quantized and released before v is copied for its quantization, a peak lower by half of k; same results."""
    settings = sage2_staged_settings(qkv_list[0].device, force_attention)
    if settings is None:
        return pay_attention(qkv_list, force_attention=force_attention, recycle_q=True)
    from shared import sage2_core
    query, key_list, value_list = qkv_list[0], qkv_list[1:2], qkv_list[2:3]
    qkv_list.clear()
    quantized = [None, sage2_core.staged_quantize_k(key_list, settings), sage2_core.staged_quantize_v(value_list, settings)]
    quantized[0] = sage2_core.staged_quantize_q(query, settings)
    return sage2_core.staged_attention(query, quantized, settings)


def head_groups(kv_heads, tokens, level=None):
    """Groups of heads of an attention over tokens (per batch item): the divisor of kv_heads closest to the level's target, 1 when off."""
    level = head_split if level is None else level
    if level == 0 or tokens < MIN_SPLIT_TOKENS:
        return 1
    return min((groups for groups in range(1, kv_heads + 1) if kv_heads % groups == 0), key=lambda groups: (abs(groups - _TARGET_GROUPS[level]), groups))


@torch.compiler.disable()
def qkv_attention(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, kv_heads=None, norm_spans_heads=False, split_heads=True):
    """Attention of the q/k/v projections of x (handed off in x_list, shape (batch, tokens, features) or (tokens, features)), returned
    as (batch, tokens, heads, head_dim), batch 1 for a 2D x.

    norm_rope(query, key, group) normalizes and rotates in place query and/or key, the other one may be None: (batch, tokens, heads of
    the group, head_dim) for the head ranges group.q and group.kv. With norm_spans_heads, a split group gets group.mean_squares to scale
    its rows; otherwise the tensors hold all heads. split_heads=False keeps all heads together (masked or sparse variants)."""
    global _unsplit_notice
    x = x_list[0]
    x_list.clear()
    shape = tuple(x.shape[:-1]) if x.dim() > 2 else (1, x.shape[0])
    kv_heads = kv_heads or heads
    groups = head_groups(kv_heads, shape[-1]) if split_heads else 1
    settings = sage2_staged_settings(x.device)
    projections = (q_proj, k_proj, v_proj)
    linear_input = None
    if offload.linear_rows_supported(projections, x):
        linear_input = offload.prepare_linear_input([x], projections)
        x = None
    elif groups > 1:
        if not _unsplit_notice:
            print("[Attention] Head Split is not applied: these q/k/v projections cannot be computed by groups of heads (weight format or LoRA type).")
            _unsplit_notice = True
        groups = 1

    def project(module, head_range):
        if linear_input is None:
            return module(x).view(*shape, -1, head_dim)
        return offload.linear_rows(module, linear_input, head_range.start * head_dim, head_range.stop * head_dim).view(*shape, -1, head_dim)

    mean_squares = None
    if norm_spans_heads and groups > 1:
        mean_squares = []
        for module, total in ((q_proj, heads), (k_proj, kv_heads)):
            step, squares = total // groups, None
            for group in range(groups):
                part = project(module, range(group * step, (group + 1) * step)).float().square_().sum(dim=(-2, -1))
                squares = part if squares is None else squares.add_(part)
                del part
            mean_squares.append(squares.div_(total * head_dim).unsqueeze(-1))
        mean_squares = tuple(mean_squares)

    if settings is not None:
        from shared import sage2_core
    q_heads, kv_group_heads = heads // groups, kv_heads // groups
    output = None
    for index in range(groups):
        group = HeadGroup(range(index * q_heads, (index + 1) * q_heads), range(index * kv_group_heads, (index + 1) * kv_group_heads), mean_squares)
        last = index == groups - 1
        if settings is None:
            qkv_list = [project(q_proj, group.q), project(k_proj, group.kv), project(v_proj, group.kv)]
            if last:
                linear_input = x = None
            norm_rope(qkv_list[0], qkv_list[1], group)
            attention = pay_attention(qkv_list, recycle_q=True)
            if groups == 1:
                return attention
            if output is None:
                output = attention.new_empty((*shape, heads, head_dim))
            output[..., group.q.start:group.q.stop, :].copy_(attention)
            del attention
            continue
        quantized = [None, None, sage2_core.staged_quantize_v([project(v_proj, group.kv)], settings)]
        if groups == 1:  # all heads: k is quantized and released before q is projected
            key = project(k_proj, group.kv)
            norm_rope(None, key, group)
            quantized[1] = sage2_core.staged_quantize_k([key], settings)
            del key
            query = project(q_proj, group.q)
            linear_input = x = None
            norm_rope(query, None, group)
            quantized[0] = sage2_core.staged_quantize_q(query, settings)
            return sage2_core.staged_attention(query, quantized, settings)
        query, key = project(q_proj, group.q), project(k_proj, group.kv)
        if last:
            linear_input = x = None
        norm_rope(query, key, group)
        quantized[1] = sage2_core.staged_quantize_k([key], settings)
        del key
        quantized[0] = sage2_core.staged_quantize_q(query, settings)
        if output is None:
            output = query.new_empty((*shape, heads, head_dim))
        del query
        sage2_core.staged_attention(output[..., group.q.start:group.q.stop, :], quantized, settings)
    return output
