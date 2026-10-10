"""VRAM savings shared by model attentions: a model calls qkv_attention() (joint_qkv_attention() for several token streams) instead of
projecting q/k/v and calling pay_attention.

- SageAttention 2 (sm86/sm89/sm120, and its masked Triton path on every architecture): v, k and q are quantized as soon as each is
  projected (shared.sage2_core staged functions), so at most one of them exists in 16 bits (with the Triton kernels v stays in fp16);
  same results as pay_attention.
- The three projections share one input prepared by mmgp (offload.prepare_linear_input) when their weight format supports it: the
  input is quantized once instead of once per projection, and the 16-bit input is released before the projections.
- Attention Head Split (Performance tab, off by default): long sequences (all batch items together) are attended one group of heads
  at a time into one output. q/k/v rows, per-head norms, RoPE and attention are per head; norms over all heads get per token
  statistics computed beforehand.
- Joint attentions: the projections of several streams (text and image tokens) are joined along the tokens in their order, and keys
  and values computed beforehand (a cached prefix) can be put before the projected ones.
"""
from collections import namedtuple

import torch
from mmgp import offload

from shared.attention import pay_attention, sage2_staged_settings, sage_attention_mask

HEAD_SPLIT_CHOICES = [("Off", 0), ("Low: saves some VRAM", 1), ("Medium (good balance): saves more VRAM", 2), ("High: saves the most VRAM", 3)]
_TARGET_GROUPS = (1, 4, 8, 16)
MIN_SPLIT_TOKENS = 8192  # fewer query tokens (all batch items together) are never split: little VRAM to save
_ROW_ELEMENTS = 1 << 23  # rows_ chunk size, in elements of the tensor
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
    (sm86/sm89/sm120), k is quantized and released before v is copied for its quantization, a peak lower by half of k; same results."""
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
    """Groups of heads of an attention over tokens (of all batch items): the divisor of kv_heads closest to the level's target, 1 when
    off."""
    level = head_split if level is None else level
    if level == 0 or tokens < MIN_SPLIT_TOKENS:
        return 1
    return min((groups for groups in range(1, kv_heads + 1) if kv_heads % groups == 0), key=lambda groups: (abs(groups - _TARGET_GROUPS[level]), groups))


def rows_(tensor, fn, start=0, stop=None):
    """tensor[:, first:last] = fn(tensor[:, first:last], first, last) by chunks of tokens of [start, stop), in place: fn computes each
    row on its own (norm, RoPE, possibly in place), so the result is that of fn on the whole range, with a bounded scratch."""
    stop = tensor.shape[1] if stop is None else stop
    step = max(1, _ROW_ELEMENTS // max(1, tensor[:, :1].numel()))
    for first in range(start, stop, step):
        last = min(first + step, stop)
        part = tensor[:, first:last]
        part.copy_(fn(part, first, last))
        del part


def _head_slice(mask, dim, heads):
    """mask restricted to the range heads along its head dimension dim, when it has one row per head."""
    return mask if mask is None or mask.dim() < 4 or mask.shape[dim] == 1 else mask.narrow(dim, heads.start, heads.stop - heads.start)


def qkv_attention(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, **kwargs):
    """Attention of the q/k/v projections of x (handed off in x_list, shape (batch, tokens, features) or (tokens, features)), returned
    as (batch, tokens, heads, head_dim), batch 1 for a 2D x. See joint_qkv_attention for norm_rope and the options."""
    return joint_qkv_attention([(x_list, (q_proj, k_proj, v_proj))], heads, head_dim, norm_rope, **kwargs)


@torch.compiler.disable()
def joint_qkv_attention(streams, heads, head_dim, norm_rope, kv_heads=None, norm_spans_heads=False, split_heads=True, kv_prefix=None,
                        attention_mask=None, force_attention=None):
    """Attention of the q/k/v projections of several streams joined along the tokens in their order: streams is a list of
    (x_list, (q_proj, k_proj, v_proj)), each x handed off in its list, of shape (batch, tokens, features) or (tokens, features).
    Returns (batch, all tokens, heads, head_dim).

    norm_rope(query, key, group) normalizes and rotates in place query and/or key, the other one may be None: the joined projections,
    (batch, all tokens, heads of the group, head_dim), for the head ranges group.q and group.kv. With norm_spans_heads, a split group
    gets group.mean_squares to scale its rows; otherwise the tensors hold all heads. split_heads=False keeps all heads together.
    kv_prefix: (key, value) of shape (batch, prefix tokens, kv_heads, head_dim), already normalized and rotated, put before the
    projected keys and values. attention_mask: pay_attention's mask. kv_heads < heads: grouped query attention."""
    global _unsplit_notice
    inputs, projections = [], []
    for x_list, stream_projections in streams:
        x = x_list[0]
        x_list.clear()
        inputs.append(x if x.dim() > 2 else x.unsqueeze(0))
        projections.append(stream_projections)
    streams = x = None
    batch, lengths, dtype = inputs[0].shape[0], [x.shape[1] for x in inputs], inputs[0].dtype
    tokens = sum(lengths)
    kv_heads = kv_heads or heads
    groups = head_groups(kv_heads, batch * tokens) if split_heads else 1
    settings = sage2_staged_settings(inputs[0].device, force_attention, attention_mask is not None)
    linear_inputs = [None] * len(inputs)
    for index, modules in enumerate(projections):
        if offload.linear_rows_supported(modules, inputs[index]):
            x_holder, inputs[index] = [inputs[index]], None
            linear_inputs[index] = offload.prepare_linear_input(x_holder, modules)
    # a stream whose projections cannot be computed by groups of heads is projected whole once and sliced, if it is too short to be split
    if groups > 1 and batch * sum(length for length, prepared in zip(lengths, linear_inputs) if prepared is None) >= MIN_SPLIT_TOKENS:
        if not _unsplit_notice:
            print("[Attention] Head Split is not applied: these q/k/v projections cannot be computed by groups of heads (weight format or LoRA type).")
            _unsplit_notice = True
        groups = 1
    whole = {}

    def project_stream(index, which, head_range):
        module, linear_input = projections[index][which], linear_inputs[index]
        if linear_input is not None:
            return offload.linear_rows(module, linear_input, head_range.start * head_dim, head_range.stop * head_dim).view(batch, lengths[index], -1, head_dim)
        if groups == 1:
            return module(inputs[index]).view(batch, lengths[index], -1, head_dim)
        if (index, which) not in whole:
            whole[index, which] = module(inputs[index]).view(batch, lengths[index], -1, head_dim)
        return whole[index, which][:, :, head_range.start:head_range.stop]

    def project(which, head_range):
        """Projection which (0 q, 1 k, 2 v) of the heads head_range, the streams joined along the tokens."""
        if len(lengths) == 1:
            return project_stream(0, which, head_range)
        joined, start = None, 0
        for index, length in enumerate(lengths):
            part = project_stream(index, which, head_range)
            if joined is None:
                joined = part.new_empty((batch, tokens, *part.shape[2:]))
            joined[:, start:start + length].copy_(part)
            start += length
            del part
        return joined

    def release():
        inputs.clear()
        linear_inputs.clear()
        whole.clear()

    def with_prefix(tensor, which, head_range):
        if kv_prefix is None:
            return tensor
        return torch.cat([kv_prefix[which - 1][:, :, head_range.start:head_range.stop], tensor], dim=1)

    mean_squares = None
    if norm_spans_heads and groups > 1:
        mean_squares = []
        for which, total in ((0, heads), (1, kv_heads)):
            step, squares = total // groups, None
            for group in range(groups):
                part = project(which, range(group * step, (group + 1) * step)).float().square_().sum(dim=(-2, -1))
                squares = part if squares is None else squares.add_(part)
                del part
            mean_squares.append(squares.div_(total * head_dim).unsqueeze(-1))
        mean_squares = tuple(mean_squares)

    if settings is not None:
        from shared import sage2_core
        if attention_mask is not None:  # in SageAttention's layout (batch, heads, q, k)
            attention_mask = sage_attention_mask(attention_mask, dtype)
    q_heads, kv_group_heads = heads // groups, kv_heads // groups
    output = None
    for index in range(groups):
        group = HeadGroup(range(index * q_heads, (index + 1) * q_heads), range(index * kv_group_heads, (index + 1) * kv_group_heads), mean_squares)
        last = index == groups - 1
        if settings is None:
            qkv_list = [project(0, group.q), project(1, group.kv), project(2, group.kv)]
            if last:
                release()
            norm_rope(qkv_list[0], qkv_list[1], group)
            qkv_list[1:] = [with_prefix(qkv_list[which], which, group.kv) for which in (1, 2)]
            if q_heads != kv_group_heads:  # pay_attention's backends do not all take fewer key/value heads
                qkv_list[1:] = [tensor.repeat_interleave(q_heads // kv_group_heads, dim=2) for tensor in qkv_list[1:]]
            attention = pay_attention(qkv_list, attention_mask=_head_slice(attention_mask, 2, group.q), force_attention=force_attention, recycle_q=True)
            if groups == 1:
                return attention
            if output is None:
                output = attention.new_empty((batch, tokens, heads, head_dim))
            output[..., group.q.start:group.q.stop, :].copy_(attention)
            del attention
            continue
        quantized = [None, None, sage2_core.staged_quantize_v([with_prefix(project(2, group.kv), 2, group.kv)], settings)]
        if groups == 1:  # all heads: k is quantized and released before q is projected
            key = project(1, group.kv)
            norm_rope(None, key, group)
            quantized[1] = sage2_core.staged_quantize_k([with_prefix(key, 1, group.kv)], settings)
            del key
            query = project(0, group.q)
            release()
            norm_rope(query, None, group)
            quantized[0] = sage2_core.staged_quantize_q(query, settings)
            return sage2_core.staged_attention(query, quantized, settings, attention_mask)
        query, key = project(0, group.q), project(1, group.kv)
        if last:
            release()
        norm_rope(query, key, group)
        quantized[1] = sage2_core.staged_quantize_k([with_prefix(key, 1, group.kv)], settings)
        del key
        quantized[0] = sage2_core.staged_quantize_q(query, settings)
        if output is None:
            output = query.new_empty((batch, tokens, heads, head_dim))
        sage2_core.staged_attention(output[..., group.q.start:group.q.stop, :], quantized, settings, _head_slice(attention_mask, 1, group.q))
        del query
    return output
