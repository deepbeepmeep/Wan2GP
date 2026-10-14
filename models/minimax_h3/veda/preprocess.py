# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fused tile packing, TripPool and compact attention-index construction."""
import torch
import triton
import triton.language as tl

from . import selection


@triton.jit
def _residual(Projected, Pool, Out, elements, D: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    projected = tl.load(Projected + offset, offset < elements, 0).to(tl.float32)
    mean = tl.load(Pool + offset // D * (3 * D) + offset % D, offset < elements, 0)
    tl.store(Out + offset, projected + mean, offset < elements)


def residual(projected, pooled):
    out = torch.empty_like(projected, dtype=torch.float32)
    _residual[(triton.cdiv(out.numel(), 512),)](projected, pooled, out, out.numel(), D=projected.shape[-1], B=512)
    return out


@triton.jit
def _gather_pool(X, Perm, Heads, Valid, Out, Pool, sxn, sxh, heads, video_tiles, D: tl.constexpr, C: tl.constexpr):
    tile, head, channel = tl.program_id(0), tl.program_id(1), tl.program_id(2) * C + tl.arange(0, C)
    row = tile * 128 + tl.arange(0, 128)
    source = tl.load(Perm + row)
    source_head = tl.load(Heads + head)
    x = tl.load(X + source[:, None] * sxn + source_head * sxh + channel[None, :], source[:, None] >= 0, 0)
    tl.store(Out + (row[:, None] * heads + head) * D + channel[None, :], x)
    if tile < video_tiles:
        valid = tl.load(Valid + tile)
        xf = x.to(tl.float32)
        mean = tl.sum(xf, 0) / tl.maximum(valid, 1)
        high = tl.max(tl.where(source[:, None] >= 0, xf, -float('inf')), 0)
        low = tl.min(tl.where(source[:, None] >= 0, xf, float('inf')), 0)
        base = Pool + (head * video_tiles + tile) * (3 * D) + channel
        tl.store(base, mean)
        tl.store(base + D, tl.where(valid > 0, high, 0.))
        tl.store(base + 2 * D, tl.where(valid > 0, low, 0.))


def gather_pool(x, layout, heads):
    count, dim = heads.numel(), x.shape[-1]
    out = x.new_empty((layout.num_slots, count, dim))
    pooled = torch.empty((count, layout.n_video_tiles, dim * 3), device=x.device, dtype=torch.float32)
    _gather_pool[(layout.n_tiles, count, dim // 32)](x, layout.perm, heads, layout.valid_count, out, pooled, x.stride(0), x.stride(1), count, layout.n_video_tiles, D=dim, C=32, num_warps=4, enable_fp_fusion=False)
    return out, pooled


@triton.jit
def _gather(X, Perm, Heads, Out, sxn, sxh, heads, elements, D: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    row, head, channel = offset // (heads * D), offset // D % heads, offset % D
    source = tl.load(Perm + row, offset < elements, -1)
    source_head = tl.load(Heads + head)
    x = tl.load(X + source * sxn + source_head * sxh + channel, (offset < elements) & (source >= 0), 0)
    tl.store(Out + offset, x, offset < elements)


def gather(x, layout, heads):
    out = x.new_empty((layout.num_slots, heads.numel(), x.shape[-1]))
    _gather[(triton.cdiv(out.numel(), 1024),)](x, layout.perm, heads, out, x.stride(0), x.stride(1), heads.numel(), out.numel(), D=x.shape[-1], B=1024)
    return out


@triton.jit
def _scatter(X, Perm, Heads, Out, out_heads, heads, elements, D: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    row, head, channel = offset // (heads * D), offset // D % heads, offset % D
    destination = tl.load(Perm + row, offset < elements, -1)
    destination_head = tl.load(Heads + head)
    x = tl.load(X + offset, offset < elements, 0)
    tl.store(Out + (destination * out_heads + destination_head) * D + channel, x, (offset < elements) & (destination >= 0))


def scatter(out, tiled, layout, heads):
    _scatter[(triton.cdiv(tiled.numel(), 1024),)](tiled, layout.perm, heads, out, out.shape[1], heads.numel(), tiled.numel(), D=out.shape[-1], B=1024)
    torch.autograd.graph.increment_version(out)


@triton.jit
def _mask_scores(Scores, Valid, elements, tiles, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    row, col = offset // tiles % tiles, offset % tiles
    valid = tl.load(Valid + col)
    score = tl.load(Scores + offset, offset < elements, 0)
    score = tl.where(row == col, float('inf'), tl.where(valid > 0, score, -float('inf')))
    tl.store(Scores + offset, score, offset < elements)


@triton.jit
def _indices(I0, V0, I1, V1, Valid, Index, Count, tiles, video_tiles, split,
             k0, k1, lo0, lo1, frac0: tl.constexpr, frac1: tl.constexpr, start0,
             ALL0: tl.constexpr, ALL1: tl.constexpr, SECOND: tl.constexpr,
             N: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, N)
    valid = tl.load(Valid + col, col < tiles, 0) > 0
    if row < video_tiles:
        selected = tl.full((N,), 0, tl.int32)
        if ALL0:
            selected = tl.where((col >= start0) & (col < split) & valid, 1, selected)
        else:
            rank = tl.arange(0, K0)
            index = tl.load(I0 + (head * video_tiles + row) * k0 + rank, rank < k0, 0).to(tl.int32) + start0
            value = tl.load(V0 + (head * video_tiles + row) * k0 + rank, rank < k0, -float('inf'))
            fraction = tl.full((), frac0, tl.float64)
            extra = tl.floor((row + 1).to(tl.float64) * fraction) > tl.floor(row.to(tl.float64) * fraction)
            keep = (rank < lo0 + extra) & (rank < k0) & (value > -float('inf'))
            # Invalid ranks use a spare lane; they never overwrite a valid selected column.
            selected = selected | (tl.histogram(tl.where(keep, index, N - 1), N) > 0).to(tl.int32)
        if SECOND:
            if ALL1:
                selected = tl.where((col >= split) & (col < video_tiles) & valid, 1, selected)
            else:
                rank = tl.arange(0, K1)
                index = tl.load(I1 + (head * video_tiles + row) * k1 + rank, rank < k1, 0).to(tl.int32) + split
                value = tl.load(V1 + (head * video_tiles + row) * k1 + rank, rank < k1, -float('inf'))
                fraction = tl.full((), frac1, tl.float64)
                extra = tl.floor((row + 1).to(tl.float64) * fraction) > tl.floor(row.to(tl.float64) * fraction)
                keep = (rank < lo1 + extra) & (rank < k1) & (value > -float('inf'))
                selected = selected | (tl.histogram(tl.where(keep, index, N - 1), N) > 0).to(tl.int32)
        selected = tl.where(col >= video_tiles, valid.to(tl.int32), selected)
    else:
        selected = valid.to(tl.int32)
    positions = tl.cumsum(selected) - 1
    tl.store(Index + (head * tiles + row) * tiles + positions, col, selected > 0)
    tl.store(Count + head * tiles + row, tl.sum(selected))


def indices(scores, layout, blocks):
    """Same torch.topk selection; compact sorted tile IDs without a dense mask or argsort."""
    _mask_scores[(triton.cdiv(scores.numel(), 512),)](scores, layout.valid_count, scores.numel(), layout.n_video_tiles, B=512)
    torch.autograd.graph.increment_version(scores)
    selected = []
    for block in blocks:
        lo, hi, fraction = selection.split_budget(block.budget.per_row(block.real_tokens, block.stop - block.start), block.stop - block.start)
        if block.budget.keeps_all:
            values = index = scores
        else:
            values, index = torch.topk(scores[:, :, block.start:block.stop], hi, dim=-1, sorted=True)
        selected.append((index, values, hi, lo, fraction, block.budget.keeps_all))
    first = selected[0]
    second = selected[-1]
    index = torch.empty((scores.shape[0], layout.n_tiles, layout.n_tiles), device=scores.device, dtype=torch.int32)
    count = torch.empty((scores.shape[0], layout.n_tiles), device=scores.device, dtype=torch.int32)
    _indices[(layout.n_tiles, scores.shape[0])](first[0], first[1], second[0], second[1], layout.valid_count, index, count,
        layout.n_tiles, layout.n_video_tiles, blocks[0].stop, first[2], second[2], first[3], second[3], first[4], second[4], blocks[0].start,
        ALL0=first[5], ALL1=second[5], SECOND=len(blocks) == 2, N=triton.next_power_of_2(layout.n_tiles + 1), K0=triton.next_power_of_2(first[2]), K1=triton.next_power_of_2(second[2]), enable_fp_fusion=False)
    return index, count
