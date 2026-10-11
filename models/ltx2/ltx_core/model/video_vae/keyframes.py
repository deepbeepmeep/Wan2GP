# SPDX-FileCopyrightText: Copyright (c) 2025 Lightricks. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keyframe-aware NAD decoding: plane geometry and joint (video + keyframe planes) neighborhood attention.

A keyframe decode carries a second stream of single-pixel-frame planes through every decoder stage. Both streams
share all weights and only meet inside one joint attention softmax:

* a video query at ``(t, h, w)`` sees its ``Kt x Kh x Kw`` video window, centered and masked at the volume edges
  (no NATTEN inward shift), plus the ``Kh x Kw`` window at the same ``(h, w)`` on its two nearest planes;
* a plane query sees the ``Kh x Kw`` window on its own plane plus the same window on its two nearest video frames.

Plane times are expressed in each stage's temporal units with the same origin as the video RoPE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

KEYFRAME_CONTEXT_SLOTS = 2
_JOINT_SCORE_BUDGET = 2**25
_JOINT_KV_STACK_BUDGET = 2**28


@dataclass(frozen=True)
class DecodeKeyframes:
    """Generated keyframe latents ``(B, C, P, H, W)``, each a standalone one-pixel-frame latent, and their global pixel frames."""

    latents: torch.Tensor
    pixel_frame_indices: tuple[int, ...]


def keyframe_stage_times(pixel_frame_indices: tuple[int, ...], remaining_time_stride: int, origin: float, device: torch.device) -> torch.Tensor:
    """Center of the stage cell holding each plane: ``t_s(0) = 0`` and ``t_s(f) = (f + (r - 1) / 2) / r``, minus ``origin``."""
    frames = torch.tensor(pixel_frame_indices, dtype=torch.float32, device=device)
    times = (frames + (remaining_time_stride - 1) / 2) / remaining_time_stride
    return torch.where(frames == 0, torch.zeros_like(times), times) - origin


def planes_for_tile(pixel_frame_indices: tuple[int, ...], frame_lo: int, frame_hi: int) -> list[int]:
    """Planes inside pixel frames ``[lo, hi]`` plus the nearest plane on each side, so tile edges rank the same planes as a whole decode."""
    keep = [index for index, frame in enumerate(pixel_frame_indices) if frame_lo <= frame <= frame_hi]
    before = [(frame, index) for index, frame in enumerate(pixel_frame_indices) if frame < frame_lo]
    after = [(frame, index) for index, frame in enumerate(pixel_frame_indices) if frame > frame_hi]
    if before:
        keep.append(max(before)[1])
    if after:
        keep.append(min(after)[1])
    return sorted(keep)


def _nearest_slots(query_times: torch.Tensor, candidate_times: torch.Tensor) -> torch.Tensor:
    """``(Q, slots)`` candidate indices ranked by ``(|dt|, index)``, ``-1`` when there are fewer candidates than slots."""
    chosen = torch.argsort((query_times[:, None] - candidate_times[None, :]).abs(), dim=-1, stable=True)[:, :KEYFRAME_CONTEXT_SLOTS]
    if chosen.shape[1] < KEYFRAME_CONTEXT_SLOTS:
        chosen = torch.cat([chosen, chosen.new_full((chosen.shape[0], KEYFRAME_CONTEXT_SLOTS - chosen.shape[1]), -1)], dim=1)
    return chosen


def video_keyframe_slots(keyframe_times: torch.Tensor, video_length: int) -> torch.Tensor:
    """``(T, slots)`` nearest planes per video frame, independent of the temporal kernel."""
    return _nearest_slots(torch.arange(video_length, dtype=torch.float32, device=keyframe_times.device), keyframe_times)


def keyframe_video_slots(keyframe_times: torch.Tensor, video_length: int) -> torch.Tensor:
    """``(P, slots)`` nearest video frames per plane."""
    return _nearest_slots(keyframe_times, torch.arange(video_length, dtype=torch.float32, device=keyframe_times.device))


def _centered_bounds(length: int, kernel: int, start: int, stop: int) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    """Key range of the clamp-and-mask windows of queries ``[start, stop)`` and each window relative to that range."""
    half = kernel // 2
    lo = [max(index - half, 0) for index in range(start, stop)]
    hi = [min(index - half + kernel, length) for index in range(start, stop)]
    return lo[0], hi[-1], tuple(value - lo[0] for value in lo), tuple(value - lo[0] for value in hi)


def _axis_tiles(length: int, kernel: int, tile: int) -> list[tuple[slice, slice, tuple]]:
    """``(query slice, key slice, relative window bounds)`` per tile along one axis."""
    tiles = []
    for start in range(0, length, tile):
        stop = min(start + tile, length)
        key_start, key_stop, rel_lo, rel_hi = _centered_bounds(length, kernel, start, stop)
        tiles.append((slice(start, stop), slice(key_start, key_stop), (rel_lo, rel_hi)))
    return tiles


def _visible(bounds: tuple, device: torch.device) -> torch.Tensor:
    starts, ends = (torch.tensor(values, device=device) for values in bounds)
    key = torch.arange(int(ends[-1]), device=device)
    return (key[None] >= starts[:, None]) & (key[None] < ends[:, None])


def _joint_mask(rel_t: tuple | None, rel_h: tuple, rel_w: tuple, planes: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``(1, 1, Nq, Nk)`` additive mask: the clamped video window (if any), then ``planes`` spatial windows."""
    spatial = (_visible(rel_h, device)[:, None, :, None] & _visible(rel_w, device)[None, :, None, :]).flatten(2).flatten(0, 1)
    queries = 1 if rel_t is None else len(rel_t[0])
    parts = [] if rel_t is None else [(_visible(rel_t, device)[:, None, :, None] & spatial[None, :, None, :]).flatten(2).flatten(0, 1)]
    parts.append(spatial[None, :, None, :].expand(queries, -1, planes, -1).flatten(2).flatten(0, 1))
    visible = torch.cat(parts, dim=1)
    return torch.zeros(visible.shape, dtype=dtype, device=device).masked_fill_(~visible, torch.finfo(dtype).min)[None, None]


def _tokens(x: torch.Tensor) -> torch.Tensor:
    """``(B, A, H, W, NH, HD)`` -> ``(B, NH, A * H * W, HD)``."""
    return x.permute(0, 4, 1, 2, 3, 5).flatten(2, 4)


def _pick_tiles(dims: tuple[int, ...], kernels: tuple[int, ...], other_planes: int) -> list[int]:
    """Query tile sizes keeping ``queries x keys`` within the score budget (the leading axis is T when ``len(dims) == 3``)."""
    tiles = list(dims)

    def cost() -> int:
        spatial_keys = math.prod(min(dim, tile + kernel - 1) for tile, kernel, dim in zip(tiles[-2:], kernels[-2:], dims[-2:]))
        own_keys = min(dims[0], tiles[0] + kernels[0] - 1) if len(dims) == 3 else 1
        return math.prod(tiles) * spatial_keys * (own_keys + other_planes)

    while cost() > _JOINT_SCORE_BUDGET and max(tiles) > 1:
        axis = max(range(len(tiles)), key=lambda index: tiles[index] / kernels[index])
        tiles[axis] = max(1, (tiles[axis] + 1) // 2)
    return tiles


def _attend_groups(groups: dict, out: torch.Tensor, gather) -> None:
    """One SDPA per chunk of query tiles sharing a mask geometry; ``gather(tile)`` returns ``(query slice, q, k, v)``."""
    batch, heads, head_dim = out.shape[0], out.shape[4], out.shape[5]
    for key, tiles in groups.items():
        mask = _joint_mask(*key, out.dtype, out.device)
        group_max = max(1, _JOINT_KV_STACK_BUDGET // max(1, batch * heads * mask.shape[-1] * head_dim * 2)) if out.is_cuda else 1
        for start in range(0, len(tiles), group_max):
            gathered = [gather(tile) for tile in tiles[start:start + group_max]]
            q, k, v = (torch.cat([item[index] for item in gathered]) for index in (1, 2, 3))
            result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
            q = k = v = None
            for index, (query_slices, *_) in enumerate(gathered):
                region = out[query_slices]
                region.copy_(result[index * batch:(index + 1) * batch].unflatten(2, region.shape[1:4]).permute(0, 2, 3, 4, 1, 5))
            gathered = result = None


def joint_na3d_eager(qkv_list: list[torch.Tensor], keyframe_times: torch.Tensor, kernel_size: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded-workspace SDPA joint neighborhood attention; inputs are ``(B, A, H, W, NH, HD)`` with Q pre-scaled."""
    q, k, v, keyframe_q, keyframe_k, keyframe_v = qkv_list
    qkv_list.clear()
    time, height, width = q.shape[1:4]
    kernel_t, kernel_h, kernel_w = kernel_size
    video_slots = video_keyframe_slots(keyframe_times, time).tolist()
    keyframe_slots = keyframe_video_slots(keyframe_times, time).tolist()

    runs, run_start = [], 0
    for index in range(1, time + 1):
        if index == time or video_slots[index] != video_slots[run_start]:
            runs.append((run_start, index, [plane for plane in video_slots[run_start] if plane >= 0]))
            run_start = index
    tile_t, tile_h, tile_w = _pick_tiles((time, height, width), kernel_size, KEYFRAME_CONTEXT_SLOTS)
    h_tiles, w_tiles = _axis_tiles(height, kernel_h, tile_h), _axis_tiles(width, kernel_w, tile_w)
    groups: dict = {}
    for run_start, run_stop, planes in runs:
        for t0 in range(run_start, run_stop, tile_t):
            t1 = min(t0 + tile_t, run_stop)
            key_t0, key_t1, *rel_t = _centered_bounds(time, kernel_t, t0, t1)
            for query_h, key_h, rel_h in h_tiles:
                for query_w, key_w, rel_w in w_tiles:
                    tile = ((slice(None), slice(t0, t1), query_h, query_w), (slice(key_t0, key_t1), key_h, key_w), planes)
                    groups.setdefault((tuple(rel_t), rel_h, rel_w, len(planes)), []).append(tile)

    def gather_video(tile):
        query, (key_t, key_h, key_w), planes = tile
        index = torch.tensor(planes, device=q.device)
        keys = torch.cat([_tokens(k[:, key_t, key_h, key_w]), _tokens(keyframe_k[:, index, key_h, key_w])], dim=2)
        values = torch.cat([_tokens(v[:, key_t, key_h, key_w]), _tokens(keyframe_v[:, index, key_h, key_w])], dim=2)
        return query, _tokens(q[query]), keys, values

    video_out = q
    _attend_groups(groups, video_out, gather_video)

    tile_h, tile_w = _pick_tiles((height, width), (kernel_h, kernel_w), KEYFRAME_CONTEXT_SLOTS)
    h_tiles, w_tiles = _axis_tiles(height, kernel_h, tile_h), _axis_tiles(width, kernel_w, tile_w)
    groups = {}
    for plane, frames in enumerate(keyframe_slots):
        frames = [frame for frame in frames if frame >= 0]
        for query_h, key_h, rel_h in h_tiles:
            for query_w, key_w, rel_w in w_tiles:
                groups.setdefault((None, rel_h, rel_w, 1 + len(frames)), []).append(((slice(None), slice(plane, plane + 1), query_h, query_w), (key_h, key_w), plane, frames))

    def gather_keyframe(tile):
        query, (key_h, key_w), plane, frames = tile
        index = torch.tensor(frames, device=q.device)
        keys = torch.cat([_tokens(keyframe_k[:, plane:plane + 1, key_h, key_w]), _tokens(k[:, index, key_h, key_w])], dim=2)
        values = torch.cat([_tokens(keyframe_v[:, plane:plane + 1, key_h, key_w]), _tokens(v[:, index, key_h, key_w])], dim=2)
        return query, _tokens(keyframe_q[query]), keys, values

    keyframe_out = keyframe_q
    _attend_groups(groups, keyframe_out, gather_keyframe)
    return video_out, keyframe_out
