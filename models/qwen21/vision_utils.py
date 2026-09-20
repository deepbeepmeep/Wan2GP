# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
# Adapted from transformers v5.17.0, modeling_qwen3_vl.py for inference.
from __future__ import annotations
import torch
import torch.nn.functional as F
def get_vision_cu_seqlens(grid_thw: torch.Tensor, merge_temporal: bool=False, kwargs: dict | None=None) -> torch.Tensor:
    if kwargs is not None and (cu_seqlens := kwargs.pop('cu_seqlens', None)) is not None:
        return cu_seqlens
    dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    if merge_temporal:
        seqlens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    else:
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=dtype), (1, 0), value=0)

def get_vision_attention_seqlens(grid_thw: torch.Tensor, config: PreTrainedConfig, merge_temporal: bool=False, kwargs: dict | None=None) -> tuple[torch.Tensor, int | None]:
    cu_seqlens = get_vision_cu_seqlens(grid_thw, merge_temporal=merge_temporal, kwargs=kwargs)
    max_seqlen = None
    return (cu_seqlens, max_seqlen)

def get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int | torch.Tensor, include_temporal: bool=False, kwargs: dict | None=None) -> torch.Tensor:
    if kwargs is not None and (position_ids := kwargs.pop('position_ids', None)) is not None:
        return position_ids
    device = grid_thw.device
    if isinstance(spatial_merge_size, int):
        spatial_merge_size = torch.tensor([spatial_merge_size], device=device).expand(len(grid_thw))
    position_ids = []
    for (t, h, w), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
        hpos_ids, wpos_ids = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        if include_temporal:
            tpos_ids = torch.arange(t, device=device).repeat_interleave(h * w)
            position_ids.append(torch.stack([tpos_ids, hpos_ids.repeat(t), wpos_ids.repeat(t)], dim=-1))
        else:
            position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)

def _interpolation_axis_taps_weights(index: torch.Tensor, size: torch.Tensor, side: int, mode: str, align_corners: bool, padding: str='border') -> tuple[torch.Tensor, torch.Tensor]:
    index = index.to(torch.float32)
    if align_corners:
        src = index * (side - 1) / torch.clamp(size - 1, min=1)
    else:
        src = (index + 0.5) * side / size - 0.5
    floor = torch.floor(src)
    if mode == 'bilinear':
        offsets = torch.arange(0, 2, device=index.device)
    elif mode == 'bicubic':
        offsets = torch.arange(-1, 3, device=index.device)
    else:
        raise ValueError(f"Unsupported interpolation mode {mode!r} (expected 'bilinear' or 'bicubic').")
    raw_taps = floor.long()[:, None] + offsets
    taps = raw_taps.clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    if mode == 'bilinear':
        weights = (1 - distance).clamp(min=0)
    else:
        a = -0.75
        near = ((a + 2) * distance - (a + 3)) * distance * distance + 1
        far = ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a
        weights = torch.where(distance <= 1, near, far)
    if padding == 'zeros':
        weights = weights * ((raw_taps >= 0) & (raw_taps <= side - 1))
    return (taps, weights)

def get_vision_interpolation_indices_and_weights(grid_thw: torch.Tensor, num_grid_per_side: int, mode: str='bilinear', align_corners: bool=False, spatial_merge_size: int=1, padding: str='border', kwargs: dict | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    if kwargs is not None:
        interp_indices = kwargs.pop('interp_indices', None)
        interp_weights = kwargs.pop('interp_weights', None)
        if interp_indices is not None and interp_weights is not None:
            return (interp_indices, interp_weights)
    side = num_grid_per_side
    merge = spatial_merge_size
    device = grid_thw.device
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within = (torch.arange(counts.sum(), device=device) - starts) % (heights * widths)
    blocks_w = widths // merge
    in_col = within % merge
    in_row = within // merge % merge
    block_col = within // (merge * merge) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col
    h_taps, h_weights = _interpolation_axis_taps_weights(row, heights, side, mode, align_corners, padding)
    w_taps, w_weights = _interpolation_axis_taps_weights(col, widths, side, mode, align_corners, padding)
    n = h_taps.shape[1]
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, n * n)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, n * n)
    return (indices, weights)
