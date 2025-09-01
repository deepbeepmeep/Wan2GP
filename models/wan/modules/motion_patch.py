# Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union
import torch


# Refer to https://github.com/Angtian/VoGE/blob/main/VoGE/Utils.py
def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int = 1):
    """
    :param target: [... (can be k or 1), n > M, ...]
    :param ind: [... (k), M]
    :param dim: dim to apply index on
    :return: sel_target [... (k), M, ...]
    """
    assert (
        len(ind.shape) > dim
    ), "Index must have the target dim, but get dim: %d, ind shape: %s" % (dim, str(ind.shape))

    target = target.expand(
        *tuple(
            [ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)]
            + [
                -1,
            ]
            * (len(target.shape) - dim)
        )
    )

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1,) * (dim + 1), *target.shape[(dim + 1) : :])

    return torch.gather(target, dim=dim, index=ind_pad)


def merge_final(vert_attr: torch.Tensor, weight: torch.Tensor, vert_assign: torch.Tensor):
    """

    :param vert_attr: [n, d] or [b, n, d] color or feature of each vertex
    :param weight: [b(optional), w, h, M] weight of selected vertices
    :param vert_assign: [b(optional), w, h, M] selective index
    :return:
    """
    target_dim = len(vert_assign.shape) - 1
    if len(vert_attr.shape) == 2:
        assert vert_attr.shape[0] > vert_assign.max()
        # [n, d] ind: [b(optional), w, h, M]-> [b(optional), w, h, M, d]
        sel_attr = ind_sel(
            vert_attr[(None,) * target_dim], vert_assign.type(torch.long), dim=target_dim
        )
    else:
        assert vert_attr.shape[1] > vert_assign.max()
        sel_attr = ind_sel(
            vert_attr[(slice(None),) + (None,)*(target_dim-1)], vert_assign.type(torch.long), dim=target_dim
        )

    # [b(optional), w, h, M]
    final_attr = torch.sum(sel_attr * weight.unsqueeze(-1), dim=-2)
    return final_attr


def patch_motion(
    tracks: torch.FloatTensor,  # (B, T, N, 4)
    vid: torch.FloatTensor,  # (C, T, H, W)
    temperature: float = 220.0,
    training: bool = True,
    tail_dropout: float = 0.2,
    vae_divide: tuple = (4, 16),
    topk: int = 2,
):
    with torch.no_grad():
        _, T, H, W = vid.shape
        N = tracks.shape[2]
        _, tracks, visible = torch.split(
            tracks, [1, 2, 1], dim=-1
        )  # (B, T, N, 2) | (B, T, N, 1)
        tracks_n = tracks / torch.tensor([W / min(H, W), H / min(H, W)], device=tracks.device)
        tracks_n = tracks_n.clamp(-1, 1)
        visible = visible.clamp(0, 1)

        if tail_dropout > 0 and training:
            TT = visible.shape[1]
            rrange = torch.arange(TT, device=visible.device, dtype=visible.dtype)[
                None, :, None, None
            ]
            rand_nn = torch.rand_like(visible[:, :1])
            rand_rr = torch.rand_like(visible[:, :1]) * (TT - 1)
            visible = visible * (
                (rand_nn > tail_dropout).type_as(visible)
                + (rrange < rand_rr).type_as(visible)
            ).clamp(0, 1)

        xx = torch.linspace(-W / min(H, W), W / min(H, W), W)
        yy = torch.linspace(-H / min(H, W), H / min(H, W), H)

        grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1).to(
            tracks.device
        )

        tracks_pad = tracks[:, 1:]
        visible_pad = visible[:, 1:]

        visible_align = visible_pad.view(T - 1, 4, *visible_pad.shape[2:]).sum(1)
        tracks_align = (tracks_pad * visible_pad).view(T - 1, 4, *tracks_pad.shape[2:]).sum(
            1
        ) / (visible_align + 1e-5)
        dist_ = (
            (tracks_align[:, None, None] - grid[None, :, :, None]).pow(2).sum(-1)
        )  # T, H, W, N
        weight = torch.exp(-dist_ * temperature) * visible_align.clamp(0, 1).view(
            T - 1, 1, 1, N
        )
        vert_weight, vert_index = torch.topk(
            weight, k=min(topk, weight.shape[-1]), dim=-1
        )

    grid_mode = "bilinear"
    point_feature = torch.nn.functional.grid_sample(
        vid[vae_divide[0]:].permute(1, 0, 2, 3)[:1],
        tracks_n[:, :1].type(vid.dtype),
        mode=grid_mode,
        padding_mode="zeros",
        align_corners=None,
    )
    point_feature = point_feature.squeeze(0).squeeze(1).permute(1, 0) # N, C=16

    out_feature = merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2) # T - 1, H, W, C => C, T - 1, H, W
    out_weight = vert_weight.sum(-1) # T - 1, H, W

    # out feature -> already soft weighted
    mix_feature = out_feature + vid[vae_divide[0]:, 1:] * (1 - out_weight.clamp(0, 1))

    out_feature_full = torch.cat([vid[vae_divide[0]:, :1], mix_feature], dim=1) # C, T, H, W
    out_mask_full = torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0)  # T, H, W
    return torch.cat([out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full], dim=0)
