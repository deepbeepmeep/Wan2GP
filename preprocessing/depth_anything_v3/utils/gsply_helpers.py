# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for WanGP: the PLY is written by shared.mesh3d.splats (numpy) instead of plyfile (GPL).
from pathlib import Path
from typing import Optional
import torch
from einops import rearrange
from torch import Tensor

from shared.mesh3d.splats import splats_to_ply
from ..specs import Gaussians


def export_ply(
    means: Tensor,  # "gaussian 3"
    scales: Tensor,  # "gaussian 3"
    rotations: Tensor,  # "gaussian 4"
    harmonics: Tensor,  # "gaussian 3 d_sh"
    opacities: Tensor,  # "gaussian", logits as stored in the PLY
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    match_3dgs_mcmc_dev: Optional[bool] = False,
):
    """INRIA 3DGS PLY (binary little-endian float32: x y z nx ny nz f_dc_0..2 f_rest_* opacity scale_0..2 rot_0..3, f_rest channel-major,
    plus a `comment sh_degree N` header line). match_3dgs_mcmc_dev writes degree 3 with zero higher coefficients (normals included)."""
    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Since current model use SH_degree = 4, which require large memory to store, we can only save the DC band to save memory.
    if match_3dgs_mcmc_dev:
        sh_rest = torch.zeros(len(means), 15, 3, dtype=torch.float32, device=means.device)
    else:
        sh_rest = None if save_sh_dc_only else harmonics[..., 1:].transpose(1, 2)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(splats_to_ply(means, scales, rotations, opacities.float().sigmoid(), harmonics[..., 0], sh_rest))


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def save_gaussian_ply(
    gaussians: Gaussians,
    save_path: str,
    ctx_depth: torch.Tensor,  # depth of input views; for getting shape and filtering, "v h w 1"
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    gs_views_interval: int = 1,
    inv_opacity: Optional[bool] = True,
    prune_by_depth_percent: Optional[float] = 1.0,
    prune_border_gs: Optional[bool] = True,
    match_3dgs_mcmc_dev: Optional[bool] = False,
):
    b = gaussians.means.shape[0]
    assert b == 1, "must set batch_size=1 when exporting 3D gaussians"
    src_v, out_h, out_w, _ = ctx_depth.shape

    # extract gs params
    world_means = gaussians.means
    world_shs = gaussians.harmonics
    world_rotations = gaussians.rotations
    gs_scales = gaussians.scales
    gs_opacities = inverse_sigmoid(gaussians.opacities) if inv_opacity else gaussians.opacities

    # Create a mask to filter the Gaussians.

    # TODO: prune the sky region here

    # throw away Gaussians at the borders, since they're generally of lower quality.
    if prune_border_gs:
        mask = torch.zeros_like(ctx_depth, dtype=torch.bool)
        gstrim_h = int(8 / 256 * out_h)
        gstrim_w = int(8 / 256 * out_w)
        mask[:, gstrim_h:-gstrim_h, gstrim_w:-gstrim_w, :] = 1
    else:
        mask = torch.ones_like(ctx_depth, dtype=torch.bool)

    # trim the far away point based on depth;
    if prune_by_depth_percent is not None and prune_by_depth_percent < 1:
        in_depths = ctx_depth
        d_percentile = torch.quantile(
            in_depths.view(in_depths.shape[0], -1), q=prune_by_depth_percent, dim=1
        ).view(-1, 1, 1)
        d_mask = (in_depths[..., 0] <= d_percentile).unsqueeze(-1)
        mask = mask & d_mask
    mask = mask.squeeze(-1)  # v h w

    # helper fn, must place after mask
    def trim_select_reshape(element):
        selected_element = rearrange(
            element[0], "(v h w) ... -> v h w ...", v=src_v, h=out_h, w=out_w
        )
        selected_element = selected_element[::gs_views_interval][mask[::gs_views_interval]]
        return selected_element

    export_ply(
        means=trim_select_reshape(world_means),
        scales=trim_select_reshape(gs_scales),
        rotations=trim_select_reshape(world_rotations),
        harmonics=trim_select_reshape(world_shs),
        opacities=trim_select_reshape(gs_opacities),
        path=Path(save_path),
        shift_and_scale=shift_and_scale,
        save_sh_dc_only=save_sh_dc_only,
        match_3dgs_mcmc_dev=match_3dgs_mcmc_dev,
    )
