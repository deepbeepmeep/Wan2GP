"""PyTorch 3D utilities for NLF."""
from typing import List, Optional

import numpy as np
import torch

from . import ptu


def reconstruct_absolute(
    coords2d: torch.Tensor,
    coords3d_rel: torch.Tensor,
    intrinsics: torch.Tensor,
    proc_side: int,
    stride: int,
    centered_stride: bool,
    mix_3d_inside_fov: Optional[float] = None,
    weak_perspective: bool = None,
    point_validity_mask: Optional[torch.Tensor] = None,
    border_factor1: float = 0.75,
    border_factor2: Optional[float] = None,
    mix_based_on_3d: bool = True,
):
    inv_intrinsics = torch.linalg.inv(intrinsics.to(coords2d.dtype))
    coords2d_normalized = (to_homogeneous(coords2d) @ inv_intrinsics.transpose(1, 2))[..., :2]

    if border_factor2 is None:
        border_factor2 = border_factor1

    is_predicted_to_be_in_fov1 = is_within_fov(
        coords2d, proc_side, stride, centered_stride, border_factor1
    )

    if point_validity_mask is not None:
        is_predicted_to_be_in_fov1 = torch.logical_and(
            is_predicted_to_be_in_fov1, point_validity_mask
        )

    if weak_perspective:
        ref = reconstruct_ref_weakpersp(
            coords2d_normalized, coords3d_rel, is_predicted_to_be_in_fov1
        )
    else:
        ref = reconstruct_ref_fullpersp(
            coords2d_normalized, coords3d_rel, is_predicted_to_be_in_fov1
        )
    coords_abs_3d_based = coords3d_rel + ref.unsqueeze(1)
    reference_depth = ref[:, 2]
    relative_depths = coords3d_rel[..., 2]
    coords_abs_2d_based = back_project(coords2d_normalized, relative_depths, reference_depth)

    if mix_3d_inside_fov is not None:
        coords_abs_2d_based = (
            mix_3d_inside_fov * coords_abs_3d_based + (1 - mix_3d_inside_fov) * coords_abs_2d_based
        )

    if mix_based_on_3d:
        coords_proj_for_mix_decision = project_pose(coords_abs_3d_based, intrinsics)
    else:
        coords_proj_for_mix_decision = project_pose(coords_abs_2d_based, intrinsics)

    is_predicted_to_be_in_fov2 = torch.logical_and(
        is_within_fov(
            coords_proj_for_mix_decision, proc_side, stride, centered_stride, border_factor2
        ),
        coords_abs_3d_based[..., 2] > 0.001,
    )

    return torch.where(
        is_predicted_to_be_in_fov2[..., np.newaxis], coords_abs_2d_based, coords_abs_3d_based
    )


def reconstruct_ref_weakpersp(normalized_2d, coords3d_rel, validity_mask):
    mean3d, stdev3d = ptu.mean_stdev_masked(
        coords3d_rel[..., :2], validity_mask, items_dim=1, dimensions_dim=2
    )

    mean2d, stdev2d = ptu.mean_stdev_masked(
        normalized_2d[..., :2], validity_mask, items_dim=1, dimensions_dim=2
    )

    stdev2d = torch.maximum(stdev2d, torch.tensor(1e-5))
    stdev3d = torch.maximum(stdev3d, torch.tensor(1e-5))

    old_mean = ptu.reduce_mean_masked(coords3d_rel, validity_mask, dim=[1], keepdim=True)
    new_mean_z = torch.nan_to_num(stdev3d / stdev2d)
    new_mean = to_homogeneous(mean2d) * new_mean_z
    return torch.squeeze(new_mean - old_mean, 1)


def to_homogeneous(x):
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def reconstruct_ref_fullpersp(normalized_2d, coords3d_rel, validity_mask):
    n_batch = normalized_2d.shape[0]
    n_points = normalized_2d.shape[1]

    eyes2 = (
        torch.eye(2, device=normalized_2d.device, dtype=normalized_2d.dtype)
        .unsqueeze(0)
        .repeat(n_batch, n_points, 1)
    )
    scale2d, reshaped2d = rms_normalize_and_reshape(normalized_2d, validity_mask, n_points)
    A = torch.cat([eyes2, -reshaped2d], dim=2)

    rel_backproj = normalized_2d * coords3d_rel[:, :, 2:] - coords3d_rel[:, :, :2]
    scale_rel_backproj, b = rms_normalize_and_reshape(rel_backproj, validity_mask, n_points)

    weights = validity_mask.to(normalized_2d.dtype) + 1e-8
    weights = torch.repeat_interleave(weights, 2, 1)
    ref = lstsq_cholesky(A, b, weights, l2_regularizer=1e-4)

    ref = torch.cat(
        [ref[:, :2] * scale_rel_backproj, ref[:, 2:] * (scale_rel_backproj / scale2d)], dim=1
    )
    return torch.squeeze(ref, dim=-1)


def rms_normalize_and_reshape(x: torch.Tensor, validity_mask: torch.Tensor, n_points: int):
    scale = torch.sqrt(
        ptu.reduce_mean_masked(torch.square(x), validity_mask, dim=[1, 2], keepdim=True)
        + 1e-10
    )
    normalized = x / scale
    reshaped = normalized.reshape(-1, n_points * 2, 1)
    return scale, reshaped


def lstsq_cholesky(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    weights: torch.Tensor,
    l2_regularizer: Optional[float] = None,
) -> torch.Tensor:
    weighted_matrix = weights.unsqueeze(-1) * matrix
    regularized_gramian = weighted_matrix.mT @ matrix
    if l2_regularizer is not None:
        regularized_gramian.diagonal(dim1=-2, dim2=-1).add_(l2_regularizer)

    ATb = weighted_matrix.mT @ rhs
    chol, _ = torch.linalg.cholesky_ex(regularized_gramian)
    return torch.cholesky_solve(ATb, chol)


def back_project(camcoords2d, delta_z, z_offset):
    return to_homogeneous(camcoords2d) * torch.unsqueeze(
        delta_z + torch.unsqueeze(z_offset, -1), -1
    )


def is_within_fov(
    imcoords: torch.Tensor,
    proc_side: int,
    stride: int,
    centered_stride: bool,
    border_factor: float = 0.75,
):
    offset = -stride / 2.0 if not centered_stride else 0.0
    lower = stride * border_factor + offset
    upper = proc_side - stride * border_factor + offset
    proj_in_fov = torch.all(torch.logical_and(imcoords >= lower, imcoords <= upper), dim=-1)
    return proj_in_fov


def project_pose(coords3d, intrinsic_matrix):
    projected = coords3d / torch.maximum(
        torch.tensor(0.1, device=coords3d.device), coords3d[..., 2:]
    )
    return torch.einsum('bnk,bjk->bnj', projected, intrinsic_matrix[..., :2, :])


def lookat_matrix(forward_vector, up_vector):
    new_z = forward_vector / torch.linalg.norm(forward_vector, dim=-1, keepdim=True)
    new_x = torch.linalg.cross(new_z, up_vector)
    new_x_alt = torch.stack([new_z[:, 2], torch.zeros_like(new_z[:, 2]), -new_z[:, 0]], dim=1)
    new_x = torch.where(torch.linalg.norm(new_x, dim=-1, keepdim=True) == 0, new_x_alt, new_x)
    new_x = new_x / torch.linalg.norm(new_x, dim=-1, keepdim=True)
    new_y = torch.linalg.cross(new_z, new_x)
    return torch.stack([new_x, new_y, new_z], dim=1)


def project(points):
    return points[..., :2] / points[..., 2:3]


def intrinsic_matrix_from_field_of_view(
    fov_degrees: float, imshape: List[int], device: Optional[torch.device] = None
):
    imshape = torch.tensor(imshape, dtype=torch.float32, device=device)
    fov_radians = fov_degrees * torch.tensor(torch.pi / 180, dtype=torch.float32, device=device)
    larger_side = torch.max(imshape)
    focal_length = larger_side / (torch.tan(fov_radians / 2) * 2)
    _0 = torch.tensor(0, dtype=torch.float32, device=device)
    _1 = torch.tensor(1, dtype=torch.float32, device=device)

    return (
        torch.stack(
            [
                focal_length,
                _0,
                (imshape[1] - 1) / 2,
                _0,
                focal_length,
                (imshape[0] - 1) / 2,
                _0,
                _0,
                _1,
            ],
            dim=-1,
        )
        .unflatten(-1, (3, 3))
        .unsqueeze(0)
    )


def rotation_mat(angle: torch.Tensor, rot_axis: str):
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    _0 = torch.zeros_like(angle)
    _1 = torch.ones_like(angle)

    if rot_axis == 'x':
        elems = [_1, _0, _0, _0, cos, sin, _0, -sin, cos]
    elif rot_axis == 'y':
        elems = [cos, _0, -sin, _0, _1, _0, sin, _0, cos]
    else:
        elems = [cos, -sin, _0, sin, cos, _0, _0, _0, _1]

    return torch.stack(elems, dim=-1).unflatten(-1, (3, 3))
