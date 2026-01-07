"""Image warping utilities for NLF."""
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from . import ptu3d


def warp_images_with_pyramid(
    images: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
    new_invprojmats: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    crop_scales: torch.Tensor,
    output_shape: Tuple[int, int],
    image_ids: torch.Tensor,
    n_pyramid_levels: int = 3,
):
    image_levels = [images]
    for _ in range(1, n_pyramid_levels):
        image_levels.append(F.avg_pool2d(image_levels[-1], 2, 2))

    intrinsic_matrix_levels = [
        corner_aligned_scale_mat(1 / 2**i_level, device=intrinsic_matrix.device) @ intrinsic_matrix
        for i_level in range(n_pyramid_levels)
    ]

    i_pyramid_levels = torch.floor(-torch.log2(crop_scales))
    i_pyramid_levels = torch.clip(i_pyramid_levels, 0, n_pyramid_levels - 1).int()

    return torch.stack(
        [
            warp_single_image(
                image_levels[i_pyramid_levels[i]][image_ids[i]],
                intrinsic_matrix_levels[i_pyramid_levels[i]][i],
                new_invprojmats[i],
                distortion_coeffs[i],
                output_shape,
            )
            for i in range(len(image_ids))
        ]
    )


def warp_single_image(
    image: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
    new_invprojmat: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    output_shape: Tuple[int, int],
):
    device = image.device
    new_coords = torch.stack(
        torch.meshgrid(
            torch.arange(output_shape[1], device=device),
            torch.arange(output_shape[0], device=device),
            indexing='xy',
        ),
        dim=-1,
    ).float()
    new_coords_homog = ptu3d.to_homogeneous(new_coords)
    old_coords_homog = torch.einsum('hwc,Cc->hwC', new_coords_homog, new_invprojmat)
    old_coords_homog = ptu3d.to_homogeneous(
        distort_points(ptu3d.project(old_coords_homog), distortion_coeffs)
    )
    old_coords = torch.einsum('hwc,Cc->hwC', old_coords_homog, intrinsic_matrix)[..., :2]
    size = torch.tensor([image.shape[2], image.shape[1]], dtype=old_coords.dtype, device=device)
    old_coords_normalized = old_coords.mul_(2.0 / (size - 1)).sub_(1.0)
    return F.grid_sample(
        image.unsqueeze(0),
        old_coords_normalized.unsqueeze(0).to(image.dtype),
        align_corners=True,
        mode='bilinear',
        padding_mode='zeros',
    ).squeeze(0)


def distort_points(undist_points2d: torch.Tensor, distortion_coeffs: torch.Tensor):
    if torch.all(distortion_coeffs == 0):
        return undist_points2d
    elif distortion_coeffs.shape[-1] == 4:
        return distort_points_fisheye(undist_points2d, distortion_coeffs)
    else:
        a, b, c = distortion_formula_parts(undist_points2d, distortion_coeffs)
        return undist_points2d * (a + b) + c


def undistort_points(dist_points2d: torch.Tensor, distortion_coeffs: torch.Tensor):
    if torch.all(distortion_coeffs == 0):
        return dist_points2d
    elif distortion_coeffs.shape[-1] == 4:
        return undistort_points_fisheye(dist_points2d, distortion_coeffs)
    else:
        undist_points2d = dist_points2d
        for _ in range(5):
            a, b, c = distortion_formula_parts(undist_points2d, distortion_coeffs)
            undist_points2d = (dist_points2d - c - undist_points2d * b) / a
        return undist_points2d


def distortion_formula_parts(undist_points2d: torch.Tensor, distortion_coeffs: torch.Tensor):
    d = pad_axis_to_size(distortion_coeffs, 12, -1)
    broadcast_shape = (
        ([-1] if d.ndim > 1 else [])
        + [1] * (undist_points2d.ndim - d.ndim)
        + [12]
    )
    d = torch.reshape(d, broadcast_shape)

    r2 = torch.sum(torch.square(undist_points2d), dim=-1, keepdim=True)
    a = (((d[..., 4:5] * r2 + d[..., 1:2]) * r2 + d[..., 0:1]) * r2 + 1) / (
        ((d[..., 7:8] * r2 + d[..., 6:7]) * r2 + d[..., 5:6]) * r2 + 1
    )

    p2_1 = torch.flip(d[..., 2:4], dims=[-1])
    b = 2 * torch.sum(undist_points2d * p2_1, dim=-1, keepdim=True)
    c = (d[..., 9:12:2] * r2 + p2_1 + d[..., 8:11:2]) * r2

    return a, b, c


def pad_axis_to_size(x: torch.Tensor, size: int, dim: int):
    paddings = [[0, 0]] * x.ndim
    paddings[dim] = [0, size - x.shape[dim]]

    ps: List[int] = []
    for p_ in paddings[::-1]:
        ps.extend(p_)

    return F.pad(x, ps)


def distort_points_fisheye(
    undist_points2d: torch.Tensor, distortion_coeffs: torch.Tensor
) -> torch.Tensor:
    d = distortion_coeffs
    broadcast_shape = (
        ([-1] if d.ndim > 1 else [])
        + [1] * (undist_points2d.ndim - d.ndim)
        + [4]
    )
    d = torch.reshape(d, broadcast_shape)

    if torch.all(distortion_coeffs == 0):
        return undist_points2d
    else:
        r = torch.linalg.vector_norm(undist_points2d, dim=-1, keepdim=True)
        t = torch.atan(r)
        t2 = torch.square(t)
        t_d = (
            (((d[..., 3:4] * t2 + d[..., 2:3]) * t2 + d[..., 1:2]) * t2 + d[..., 0:1]) * t2 + 1
        ) * t
        return undist_points2d * torch.nan_to_num(t_d / r)


def undistort_points_fisheye(
    dist_points2d: torch.Tensor, distortion_coeffs: torch.Tensor
) -> torch.Tensor:
    d = distortion_coeffs
    broadcast_shape = (
        ([-1] if d.ndim > 1 else [])
        + [1] * (dist_points2d.ndim - d.ndim)
        + [4]
    )
    d = torch.reshape(d, broadcast_shape)

    if torch.all(distortion_coeffs == 0):
        return dist_points2d

    t_d = torch.linalg.vector_norm(dist_points2d, dim=-1, keepdim=True)
    t = t_d
    for _ in range(5):
        t2 = torch.square(t)
        t4 = torch.square(t2)
        t6 = t2 * t4
        t8 = torch.square(t4)
        k0_t2 = d[..., 0:1] * t2
        k1_t4 = d[..., 1:2] * t4
        k2_t6 = d[..., 2:3] * t6
        k3_t8 = d[..., 3:4] * t8
        t = t - (
            (t * (1 + k0_t2 + k1_t4 + k2_t6 + k3_t8) - t_d)
            / (1 + 3 * k0_t2 + 5 * k1_t4 + 7 * k2_t6 + 9 * k3_t8)
        )

    undist_points2d = dist_points2d * torch.nan_to_num(torch.tan(t) / t_d)
    return undist_points2d


def corner_aligned_scale_mat(factor: float, device: Optional[torch.device] = None) -> torch.Tensor:
    shift = (factor - 1.0) / 2.0
    return torch.tensor(
        [[factor, 0.0, shift], [0.0, factor, shift], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
