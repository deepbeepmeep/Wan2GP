# WorldMirror 2.0 reconstruction post-processing, ported from Tencent HY-World 2.0 (hyworld2/worldrecon/hyworldmirror: models/utils/
# {camera_utils,rotation,geometry,sh_utils,act_gs}.py, utils/geometry.py (depth_edge, normals_edge; references DUSt3R and MoGe),
# utils/inference_utils.py (filter masks, voxel merging, point compression), utils/save_utils.py (scale filter)). Copyright (C) 2026
# Tencent, licensed under the Tencent HY-WORLD 2.0 Community License Agreement (LICENSE in this folder). Modified by DeepBeepMeep for
# WanGP (2026-10-08): PyTorch on the GPU instead of NumPy / CPU loops, the sky mask from the model only (no skyseg ONNX download),
# FP64 accumulation in the voxel merges, merged opacities written as linear values (the PLY writer stores their logit).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import numpy as np
import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814
OPENCV_TO_GLTF = (1.0, -1.0, -1.0)  # a 180 degree turn about X: +Y up, +Z towards the viewer


def quat_to_rotmat(quaternions):
    """Rotation matrices (..., 3, 3) of quaternions (..., 4) with the real part last (x y z w), as PyTorch3D."""
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    matrix = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                          two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                          two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def camera_matrices(codes, height, width):
    """Camera codes (views, 9: translation, quaternion x y z w, vertical / horizontal field of view) -> OpenCV camera-to-world (views, 4,
    4) and pixel intrinsics (views, 3, 3), principal point at the image centre. The code holds the world-to-camera transform."""
    world_to_camera = torch.zeros(len(codes), 4, 4, dtype=codes.dtype, device=codes.device)
    world_to_camera[:, :3, :3] = quat_to_rotmat(codes[:, 3:7])
    world_to_camera[:, :3, 3] = codes[:, :3]
    world_to_camera[:, 3, 3] = 1
    intrinsics = torch.zeros(len(codes), 3, 3, dtype=codes.dtype, device=codes.device)
    intrinsics[:, 0, 0] = width * 0.5 / torch.tan(codes[:, 8] * 0.5)
    intrinsics[:, 1, 1] = height * 0.5 / torch.tan(codes[:, 7] * 0.5)
    intrinsics[:, 0, 2], intrinsics[:, 1, 2], intrinsics[:, 2, 2] = width * 0.5, height * 0.5, 1.0
    return torch.linalg.inv(world_to_camera), intrinsics


def unproject(depth, camera_to_world, intrinsics):
    """depth_to_world_coords_points: world points (views, H, W, 3) of depth maps (views, H, W), pixel centres at integer coordinates."""
    views, height, width = depth.shape
    v, u = torch.meshgrid(torch.arange(height, dtype=depth.dtype, device=depth.device), torch.arange(width, dtype=depth.dtype, device=depth.device), indexing="ij")
    fx, fy, cx, cy = (intrinsics[:, a, b].view(views, 1, 1) for a, b in ((0, 0), (1, 1), (0, 2), (1, 2)))
    camera = torch.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dim=-1)
    return torch.einsum("bhwi,bji->bhwj", camera, camera_to_world[:, :3, :3]) + camera_to_world[:, None, None, :3, 3]


def _max_pool(x):
    """3x3 max over the valid neighbours (NumPy max_pool_2d with NaN padding and nanmax): NaNs and the border are ignored."""
    return F.max_pool2d(x.nan_to_num(nan=-math.inf)[:, None], 3, 1, 1)[:, 0]


def depth_edges(depth, rtol):
    """Pixels whose 3x3 neighbourhood spans a relative depth range above rtol (depth (views, H, W))."""
    return (_max_pool(depth) + _max_pool(-depth)) / depth > rtol


def normal_edges(normals, degrees):
    """Pixels near a normal change above `degrees` (normals (views, 3, H, W)): the largest angle to a 3x3 neighbour (edge-padded), then
    a 3x3 max. The NumPy reference propagates arccos NaNs in the first max and ignores them in the second, as done here."""
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-12)
    padded = F.pad(normals, (1, 1, 1, 1), mode="replicate")
    height, width = normals.shape[-2:]
    angle = None
    for dy in range(3):
        for dx in range(3):
            cosine = (normals * padded[:, :, dy:dy + height, dx:dx + width]).sum(dim=1)
            current = torch.arccos(cosine)
            angle = current if angle is None else torch.maximum(angle, current)  # NaN propagates, as np.max
    return _max_pool(angle) > math.radians(degrees)


def filter_masks(depth, normals, gs_depth, keep_logits, sky_threshold, remove_sky, remove_edges, normal_degrees, depth_tolerance):
    """compute_filter_mask with the upstream defaults (no confidence filter): points keep the pixels that are neither depth-and-normal edges
    nor sky; Gaussians use the edges of their own depth. keep_logits: the model's depth-mask logits (sky = low)."""
    keep = torch.ones_like(depth, dtype=torch.bool)
    gs_keep = None if gs_depth is None else keep.clone()
    if remove_edges:
        bends = normal_edges(normals, normal_degrees)
        keep &= ~(depth_edges(depth, depth_tolerance) & bends)
        if gs_depth is not None:
            gs_keep &= ~(depth_edges(gs_depth, depth_tolerance) & bends)
    if remove_sky:
        ground = keep_logits.sigmoid() > sky_threshold
        keep &= ground
        if gs_keep is not None:
            gs_keep &= ground
    return keep, gs_keep


def voxel_groups(points, voxel_size):
    """Voxel of each point as a sorted group index: (inverse (N,), group count). The order is NumPy's lexicographic unique order."""
    voxels = (points / voxel_size).floor().long()
    voxels -= voxels.amin(dim=0)
    extent = voxels.amax(dim=0) + 1
    keys = (voxels[:, 0] * extent[1] + voxels[:, 1]) * extent[2] + voxels[:, 2]
    _, inverse = torch.unique(keys, return_inverse=True)
    return inverse, int(inverse.max()) + 1


def merge_points(points, colors, voxel_size, max_points):
    """_compress_points_voxel_then_sample: points averaged per voxel (colors rounded), then a seeded random subset of max_points.
    colors: uint8 (N, 3)."""
    inverse, count = voxel_groups(points, voxel_size)
    if count < len(points):
        sizes = torch.bincount(inverse, minlength=count).double().clamp_min(1)
        sums = points.new_zeros(count, 3, dtype=torch.float64).index_add_(0, inverse, points.double())
        points = sums.float() / sizes.float()[:, None]
        color_sums = points.new_zeros(count, 3, dtype=torch.float64).index_add_(0, inverse, colors.double())
        colors = torch.round(color_sums / sizes[:, None]).clamp(0, 255).to(torch.uint8)
    if len(points) > max_points:
        keep = torch.from_numpy(np.random.default_rng(42).choice(len(points), size=max_points, replace=False)).to(points.device)
        points, colors = points[keep], colors[keep]
    return points, colors


def gaussian_attributes(raw, images, depth, camera_to_world, intrinsics):
    """GaussianSplatRenderer.prepare_splats (inference, positions from the Gaussian depth and the predicted cameras): raw (views, 12, H, W)
    -> per-pixel means, rotations (w x y z), linear scales, SH DC and merge weights, flattened over views and pixels."""
    raw = raw.permute(0, 2, 3, 1).reshape(-1, 12)
    rotations = raw[:, :4] / (raw[:, :4].norm(dim=-1, keepdim=True) + 1e-8)
    scales = raw[:, 4:7].exp().clamp_max(0.3)
    sh = (images.permute(0, 2, 3, 1).reshape(-1, 3) - 0.5) / SH_C0 + raw[:, 8:11]
    weights = raw[:, 11].sigmoid()
    means = unproject(depth, camera_to_world, intrinsics).reshape(-1, 3)
    return means, rotations, scales, sh, weights


def merge_gaussians(means, rotations, scales, sh, weights, voxel_size, max_gaussians, scale_quantile):
    """_voxel_prune_gaussians (weighted averages per voxel, opacity = sum(w^2) / sum(w), as the upstream renderer and exporter), a seeded
    random subset of max_gaussians, then save_gs_ply's filter of the Gaussians larger than the scale_quantile quantile."""
    inverse, count = voxel_groups(means, voxel_size)
    weights64 = weights.double()
    total = weights64.new_zeros(count).index_add_(0, inverse, weights64).clamp_min(1e-8)

    def average(values):
        return (values.new_zeros(count, values.shape[1], dtype=torch.float64).index_add_(0, inverse, values.double() * weights64[:, None]) / total[:, None]).float()

    opacities = (weights64.new_zeros(count).index_add_(0, inverse, weights64 * weights64) / total).float()
    rotations = average(rotations)
    rotations = rotations / rotations.norm(dim=1, keepdim=True).clamp_min(1e-8)
    means, scales, sh = average(means), average(scales), average(sh)
    if count > max_gaussians:
        keep = torch.from_numpy(np.random.default_rng(42).choice(count, size=max_gaussians, replace=False)).to(means.device)
        means, rotations, scales, sh, opacities = means[keep], rotations[keep], scales[keep], sh[keep], opacities[keep]
    largest = scales.amax(dim=-1)
    keep = largest <= torch.quantile(largest, scale_quantile)
    return means[keep], rotations[keep], scales[keep], sh[keep], opacities[keep]


def opencv_to_gltf_cameras(camera_to_world):
    """OpenCV camera-to-world (x right, y down, z forward, OpenCV world) -> glTF world, OpenGL camera (x right, y up, looking down -Z)."""
    flip = torch.diag(torch.tensor((*OPENCV_TO_GLTF, 1.0), dtype=camera_to_world.dtype, device=camera_to_world.device))
    return flip @ camera_to_world @ flip
