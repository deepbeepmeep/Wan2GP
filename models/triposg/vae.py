# TripoSG VAE decoder and occupancy volume extraction, ported from VAST-AI-Research/TripoSG
# triposg/models/autoencoders/autoencoder_kl_triposg.py, triposg/models/embeddings.py and triposg/inference_utils.py (Copyright (c) 2025
# VAST-AI-Research and contributors, MIT) and wgsxm/PartCrafter src/utils/inference_utils.py (Copyright (c) 2025 Yuchen Lin, MIT).
# flash_volume derives from Tencent FlashVDM (see LICENSE_TENCENT_FLASHVDM.txt in this folder): Tencent Hunyuan FlashVDM is licensed
# under the Tencent Hunyuan FlashVDM Community License Agreement, Copyright (C) 2025 Tencent. All Rights Reserved. The trademark rights
# of "Tencent Hunyuan" are owned by Tencent or its affiliate.
# Modified by DeepBeepMeep for WanGP: decoder only (the point-cloud encoder is not used); attention through shared.attention_kit and
# pay_attention on per-head q/k/v weights (rows reordered in the WanGP checkpoint, same products); the latent transformer and the
# geometry decoder keys and values are computed once per mesh instead of once per decoder call (same values); query points decoded by
# chunks in any order (TripoSG decodes with full attention: the order and grouping of the queries only change their values by the
# rounding of the attention kernels); the surface extracted with the shared marching cubes instead of diso's DiffDMC (CC BY-NC), on the
# box of the inside voxels, placed at the positions the grid samples and wound outwards (the logits are negative inside: upstream
# PartCrafter's marching cubes winds its faces inwards); PartCrafter's volume refined from a 128^3 dense grid instead of 256^3, with a
# narrower last band, and its refined values written at their integer voxel coordinates instead of coordinates rounded through FP16 and
# truncated.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from shared.mesh3d import Mesh, marching_cubes
from shared.mesh3d.mesh import submesh
from shared.utils.phase_progress import check_abort
from .dit import DiTBlock


class TripoSGVAEDecoder(nn.Module):
    def __init__(self, latent_channels=64, width_decoder=1024, num_attention_heads=8, num_layers_decoder=16, embed_frequency=8, **_):
        super().__init__()
        self.post_quant = nn.Linear(latent_channels, width_decoder)
        self.decoder = nn.Module()
        self.decoder.blocks = nn.ModuleList([DiTBlock(width_decoder, num_attention_heads, cross_attention=False, qk_norm=False) for _ in range(num_layers_decoder)]
                                            + [DiTBlock(width_decoder, num_attention_heads, width_decoder, self_attention=False, qk_norm=False, norm_cross=True)])
        self.decoder.proj_query = nn.Linear(3 * (2 * embed_frequency + 1), width_decoder)
        self.decoder.norm_out = nn.LayerNorm(width_decoder)
        self.decoder.proj_out = nn.Linear(width_decoder, 1)
        self.embed_frequency = embed_frequency

    def forward(self, latents):
        """Keys and values of the geometry decoder for latents (1, tokens, channels): the latent transformer then the cross-attention
        projections."""
        x = self.post_quant(latents)
        for block in self.decoder.blocks[:-1]:
            x = block(x)
        return self.decoder.blocks[-1].attn2.keys_values(x)

    def query(self, points, keys_values):
        """Occupancy logits (points,) of points (points, 3), negative inside (upstream's logits * -1). FrequencyPositionalEmbedding
        (include_pi=False) computed in the points dtype, as upstream's FP16 pipeline does (its frequencies buffer is cast with the model)."""
        frequencies = 2.0 ** torch.arange(self.embed_frequency, dtype=torch.float32, device=points.device)
        embed = (points[..., None] * frequencies.to(points.dtype)).view(*points.shape[:-1], -1)
        x = self.decoder.proj_query(torch.cat((points, embed.sin(), embed.cos()), dim=-1))[None]
        x = self.decoder.blocks[-1](x, keys_values)
        return self.decoder.proj_out(self.decoder.norm_out(x))[0, :, 0].neg_()


def _decode(vae, keys_values, points, chunk):
    values = torch.empty(points.shape[0], dtype=keys_values[0].dtype, device=points.device)
    for start in range(0, points.shape[0], chunk):
        values[start:start + chunk] = vae.query(points[start:start + chunk].to(keys_values[0].dtype), keys_values)
        check_abort()
    return values


def _near_surface(volume, level):
    """Upstream extract_near_surface_volume_fn: voxels whose sign differs from one of their 6 neighbours (edges replicated), among the
    decoded ones."""
    val = volume + level
    padded = F.pad(val[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    neighbors = [padded[2:, 1:-1, 1:-1], padded[:-2, 1:-1, 1:-1], padded[1:-1, 2:, 1:-1], padded[1:-1, :-2, 1:-1], padded[1:-1, 1:-1, 2:], padded[1:-1, 1:-1, :-2]]
    sign = torch.sign(val.to(torch.float32))
    same = torch.ones_like(val, dtype=torch.bool)
    for neighbor in neighbors:
        same &= torch.sign(torch.where(neighbor > -9000, neighbor, val).to(torch.float32)) == sign
    return (~same).to(torch.int32) * (val > -9000).to(torch.int32)


def _dilate(volume):
    return F.conv3d(volume[None, None], torch.ones((1, 1, 3, 3, 3), dtype=volume.dtype, device=volume.device), padding=1)[0, 0]


@torch.no_grad()
def flash_volume(vae, keys_values, octree_depth=9, bounds=1.005, num_chunks=50000, min_resolution=63, mini_grid_num=4, mc_level=0.0):
    """TripoSG flash_extract_geometry: a coarse grid decoded densely, then the voxels near the surface refined level by level up to
    2**octree_depth. Returns the logits grid (NaN where not decoded) and its bounds."""
    device, dtype = keys_values[0].device, keys_values[0].dtype
    resolution = 2 ** octree_depth
    resolutions = [resolution] if resolution < min_resolution else []
    while resolution >= min_resolution:
        resolutions.append(resolution)
        resolution //= 2
    resolutions.reverse()
    resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
    for i in range(1, len(resolutions)):
        resolutions[i] = resolutions[0] * 2 ** i
    bbox_min, bbox_max = np.full(3, -bounds), np.full(3, bounds)
    axes = [np.linspace(bbox_min[i], bbox_max[i], resolutions[0] + 1, dtype=np.float32) for i in range(3)]
    xyz = torch.from_numpy(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)).to(device, dtype=dtype)
    grid = _decode(vae, keys_values, xyz, num_chunks).view(*[resolutions[0] + 1] * 3)
    del xyz
    for resolution in resolutions[1:]:
        size, step = resolution + 1, (bbox_max - bbox_min) / resolution
        current = _near_surface(grid, mc_level)
        current += grid.abs() < 0.95
        expand = 0 if resolution == resolutions[-1] else 1
        for _ in range(expand):
            current = _dilate(_dilate(current.to(dtype)))
        next_index = torch.zeros((size, size, size), dtype=dtype, device=device)
        cidx_x, cidx_y, cidx_z = torch.where(current > 0)
        next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
        del current, cidx_x, cidx_y, cidx_z
        for _ in range(2 - expand):
            next_index = _dilate(next_index)
        index = torch.where(next_index > 0)
        del next_index
        points = torch.stack(index, dim=1) * torch.tensor(step, dtype=torch.float32, device=device) + torch.tensor(bbox_min, dtype=torch.float32, device=device)
        grid = torch.full((size, size, size), -10000.0, dtype=dtype, device=device)
        grid[index] = _decode(vae, keys_values, points, num_chunks)
        del points, index
    grid[grid == -10000.0] = float("nan")
    return grid, (-bounds, bounds)


def _band_children(grid, max_logit=None):
    """Coordinates (int16) in the grid of twice the resolution of the children of the band voxels near the surface: upstream's
    find_candidates_band (inner voxels with |sigmoid * 2 - 1| < 1 in the grid dtype, about |logit| < 8.3) or the inner voxels with
    |logit| < max_logit, dilated as upstream's expand_edge_region_fast (max pool of 5, or 3 from 512 voxels), each split into its 8
    children."""
    core, kernel = grid[1:-1, 1:-1, 1:-1], 5 if grid.shape[0] < 512 else 3
    band = torch.zeros_like(grid)
    band[1:-1, 1:-1, 1:-1] = (torch.sigmoid(core) * 2 - 1).abs() < 1 if max_logit is None else core.abs() < max_logit
    low = torch.nonzero(F.max_pool3d(band[None, None], kernel_size=kernel, stride=1, padding=kernel // 2)[0, 0]).to(torch.int16)
    del band
    children = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=torch.int16, device=grid.device)
    return (low[None] * 2 + children[:, None]).reshape(-1, 3)


@torch.no_grad()
def hierarchical_volume(vae, keys_values, dense_octree_depth=7, hierarchical_octree_depth=9, bounds=1.005, num_chunks=50000, surface_logit=4.0):
    """PartCrafter hierarchical_extract_geometry from a coarser dense grid: 2**dense_octree_depth points per axis (upstream: 2**8, 16.8M
    queries) on the lattice of the finer levels (upstream's linspace grid is offset from it by up to one voxel), upsampled (nearest) level
    by level with the band near the surface decoded again, upstream's band (about |logit| < 8.3) at the intermediate levels and the voxels
    with |logit| < surface_logit at the last one, both dilated by 2 voxels as upstream. Across the surface the logits change by about 0.65
    per final voxel: upstream's last band reaches about 15 voxels away from it. A 2**6 dense grid would miss thin parts (railings,
    antennas) that 2**7 still finds. Returns the logits grid and its bounds."""
    device, dtype = keys_values[0].device, keys_values[0].dtype
    cells = 2 ** dense_octree_depth
    axis = (torch.arange(cells, dtype=dtype, device=device) - cells / 2) * (bounds / (cells / 2))
    grid = _decode(vae, keys_values, torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).view(-1, 3), num_chunks).view(cells, cells, cells)
    del axis
    for depth in range(dense_octree_depth + 1, hierarchical_octree_depth + 1):
        offset = 2 ** depth / 2
        coords = _band_children(grid, surface_logit if depth == hierarchical_octree_depth else None)
        fine = F.interpolate(grid[None, None], scale_factor=2)[0, 0]
        del grid
        points = (coords.to(dtype) - offset) * (bounds / offset)
        coords = coords.long()
        fine[coords[:, 0], coords[:, 1], coords[:, 2]] = _decode(vae, keys_values, points, num_chunks)
        grid = fine
        del points, coords, fine
    return grid, (-bounds, -bounds + (2 ** hierarchical_octree_depth - 1) * 2 * bounds / 2 ** hierarchical_octree_depth)


def extract_mesh(grid, bounds):
    """Surface at level 0 of a logits grid (inside < 0, NaN where not decoded), or None when no cell has inside and outside corners
    (upstream's marching cubes raises when the grid has no inside or no outside voxel). Scikit-image's marching cubes (Lewiner) runs on
    the box of the inside voxels grown by one voxel, out of which no cell has an inside corner, masked to the cells with inside and
    outside corners and their neighbours: the same faces without visiting the whole grid. Faces touching an undecoded voxel are dropped."""
    inside = grid < 0
    if not inside.any():
        return None
    low, high = [], []
    for axis in range(3):
        hits = inside.any(dim=[other for other in range(3) if other != axis]).nonzero()[:, 0]
        low.append(max(int(hits[0]) - 1, 0))
        high.append(min(int(hits[-1]) + 2, grid.shape[axis]))
    del inside
    volume = grid[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    corners = F.max_pool3d((volume < 0).half()[None, None], 2, 1)
    corners = F.max_pool3d((volume > 0).half()[None, None], 2, 1).mul_(corners)
    if not corners.any():
        return None
    mask = F.max_pool3d(F.pad(corners, (0, 1, 0, 1, 0, 1)), 3, 1, 1)[0, 0] > 0
    del corners
    step = (bounds[1] - bounds[0]) / (grid.shape[0] - 1)
    box = ([bounds[0] + i * step for i in low], [bounds[0] + (i - 1) * step for i in high])
    vertices, faces = marching_cubes(volume.cpu(), 0.0, box, inside="below", mask=mask.cpu())
    del mask
    mesh = Mesh(vertices, faces)
    if not torch.isfinite(vertices).all():
        mesh = submesh(mesh, torch.isfinite(vertices).all(1)[faces].all(1))
    return mesh.to(grid.device)
