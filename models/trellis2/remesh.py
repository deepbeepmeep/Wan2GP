# Narrow-band remeshing of TRELLIS.2 meshes in PyTorch: dual contouring of (unsigned distance - band) on the voxel grid, the algorithm of
# CuMesh remesh_narrow_band_dc (github.com/JeffreyXiang/CuMesh, cumesh/remeshing.py and src/remesh, MIT License) used by o-voxel's
# to_glb(remesh=True). The raw flexible-dual-grid surface has non-manifold edges and cracks wherever the predicted edge crossings disagree;
# its band offset is a closed, consistently oriented surface that decimates and unwraps cleanly. Unsigned distances are splatted from each
# triangle to the grid corners around it (no BVH), and the inner offset shells of solid parts (negative volume) are dropped. On CUDA GPUs
# with Triton, the splat and the corner lookups run as bit-identical kernels (remesh_kernels.py).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch

from shared.utils.phase_progress import check_abort
from .fdg_mesh import compact, face_components
from .sparse import SparseTensor

try:
    from . import remesh_kernels as _kernels
except (ImportError, OSError, RuntimeError):
    _kernels = None

_EDGE_VOXELS = [[[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]]
_CORNERS = [[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)]
_CELL_EDGES = [(a, b) for a in range(8) for b in range(8) if a < b and bin(a ^ b).count("1") == 1]  # the 12 edges of a cell, corner pairs
_SPLAT_POINTS = 1 << 21  # bound of the (triangle, corner) pairs of one splat chunk (about 0.7 KB each at the peak)
_CELL_ROWS = 1 << 21  # corners examined at a time for crossing cells
_DILATE_ROWS = 32  # x values of the grid dilated at a time
_VOLUME_FACES = 1 << 22  # faces of a component volume chunk


def _dilate(cells, low, high):
    """Integer points cells + [low, high]^3 (unique, sorted), dilated one axis at a time on 1-D keys, by slabs of _DILATE_ROWS x values
    (the full dilation would hold (high - low + 1) shifted copies of every key at once)."""
    base = cells.amin(0) + low
    span = int((cells.amax(0) - base).max()) + high + 1
    plane = span * span
    local = (cells - base).long()
    keys = torch.unique((local[:, 0] * span + local[:, 1]) * span + local[:, 2])
    del local
    slabs = []
    for x0 in range(0, span, _DILATE_ROWS):
        x1 = min(span, x0 + _DILATE_ROWS)
        # the sources of the slab's points have x in [x0 - high, x1 - 1 - low]
        start, stop = (int(torch.searchsorted(keys, bound * plane)) for bound in (x0 - high, x1 - low))
        part = keys[start:stop]
        for stride in (plane, span, 1):
            part = torch.unique(torch.cat([part + offset * stride for offset in range(low, high + 1)]))
        slabs.append(part[(part >= x0 * plane) & (part < x1 * plane)])
        del part
    del keys
    # int32 coordinates written slab by slab (no full-size int64 copies)
    points = torch.empty((sum(len(part) for part in slabs), 3), dtype=torch.int32, device=cells.device)
    start = 0
    while slabs:
        part = slabs.pop(0)
        points[start:start + len(part)] = torch.stack([part // plane, part // span % span, part % span], 1)
        start += len(part)
        del part
    return points.add_(base.int())


def _point_triangle_distance(p, a, b, c):
    """Euclidean distance of points p to triangles (a, b, c), all (..., 3)."""
    ab, ac, ap = b - a, c - a, p - a
    normal = torch.cross(ab, ac, dim=-1)
    area2 = (normal * normal).sum(-1)
    # barycentric coordinates of the projection, inside test
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (ap * ab).sum(-1), (ap * ac).sum(-1)
    denominator = (d00 * d11 - d01 * d01).clamp_min(1e-20)
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    inside = (v >= 0) & (w >= 0) & (v + w <= 1) & (area2 > 1e-20)
    plane = (ap * normal).sum(-1).abs() / area2.clamp_min(1e-20).sqrt()

    def segment(start, end):
        direction = end - start
        t = (((p - start) * direction).sum(-1) / (direction * direction).sum(-1).clamp_min(1e-20)).clamp(0, 1)
        return (p - start - t.unsqueeze(-1) * direction).norm(dim=-1)

    edges = torch.minimum(torch.minimum(segment(a, b), segment(b, c)), segment(c, a))
    return torch.where(inside, plane, edges)


def _unsigned_distance(points, faces, corners, reach):
    """Distance from each grid corner (SparseTensor coords) to the triangles, exact below `reach` voxels and `reach` beyond."""
    udf = torch.full((len(corners),), float(reach), device=points.device)
    triangles = points[faces]
    low = torch.floor(triangles.amin(1) - reach).long()
    extent = torch.ceil(triangles.amax(1) + reach).long() - low + 1
    counts = extent.prod(1)
    start = 0
    while start < len(faces):
        total = torch.cumsum(counts[start:], 0)
        stop = start + max(1, int(torch.searchsorted(total, _SPLAT_POINTS)))
        n = counts[start:stop]
        triangle = torch.repeat_interleave(torch.arange(start, stop, device=points.device), n)
        rank = torch.arange(len(triangle), device=points.device) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        ex = extent[triangle]
        offset = torch.stack([rank % ex[:, 0], rank // ex[:, 0] % ex[:, 1], rank // (ex[:, 0] * ex[:, 1])], 1)
        grid = low[triangle] + offset
        tri = triangles[triangle]
        distance = _point_triangle_distance(grid.float(), tri[:, 0], tri[:, 1], tri[:, 2])
        near = distance < reach
        slot = corners.find(grid[near])
        valid = slot >= 0
        udf.scatter_reduce_(0, slot[valid].long(), distance[near][valid], "amin")
        del triangle, rank, ex, offset, grid, tri, distance, near, slot, valid
        start = stop
        check_abort()
    return udf


def _crossing_cells(corners, value):
    """(min corner coords, corner values (M, 8)) of the cells whose 8 corners are in the corner set and change sign, in corner order."""
    corner_offsets = torch.tensor(_CORNERS, dtype=torch.int32, device=value.device)
    found = []
    for start in range(0, len(corners), _CELL_ROWS):
        index = corners.find(corners.coords[start:start + _CELL_ROWS, None] + corner_offsets)
        complete = (index >= 0).all(1)
        values = value[index[complete].long()]
        negative = values < 0
        crossing = negative.any(1) & ~negative.all(1)
        found.append((corners.coords[start:start + _CELL_ROWS][complete][crossing], values[crossing]))
        del index, complete, values, negative, crossing
    return [torch.cat(parts) for parts in zip(*found)]


def remesh_narrow_band(mesh, resolution, band=1.0):
    """Closed surface at `band` voxels from the mesh (vertices in [-0.5, 0.5]^3, resolution voxels across), triangulated quads of dual
    contouring with outward windings; components of negative volume (inner offsets of solids) are removed. mesh: [vertices, faces],
    handed over (the list is emptied) so that the input is released once the distances are splatted."""
    vertices, faces = mesh
    mesh.clear()
    device = vertices.device
    fused = _kernels is not None and _kernels.supported(device)
    points = (vertices.float() + 0.5) * resolution
    del vertices
    reach = band + 1.0
    # grid corners within reach of the vertices (the surface lies within a voxel of its vertices)
    coords = _dilate(torch.floor(points).long(), -int(reach) - 1, int(reach) + 2)
    corners = _kernels.ColumnIndex(coords) if fused else SparseTensor(None, coords)
    del coords
    value = (_kernels.unsigned_distance if fused else _unsigned_distance)(points, faces, corners, reach).sub_(band)
    del points, faces
    # cells (named by their min corner) whose 8 corners are in the band set and change sign
    cell_coords, cell_values = _kernels.crossing_cells(corners, value) if fused else _crossing_cells(corners, value)
    del corners, value
    check_abort()
    sign = cell_values < 0
    # dual vertex: mean of the crossing points on the cell edges
    corner_offsets = torch.tensor(_CORNERS, dtype=torch.int32, device=device)
    sums = torch.zeros((len(cell_coords), 3), device=device)
    hits = torch.zeros(len(cell_coords), device=device)
    for a, b in _CELL_EDGES:
        va, vb = cell_values[:, a], cell_values[:, b]
        cut = (va < 0) != (vb < 0)
        t = (-va / (vb - va).where(cut, torch.ones_like(va))).clamp(0, 1)
        position = corner_offsets[a].float() + t[:, None] * (corner_offsets[b] - corner_offsets[a]).float()
        sums += position * cut[:, None]
        hits += cut
        del va, vb, cut, t, position
    del cell_values
    dual = cell_coords.float() + sums / hits.clamp_min(1)[:, None]
    del sums, hits
    # quads around the crossing edges (o-voxel convention: edge along axis a at the cell corner +1 on the other two axes)
    cells = _kernels.ColumnIndex(cell_coords) if fused else SparseTensor(None, cell_coords)
    find = (lambda query: _kernels.find(cells, query)) if fused else cells.find
    offsets = torch.tensor(_EDGE_VOXELS, dtype=torch.int32, device=device)
    split_02, split_13 = torch.tensor([0, 1, 2, 0, 2, 3], device=device), torch.tensor([0, 1, 3, 3, 1, 2], device=device)
    edge_corners = {0: (6, 7), 1: (5, 7), 2: (3, 7)}  # (low, high) corner of the edge along x, y, z at (+1, +1) on the other axes
    triangles = []
    for axis in range(3):
        low, high = edge_corners[axis]
        cut = sign[:, low] != sign[:, high]
        quads = find(cell_coords[cut][:, None] + offsets[axis])
        keep = (quads >= 0).all(1)
        quads = quads[keep].long()
        # the default quad order winds toward -axis; the outside (positive values) is on the high side when the low end is inside
        outward = sign[cut][keep][:, low]
        quads = torch.where(outward[:, None], quads.flip(1), quads)
        q = dual[quads]
        align = []
        for split in (split_02, split_13):
            t = q[:, split].view(-1, 2, 3, 3)
            normals = torch.cross(t[:, :, 1] - t[:, :, 0], t[:, :, 2] - t[:, :, 0], dim=-1)
            align.append((normals[:, 0] * normals[:, 1]).sum(-1))
            del t, normals
        del q
        triangles.append(torch.where((align[0] > align[1])[:, None], quads[:, split_02], quads[:, split_13]).reshape(-1, 3))
        del cut, quads, keep, outward, align
    del cells, find, cell_coords, sign
    faces = torch.cat(triangles)
    del triangles
    vertices, faces = compact(dual / resolution - 0.5, faces)
    del dual
    # drop the inner offset shells (and empty components): signed volume of each component, by chunks of faces
    component, count = face_components(len(vertices), faces)
    volume = torch.zeros(count, dtype=torch.float64, device=device)
    for start in range(0, len(faces), _VOLUME_FACES):
        v = vertices[faces[start:start + _VOLUME_FACES]].double()
        volume.index_add_(0, component[start:start + _VOLUME_FACES], (v[:, 0] * torch.cross(v[:, 1], v[:, 2], dim=1)).sum(1))
        del v
    faces = faces[volume[component] > 0]
    del component, volume
    return compact(vertices, faces)
