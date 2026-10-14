# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Iso-surface extraction from a sampled scalar volume (occupancy logits, SDF, density).

Coordinates: volume[i, j, k] is the sample at bounds_min + (i, j, k) / (shape - 1) * (bounds_max - bounds_min), the
first and last samples lying on the bounds; axis 0 is x, 1 is y, 2 is z. Output faces are counter-clockwise seen from
outside the object (outward right-hand-rule normals), the object being where the volume is above `level` (inside =
"above", e.g. occupancy) or below it ("below", e.g. signed distance).

Methods:
- "lewiner" (default): scikit-image's marching cubes (Lewiner et al. 2003, topologically resolved ambiguities), on the
  CPU (the volume is copied to the CPU, the mesh is returned on the volume's device).
- "torch": vectorized classic marching cubes in PyTorch on the volume's device. Its vertices are exactly the edge
  crossings that scikit-image also produces (same linear interpolation); its case table is built here so that
  ambiguous cube faces always separate the inside corners, which keeps neighbouring cells consistent: the surface is
  crack-free and manifold (closed for a volume whose border is outside). Unlike Lewiner's method it adds no vertices
  inside the cells and resolves ambiguous cells by separating the inside corners, so it is not topologically identical
  to scikit-image there (on smooth fields the two meshes coincide almost everywhere).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

_CORNERS = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]  # corner c = x + 2y + 4z
_EDGES = [(c, c | (1 << axis)) for axis in range(3) for c in range(8) if not c & (1 << axis)]  # 12 edges, 4 per axis


def _face_cycles() -> list[list[int]]:
    """The 6 cube faces as corner cycles, counter-clockwise seen from outside the cube."""
    cycles = []
    for axis, side in itertools.product(range(3), (0, 1)):
        u, v = [a for a in range(3) if a != axis]
        corners = [sum(bit << b for bit, b in ((side, axis), (du, u), (dv, v))) for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1))]
        normal = np.zeros(3)
        normal[axis] = 1 if side else -1
        p = [np.array(_CORNERS[c], float) for c in corners]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), normal) < 0:
            corners.reverse()
        cycles.append(corners)
    return cycles


def _case_table() -> tuple[np.ndarray, np.ndarray]:
    """Triangles (as cube edge triples, counter-clockwise seen from the outside corners) of the 256 corner configurations
    (bit c set = corner c inside), padded with -1, and their counts.

    On every cube face, each run of inside corners is cut off by a segment, so diagonal inside corners are separated and
    the two cells sharing a face always agree. The segments, oriented with the inside corners on their left seen from
    outside the cube, chain into loops; each loop is fan-triangulated from a vertex whose diagonals join no two vertices
    of a common cube face, so diagonals lie inside the cell and never coincide with a neighbour's edges (manifold)."""
    edge_of = {frozenset(e): i for i, e in enumerate(_EDGES)}
    faces = _face_cycles()
    edge_faces = [{f for f, cycle in enumerate(faces) if a in cycle and b in cycle} for a, b in _EDGES]
    triangles = []
    for config in range(256):
        inside = [bool(config >> c & 1) for c in range(8)]
        following = {}
        for cycle in faces:
            for i, corner in enumerate(cycle):  # each run of inside corners: segment from its exit edge to its entry edge
                if inside[corner] and not inside[cycle[i - 1]]:
                    j = i
                    while inside[cycle[(j + 1) % 4]]:
                        j += 1
                    exit_edge = edge_of[frozenset((cycle[j % 4], cycle[(j + 1) % 4]))]
                    following[exit_edge] = edge_of[frozenset((cycle[i - 1], corner))]
        polygons = []
        while following:
            start, loop = next(iter(following)), []
            edge = start
            while True:
                loop.append(edge)
                edge = following.pop(edge)
                if edge == start:
                    break
            n = len(loop)
            apex = next(a for a in range(n) if all(not edge_faces[loop[a]] & edge_faces[loop[j]] for j in range(n) if j not in (a, (a - 1) % n, (a + 1) % n)))
            polygons.append(loop[apex:] + loop[:apex])
        triangles.append([(p[0], p[i + 1], p[i]) for p in polygons for i in range(1, len(p) - 1)])
    width = max(map(len, triangles))
    table = np.full((256, width, 3), -1, np.int64)
    for config, tris in enumerate(triangles):
        if tris:
            table[config, :len(tris)] = tris
    return table, np.array([len(t) for t in triangles], np.int64)


_TABLE, _COUNTS = _case_table()
_EDGE_AXIS = np.array([int(np.log2(b ^ a)) for a, b in _EDGES])
_EDGE_BASE = np.array([_CORNERS[a] for a, _ in _EDGES])


def _bounds(bounds, device) -> tuple[torch.Tensor, torch.Tensor]:
    low, high = (torch.as_tensor(np.asarray(b, np.float64), device=device).expand(3) for b in (bounds[0], bounds[1]))
    return low, high


def _torch_marching_cubes(volume: torch.Tensor, level: float, above: bool, slab: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Vertices in index space [V, 3] (float64) and faces [F, 3] int64 (orientation of the case table)."""
    device = volume.device
    table, counts = torch.from_numpy(_TABLE).to(device), torch.from_numpy(_COUNTS).to(device)
    edge_axis, edge_base = torch.from_numpy(_EDGE_AXIS).to(device), torch.from_numpy(_EDGE_BASE).to(device)
    nx, ny, nz = volume.shape
    triangle_edges = []
    for x0 in range(0, nx - 1, slab):
        x1 = min(x0 + slab, nx - 1)
        inside = volume[x0:x1 + 1] > level if above else volume[x0:x1 + 1] < level
        config = torch.zeros(x1 - x0, ny - 1, nz - 1, dtype=torch.uint8, device=device)
        for c, (dx, dy, dz) in enumerate(_CORNERS):
            config |= inside[dx:dx + x1 - x0, dy:dy + ny - 1, dz:dz + nz - 1].to(torch.uint8) << c
        cells = ((config > 0) & (config < 255)).nonzero()
        config = config[cells[:, 0], cells[:, 1], cells[:, 2]].long()
        cells[:, 0] += x0
        cell = torch.repeat_interleave(torch.arange(len(cells), device=device), counts[config])
        slot = torch.arange(len(cell), device=device) - torch.repeat_interleave(counts[config].cumsum(0) - counts[config], counts[config])
        local = table[config[cell], slot]  # [T, 3] cube edges
        point = cells[cell][:, None, :] + edge_base[local]  # grid point at the start of each edge
        triangle_edges.append(((point[..., 0] * ny + point[..., 1]) * nz + point[..., 2]) * 3 + edge_axis[local])
    edges = torch.cat(triangle_edges) if triangle_edges else torch.zeros(0, 3, dtype=torch.int64, device=device)
    unique, faces = torch.unique(edges, return_inverse=True)
    axis, start = unique % 3, unique // 3
    start = torch.stack([start // (ny * nz), start // nz % ny, start % nz], 1)
    end = start + torch.nn.functional.one_hot(axis, 3)
    v0, v1 = (volume[p[:, 0], p[:, 1], p[:, 2]].double() for p in (start, end))
    t = (level - v0) / (v1 - v0)
    return start.double() + t[:, None] * (end - start).double(), faces


def marching_cubes(volume, level: float = 0.0, bounds=(-1.0, 1.0), *, inside: str = "above", method: str = "lewiner", mask=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Iso-surface of volume [X, Y, Z] (torch tensor or numpy array) at level: vertices [V, 3] float32 in world
    coordinates and faces [F, 3] int64, outward counter-clockwise (see the module docstring for the conventions and the
    methods). bounds: (min, max) as scalars or xyz triples (e.g. Hunyuan3D's [-1.01, 1.01]). Tensors are returned on the
    volume's device (CPU for numpy input). mask ("lewiner" only): boolean [X, Y, Z] restricting the cells visited
    (scikit-image tests it at one corner of each cell: mark the cells crossed by the surface dilated by one voxel); a
    mask that covers them gives the same mesh faster on large, mostly empty volumes."""
    if inside not in ("above", "below"):
        raise ValueError(f"inside must be 'above' or 'below', not {inside!r}")
    tensor = volume if isinstance(volume, torch.Tensor) else torch.from_numpy(np.asarray(volume))
    device = tensor.device
    if method == "torch":
        vertices, faces = _torch_marching_cubes(tensor, level, inside == "above")
        flip = False
    elif method == "lewiner":
        from skimage.measure import marching_cubes as skimage_marching_cubes
        array = tensor.detach().float().cpu().numpy()
        gradient = "descent" if inside == "above" else "ascent"
        mask = None if mask is None else (mask if isinstance(mask, np.ndarray) else mask.cpu().numpy())
        vertices, faces, _, _ = skimage_marching_cubes(array, level, method="lewiner", gradient_direction=gradient, mask=mask)
        vertices, faces = torch.from_numpy(vertices.astype(np.float64)).to(device), torch.from_numpy(faces.astype(np.int64)).to(device)
        flip = True  # scikit-image winds its faces clockwise seen from outside in index space
    else:
        raise ValueError(f"Unknown marching cubes method {method!r}")
    low, high = _bounds(bounds, device)
    scale = (high - low) / (torch.tensor(tensor.shape, dtype=torch.float64, device=device) - 1)
    vertices = (vertices * scale + low).float()
    return vertices, faces.flip(1) if flip else faces
