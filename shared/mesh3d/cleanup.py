# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Mesh cleanup, decimation, UV unwrapping and texture / vertex-colour inpainting.

Results stay on the device of their inputs; decimation (fast-simplification, MIT), UV unwrapping (xatlas, MIT),
connected components (scipy) and image inpainting (OpenCV) run on the CPU.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import threading

import numpy as np
import torch

from .mesh import Mesh, face_normals, select_vertices, submesh, weld


def remove_degenerate(mesh: Mesh, *, min_area: float = 0.0) -> Mesh:
    """Drops faces with a repeated vertex index or an area <= min_area, then the vertices no face uses."""
    faces = mesh.faces
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    keep &= face_normals(mesh.vertices, faces).norm(dim=1) > 2 * min_area
    return submesh(mesh, keep)


def merge_vertices(mesh: Mesh, *, tolerance: float = 0.0) -> Mesh:
    """Merges vertices whose positions are equal (or round to the same multiple of tolerance) and whose normals, uvs and
    colours are identical, so UV seams and hard edges stay split; to weld through them, pass a Mesh without those
    attributes. The first vertex of each group is kept; faces that collapse are removed."""
    position = torch.round(mesh.vertices / tolerance).long() if tolerance > 0 else (mesh.vertices.float() + 0.0).view(torch.int32).long()
    keys = [position] + [(getattr(mesh, k).float() + 0.0).view(torch.int32).long() for k in ("normals", "uvs", "colors") if getattr(mesh, k) is not None]
    _, inverse = torch.unique(torch.cat(keys, 1), dim=0, return_inverse=True)
    order = torch.arange(len(inverse), device=inverse.device)
    first = torch.full((int(inverse.max()) + 1,), len(inverse), device=inverse.device).scatter_reduce(0, inverse, order, "amin")
    rank = torch.empty_like(first)
    rank[first.argsort()] = torch.arange(len(first), device=first.device)  # keep the original vertex order
    faces = rank[inverse][mesh.faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    return select_vertices(mesh, first.sort().values, faces[keep])


def _components(mesh: Mesh) -> tuple[np.ndarray, int]:
    """Connected component of each face, faces being connected through shared positions (seams welded)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    _, inverse = weld(mesh.vertices)
    faces = inverse[mesh.faces].cpu().numpy()
    count = int(inverse.max()) + 1 if len(inverse) else 0
    rows, cols = faces[:, [0, 1, 2]].reshape(-1), faces[:, [1, 2, 0]].reshape(-1)
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(count, count))
    components, labels = connected_components(graph, directed=False)
    return labels[faces[:, 0]], components


def remove_floaters(mesh: Mesh, min_face_ratio: float = 0.01, *, min_diameter_ratio: float = 0.0) -> Mesh:
    """Keeps the connected components (faces connected through shared positions) with at least min_face_ratio of the
    faces and a bounding box diagonal of at least min_diameter_ratio of the mesh's; the largest component is always kept.
    Hunyuan3D's pymeshlab cleanup corresponds to min_face_ratio=0, min_diameter_ratio=0.1."""
    if len(mesh.faces) == 0:
        return mesh
    labels, components = _components(mesh)
    labels = torch.from_numpy(labels).to(mesh.faces.device)
    sizes = torch.bincount(labels, minlength=components)
    corners = mesh.vertices[mesh.faces].float()
    index = labels[:, None].expand(-1, 3)
    low = torch.full((components, 3), float("inf"), dtype=torch.float32, device=corners.device).scatter_reduce(0, index, corners.amin(1), "amin")
    high = torch.full((components, 3), float("-inf"), dtype=torch.float32, device=corners.device).scatter_reduce(0, index, corners.amax(1), "amax")
    diagonal = (high - low).norm(dim=1)
    whole = (corners.reshape(-1, 3).amax(0) - corners.reshape(-1, 3).amin(0)).norm()
    keep = (sizes >= min_face_ratio * len(mesh.faces)) & (diagonal >= min_diameter_ratio * whole)
    keep[sizes.argmax()] = True
    return submesh(mesh, keep[labels])


class _Worker:
    """A mesh_worker.py job in a child process: the input arrays are sent and the outputs collected by a thread, so that the caller
    keeps running (CPU or GPU work) meanwhile."""

    SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_worker.py")

    def __init__(self, job: str, arguments, arrays):
        payload = io.BytesIO()
        for array in arrays:
            np.save(payload, array)
        self.output = {}
        self.process = subprocess.Popen([sys.executable, "-I", self.SCRIPT, job, *(str(int(value)) for value in arguments)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.thread = threading.Thread(target=lambda: self.output.update(zip(("stdout", "stderr"), self.process.communicate(payload.getvalue()))), daemon=True)
        self.thread.start()

    def result(self, count: int) -> list[np.ndarray]:
        self.thread.join()
        if self.process.returncode:
            raise RuntimeError(f"Mesh worker failed: {self.output['stderr'].decode(errors='replace').strip()}")
        stream = io.BytesIO(self.output["stdout"])
        return [np.load(stream) for _ in range(count)]

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.thread.join()


def _decimate_parts(points: np.ndarray, triangles: np.ndarray, target_faces: int, aggressiveness: float, parts: int) -> tuple[np.ndarray, np.ndarray]:
    """The cells of a parts^3 grid decimated in parallel child processes, each to its share of target_faces with its cut border kept,
    then welded back on their unchanged border vertices. The cells are extracted and their processes started by parallel threads (the
    numpy work releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    from .mesh_worker import cell_mesh, grid_parts

    def start(group):
        used, local = cell_mesh(points, triangles, group)
        share = round(target_faces * len(group) / len(triangles))
        return _Worker("decimate_cell", (share, aggressiveness), (points[used], local.astype(np.int32)))

    groups = grid_parts(points, triangles, parts, centers=False)
    with ThreadPoolExecutor(len(groups)) as pool:
        starts = [pool.submit(start, group) for group in groups]
    workers = [job.result() for job in starts if job.exception() is None]
    try:
        results = [job.result().result(2) for job in starts]  # raises the error of a failed start
    finally:
        for worker in workers:
            worker.close()
    offsets = np.cumsum([0] + [len(p) for p, _ in results])
    welded, inverse = np.unique(np.concatenate([p for p, _ in results]), axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[np.concatenate([f + offset for (_, f), offset in zip(results, offsets)])]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
    return welded, faces.astype(np.int32)


def decimate(mesh: Mesh, target_faces: int, *, aggressiveness: float = 7.0, preserve_border: bool = False, parts: int = 1,
             welded: bool = False) -> Mesh:
    """Quadric edge-collapse decimation (fast-simplification) to about target_faces faces, on positions welded through
    seams. Kept: positions (optimized by the collapses) and vertex colours (averaged over the merged vertices). Lost:
    normals (recomputed by the writers and the renderer), uvs and textures, which no longer match the new topology:
    decimate before unwrap_uv and baking. A mesh with at most target_faces faces is returned unchanged.
    Welding runs on the mesh's device (much faster on a GPU for large meshes); welded=True skips it for a mesh whose positions are
    already unique, faces non-degenerate and vertices all used (e.g. fresh from dual contouring). parts > 1 first decimates the cells
    of a parts^3 grid in parallel child processes (their cut borders kept), then the welded result in a last pass: fast-simplification
    is single-threaded and keeps the GIL (16.9M faces to 100K: 46 s in one pass, about 20 s with parts=2, at the same distance to the
    input surface). Not available with vertex colours."""
    import fast_simplification
    if len(mesh.faces) <= target_faces:
        return mesh
    if parts > 1 and mesh.colors is not None:
        raise ValueError("decimate(parts > 1) does not carry vertex colours")
    device = mesh.vertices.device
    if welded:
        positions, inverse, faces = mesh.vertices, torch.arange(len(mesh.vertices), device=device), mesh.faces
        used, points = inverse, positions.cpu().numpy().astype(np.float64)
    else:
        positions, inverse = weld(mesh.vertices)
        faces = inverse[mesh.faces]
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
        used, faces = torch.unique(faces, return_inverse=True)
        points = positions[used].cpu().numpy().astype(np.float64)
    triangles = faces.cpu().numpy().astype(np.int32)
    if parts > 1:
        points, triangles = _decimate_parts(points, triangles, target_faces, aggressiveness, parts)
    simplified = fast_simplification.simplify(points, triangles, target_count=target_faces, agg=aggressiveness, preserve_border=preserve_border,
                                              return_collapses=mesh.colors is not None)
    points_out, faces_out = simplified[:2]
    result = Mesh(torch.from_numpy(points_out.astype(np.float32)).to(device), torch.from_numpy(faces_out.astype(np.int64)).to(device))
    if mesh.colors is not None:
        _, _, mapping = fast_simplification.replay_simplification(points.astype(np.float32), triangles, simplified[2])
        mapping = torch.from_numpy(mapping.astype(np.int64)).to(device)
        first = torch.full((len(positions),), len(inverse), dtype=torch.int64, device=device)
        first = first.scatter_reduce(0, inverse, torch.arange(len(inverse), device=device), "amin")  # first vertex of each position
        colors = mesh.colors.float()[first[used]]
        sums = torch.zeros(len(points_out), colors.shape[1], device=device).index_add_(0, mapping, colors)
        result.colors = sums / torch.bincount(mapping, minlength=len(points_out)).clamp_min(1)[:, None]
    return result


def _atlas_mesh(mesh: Mesh, mapping, indices, uvs) -> Mesh:
    device = mesh.vertices.device
    unwrapped = select_vertices(mesh, torch.from_numpy(mapping.astype(np.int64)).to(device), torch.from_numpy(indices.astype(np.int64).reshape(-1, 3)).to(device))
    return unwrapped.replace(uvs=torch.from_numpy(uvs).to(device), base_color=None, metallic_roughness=None, normal_map=None)


def _xatlas_arrays(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    return mesh.vertices.detach().float().cpu().numpy(), mesh.faces.cpu().numpy().astype(np.uint32)


def unwrap_uv(mesh: Mesh, *, resolution: int = 0, padding: int = 2, max_iterations: int = 1, parts: int = 1) -> Mesh:
    """UV atlas from xatlas: a Mesh whose vertices are split along the chart seams (faces keep their order), with uvs in [0, 1] (glTF
    convention, see mesh.py). Normals and colours follow their vertices; textures are dropped (they belong to the old uvs). For good
    charts, weld first (merge_vertices on a Mesh without uvs) and decimate large meshes before unwrapping. resolution > 0 packs for that
    texture size with `padding` texels between charts. xatlas is CPU code whose chart computation grows faster than the mesh: parts > 1
    unwraps the cells of a parts^3 grid as separate xatlas meshes, computed in parallel (100K faces: 26-46 s whole, under 2 s with
    parts=2), with slightly longer seams; BackgroundUnwrap runs it in a child process."""
    from .mesh_worker import unwrap_arrays
    return _atlas_mesh(mesh, *unwrap_arrays(*_xatlas_arrays(mesh), resolution, padding, max_iterations, parts))


class BackgroundUnwrap:
    """unwrap_uv(mesh, ...) computed in a child process (mesh_worker.py), so that it overlaps other work: xatlas keeps the GIL while it
    runs, which would stall the kernel launches of a thread. result() returns the same Mesh as unwrap_uv; close() stops the process
    (call it on cancellation)."""

    def __init__(self, mesh: Mesh, *, resolution: int = 0, padding: int = 2, max_iterations: int = 1, parts: int = 1):
        self.mesh = mesh
        self.worker = _Worker("unwrap", (resolution, padding, max_iterations, parts), _xatlas_arrays(mesh))

    def result(self) -> Mesh:
        return _atlas_mesh(self.mesh, *self.worker.result(3))

    def close(self):
        self.worker.close()


def inpaint_vertex_colors(texture: torch.Tensor, mask: torch.Tensor, vtx_pos: torch.Tensor, vtx_uv: torch.Tensor, pos_idx: torch.Tensor,
                          uv_idx: torch.Tensor | None = None, *, flip_v: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Fills texels at the mesh's UV corners from the colours of neighbouring vertices: a vectorized, behaviour-compatible
    reimplementation of the "smooth" mode of Hunyuan3D's meshVerticeInpaint (written independently, no Tencent code).

    texture [H, W, C] (float or uint8) and mask [H, W] (> 0 = painted) are images whose rows run top to bottom; vertex
    positions vtx_pos [P, 3] with faces pos_idx [F, 3], uvs vtx_uv [T, 2] with faces uv_idx [F, 3] (default pos_idx).
    The texel of a uv is (round(v * (H - 1)), round(u * (W - 1))) with v = 1 - v when flip_v (the reference uses OpenGL
    uvs, i.e. flip_v=True; WanGP meshes use glTF uvs, flip_v=False).
    1. A vertex whose corner texel is painted takes its colour (the last such corner in face order).
    2. Vertices with an unpainted corner are then repeatedly set to the inverse-squared-distance weighted mean of their
       coloured neighbours (each corner's next corner in its face), until no more vertices can be reached, plus as many
       further smoothing passes as the reference's termination rule gives. Unlike the reference, which updates vertices
       in place one after the other, each pass updates all vertices at once (order independent), so colours can differ
       slightly where several passes propagate.
    3. Every corner of a coloured vertex writes its vertex colour into its texel (the last corner wins) and marks it in
       the mask (255, or True for a bool mask).
    Returns the new texture and mask on the input device."""
    device, (height, width, channels) = texture.device, texture.shape
    uv_idx = pos_idx if uv_idx is None else uv_idx
    uvs = vtx_uv.double()
    rows = torch.round(((1 - uvs[:, 1]) if flip_v else uvs[:, 1]) * (height - 1)).long()
    cols = torch.round(uvs[:, 0] * (width - 1)).long()
    corner_vertex, corner_uv = pos_idx.reshape(-1).long(), uv_idx.reshape(-1).long()
    texel = rows[corner_uv] * width + cols[corner_uv]
    painted = mask.reshape(-1)[texel] > 0
    order = torch.arange(len(texel), device=device)
    vertices = len(vtx_pos)
    flat = texture.reshape(-1, channels)

    last = torch.full((vertices,), -1, dtype=torch.int64, device=device).scatter_reduce(0, corner_vertex[painted], order[painted], "amax")
    colored = last >= 0
    color = torch.zeros(vertices, channels, dtype=torch.float32, device=device)
    color[colored] = flat[texel[last[colored]]].float()
    entries = torch.bincount(corner_vertex[~painted], minlength=vertices)  # corners waiting in the reference's list
    pending = entries > 0

    source, target = pos_idx.reshape(-1).long(), pos_idx.roll(-1, 1).reshape(-1).long()
    weight = (vtx_pos[source].double() - vtx_pos[target].double()).norm(dim=1).clamp_min(1e-4).pow(-2).float()
    budget, previous = 2, 0
    while budget > 0:
        active = weight * colored[target]
        total = torch.zeros(vertices, dtype=torch.float32, device=device).index_add_(0, source, active)
        sums = torch.zeros(vertices, channels, dtype=torch.float32, device=device).index_add_(0, source, active[:, None] * color[target])
        update = pending & (total > 0)
        color = torch.where(update[:, None], sums / total.clamp_min(1e-30)[:, None], color)
        colored |= update
        missing = int(entries[pending & ~update].sum())
        budget += -1 if missing == previous else 1
        previous = missing

    write = colored[corner_vertex]
    winner = torch.full((height * width,), -1, dtype=torch.int64, device=device).scatter_reduce(0, texel[write], order[write], "amax")
    written = winner >= 0
    new_texture, new_mask = texture.clone(), mask.clone()
    values = color[corner_vertex[winner[written]]]
    new_texture.view(-1, channels)[written] = values.round().clamp(0, 255).to(texture.dtype) if texture.dtype == torch.uint8 else values.to(texture.dtype)
    new_mask.view(-1)[written] = True if mask.dtype == torch.bool else 255
    return new_texture, new_mask


def inpaint_texture(texture: torch.Tensor, mask: torch.Tensor, *, radius: float = 3.0, method: str = "ns") -> torch.Tensor:
    """Fills the texels where mask > 0 with OpenCV inpainting (method "ns": Navier-Stokes, as Hunyuan3D; "telea"), on the
    CPU. texture [H, W, C] uint8 or float 0..1 (C = 1, 3 or 4; alpha is inpainted separately); returns the same dtype on
    the input device. Note: mask marks the texels to fill (OpenCV convention), i.e. the complement of a painted mask."""
    import cv2
    flags = {"ns": cv2.INPAINT_NS, "telea": cv2.INPAINT_TELEA}[method]
    array = texture.detach().cpu()
    array = (array if array.dtype == torch.uint8 else (array.float() * 255).round().clamp(0, 255).to(torch.uint8)).numpy()
    holes = (mask.detach().cpu().numpy() > 0).astype(np.uint8)
    groups = [array[..., :3], array[..., 3:]] if array.shape[2] == 4 else [array]
    filled = [cv2.inpaint(np.ascontiguousarray(g), holes, radius, flags).reshape(*g.shape) for g in groups]
    result = torch.from_numpy(np.concatenate(filled, 2)).to(texture.device)
    return result if texture.dtype == torch.uint8 else (result.float() / 255).to(texture.dtype)
