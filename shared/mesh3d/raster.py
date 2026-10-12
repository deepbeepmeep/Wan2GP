# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Exact triangle rasterizer in pure PyTorch (any device), with a z-buffer, perspective-correct barycentrics and UV-space
rasterization for texture baking.

Conventions (those of nvdiffrast, also followed by Hunyuan3D's custom_rasterizer):
- Inputs are clip-space vertices (x, y, z, w) after the projection matrix. Pixel (row r, column c) of an H x W image is the
  sample at NDC x = (2c + 1) / W - 1, y = (2r + 1) / H - 1: row 0 is NDC y = -1, so images of a camera with +Y up come
  out upside down (render.py flips them). custom_rasterizer puts NDC -1 / +1 on the first / last pixel centres instead:
  scale clip x by (W - 1) / W and clip y by (H - 1) / H to reproduce it.
- A pixel is covered by a face when its sample lies inside the triangle or on its edges and its depth z / w is within
  [-1, 1] (near / far planes). The nearest z / w wins; equal depths keep the smallest face id. There is no culling of
  back faces.
- Coverage uses the 2D homogeneous edge functions of Olano & Greer (1997) in float64: triangles crossing the camera plane
  (w <= 0) are rasterized correctly without clipping, and the edge functions of a shared edge are exact opposites, so
  neighbouring faces leave no cracks and overlap only on samples exactly on their common edge.
- Outputs: face_id int32 (-1 where empty), barycentrics float32 (weights of the face's corners 0, 1, 2, perspective
  correct, summing to 1, zero where empty) and depth float32 (z / w of the visible surface, zero where empty).

Work is bounded: faces are set up in chunks, and the pixels of their bounding boxes are enumerated in chunks of
`chunk_size` candidates (about 150 bytes each, so ~150 MiB at the default 1M); the z-buffer is one int64 per pixel (min of packed (depth, face id)).
"""
from __future__ import annotations

import torch

EMPTY = torch.iinfo(torch.int64).max
FACE_CHUNK = 1 << 17


def _resolution(resolution) -> tuple[int, int]:
    return (resolution, resolution) if isinstance(resolution, int) else (int(resolution[0]), int(resolution[1]))


def _edge_planes(corners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """corners [N, 3, 4] clip coordinates -> edge functions n [N, 3, 3] (coefficients of x, y, 1 in NDC; n_i vanishes on
    the edge opposite corner i) and det [N] (n_i . corner_i)."""
    x, y, w = corners[..., 0], corners[..., 1], corners[..., 3]
    x1, y1, w1, x2, y2, w2 = x.roll(-1, 1), y.roll(-1, 1), w.roll(-1, 1), x.roll(-2, 1), y.roll(-2, 1), w.roll(-2, 1)
    n = torch.stack([y1 * w2 - w1 * y2, w1 * x2 - x1 * w2, x1 * y2 - y1 * x2], -1)
    return n, n[:, 0, 0] * x[:, 0] + n[:, 0, 1] * y[:, 0] + n[:, 0, 2] * w[:, 0]


def _pixel_centers(pixels: torch.Tensor, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = pixels.div(width, rounding_mode="floor"), pixels % width
    return (2 * cols + 1).double() / width - 1, (2 * rows + 1).double() / height - 1


def _depth_keys(depth: torch.Tensor) -> torch.Tensor:
    """float32 depths -> int64 keys whose high 32 bits sort like the depths (-0 counts as +0)."""
    bits = (depth + 0.0).view(torch.int32)
    return torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64) << 32


def _setup(clip: torch.Tensor, faces: torch.Tensor, width: int, height: int):
    """Face ids kept, their [n, 4, 3] planes (3 oriented edge functions + depth z/w, as coefficients of NDC x, y, 1) and
    their pixel bounding boxes (first and last (column, row))."""
    corners = clip[faces].double()
    x, y, z, w = corners.unbind(-1)
    n, det = _edge_planes(corners)
    keep = (det != 0) & (w > 0).any(1)
    for value, bound in ((x, w), (-x, w), (y, w), (-y, w), (z, w), (-z, w)):  # outside one frustum plane
        keep &= ~(value > bound).all(1)
    ids = keep.nonzero().squeeze(1)
    corners, n, det, z = corners[ids], n[ids], det[ids], z[ids]
    planes = torch.cat([n * det.sign()[:, None, None], (torch.einsum("fi,fic->fc", z, n) / det[:, None])[:, None]], 1)

    # Bounding box of the triangle clipped by the near plane (z + w >= 0): kept corners and edge crossings, projected.
    near = corners[..., 2] + corners[..., 3]
    following, near_following = corners.roll(-1, 1), near.roll(-1, 1)
    crossing = (near >= 0) != (near_following >= 0)
    t = (near / torch.where(crossing, near - near_following, 1.0))[..., None]
    points = torch.cat([corners, corners + t * (following - corners)], 1)
    valid = torch.cat([near >= 0, crossing], 1)
    pw = points[..., 3]
    unbounded = (valid & (pw <= 0)).any(1)
    valid &= pw > 0
    ndc = points[..., :2] / torch.where(valid, pw, 1.0)[..., None]
    big = torch.tensor(float("inf"), dtype=torch.float64, device=clip.device)
    low = torch.where(valid[..., None], ndc, big).amin(1)
    high = torch.where(valid[..., None], ndc, -big).amax(1)
    low = torch.where(unbounded[:, None], -big, low)
    high = torch.where(unbounded[:, None], big, high)
    size = torch.tensor([width, height], dtype=torch.float64, device=clip.device)
    first = torch.ceil(((low + 1) * size - 1) / 2 - 1e-6).clamp(0, max(width, height)).long()
    last = torch.minimum(torch.floor(((high + 1) * size - 1) / 2 + 1e-6).clamp(-1, max(width, height)), size - 1).long()
    inside = (last >= first).all(1)
    return ids[inside], planes[inside], first[inside], last[inside]


def _rasterize_view(clip: torch.Tensor, faces: torch.Tensor, height: int, width: int, chunk_size: int, face_id, bary, depth):
    """Rasterizes one view into the flat outputs face_id [H * W], bary [H * W, 3] and depth [H * W]."""
    device = clip.device
    if len(faces) == 0:
        face_id.fill_(-1), bary.zero_(), depth.zero_()
        return
    zbuffer = torch.full((height * width,), EMPTY, dtype=torch.int64, device=device)
    for start in range(0, len(faces), FACE_CHUNK):
        ids, planes, first, last = _setup(clip, faces[start:start + FACE_CHUNK], width, height)
        if len(ids) == 0:
            continue
        ids += start
        planes = [planes[:, j].contiguous() for j in range(4)]
        box_width = last[:, 0] - first[:, 0] + 1
        counts = box_width * (last[:, 1] - first[:, 1] + 1)
        ends = counts.cumsum(0)
        starts, total = ends - counts, int(ends[-1])
        for offset in range(0, total, chunk_size):
            # Candidate pixels of the bounding boxes; misses scatter EMPTY keys (no compaction, no host sync).
            face = torch.searchsorted(ends, torch.arange(offset, min(offset + chunk_size, total), device=device), right=True)
            local = torch.arange(offset, offset + len(face), device=device) - starts[face]
            row = local.div(box_width[face], rounding_mode="floor")
            col = local - row * box_width[face] + first[face, 0]
            row += first[face, 1]
            x = (2 * col + 1).double() / width - 1
            y = (2 * row + 1).double() / height - 1
            hit = None
            for plane in planes[:3]:
                coefficients = plane[face]
                inside = coefficients[:, 0] * x + coefficients[:, 1] * y + coefficients[:, 2] >= 0
                hit = inside if hit is None else hit & inside
            coefficients = planes[3][face]
            z = (coefficients[:, 0] * x + coefficients[:, 1] * y + coefficients[:, 2]).float()
            keys = torch.where(hit & (z.abs() <= 1), _depth_keys(z) | ids[face], EMPTY)
            zbuffer.scatter_reduce_(0, row * width + col, keys, "amin")
    step = max(chunk_size // 4, 1)
    for offset in range(0, height * width, step):  # dense pass over the pixels, empty ones masked
        ids = zbuffer[offset:offset + step]
        covered = ids != EMPTY
        ids = torch.where(covered, ids & 0xFFFFFFFF, 0)
        corners = clip[faces[ids]].double()
        n, _ = _edge_planes(corners)
        x, y = _pixel_centers(torch.arange(offset, offset + len(ids), device=device), width, height)
        edges = n[..., 0] * x[:, None] + n[..., 1] * y[:, None] + n[..., 2]
        weights = edges / edges.sum(1, keepdim=True)
        face_id[offset:offset + len(ids)] = torch.where(covered, ids, -1)
        bary[offset:offset + len(ids)] = torch.where(covered[:, None], weights, 0.0)
        depth[offset:offset + len(ids)] = torch.where(covered, (weights * corners[..., 2]).sum(1) / (weights * corners[..., 3]).sum(1), 0.0)


def rasterize(clip_vertices: torch.Tensor, faces: torch.Tensor, resolution, *, chunk_size: int = 1 << 20):
    """Rasterizes triangles given in clip space (see the module docstring for the conventions).

    clip_vertices: [V, 4] or [B, V, 4] (one view per batch item, shared faces); faces [F, 3] integer; resolution: int or
    (height, width). Returns face_id [B, H, W] int32 (-1 = empty), bary [B, H, W, 3] float32 and depth [B, H, W] float32
    (z / w), on the device of clip_vertices (faces must be on the same device)."""
    height, width = _resolution(resolution)
    views = clip_vertices[None] if clip_vertices.ndim == 2 else clip_vertices
    device, faces = views.device, faces.long()
    face_id = torch.empty(len(views), height, width, dtype=torch.int32, device=device)
    bary = torch.empty(len(views), height, width, 3, dtype=torch.float32, device=device)
    depth = torch.empty(len(views), height, width, dtype=torch.float32, device=device)
    for b, view in enumerate(views):
        _rasterize_view(view, faces, height, width, chunk_size, face_id[b].view(-1), bary[b].view(-1, 3), depth[b].view(-1))
    return face_id, bary, depth


def interpolate(attributes: torch.Tensor, faces: torch.Tensor, face_id: torch.Tensor, bary: torch.Tensor, *, chunk_size: int = 1 << 20) -> torch.Tensor:
    """Per-pixel attributes sum_k bary[k] * attributes[faces[face_id, k]], zero where face_id < 0.

    attributes [V, C] (shared) or [B, V, C] (per view); face_id [B, H, W] with bary [B, H, W, 3], or unbatched [H, W] with
    [H, W, 3] (then the result is [H, W, C]). Computed in float32 (float64 attributes stay float64)."""
    unbatched = face_id.ndim == 2
    if unbatched:
        face_id, bary = face_id[None], bary[None]
    dtype = torch.promote_types(attributes.dtype, torch.float32)
    out = torch.zeros(*face_id.shape, attributes.shape[-1], dtype=dtype, device=face_id.device)
    faces = faces.long()
    for b in range(face_id.shape[0]):
        values = attributes[b] if attributes.ndim == 3 else attributes
        ids, weights, target = face_id[b].reshape(-1), bary[b].reshape(-1, 3), out[b].view(-1, out.shape[-1])
        covered = (ids >= 0).nonzero().squeeze(1)
        for offset in range(0, len(covered), chunk_size):
            pixels = covered[offset:offset + chunk_size]
            corners = values[faces[ids[pixels].long()]].to(dtype)
            target[pixels] = (weights[pixels].to(dtype)[..., None] * corners).sum(1)
    return out[0] if unbatched else out


def rasterize_uv(uvs: torch.Tensor, faces: torch.Tensor, resolution, *, chunk_size: int = 1 << 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Texture-space rasterization for baking: texel [r, c] of an (H, W) texture is the sample at uv ((c + 0.5) / W,
    (r + 0.5) / H) (glTF uvs, v downwards, see mesh.py). Returns face_id [H, W] int32 (-1 = no face) and bary [H, W, 3];
    pass them to interpolate() with any per-vertex attribute (positions, normals, colours). Seams must be duplicated
    vertices (one index buffer); overlapping charts keep the smallest face id."""
    clip = torch.cat([uvs.float() * 2 - 1, torch.zeros_like(uvs[:, :1], dtype=torch.float32), torch.ones_like(uvs[:, :1], dtype=torch.float32)], 1)
    face_id, bary, _ = rasterize(clip, faces, resolution, chunk_size=chunk_size)
    return face_id[0], bary[0]
