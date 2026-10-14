# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Gaussian splats in the INRIA 3D Gaussian Splatting PLY layout, point clouds as splats, and plain point-cloud PLYs.

Splat dict (float32 torch tensors on one device; `ply_to_splats` / `points_to_splats` also return "comments"):
- means [N, 3]: centres, glTF axes (+Y up, +Z front; see splats_opencv_to_gltf for OpenCV-frame outputs);
- scales [N, 3]: LINEAR standard deviations along the local axes, in world units (the PLY stores log(scale));
- rotations [N, 4]: unit quaternions (w, x, y, z), local-to-world;
- opacities [N]: LINEAR opacities in 0..1 (the PLY stores logit(opacity));
- sh_dc [N, 3]: degree-0 spherical-harmonic coefficients, colour = 0.5 + SH_C0 * sh_dc (sRGB-like display values);
- sh_rest [N, K, 3] or None: higher degrees, K = (degree + 1)^2 - 1, in the 3DGS basis order;
- sh_degree: int.

PLY layout written (INRIA, binary little-endian, every property float32): x y z nx ny nz f_dc_0..2 f_rest_0..(3K-1)
opacity scale_0..2 rot_0..3, f_rest stored channel-major (all K red coefficients, then green, then blue), normals zero,
and a `comment sh_degree N` header line. Our own numpy code (no plyfile).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .formats import _ply_header, _read_ply_ascii, _read_ply_element

SH_C0 = 0.28209479177387814
SPLAT_PROPERTIES = {"f_dc_0", "opacity", "scale_0", "rot_0"}
_OPACITY_EPSILON = 1e-6  # opacities are clamped to [eps, 1 - eps] before the logit
# Sign of each 3DGS SH basis function (degrees 1..3) under (x, y, z) -> (x, -y, -z), i.e. a 180 degree turn about X.
_SH_FLIP_SIGNS = [-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1]


def _array(value, dtype=np.float32) -> np.ndarray:
    return value.detach().to(device="cpu").numpy().astype(dtype, copy=False) if isinstance(value, torch.Tensor) else np.asarray(value, dtype)


def _header_text(data: bytes) -> tuple[list[str], int]:
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError("Not a PLY file")
    body = data.index(b"\n", end) + 1
    return data[:body].decode("utf-8").splitlines(), body


def _read_header(source) -> bytes:
    """The PLY header bytes (up to and including the end_header line) of a path or bytes, reading only what is needed."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source[:1 << 20])
        return data[:data.index(b"\n", data.index(b"end_header")) + 1] if b"end_header" in data else data
    data = b""
    with open(source, "rb") as handle:
        while b"end_header" not in data or b"\n" not in data[data.index(b"end_header"):]:
            chunk = handle.read(1 << 16)
            if not chunk:
                break
            data += chunk
    return data


def splats_to_ply(means, scales, rotations, opacities, sh_dc, sh_rest=None, comments=()) -> bytes:
    """INRIA 3DGS PLY bytes from linear scales / opacities (see the module docstring); torch tensors or numpy arrays.
    comments are extra header lines (`comment <text>`, one line each)."""
    means, scales, rotations, sh_dc = (_array(v) for v in (means, scales, rotations, sh_dc))
    opacities = np.clip(_array(opacities, np.float64).reshape(-1), _OPACITY_EPSILON, 1 - _OPACITY_EPSILON)
    rest = np.zeros((len(means), 0, 3), np.float32) if sh_rest is None else _array(sh_rest)
    degree = math.isqrt(rest.shape[1] + 1) - 1
    if (degree + 1) ** 2 - 1 != rest.shape[1]:
        raise ValueError(f"sh_rest has {rest.shape[1]} coefficients, not (degree + 1)^2 - 1")
    columns = [means, np.zeros_like(means), sh_dc, rest.transpose(0, 2, 1).reshape(len(means), -1),
               np.log(opacities / (1 - opacities))[:, None].astype(np.float32), np.log(np.maximum(scales, 1e-30)), rotations]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    for comment in comments:
        if "\n" in comment or "\r" in comment:
            raise ValueError("PLY comments must be single lines")
    header = ["ply", "format binary_little_endian 1.0", f"comment sh_degree {degree}"] + [f"comment {c}" for c in comments]
    header += [f"element vertex {len(means)}"] + [f"property float {n}" for n in names] + ["end_header"]
    body = np.ascontiguousarray(np.concatenate(columns, 1), dtype="<f4")
    return ("\n".join(header) + "\n").encode("utf-8") + body.tobytes()


def ply_to_splats(source, *, device="cpu") -> dict:
    """Splat dict (see the module docstring, with linear scales and opacities, normalized rotations) of an INRIA 3DGS
    PLY file (path or bytes; binary or ASCII); "comments" holds the header comment lines."""
    data = bytes(source) if isinstance(source, (bytes, bytearray, memoryview)) else Path(source).read_bytes()
    fmt, elements, comments, pos = _ply_header(data)
    vertex_element = next(e for e in elements if e["name"] == "vertex")
    if elements[0] is not vertex_element:
        raise ValueError("The vertex element must come first in a splat PLY")
    if fmt == "ascii":
        vertex = _read_ply_ascii(data[pos:].decode("ascii"), [vertex_element])["vertex"]
    else:
        vertex = _read_ply_element(data, pos, vertex_element, {"binary_little_endian": "<", "binary_big_endian": ">"}[fmt])[1]
    if not SPLAT_PROPERTIES <= set(vertex):
        raise ValueError("Not a Gaussian splat PLY")
    column = lambda *names: np.stack([np.asarray(vertex[n], np.float64) for n in names], 1)
    rest_names = sorted((n for n in vertex if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    count = len(vertex["x"])
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
    rotations = column("rot_0", "rot_1", "rot_2", "rot_3")
    splats = {"means": tensor(column("x", "y", "z")), "scales": tensor(np.exp(column("scale_0", "scale_1", "scale_2"))),
              "rotations": tensor(rotations / np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-30)),
              "opacities": tensor(1 / (1 + np.exp(-np.asarray(vertex["opacity"], np.float64)))), "sh_dc": tensor(column("f_dc_0", "f_dc_1", "f_dc_2")),
              "sh_rest": tensor(column(*rest_names).reshape(count, 3, -1).transpose(0, 2, 1)) if rest_names else None,
              "sh_degree": math.isqrt(len(rest_names) // 3 + 1) - 1, "comments": comments}
    return splats


def ply_comments(source) -> list[str]:
    """Text of the `comment` header lines of a PLY file (path or bytes), reading only the header."""
    lines, _ = _header_text(_read_header(source))
    return [line[8:] for line in lines if line.startswith("comment ")]


def ply_write_comments(data: bytes, comments) -> bytes:
    """PLY bytes with the given comment lines added to the header. An existing comment whose first word is the first
    word of a new comment is replaced (so rewriting "wangp {...}" replaces the previous settings and keeps
    "sh_degree 0"). Only the header changes; the data that follows is kept byte-identical."""
    lines, body = _header_text(bytes(data[:data.index(b"\n", data.index(b"end_header")) + 1]))
    for comment in comments:
        if "\n" in comment or "\r" in comment:
            raise ValueError("PLY comments must be single lines")
    keys = {c.split(" ", 1)[0] for c in comments}
    kept = [line for line in lines if not (line.startswith("comment ") and line[8:].split(" ", 1)[0] in keys)]
    at = max([i for i, line in enumerate(kept) if line.startswith(("format ", "comment "))]) + 1
    kept[at:at] = [f"comment {c}" for c in comments]
    return ("\n".join(kept) + "\n").encode("utf-8") + bytes(data[body:])


def is_splat_ply(source) -> bool:
    """True for a PLY whose vertex element has the 3DGS properties (f_dc_0, opacity, scale_0, rot_0); header sniff only."""
    try:
        lines, _ = _header_text(_read_header(source))
    except (OSError, ValueError):
        return False
    names, in_vertex = set(), False
    for line in lines:
        words = line.split()
        if words[:1] == ["element"]:
            in_vertex = words[1:2] == ["vertex"]
        elif words[:1] == ["property"] and in_vertex:
            names.add(words[-1])
    return SPLAT_PROPERTIES <= names


def splat_file_stats(source) -> dict:
    """Header-only statistics of a 3DGS splat PLY (path or bytes): {"format": "ply", "gaussians": N, "sh_degree": d
    (from the `comment sh_degree N` line, else from the f_rest_* count), "file_size": bytes}. Raises ValueError for a
    file that is not a splat PLY."""
    lines, _ = _header_text(_read_header(source))
    count, names, in_vertex, degree = 0, set(), False, None
    for line in lines:
        words = line.split()
        if words[:1] == ["element"]:
            in_vertex = words[1:2] == ["vertex"]
            count = int(words[2]) if in_vertex else count
        elif words[:1] == ["property"] and in_vertex:
            names.add(words[-1])
        elif words[:2] == ["comment", "sh_degree"] and len(words) > 2:
            degree = int(words[2])
    if not SPLAT_PROPERTIES <= names:
        raise ValueError("Not a Gaussian splat PLY")
    if degree is None:
        degree = math.isqrt(sum(n.startswith("f_rest_") for n in names) // 3 + 1) - 1
    size = len(source) if isinstance(source, (bytes, bytearray, memoryview)) else Path(source).stat().st_size
    return {"format": "ply", "gaussians": count, "sh_degree": degree, "file_size": size}


def points_to_splats(points: torch.Tensor, colors: torch.Tensor, scale: float | None = None, *, opacity: float = 0.99) -> dict:
    """Splat dict of small isotropic Gaussians for a point cloud: colors [N, 3] in 0..1 (sRGB) become sh_dc (degree 0),
    rotations identity, opacity `opacity`. scale None estimates each point's scale as the 3DGS initialisation does: the
    root mean square distance to its 3 nearest neighbours (k-d tree on the CPU), at least 1e-4 of the bounding box
    diagonal and at most the 98th percentile of these distances; otherwise every point gets `scale`. Tensors are on the device of points."""
    from scipy.spatial import cKDTree
    device, count = points.device, len(points)
    if scale is None:
        if count < 2:
            raise ValueError("Estimating splat scales needs at least two points")
        array = _array(points, np.float64)
        distances, _ = cKDTree(array).query(array, k=min(4, count), workers=-1)
        sigma = np.sqrt((distances[:, 1:] ** 2).mean(1))
        sigma = np.maximum(sigma, 1e-4 * np.linalg.norm(array.max(0) - array.min(0)))
        sigma = np.minimum(sigma, np.percentile(sigma, 98))  # isolated points would otherwise become large blobs
        scales = torch.from_numpy(np.repeat(sigma[:, None], 3, 1).astype(np.float32)).to(device)
    else:
        scales = torch.full((count, 3), float(scale), dtype=torch.float32, device=device)
    rotations = torch.zeros(count, 4, dtype=torch.float32, device=device)
    rotations[:, 0] = 1
    return {"means": points.float(), "scales": scales, "rotations": rotations, "opacities": torch.full((count,), opacity, dtype=torch.float32, device=device),
            "sh_dc": (colors.float() - 0.5) / SH_C0, "sh_rest": None, "sh_degree": 0, "comments": []}


def splats_opencv_to_gltf(splats: dict) -> dict:
    """Splats converted from OpenCV axes (+X right, +Y down, +Z forward) to the glTF axes of the 3D gallery (+Y up,
    +Z towards the viewer): a 180 degree turn about X applied to means, rotations and SH coefficients (exact)."""
    w, x, y, z = splats["rotations"].unbind(1)
    converted = dict(splats, means=splats["means"] * splats["means"].new_tensor([1.0, -1.0, -1.0]), rotations=torch.stack([-x, w, -z, y], 1))
    if splats["sh_rest"] is not None:
        signs = splats["sh_rest"].new_tensor(_SH_FLIP_SIGNS[:splats["sh_rest"].shape[1]])
        converted["sh_rest"] = splats["sh_rest"] * signs[None, :, None]
    return converted


def write_points_ply(points, colors) -> bytes:
    """Standard point-cloud PLY (binary little-endian): float x y z and uchar red green blue from colors in 0..1."""
    points, colors = _array(points), np.rint(np.clip(_array(colors), 0, 1) * 255).astype(np.uint8)
    records = np.empty(len(points), np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
    for i, axis in enumerate("xyz"):
        records[axis] = points[:, i]
    for i, channel in enumerate(("red", "green", "blue")):
        records[channel] = colors[:, i]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}", "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    return ("\n".join(header) + "\n").encode("ascii") + records.tobytes()
