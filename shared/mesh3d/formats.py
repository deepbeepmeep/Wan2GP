# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""OBJ (+ MTL + textures), PLY and STL readers, writers and quick statistics, in Mesh conventions (see mesh.py).

OBJ and PLY texture coordinates have v upwards (OpenGL); they are converted from / to the glTF convention (v' = 1 - v).
Vertex colours are sRGB in all three formats. Readers create their tensors on the `device` they are given.
"""
from __future__ import annotations

import os
import re
import struct
from pathlib import Path

import numpy as np
import torch

from .images import decode_image, encode_image, file_reader, image_size, pack_textures, remap_uvs, to_channels
from .mesh import Mesh, linear_to_srgb, srgb_to_linear, vertex_normals


def unique_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows of an [N, K] array (sorted) and, for each input row, the index of its unique row."""
    if len(rows) == 0:
        return rows, np.zeros(0, np.int64)
    order = np.lexsort(rows.T[::-1])
    ordered = rows[order]
    first = np.ones(len(rows), bool)
    first[1:] = (ordered[1:] != ordered[:-1]).any(1)
    inverse = np.empty(len(rows), np.int64)
    inverse[order] = np.cumsum(first) - 1
    return ordered[first], inverse


def _fan(counts: np.ndarray) -> np.ndarray:
    """[T, 3] corner indices of the fan triangulation of polygons with `counts` corners stored one after the other."""
    counts = np.asarray(counts, np.int64)
    triangles = np.maximum(counts - 2, 0)
    first = np.repeat(np.cumsum(counts) - counts, triangles)
    step = np.arange(triangles.sum()) - np.repeat(np.cumsum(triangles) - triangles, triangles) + 1
    return np.stack([first, first + step, first + step + 1], 1)


def _to_mesh(device, vertices, faces, normals=None, uvs=None, colors=None, **textures) -> Mesh:
    tensor = lambda a, dtype: None if a is None else torch.from_numpy(np.ascontiguousarray(a, dtype=dtype)).to(device)
    return Mesh(vertices=tensor(vertices, np.float32), faces=tensor(faces, np.int64), normals=tensor(normals, np.float32), uvs=tensor(uvs, np.float32),
                colors=tensor(colors, np.float32), **{k: tensor(v, np.uint8) for k, v in textures.items()})


def _texture_file(path: Path) -> np.ndarray:
    return decode_image(path.read_bytes())


def _base_color_array(array: np.ndarray) -> np.ndarray:
    array = to_channels(array, 4)
    return np.ascontiguousarray(array[..., :3]) if (array[..., 3] == 255).all() else array


def _sample(texture: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Bilinear sample (repeat wrap) of a uint8 [H, W, C] texture at glTF uvs: float 0..1 [N, C]."""
    grid = torch.from_numpy(np.ascontiguousarray((uvs - np.floor(uvs)) * 2 - 1, dtype=np.float32)).view(1, 1, -1, 2)
    image = torch.from_numpy(np.ascontiguousarray(texture)).permute(2, 0, 1)[None].float() / 255
    return torch.nn.functional.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T.numpy()


def _numpy(tensor, dtype=None) -> np.ndarray:
    array = tensor.detach().to(device="cpu").numpy()
    return array if dtype is None else array.astype(dtype, copy=False)


# ------------------------------------------------------------------------------------------------------------ OBJ
_MAP_OPTIONS = {"-blendu", "-blendv", "-bm", "-boost", "-cc", "-clamp", "-imfchan", "-mm", "-o", "-s", "-t", "-texres", "-type"}


def _map_file(arguments: str) -> str:
    """File name of an MTL map statement, skipping its options (which take one to three values)."""
    tokens, i = arguments.split(), 0
    while i < len(tokens) and tokens[i] in _MAP_OPTIONS:
        i += 1
        if i < len(tokens) and not re.fullmatch(r"[-+]?[\d.]+(e[-+]?\d+)?", tokens[i], re.I):
            i += 1  # on/off or a channel name
            continue
        for _ in range(3):
            if i < len(tokens) - 1 and re.fullmatch(r"[-+]?[\d.]+(e[-+]?\d+)?", tokens[i], re.I):
                i += 1
    return " ".join(tokens[i:])


def _read_mtl(path: Path) -> dict[str, dict]:
    materials, current = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        key = parts[0].lower()
        if key == "newmtl":
            current = materials.setdefault(parts[1].strip(), {})
        elif current is not None and key == "kd":
            current["kd"] = [float(x) for x in parts[1].split()[:3]]
        elif current is not None and key in ("map_kd", "map_pr", "map_pm", "norm"):
            current[key] = _map_file(parts[1])
    return materials


def _float_rows(lines: list[str]) -> np.ndarray:
    rows = [line.split()[1:] for line in lines]
    width = max(map(len, rows), default=0)
    if any(len(r) != width for r in rows):
        rows = [r + ["nan"] * (width - len(r)) for r in rows]
    return np.array(rows, np.float64).reshape(len(rows), width)


def _obj_corners(face_lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Corner (v, vt, vn) indices as written (0 = absent) and the corner count of each polygon."""
    tokens = [line.split()[1:] for line in face_lines]
    counts = np.fromiter(map(len, tokens), np.int64, len(tokens))
    flat = [t for face in tokens for t in face]
    joined = " ".join(flat)
    slashes = flat[0].count("/") if flat else 0
    doubles = "//" in flat[0] if flat else False
    if joined.count("/") == slashes * len(flat) and joined.count("//") == doubles * len(flat):
        numbers = np.array(joined.replace("//", "/0/").replace("/", " ").split(), np.int64).reshape(len(flat), slashes + 1)
    else:
        parts = [(t.split("/") + ["", ""])[:3] for t in flat]
        numbers = np.array([[int(p) if p else 0 for p in row] for row in parts], np.int64).reshape(len(flat), 3)
    return np.pad(numbers, ((0, 0), (0, 3 - numbers.shape[1]))), counts


def read_obj(path, *, device="cpu") -> Mesh:
    """Mesh from an OBJ file. Polygons are fan-triangulated; vertices are split per distinct (v, vt, vn) corner; "v x y z
    r g b" colours (0..1, sRGB) are read. One material with maps keeps its textures (map_Kd, map_Pr + map_Pm packed into
    metallic_roughness, norm; its Kd is ignored, the texture replacing it). With several materials, their map_Kd images
    are packed into one atlas (uvs remapped, other maps dropped) and the Kd of untextured materials becomes vertex
    colours (Kd read as linear)."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").replace("\t", " ").splitlines()
    v_lines, vt_lines, vn_lines, f_lines, face_materials, material_names, libraries = [], [], [], [], [], {}, []
    current = -1
    for line in lines:
        if line.startswith("v "):
            v_lines.append(line)
        elif line.startswith("vt "):
            vt_lines.append(line)
        elif line.startswith("vn "):
            vn_lines.append(line)
        elif line.startswith("f "):
            f_lines.append(line)
            face_materials.append(current)
        elif line.startswith("usemtl"):
            current = material_names.setdefault(line[6:].strip(), len(material_names))
        elif line.startswith("mtllib"):
            libraries.append(line[6:].strip())
    positions = _float_rows(v_lines)
    texcoords = _float_rows(vt_lines)[:, :2] if vt_lines else None
    normals = _float_rows(vn_lines)[:, :3] if vn_lines else None
    corners, counts = _obj_corners(f_lines)
    if (corners < 0).any():  # negative indices are relative to the elements read so far
        seen, totals = np.zeros((len(f_lines), 3), np.int64), [0, 0, 0]
        face = 0
        for line in lines:
            if line.startswith(("v ", "vt ", "vn ")):
                totals[{"v ": 0, "vt": 1, "vn": 2}[line[:2]]] += 1
            elif line.startswith("f "):
                seen[face] = totals
                face += 1
        relative = np.repeat(seen, counts, axis=0)
        corners = np.where(corners < 0, relative + corners + 1, corners)
    corners = corners - 1  # 0-based, -1 = absent

    materials = {}
    for library in libraries:
        if (path.parent / library).is_file():
            materials |= _read_mtl(path.parent / library)
    names = {i: n for n, i in material_names.items()}
    face_materials = np.array(face_materials, np.int64)
    used = np.unique(face_materials)
    maps = [materials.get(names.get(i), {}) for i in used]
    keep_maps = len(used) == 1 and texcoords is not None and any(k in maps[0] for k in ("map_kd", "map_pr", "map_pm", "norm"))
    split = len(used) > 1  # several materials: vertices split per material (own atlas box and colour)

    triangles = _fan(counts)
    keys = np.concatenate([corners, np.repeat(face_materials, counts)[:, None]], 1) if split else corners
    unique, inverse = unique_rows(keys)
    vertices = positions[unique[:, 0], :3]
    colors = positions[unique[:, 0], 3:6].clip(0, 1) if positions.shape[1] >= 6 else None
    uvs = None if texcoords is None or (unique[:, 1] < 0).any() else texcoords[unique[:, 1]] * [1, -1] + [0, 1]
    vertex_normals_ = None if normals is None or (unique[:, 2] < 0).any() else normals[unique[:, 2]]
    textures = {}
    if keep_maps:
        found = {k: _texture_file(path.parent / f) for k, f in maps[0].items() if k != "kd" and (path.parent / f).is_file()}
        if "map_kd" in found:
            textures["base_color"] = _base_color_array(found["map_kd"])
        if "map_pr" in found or "map_pm" in found:
            shape = (found.get("map_pr", found.get("map_pm"))).shape[:2]
            roughness = to_channels(found["map_pr"], 1)[..., 0] if "map_pr" in found else np.full(shape, 255, np.uint8)
            metallic = to_channels(found["map_pm"], 1)[..., 0] if "map_pm" in found else np.zeros(shape, np.uint8)
            textures["metallic_roughness"] = np.stack([np.zeros_like(roughness), roughness, metallic], 2)
        if "norm" in found:
            textures["normal_map"] = to_channels(found["norm"], 3)
    else:  # map_Kd of several materials packed into one atlas; Kd (linear) of untextured materials into vertex colours
        material_of = unique[:, 3] if split else np.full(len(unique), used[0])
        files = {i: path.parent / m["map_kd"] for i, m in zip(used, maps) if "map_kd" in m and uvs is not None and (path.parent / m["map_kd"]).is_file()}
        linear = np.ones((len(unique), 3)) if colors is None else srgb_to_linear(torch.from_numpy(colors)).numpy()
        for index, material in zip(used, maps):
            if index not in files:
                linear[material_of == index] *= material.get("kd", [1.0, 1.0, 1.0])
        if files:
            atlas, boxes = pack_textures([to_channels(_texture_file(f), 4) for f in files.values()] + [np.full((4, 4, 4), 255, np.uint8)])
            packed = remap_uvs(np.full((len(unique), 2), 0.5), boxes[-1], atlas.shape)
            for box, index in zip(boxes, files):
                rows = material_of == index
                packed[rows] = remap_uvs(uvs[rows], box, atlas.shape)
            uvs, textures["base_color"] = packed, _base_color_array(atlas)
        colors = None if (linear == 1).all() else linear_to_srgb(torch.from_numpy(linear.clip(0, 1))).numpy()
    return _to_mesh(device, vertices, inverse[triangles], vertex_normals_, uvs, colors, **textures)


def _format_rows(line: str, array: np.ndarray) -> str:
    """One formatted line per row of array (line holds one placeholder per column)."""
    return (line * len(array)) % tuple(array.ravel().tolist()) if len(array) else ""


def write_obj(mesh: Mesh, name: str = "mesh") -> dict[str, bytes]:
    """OBJ + MTL (+ PNG textures): positions, uvs (v = 1 - glTF v) and normals are each welded (exact values) with their
    own indices; vertex colours are appended to the "v" lines; metallic_roughness is split into map_Pr / map_Pm images
    and the normal map is referenced by "norm" and by "map_Bump -bm 1" (read as a normal map by Blender)."""
    faces = _numpy(mesh.faces, np.int64)
    positions, position_index = unique_rows(_numpy(mesh.vertices, np.float32))
    columns = [positions]
    if mesh.colors is not None:
        first = np.unique(position_index, return_index=True)[1]  # first vertex of each position
        columns.append(_numpy(mesh.colors[:, :3], np.float32)[first])
    values = np.concatenate(columns, 1)
    text = [f"# WanGP mesh3d\nmtllib {name}.mtl\no {name}\n", _format_rows("v" + " %.9g" * values.shape[1] + "\n", values)]
    corner = [position_index[faces] + 1]
    if mesh.uvs is not None:
        uvs, uv_index = unique_rows(_numpy(mesh.uvs, np.float32) * np.float32([1, -1]) + np.float32([0, 1]))
        text.append(_format_rows("vt %.9g %.9g\n", uvs))
        corner.append(uv_index[faces] + 1)
    normals, normal_index = unique_rows(_numpy(vertex_normals(mesh.vertices, mesh.faces) if mesh.normals is None else mesh.normals, np.float32))
    text.append(_format_rows("vn %.9g %.9g %.9g\n", normals))
    corner.append(normal_index[faces] + 1)
    text.append("usemtl material\n")
    line = "f %d//%d %d//%d %d//%d\n" if mesh.uvs is None else "f %d/%d/%d %d/%d/%d %d/%d/%d\n"
    text.append(_format_rows(line, np.stack(corner, 2).reshape(len(faces), -1)))
    files = {f"{name}.obj": "".join(text).encode("utf-8")}
    mtl = ["# WanGP mesh3d", "newmtl material", "Ka 0 0 0", "Kd 1 1 1", "Ks 0 0 0", "d 1", "illum 1"]
    if mesh.base_color is not None:
        files[f"{name}_basecolor.png"] = encode_image(_numpy(mesh.base_color))
        mtl.append(f"map_Kd {name}_basecolor.png")
    if mesh.metallic_roughness is not None:
        mr = _numpy(mesh.metallic_roughness)
        files[f"{name}_roughness.png"], files[f"{name}_metallic.png"] = encode_image(mr[..., 1]), encode_image(mr[..., 2])
        mtl += [f"map_Pr {name}_roughness.png", f"map_Pm {name}_metallic.png"]
    if mesh.normal_map is not None:
        files[f"{name}_normal.png"] = encode_image(_numpy(mesh.normal_map))
        mtl += [f"norm {name}_normal.png", f"map_Bump -bm 1 {name}_normal.png"]
    files[f"{name}.mtl"] = ("\n".join(mtl) + "\n").encode("utf-8")
    return files


def obj_stats(path: str) -> dict:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").replace("\t", " ").splitlines()
    v_lines = [line for line in lines if line.startswith("v ")]
    positions = _float_rows(v_lines)
    triangles = sum(max(len(line.split()) - 3, 0) for line in lines if line.startswith("f "))
    materials, textures = {}, []
    for line in lines:
        if line.startswith("mtllib") and (path.parent / line[6:].strip()).is_file():
            materials |= _read_mtl(path.parent / line[6:].strip())
    for material in materials.values():
        for key in ("map_kd", "map_pr", "map_pm", "norm"):
            if key in material and (path.parent / material[key]).is_file():
                with open(path.parent / material[key], "rb") as handle:
                    textures.append(image_size(file_reader(handle)))
    used = {line[6:].strip() for line in lines if line.startswith("usemtl")}
    return {"format": "obj", "vertices": len(v_lines), "faces": triangles, "materials": len(materials) or len(used), "textures": textures,
            "bbox": [positions[:, :3].min(0).tolist(), positions[:, :3].max(0).tolist()] if len(positions) else None,
            "has_uv": any(line.startswith("vt ") for line in lines), "has_colors": positions.shape[1] >= 6, "animations": 0, "animation": None, "splats": 0, "file_size": os.path.getsize(path)}


# ------------------------------------------------------------------------------------------------------------ PLY
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _ply_header(data: bytes) -> tuple[str, list[dict], list[str], int]:
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError("Not a PLY file")
    body = data.index(b"\n", end) + 1
    fmt, elements, comments = None, [], []
    for line in data[:end].decode("utf-8", errors="replace").splitlines()[1:]:
        words = line.split()
        if not words:
            continue
        if words[0] == "format":
            fmt = words[1]
        elif words[0] in ("comment", "obj_info"):
            comments.append(line.split(None, 1)[1] if len(words) > 1 else "")
        elif words[0] == "element":
            elements.append({"name": words[1], "count": int(words[2]), "properties": []})
        elif words[0] == "property":
            prop = (words[4], _PLY_TYPES[words[2]], _PLY_TYPES[words[3]]) if words[1] == "list" else (words[2], _PLY_TYPES[words[1]], None)
            elements[-1]["properties"].append(prop)
    return fmt, elements, comments, body


def _read_ply_element(data, pos: int, element: dict, endian: str) -> tuple[int, dict]:
    """Reads one binary element: {property: array} (lists: (flat values, counts)) and the position after it."""
    props, count = element["properties"], element["count"]
    if all(p[2] is None for p in props):
        dtype = np.dtype([(p[0], endian + p[1]) for p in props])
        array = np.frombuffer(data, dtype, count, pos)
        return pos + count * dtype.itemsize, {p[0]: array[p[0]] for p in props}
    # Fast path: every list has the length found in the first record.
    first, cursor = [], pos
    for name, kind, item in props:
        if item is None:
            cursor += np.dtype(kind).itemsize
        else:
            n = int(np.frombuffer(data, endian + kind, 1, cursor)[0])
            first.append(n)
            cursor += np.dtype(kind).itemsize + n * np.dtype(item).itemsize
    fields, lists = [], iter(first)
    for name, kind, item in props:
        fields += [(name, endian + kind)] if item is None else [(name + "#count", endian + kind), (name, endian + item, (next(lists),))]
    dtype = np.dtype(fields)
    if count * dtype.itemsize <= len(data) - pos:
        array = np.frombuffer(data, dtype, count, pos)
        if all((array[p[0] + "#count"] == n).all() for p, n in zip([p for p in props if p[2] is not None], first)):
            return _uniform_result(array, props, first, count, pos + count * dtype.itemsize)
    # General path: walk the records to find the list starts, then gather the items.
    starts = {p[0]: np.empty(count, np.int64) for p in props}
    lengths = {p[0]: np.zeros(count, np.int64) for p in props if p[2] is not None}
    sizes = [(p[0], np.dtype(p[1]).itemsize, None if p[2] is None else np.dtype(p[2]).itemsize, endian + p[1]) for p in props]
    for record in range(count):
        for name, size, item_size, kind in sizes:
            if item_size is None:
                starts[name][record] = pos
                pos += size
            else:
                n = int(np.frombuffer(data, kind, 1, pos)[0])
                starts[name][record], lengths[name][record] = pos + size, n
                pos += size + n * item_size
    raw = np.frombuffer(data, np.uint8)
    result = {}
    for name, size, item_size, kind in sizes:
        prop = next(p for p in props if p[0] == name)
        if item_size is None:
            result[name] = raw[starts[name][:, None] + np.arange(size)].copy().view(endian + prop[1]).reshape(-1)
        else:
            n = lengths[name]
            item_starts = np.repeat(starts[name], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) * item_size
            result[name] = (raw[item_starts[:, None] + np.arange(item_size)].copy().view(endian + prop[2]).reshape(-1), n)
    return pos, result


def _uniform_result(array, props, first, count, end) -> tuple[int, dict]:
    result, lists = {}, iter(first)
    for name, kind, item in props:
        result[name] = array[name] if item is None else (np.ascontiguousarray(array[name]).reshape(-1), np.full(count, next(lists), np.int64))
    return end, result


def _read_ply_ascii(text: str, elements: list[dict]) -> dict[str, dict]:
    tokens, at, result = text.split(), 0, {}
    for element in elements:
        props, count = element["properties"], element["count"]
        if all(p[2] is None for p in props):
            values = np.array(tokens[at:at + count * len(props)], np.float64).reshape(count, len(props))
            at += count * len(props)
            result[element["name"]] = {p[0]: values[:, i] for i, p in enumerate(props)}
            continue
        columns = {p[0]: [] for p in props}
        lengths = {p[0]: [] for p in props if p[2] is not None}
        for _ in range(count):
            for name, kind, item in props:
                if item is None:
                    columns[name].append(tokens[at])
                    at += 1
                else:
                    n = int(tokens[at])
                    columns[name] += tokens[at + 1:at + 1 + n]
                    lengths[name].append(n)
                    at += 1 + n
        result[element["name"]] = {name: np.array(values, np.float64) if name not in lengths else (np.array(values, np.float64), np.array(lengths[name], np.int64))
                                   for name, values in columns.items()}
    return result


def _read_ply_elements(data: bytes, wanted=("vertex", "face")) -> tuple[dict, list[dict], list[str]]:
    fmt, elements, comments, pos = _ply_header(data)
    if fmt == "ascii":
        return _read_ply_ascii(data[pos:].decode("ascii", errors="replace"), elements), elements, comments
    endian = {"binary_little_endian": "<", "binary_big_endian": ">"}[fmt]
    result = {}
    for element in elements:
        if not set(wanted) - set(result):
            break
        pos, result[element["name"]] = _read_ply_element(data, pos, element, endian)
    return result, elements, comments


def _ply_color(values: np.ndarray, kind: str) -> np.ndarray:
    scale = {"u1": 255.0, "u2": 65535.0}.get(kind[-2:] if kind else "", 1.0)
    return values.astype(np.float64) / scale


def read_ply(path, *, device="cpu") -> Mesh:
    """Mesh from an ASCII or binary PLY: x y z, nx ny nz, red green blue (alpha) colours (uchar/ushort scaled, float as is,
    sRGB), per-vertex s t / u v / texture_u texture_v coordinates or per-face "texcoord" lists (v upwards), polygon faces
    (fan-triangulated) and a "comment TextureFile" base colour texture."""
    path = Path(path)
    elements, header, comments = _read_ply_elements(path.read_bytes())
    vertex = elements["vertex"]
    types = {p[0]: p[1] for e in header if e["name"] == "vertex" for p in e["properties"]}
    vertices = np.stack([vertex["x"], vertex["y"], vertex["z"]], 1).astype(np.float64)
    normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], 1) if {"nx", "ny", "nz"} <= set(vertex) else None
    names = next((n for n in (("red", "green", "blue"), ("diffuse_red", "diffuse_green", "diffuse_blue"), ("r", "g", "b")) if set(n) <= set(vertex)), None)
    colors = None
    if names:
        alpha = next((a for a in ("alpha", "diffuse_alpha", "a") if a in vertex), None)
        colors = np.stack([_ply_color(vertex[n], types[n]) for n in names + ((alpha,) if alpha else ())], 1)
    uv_names = next((n for n in (("s", "t"), ("u", "v"), ("texture_u", "texture_v"), ("texture_s", "texture_t")) if set(n) <= set(vertex)), None)
    uvs = np.stack([vertex[uv_names[0]], 1 - vertex[uv_names[1]].astype(np.float64)], 1) if uv_names else None
    face = elements.get("face", {})
    index_name = next((n for n in ("vertex_indices", "vertex_index") if n in face), None)
    if index_name is None:
        raise ValueError("PLY file without faces")
    flat, counts = face[index_name]
    flat = flat.astype(np.int64)
    triangles = _fan(counts)
    corner_vertices = flat[triangles]
    if "texcoord" in face:  # per-corner uvs: split the vertices on distinct (vertex, u, v)
        coords, coord_counts = face["texcoord"]
        if not np.array_equal(coord_counts, 2 * counts):
            raise ValueError("PLY texcoord lists do not match the face corners")
        corner_uvs = coords.reshape(-1, 2)[triangles]
        keys = np.concatenate([corner_vertices.reshape(-1, 1).astype(np.float64), corner_uvs.reshape(-1, 2)], 1)
        unique, inverse = unique_rows(keys)
        source = unique[:, 0].astype(np.int64)
        uvs = np.stack([unique[:, 1], 1 - unique[:, 2]], 1)
        vertices, normals, colors = vertices[source], None if normals is None else normals[source], None if colors is None else colors[source]
        faces = inverse.reshape(-1, 3)
    else:
        faces = corner_vertices
    textures = {}
    texture_name = next((c.split(None, 1)[1] for c in comments if c.startswith("TextureFile") and len(c.split(None, 1)) > 1), None)
    if texture_name and uvs is not None and (path.parent / texture_name).is_file():
        textures["base_color"] = _base_color_array(_texture_file(path.parent / texture_name))
    return _to_mesh(device, vertices, faces, normals, uvs, None if colors is None else colors.clip(0, 1), **textures)


def write_ply(mesh: Mesh, name: str = "mesh") -> dict[str, bytes]:
    """Binary little-endian PLY: float x y z, nx ny nz, s t (v upwards) when the mesh has uvs, uchar red green blue (alpha)
    sRGB colours (the vertex colours, or the base colour texture sampled at the vertex uvs), uchar/int faces."""
    vertices = _numpy(mesh.vertices, np.float32)
    normals = _numpy(vertex_normals(mesh.vertices, mesh.faces) if mesh.normals is None else mesh.normals, np.float32)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    columns = {"x": vertices[:, 0], "y": vertices[:, 1], "z": vertices[:, 2], "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2]}
    if mesh.uvs is not None:
        uvs = _numpy(mesh.uvs, np.float32)
        fields += [("s", "<f4"), ("t", "<f4")]
        columns |= {"s": uvs[:, 0], "t": 1 - uvs[:, 1]}
    colors = None
    if mesh.colors is not None:
        colors = _numpy(mesh.colors, np.float32)
    elif mesh.base_color is not None and mesh.uvs is not None:
        colors = _sample(_numpy(mesh.base_color), _numpy(mesh.uvs, np.float32))
    if colors is not None:
        names = ("red", "green", "blue", "alpha")[:colors.shape[1]]
        fields += [(n, "u1") for n in names]
        columns |= {n: np.rint(colors[:, i].clip(0, 1) * 255).astype(np.uint8) for i, n in enumerate(names)}
    records = np.empty(len(vertices), np.dtype(fields))
    for key, values in columns.items():
        records[key] = values
    faces = np.empty(len(mesh.faces), np.dtype([("n", "u1"), ("vertex_indices", "<i4", (3,))]))
    faces["n"], faces["vertex_indices"] = 3, _numpy(mesh.faces, np.int32)
    header = ["ply", "format binary_little_endian 1.0", "comment WanGP mesh3d", f"element vertex {len(vertices)}"]
    header += [f"property {'float' if kind == '<f4' else 'uchar'} {n}" for n, kind in fields]
    header += [f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    return {f"{name}.ply": ("\n".join(header) + "\n").encode("ascii") + records.tobytes() + faces.tobytes()}


def ply_stats(path: str) -> dict:
    data = Path(path).read_bytes()
    elements, header, comments = _read_ply_elements(data, wanted=("vertex",))
    counts = {e["name"]: e["count"] for e in header}
    vertex_props = {p[0] for e in header if e["name"] == "vertex" for p in e["properties"]}
    face_props = {p[0] for e in header if e["name"] == "face" for p in e["properties"]}
    vertex = elements["vertex"]
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], 1) if counts.get("vertex") else np.zeros((0, 3))
    textures = []
    for comment in comments:
        if comment.startswith("TextureFile") and len(comment.split(None, 1)) > 1 and (Path(path).parent / comment.split(None, 1)[1]).is_file():
            with open(Path(path).parent / comment.split(None, 1)[1], "rb") as handle:
                textures.append(image_size(file_reader(handle)))
    return {"format": "ply", "vertices": counts.get("vertex", 0), "faces": counts.get("face", 0), "materials": int(bool(textures)), "textures": textures,
            "bbox": [positions.min(0).tolist(), positions.max(0).tolist()] if len(positions) else None,
            "has_uv": bool({"s", "u", "texture_u", "texture_s"} & vertex_props or "texcoord" in face_props),
            "has_colors": bool({"red", "diffuse_red", "r", "f_dc_0"} & vertex_props), "animations": 0, "animation": None,
            "splats": counts.get("vertex", 0) if {"f_dc_0", "opacity", "scale_0", "rot_0"} <= vertex_props else 0, "file_size": len(data)}


# ------------------------------------------------------------------------------------------------------------ STL
_STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])


def _stl_corners(data: bytes) -> np.ndarray:
    if len(data) >= 84 and 84 + 50 * struct.unpack_from("<I", data, 80)[0] == len(data):
        return np.frombuffer(data, _STL_RECORD, struct.unpack_from("<I", data, 80)[0], 84)["corners"].reshape(-1, 3)
    if data[:5].lower() == b"solid":
        return np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data), np.float32).reshape(-1, 3)
    raise ValueError("Not an STL file")


def read_stl(path, *, device="cpu") -> Mesh:
    """Mesh from a binary or ASCII STL; corners with identical positions are welded (stored normals are ignored)."""
    vertices, inverse = unique_rows(_stl_corners(Path(path).read_bytes()))
    return _to_mesh(device, vertices, inverse.reshape(-1, 3))


def write_stl(mesh: Mesh, name: str = "mesh") -> dict[str, bytes]:
    """Binary STL with unit face normals (STL has no units; see export_mesh for scale and Z-up)."""
    corners = _numpy(mesh.vertices[mesh.faces], np.float32)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    records = np.zeros(len(corners), _STL_RECORD)
    records["normal"] = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    records["corners"] = corners
    return {f"{name}.stl": b"WanGP mesh3d binary STL".ljust(80, b" ") + struct.pack("<I", len(records)) + records.tobytes()}


def stl_stats(path: str) -> dict:
    data = Path(path).read_bytes()
    corners = _stl_corners(data)
    return {"format": "stl", "vertices": len(unique_rows(corners)[0]), "faces": len(corners) // 3, "materials": 0, "textures": [],
            "bbox": [corners.min(0).tolist(), corners.max(0).tolist()] if len(corners) else None, "has_uv": False, "has_colors": False, "animations": 0, "animation": None, "splats": 0, "file_size": len(data)}
