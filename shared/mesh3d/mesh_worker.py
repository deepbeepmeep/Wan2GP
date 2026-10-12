# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU mesh jobs on numpy arrays (xatlas UV unwrapping, fast-simplification decimation), also runnable as child processes by cleanup.py:
both libraries keep the GIL while they run, so threads would neither run them in parallel nor let GPU work proceed meanwhile.
As a script: python -I mesh_worker.py <job> <integer arguments...>, the input arrays as .npy on stdin, the output arrays as .npy on
stdout. Imports only numpy and the library of the job."""
import io
import sys

import numpy as np


def grid_parts(vertices, faces, parts, centers=True):
    """Face indices of each non-empty cell of a parts^3 grid over the bounding box of the vertices, a face going to the cell of its
    center (straight cell borders: fewer seams for the unwrapping) or, with centers=False, of its first vertex (faster on millions of
    faces; the decimation re-decimates its cut borders anyway)."""
    if parts <= 1:
        return [np.arange(len(faces))]
    # per-axis reductions and in-place arithmetic: the values of (points - low) / max(high - low, 1e-12) * parts at a fraction of the cost
    low, high = (np.array([reduce(vertices[:, axis]) for axis in range(3)]) for reduce in (np.min, np.max))
    scaled = (vertices[faces].mean(1) if centers else vertices) - low
    scaled /= np.maximum(high - low, 1e-12)
    scaled *= parts
    cells = scaled.astype(np.int64)
    del scaled
    np.minimum(cells, parts - 1, out=cells)
    cell = ((cells[:, 0] * parts + cells[:, 1]) * parts + cells[:, 2]).astype(np.uint16)
    if not centers:
        cell = cell[faces[:, 0]]
    order = np.argsort(cell, kind="stable")  # radix sort of 16-bit keys
    return np.split(order, np.flatnonzero(np.diff(cell[order])) + 1)


def cell_mesh(points, triangles, group):
    """The vertices and faces (local indices) of the faces `group`, vertices in their original order (as np.unique, without a sort)."""
    used = np.zeros(len(points), dtype=bool)
    used[triangles[group]] = True
    remap = np.cumsum(used) - 1
    return np.flatnonzero(used), remap[triangles[group]]


def unwrap_arrays(vertices, faces, resolution=0, padding=2, max_iterations=1, parts=1):
    """xatlas atlas of the mesh: (mapping [V'] original vertex of each atlas vertex, faces [F, 3] in the original face order, uvs [V', 2]
    in [0, 1] with v down). parts > 1 unwraps each cell of a parts^3 grid as a separate xatlas mesh: xatlas computes the charts of the
    meshes in parallel and its chart growing is superlinear in the mesh size (100K faces: 26-46 s whole, 1.5-1.8 s in 2^3 cells), at the
    cost of seams along the cell borders (2-8 % longer seams, same utilization)."""
    import xatlas

    groups = grid_parts(vertices, faces, parts)
    atlas = xatlas.Atlas()
    used = []
    if len(groups) == 1:
        atlas.add_mesh(vertices, faces)
    else:
        for group in groups:
            indices, local = cell_mesh(vertices, faces, group)
            atlas.add_mesh(vertices[indices], local.astype(np.uint32))
            used.append(indices)
    charts, packing = xatlas.ChartOptions(), xatlas.PackOptions()
    charts.max_iterations = max_iterations
    packing.padding = padding
    if resolution:
        packing.resolution = resolution
    atlas.generate(charts, packing)
    if atlas.atlas_count != 1:
        raise ValueError(f"xatlas split the charts into {atlas.atlas_count} atlases; raise the resolution")
    if len(groups) == 1:
        return atlas[0]
    mappings, uvs, out_faces, offset = [], [], np.empty((len(faces), 3), dtype=np.uint32), 0
    for index, (group, indices) in enumerate(zip(groups, used)):
        mapping, part_faces, part_uvs = atlas[index]
        out_faces[group] = part_faces.reshape(-1, 3) + offset
        mappings.append(indices[mapping])
        uvs.append(part_uvs)
        offset += len(mapping)
    return np.concatenate(mappings).astype(np.uint32), out_faces, np.concatenate(uvs)


def decimate_cell(points, triangles, target_faces, aggressiveness=7):
    """fast-simplification of one grid cell (points float64 [V, 3], triangles int32 [F, 3]) with its border kept, so that the cells weld
    back; target_faces counts the inner faces: two faces per border edge are added to it, else the kept border would starve the inside."""
    import fast_simplification

    first, second = triangles.T.astype(np.int64), triangles[:, [1, 2, 0]].T.astype(np.int64)
    keys = np.sort((np.minimum(first, second) * len(points) + np.maximum(first, second)).reshape(-1))  # undirected edge keys
    single = np.ones(len(keys), dtype=bool)
    single[1:] &= keys[1:] != keys[:-1]
    single[:-1] &= keys[:-1] != keys[1:]
    target = int(target_faces) + 2 * int(single.sum())
    return fast_simplification.simplify(points, triangles, target_count=target, agg=aggressiveness, preserve_border=True)


JOBS = {"unwrap": unwrap_arrays, "decimate_cell": decimate_cell}


def main(job, arguments):
    stream = io.BytesIO(sys.stdin.buffer.read())
    arrays = [np.load(stream), np.load(stream)]
    try:
        results = JOBS[job](*arrays, *arguments)
    except ValueError as error:
        sys.exit(str(error))
    output = io.BytesIO()
    for array in results:
        np.save(output, array)
    sys.stdout.buffer.write(output.getvalue())


if __name__ == "__main__":
    main(sys.argv[1], [int(value) for value in sys.argv[2:]])
