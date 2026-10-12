# Flexible dual grid to mesh, hole filling and face orientation for TRELLIS.2 meshes, in PyTorch/NumPy without compiled extensions.
# flexible_dual_grid_to_mesh follows o-voxel (github.com/microsoft/TRELLIS.2/o-voxel, o_voxel/convert/flexible_dual_grid.py, MIT License)
# with the CUDA hash map replaced by a sorted-key search. fill_holes and unify_orientations follow the algorithms of CuMesh
# (github.com/JeffreyXiang/CuMesh, src/clean_up.cu, MIT License): small boundary loops are closed by a fan around the mean of their
# edge midpoints, and faces are flipped to agree with their neighbors across manifold edges.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from .sparse import SparseTensor

# the 4 voxels around the x, y and z edges of a voxel, in quad order (o-voxel edge_neighbor_voxel_offset)
_EDGE_VOXELS = [[[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]]
_SPLIT_02 = [0, 1, 2, 0, 2, 3]
_SPLIT_13 = [0, 1, 3, 3, 1, 2]
_CHUNK_VOXELS = 1 << 21
_COMPONENT_FACES = 1 << 22  # faces hooked at a time by the GPU connected components


def decode_fdg(fields, voxel_margin=0.5):
    """FlexiDualGridVaeDecoder outputs (N, 7) FP32 -> dual vertices in voxel units, per-axis intersection flags, quad split weights."""
    vertices = (1 + 2 * voxel_margin) * torch.sigmoid(fields[:, 0:3]) - voxel_margin
    return vertices, fields[:, 3:6] > 0, torch.nn.functional.softplus(fields[:, 6:7])


def flexible_dual_grid_to_mesh(x, dual_vertices, intersected, split_weight, grid_size):
    """o-voxel flexible_dual_grid_to_mesh (inference): one vertex per voxel at coords + dual vertex, one quad per intersected voxel edge
    whose 4 surrounding voxels are active, split along the diagonal with the larger product of split weights. x: SparseTensor of the
    voxels (its coords). Returns vertices (N, 3) FP32 in [-0.5, 0.5]^3 and faces (F, 3) int64 (unreferenced vertices included)."""
    coords = x.coords
    offsets = torch.tensor(_EDGE_VOXELS, dtype=coords.dtype, device=coords.device)
    split_02 = torch.tensor(_SPLIT_02, device=coords.device)
    split_13 = torch.tensor(_SPLIT_13, device=coords.device)
    weight = split_weight.reshape(-1)
    faces = []
    for start in range(0, coords.shape[0], _CHUNK_VOXELS):
        voxel, axis = intersected[start:start + _CHUNK_VOXELS].nonzero(as_tuple=True)
        corners = x.find(coords[start + voxel][:, None] + offsets[axis])
        quads = corners[(corners >= 0).all(1)].long()
        quad_weight = weight[quads]
        faces.append(torch.where((quad_weight[:, 0] * quad_weight[:, 2] > quad_weight[:, 1] * quad_weight[:, 3])[:, None],
                                 quads[:, split_02], quads[:, split_13]).reshape(-1, 3))
        del voxel, axis, corners, quads, quad_weight
    vertices = (coords.float() + dual_vertices) * (1.0 / grid_size) - 0.5
    return vertices, torch.cat(faces)


def compact(vertices, faces):
    """Removes the unreferenced vertices."""
    used = torch.zeros(vertices.shape[0], dtype=torch.bool, device=vertices.device)
    used[faces.reshape(-1)] = True
    remap = torch.cumsum(used, 0) - 1
    return vertices[used], remap[faces]


def _edges(faces):
    """Directed edges (3F, 2) of the faces in their winding and the undirected key of each."""
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    low, high = directed.min(1).values, directed.max(1).values
    return directed, low * (int(faces.max()) + 1) + high


def fill_holes(vertices, faces, max_perimeter=3e-2):
    """Closes the boundary loops shorter than max_perimeter (CuMesh fill_holes): one new vertex at the mean of the loop's edge midpoints,
    one triangle per boundary edge, wound against the boundary face. Loops through non-manifold boundary vertices are left open."""
    directed, keys = _edges(faces)
    _, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1]
    if boundary.shape[0] == 0:
        return vertices, faces
    vertex_degree = torch.bincount(boundary.reshape(-1), minlength=vertices.shape[0])
    boundary = boundary[(vertex_degree[boundary] == 2).all(1)]
    if boundary.shape[0] == 0:
        return vertices, faces
    edges = boundary.cpu().numpy()
    used, local = np.unique(edges, return_inverse=True)
    local = local.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(local)), (local[:, 0], local[:, 1])), shape=(len(used), len(used)))
    _, labels = connected_components(graph, directed=False)
    loop = torch.from_numpy(labels[local[:, 0]]).to(vertices.device)
    loops = int(loop.max()) + 1
    ends = vertices[boundary]
    length = torch.zeros(loops, device=vertices.device).index_add_(0, loop, (ends[:, 1] - ends[:, 0]).norm(dim=1))
    edge_count = torch.bincount(loop, minlength=loops)
    # a component of degree-2 vertices is a simple cycle when it has as many edges as vertices
    vertex_count = torch.from_numpy(np.bincount(labels, minlength=loops)).to(vertices.device)
    small = (length < max_perimeter) & (edge_count == vertex_count) & (edge_count >= 3)
    if not bool(small.any()):
        return vertices, faces
    keep = small[loop]
    boundary, loop, ends = boundary[keep], loop[keep], ends[keep]
    new_index = torch.cumsum(small, 0) - 1 + vertices.shape[0]
    # FP64 sums: the order of the GPU atomic additions then (practically) never changes the FP32 result between runs
    centers = (torch.zeros((loops, 3), dtype=torch.float64, device=vertices.device).index_add_(0, loop, ends.mean(1).double()) / edge_count.clamp(min=1)[:, None]).float()
    new_faces = torch.stack([boundary[:, 1], boundary[:, 0], new_index[loop]], dim=1)
    return torch.cat([vertices, centers[small]]), torch.cat([faces, new_faces.to(faces.dtype)])


def unify_orientations(faces):
    """Flips faces so that the two faces of every manifold edge traverse it in opposite directions (CuMesh unify_face_orientations):
    a breadth-first spanning forest of the face adjacency carries the relative flips from one root face per component."""
    count = faces.shape[0]
    directed, keys = _edges(faces)
    order = torch.argsort(keys)
    _, counts = torch.unique_consecutive(keys[order], return_counts=True)
    starts = (torch.cumsum(counts, 0) - counts)[counts == 2]
    first, second = order[starts], order[starts + 1]
    # weight 2: the faces traverse the shared edge in the same direction, so one of them must flip; weight 1: consistent
    weight = (directed[first, 0] == directed[second, 0]).cpu().numpy().astype(np.int64) + 1
    face_a, face_b = (first // 3).cpu().numpy(), (second // 3).cpu().numpy()
    graph = coo_matrix((weight, (face_a, face_b)), shape=(count, count)).tocsr()
    graph = (graph + graph.T).tocsr()
    _, labels = connected_components(graph, directed=False)
    _, roots = np.unique(labels, return_index=True)
    # one BFS from a virtual node linked to a root face of every component
    root = count
    links = coo_matrix((np.ones(len(roots), dtype=np.int64), (np.full(len(roots), root), roots)), shape=(count + 1, count + 1))
    graph.resize((count + 1, count + 1))
    graph = (graph + links + links.T).tocsr()
    _, parent = breadth_first_order(graph, root, directed=False, return_predecessors=True)
    parent[root] = root
    nodes = np.arange(count)
    parity = np.zeros(count + 1, dtype=np.int64)
    parity[:count] = np.asarray(graph[nodes, parent[:count]]).reshape(-1) % 2 == 0
    # parity relative to the root: XOR along the tree path, by pointer jumping
    while True:
        grand = parent[parent]
        parity ^= parity[parent]
        if np.array_equal(grand, parent):
            break
        parent = grand
    flip = torch.from_numpy(parity[:count].astype(bool)).to(faces.device)
    faces = faces.clone()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def _vertex_roots(vertex_count, faces):
    """Smallest vertex index of the connected component of each vertex (GPU): edge hooking of the larger root under the smaller one,
    then pointer jumping, until no root changes."""
    root = torch.arange(vertex_count, device=faces.device)
    while True:
        before = root.clone()
        for start in range(0, faces.shape[0], _COMPONENT_FACES):
            part = faces[start:start + _COMPONENT_FACES]
            for a, b in ((0, 1), (1, 2)):
                ra, rb = root[part[:, a]], root[part[:, b]]
                root.scatter_reduce_(0, torch.maximum(ra, rb), torch.minimum(ra, rb), "amin")
                del ra, rb
        while True:
            jumped = root[root]
            if torch.equal(jumped, root):
                break
            root = jumped
        if torch.equal(root, before):
            return root


def face_components(vertex_count, faces):
    """Connected component of each face (faces sharing a vertex are connected), and the component count. Components are numbered in
    the order of their smallest vertex index, as scipy's connected_components numbers them (CUDA faces: computed on the GPU)."""
    if faces.is_cuda:
        root = _vertex_roots(vertex_count, faces)
        rank = torch.cumsum(root == torch.arange(vertex_count, device=faces.device), 0) - 1
        count = int(rank[-1]) + 1 if vertex_count else 0
        return rank[root[faces[:, 0]]], count
    f = faces.cpu().numpy()
    graph = coo_matrix((np.ones(f.shape[0] * 2), (np.repeat(f[:, 0], 2), f[:, 1:].reshape(-1))), shape=(vertex_count, vertex_count))
    count, labels = connected_components(graph, directed=False)
    return torch.from_numpy(labels[f[:, 0]]).to(device=faces.device, dtype=torch.int64), count


def orient_outward(vertices, faces):
    """Flips the connected components whose signed volume is negative (closed components then face outward)."""
    component, count = face_components(vertices.shape[0], faces)
    v = vertices[faces].double()
    volume = torch.zeros(count, dtype=torch.float64, device=faces.device).index_add_(0, component, (v[:, 0] * torch.cross(v[:, 1], v[:, 2], dim=1)).sum(1))
    flip = volume[component] < 0
    faces = faces.clone()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def remove_duplicate_faces(faces):
    """Keeps one face per vertex set (CuMesh remove_duplicate_faces), in the original order."""
    _, first = np.unique(np.sort(faces.cpu().numpy(), axis=1), axis=0, return_index=True)
    return faces[torch.from_numpy(np.sort(first)).to(faces.device)]


def remove_small_components(vertices, faces, min_area):
    """Drops the connected components whose total area is below min_area (CuMesh remove_small_connected_components)."""
    component, count = face_components(vertices.shape[0], faces)
    v = vertices[faces]
    area = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1).norm(dim=1) / 2
    total = torch.zeros(count, dtype=area.dtype, device=faces.device).index_add_(0, component, area)
    return compact(vertices, faces[total[component] >= min_area])

