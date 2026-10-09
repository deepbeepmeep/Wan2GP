# TRELLIS.2 mesh post-processing and GLB export with the shared mesh toolkit (shared.mesh3d), following o-voxel's to_glb with remesh=True
# (github.com/microsoft/TRELLIS.2/o-voxel, o_voxel/postprocess.py, MIT License; the official demo's settings) without CuMesh,
# nvdiffrast or trimesh: narrow-band remeshing (GPU), decimation with cleanup, outward orientation and UV unwrapping (CPU, in child
# processes overlapping the texture stage on the GPU), PBR baking by trilinear sampling of the sparse attribute voxels on the original
# surface, texture inpainting, then glTF Y-up export.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from shared.utils.phase_progress import check_abort, set_phase_status
from .fdg_mesh import compact, fill_holes, orient_outward, remove_duplicate_faces, remove_small_components, unify_orientations
from .remesh import remesh_narrow_band
from .sparse import sparse_trilinear

HOLE_PERIMETER = 3e-2  # o-voxel to_glb / Mesh.fill_holes
MIN_COMPONENT_AREA = 1e-5  # o-voxel to_glb remove_small_connected_components
REMESH_BAND = 1.0  # o-voxel to_glb remesh_band of the official demo, in voxels
PBR_LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
_SAMPLE_POINTS = 1 << 20
SURFACE_PARTS = 2  # grid cells per axis of the parallel decimation and unwrapping (2^3 cells)


def remesh_surface(mesh, resolution):
    """The raw flexible-dual-grid surface [vertices, faces] (non-manifold where the predicted crossings disagree; handed over, the list
    is emptied) remeshed as the closed surface at REMESH_BAND voxels from it (GPU). Returns vertices, faces and the offset of the new
    surface from the original one (object units)."""
    set_phase_status("Remeshing")
    mesh[:] = compact(*mesh)
    vertices, faces = remesh_narrow_band(mesh, resolution, REMESH_BAND)
    check_abort()
    return vertices, faces, REMESH_BAND / resolution


def finish_surface(vertices, faces, max_faces, texture_size=0, stop=None):
    """CPU: the remeshed surface decimated to max_faces (cells of a 2^3 grid in parallel processes, then a last pass), cleaned (duplicate
    faces, components smaller than MIN_COMPONENT_AREA, small holes), oriented outward and, with a texture_size, UV unwrapped (xatlas,
    the 2^3 cells unwrapped as separate meshes in parallel). Returns a shared.mesh3d Mesh on the CPU. `stop` (threading.Event) cancels
    before the unwrapping."""
    from shared.mesh3d import BackgroundUnwrap, Mesh
    from shared.mesh3d.cleanup import decimate

    if faces.shape[0] > max_faces:
        mesh = decimate(Mesh(vertices, faces), max_faces, parts=SURFACE_PARTS, welded=True)  # dual contouring vertices are unique
        vertices, faces = mesh.vertices, mesh.faces
        del mesh
    vertices, faces = remove_small_components(vertices, remove_duplicate_faces(faces), MIN_COMPONENT_AREA)
    vertices, faces = fill_holes(vertices, faces, HOLE_PERIMETER)
    mesh = Mesh(vertices, orient_outward(vertices, unify_orientations(faces)))
    if not texture_size or (stop is not None and stop.is_set()):
        return mesh
    unwrap = BackgroundUnwrap(mesh, resolution=texture_size, parts=SURFACE_PARTS)  # a child process: xatlas keeps the GIL
    try:
        return unwrap.result()
    finally:
        unwrap.close()


class SurfaceJob:
    """finish_surface of a remeshed GPU surface in a thread (decimation and unwrapping run in child processes), so that it overlaps the
    texture flows and decoder on the GPU; result() waits for it, close() cancels and joins it."""

    def __init__(self, vertices, faces, max_faces, texture_size=0):
        self.stop, self.output = threading.Event(), {}
        vertices, faces = vertices.cpu(), faces.cpu()

        def run():
            try:
                self.output["mesh"] = finish_surface(vertices, faces, max_faces, texture_size, self.stop)
            except BaseException as error:  # re-raised by result()
                self.output["error"] = error

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def result(self):
        self.thread.join()
        if "error" in self.output:
            raise self.output["error"]
        return self.output["mesh"]

    def close(self):
        self.stop.set()
        self.thread.join()


def sample_attributes(voxels, attrs, points, grid_size):
    """PBR attributes (P, C) at points (P, 3) in [-0.5, 0.5]^3 of the sparse voxels (grid_size^3 voxels over the unit cube), and
    whether any surrounding voxel is active at each point."""
    values = attrs.new_zeros((points.shape[0], attrs.shape[1]))
    found = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for start in range(0, points.shape[0], _SAMPLE_POINTS):
        part, weight = sparse_trilinear(attrs, voxels, (points[start:start + _SAMPLE_POINTS] + 0.5) * grid_size)
        values[start:start + _SAMPLE_POINTS], found[start:start + _SAMPLE_POINTS] = part, weight > 0
    return values, found


def _to_uint8(values):
    return (values * 255).clamp_(0, 255).to(torch.uint8)  # truncation, as upstream np.clip(x * 255, 0, 255).astype(np.uint8)


def _on_surface(points, normals, offset):
    """Points of the remeshed surface moved back to the original one, `offset` along their normals (o-voxel bakes at the closest point of
    the original mesh)."""
    return points - offset * F.normalize(normals, dim=-1)


def bake_textures(mesh, voxels, attrs, grid_size, texture_size, offset):
    """Base color + alpha and metallic-roughness textures of an unwrapped mesh (finish_surface), baked from the attribute voxels at the
    texel positions moved back onto the original surface. Texels whose position has no active voxel around it are inpainted with the
    uncovered ones."""
    from shared.mesh3d import interpolate, rasterize_uv, vertex_normals
    from shared.mesh3d.cleanup import inpaint_texture

    device = attrs.device
    set_phase_status("Baking Texture")
    uvs, faces = mesh.uvs.to(device=device, dtype=torch.float32), mesh.faces.to(device=device, dtype=torch.int64)
    vertices = mesh.vertices.to(device=device, dtype=torch.float32)
    face_id, bary = rasterize_uv(uvs, faces, texture_size)
    covered = face_id >= 0
    normals = vertex_normals(vertices, faces)
    positions = _on_surface(interpolate(vertices, faces, face_id, bary)[covered], interpolate(normals, faces, face_id, bary)[covered], offset)
    # faces smaller than a texel cover no texel center: their center is sampled into the free texel under their uv center, so that the
    # inpainting does not fill them from unrelated neighboring charts
    missing = torch.ones(faces.shape[0], dtype=torch.bool, device=device)
    missing[face_id[covered]] = False
    del face_id, bary
    corners = faces[missing]
    texel = (uvs[corners].mean(1) * texture_size).long().clamp_(0, texture_size - 1)
    row, column = texel[:, 1], texel[:, 0]
    free = ~covered[row, column]
    row, column = row[free], column[free]
    centers = _on_surface(vertices[corners[free]].mean(1), normals[corners[free]].mean(1), offset)
    del normals, missing, corners, texel, free
    texels = attrs.new_zeros((texture_size, texture_size, attrs.shape[1]))
    values, found = sample_attributes(voxels, attrs, positions, grid_size)
    texels[covered] = values
    holes = ~covered
    holes[covered] = ~found  # covered texels whose position has no active voxel around it are filled too
    values, found = sample_attributes(voxels, attrs, centers, grid_size)
    texels[row, column] = values
    holes[row, column] = ~found
    del positions, values, found, covered, centers, row, column
    texels = _to_uint8(texels).cpu()
    holes = holes.cpu()
    # o-voxel to_glb: Telea inpainting, radius 3 for the base color and 1 for metallic, roughness and alpha (OpenCV releases the GIL:
    # both run at once)
    with ThreadPoolExecutor(2) as pool:
        base_color = pool.submit(inpaint_texture, texels[..., PBR_LAYOUT["base_color"]], holes, radius=3, method="telea")
        extra = pool.submit(inpaint_texture, torch.cat([texels[..., PBR_LAYOUT[name]] for name in ("alpha", "roughness", "metallic")], dim=-1), holes, radius=1, method="telea")
        base_color, extra = base_color.result(), extra.result()
    # glTF packing: the alpha attribute in the base color alpha channel, metallic-roughness G = roughness, B = metallic
    metallic_roughness = extra.clone()
    metallic_roughness[..., 0] = 0
    return vertices, faces, uvs, torch.cat([base_color, extra[..., :1]], dim=-1), metallic_roughness


def vertex_colors(vertices, faces, voxels, attrs, grid_size, offset):
    """Base color and alpha (V, 4) in [0, 1] sampled on the original surface below the vertices; vertices without active voxels around
    them get the mean color."""
    from shared.mesh3d import vertex_normals

    values, found = sample_attributes(voxels, attrs, _on_surface(vertices, vertex_normals(vertices, faces), offset), grid_size)
    colors = torch.cat([values[:, PBR_LAYOUT["base_color"]], values[:, PBR_LAYOUT["alpha"]]], dim=1)
    if not bool(found.all()):
        colors[~found] = colors[found].mean(0) if bool(found.any()) else 0.5
    return colors.clamp_(0, 1)


def to_gltf_mesh(vertices, faces, uvs=None, base_color=None, metallic_roughness=None, colors=None):
    """shared.mesh3d Mesh in glTF axes: TRELLIS meshes are Z-up, o-voxel's to_glb maps (x, y, z) to (x, z, -y) (Y-up, a rotation, so
    the windings stay outward). The GLB writer computes the smooth vertex normals."""
    from shared.mesh3d import Mesh

    vertices = torch.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], dim=1).float()
    return Mesh(vertices.cpu(), faces.cpu().long(), uvs=None if uvs is None else uvs.cpu(), colors=None if colors is None else colors.cpu(),
                base_color=base_color, metallic_roughness=metallic_roughness)


def export_glb(mesh):
    from shared.mesh3d import mesh_to_glb

    return mesh_to_glb(mesh)


def render_preview(mesh, size=512):
    from shared.mesh3d import preview

    return preview(mesh, size=size)
