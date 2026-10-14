# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared 3D mesh toolkit (pure PyTorch / numpy, device explicit): the Mesh container, GLB/OBJ/PLY/STL I/O (static,
multi-part and morph-animated GLB), Gaussian splat PLYs, a triangle rasterizer, previews and turntables, mesh cleanup /
decimation / UV unwrapping and surface extraction.

Conventions are those of glTF 2.0 (see mesh.py): +Y up, +Z front, counter-clockwise outward faces, uv (0, 0) at the
top-left of the texture image, sRGB vertex colours and base colour textures. The CPU libraries (xatlas,
fast-simplification, scikit-image, scipy, OpenCV) are imported only by the functions that use them.
"""
from .cleanup import BackgroundUnwrap, decimate, inpaint_texture, inpaint_vertex_colors, merge_vertices, remove_degenerate, remove_floaters, unwrap_uv
from .glb import (EXPORT_FORMATS, MESH_EXTENSIONS, animated_mesh_to_glb, export_mesh, glb_read_extras, glb_to_mesh, glb_write_extras, load_mesh,
                  mesh_file_stats, mesh_to_glb, meshes_to_glb)
from .mesh import Mesh, bounding_box, face_normals, linear_to_srgb, srgb_to_linear, submesh, vertex_normals, weld
from .raster import interpolate, rasterize, rasterize_uv
from .render import look_at, orthographic, perspective, preview, render, splat_colors, splat_preview, turntable
from .splats import (SH_C0, is_splat_ply, ply_comments, ply_to_splats, ply_write_comments, points_to_splats, splat_file_stats, splats_opencv_to_gltf,
                     splats_to_ply, write_points_ply)
from .surface import marching_cubes
