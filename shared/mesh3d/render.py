# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Cameras, multi-mode rendering, previews and turntables, built on raster.py.

Cameras follow OpenGL: look_at() returns world-to-camera (view) matrices, the camera looking down its -Z axis with +Y up
and +X right; perspective() / orthographic() return projection matrices to clip space (z / w = -1 at the near plane, +1
at the far plane). A camera set is a (view, projection) pair, each [4, 4] or [B, 4, 4]. Rendered images are channels
last, top row first: [B, H, W, C]. Colours are sRGB in 0..1; shading is computed in linear light.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .mesh import Mesh, linear_to_srgb, srgb_to_linear, vertex_normals
from .raster import interpolate, rasterize

MODES = ("shaded", "albedo", "normal", "depth", "position", "mask")
COLOR_MODES = ("shaded", "albedo")
DEFAULT_ALBEDO_LINEAR = ((0.8 + 0.055) / 1.055) ** 2.4  # sRGB grey 0.8 of meshes without texture or vertex colours
LIGHT = (-0.45, 0.65, 0.6)                 # camera-space direction towards the key light (upper left, in front)
PREVIEW_VIEW = (35.0, 20.0)                # azimuth from +Z towards +X, elevation (degrees)
FIELD_OF_VIEW = 30.0


def _tensor(value, device) -> torch.Tensor:
    return value.to(device=device, dtype=torch.float32) if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32, device=device)


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), *, device="cpu") -> torch.Tensor:
    """World-to-camera matrices [4, 4] (eye [3]) or [B, 4, 4] (eye [B, 3]) of cameras at eye looking at target, with up
    towards the top of the image. Tensors keep their device; tuples are created on `device`."""
    device = eye.device if isinstance(eye, torch.Tensor) else device
    eye, target, up = (_tensor(v, device) for v in (eye, target, up))
    batched = eye.ndim == 2
    eye = eye.reshape(-1, 3)
    forward = F.normalize(target.expand_as(eye) - eye, dim=-1)
    right = F.normalize(torch.linalg.cross(forward, up.expand_as(eye)), dim=-1)
    rotation = torch.stack([right, torch.linalg.cross(right, forward), -forward], 1)
    view = torch.eye(4, dtype=torch.float32, device=device).repeat(len(eye), 1, 1)
    view[:, :3, :3] = rotation
    view[:, :3, 3] = -(rotation @ eye[..., None])[..., 0]
    return view if batched else view[0]


def perspective(fov_y: float = 40.0, aspect: float = 1.0, near: float = 0.1, far: float = 100.0, *, device="cpu") -> torch.Tensor:
    """OpenGL perspective projection [4, 4]: vertical field of view in degrees, aspect = width / height."""
    f = 1.0 / math.tan(math.radians(fov_y) / 2)
    return torch.tensor([[f / aspect, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]],
                        dtype=torch.float32, device=device)


def orthographic(height: float = 2.0, aspect: float = 1.0, near: float = 0.1, far: float = 100.0, *, device="cpu") -> torch.Tensor:
    """OpenGL orthographic projection [4, 4] of a view box `height` high (world units) and height * aspect wide."""
    return torch.tensor([[2 / (height * aspect), 0, 0, 0], [0, 2 / height, 0, 0], [0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1]],
                        dtype=torch.float32, device=device)


def _texture(texture: torch.Tensor, max_height: int, max_width: int) -> torch.Tensor:
    """uint8 sRGB [H, W, C] texture -> linear float [1, 3, h, w], area-downsampled by an integer factor to at most twice the
    render size (a cheap stand-in for mipmapping), converted by bands of rows to bound memory."""
    factor = max(1, math.ceil(max(texture.shape[0] / (2 * max_height), texture.shape[1] / (2 * max_width))))
    height, width = texture.shape[0] // factor * factor, texture.shape[1] // factor * factor
    band = factor * max(1, (1 << 22) // (width * 3 * factor))
    bands = [F.avg_pool2d(texture[r:min(r + band, height), :width, :3].permute(2, 0, 1)[None].float() / 255, factor) for r in range(0, height, band)]
    return srgb_to_linear(torch.cat(bands, 2))


def _sample(texture: torch.Tensor, uvs: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup (repeat wrap) of a [1, C, h, w] texture at glTF uvs [N, 2] -> [N, C]."""
    grid = ((uvs - uvs.floor()) * 2 - 1).view(1, 1, -1, 2)
    return F.grid_sample(texture, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T


def _cameras(cameras, device) -> tuple[torch.Tensor, torch.Tensor]:
    view, projection = (_tensor(m, device) for m in cameras)
    view, projection = view.reshape(-1, 4, 4), projection.reshape(-1, 4, 4)
    count = max(len(view), len(projection))
    return view.expand(count, 4, 4), projection.expand(count, 4, 4)


def _render(mesh: Mesh, cameras, resolution, mode: str, supersample: int, normal_space: str, ambient: float):
    """Premultiplied values [B, H, W, C] (values x coverage, averaged over the supersamples) and coverage [B, H, W, 1]."""
    if mode not in MODES:
        raise ValueError(f"Unknown render mode {mode!r}")
    device = mesh.vertices.device
    view, projection = _cameras(cameras, device)
    height, width = (resolution, resolution) if isinstance(resolution, int) else resolution
    h, w = height * supersample, width * supersample
    homogeneous = torch.cat([mesh.vertices.float(), torch.ones_like(mesh.vertices[:, :1], dtype=torch.float32)], 1)
    camera = homogeneous @ view.transpose(1, 2)
    clip = camera @ projection.transpose(1, 2)
    clip[..., 1].neg_()  # row 0 = top of the view
    face_id, bary, _ = rasterize(clip, mesh.faces, (h, w))
    covered = (face_id >= 0)[..., None].float()

    if mode == "mask":
        values = covered
    elif mode == "position":
        values = interpolate(mesh.vertices.float(), mesh.faces, face_id, bary)
    elif mode == "depth":
        values = -interpolate(camera[..., 2:3], mesh.faces, face_id, bary)
    else:
        normals = vertex_normals(mesh.vertices, mesh.faces) if mesh.normals is None else mesh.normals.float()
        textured = mesh.uvs is not None and mesh.base_color is not None
        attributes = [normals]
        if mode in COLOR_MODES:
            attributes += [mesh.uvs.float()] if textured else []
            attributes += [] if mesh.colors is None else [torch.cat([srgb_to_linear(mesh.colors[:, :3].float()), mesh.colors[:, 3:].float()], 1)]
        pixels = interpolate(torch.cat(attributes, 1), mesh.faces, face_id, bary)
        normal = F.normalize(pixels[..., :3], dim=-1)
        if mode == "normal":
            values = torch.einsum("bij,bhwj->bhwi", view[:, :3, :3], normal) if normal_space == "camera" else normal
        else:
            albedo = torch.full_like(pixels[..., :3], DEFAULT_ALBEDO_LINEAR)
            if textured:
                hit = face_id >= 0
                albedo[hit] = _sample(_texture(mesh.base_color, h, w), pixels[..., 3:5][hit])
            if mesh.colors is not None:
                color = pixels[..., 5 if textured else 3:][..., :3]
                albedo = albedo * color if textured else color
            if mode == "shaded":
                normal_camera = torch.einsum("bij,bhwj->bhwi", view[:, :3, :3], normal)
                normal_camera = torch.where(normal_camera[..., 2:3] < 0, -normal_camera, normal_camera)
                light = F.normalize(torch.tensor(LIGHT, dtype=torch.float32, device=device), dim=0)
                albedo = albedo * (ambient + (1 - ambient) * (normal_camera @ light).clamp_min(0)[..., None])
            values = linear_to_srgb(albedo.clamp(0, 1))
    values = values * covered
    if supersample > 1:
        pool = lambda x: F.avg_pool2d(x.permute(0, 3, 1, 2), supersample).permute(0, 2, 3, 1)
        values, covered = pool(values), pool(covered)
    return values, covered


def render(mesh: Mesh, cameras, resolution, mode: str = "shaded", background=None, *, supersample: int = 1, normal_space: str = "world",
           ambient: float = 0.3) -> torch.Tensor:
    """Renders B views of a mesh: [B, H, W, C] float32 on the mesh's device (resolution: int or (H, W)).

    mode:
    - "shaded": sRGB colour, Lambert key light from the camera's upper left plus `ambient`, two-sided;
    - "albedo": unlit sRGB colour (base colour texture x vertex colours, or grey 0.8);
    - "normal": unit normals (C = 3) in world space, or camera space (x right, y up, z towards the camera) with
      normal_space="camera"; normal maps are ignored;
    - "depth": distance along the camera axis (C = 1);
    - "position": world positions (C = 3);
    - "mask": coverage (C = 1).
    With background None, colour modes return RGBA (straight alpha, C = 4) and the other modes are zero where empty;
    otherwise every mode is composited over background (a value or per-channel sequence). supersample renders at that
    factor and averages down (anti-aliasing, fractional coverage on edges)."""
    values, covered = _render(mesh, cameras, resolution, mode, supersample, normal_space, ambient)
    if background is None:
        return torch.cat([values / covered.clamp_min(1e-12), covered], -1) if mode in COLOR_MODES else values
    return values + _tensor(background, values.device) * (1 - covered)


def _framing(mesh: Mesh) -> tuple[torch.Tensor, float]:
    low, high = mesh.vertices.float().amin(0), mesh.vertices.float().amax(0)
    center = (low + high) / 2
    return center, max(float((mesh.vertices.float() - center).norm(dim=1).max()), 1e-6)


def _orbit(center: torch.Tensor, radius: float, azimuth: float, elevation: float):
    """View and projection of a camera orbiting center at the distance where the bounding sphere fills the view."""
    distance = radius / math.sin(math.radians(FIELD_OF_VIEW) / 2)
    a, e = math.radians(azimuth), math.radians(elevation)
    direction = torch.tensor([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)], dtype=torch.float32, device=center.device)
    view = look_at(center + direction * distance, center)
    projection = perspective(FIELD_OF_VIEW, 1.0, max(distance - 1.05 * radius, 1e-3 * distance), distance + 1.05 * radius, device=center.device)
    return view, projection, distance


def _zoom(projection: torch.Tensor, low, high, fill: float = 0.92) -> torch.Tensor:
    """Projection zoomed and shifted so that the NDC box [low, high] fills `fill` of the image (aspect kept)."""
    center, half = (low + high) / 2, float(((high - low) / 2).max().clamp_min(1e-6))
    scale = fill / half
    adjust = torch.eye(4, dtype=torch.float32, device=projection.device)
    adjust[0, 0] = adjust[1, 1] = scale
    adjust[0, 3], adjust[1, 3] = -scale * center[0], -scale * center[1]
    return adjust @ projection


def _ndc_box(mesh: Mesh, view: torch.Tensor, projection: torch.Tensor):
    homogeneous = torch.cat([mesh.vertices.float(), torch.ones_like(mesh.vertices[:, :1], dtype=torch.float32)], 1)
    clip = homogeneous @ (projection @ view).T
    ndc = clip[:, :2] / clip[:, 3:]
    return ndc.amin(0), ndc.amax(0)


def preview(mesh: Mesh, size: int = 512, *, supersample: int = 2) -> torch.Tensor:
    """Canonical 3/4 view (azimuth 35 degrees from the front towards +X, elevation 20 degrees), shaded, auto-framed on the
    mesh: RGBA [4, size, size] in [-1, 1] with a transparent background (alpha -1), on the mesh's device."""
    center, radius = _framing(mesh)
    view, projection, _ = _orbit(center, radius, *PREVIEW_VIEW)
    projection = _zoom(projection, *_ndc_box(mesh, view, projection))
    image = render(mesh, (view, projection), size, "shaded", supersample=supersample)[0]
    return image.permute(2, 0, 1) * 2 - 1


def _encoded_views(mesh: Mesh, view: torch.Tensor, projection: torch.Tensor, resolution, mode: str, background, supersample: int) -> torch.Tensor:
    """Views [B, H, W, C] in 0..1 rendered by groups (bounded memory). Colour modes: RGBA with straight alpha when
    background is None (C = 4), else sRGB over background. Other modes (C = 3, over black by default): normal:
    camera-space (n + 1) / 2; depth: linear from 1 at the nearest to 0.15 at the farthest surface seen in all the views
    (no flicker between views); mask: coverage; position: world coordinates mapped from the bounding sphere to 0..1."""
    if mode not in MODES:
        raise ValueError(f"Unknown render mode {mode!r}")
    device = mesh.vertices.device
    height, width = (resolution, resolution) if isinstance(resolution, int) else resolution
    transparent = background is None and mode in COLOR_MODES
    background = None if transparent else _tensor(0.0 if background is None else background, device)
    center, radius = _framing(mesh)
    count = len(view)
    group = max(1, (1 << 22) // (height * width * supersample * supersample))
    images = torch.empty(count, height, width, 4 if transparent else 3, dtype=torch.float32, device=device)
    for start in range(0, count, group):
        stop = min(start + group, count)
        values, coverage = _render(mesh, (view[start:stop], projection[start:stop]), (height, width), mode, supersample, "camera", 0.3)
        if mode == "depth":  # premultiplied depth and coverage, normalized once every view is known
            images[start:stop, ..., 0], images[start:stop, ..., 1] = values[..., 0], coverage[..., 0]
            continue
        if transparent:
            images[start:stop] = torch.cat([values / coverage.clamp_min(1e-12), coverage], -1)
            continue
        if mode == "normal":
            values = (values + coverage) / 2
        elif mode == "position":
            values = (values - (center - radius) * coverage) / (2 * radius)
        images[start:stop] = (values + background * (1 - coverage)).expand(-1, -1, -1, 3)
    if mode == "depth":
        coverage = images[..., 1].clone()
        depth = images[..., 0] / coverage.clamp_min(1e-12)
        solid = depth[coverage > 0.999]
        near, far = (solid.min(), solid.max()) if len(solid) else (torch.zeros((), device=device), torch.ones((), device=device))
        values = coverage * (1 - 0.85 * (depth - near) / (far - near).clamp_min(1e-6 * radius)).clamp(0, 1)
        images[:] = (values[..., None] + background.reshape(1, 1, 1, -1) * (1 - coverage[..., None])).expand(-1, -1, -1, 3)
    return images


def turntable(mesh: Mesh, frames: int = 48, size: int = 512, mode: str = "shaded", *, elevation: float = 15.0, background=None,
              supersample: int = 2) -> torch.Tensor:
    """Turntable video [3, frames, size, size] in [-1, 1]: the camera orbits the mesh (starting at the front, +Z, moving
    towards +X) with the key light attached to it, framed so that the mesh stays in view on every frame. Encodings as
    _encoded_views; shaded / albedo over a white background by default."""
    if mode not in MODES:
        raise ValueError(f"Unknown render mode {mode!r}")
    center, radius = _framing(mesh)
    orbits = [_orbit(center, radius, 360.0 * i / frames, elevation) for i in range(frames)]
    boxes = [_ndc_box(mesh, view, projection) for view, projection, _ in orbits]
    low, high = torch.stack([b[0] for b in boxes]).amin(0), torch.stack([b[1] for b in boxes]).amax(0)
    half = torch.maximum(low.abs(), high.abs())  # keep the orbit centre in the middle of the image
    view, projection = torch.stack([v for v, _, _ in orbits]), torch.stack([_zoom(p, -half, half) for _, p, _ in orbits])
    background = background if background is not None else (1.0, 1.0, 1.0) if mode in COLOR_MODES else 0.0
    return _encoded_views(mesh, view, projection, size, mode, background, supersample).permute(3, 0, 1, 2) * 2 - 1


# ------------------------------------------------------------------------------------------------------------ views
VIEWS = {"front": (0.0, 0.0), "back": (180.0, 0.0), "left": (90.0, 0.0), "right": (-90.0, 0.0), "top": (0.0, 90.0), "bottom": (0.0, -90.0),
         "three_quarter": PREVIEW_VIEW}  # (azimuth, elevation) in degrees; left / right are the object's own sides


def orbit_cameras(points: torch.Tensor, angles, resolution, *, projection: str = "orthographic", zoom: float = 1.0, focus=(0.5, 0.5, 0.5),
                  quantile: float = 0.0, fill: float = 0.92) -> tuple[torch.Tensor, torch.Tensor]:
    """Cameras (view [B, 4, 4], projection [B, 4, 4]) around points [N, 3] at angles [(azimuth, elevation)] in degrees:
    azimuth 0 looks at the front (+Z), 90 at the +X side (an object's left when it faces the front camera), 180 at the
    back; elevation 90 looks from above. Every view shares one scale: the points fill `fill` of the image in the view
    where they extend most (their quantile .. 1 - quantile range when quantile > 0, against splat floaters), magnified
    by zoom, centred on focus (fractions of the bounding box: 0.5 is the centre, y 1 the top, z 1 the front)."""
    height, width = (resolution, resolution) if isinstance(resolution, int) else resolution
    device = points.device
    sample = points[::max(1, len(points) // 500000)].float()
    low, high = (torch.quantile(sample, quantile, dim=0), torch.quantile(sample, 1 - quantile, dim=0)) if quantile > 0 else (sample.amin(0), sample.amax(0))
    center, radius = (low + high) / 2, max(float((high - low).norm() / 2), 1e-6)
    target = low + _tensor(focus, device) * (high - low)
    distance = radius / math.sin(math.radians(FIELD_OF_VIEW) / 2)
    near, far = max(distance - 1.05 * radius, 1e-3 * distance), distance + 1.05 * radius
    base = perspective(FIELD_OF_VIEW, width / height, near, far, device=device) if projection == "perspective" else orthographic(2 * radius, width / height, near, far, device=device)
    homogeneous = torch.cat([sample, torch.ones_like(sample[:, :1])], 1)
    views, half = [], torch.zeros(2, device=device)
    for azimuth, elevation in angles:
        a, e = math.radians(azimuth), math.radians(max(-90.0, min(90.0, elevation)))
        direction = torch.tensor([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)], dtype=torch.float32, device=device)
        sign = math.copysign(1.0, elevation)  # straight above / below: the image top is the far side of the azimuth
        up = (0.0, 1.0, 0.0) if abs(elevation) < 89.9 else (-math.sin(a) * sign, 0.0, -math.cos(a) * sign)
        clip = homogeneous @ (base @ look_at(center + direction * distance, center, up)).T
        ndc = (clip[:, :2] / clip[:, 3:]).abs()
        half = torch.maximum(half, torch.quantile(ndc, 1 - quantile, dim=0) if quantile > 0 else ndc.amax(0))
        views.append(look_at(target + direction * distance, target, up))
    adjust = torch.eye(4, dtype=torch.float32, device=device)
    adjust[0, 0] = adjust[1, 1] = fill * zoom / float(half.max().clamp_min(1e-6))
    return torch.stack(views), (adjust @ base).expand(len(views), 4, 4)


def render_views(mesh: Mesh, angles, resolution, mode: str = "shaded", *, projection: str = "orthographic", zoom: float = 1.0,
                 focus=(0.5, 0.5, 0.5), background=None, supersample: int = 2) -> torch.Tensor:
    """Views [B, H, W, C] in 0..1 of a mesh from orbit_cameras (one scale shared by every view), encoded as
    _encoded_views: RGBA with a transparent background for the colour modes by default."""
    view, projection = orbit_cameras(mesh.vertices, angles, resolution, projection=projection, zoom=zoom, focus=focus)
    return _encoded_views(mesh, view, projection, resolution, mode, background, supersample)


# ------------------------------------------------------------------------------------------------------------ splats
SH_C0, SH_C1 = 0.28209479177387814, 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def splat_colors(splats: dict, directions: torch.Tensor) -> torch.Tensor:
    """Colours [N, 3] of splats seen along unit directions [N, 3] (from the camera to each splat), evaluating their
    spherical harmonics (degrees 0 to 3) as 3D Gaussian Splatting does: 0.5 + sum of the SH terms, clamped at 0."""
    sh = [splats["sh_dc"].float()] + ([] if splats["sh_rest"] is None else list(splats["sh_rest"].float().unbind(1)))
    x, y, z = (d[:, None] for d in directions.unbind(1))
    result = SH_C0 * sh[0]
    if len(sh) > 1:
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]
    if len(sh) > 4:
        xx, yy, zz = x * x, y * y, z * z
        result = (result + SH_C2[0] * x * y * sh[4] + SH_C2[1] * y * z * sh[5] + SH_C2[2] * (2 * zz - xx - yy) * sh[6] + SH_C2[3] * x * z * sh[7]
                  + SH_C2[4] * (xx - yy) * sh[8])
        if len(sh) > 9:
            result = (result + SH_C3[0] * y * (3 * xx - yy) * sh[9] + SH_C3[1] * x * y * z * sh[10] + SH_C3[2] * y * (4 * zz - xx - yy) * sh[11]
                      + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[12] + SH_C3[4] * x * (4 * zz - xx - yy) * sh[13]
                      + SH_C3[5] * z * (xx - yy) * sh[14] + SH_C3[6] * x * (xx - 3 * yy) * sh[15])
    return (result + 0.5).clamp_min(0)


def _quaternion_matrices(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = F.normalize(q.float(), dim=1).unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)


def _splat_framing(means: torch.Tensor, quantile: float = 0.02):
    """Canonical 3/4 camera on the bulk of the splats (quantiles, so that floaters do not shrink the object)."""
    sample = means[::max(1, len(means) // 500000)].float()
    low, high = torch.quantile(sample, quantile, dim=0), torch.quantile(sample, 1 - quantile, dim=0)
    center = (low + high) / 2
    view, projection, _ = _orbit(center, max(float((high - low).norm() / 2), 1e-6), *PREVIEW_VIEW)
    homogeneous = torch.cat([sample, torch.ones_like(sample[:, :1])], 1) @ (projection @ view).T
    ndc = homogeneous[:, :2] / homogeneous[:, 3:]
    return view, _zoom(projection, torch.quantile(ndc, quantile, dim=0), torch.quantile(ndc, 1 - quantile, dim=0), fill=0.85)


def splat_preview(splats: dict, size=512, *, cameras=None, max_radius: int | None = None, fragment_budget: int = 1 << 20) -> torch.Tensor:
    """Thumbnail of Gaussian splats (see splats.py for the dict): RGBA [4, H, W] in [-1, 1] (size: int or (H, W)), transparent
    background, on the device of the splats. Pure PyTorch (no CUDA extension): each splat is projected as an elliptical
    2D Gaussian (3DGS EWA projection with the 0.3 pixel low-pass), coloured by its spherical harmonics for the view
    direction, and alpha-blended front to back in depth order (one depth per splat, as 3DGS). cameras: (view,
    projection) like render(); by default the canonical 3/4 view framed on the 2-98 % quantiles of the centres. Splat
    footprints are clipped to max_radius pixels (a quarter of the smaller side by default); fragment_budget bounds the memory
    (about 190 bytes per fragment)."""
    means = splats["means"].float()
    device = means.device
    height, width = (size, size) if isinstance(size, int) else size
    view, projection = _splat_framing(means) if cameras is None else (m[0] for m in _cameras(cameras, device))
    max_radius = max_radius or max(2, min(height, width) // 4)
    camera = torch.cat([means, torch.ones_like(means[:, :1])], 1) @ view.T
    clip = camera @ projection.T
    w = clip[:, 3]
    centre = torch.stack([(clip[:, 0] / w + 1) * width / 2 - 0.5, (1 - clip[:, 1] / w) * height / 2 - 0.5], 1)  # pixel centres at integers
    jacobian = (projection[None, :2, :3] * w[:, None, None] - clip[:, :2, None] * projection[None, 3:, :3]) / (w * w)[:, None, None]
    jacobian = (jacobian * torch.tensor([width / 2, -height / 2], device=device)[None, :, None]) @ view[None, :3, :3]
    footprint = jacobian @ (_quaternion_matrices(splats["rotations"]) * splats["scales"].float()[:, None, :])
    covariance = footprint @ footprint.transpose(1, 2) + 0.3 * torch.eye(2, device=device)
    a, b, c = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
    determinant = a * c - b * b
    conic = torch.stack([c, -b, a], 1) / determinant[:, None]
    radius = torch.ceil(3 * ((a + c) / 2 + (((a - c) / 2) ** 2 + b * b).sqrt()).sqrt()).clamp(1, max_radius).long()
    visible = (-camera[:, 2] > 0) & (w > 0) & (splats["opacities"] > 1 / 255) & (determinant > 0)
    visible &= ((centre + radius[:, None] >= 0) & (centre - radius[:, None] <= torch.tensor([width - 1, height - 1], device=device))).all(1)
    order = visible.nonzero().squeeze(1)
    order = order[torch.argsort(-camera[order, 2], stable=True)]  # front to back
    eye = -view[:3, :3].T @ view[:3, 3]
    rest = None if splats["sh_rest"] is None else splats["sh_rest"][order]
    colors = splat_colors({"sh_dc": splats["sh_dc"][order], "sh_rest": rest}, F.normalize(means[order] - eye, dim=1))
    centre, conic, radius, opacity = centre[order], conic[order], radius[order], splats["opacities"][order].float()

    image = torch.zeros(height * width, 3, dtype=torch.float32, device=device)
    log_transmittance = torch.zeros(height * width, dtype=torch.float64, device=device)
    counts = (2 * radius + 1) ** 2
    ends = counts.cumsum(0)
    start = 0
    while start < len(order):  # chunks of whole splats, front to back, about fragment_budget fragments each
        stop = max(int(torch.searchsorted(ends, ends[start] - counts[start] + fragment_budget, right=True)), start + 1)
        span = counts[start:stop]
        splat = torch.repeat_interleave(torch.arange(start, stop, device=device), span)
        local = torch.arange(len(splat), device=device) - torch.repeat_interleave(span.cumsum(0) - span, span)
        side = 2 * radius[splat] + 1
        row = torch.round(centre[splat, 1]).long() + local.div(side, rounding_mode="floor") - radius[splat]
        col = torch.round(centre[splat, 0]).long() + local % side - radius[splat]
        ox, oy = col - centre[splat, 0], row - centre[splat, 1]
        power = -0.5 * (conic[splat, 0] * ox * ox + conic[splat, 2] * oy * oy) - conic[splat, 1] * ox * oy
        alpha = (opacity[splat] * power.exp()).clamp_max(0.99)
        keep = (alpha >= 1 / 255) & (col >= 0) & (col < width) & (row >= 0) & (row < height)
        pixel, alpha, splat = (row * width + col)[keep], alpha[keep].double(), splat[keep]
        pixel, sort = torch.sort(pixel, stable=True)  # stable: the front-to-back order is kept within each pixel
        alpha, splat = alpha[sort], splat[sort]
        attenuation = torch.log1p(-alpha)
        before = attenuation.cumsum(0) - attenuation  # exclusive running sum, made relative to each pixel's first fragment
        first = torch.ones_like(pixel, dtype=torch.bool)
        first[1:] = pixel[1:] != pixel[:-1]
        before = before - before[torch.cummax(torch.where(first, torch.arange(len(pixel), device=device), 0), 0).values]
        weight = ((log_transmittance[pixel] + before).exp() * alpha).float()
        image.index_add_(0, pixel, weight[:, None] * colors[splat])
        log_transmittance.index_add_(0, pixel, attenuation)
        start = stop
    coverage = (1 - log_transmittance.exp()).float()
    rgb = (image / coverage.clamp_min(1e-12)[:, None]).clamp(0, 1)
    return torch.cat([rgb, coverage[:, None]], 1).T.reshape(4, height, width) * 2 - 1


def render_splat_views(splats: dict, angles, resolution, mode: str = "shaded", *, projection: str = "orthographic", zoom: float = 1.0,
                       focus=(0.5, 0.5, 0.5), background=None) -> torch.Tensor:
    """Views [B, H, W, C] in 0..1 of Gaussian splats from orbit_cameras (framed on the 2-98 % quantiles of the centres,
    against floaters): shaded / albedo give the splat colours, RGBA with a transparent background when background is
    None; mask gives the coverage (C = 3)."""
    if mode not in ("shaded", "albedo", "mask"):
        raise ValueError(f"Gaussian splats render in shaded, albedo or mask mode, not {mode!r}")
    height, width = (resolution, resolution) if isinstance(resolution, int) else resolution
    view, projection = orbit_cameras(splats["means"], angles, (height, width), projection=projection, zoom=zoom, focus=focus, quantile=0.02, fill=0.85)
    images = torch.stack([(splat_preview(splats, (height, width), cameras=(v, p)).permute(1, 2, 0) + 1) / 2 for v, p in zip(view, projection)])
    rgb, coverage = images[..., :3], images[..., 3:]
    if mode == "mask":
        return coverage.expand(-1, -1, -1, 3)
    return images if background is None else rgb * coverage + _tensor(background, rgb.device) * (1 - coverage)
