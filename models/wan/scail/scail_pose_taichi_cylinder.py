"""
Taichi cylinder renderer (GPU) from the official SCAIL-Pose implementation.

Upstream: https://github.com/zai-org/SCAIL-Pose (render_3d/taichi_cylinder.py)

Notes for WanGP integration:
- Initialization is done lazily to avoid import-time side effects.
- If there are no cylinder specs at all, we return black frames instead of crashing.
"""

from __future__ import annotations

import numpy as np

# Lazy import to avoid printing version message at app startup
ti = None
_TAICHI_INITIALIZED = False


def _ensure_taichi_init() -> None:
    global ti, _TAICHI_INITIALIZED
    if _TAICHI_INITIALIZED:
        return
    import taichi as _ti
    ti = _ti
    try:
        ti.init(arch=ti.cuda, log_level=ti.ERROR)
    except Exception as exc:  # pragma: no cover
        # Allow re-import/re-init patterns; keep GPU-only otherwise.
        msg = str(exc).lower()
        if "already" in msg and "init" in msg:
            _TAICHI_INITIALIZED = True
            return
        raise
    _TAICHI_INITIALIZED = True


def flatten_specs(specs_list):
    """Flatten `specs_list` into numpy arrays + frame offsets/counts."""
    starts, ends, colors = [], [], []
    frame_offset, frame_count = [], []
    offset = 0
    for specs in specs_list:
        frame_offset.append(offset)
        frame_count.append(len(specs))
        for (start, end, color) in specs:
            starts.append(start)
            ends.append(end)
            colors.append(color)
        offset += len(specs)

    starts_np = np.asarray(starts, dtype=np.float32).reshape(-1, 3)
    ends_np = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
    colors_np = np.asarray(colors, dtype=np.float32).reshape(-1, 4)
    frame_offset_np = np.asarray(frame_offset, dtype=np.int32)
    frame_count_np = np.asarray(frame_count, dtype=np.int32)
    return starts_np, ends_np, colors_np, frame_offset_np, frame_count_np


def render_whole(specs_list, H=480, W=640, fx=500, fy=500, cx=240, cy=320, radius=21.5):
    _ensure_taichi_init()

    if len(specs_list) == 0:
        return []

    img = ti.Vector.field(4, dtype=ti.f32, shape=(H, W))
    starts, ends, colors, frame_offset, frame_count = flatten_specs(specs_list)
    total_cyl = int(starts.shape[0])
    n_frames = len(specs_list)

    if total_cyl == 0:
        return [np.zeros((H, W, 4), dtype=np.uint8) for _ in range(n_frames)]

    z_min = float(min(starts[:, 2].min(), ends[:, 2].min()))
    z_max = float(max(starts[:, 2].max(), ends[:, 2].max()))

    # ========= camera =========
    znear = 0.1
    zfar = max(min(z_max, 25000), 10000)
    C = ti.Vector([0.0, 0.0, 0.0])
    light_dir = ti.Vector([0.0, 0.0, 1.0])

    c_start = ti.Vector.field(3, dtype=ti.f32, shape=total_cyl)
    c_end = ti.Vector.field(3, dtype=ti.f32, shape=total_cyl)
    c_rgba = ti.Vector.field(4, dtype=ti.f32, shape=total_cyl)
    f_offset = ti.field(dtype=ti.i32, shape=n_frames)
    f_count = ti.field(dtype=ti.i32, shape=n_frames)
    frame_id = ti.field(dtype=ti.i32, shape=())
    z_min_field = ti.field(dtype=ti.f32, shape=())
    z_max_field = ti.field(dtype=ti.f32, shape=())

    z_min_field[None] = z_min
    z_max_field[None] = z_max

    c_start.from_numpy(starts)
    c_end.from_numpy(ends)
    c_rgba.from_numpy(colors)
    f_offset.from_numpy(frame_offset)
    f_count.from_numpy(frame_count)

    @ti.func
    def sd_cylinder(p, a, b, r):
        pa = p - a
        ba = b - a
        h = ba.norm()
        eps = 1e-8
        res = 0.0
        if h < eps:
            res = pa.norm() - r
        else:
            ba_n = ba / h
            proj = pa.dot(ba_n)
            proj_clamped = min(max(proj, 0.0), h)
            res = (pa - proj_clamped * ba_n).norm() - r
        return res

    @ti.func
    def scene_sdf(p):
        best_d = 1e6
        best_col = ti.Vector([0.0, 0.0, 0.0, 0.0])
        fid = frame_id[None]
        off = f_offset[fid]
        cnt = f_count[fid]
        for i in range(cnt):
            a = c_start[off + i]
            b = c_end[off + i]
            col = c_rgba[off + i]
            d = sd_cylinder(p, a, b, radius)
            if d < best_d:
                best_d = d
                best_col = col
        return best_d, best_col

    @ti.func
    def get_normal(p):
        e = 1e-3
        dx = scene_sdf(p + ti.Vector([e, 0.0, 0.0]))[0] - scene_sdf(p - ti.Vector([e, 0.0, 0.0]))[0]
        dy = scene_sdf(p + ti.Vector([0.0, e, 0.0]))[0] - scene_sdf(p - ti.Vector([0.0, e, 0.0]))[0]
        dz = scene_sdf(p + ti.Vector([0.0, 0.0, e]))[0] - scene_sdf(p - ti.Vector([0.0, 0.0, e]))[0]
        n = ti.Vector([dx, dy, dz])
        return n.normalized()

    @ti.func
    def pixel_to_ray(xi, yi):
        u = (xi - cx) / fx
        v = (yi - cy) / fy
        dir_cam = ti.Vector([u, v, 1.0]).normalized()
        Rcw = ti.Matrix.identity(ti.f32, 3)
        rd_world = Rcw @ dir_cam
        ro_world = C
        return ro_world, rd_world

    @ti.kernel
    def render():
        depth_near = ti.max(z_min_field[None], 0.1)
        depth_far = ti.min(z_max_field[None] + 6000, 20000)
        for y, x in img:
            ro, rd = pixel_to_ray(x, y)
            t = znear
            col_out = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for _ in range(300):
                p = ro + rd * t
                d, col = scene_sdf(p)
                if d < 1e-3:
                    n = get_normal(p)
                    diff = max(n.dot(-light_dir), 0.0)

                    view_dir = -rd.normalized()
                    half_dir = (view_dir + -light_dir).normalized()
                    spec = max(n.dot(half_dir), 0.0) ** 32

                    depth_factor = 1.0 - (p.z - depth_near) / (depth_far - znear)
                    depth_factor = ti.max(0.0, ti.min(1.0, depth_factor))

                    diffuse_term = 0.3 + 0.7 * diff
                    base = col.xyz * diffuse_term * depth_factor
                    highlight = ti.Vector([1.0, 1.0, 1.0]) * (0.5 * spec) * depth_factor

                    col_out = ti.Vector(
                        [base.x + highlight.x, base.y + highlight.y, base.z + highlight.z, col.w]
                    )
                    break

                if t > zfar:
                    break
                t += max(d, 1e-4)
            img[y, x] = col_out

    frames_np_rgba = []
    for f in range(n_frames):
        frame_id[None] = f
        render()
        arr = np.clip(img.to_numpy(), 0, 1)
        arr8 = (arr * 255).astype(np.uint8)
        frames_np_rgba.append(arr8)

    return frames_np_rgba

