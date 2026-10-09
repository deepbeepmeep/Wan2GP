# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Triton kernels of the narrow-band remesh (remesh.py), bit-identical to its PyTorch path.

- Lookups: the grid corners and the crossing cells are unique int32 coordinates sorted in (x, y, z) order, so a column table (the first
  row of every (x, y) column) and a binary search of z within the column find a voxel; no sorted 64-bit keys and no searchsorted over
  millions of keys.
- Splat: one program per group of triangles walks the grid points of each triangle's box, computes the point-triangle distance with the
  arithmetic of remesh._point_triangle_distance on PyTorch's CUDA kernels (separately rounded operations; three-term sums in the order
  (0 + 2) + 1 of PyTorch's reductions; the cross product's first product fused, as nvcc compiles it; IEEE division and square roots), and
  lowers the corner's distance with an integer atomic minimum (non-negative floats order as their bits). No (triangle, point) pairs are
  materialized.

Counts and coordinates are not specialized: one binary per kernel. Launch configurations start from a static choice and are tuned once
per kernel and GPU on scratch outputs."""
import functools

import torch
import triton
import triton.language as tl

from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_trellis2_remesh_kernels", 2, "TRELLIS.2 Remesh")
_TUNE_ROWS = 1 << 20  # calls with at least this many rows (or triangles) tune; the scratch work is bounded to this many


@functools.cache
def supported(device):
    """CUDA GPUs with compute capability 8.0+; other devices use the PyTorch path."""
    return device.type == "cuda" and torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0)


class ColumnIndex:
    """Lookup of unique int32 coords (N, 3) sorted in (x, y, z) order: start row of each (x, y) column of their bounding rectangle."""

    def __init__(self, coords):
        self.coords = coords
        (self.x0, self.y0), high = coords[:, :2].amin(0).tolist(), coords[:, :2].amax(0).tolist()
        self.sx, self.sy = high[0] - self.x0 + 1, high[1] - self.y0 + 1
        column_of = (coords[:, 0] - self.x0).mul_(self.sy).add_(coords[:, 1]).sub_(self.y0)  # int32: below sx * sy
        self.table = torch.searchsorted(column_of, torch.arange(self.sx * self.sy + 1, device=coords.device, dtype=torch.int32), out_int32=True)
        del column_of
        self.steps = max(1, int((self.table[1:] - self.table[:-1]).max()).bit_length())  # binary search steps of the longest column

    def args(self):
        return self.table, self.coords, self.x0, self.y0, self.sx, self.sy, self.steps


@triton.jit
def _lower_bound(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask):
    """(first row of column (x, y) whose z >= z, end row of the column, valid) for unique coords sorted in (x, y, z) order."""
    cx = x - X0
    cy = y - Y0
    valid = mask & (cx >= 0) & (cx < SX) & (cy >= 0) & (cy < SY)
    column = tl.where(valid, cx.to(tl.int64) * SY + cy, 0)
    low = tl.load(TABLE + column, valid, 0)
    high = tl.load(TABLE + column + 1, valid, 0)
    end = high
    for _ in range(STEPS):
        active = low < high
        middle = (low + high) // 2
        right = tl.load(COORDS + middle.to(tl.int64) * 3 + 2, active, 0) < z
        low = tl.where(active & right, middle + 1, low)
        high = tl.where(active & ~right, middle, high)
    return low, end, valid


@triton.jit
def _find(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask):
    """Row of voxel (x, y, z), -1 where there is none."""
    row, end, valid = _lower_bound(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask)
    valid = valid & (row < end)
    hit = valid & (tl.load(COORDS + row.to(tl.int64) * 3 + 2, valid, 0) == z)
    return tl.where(hit, row, -1)


@triton.jit
def _find_pair(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask):
    """Rows of voxels (x, y, z) and (x, y, z + 1), -1 where there is none."""
    row, end, valid = _lower_bound(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask)
    valid = valid & (row < end)
    hit = valid & (tl.load(COORDS + row.to(tl.int64) * 3 + 2, valid, 0) == z)
    next_row = row + hit.to(tl.int32)
    valid = valid & (next_row < end)
    next_hit = valid & (tl.load(COORDS + next_row.to(tl.int64) * 3 + 2, valid, 0) == z + 1)
    return tl.where(hit, row, -1), tl.where(next_hit, next_row, -1)


@triton.jit(do_not_specialize=["N", "X0", "Y0", "SX", "SY", "STEPS"])
def _find_kernel(QUERY, OUT, TABLE, COORDS, N, X0, Y0, SX, SY, STEPS, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < N
    x = tl.load(QUERY + rows * 3, mask, 0)
    y = tl.load(QUERY + rows * 3 + 1, mask, 0)
    z = tl.load(QUERY + rows * 3 + 2, mask, 0)
    tl.store(OUT + rows, _find(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask), mask)


def find(index, query):
    """ColumnIndex rows of the voxels at query (..., 3) int32, int32, -1 where there is none (SparseTensor.find)."""
    query = query.contiguous()
    out = torch.empty(query.shape[:-1], dtype=torch.int32, device=query.device)
    count = out.numel()

    def launch(target, rows, config):
        block, warps = config
        _find_kernel[(triton.cdiv(rows, block),)](query, target, *index.args()[:2], rows, *index.args()[2:], block, num_warps=warps)

    def scratch():
        target = torch.empty(min(count, _TUNE_ROWS), dtype=torch.int32, device=query.device)
        return lambda config: launch(target, target.numel(), config)

    if count:
        launch(out, count, _TUNER.config("find", "int32", query.device, ((256, 4), (512, 4), (128, 4), (1024, 8)), scratch, count >= _TUNE_ROWS))
    return out


@triton.jit
def _cell_corners(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask):
    """Rows of the 8 corners of the cells whose min corner is (x, y, z), in remesh._CORNERS order (i + 2 j + 4 k)."""
    c0, c4 = _find_pair(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask)
    c1, c5 = _find_pair(TABLE, COORDS, x + 1, y, z, X0, Y0, SX, SY, STEPS, mask)
    c2, c6 = _find_pair(TABLE, COORDS, x, y + 1, z, X0, Y0, SX, SY, STEPS, mask)
    c3, c7 = _find_pair(TABLE, COORDS, x + 1, y + 1, z, X0, Y0, SX, SY, STEPS, mask)
    return c0, c1, c2, c3, c4, c5, c6, c7


@triton.jit
def _negative(VALUE, row, mask):
    return (tl.load(VALUE + row, mask, 0.0) < 0).to(tl.int32)


@triton.jit(do_not_specialize=["N", "X0", "Y0", "SX", "SY", "STEPS"])
def _crossing_kernel(VALUE, FLAG, TABLE, COORDS, N, X0, Y0, SX, SY, STEPS, BLOCK: tl.constexpr):
    """FLAG[i] = the cell whose min corner is corner i has its 8 corners and a sign change of VALUE."""
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < N
    x = tl.load(COORDS + rows * 3, mask, 0)
    y = tl.load(COORDS + rows * 3 + 1, mask, 0)
    z = tl.load(COORDS + rows * 3 + 2, mask, 0)
    c0, c1, c2, c3, c4, c5, c6, c7 = _cell_corners(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask)
    complete = (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (c3 >= 0) & (c4 >= 0) & (c5 >= 0) & (c6 >= 0) & (c7 >= 0)
    negative = _negative(VALUE, c0, complete) + _negative(VALUE, c1, complete) + _negative(VALUE, c2, complete) + _negative(VALUE, c3, complete)
    negative += _negative(VALUE, c4, complete) + _negative(VALUE, c5, complete) + _negative(VALUE, c6, complete) + _negative(VALUE, c7, complete)
    tl.store(FLAG + rows, (complete & (negative > 0) & (negative < 8)).to(tl.int8), mask)


@triton.jit(do_not_specialize=["N", "X0", "Y0", "SX", "SY", "STEPS"])
def _cell_values_kernel(CELLS, VALUE, OUT, TABLE, COORDS, N, X0, Y0, SX, SY, STEPS, BLOCK: tl.constexpr):
    """OUT[i, c] = VALUE at corner c of the cell whose min corner is corner CELLS[i] (complete cells)."""
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < N
    cell = tl.load(CELLS + rows, mask, 0)
    x = tl.load(COORDS + cell * 3, mask, 0)
    y = tl.load(COORDS + cell * 3 + 1, mask, 0)
    z = tl.load(COORDS + cell * 3 + 2, mask, 0)
    c0, c1, c2, c3, c4, c5, c6, c7 = _cell_corners(TABLE, COORDS, x, y, z, X0, Y0, SX, SY, STEPS, mask)
    tl.store(OUT + rows * 8, tl.load(VALUE + c0, mask, 0.0), mask)
    tl.store(OUT + rows * 8 + 1, tl.load(VALUE + c1, mask, 0.0), mask)
    tl.store(OUT + rows * 8 + 2, tl.load(VALUE + c2, mask, 0.0), mask)
    tl.store(OUT + rows * 8 + 3, tl.load(VALUE + c3, mask, 0.0), mask)
    tl.store(OUT + rows * 8 + 4, tl.load(VALUE + c4, mask, 0.0), mask)
    tl.store(OUT + rows * 8 + 5, tl.load(VALUE + c5, mask, 0.0), mask)
    tl.store(OUT + rows * 8 + 6, tl.load(VALUE + c6, mask, 0.0), mask)
    tl.store(OUT + rows * 8 + 7, tl.load(VALUE + c7, mask, 0.0), mask)


def crossing_cells(index, value):
    """(min corner coords (M, 3) int32, corner values (M, 8)) of the cells of the ColumnIndex corners whose 8 corners exist and whose
    values change sign, in corner order (remesh_narrow_band's crossing cells)."""
    count = len(index.coords)
    flag = torch.empty(count, dtype=torch.int8, device=value.device)

    def launch(target, rows, config):
        block, warps = config
        _crossing_kernel[(triton.cdiv(rows, block),)](value, target, *index.args()[:2], rows, *index.args()[2:], block, num_warps=warps)

    def scratch():
        target = torch.empty(min(count, _TUNE_ROWS), dtype=torch.int8, device=value.device)
        return lambda config: launch(target, target.numel(), config)

    launch(flag, count, _TUNER.config("crossing", "fp32", value.device, ((256, 4), (512, 4), (128, 4), (256, 8)), scratch, count >= _TUNE_ROWS))
    cells = flag.nonzero().view(-1)
    del flag
    values = torch.empty((len(cells), 8), dtype=value.dtype, device=value.device)

    def gather(target, rows, config):
        block, warps = config
        _cell_values_kernel[(triton.cdiv(rows, block),)](cells, value, target, *index.args()[:2], rows, *index.args()[2:], block, num_warps=warps)

    def gather_scratch():
        target = torch.empty((min(len(cells), _TUNE_ROWS), 8), dtype=value.dtype, device=value.device)
        return lambda config: gather(target, len(target), config)

    if len(cells):
        gather(values, len(cells), _TUNER.config("cell_values", "fp32", value.device, ((256, 4), (512, 4), (128, 4), (256, 8)), gather_scratch, len(cells) >= _TUNE_ROWS))
    return index.coords[cells], values


@triton.jit
def _sum3(a0, a1, a2):
    return (a0 + a2) + a1  # PyTorch's CUDA sum(-1) of three contiguous values


@triton.jit
def _segment(px, py, pz, sx, sy, sz, ex, ey, ez):
    """remesh._point_triangle_distance's segment(start, end): distance of the points to the segment."""
    dx = ex - sx
    dy = ey - sy
    dz = ez - sz
    qx = px - sx
    qy = py - sy
    qz = pz - sz
    t = tl.div_rn(_sum3(qx * dx, qy * dy, qz * dz), tl.maximum(_sum3(dx * dx, dy * dy, dz * dz), 1e-20))
    t = tl.minimum(tl.maximum(t, 0.0), 1.0)
    rx = qx - t * dx
    ry = qy - t * dy
    rz = qz - t * dz
    return tl.sqrt_rn(_sum3(rx * rx, ry * ry, rz * rz))  # PyTorch's CUDA norm(dim=-1) of three values


@triton.jit(do_not_specialize=["FACE_COUNT", "X0", "Y0", "SX", "SY", "STEPS", "TRIANGLES"])
def _splat_kernel(POINTS, FACES, UDF, TABLE, COORDS, FACE_COUNT, X0, Y0, SX, SY, STEPS, REACH, TRIANGLES, BLOCK: tl.constexpr):
    """Lowers UDF (as int32 bits) at the corners within REACH of each triangle of the program's group to their distance."""
    lanes = tl.arange(0, BLOCK)
    first = tl.program_id(0).to(tl.int64) * TRIANGLES
    last = tl.minimum(first + TRIANGLES, FACE_COUNT)
    for face in range(first, last):
        ia = tl.load(FACES + face * 3)
        ib = tl.load(FACES + face * 3 + 1)
        ic = tl.load(FACES + face * 3 + 2)
        ax = tl.load(POINTS + ia * 3)
        ay = tl.load(POINTS + ia * 3 + 1)
        az = tl.load(POINTS + ia * 3 + 2)
        bx = tl.load(POINTS + ib * 3)
        by = tl.load(POINTS + ib * 3 + 1)
        bz = tl.load(POINTS + ib * 3 + 2)
        cx = tl.load(POINTS + ic * 3)
        cy = tl.load(POINTS + ic * 3 + 1)
        cz = tl.load(POINTS + ic * 3 + 2)
        # box of grid points: floor(min - reach) .. ceil(max + reach)
        lx = tl.floor(tl.minimum(tl.minimum(ax, bx), cx) - REACH).to(tl.int32)
        ly = tl.floor(tl.minimum(tl.minimum(ay, by), cy) - REACH).to(tl.int32)
        lz = tl.floor(tl.minimum(tl.minimum(az, bz), cz) - REACH).to(tl.int32)
        ex = tl.ceil(tl.maximum(tl.maximum(ax, bx), cx) + REACH).to(tl.int32) - lx + 1
        ey = tl.ceil(tl.maximum(tl.maximum(ay, by), cy) + REACH).to(tl.int32) - ly + 1
        ez = tl.ceil(tl.maximum(tl.maximum(az, bz), cz) + REACH).to(tl.int32) - lz + 1
        # per-triangle terms of the distance
        abx = bx - ax
        aby = by - ay
        abz = bz - az
        acx = cx - ax
        acy = cy - ay
        acz = cz - az
        nx = tl.fma(aby, acz, -(abz * acy))
        ny = tl.fma(abz, acx, -(abx * acz))
        nz = tl.fma(abx, acy, -(aby * acx))
        area2 = _sum3(nx * nx, ny * ny, nz * nz)
        d00 = _sum3(abx * abx, aby * aby, abz * abz)
        d01 = _sum3(abx * acx, aby * acy, abz * acz)
        d11 = _sum3(acx * acx, acy * acy, acz * acz)
        denominator = tl.maximum(d00 * d11 - d01 * d01, 1e-20)
        area = tl.sqrt_rn(tl.maximum(area2, 1e-20))
        flat = area2 > 1e-20
        count = ex * ey * ez
        for start in range(0, count, BLOCK):
            rank = start + lanes
            valid = rank < count
            gx = lx + rank % ex
            gy = ly + rank // ex % ey
            gz = lz + rank // (ex * ey)
            px = gx.to(tl.float32)
            py = gy.to(tl.float32)
            pz = gz.to(tl.float32)
            apx = px - ax
            apy = py - ay
            apz = pz - az
            d20 = _sum3(apx * abx, apy * aby, apz * abz)
            d21 = _sum3(apx * acx, apy * acy, apz * acz)
            v = tl.div_rn(d11 * d20 - d01 * d21, denominator)
            w = tl.div_rn(d00 * d21 - d01 * d20, denominator)
            inside = (v >= 0) & (w >= 0) & (v + w <= 1) & flat
            plane = tl.div_rn(tl.abs(_sum3(apx * nx, apy * ny, apz * nz)), area)
            edges = tl.minimum(tl.minimum(_segment(px, py, pz, ax, ay, az, bx, by, bz), _segment(px, py, pz, bx, by, bz, cx, cy, cz)),
                               _segment(px, py, pz, cx, cy, cz, ax, ay, az))
            distance = tl.where(inside, plane, edges)
            near = valid & (distance < REACH)
            row = _find(TABLE, COORDS, gx, gy, gz, X0, Y0, SX, SY, STEPS, near)
            hit = row >= 0
            bits = distance.to(tl.int32, bitcast=True)
            target = UDF + row.to(tl.int64)
            lower = hit & (bits < tl.load(target, hit, 0))
            tl.atomic_min(target, bits, mask=lower)


def unsigned_distance(points, faces, index, reach):
    """remesh._unsigned_distance with the corners of a ColumnIndex: distance from each corner to the triangles, exact below `reach`
    voxels and `reach` beyond. points (V, 3) FP32 and faces (F, 3) int64 contiguous."""
    udf = torch.full((len(index.coords),), float(reach), device=points.device)
    bits = udf.view(torch.int32)
    count = len(faces)
    triangles = 8

    def launch(target, faces_count, config):
        block, warps = config
        _splat_kernel[(triton.cdiv(faces_count, triangles),)](points, faces, target, *index.args()[:2], faces_count, *index.args()[2:],
                                                             float(reach), triangles, block, num_warps=warps, enable_fp_fusion=False)

    def scratch():
        target = torch.full_like(bits, int(udf[:1].view(torch.int32)))
        return lambda config: launch(target, min(count, _TUNE_ROWS), config)

    # one point per thread: with several, Triton packs the FP32 arithmetic into paired instructions that ptxas fuses into FMAs on Blackwell
    # GPUs (sm_120, Triton 3.6), whose roundings differ from PyTorch's
    launch(bits, count, _TUNER.config("splat", "fp32", points.device, ((128, 4), (64, 2), (256, 8), (32, 1)), scratch, count >= _TUNE_ROWS))
    return udf


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
