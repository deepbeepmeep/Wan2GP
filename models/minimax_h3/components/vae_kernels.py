"""H3 decoder inference fusions. Preserve the eager path's dtype boundaries."""
import torch
import triton
import triton.language as tl


@triton.jit
def _cast_rope(X, OUT, C, S, ROWS: tl.constexpr, TOKENS: tl.constexpr, HEADS: tl.constexpr,
             SB: tl.constexpr, ST: tl.constexpr, SH: tl.constexpr, CB: tl.constexpr, CT: tl.constexpr,
             D: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, D)
    token = rows // HEADS % TOKENS
    batch = rows // (TOKENS * HEADS)
    offsets = batch * SB + token * ST + (rows % HEADS) * SH
    ptr = X
    out = OUT
    x = tl.load(ptr + offsets[:, None] + cols[None, :], rows[:, None] < ROWS, 0).to(tl.float32)
    # Keep native FP32 RMSNorm outside this kernel. Preserve its BF16 cast and
    # the rounding after each eager BF16 multiply before the rotation add/sub.
    x = x.to(out.dtype.element_ty).to(tl.float32)
    if R:
        partner = tl.where(cols < R // 2, cols + R // 2, cols - R // 2)
        partner = tl.where(cols < R, partner, cols)
        other = tl.gather(x, tl.broadcast_to(partner[None, :], (BLOCK, D)), 1)
        co = batch[:, None] * CB + token[:, None] * CT + cols[None, :]
        cos = tl.load(C + co, (rows[:, None] < ROWS) & (cols[None, :] < R), 0).to(out.dtype.element_ty).to(tl.float32)
        sin = tl.load(S + co, (rows[:, None] < ROWS) & (cols[None, :] < R), 0).to(out.dtype.element_ty).to(tl.float32)
        a = (x * cos).to(out.dtype.element_ty).to(tl.float32)
        b = (other * sin).to(out.dtype.element_ty).to(tl.float32)
        rotated = tl.where(cols[None, :] < R // 2, a - b, a + b)
        x = tl.where(cols[None, :] < R, rotated, x)
    tl.store(out + rows[:, None] * D + cols[None, :], x, rows[:, None] < ROWS)


@triton.jit
def _swiglu(X, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = (i // D) * (2 * D) + i % D
    gate = tl.load(X + offset, i < N, 0).to(tl.float32)
    value = tl.load(X + offset + D, i < N, 0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty).to(tl.float32)
    tl.store(X + offset, activated * value, i < N)


@triton.jit
def _residual(X, SCALE, RESIDUAL, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    scale = tl.load(SCALE + i % D).to(tl.float32)
    residual = tl.load(RESIDUAL + i, i < N, 0).to(tl.float32)
    scaled = (x * scale).to(X.dtype.element_ty).to(tl.float32)
    tl.store(X + i, scaled + residual, i < N)


def cast_rope(q, rotary):
    b, t, heads, dim = q.shape
    cos, sin = (q, q) if rotary is None else rotary
    r = 0 if rotary is None else cos.shape[-1]
    oq = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    _cast_rope[(triton.cdiv(b * t * heads, 16),)](q, oq, cos, sin, b * t * heads, t, heads, *q.stride()[:3], cos.stride(0), cos.stride(1), dim, r, 16, enable_fp_fusion=False)
    return oq


def swiglu_(expanded):
    dim = expanded.shape[-1] // 2
    n = expanded.numel() // 2
    _swiglu[(triton.cdiv(n, 512),)](expanded, n, dim, 512, enable_fp_fusion=False)
    return expanded[..., :dim]


def residual_(branch, scale, residual):
    n = branch.numel()
    _residual[(triton.cdiv(n, 512),)](branch, scale, residual, n, branch.shape[-1], 512, enable_fp_fusion=False)
    return branch


@triton.jit
def _swiglu_contiguous(X, OUT, N, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = (i // D) * (2 * D) + i % D
    gate = tl.load(X + offset, i < N, 0).to(tl.float32)
    value = tl.load(X + offset + D, i < N, 0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty).to(tl.float32)
    tl.store(OUT + i, activated * value, i < N)


def swiglu(expanded):
    """Release the twice-wide projection before the following linear operation."""
    dim = expanded.shape[-1] // 2
    out = torch.empty((*expanded.shape[:-1], dim), device=expanded.device, dtype=expanded.dtype)
    n = out.numel()
    _swiglu_contiguous[(triton.cdiv(n, 512),)](expanded, out, n, dim, 512, enable_fp_fusion=False)
    return out


@triton.jit
def _silu_pad_frames(X, OUT, N, C: tl.constexpr, T: tl.constexpr,
                     H: tl.constexpr, W: tl.constexpr, P: tl.constexpr,
                     FRONT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = i % (W + 2 * P) - P
    row = i // (W + 2 * P) % (H + 2 * P) - P
    t = i // ((W + 2 * P) * (H + 2 * P)) % (T + FRONT) - FRONT
    c = i // ((W + 2 * P) * (H + 2 * P) * (T + FRONT)) % C
    b = i // (C * (W + 2 * P) * (H + 2 * P) * (T + FRONT))
    col = tl.where(col < 0, -col, tl.where(col >= W, 2 * W - 2 - col, col))
    row = tl.where(row < 0, -row, tl.where(row >= H, 2 * H - 2 - row, row))
    offset = (((b * T + t) * C + c) * H + row) * W + col
    x = tl.load(X + offset, (i < N) & (t >= 0), 0).to(tl.float32)
    tl.store(OUT + i, x / (1.0 + tl.exp(-x)), i < N)


def silu_pad_frames(normalized, batch, frames, padding, front):
    _, channels, _, height, width = normalized.shape
    out = torch.empty((batch, channels, frames + front, height + 2 * padding, width + 2 * padding),
                      device=normalized.device, dtype=normalized.dtype)
    n = out.numel()
    _silu_pad_frames[(triton.cdiv(n, 256),)](normalized, out, n, channels, frames, height, width,
                                        padding, front, 256, enable_fp_fusion=False)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
