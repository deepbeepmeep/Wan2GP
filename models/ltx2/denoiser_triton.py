"""LTX modulation and compact split RoPE, preserving native rounding and layout."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _scale_shift(X, SCALE, SHIFT, Y, N, D: tl.constexpr, L: tl.constexpr,
                 T: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D
    s = (row // L * T + row % L // (L // T)) * D + col
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    scale = tl.load(SCALE + s, i < N, 0).to(tl.float32)
    shift = tl.load(SHIFT + s, i < N, 0).to(tl.float32)
    factor = (1.0 + scale).to(X.dtype.element_ty).to(tl.float32)
    value = (x * factor).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, value + shift, i < N)


def scale_shift(x, scale, shift, in_place):
    out = x if in_place else torch.empty_like(x)
    _scale_shift[(triton.cdiv(x.numel(), 1024),)](
        x, scale, shift, out, x.numel(), x.shape[2], x.shape[1], scale.shape[1], 1024,
        enable_fp_fusion=False)
    return out


@triton.jit
def _split_rope(X, C0, S0, C1, S1, C2, S2, N, TOKENS: tl.constexpr,
                WIDTH: tl.constexpr, HALF: tl.constexpr, PAD: tl.constexpr,
                AXES: tl.constexpr, GRID: tl.constexpr, AXIS_IDS: tl.constexpr,
                FREQ_WIDTH: tl.constexpr, BATCH_STRIDE: tl.constexpr,
                FP32: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    f = i % (WIDTH // 2)
    token = i // (WIDTH // 2) % TOKENS
    batch = i // (WIDTH // 2) // TOKENS
    head, local = f // HALF, f % HALF
    offset = batch * BATCH_STRIDE + token * WIDTH + head * HALF * 2 + local
    axis = (f - PAD) % AXES
    col = (f - PAD) // AXES
    c = tl.full((BLOCK,), 1.0, tl.float32)
    s = tl.full((BLOCK,), 0.0, tl.float32)
    for a in tl.static_range(AXES):
        divisor = 1
        for j in tl.static_range(len(GRID)):
            if j > AXIS_IDS[a]:
                divisor *= GRID[j]
        row = token // divisor % GRID[AXIS_IDS[a]]
        valid = (i < N) & (f >= PAD) & (axis == a)
        if a == 0:
            cv = tl.load(C0 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
            sv = tl.load(S0 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
        elif a == 1:
            cv = tl.load(C1 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
            sv = tl.load(S1 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
        else:
            cv = tl.load(C2 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
            sv = tl.load(S2 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
        c = tl.where(valid, cv, c)
        s = tl.where(valid, sv, s)
    x0 = tl.load(X + offset, i < N, 0).to(tl.float32)
    x1 = tl.load(X + offset + HALF, i < N, 0).to(tl.float32)
    # Explicit rounding intrinsics also prevent packed arithmetic substitutions.
    a, b = libdevice.mul_rn(x0, c), libdevice.mul_rn(x1, c)
    if not FP32:
        a = a.to(X.dtype.element_ty).to(tl.float32)
        b = b.to(X.dtype.element_ty).to(tl.float32)
    # Preserve the native grouped layout's padding in every head group.
    valid = (i < N) & ((head % AXES != 0) | (local >= PAD))
    tl.store(X + offset, libdevice.sub_rn(a, libdevice.mul_rn(x1, s)), valid)
    tl.store(X + offset + HALF, libdevice.add_rn(b, libdevice.mul_rn(x0, s)), valid)


def split_rope(x, cache, layout):
    cos, sin = layout.axis_cos, layout.axis_sin
    pointers = tuple(value for a in range(3) for value in (cos[min(a, len(cos)-1)], sin[min(a, len(sin)-1)]))
    size = x.numel() // 2
    return _split_rope[(triton.cdiv(size, 256),)](
        x, *pointers, size, x.shape[1], x.shape[2], x.shape[2] // cache.num_attention_heads // 2,
        cache.pad_size, len(cos), layout.grid_sizes, cache.rope_axes, cos[0].shape[1],
        x.stride(0), cache.use_fp32_freqs, 256, enable_fp_fusion=False)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
