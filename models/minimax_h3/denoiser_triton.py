"""H3 modulation fusion; retain native RMSNorm and BF16 rounding boundaries."""
import triton
import triton.language as tl


@triton.jit
def _modulate(X, SCALE, SHIFT, N, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    col = i % D
    x = tl.load(X + i, valid, 0).to(tl.float32)
    scale = tl.load(SCALE + col, valid, 0).to(X.dtype.element_ty).to(tl.float32)
    shift = tl.load(SHIFT + col, valid, 0).to(X.dtype.element_ty).to(tl.float32)
    factor = (1.0 + scale).to(X.dtype.element_ty).to(tl.float32)
    scaled = (x * factor).to(X.dtype.element_ty).to(tl.float32)
    tl.store(X + i, scaled + shift, valid)


def norm_modulate(x, norm, shift, scale, segments):
    output = norm(x)
    width = x.shape[-1]
    for start, stop, row in segments:
        if stop > start:
            size = (stop-start) * width
            _modulate[(triton.cdiv(size, 1024),)](
                output[start:stop], scale[row], shift[row], size, width, 1024,
                enable_fp_fusion=False)
    return output


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
