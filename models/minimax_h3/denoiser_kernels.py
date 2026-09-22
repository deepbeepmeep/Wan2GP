"""Optional H3 denoiser fusions; keep attention and AdaLN semantics unchanged."""
import torch
from shared.kernels import int8_backend, kernel_policy

try:
    from comfy_kitchen.backends.cuda import rms_rope_split_half_ as _rms_rope
except (ImportError, OSError, RuntimeError):
    _rms_rope = None
except ValueError as exc:
    if not str(exc).startswith("infer_schema(func):"):
        raise
    _rms_rope = None

try:
    from shared.sol_attn import qk_rms_norm_rope_ as _triton_rms_rope
except (ImportError, OSError, RuntimeError):
    _triton_rms_rope = None

try:
    from .denoiser_triton import norm_modulate as _norm_modulate
except (ImportError, OSError, RuntimeError):
    _norm_modulate = None

ENABLED = True
DEEP_FUSIONS = True
_announced = False


def available(x):
    return (ENABLED and kernel_policy.allow_approximate() and not torch.compiler.is_compiling()
            and torch.is_inference_mode_enabled() and x.is_cuda and x.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(x.device) == (12, 0))


def can_linear(layer, x):
    return not torch.compiler.is_compiling() and ENABLED and int8_backend.can_fuse_linear(layer, x)


def norm_modulate(x, norm, shift, scale, segments):
    if (DEEP_FUSIONS and _norm_modulate is not None and available(x) and x.is_contiguous()
            and norm.weight.device == x.device and norm.weight.dtype == x.dtype
            and x.shape[-1] <= 8192):
        return _norm_modulate(x, norm, shift, scale, segments)
    return None


def prepare_rope(rope):
    if not available(rope):
        return rope
    if _rms_rope is None:
        return (rope, None) if _triton_rms_rope is not None else rope
    c, s = rope[..., 0], rope[..., 1]
    return rope, torch.stack((c, -s, s, c), dim=-1).reshape(*c.shape, 2, 2)


def can_rms(x, packed, q_norm, k_norm):
    return (available(x) and ((_rms_rope is not None and packed is not None) or _triton_rms_rope is not None)
            and all(p.device == x.device and p.dtype == x.dtype for p in (q_norm.weight, k_norm.weight)))


def rms_rope(q, k, rope, packed, q_norm, k_norm):
    global _announced
    kitchen = _rms_rope is not None and packed is not None
    if not _announced:
        print(f'[H3] {"Kitchen" if kitchen else "Triton"} denoiser Q/K RMSNorm + RoPE fusion is being used.')
        _announced = True
    if not kitchen:
        _triton_rms_rope(q, k, q_norm.weight, k_norm.weight, rope, q_norm.eps)
        return q, k
    return _rms_rope(q, k, packed, q_norm.weight, k_norm.weight,
                     epsilon=q_norm.eps, rot_dim=packed.shape[-3] * 2)


def gated_linear(layer, x, residual, gate, segments):
    """Each modality/timestep segment has its own residual gate."""
    for start, stop, row in segments:
        if stop <= start:
            continue
        out = int8_backend.linear_with_fusion(layer, x[start:stop], residual=residual[start:stop],
                                              residual_scale=gate[row].to(x.dtype),
                                              out=residual[start:stop] if DEEP_FUSIONS else None)
        residual[start:stop].copy_(out)
    return residual
