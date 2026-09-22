"""Optional Kitchen encoder fusion; keep precision and convolution math unchanged."""
import torch
from shared.kernels import kernel_policy

try:
    from comfy_kitchen.backends.cuda import group_norm_silu_pad3d as _group_norm_silu_pad3d
except (ImportError, OSError, RuntimeError):
    _group_norm_silu_pad3d = None
except ValueError as exc:
    if not str(exc).startswith("infer_schema(func):"):
        raise
    _group_norm_silu_pad3d = None

try:
    from comfy_kitchen.backends.cuda import rms_rope_split_half_ as _rms_rope
except (ImportError, OSError, RuntimeError):
    _rms_rope = None
except ValueError as exc:
    if not str(exc).startswith("infer_schema(func):"):
        raise
    _rms_rope = None

_rms_rope_logged = False


class RotaryEmbedding(tuple):
    """Ordinary cos/sin pair with per-decode Kitchen operands; no persistent cache."""
    pass


def prepare_rotary(cos, sin):
    if (kernel_policy.allow_approximate() and not torch.compiler.is_compiling()
            and _rms_rope is not None and cos.is_cuda
            and torch.cuda.get_device_capability(cos.device) == (12, 0)
            and torch.is_inference_mode_enabled()):
        pair = RotaryEmbedding((cos, sin))
        c, s = cos[..., :cos.shape[-1] // 2], sin[..., :sin.shape[-1] // 2]
        pair.kitchen_table = torch.stack((c, -s, s, c), dim=-1).reshape(*c.shape, 2, 2)
        pair.kitchen_scale = torch.ones(64, device=cos.device, dtype=torch.float32)
        return pair
    return cos, sin


def can_fuse_rms_rope(x, rotary):
    return (kernel_policy.allow_approximate() and _rms_rope is not None
            and x.dtype == torch.bfloat16 and hasattr(rotary, 'kitchen_table'))


def rms_rope(q, k, rotary, eps):
    global _rms_rope_logged
    if not _rms_rope_logged:
        print("[H3 VAE] Comfy Kitchen fused Q/K RMSNorm + RoPE is being used.")
        _rms_rope_logged = True
    return _rms_rope(q, k, rotary.kitchen_table, rotary.kitchen_scale,
                     epsilon=eps, rot_dim=rotary[0].shape[-1])


def available(x, norm, pad):
    batch, channels, frames, height, width = x.shape
    left, right, top, bottom, front = pad
    return (kernel_policy.allow_approximate() and _group_norm_silu_pad3d is not None and x.is_cuda
            and torch.cuda.get_device_capability(x.device) == (12, 0)
            and torch.is_inference_mode_enabled()
            and x.dtype in (torch.float16, torch.bfloat16)
            and norm.weight.dtype == x.dtype and norm.bias.dtype == x.dtype
            and channels % 8 == 0 and 256 % (channels // 8) == 0
            and channels % norm.num_groups == 0 and norm.num_groups <= 1024
            and max(left, right) < width and max(top, bottom) < height
            and batch * (frames + front) <= 65535)


def group_norm_silu_pad3d(x, norm, pad):
    return _group_norm_silu_pad3d(x, norm.weight, norm.bias, norm.num_groups, norm.eps, list(pad), True)
