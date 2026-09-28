"""Optional shared-input projection fusion; normal module calls own all fallbacks."""
import torch
from shared.kernels import int8_backend, kernel_policy

try:
    from . import denoiser_triton
except ImportError:
    denoiser_triton = None

ENABLED = True
_reported = False
_rope_reported = False


def project_many(modules, x):
    global _reported
    if not torch.compiler.is_compiling() and ENABLED and x.is_contiguous() and all(int8_backend.can_fuse_linear(m, x) for m in modules):
        if not _reported:
            print('[LTX] Comfy Kitchen shared-input projection quantization is being used.')
            _reported = True
        outputs = int8_backend.linear_multi(modules, x.reshape(-1, x.shape[-1]))
        return tuple(out.reshape(*x.shape[:-1], out.shape[-1]) for out in outputs)
    return tuple(module(x) for module in modules)


def scale_shift(x, scale, shift, in_place):
    if (not ENABLED or denoiser_triton is None or not kernel_policy.allow_approximate()
            or torch.is_grad_enabled() or torch.compiler.is_compiling() or not x.is_cuda
            or int8_backend.triton._is_fake_tensor(x)
            or x.ndim != 3 or scale.ndim != 3 or scale.shape != shift.shape
            or x.dtype != torch.bfloat16 or scale.dtype != x.dtype or shift.dtype != x.dtype
            or scale.device != x.device or shift.device != x.device
            or scale.shape[0] != x.shape[0] or scale.shape[-1] != x.shape[-1]
            or scale.shape[1] == 0 or x.shape[1] % scale.shape[1]
            or not all(t.is_contiguous() for t in (x, scale, shift))
            or torch.cuda.get_device_capability(x.device) != (12, 0)):
        return None
    return denoiser_triton.scale_shift(x, scale, shift, in_place)


def split_rope(x, cache, layout):
    global _rope_reported
    if (not ENABLED or denoiser_triton is None or not kernel_policy.allow_approximate()
            or torch.is_grad_enabled() or torch.compiler.is_compiling() or not x.is_cuda
            or int8_backend.triton._is_fake_tensor(x)
            or x.dtype != torch.bfloat16 or x.stride(-1) != 1 or x.stride(1) != x.shape[2]
            or len(layout.axis_cos) not in (1, 3)
            or not 0 <= cache.pad_size < len(layout.axis_cos)
            or (x.shape[0] > 1 and x.stride(0) < x.shape[1] * x.shape[2])
            or torch.cuda.get_device_capability(x.device) != (12, 0)):
        return False
    dtype = torch.float32 if cache.use_fp32_freqs else x.dtype
    if any(t.device != x.device or t.dtype != dtype or not t.is_contiguous()
           for t in (*layout.axis_cos, *layout.axis_sin)):
        return False
    denoiser_triton.split_rope(x, cache, layout)
    if not _rope_reported:
        print('[LTX] Triton compact rotary kernel is being used.')
        _rope_reported = True
    return True
