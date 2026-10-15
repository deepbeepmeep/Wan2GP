# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Activation operations shared by denoisers.

Owned inputs use x_list through the dispatch boundary. In-place operations borrow
their destination; residuals, weights and conditioning are always borrowed. Math
contracts are explicit: split-half RMS/RoPE rounds after normalization, matrix
RoPE keeps FP32 until its final cast, and complex RoPE rounds before rotation.
Whole-head normalization and compact rotary layouts remain model supplied.
"""
import torch
import torch.nn.functional as F
from . import int8_backend, native_ops
from .dispatch import choose, cuda_inference, take

try:
    from . import denoiser_triton as _triton
except (ImportError, OSError, RuntimeError):
    _triton = None

try:
    from . import adaln_triton as _adaln_triton
except (ImportError, OSError):
    _adaln_triton = None

try:
    from comfy_kitchen import rms_rope_split_half_ as _kitchen_rms_rope
except (ImportError, OSError, RuntimeError):
    _kitchen_rms_rope = None
except ValueError as exc:
    if not str(exc).startswith("infer_schema(func):"):
        raise
    _kitchen_rms_rope = None


def can_linear(layer, x, *, dtypes=(torch.bfloat16,)):
    return int8_backend.can_fuse_linear(layer, x, dtypes=dtypes)


def linear(layer, x_list, **fusion):
    """Use the shared qtype implementation and normal module hooks."""
    if not fusion:
        return layer(take(x_list))
    return int8_backend.linear_with_fusion(layer, take(x_list), **fusion)


def project_many(modules, x_list):
    x = take(x_list)
    if (not torch.compiler.is_compiling() and x.is_contiguous()
            and all(can_linear(module, x) for module in modules)):
        outputs = int8_backend.linear_multi(modules, x.reshape(-1, x.shape[-1]))
        return tuple(out.reshape(*x.shape[:-1], out.shape[-1]) for out in outputs)
    return tuple(module(x) for module in modules)


def norm_modulate(x_list, norm, shift, scale, segments):
    x = take(x_list)
    fused = (_triton is not None and cuda_inference(x, approximate=True) and x.is_contiguous()
             and norm.weight.device == x.device and norm.weight.dtype == x.dtype and x.shape[-1] <= 8192)
    if choose("Segmented Norm Modulation", owned_triton=fused):
        return _triton.norm_modulate(x, norm, shift, scale, segments)
    x = norm(x)
    shift, scale = shift.to(x.dtype), scale.to(x.dtype)
    for start, stop, row in segments:
        x[start:stop].mul_(1.0 + scale[row]).add_(shift[row])
    return x


def scale_shift(x_list, scale, shift=None, *, in_place=True, frames=False, fused_add=False):
    """Native dtype arithmetic; optionally broadcast modulation per frame."""
    x = take(x_list)
    broadcast = frames or (x.ndim == 3 and scale.ndim == 3 and scale.shape[1] in (1, x.shape[1]))
    fused = (broadcast and not fused_add and _triton is not None and cuda_inference(x, approximate=True, inference=False)
             and x.ndim == 3 and scale.ndim == 3 and shift is not None and scale.shape == shift.shape
             and all(t.dtype == x.dtype and t.device == x.device and t.is_contiguous() for t in (x, scale, shift))
             and scale.shape[0] == x.shape[0] and scale.shape[-1] == x.shape[-1]
             and scale.shape[1] > 0 and x.shape[1] % scale.shape[1] == 0
             and (scale.shape[1] == x.shape[1] or scale.shape[0] * scale.shape[1] <= 65535))
    if choose("Frame Scale/Shift" if frames else "Scale/Shift", owned_triton=fused):
        return _triton.scale_shift(x, scale, shift, in_place)
    shape = x.shape
    if frames and scale.shape[1] != x.shape[1]:
        x = x.reshape(x.shape[0], scale.shape[1], -1, x.shape[-1])
        scale = scale.unsqueeze(2)
        shift = None if shift is None else shift.unsqueeze(2)
    if fused_add:
        return torch.addcmul(shift, 1 + scale, x, out=x if in_place else None).reshape(shape)
    x = x.mul_(1 + scale) if in_place else x * (1 + scale)
    if shift is not None:
        x = x.add_(shift) if in_place else x + shift
    return x.reshape(shape)


def norm_scale_shift(x_list, norm, scale, shift, *, in_place=True, fused_add=False):
    """Keep the supplied norm's reduction/affine contract, then modulate.

    Unlike fused AdaLN this retains the native norm reduction and its rounding.
    Residual inputs may be borrowed by passing [residual]; the norm allocates.
    """
    x = norm(take(x_list))
    x_list, x = [x], None
    return scale_shift(x_list, scale, shift, in_place=in_place, fused_add=fused_add)


def layer_norm_modulate(x_list, shift, scale, eps=1e-6):
    x = take(x_list)
    valid = (cuda_inference(x, dtypes=(torch.bfloat16, torch.float16), approximate=True)
             and x.ndim in (2, 3) and x.is_contiguous() and x.numel() > 0 and x.shape[-1] <= 8192
             and all(t.ndim > 0 and t.ndim <= x.ndim and t.dtype == x.dtype and t.device == x.device
                     and t.shape[-1] == x.shape[-1] and t.stride(-1) == 1
                     and all(a == 1 or a == b for a, b in zip(t.shape, x.shape[-t.ndim:])) for t in (shift, scale)))
    family = choose("AdaLN", wgpkernels=valid and native_ops.available(x.device),
                    triton=valid and _adaln_triton is not None)
    if family == "wgpkernels":
        return native_ops.implementation.adaln(x, scale, shift, eps)
    if family == "triton":
        return _adaln_triton.adaln(x, scale, shift, eps)
    x = F.layer_norm(x, (x.shape[-1],), eps=eps)
    # Preserve promotion by a higher-precision modulation, if supplied.
    x_list, x = [x], None
    return scale_shift(x_list, scale, shift, in_place=False)


def swiglu(x_list, *, packed=True, in_place=True):
    """Native SiLU rounding, followed by native multiplication, without concat.

    Packed input is gate|value. Separate projections pass [gate, value]; both
    are consumed. No approximate kernel is promoted to a new model contract.
    """
    if packed:
        gate, value = take(x_list).chunk(2, -1)
    else:
        gate, value = x_list
        x_list.clear()
    return F.silu(gate, inplace=in_place).mul_(value)


def gated_mlp(x_list, gate_proj, up_proj, down_proj, activation):
    """Separate projections with native activation rounding and module hooks.

    Evaluate activation before up_proj so its source and result do not coexist
    with the up projection. All three projections keep their MMGP/LoRA hooks.
    """
    x = take(x_list)
    gate = activation(gate_proj(x))
    gate.mul_(up_proj(x))
    del x
    gate_list, gate = [gate], None
    return linear(down_proj, gate_list)


def swiglu_linear(layer, x_list, residual=None, residual_scale=None):
    if can_linear(layer, x_list[0]):
        return linear(layer, x_list, input_act="swiglu", residual=residual,
                      residual_scale=residual_scale, out=residual)
    output = layer(swiglu(x_list))
    if residual is not None:
        output.mul_(residual_scale).add_(residual)
    return output


def gated_residual_(residual, x_list, gate, *, tanh=False, fused=False):
    x = take(x_list)
    gate = gate.tanh() if tanh else gate
    if fused:
        return residual.addcmul_(x, gate)
    x.mul_(gate)
    return residual.add_(x)


def gated_linear(layer, x_list, residual, gate, segments):
    x = take(x_list)
    for start, stop, row in segments:
        if stop > start:
            out = linear(layer, [x[start:stop]], residual=residual[start:stop],
                         residual_scale=gate[row].to(x.dtype), out=residual[start:stop])
            residual[start:stop].copy_(out)
    return residual


def prepare_split_half_rope(rope):
    if not cuda_inference(rope) or choose("Split-Half RMSNorm + RoPE", owned_triton=_triton is not None, kitchen=_kitchen_rms_rope is not None) != "kitchen":
        return rope
    c, s = rope[..., 0], rope[..., 1]
    return rope, torch.stack((c, -s, s, c), dim=-1).reshape(*c.shape, 2, 2)


def can_split_half_rms_rope(x, packed, q_norm, k_norm):
    return (cuda_inference(x) and (packed is not None or _triton is not None)
            and all(p.device == x.device and p.dtype == x.dtype for p in (q_norm.weight, k_norm.weight)))


def _native_split_half_norm_rope_(tensors, rope):
    for tensor, norm in tensors:
        rows = max(1, (256 << 20) // (tensor[0, 0].numel() * tensor.element_size()))
        for start in range(0, tensor.shape[1], rows):
            part = tensor[:, start:start + rows]
            part.copy_(norm(part))
    if rope is not None:
        pairs = rope.shape[-2]
        cosine, sine = rope[..., 0], rope[..., 1]
        scratch = torch.empty_like(tensors[0][0][..., :pairs])
        for tensor, _ in tensors:
            first, second = tensor[..., :pairs], tensor[..., pairs:2 * pairs]
            scratch.copy_(first)
            first.mul_(cosine).addcmul_(second, sine, value=-1)
            second.mul_(cosine).addcmul_(scratch, sine)


def split_half_norm_rope_(q, k, rope, packed, q_norm, k_norm, *, allow_fusion=True):
    x = q if q is not None else k
    if rope is not None and allow_fusion and can_split_half_rms_rope(x, packed, q_norm, k_norm):
        return split_half_rms_rope_(q, k, rope, packed, q_norm, k_norm)
    _native_split_half_norm_rope_([(x, norm) for x, norm in ((q, q_norm), (k, k_norm)) if x is not None], rope)
    return q, k


def split_half_rms_rope_(q, k, rope, packed, q_norm, k_norm):
    """Borrow Q/K destinations. Either can be absent for staged attention."""
    if packed is not None and q is not None and k is not None:
        return _kitchen_rms_rope(q, k, packed, q_norm.weight, k_norm.weight,
                                 epsilon=q_norm.eps, rot_dim=packed.shape[-3] * 2)
    for x, norm in ((q, q_norm), (k, k_norm)):
        if x is None:
            continue
        if packed is not None and x.shape[2] > 1:
            half = x.shape[2] // 2
            _kitchen_rms_rope(x[:, :, :half], x[:, :, half:], packed, norm.weight, norm.weight,
                               epsilon=norm.eps, rot_dim=packed.shape[-3] * 2)
        elif _triton is not None:
            _triton.rms_rope_(x, norm.weight, rope, norm.eps)
        else:
            # A one-head group cannot be divided for Kitchen's paired API.
            _native_split_half_norm_rope_([(x, norm)], rope)
    return q, k


def sol_rms_rope_(x, weight, rope, eps):
    if choose(owned_triton=_triton is not None and cuda_inference(x)):
        return _triton.rms_rope_(x, weight, rope, eps)
    from shared.sol_attn import rms_norm_rope_
    rms_norm_rope_(x, weight, rope, eps)
    torch.autograd.graph.increment_version(x)
    return x


def compact_rope_(x, cache, layout, head_start=0, dim_head=None):
    """Try compact split RoPE without materializing token-sized tables.

    False leaves x untouched for the model's layout-specific PyTorch fallback.
    Norm over all heads and its prepared statistics are deliberately external.
    """
    valid = (_triton is not None and cuda_inference(x, approximate=True, inference=False)
             and x.stride(-1) == 1 and x.stride(1) == x.shape[2]
             and len(layout.axis_cos) in (1, 3) and 1 <= len(layout.grid_sizes) <= 3
             and 0 <= cache.pad_size < len(layout.axis_cos)
             and (x.shape[0] <= 1 or x.stride(0) >= x.shape[1] * x.shape[2]))
    dtype = torch.float32 if cache.use_fp32_freqs else x.dtype
    valid = valid and all(t.device == x.device and t.dtype == dtype and t.is_contiguous()
                          for t in (*layout.axis_cos, *layout.axis_sin))
    if not choose("Compact Split RoPE", owned_triton=valid):
        return False
    _triton.split_rope(x, cache, layout, head_start, dim_head)
    torch.autograd.graph.increment_version(x)
    return True


def zero_center_rms_norm(x_list, weight, eps):
    x = take(x_list)
    return F.rms_norm(x.float(), (x.shape[-1],), weight.float() + 1, eps).to(x.dtype)


def rms_norm_rounded(x_list, eps=1e-6, *, manual=False):
    """Unweighted FP32 RMSNorm, rounded before a caller's weight multiply.

    Flux uses explicit square/mean/rsqrt; SenseNova uses native rms_norm. Keep
    those reductions distinct, and leave weight promotion to the norm module.
    """
    x = take(x_list)
    dtype = x.dtype
    x = x.float()
    if manual:
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    else:
        x = F.rms_norm(x, (x.shape[-1],), eps=eps)
    return x.to(dtype)


def matrix_rope(x_list, freqs):
    """FP32 matrix rotation with separate products, followed by a final cast."""
    x = take(x_list)
    shape, dtype = x.shape, x.dtype
    x = x.float().reshape(*shape[:-1], -1, 1, 2)
    output = freqs[..., 0] * x[..., 0]
    x = freqs[..., 1] * x[..., 1]
    return output.add_(x).reshape(shape).to(dtype)


def paired_rope_(x, cosine, sine, scratch=None, *, interleaved=False):
    """Native multiply then addcmul; borrow the destination and optional scratch."""
    if interleaved:
        first, second = x[..., 0::2], x[..., 1::2]
    else:
        first, second = x.chunk(2, dim=-1)
    if scratch is None:
        scratch = first.clone()
    else:
        scratch = scratch[tuple(slice(0, size) for size in first.shape)]
        scratch.copy_(first)
    first.mul_(cosine).addcmul_(second, sine, value=-1)
    second.mul_(cosine).addcmul_(scratch, sine)
    return x


def interleaved_rope(x_list, freqs):
    """Qwen Image's FP32 multiply/addcmul rotation after native norm rounding."""
    x = take(x_list)
    dtype = x.dtype
    x = x.float()
    cosine, sine = (freqs[..., i].unsqueeze(0).unsqueeze(2) for i in (0, 1))
    return paired_rope_(x, cosine, sine, interleaved=True).to(dtype)


def indexed_rope(x_list, cosine, sine, partner):
    """Native-dtype rotation with model-supplied partner indices and signs."""
    x = take(x_list)
    rotated = x[..., partner]
    return x.mul_(cosine).addcmul_(rotated, sine)


def complex_norm_rope(x_list, norm, freqs):
    x = take(x_list)
    x = norm(x).to(x.dtype)
    if freqs is None:
        return x
    rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated.mul_(freqs.unsqueeze(1))
    return torch.view_as_real(rotated).flatten(3)


def matrix_rms_rope(x_list, weight, freqs, eps=1e-6):
    x = take(x_list)
    dtype = x.dtype
    x = x.float()
    x = x * x.square().mean(-1, keepdim=True).add(eps).rsqrt() * weight.float()
    pairs = x.unflatten(-1, (-1, 2)).unsqueeze(-2)
    return (freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]).flatten(-2).to(dtype)
