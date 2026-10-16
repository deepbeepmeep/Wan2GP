# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared dispatch for video/audio activation kernels.

Streaming schedules, temporal history, and neighborhood topology stay with the
model. This module owns kernel eligibility and fallbacks, never those caches.
"""
import torch
import torch.nn.functional as F
from . import kernel_policy, ops, native_ops, video_kitchen as _kitchen
from .dispatch import choose, cuda_inference, take

try:
    from . import video_triton as _video
except (ImportError, OSError, RuntimeError):
    _video = None
try:
    from . import conv3d_triton as _conv, stream_triton as _stream
except (ImportError, OSError, RuntimeError):
    _conv = _stream = None
try:
    from .audio_triton import snake_beta_aa as _snake
except (ImportError, OSError, RuntimeError):
    _snake = None

_DTYPES = (torch.bfloat16, torch.float16)
_STREAM_CONV_MIN_PIXELS = 8192  # smaller calls keep cuDNN's lower launch overhead


def prepare_rotary(cos, sin):
    return _kitchen.prepare_rotary(cos, sin)


def conv3d(x_list, conv, residual=None):
    x = take(x_list)
    if choose("Video Conv3D", kitchen=_kitchen.can_conv3d(x, conv)):
        return _kitchen.conv3d(x, conv, residual)
    x = F.conv3d(x, conv.weight, conv.bias, stride=conv.stride, padding=0, dilation=conv.dilation)
    return x if residual is None else x.add_(residual)


def frame_group_norm(x_list, norm, silu_pad=None):
    x = take(x_list)
    triton_pad = False
    if silu_pad is not None:
        left, right, top, bottom, front = silu_pad
        triton_pad = (_video is not None and cuda_inference(x, dtypes=_DTYPES)
                      and left == right == top == bottom and left < min(x.shape[-2:]))
    # Kitchen also prepares the channels-last layout for the next convolution.
    # Choosing native norm on its isolated time regresses the complete encoder.
    kitchen = silu_pad is not None and _kitchen.available(x, norm, silu_pad)
    family = choose("Frame GroupNorm + SiLU + Pad", kitchen=kitchen, owned_triton=triton_pad and not kitchen) if silu_pad is not None else None
    if family == "kitchen":
        return _kitchen.group_norm_silu_pad3d(x, norm, silu_pad)
    batch, channels, frames, height, width = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch * frames, channels, 1, height, width)
    x = F.group_norm(x, norm.num_groups, norm.weight, norm.bias, norm.eps)
    if silu_pad is not None:
        left, right, top, bottom, front = silu_pad
        if family is not None:
            return _video.silu_pad_frames(x, batch, frames, left, front)
    x = x.view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
    if silu_pad is not None:
        x = F.silu(x)
        x = F.pad(x, (left, right, top, bottom, 0, 0), mode="reflect")
        x = F.pad(x, (0, 0, 0, 0, front, 0))
    return x


def rope_plan(x, dim_head, rotary):
    triton_rms = (_video is not None and rotary is not None and cuda_inference(x, dtypes=_DTYPES)
                  and _video.can_rms_rope(dim_head, rotary[0].shape[-1]))
    family = choose("Video RMSNorm + RoPE", owned_triton=triton_rms, kitchen=_kitchen.can_fuse_rms_rope(x, rotary))
    cast = bool(choose(owned_triton=_video is not None and cuda_inference(x)))
    return family, cast


def normalize_projection(x_list, norm, rotary, plan):
    x = take(x_list)
    family, cast = plan
    if family is not None:
        return x
    dtype = x.dtype
    x = norm(x.float())
    return _video.cast_rope(x, rotary) if cast else x.to(dtype)


def finish_rope_(q, k, rotary, eps, plan):
    family, cast = plan
    if family == "kitchen":
        return _kitchen.rms_rope(q, k, rotary, eps)
    if family is not None:
        _video.rms_rope_(q, k, rotary, eps)
    elif not cast and rotary is not None:
        cos, sin = (table.to(q.dtype) for table in rotary)
        dim = cos.shape[-1]
        # Both outputs read the original pair before either half is overwritten.
        for x in (q, k):
            first, second = x[..., :dim].chunk(2, dim=-1)
            half = first.shape[-1]
            a = first * cos[..., :half] - second * sin[..., :half]
            b = second * cos[..., half:] + first * sin[..., half:]
            first.copy_(a)
            second.copy_(b)
    return q, k


def swiglu(x_list):
    if not choose("Video SwiGLU", owned_triton=_video is not None and cuda_inference(x_list[0], dtypes=_DTYPES)):
        return ops.swiglu(x_list)
    x = take(x_list)
    if x.dtype == torch.float16:
        return _video.swiglu_rn(x)
    return _video.swiglu(x) if kernel_policy.allow_approximate() else _video.swiglu_(x)


def scaled_residual(x_list, scale, residual):
    x = take(x_list)
    if choose(owned_triton=_video is not None and cuda_inference(x, dtypes=_DTYPES)):
        return _video.residual_(x, scale, residual)
    return x.mul_(scale).add_(residual)


def pixel_norm_silu(x_list, norm, act, *, channel_norm):
    x = take(x_list)
    valid = (cuda_inference(x, dtypes=_DTYPES, sm120=False, approximate=True) and x.ndim == 5 and x.numel() > 0 and channel_norm
             and isinstance(act, torch.nn.SiLU))
    family = choose("PixelNorm + SiLU", wgpkernels=valid and native_ops.available(x.device),
                    triton=valid and _stream is not None and _conv.supported(x.device))
    if family == "wgpkernels":
        return native_ops.implementation.pixel_norm_silu(x, norm.eps)
    if family == "triton":
        return _stream.norm_silu(x, norm.eps)
    return act(norm(x))


def temporal_conv(conv, parts):
    """Consume temporal parts without keeping their original tensors alive."""
    b, c, _, h, w = parts[0].shape
    t, x = sum(part.shape[2] for part in parts), parts[0]
    valid = (cuda_inference(x, dtypes=_DTYPES, sm120=False, approximate=True) and c % 32 == 0 and conv.out_channels % 32 == 0
             and conv.stride == (1, 1, 1) and b * (t - 2) * h * w >= _STREAM_CONV_MIN_PIXELS
             and conv.kernel_size == (3, 3, 3) and conv.dilation == (1, 1, 1)
             and conv.groups == 1 and conv.padding == (0, 1, 1) and conv.padding_mode in ("zeros", "reflect")
             and conv.weight.dtype == x.dtype and conv.weight.device == x.device
             and (conv.bias is None or (conv.bias.dtype == x.dtype and conv.bias.device == x.device)))
    triton_conv = valid and _conv is not None and conv.bias is not None and _conv.conv_supported(x.device)
    # Native CUDA tuning improves its tiles, but it still loses to Triton and
    # sometimes cuDNN. Keep it out of automatic dispatch until competitive.
    family = choose("Streamed Conv3D", triton=triton_conv)
    if family is None:
        x = torch.cat(parts, dim=2) if len(parts) > 1 else x
        parts.clear()
        return conv(x)
    padded = torch.empty((b, c, t, h + 2, w + 2), dtype=x.dtype, device=x.device, memory_format=torch.channels_last_3d)
    x, start = None, 0
    for part in parts:
        padded[:, :, start:start + part.shape[2], 1:-1, 1:-1] = part
        start += part.shape[2]
    del part
    parts.clear()
    if conv.padding_mode == "reflect":
        padded[..., 0, 1:-1], padded[..., -1, 1:-1] = padded[..., 2, 1:-1], padded[..., -3, 1:-1]
        padded[..., 0], padded[..., -1] = padded[..., 2], padded[..., -3]
    else:
        padded[..., 0, :], padded[..., -1, :], padded[..., 0], padded[..., -1] = 0, 0, 0, 0
    return _conv.conv(padded, conv.weight, conv.bias)


def snake_beta_aa(x_list, activation, upsample, downsample, *, fusable):
    x = take(x_list)
    # Retain this audio path's established FP32 eligibility in both precision
    # modes; it has its own end-to-end validation, unlike video fusions.
    if choose("SnakeBeta + Resampling", owned_triton=fusable and _snake is not None and x.is_cuda):
        return _snake(x, activation.alpha, activation.beta, upsample.filter, downsample.lowpass.filter)
    return downsample(activation(upsample(x)))


def neighborhood_status():
    from packaging.version import InvalidVersion, Version
    try:
        import triton
    except (ImportError, OSError):
        return False, "not installed"
    try:
        version = Version(triton.__version__)
    except InvalidVersion:
        return False, f"unknown version {triton.__version__}"
    if version < Version("3.2.0"):
        return False, str(version)
    return torch.cuda.is_available(), str(version)


_NA_AVAILABLE, _NA_VERSION = neighborhood_status()


def neighborhood_attention(qkv_list, kernel_size, fallback, *, keyframe_times=None, slots=None):
    if choose("3D Neighborhood Attention", owned_triton=_NA_AVAILABLE and qkv_list[0].is_cuda):
        from . import neighborhood_triton
        if keyframe_times is None:
            return neighborhood_triton.na3d(qkv_list, kernel_size)
        time = qkv_list[0].shape[1]
        video_slots, keyframe_slots = slots
        return neighborhood_triton.joint_na3d(qkv_list, video_slots(keyframe_times, time), keyframe_slots(keyframe_times, time), kernel_size)
    if keyframe_times is None:
        return fallback(qkv_list, kernel_size)
    return fallback(qkv_list, keyframe_times, kernel_size)
