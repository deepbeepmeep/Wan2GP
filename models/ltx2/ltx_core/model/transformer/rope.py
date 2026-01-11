import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple

import numpy as np
import torch


USE_FP32_ROPE_FREQS = False


def set_use_fp32_rope_freqs(enabled: bool) -> None:
    global USE_FP32_ROPE_FREQS
    USE_FP32_ROPE_FREQS = bool(enabled)


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


@dataclass(frozen=True)
class RopeCache:
    cos: torch.Tensor | tuple[torch.Tensor, ...]
    sin: torch.Tensor | tuple[torch.Tensor, ...]
    grid_sizes: tuple[int, ...] | None
    rope_axes: tuple[int, ...] | None
    pad_size: int
    rope_type: LTXRopeType
    num_attention_heads: int | None = None
    split_head_axes: torch.Tensor | None = None
    split_head_freqs: torch.Tensor | None = None
    use_fp32_freqs: bool = False

    def is_grid(self) -> bool:
        return self.grid_sizes is not None and self.rope_axes is not None and isinstance(self.cos, tuple)


def apply_rotary_emb_inplace(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor] | RopeCache,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> torch.Tensor:
    if isinstance(freqs_cis, RopeCache):
        return apply_split_rope_cache_inplace(input_tensor, freqs_cis)
    return _apply_rotary_emb_inplace_tensor(input_tensor, freqs_cis, rope_type)


def _apply_rotary_emb_inplace_tensor(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType,
) -> torch.Tensor:
    cos, sin = freqs_cis
    use_fp32 = USE_FP32_ROPE_FREQS
    target_dtype = torch.float32 if use_fp32 else input_tensor.dtype
    if cos.device != input_tensor.device or cos.dtype != target_dtype:
        cos = cos.to(device=input_tensor.device, dtype=target_dtype)
    if sin.device != input_tensor.device or sin.dtype != target_dtype:
        sin = sin.to(device=input_tensor.device, dtype=target_dtype)

    if use_fp32 and input_tensor.dtype != torch.float32:
        x_work = input_tensor.to(torch.float32)
        apply_split_rotary_emb_inplace(x_work, cos, sin)
        input_tensor.copy_(x_work.to(input_tensor.dtype))
        return input_tensor

    return apply_split_rotary_emb_inplace(input_tensor, cos, sin)



def apply_split_rope_cache_inplace(input_tensor: torch.Tensor, rope_cache: RopeCache) -> torch.Tensor:
    grid_sizes = rope_cache.grid_sizes
    rope_axes = rope_cache.rope_axes
    cos_tables = rope_cache.cos
    sin_tables = rope_cache.sin

    b, tokens, dim = input_tensor.shape
    grid_volume = math.prod(grid_sizes)
    if grid_volume != tokens:
        return input_tensor

    axis_count = len(rope_axes)
    axis_dim = cos_tables[0].shape[-1] if axis_count > 0 else 0
    if axis_count == 0:
        return input_tensor

    num_heads = rope_cache.num_attention_heads
    if num_heads is None:
        return input_tensor
    dim_head = dim // num_heads
    if dim_head % 2 != 0:
        return input_tensor
    half_dim = dim_head // 2
    expected_freqs = num_heads * half_dim
    pad_size = rope_cache.pad_size
    dim_no_pad = expected_freqs - pad_size
    if axis_dim <= 0 or dim_no_pad <= 0 or dim_no_pad % axis_count != 0:
        return input_tensor
    if axis_dim * axis_count != dim_no_pad:
        return input_tensor

    x_view = input_tensor.reshape(b, *grid_sizes, num_heads, dim_head)
    x0 = x_view[..., :half_dim]
    x1 = x_view[..., half_dim:]
    x0_flat = x0.reshape(b, *grid_sizes, expected_freqs)
    x1_flat = x1.reshape(b, *grid_sizes, expected_freqs)

    if pad_size:
        x0_rope = x0_flat[..., pad_size:]
        x1_rope = x1_flat[..., pad_size:]
    else:
        x0_rope = x0_flat
        x1_rope = x1_flat

    rope_width = axis_dim * axis_count
    if x0_rope.shape[-1] != rope_width:
        return input_tensor

    x0_rope = x0_rope.reshape(b, *grid_sizes, axis_dim, axis_count)
    x1_rope = x1_rope.reshape(b, *grid_sizes, axis_dim, axis_count)

    x_dtype = input_tensor.dtype
    device = input_tensor.device
    work_dtype = torch.float32 if rope_cache.use_fp32_freqs else x_dtype
    if work_dtype == x_dtype:
        x0_tmp = torch.empty_like(x0_rope[..., 0])
        x1_tmp = None
        x0_orig_tmp = x0_tmp
    else:
        x0_tmp = torch.empty_like(x0_rope[..., 0], dtype=work_dtype, device=device)
        x1_tmp = torch.empty_like(x1_rope[..., 0], dtype=work_dtype, device=device)
        x0_orig_tmp = torch.empty_like(x0_tmp)

    for axis_index, axis in enumerate(rope_axes):
        cos_axis = cos_tables[axis_index]
        sin_axis = sin_tables[axis_index]
        shape = [1] * len(grid_sizes) + [axis_dim]
        shape[axis] = grid_sizes[axis]
        cos_axis = cos_axis.view(*shape)
        sin_axis = sin_axis.view(*shape)
        if cos_axis.device != device or cos_axis.dtype != work_dtype:
            cos_axis = cos_axis.to(device=device, dtype=work_dtype)
        if sin_axis.device != device or sin_axis.dtype != work_dtype:
            sin_axis = sin_axis.to(device=device, dtype=work_dtype)

        x0_axis = x0_rope[..., axis_index]
        x1_axis = x1_rope[..., axis_index]
        if work_dtype == x_dtype:
            x0_orig_tmp.copy_(x0_axis)
            x0_axis.mul_(cos_axis).addcmul_(x1_axis, sin_axis, value=-1)
            x1_axis.mul_(cos_axis).addcmul_(x0_orig_tmp, sin_axis)
        else:
            x0_orig_tmp.copy_(x0_axis)
            x1_tmp.copy_(x1_axis)
            x0_tmp.copy_(x0_orig_tmp)
            x0_tmp.mul_(cos_axis).addcmul_(x1_tmp, sin_axis, value=-1)
            x1_tmp.mul_(cos_axis).addcmul_(x0_orig_tmp, sin_axis)
            x0_axis.copy_(x0_tmp)
            x1_axis.copy_(x1_tmp)

    x0.copy_(x0_flat.reshape_as(x0))
    x1.copy_(x1_flat.reshape_as(x1))
    return input_tensor
#used



def apply_split_rotary_emb_inplace(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    x = input_tensor
    if x.ndim == 3 and cos_freqs.ndim == 4:
        b, t, _ = x.shape
        h = cos_freqs.shape[1]
        x = x.reshape(b, t, h, -1).transpose(1, 2)
    elif x.ndim == 4 and cos_freqs.ndim == 4:
        if x.shape[1] != cos_freqs.shape[1] and x.shape[2] == cos_freqs.shape[1]:
            x = x.transpose(1, 2)

    half_dim = x.shape[-1] // 2
    x0 = x[..., :half_dim]
    x1 = x[..., half_dim:]
    x0_orig = x0.clone()
    x0.mul_(cos_freqs).addcmul_(x1, sin_freqs, value=-1)
    x1.mul_(cos_freqs).addcmul_(x0_orig, sin_freqs)

    return input_tensor
#used

def generate_freq_grid_np(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


def generate_freq_grid_pytorch(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count

    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
    )
    indices = indices.to(dtype=torch.float32)

    indices = indices * math.pi / 2

    return indices


def get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )
    return fractional_positions


def generate_freqs(
    indices: torch.Tensor, indices_grid: torch.Tensor, max_pos: list[int], use_middle_indices_grid: bool
) -> torch.Tensor:
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)

    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def split_freqs_cis(freqs: torch.Tensor, pad_size: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape freqs to be compatible with multi-head attention
    b = cos_freq.shape[0]
    t = cos_freq.shape[1]

    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)

    cos_freq = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)
    return cos_freq, sin_freq


def interleaved_freqs_cis(freqs: torch.Tensor, pad_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    freq_grid_generator: Callable[[float, int, int, torch.device], torch.Tensor] = generate_freq_grid_pytorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_pos is None:
        max_pos = [20, 2048, 2048]
    freqs_dtype = torch.float32 if USE_FP32_ROPE_FREQS else out_dtype

    indices = freq_grid_generator(theta, indices_grid.shape[1], dim)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        # 2 because of cos and sin by 3 for (t, x, y), 1 for temporal only
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
    return cos_freq.to(freqs_dtype), sin_freq.to(freqs_dtype)


def _reduce_positions(positions: torch.Tensor, use_middle_indices_grid: bool) -> torch.Tensor:
    if positions.ndim == 4:
        if use_middle_indices_grid:
            return positions.mean(dim=-1)
        return positions[..., 0]
    return positions


def _infer_grid_sizes(positions: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    axis_values = []
    for axis in range(positions.shape[1]):
        axis_values.append(torch.unique(positions[0, axis], sorted=True))
    grid_sizes = tuple(values.numel() for values in axis_values)
    return axis_values, grid_sizes


def _build_axis_freqs(indices: torch.Tensor, axis_values: torch.Tensor, axis_max: int) -> torch.Tensor:
    fractional_positions = axis_values / axis_max
    return indices * (fractional_positions.unsqueeze(-1) * 2 - 1)


def _can_broadcast_frames(frame_indices: torch.Tensor, frame_count: int) -> bool:
    if frame_indices.ndim != 2:
        return False
    token_count = frame_indices.shape[1]
    if frame_count <= 0 or token_count % frame_count != 0:
        return False
    tokens_per_frame = token_count // frame_count
    if tokens_per_frame == 0:
        return False
    try:
        frame_view = frame_indices.reshape(frame_indices.shape[0], frame_count, tokens_per_frame)
    except RuntimeError:
        return False
    expected = torch.arange(frame_count, device=frame_indices.device).view(1, frame_count, 1)
    return torch.equal(frame_view, expected.expand(frame_indices.shape[0], -1, tokens_per_frame))


def build_rope_cache(
    positions: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    rope_axes: tuple[int, ...] | None = None,
    frame_indices: torch.Tensor | None = None,
    grid_sizes: tuple[int, ...] | None = None,
    rope_max_pos: list[int] | None = None,
    freq_grid_generator: Callable[[float, int, int], torch.Tensor] = generate_freq_grid_pytorch,
) -> RopeCache | tuple[torch.Tensor, torch.Tensor]:
    if max_pos is None:
        max_pos = [20, 2048, 2048]
    freqs_dtype = torch.float32 if USE_FP32_ROPE_FREQS else out_dtype

    positions_mid = _reduce_positions(positions, use_middle_indices_grid)
    pos_dims = positions_mid.shape[1]
    if rope_axes is None:
        rope_axes = tuple(range(pos_dims))

    axis_values_all, grid_sizes = _infer_grid_sizes(positions_mid)
    token_count = positions_mid.shape[-1]
    grid_volume = math.prod(grid_sizes)

    rope_positions = positions[:, rope_axes, ...]
    rope_max = rope_max_pos if rope_max_pos is not None else [max_pos[axis] for axis in rope_axes]
    num_rope_axes = len(rope_axes)

    # Use per-axis RoPE tables when tokens form a full grid; fall back for irregular layouts.
    can_use_grid = token_count == grid_volume
    if can_use_grid and frame_indices is not None:
        frame_count = int(frame_indices.max().item()) + 1
        can_use_grid = _can_broadcast_frames(frame_indices, frame_count)

    indices = freq_grid_generator(theta, num_rope_axes, dim).to(device=positions_mid.device)
    if can_use_grid and num_rope_axes > 0:
        axis_cos_tables = []
        axis_sin_tables = []
        if rope_type == LTXRopeType.SPLIT:
            expected_freqs = dim // 2
            current_freqs = indices.shape[0] * num_rope_axes
            pad_size = expected_freqs - current_freqs
        else:
            pad_size = dim % (2 * num_rope_axes)

        for axis_index, axis in enumerate(rope_axes):
            axis_values = axis_values_all[axis].to(device=positions_mid.device)
            axis_freqs = _build_axis_freqs(indices, axis_values, rope_max[axis_index])
            if rope_type == LTXRopeType.SPLIT:
                cos_axis = axis_freqs.cos()
                sin_axis = axis_freqs.sin()
            else:
                cos_axis = axis_freqs.cos().repeat_interleave(2, dim=-1)
                sin_axis = axis_freqs.sin().repeat_interleave(2, dim=-1)
            axis_cos_tables.append(cos_axis.to(freqs_dtype))
            axis_sin_tables.append(sin_axis.to(freqs_dtype))

        return RopeCache(
            cos=tuple(axis_cos_tables),
            sin=tuple(axis_sin_tables),
            grid_sizes=grid_sizes,
            rope_axes=rope_axes,
            pad_size=pad_size,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
            use_fp32_freqs=USE_FP32_ROPE_FREQS,
        )

    if num_rope_axes > 0:
        warnings.warn(
            "LTX2 RoPE fallback: using per-token freqs because tokens are not a regular grid.",
            RuntimeWarning,
            stacklevel=2,
        )

    freqs = generate_freqs(indices, rope_positions, rope_max, use_middle_indices_grid)

    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        pad_size = dim % (2 * num_rope_axes)
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, pad_size)

    cos_freq = cos_freq.to(freqs_dtype)
    sin_freq = sin_freq.to(freqs_dtype)
    if cos_freq.ndim == 3 and cos_freq.shape[0] == 1:
        cos_freq = cos_freq[0]
        sin_freq = sin_freq[0]

    return cos_freq, sin_freq
