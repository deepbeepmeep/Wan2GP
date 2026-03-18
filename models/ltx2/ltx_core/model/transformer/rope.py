import math
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
class RopeAxisCache:
    values: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass(frozen=True)
class RopeLayoutCache:
    token_start: int
    token_stop: int
    grid_sizes: tuple[int, ...]
    axis_cos: tuple[torch.Tensor, ...]
    axis_sin: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class RopeSplitGroup:
    head_start: int
    head_count: int
    local_start: int
    freq_start: int
    freq_width: int


@dataclass(frozen=True)
class RopeCache:
    axes: tuple[RopeAxisCache, ...]
    layouts: tuple[RopeLayoutCache, ...]
    rope_axes: tuple[int, ...] | None
    pad_size: int
    rope_type: LTXRopeType
    freq_dim: int
    token_count: int
    num_attention_heads: int | None = None
    use_fp32_freqs: bool = False
    split_groups: tuple[tuple[RopeSplitGroup, ...], ...] = ()
    split_max_head_count: int = 0
    split_max_freq_width: int = 0

    def is_grid(self) -> bool:
        return self.rope_axes is not None and bool(self.layouts)


def apply_rotary_emb_inplace(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor] | RopeCache,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> torch.Tensor:
    if isinstance(freqs_cis, RopeCache):
        return apply_rope_cache_inplace(input_tensor, freqs_cis)
    return _apply_rotary_emb_inplace_tensor(input_tensor, freqs_cis, rope_type)


def _prepare_freqs_for_input(
    input_tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    use_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_dtype = torch.float32 if use_fp32 else input_tensor.dtype
    if cos.device != input_tensor.device or cos.dtype != target_dtype:
        cos = cos.to(device=input_tensor.device, dtype=target_dtype)
    if sin.device != input_tensor.device or sin.dtype != target_dtype:
        sin = sin.to(device=input_tensor.device, dtype=target_dtype)
    return cos, sin


def _apply_rotary_emb_inplace_prepared(
    input_tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_type: LTXRopeType,
    use_fp32: bool,
) -> torch.Tensor:
    apply_fn = apply_split_rotary_emb_inplace if rope_type == LTXRopeType.SPLIT else apply_interleaved_rotary_emb_inplace
    if use_fp32 and input_tensor.dtype != torch.float32:
        x_work = input_tensor.to(torch.float32)
        apply_fn(x_work, cos, sin)
        input_tensor.copy_(x_work.to(input_tensor.dtype))
        return input_tensor
    apply_fn(input_tensor, cos, sin)
    return input_tensor


def _apply_rotary_emb_inplace_tensor(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType,
) -> torch.Tensor:
    cos, sin = _prepare_freqs_for_input(input_tensor, freqs_cis[0], freqs_cis[1], USE_FP32_ROPE_FREQS)
    return _apply_rotary_emb_inplace_prepared(input_tensor, cos, sin, rope_type, USE_FP32_ROPE_FREQS)


def _adjacent_pair_view(x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    last_stride = x.stride(-1)
    return torch.as_strided(
        x,
        size=(*x.shape[:-1], half_dim, 2),
        stride=(*x.stride()[:-1], 2 * last_stride, last_stride),
    )


def _split_pair_view(x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    last_stride = x.stride(-1)
    return torch.as_strided(
        x,
        size=(*x.shape[:-1], half_dim, 2),
        stride=(*x.stride()[:-1], last_stride, half_dim * last_stride),
    )


def _apply_interleaved_rope_inplace_inner(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    x_view = _adjacent_pair_view(x)
    cos_view = _adjacent_pair_view(cos)
    sin_view = _adjacent_pair_view(sin)
    x0 = x_view[..., 0]
    x1 = x_view[..., 1]
    x0_orig = x0.clone()
    x0.mul_(cos_view[..., 0]).addcmul_(x1, sin_view[..., 0], value=-1)
    x1.mul_(cos_view[..., 1]).addcmul_(x0_orig, sin_view[..., 1])


def _apply_split_rope_inplace_inner(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    x_view = _split_pair_view(x)
    x0 = x_view[..., 0]
    x1 = x_view[..., 1]
    x0_orig = x0.clone()
    x0.mul_(cos).addcmul_(x1, sin, value=-1)
    x1.mul_(cos).addcmul_(x0_orig, sin)


def apply_interleaved_rotary_emb_inplace(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    x = input_tensor
    if x.shape[-1] % 2 != 0:
        return input_tensor
    if x.ndim == 4 and cos_freqs.ndim == 4 and x.shape[1] != cos_freqs.shape[1] and x.shape[2] == cos_freqs.shape[1]:
        x = x.transpose(1, 2)
    _apply_interleaved_rope_inplace_inner(x, cos_freqs, sin_freqs)
    return input_tensor


def apply_split_rotary_emb_inplace(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    x = input_tensor
    if x.ndim == 3 and cos_freqs.ndim == 4:
        b, t, _ = x.shape
        h = cos_freqs.shape[1]
        x = x.reshape(b, t, h, -1).transpose(1, 2)
    elif x.ndim == 4 and cos_freqs.ndim == 4 and x.shape[1] != cos_freqs.shape[1] and x.shape[2] == cos_freqs.shape[1]:
        x = x.transpose(1, 2)
    if x.shape[-1] % 2 != 0:
        return input_tensor
    _apply_split_rope_inplace_inner(x, cos_freqs, sin_freqs)
    return input_tensor


def _make_axis_buffers(
    example: torch.Tensor,
    work_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if example.dtype == work_dtype:
        x0_orig_tmp = torch.empty_like(example)
        return None, None, x0_orig_tmp
    x0_tmp = torch.empty_like(example, dtype=work_dtype, device=example.device)
    x1_tmp = torch.empty_like(example, dtype=work_dtype, device=example.device)
    return x0_tmp, x1_tmp, None


def _make_axis_buffers_for_shape(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    work_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if dtype == work_dtype:
        return None, None, torch.empty(shape, dtype=dtype, device=device)
    return (
        torch.empty(shape, dtype=work_dtype, device=device),
        torch.empty(shape, dtype=work_dtype, device=device),
        None,
    )


def _slice_axis_buffers(
    buffers: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    head_count: int,
    freq_width: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    x0_tmp, x1_tmp, x0_orig_tmp = buffers
    if x0_tmp is not None:
        x0_tmp = x0_tmp[..., :head_count, :freq_width]
    if x1_tmp is not None:
        x1_tmp = x1_tmp[..., :head_count, :freq_width]
    if x0_orig_tmp is not None:
        x0_orig_tmp = x0_orig_tmp[..., :head_count, :freq_width]
    return x0_tmp, x1_tmp, x0_orig_tmp


def _is_compiling_graph() -> bool:
    return torch.compiler.is_compiling()


def _broadcast_group_freqs(freqs: torch.Tensor, axis: int, grid_ndim: int) -> torch.Tensor:
    for dim in range(grid_ndim):
        if dim != axis:
            freqs = freqs.unsqueeze(dim)
    return freqs


def _split_group_freq_view(axis_freqs: torch.Tensor, group: RopeSplitGroup, half_dim: int) -> torch.Tensor:
    last_index = group.freq_start + (group.head_count - 1) * half_dim + group.freq_width - 1
    if last_index >= axis_freqs.shape[1]:
        raise RuntimeError(
            "LTX2 split RoPE group exceeded cached frequency bounds: "
            f"last_index={last_index}, width={axis_freqs.shape[1]}, group={group}"
        )
    if _is_compiling_graph():
        # Keep the compile-safe path limited to the small frequency tensor.
        head_offsets = torch.arange(group.head_count, device=axis_freqs.device, dtype=torch.long) * half_dim
        freq_offsets = torch.arange(group.freq_width, device=axis_freqs.device, dtype=torch.long)
        freq_indices = group.freq_start + head_offsets.unsqueeze(1) + freq_offsets.unsqueeze(0)
        return axis_freqs.index_select(1, freq_indices.reshape(-1)).view(axis_freqs.shape[0], group.head_count, group.freq_width)
    return torch.as_strided(
        axis_freqs,
        size=(axis_freqs.shape[0], group.head_count, group.freq_width),
        stride=(axis_freqs.stride(0), half_dim * axis_freqs.stride(1), axis_freqs.stride(1)),
        storage_offset=axis_freqs.storage_offset() + group.freq_start * axis_freqs.stride(1),
    )


def _apply_axis_rotation(
    x0_axis: torch.Tensor,
    x1_axis: torch.Tensor,
    cos_axis: torch.Tensor,
    sin_axis: torch.Tensor,
    work_dtype: torch.dtype,
    buffers: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None,
) -> None:
    x0_tmp, x1_tmp, x0_orig_tmp = _make_axis_buffers(x0_axis, work_dtype) if buffers is None else buffers
    if work_dtype == x0_axis.dtype:
        if x0_orig_tmp is None:
            raise RuntimeError("LTX2 RoPE expected an in-dtype scratch buffer.")
        x0_orig_tmp.copy_(x0_axis)
        x0_axis.mul_(cos_axis).addcmul_(x1_axis, sin_axis, value=-1)
        x1_axis.mul_(cos_axis).addcmul_(x0_orig_tmp, sin_axis)
        return
    if x0_tmp is None or x1_tmp is None:
        raise RuntimeError("LTX2 RoPE expected fp32 scratch buffers.")
    torch.mul(x1_axis, cos_axis, out=x1_tmp)
    x1_tmp.addcmul_(x0_axis, sin_axis)
    torch.mul(x0_axis, cos_axis, out=x0_tmp)
    x0_tmp.addcmul_(x1_axis, sin_axis, value=-1)
    x0_axis.copy_(x0_tmp)
    x1_axis.copy_(x1_tmp)


def _apply_split_rope_layout_inplace(input_tensor: torch.Tensor, rope_cache: RopeCache, layout: RopeLayoutCache) -> None:
    num_heads = rope_cache.num_attention_heads
    if num_heads is None or not rope_cache.split_groups:
        return
    b, tokens, dim = input_tensor.shape
    grid_sizes = layout.grid_sizes
    if math.prod(grid_sizes) != tokens:
        return

    dim_head = dim // num_heads
    if dim_head % 2 != 0:
        return
    half_dim = dim_head // 2
    expected_freqs = num_heads * half_dim
    x_dtype = input_tensor.dtype
    work_dtype = torch.float32 if rope_cache.use_fp32_freqs else x_dtype
    axis_count = len(rope_cache.axes)
    axis_width = rope_cache.axes[0].cos.shape[-1] if axis_count else 0
    dim_no_pad = expected_freqs - rope_cache.pad_size
    if axis_width <= 0 or dim_no_pad <= 0 or dim_no_pad % axis_count != 0 or axis_width * axis_count != dim_no_pad:
        return

    x_view = input_tensor.reshape(b, *grid_sizes, num_heads, dim_head)
    x0 = x_view[..., :half_dim]
    x1 = x_view[..., half_dim:]
    if rope_cache.split_max_head_count <= 0 or rope_cache.split_max_freq_width <= 0:
        return
    buffer_shape = (*x0.shape[:-2], rope_cache.split_max_head_count, rope_cache.split_max_freq_width)
    buffers = _make_axis_buffers_for_shape(buffer_shape, x_dtype, input_tensor.device, work_dtype)

    for axis_index, axis in enumerate(rope_cache.rope_axes or ()):
        cos_axis = layout.axis_cos[axis_index]
        sin_axis = layout.axis_sin[axis_index]
        if cos_axis.device != input_tensor.device or cos_axis.dtype != work_dtype:
            cos_axis = cos_axis.to(device=input_tensor.device, dtype=work_dtype)
        if sin_axis.device != input_tensor.device or sin_axis.dtype != work_dtype:
            sin_axis = sin_axis.to(device=input_tensor.device, dtype=work_dtype)
        for group in rope_cache.split_groups[axis_index]:
            x0_group = x0[..., group.head_start :: axis_count, group.local_start :: axis_count]
            x1_group = x1[..., group.head_start :: axis_count, group.local_start :: axis_count]
            cos_group = _broadcast_group_freqs(_split_group_freq_view(cos_axis, group, half_dim), axis, len(grid_sizes))
            sin_group = _broadcast_group_freqs(_split_group_freq_view(sin_axis, group, half_dim), axis, len(grid_sizes))
            _apply_axis_rotation(
                x0_group,
                x1_group,
                cos_group,
                sin_group,
                work_dtype,
                _slice_axis_buffers(buffers, group.head_count, group.freq_width),
            )
        del cos_axis, sin_axis


def _apply_interleaved_rope_layout_inplace(input_tensor: torch.Tensor, rope_cache: RopeCache, layout: RopeLayoutCache) -> None:
    b, tokens, dim = input_tensor.shape
    grid_sizes = layout.grid_sizes
    if math.prod(grid_sizes) != tokens:
        return

    axis_count = len(rope_cache.axes)
    axis_width = rope_cache.axes[0].cos.shape[-1] if axis_count else 0
    dim_no_pad = dim - rope_cache.pad_size
    if axis_width <= 0 or dim_no_pad <= 0 or dim_no_pad % (2 * axis_count) != 0 or 2 * axis_width * axis_count != dim_no_pad:
        return

    x_view = input_tensor.reshape(b, *grid_sizes, dim)
    x_rope = x_view[..., rope_cache.pad_size :] if rope_cache.pad_size else x_view
    x_pairs = _adjacent_pair_view(x_rope).reshape(b, *grid_sizes, axis_width, axis_count, 2)

    x_dtype = input_tensor.dtype
    work_dtype = torch.float32 if rope_cache.use_fp32_freqs else x_dtype
    x0_tmp, x1_tmp, x0_orig_tmp = _make_axis_buffers(x_pairs[..., 0, 0], work_dtype)

    for axis_index, axis in enumerate(rope_cache.rope_axes or ()):
        cos_axis = layout.axis_cos[axis_index]
        sin_axis = layout.axis_sin[axis_index]
        if cos_axis.device != input_tensor.device or cos_axis.dtype != work_dtype:
            cos_axis = cos_axis.to(device=input_tensor.device, dtype=work_dtype)
        if sin_axis.device != input_tensor.device or sin_axis.dtype != work_dtype:
            sin_axis = sin_axis.to(device=input_tensor.device, dtype=work_dtype)
        shape = [1] * len(grid_sizes) + [axis_width]
        shape[axis] = grid_sizes[axis]
        cos_axis = cos_axis.view(*shape)
        sin_axis = sin_axis.view(*shape)
        x_axis = x_pairs[..., axis_index, :]
        _apply_axis_rotation(x_axis[..., 0], x_axis[..., 1], cos_axis, sin_axis, work_dtype, (x0_tmp, x1_tmp, x0_orig_tmp))


def apply_rope_cache_inplace(input_tensor: torch.Tensor, rope_cache: RopeCache) -> torch.Tensor:
    if not rope_cache.layouts or input_tensor.shape[1] != rope_cache.token_count:
        return input_tensor
    apply_layout = _apply_split_rope_layout_inplace if rope_cache.rope_type == LTXRopeType.SPLIT else _apply_interleaved_rope_layout_inplace
    for layout in rope_cache.layouts:
        apply_layout(input_tensor[:, layout.token_start : layout.token_stop], rope_cache, layout)
    return input_tensor


def generate_freq_grid_np(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
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
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count
    indices = theta ** torch.linspace(math.log(start, theta), math.log(end, theta), inner_dim // n_elem, dtype=torch.float32)
    return indices.to(dtype=torch.float32) * math.pi / 2


def get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )
    return torch.stack([indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)], dim=-1)


def generate_freqs(
    indices: torch.Tensor,
    indices_grid: torch.Tensor,
    max_pos: list[int],
    use_middle_indices_grid: bool,
) -> torch.Tensor:
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid = indices_grid.mean(dim=-1)
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)
    return (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)


def split_freqs_cis(freqs: torch.Tensor, pad_size: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

    b, t = cos_freq.shape[:2]
    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)
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
        pad_size = expected_freqs - freqs.shape[-1]
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % (2 * indices_grid.shape[1]))
    return cos_freq.to(freqs_dtype), sin_freq.to(freqs_dtype)


def _reduce_positions(positions: torch.Tensor, use_middle_indices_grid: bool) -> torch.Tensor:
    if positions.ndim == 4:
        return positions.mean(dim=-1) if use_middle_indices_grid else positions[..., 0]
    return positions


def _infer_grid_sizes(positions: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    axis_values = [torch.unique(positions[0, axis], sorted=True) for axis in range(positions.shape[1])]
    grid_sizes = tuple(values.numel() for values in axis_values)
    return axis_values, grid_sizes


def _build_axis_freqs(indices: torch.Tensor, axis_values: torch.Tensor, axis_max: int) -> torch.Tensor:
    fractional_positions = axis_values / axis_max
    return indices * (fractional_positions.unsqueeze(-1) * 2 - 1)


def _build_split_groups(
    num_heads: int,
    half_dim: int,
    axis_count: int,
    axis_width: int,
    pad_size: int,
) -> tuple[tuple[tuple[RopeSplitGroup, ...], ...], int, int]:
    all_groups: list[tuple[RopeSplitGroup, ...]] = []
    max_head_count = 0
    max_freq_width = 0
    for axis_index in range(axis_count):
        axis_groups: list[RopeSplitGroup] = []
        for head_start in range(min(axis_count, num_heads)):
            head_flat_start = head_start * half_dim
            numer = head_flat_start - pad_size - axis_index
            freq_start = 0 if numer <= 0 else (numer + axis_count - 1) // axis_count
            local_start = pad_size + axis_index + axis_count * freq_start - head_flat_start
            if local_start < 0 or local_start >= half_dim or freq_start >= axis_width:
                continue
            head_count = 1 + (num_heads - 1 - head_start) // axis_count
            available_width = axis_width - freq_start - (head_count - 1) * half_dim
            freq_width = min(available_width, 1 + (half_dim - 1 - local_start) // axis_count)
            if freq_width <= 0:
                continue
            axis_groups.append(
                RopeSplitGroup(
                    head_start=head_start,
                    head_count=head_count,
                    local_start=local_start,
                    freq_start=freq_start,
                    freq_width=freq_width,
                )
            )
            max_head_count = max(max_head_count, head_count)
            max_freq_width = max(max_freq_width, freq_width)
        all_groups.append(tuple(axis_groups))
    return tuple(all_groups), max_head_count, max_freq_width


def _matches_sorted_grid_order(positions: torch.Tensor, axis_values_all: list[torch.Tensor]) -> bool:
    if positions.shape[0] <= 1:
        batches = (positions[0],)
    else:
        batches = positions
    expected = torch.stack(
        torch.meshgrid(
            *(torch.arange(values.numel(), device=positions.device) for values in axis_values_all),
            indexing="ij",
        ),
        dim=0,
    ).reshape(positions.shape[1], -1)
    for batch_positions in batches:
        actual = []
        for axis, axis_values in enumerate(axis_values_all):
            _, inverse = torch.unique(batch_positions[axis], sorted=True, return_inverse=True)
            if inverse.numel() != expected.shape[1]:
                return False
            actual.append(inverse)
        if not torch.equal(torch.stack(actual, dim=0), expected):
            return False
    return True


def _positions_match_across_batch(positions: torch.Tensor) -> bool:
    if positions.shape[0] <= 1:
        return True
    ref = positions[0]
    return all(torch.equal(positions[b], ref) for b in range(1, positions.shape[0]))


def _validate_layout_chunk(
    positions_mid: torch.Tensor,
    token_start: int,
    token_stop: int,
) -> tuple[list[torch.Tensor], tuple[int, ...]] | None:
    chunk_positions = positions_mid[:, :, token_start:token_stop]
    axis_values_all, grid_sizes = _infer_grid_sizes(chunk_positions)
    if math.prod(grid_sizes) != token_stop - token_start:
        return None
    if not _matches_sorted_grid_order(chunk_positions, axis_values_all):
        return None
    return axis_values_all, grid_sizes


def _split_constant_time_run_by_frame_starts(
    positions_mid: torch.Tensor,
    token_start: int,
    token_stop: int,
) -> list[tuple[int, int, list[torch.Tensor], tuple[int, ...]]] | None:
    if token_stop - token_start <= 1 or positions_mid.shape[1] <= 1:
        validated = _validate_layout_chunk(positions_mid, token_start, token_stop)
        if validated is None:
            return None
        axis_values_all, grid_sizes = validated
        return [(token_start, token_stop, axis_values_all, grid_sizes)]

    spatial = positions_mid[0, 1:, token_start:token_stop]
    is_frame_start = torch.all(spatial == spatial[:, :1], dim=0)
    frame_starts = torch.nonzero(is_frame_start, as_tuple=False).flatten()
    if frame_starts.numel() == 0 or int(frame_starts[0].item()) != 0:
        return None

    boundaries = token_start + frame_starts
    boundaries = torch.cat([boundaries, boundaries.new_tensor([token_stop])])
    splits: list[tuple[int, int, list[torch.Tensor], tuple[int, ...]]] = []
    for start, stop in zip(boundaries[:-1].tolist(), boundaries[1:].tolist(), strict=False):
        validated = _validate_layout_chunk(positions_mid, start, stop)
        if validated is None:
            return None
        axis_values_all, grid_sizes = validated
        splits.append((start, stop, axis_values_all, grid_sizes))
    return splits


def _split_constant_time_run(
    positions_mid: torch.Tensor,
    token_start: int,
    token_stop: int,
) -> list[tuple[int, int, list[torch.Tensor], tuple[int, ...]]] | None:
    return _split_constant_time_run_by_frame_starts(positions_mid, token_start, token_stop)


def _map_axis_values(global_values: torch.Tensor, local_values: torch.Tensor) -> torch.Tensor:
    if local_values.device != global_values.device:
        local_values = local_values.to(device=global_values.device)
    indices = torch.searchsorted(global_values, local_values)
    if indices.numel() and not torch.equal(global_values.index_select(0, indices), local_values):
        raise ValueError("LTX2 RoPE could not map layout axis values to the cached frequency table.")
    return indices


def _build_axis_cache(
    positions_mid: torch.Tensor,
    indices: torch.Tensor,
    axis: int,
    axis_max: int,
    freqs_dtype: torch.dtype,
) -> RopeAxisCache:
    axis_values = torch.unique(positions_mid[0, axis], sorted=True)
    axis_freqs = _build_axis_freqs(indices, axis_values.to(device=indices.device), axis_max)
    return RopeAxisCache(
        values=axis_values,
        cos=axis_freqs.cos().to(freqs_dtype),
        sin=axis_freqs.sin().to(freqs_dtype),
    )


def _make_layout(
    token_start: int,
    token_stop: int,
    grid_sizes: tuple[int, ...],
    axis_values_all: list[torch.Tensor],
    rope_axes: tuple[int, ...],
    axis_caches: tuple[RopeAxisCache, ...],
    rope_type: LTXRopeType,
    num_attention_heads: int | None,
    freq_dim: int,
    pad_size: int,
) -> RopeLayoutCache:
    axis_cos: list[torch.Tensor] = []
    axis_sin: list[torch.Tensor] = []
    for axis_index, axis in enumerate(rope_axes):
        axis_indices = _map_axis_values(axis_caches[axis_index].values, axis_values_all[axis])
        cos_axis = axis_caches[axis_index].cos.index_select(0, axis_indices)
        sin_axis = axis_caches[axis_index].sin.index_select(0, axis_indices)
        axis_cos.append(cos_axis)
        axis_sin.append(sin_axis)
    return RopeLayoutCache(
        token_start=token_start,
        token_stop=token_stop,
        grid_sizes=grid_sizes,
        axis_cos=tuple(axis_cos),
        axis_sin=tuple(axis_sin),
    )


def _build_layouts(
    positions_mid: torch.Tensor,
    rope_axes: tuple[int, ...],
    axis_caches: tuple[RopeAxisCache, ...],
    rope_type: LTXRopeType,
    num_attention_heads: int | None,
    freq_dim: int,
    pad_size: int,
) -> tuple[RopeLayoutCache, ...] | None:
    if positions_mid.shape[1] == 0:
        return ()
    time_positions = positions_mid[0, 0]
    if time_positions.numel() == 0:
        return ()

    changes = torch.nonzero(time_positions[1:] != time_positions[:-1], as_tuple=False).flatten() + 1
    starts = torch.cat([torch.zeros(1, device=changes.device, dtype=torch.long), changes])
    stops = torch.cat([changes, torch.tensor([time_positions.numel()], device=changes.device, dtype=torch.long)])

    runs: list[tuple[int, int, list[torch.Tensor], tuple[int, ...]]] = []
    for start, stop in zip(starts.tolist(), stops.tolist(), strict=False):
        split_runs = _split_constant_time_run(positions_mid, start, stop)
        if split_runs is None:
            return None
        runs.extend(split_runs)

    if not runs:
        return ()

    layouts: list[RopeLayoutCache] = []
    seg_start, seg_stop, seg_axis_values, seg_grid_sizes = runs[0]
    seg_times = [seg_axis_values[0][0]]
    spatial_axis_values = seg_axis_values[1:]
    spatial_grid_sizes = seg_grid_sizes[1:]
    last_time = float(seg_times[0].item())

    for start, stop, axis_values_all, grid_sizes in runs[1:]:
        run_time = float(axis_values_all[0][0].item())
        same_spatial = spatial_grid_sizes == grid_sizes[1:] and all(
            torch.equal(axis_values_all[axis], spatial_axis_values[axis - 1]) for axis in range(1, len(axis_values_all))
        )
        if same_spatial and run_time > last_time:
            seg_stop = stop
            seg_times.append(axis_values_all[0][0])
            last_time = run_time
            continue
        merged_axis_values = [torch.stack(seg_times), *spatial_axis_values]
        merged_grid_sizes = (len(seg_times), *spatial_grid_sizes)
        layouts.append(_make_layout(seg_start, seg_stop, merged_grid_sizes, merged_axis_values, rope_axes, axis_caches, rope_type, num_attention_heads, freq_dim, pad_size))
        seg_start, seg_stop, seg_axis_values, seg_grid_sizes = start, stop, axis_values_all, grid_sizes
        seg_times = [axis_values_all[0][0]]
        spatial_axis_values = axis_values_all[1:]
        spatial_grid_sizes = grid_sizes[1:]
        last_time = run_time

    merged_axis_values = [torch.stack(seg_times), *spatial_axis_values]
    merged_grid_sizes = (len(seg_times), *spatial_grid_sizes)
    layouts.append(_make_layout(seg_start, seg_stop, merged_grid_sizes, merged_axis_values, rope_axes, axis_caches, rope_type, num_attention_heads, freq_dim, pad_size))
    return tuple(layouts)


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
    rope_max_pos: list[int] | None = None,
    freq_grid_generator: Callable[[float, int, int], torch.Tensor] = generate_freq_grid_pytorch,
) -> RopeCache:
    if max_pos is None:
        max_pos = [20, 2048, 2048]
    freqs_dtype = torch.float32 if USE_FP32_ROPE_FREQS else out_dtype
    positions_mid = _reduce_positions(positions, use_middle_indices_grid)
    if not _positions_match_across_batch(positions_mid):
        raise ValueError("LTX2 RoPE expects identical token positions across the batch.")

    pos_dims = positions_mid.shape[1]
    if rope_axes is None:
        rope_axes = tuple(range(pos_dims))
    rope_max = rope_max_pos if rope_max_pos is not None else [max_pos[axis] for axis in rope_axes]
    num_rope_axes = len(rope_axes)
    token_count = positions_mid.shape[-1]
    if num_rope_axes == 0:
        return RopeCache(
            axes=(),
            layouts=(),
            rope_axes=rope_axes,
            pad_size=0,
            rope_type=rope_type,
            freq_dim=0,
            token_count=token_count,
            num_attention_heads=num_attention_heads,
            use_fp32_freqs=USE_FP32_ROPE_FREQS,
            split_groups=(),
            split_max_head_count=0,
            split_max_freq_width=0,
        )

    indices = freq_grid_generator(theta, num_rope_axes, dim).to(device=positions_mid.device)
    axis_width = indices.shape[0]
    if rope_type == LTXRopeType.SPLIT:
        freq_dim = dim // 2
        pad_size = freq_dim - axis_width * num_rope_axes
    else:
        freq_dim = dim
        pad_size = dim - 2 * axis_width * num_rope_axes
    if pad_size < 0:
        raise ValueError("LTX2 RoPE received incompatible dimensions for the requested axes.")

    axis_caches = tuple(
        _build_axis_cache(positions_mid, indices, axis, rope_max[axis_index], freqs_dtype)
        for axis_index, axis in enumerate(rope_axes)
    )
    split_groups: tuple[tuple[RopeSplitGroup, ...], ...] = ()
    split_max_head_count = 0
    split_max_freq_width = 0
    if rope_type == LTXRopeType.SPLIT:
        if num_attention_heads <= 0:
            raise ValueError("LTX2 RoPE expects a positive number of attention heads for split RoPE.")
        half_dim = freq_dim // num_attention_heads
        split_groups, split_max_head_count, split_max_freq_width = _build_split_groups(
            num_attention_heads,
            half_dim,
            num_rope_axes,
            axis_width,
            pad_size,
        )
    layouts = _build_layouts(positions_mid, rope_axes, axis_caches, rope_type, num_attention_heads, freq_dim, pad_size)
    if layouts is None:
        raise ValueError("LTX2 RoPE could not derive a broadcastable token layout from the latent positions.")

    return RopeCache(
        axes=axis_caches,
        layouts=layouts,
        rope_axes=rope_axes,
        pad_size=pad_size,
        rope_type=rope_type,
        freq_dim=freq_dim,
        token_count=token_count,
        num_attention_heads=num_attention_heads,
        use_fp32_freqs=USE_FP32_ROPE_FREQS,
        split_groups=split_groups,
        split_max_head_count=split_max_head_count,
        split_max_freq_width=split_max_freq_width,
    )
