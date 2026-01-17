"""PyTorch utilities for NLF."""
from typing import List, Optional

import torch


def mean_stdev_masked(
    input_tensor: torch.Tensor,
    is_valid: torch.Tensor,
    items_dim: int,
    dimensions_dim: int,
    fixed_ref: Optional[torch.Tensor] = None,
):
    if fixed_ref is not None:
        mean = fixed_ref
    else:
        mean = reduce_mean_masked(input_tensor, is_valid, dim=[items_dim], keepdim=True)
    centered = input_tensor - mean

    n_new_dims = input_tensor.ndim - is_valid.ndim
    is_valid = is_valid.reshape(is_valid.shape + (1,) * n_new_dims)
    n_valid = is_valid.sum(dim=items_dim, keepdim=True, dtype=input_tensor.dtype)

    sum_of_squared_deviations = reduce_sum_masked(
        torch.square(centered), is_valid, dim=[items_dim, dimensions_dim], keepdim=True
    )

    stdev = torch.sqrt(torch.nan_to_num(sum_of_squared_deviations / n_valid) + 1e-10)
    return mean, stdev


def reduce_mean_masked(
    input_tensor: torch.Tensor,
    is_valid: Optional[torch.Tensor],
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
):
    d: List[int] = [] if dim is None else dim

    if is_valid is None:
        return torch.mean(input_tensor, dim=d, keepdim=keepdim)

    if dim is None and not keepdim:
        return torch.masked_select(input_tensor, is_valid).mean()

    n_new_dims = input_tensor.ndim - is_valid.ndim
    is_valid = is_valid.reshape(is_valid.shape + (1,) * n_new_dims)
    replaced = torch.where(is_valid, input_tensor, torch.zeros_like(input_tensor))
    sum_valid = torch.sum(replaced, dim=d, keepdim=keepdim)
    n_valid = torch.sum(is_valid, dim=d, keepdim=keepdim, dtype=input_tensor.dtype)
    return torch.nan_to_num(sum_valid / n_valid)


def reduce_sum_masked(
    input_tensor: torch.Tensor,
    is_valid: torch.Tensor,
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
):
    if dim is None and not keepdim:
        return torch.masked_select(input_tensor, is_valid).sum()

    n_new_dims = input_tensor.ndim - is_valid.ndim
    is_valid = is_valid.reshape(is_valid.shape + (1,) * n_new_dims)
    replaced = torch.where(is_valid, input_tensor, torch.zeros_like(input_tensor))
    d: List[int] = [] if dim is None else dim
    return torch.sum(replaced, dim=d, keepdim=keepdim)


def softmax(target: torch.Tensor, dim: List[int] = (-1,)):
    dim = [d if d >= 0 else d + target.ndim for d in dim]
    assert sorted(dim) == list(range(min(dim), max(dim) + 1))
    dim_min = min(dim)
    dim_max = max(dim)
    target_flattened = target.flatten(start_dim=dim_min, end_dim=dim_max)
    softmaxed_flattened = torch.softmax(target_flattened, dim=dim_min)
    return softmaxed_flattened.reshape(target.shape)


def soft_argmax(inp: torch.Tensor, dim: List[int] = (-1,)):
    return decode_heatmap(softmax(inp, dim=dim), dim=dim)


def decode_heatmap(inp: torch.Tensor, dim: List[int] = (-1,), output_coord_dim: int = -1):
    result = []
    dim = [d if d >= 0 else d + inp.ndim for d in dim]
    for d in dim:
        other_heatmap_dims = list(dim)
        other_heatmap_dims.remove(d)
        summed_over_other_heatmap_axes = torch.sum(inp, dim=other_heatmap_dims, keepdim=True)
        coords = linspace(
            0.0, 1.0, inp.shape[d], dtype=inp.dtype, device=summed_over_other_heatmap_axes.device
        )
        decoded = torch.tensordot(summed_over_other_heatmap_axes, coords, dims=([d], [0]))
        x = torch.unsqueeze(decoded, d)
        for hd in sorted(dim)[::-1]:
            x = x.squeeze(hd)
        result.append(x)
    return torch.stack(result, dim=output_coord_dim)


def linspace(
    start: float,
    stop: float,
    num: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    endpoint: bool = True,
):
    start = torch.as_tensor(start, device=device, dtype=dtype)
    stop = torch.as_tensor(stop, device=device, dtype=dtype)

    if endpoint:
        if num == 1:
            return torch.mean(torch.stack([start, stop], dim=0), dim=0, keepdim=True)
        else:
            return torch.linspace(start, stop, num, device=device, dtype=dtype)
    else:
        if num > 1:
            step = (stop - start) / num
            return torch.linspace(start, stop - step, num, device=device, dtype=dtype)
        else:
            return torch.linspace(start, stop, num, device=device, dtype=dtype)


def dynamic_partition(x: torch.Tensor, partitions: torch.Tensor, num_partitions: int):
    return [x[partitions == i] for i in torch.arange(num_partitions, device=partitions.device)]


def dynamic_stitch(index_lists: List[torch.Tensor], value_lists: List[torch.Tensor]):
    indices = torch.cat(index_lists, dim=0)
    values = torch.cat(value_lists, dim=0)
    inv_permutation = torch.argsort(indices)
    return values[inv_permutation]
