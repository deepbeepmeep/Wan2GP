# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import List, Union

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map


class Metadata:
    def __init__(self, dtype: torch.dtype, numel: int, ndim: int, shape: List[int]):
        self.dtype = dtype
        self.numel = numel
        self.ndim = ndim
        self.shape = shape

    def __repr__(self):
        return f"Metadata(dtype={self.dtype}, numel={self.numel}, ndim={self.ndim}, shape={self.shape})"


def _gather_metadata(tensor_list: List[torch.Tensor], group: dist.ProcessGroup) -> List[List[Metadata]]:
    dist.get_rank(group)
    world_size = dist.get_world_size(group)

    local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
    assert (
        local_rank == torch.cuda.current_device()
    ), f"local_rank {local_rank} != current_device {torch.cuda.current_device()}"
    device = tensor_list[0].device if len(tensor_list) > 0 else torch.device("cuda")

    # ========== Step 1: flatten local tensor list ==========

    # Metadata: [dtype_code, numel, ndim, *shape]
    local_metadata = []

    dtype_map = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2, torch.int32: 3, torch.int64: 4, torch.uint8: 5}
    reverse_dtype_map = {v: k for k, v in dtype_map.items()}

    for t in tensor_list:
        dtype_code = dtype_map[t.dtype]
        shape = list(t.shape)
        numel = t.numel()
        local_metadata.append(torch.tensor([dtype_code, numel, len(shape)] + shape, dtype=torch.int32, device=device))

    if local_metadata:
        local_metadata_tensor = torch.cat(local_metadata)
    else:
        local_metadata_tensor = torch.empty(0, dtype=torch.int32, device=device)
    local_metadata_tensor = local_metadata_tensor.contiguous()
    local_metadata_len = torch.tensor([local_metadata_tensor.numel()], dtype=torch.int32, device=device)

    # ========== Step 2: all_gather metadata lengths ==========
    metadata_lens = [torch.empty_like(local_metadata_len) for _ in range(world_size)]
    dist.all_gather(metadata_lens, local_metadata_len, group)

    # ========== Step 3: all_gather metadata payloads (with cpu tensor) ==========
    metadata_lists = [torch.empty(m.item(), dtype=torch.int32, device=device) for m in metadata_lens]
    dist.all_gather(metadata_lists, local_metadata_tensor, group)

    # ========== Step 4: decode metadata and reconstruct tensor list ==========
    result = []
    for metadata_list in metadata_lists:
        offset = 0
        local_metadata = []
        while offset < metadata_list.numel():
            dtype_code = metadata_list[offset].item()
            numel = metadata_list[offset + 1].item()
            ndim = metadata_list[offset + 2].item()
            shape = metadata_list[offset + 3 : offset + 3 + ndim].tolist()
            offset += 3 + ndim

            local_metadata.append(Metadata(reverse_dtype_map[dtype_code], numel, ndim, shape))
        result.append(local_metadata)

    return result


def _get_dtype_and_assert_consistency(metadata_lists: List[List[Metadata]]):
    dtype_set = set()
    for metadata_list in metadata_lists:
        for metadata in metadata_list:
            dtype_set.add(metadata.dtype)
    assert len(dtype_set) == 1, f"Metadata lists are not consistent: {dtype_set}"
    return dtype_set.pop()


def _get_numel_for_each_rank(metadata_lists: List[List[Metadata]]) -> List[int]:
    return [sum(meta.numel for meta in metadata_list) for metadata_list in metadata_lists]


def gather_arbitrary_tensor_list(tensor_list: List[torch.Tensor], group: dist.ProcessGroup) -> List[torch.Tensor]:
    """
    Magic gather primitive. Provide the following features:
    1. Support tensor list with different length for each rank.
    2. Support arbitrary Tensor, which means the Tensor can have different shapes but same dtype.
    3. Support empty tensor_list in some ranks without padding.

    Args:
        tensor_list: A list of tensors to gather.
        group: The process group to use.

    Returns:
        A list of tensors gathered from all ranks.
    """

    dist.get_rank(group)
    world_size = dist.get_world_size(group)

    local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
    assert (
        local_rank == torch.cuda.current_device()
    ), f"local_rank {local_rank} != current_device {torch.cuda.current_device()}"
    device = tensor_list[0].device if len(tensor_list) > 0 else torch.device("cuda")

    # Step 1: Gather metadata
    metadata_lists = _gather_metadata(tensor_list, group)
    tensor_dtype = _get_dtype_and_assert_consistency(metadata_lists)

    # Step 2: Flatten local tensors into a single 1D buffer
    if tensor_list:
        flat_tensor = torch.cat([t.flatten() for t in tensor_list], dim=0).contiguous()
    else:
        flat_tensor = torch.empty(0, dtype=tensor_dtype, device=device)  # dummy, will be ignored

    # Step 3: Gather lengths from metadata
    all_numels_int = _get_numel_for_each_rank(metadata_lists)

    # Step 4: Allocate buffers and gather flat tensor data
    output_flat_tensors = []
    for numel in all_numels_int:
        output_flat_tensors.append(torch.empty(numel, dtype=tensor_dtype, device=device))
    dist.all_gather(output_flat_tensors, flat_tensor, group)

    # Step 5: Reconstruct individual tensors using metadata
    gathered_tensor_lists = []
    for i in range(world_size):
        flat = output_flat_tensors[i]
        if flat.numel() == 0:
            continue
        metadata_list = metadata_lists[i]
        offset = 0
        for meta in metadata_list:
            numel = meta.numel
            t = flat[offset : offset + numel].view(meta.shape).to(meta.dtype)
            offset += numel
            gathered_tensor_lists.append(t)

    return gathered_tensor_lists


def _scatter_to_context_parallel_region(input: torch.Tensor, split_sizes: List[int], group: dist.ProcessGroup = None):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    # Split along first dimension with padding.
    rank = dist.get_rank(group)
    dim_offset = sum(split_sizes[:rank])
    output = input[dim_offset : dim_offset + split_sizes[rank]].contiguous()
    return output


def scatter_to_context_parallel_region(
    inputs: Union[torch.Tensor, List[torch.Tensor]], split_sizes: List[int] = None, group: dist.ProcessGroup = None
):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    if group is None or torch.distributed.get_world_size(group) == 1:
        return inputs

    if split_sizes is None:
        assert (
            inputs.shape[0] % dist.get_world_size(group) == 0
        ), f"inputs.shape[0] {inputs.shape[0]} % dist.get_world_size(group) {dist.get_world_size(group)} != 0"
        split_sizes = [inputs.shape[0] // dist.get_world_size(group)] * dist.get_world_size(group)

    partial_func = partial(_scatter_to_context_parallel_region, split_sizes=split_sizes, group=group)
    return tree_map(partial_func, inputs)


def _gather_from_context_parallel_region(
    input: Union[torch.Tensor, List[torch.Tensor]], split_sizes: List[int], group: dist.ProcessGroup = None
):
    input = input.contiguous()
    dim_size = list(input.size())
    dim_size[0] = sum(split_sizes)

    output = torch.empty(dim_size, dtype=input.dtype, device=input.device)
    outputs = list(torch.split(output, split_sizes, dim=0))
    torch.distributed.all_gather(outputs, input, group=group)
    output = torch.concat(outputs, dim=0)

    return output


def gather_from_context_parallel_region(
    inputs: Union[torch.Tensor, List[torch.Tensor]], split_sizes: List[int] = None, group: dist.ProcessGroup = None
):
    """Gather tensors and concatinate along the first dimension."""
    if group is None or torch.distributed.get_world_size(group) == 1:
        return inputs

    if split_sizes is None:
        split_sizes = [inputs.shape[0] * dist.get_world_size(group)]
    partial_func = partial(_gather_from_context_parallel_region, split_sizes=split_sizes, group=group)
    return tree_map(partial_func, inputs)
