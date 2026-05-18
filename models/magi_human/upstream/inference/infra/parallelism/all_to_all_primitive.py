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

from typing import List, Tuple, Union

import torch
import torch.distributed as dist
from einops import rearrange

from inference.utils import divide


class FakeHandle:
    def __init__(self):
        pass

    def wait(self):
        pass


def scatter_head_gather_seqlen(
    tensor: torch.Tensor, split_sizes: List[int] = None, group: dist.ProcessGroup = None, async_op: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, Union[dist.Work, FakeHandle]]]:
    """
    Scatter head_number and gather seq_len, for example:
    input: (seq_len, cp * hn, hd)
    output: (seq_len * cp, hn, hd)
    NOTE: seq_len of input maybe not equal, which depends on split_sizes[rank]
    """
    if group is None or dist.get_world_size(group) == 1:
        return tensor, FakeHandle()
    group_world_size = dist.get_world_size(group)
    if split_sizes is None:
        split_sizes = [tensor.shape[0]] * group_world_size

    _, hn, _ = tensor.shape
    if group_world_size % hn == 0 and group_world_size != hn:
        tensor = torch.repeat_interleave(tensor, repeats=divide(group_world_size, hn), dim=1).contiguous()
    assert tensor.is_contiguous()
    input_split_sizes = [tensor.shape[0]] * group_world_size
    input = rearrange(tensor, "seq (cp hn) hd -> (cp seq) hn hd", cp=group_world_size).contiguous()
    output = torch.empty([sum(split_sizes), *input.shape[1:]], device=input.device, dtype=input.dtype)
    if async_op:
        handle = dist.all_to_all_single(
            output, input, output_split_sizes=split_sizes, input_split_sizes=input_split_sizes, group=group, async_op=True
        )
        return output, handle
    else:
        dist.all_to_all_single(
            output, input, output_split_sizes=split_sizes, input_split_sizes=input_split_sizes, group=group, async_op=False
        )
        return output


def scatter_seqlen_gather_head(
    tensor: torch.Tensor, split_sizes: List[int] = None, group: dist.ProcessGroup = None, async_op: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, Union[dist.Work, FakeHandle]]]:
    """
    Scatter seq_len and gather head_number, for example:
    input: (seq_len * cp, hn, hd)
    output: (seq_len, cp * hn, hd)
    NOTE: seq_len of output maybe not equal, which depends on split_sizes[rank]
    NOTE: rearrange the tensor after communication: (cp, seq, hn, hd) -> (seq, cp * hn, hd)
    """
    if group is None or dist.get_world_size(group) == 1:
        return tensor, FakeHandle() if async_op else tensor
    group_world_size = dist.get_world_size(group)
    if split_sizes is None:
        assert (
            tensor.shape[0] % group_world_size == 0
        ), f"tensor.shape[0] {tensor.shape[0]} % group_world_size {group_world_size} != 0"
        split_sizes = [tensor.shape[0] // group_world_size] * group_world_size
    assert tensor.is_contiguous()
    assert tensor.dim() == 3, f"tensor must be 3D, but got {tensor.dim()}D"
    output = torch.empty(
        [group_world_size * split_sizes[dist.get_rank(group)], *tensor.shape[1:]], device=tensor.device, dtype=tensor.dtype
    )
    output_split_sizes = [split_sizes[dist.get_rank(group)]] * group_world_size
    if async_op:
        handle = dist.all_to_all_single(
            output, tensor, output_split_sizes=output_split_sizes, input_split_sizes=split_sizes, group=group, async_op=True
        )
        return output, handle
    else:
        dist.all_to_all_single(
            output, tensor, output_split_sizes=output_split_sizes, input_split_sizes=split_sizes, group=group, async_op=False
        )
        return output


def batch_scatter_head_gather_seqlen(
    inputs: List[torch.Tensor], split_sizes: List[int] = None, group: dist.ProcessGroup = None
) -> List[torch.Tensor]:
    """
    Batch scatter head_number and gather seq_len, for example:
    inputs[i] input: (seq_len_i, cp * hn_i, hd)
    outputs[i] output: (seq_len_i * cp, hn_i, hd)
    NOTE: seq_len of inputs maybe not equal across ranks, which depends on split_sizes[rank]
    NOTE: fuse along head dim before communication, and split back after
    """
    if group is None or dist.get_world_size(group) == 1:
        return inputs
    rank = dist.get_rank(group)
    group_world_size = dist.get_world_size(group)
    if split_sizes is None:
        split_sizes = [inputs[0].shape[0]] * group_world_size
    assert all(
        input.shape[0] == split_sizes[rank] for input in inputs
    ), f"inputs[0].shape[0] {inputs[0].shape[0]} != split_sizes[rank] {split_sizes[rank]}"
    assert all(input.dim() == 3 for input in inputs), f"inputs[0].dim() {inputs[0].dim()} != 3"
    for idx in range(len(inputs)):
        _, hn, _ = inputs[idx].shape
        if group_world_size % hn == 0 and group_world_size != hn:
            inputs[idx] = torch.repeat_interleave(inputs[idx], repeats=divide(group_world_size, hn), dim=1)
        inputs[idx] = rearrange(inputs[idx], "seq (cp hn) hd -> (cp seq) hn hd", cp=group_world_size).contiguous()

    head_split_number = [input.shape[1] for input in inputs]
    fused_input = torch.cat(inputs, dim=1).contiguous()
    input_split_sizes = [fused_input.shape[0] // group_world_size] * group_world_size

    fused_output = torch.empty([sum(split_sizes), *fused_input.shape[1:]], device=fused_input.device, dtype=fused_input.dtype)
    dist.all_to_all_single(
        fused_output,
        fused_input,
        output_split_sizes=split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=False,
    )
    outputs = torch.split(fused_output, head_split_number, dim=1)
    return outputs
