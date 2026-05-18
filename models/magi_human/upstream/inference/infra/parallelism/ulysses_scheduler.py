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

from typing import Generic, List, Optional, TypeVar

import torch
from torch.utils._pytree import tree_map

from inference.infra.distributed import get_cp_group, get_cp_world_size

from .gather_scatter_primitive import gather_from_context_parallel_region, scatter_to_context_parallel_region

T = TypeVar("T")


class UlyssesScheduler(Generic[T]):
    """
    A naive implementation of Ulysses scheduler for context parallel processing.

    This scheduler handles tensor dispatching and undispatching operations when tensors
    enter and exit the context parallel region. It supports arbitrary nested data structures
    containing tensors and automatically handles padding and splitting operations.

    The scheduler splits input tensors along the sequence dimension across multiple GPUs
    in the context parallel group, enabling parallel processing of long sequences.
    """

    def __init__(self):
        """Initialize the Ulysses scheduler."""
        self._cp_split_sizes: Optional[List[int]] = None

    @property
    def cp_split_sizes(self):
        """Get the current context parallel split sizes."""
        return self._cp_split_sizes

    def reset(self) -> None:
        """Clear any cached split sizes from a previous forward."""
        self._cp_split_sizes = None

    def _dispatch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dispatch a tensor to the context parallel region.

        This method automatically handles padding and splits the tensor along the sequence
        dimension across the context parallel group. The split sizes are calculated to
        distribute the sequence length as evenly as possible across all ranks.

        Args:
            x: Input tensor with shape [seq_len, ...] where seq_len is the sequence length.

        Returns:
            Dispatched tensor that has been split and distributed across the context parallel group.

        Raises:
            AssertionError: If the split sizes change between calls, indicating inconsistent
                          sequence lengths or context parallel group size.
        """
        seq_len = x.shape[0]
        cp_world_size = get_cp_world_size()
        if seq_len % cp_world_size == 0:
            cp_split_sizes = [seq_len // cp_world_size] * cp_world_size
        else:
            num_ranks_with_one_extra = seq_len % cp_world_size
            min_tokens_per_rank = (seq_len - num_ranks_with_one_extra) // cp_world_size
            cp_split_sizes = [min_tokens_per_rank + 1] * num_ranks_with_one_extra + [min_tokens_per_rank] * (
                cp_world_size - num_ranks_with_one_extra
            )
        if self._cp_split_sizes is not None:
            assert (
                self._cp_split_sizes == cp_split_sizes
            ), f"cp_split_sizes changed from {self._cp_split_sizes} to {cp_split_sizes}"
        self._cp_split_sizes = cp_split_sizes
        x = scatter_to_context_parallel_region(x, cp_split_sizes, group=get_cp_group())
        return x

    def _undispatch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Undispatch a tensor from the context parallel region.

        This method gathers the tensor parts from all ranks in the context parallel group
        and concatenates them back into the original sequence. It automatically handles
        unpadding if padding was applied during dispatch.

        Args:
            x: Dispatched tensor from the context parallel region.

        Returns:
            Reconstructed tensor with the original sequence length.
        """
        x = gather_from_context_parallel_region(x, self._cp_split_sizes, group=get_cp_group())
        return x

    def dispatch(self, tensors: T) -> T:
        """
        Apply dispatch operation to all tensor leaf nodes in a nested data structure.

        This method recursively applies the _dispatch operation to all tensors in the
        input data structure, preparing them for context parallel computation. The
        structure of the input is preserved in the output.

        Args:
            tensors: Arbitrary nested data structure containing tensors (single tensor,
                    tuple, list, dict, etc.). All tensors should have the same sequence
                    length in their first dimension.

        Returns:
            A new data structure with the same structure as input, where all tensors
            have been dispatched to the context parallel region.
        """
        return tree_map(self._dispatch, tensors)

    def undispatch(self, tensors: T) -> T:
        """
        Apply undispatch operation to all tensor leaf nodes in a nested data structure.

        This method recursively applies the _undispatch operation to all tensors in the
        input data structure, reconstructing them from the context parallel region. The
        structure of the input is preserved in the output.

        Args:
            tensors: Arbitrary nested data structure containing dispatched tensors.

        Returns:
            A new data structure with the same structure as input, where all tensors
            have been reconstructed from the context parallel region.
        """
        output = tree_map(self._undispatch, tensors)
        self.reset()
        return output

_ULYSSES_SCHEDULER = UlyssesScheduler()

def ulysses_scheduler() -> UlyssesScheduler:
    assert _ULYSSES_SCHEDULER is not None, "ulysses scheduler is not initialized"
    return _ULYSSES_SCHEDULER
