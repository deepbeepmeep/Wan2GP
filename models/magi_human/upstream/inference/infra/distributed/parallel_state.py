# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

"""Model and data parallel groups."""

import warnings
from datetime import timedelta
from typing import List, Optional

import torch

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Tensor parallel group information with context parallel combined.
_TENSOR_MODEL_PARALLEL_GROUP_WITH_CP = None
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each tensor model parallel group to ease calculation of
# the first local rank in the tensor model parallel group
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

_CONTEXT_PARALLEL_EXTRA_GROUP = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None


def _get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get("cga_cluster_size", 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get("max_ctas", 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get("min_ctas", 1)
        return nccl_options
    else:
        return None


def generate_masked_orthogonal_rank_groups(world_size: int, parallel_size: List[int], mask: List[bool]) -> List[List[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride) + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


class RankGenerator(object):
    def __init__(self, tp: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {"tp": self.tp, "pp": self.pp, "dp": self.dp, "cp": self.cp}
        order = order.lower()
        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({order})."
                )
            elif name not in order:
                order = order + "-" + name

        self.order = order
        self.ordered_size = [self.name_to_size[token] for token in order.split("-")]

    def get_mask(self, order: str, token: str):
        ordered_token = order.split("-")
        token = token.split("-")
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token):
        """Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
        return ranks


def initialize_model_parallel(
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-pp-dp",
) -> None:
    """Initialize model data parallel groups.
    Borrow from: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py

    Args:
        tp_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pp_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if tp_size is 4 and
            pp_size is 2, the model will be split into 2 groups of 4 GPUs.

        cp_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    if world_size % (tp_size * pp_size * cp_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tp_size "
            f"({tp_size}) x pp_size ({pp_size}) "
            f"x cp_size ({cp_size})"
        )

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError("Cannot import `yaml`. Setting custom nccl communicator configs " "requires the yaml package.")

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    dp_size: int = world_size // (tp_size * pp_size * cp_size)
    rank = torch.distributed.get_rank()
    rank_generator = RankGenerator(tp=tp_size, dp=dp_size, pp=pp_size, cp=cp_size, order=order)
    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"

    for ranks in rank_generator.get_ranks("dp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=_get_nccl_options("dp", nccl_comm_cfgs))
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GLOBAL_RANKS = ranks
    for ranks_with_cp in rank_generator.get_ranks("dp-cp"):
        group_with_cp = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, pg_options=_get_nccl_options("dp_cp", nccl_comm_cfgs)
        )
        if rank in ranks_with_cp:
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    for ranks in rank_generator.get_ranks("cp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=_get_nccl_options("cp", nccl_comm_cfgs))
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks


    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for ranks in rank_generator.get_ranks("tp-pp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=_get_nccl_options("mp", nccl_comm_cfgs))
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, "tensor model parallel group is already initialized"
    for ranks in rank_generator.get_ranks("tp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=_get_nccl_options("tp", nccl_comm_cfgs))
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the tensor + context parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP_WITH_CP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP_WITH_CP is None
    ), "tensor model parallel group with context parallel is already initialized"
    for ranks in rank_generator.get_ranks("tp-cp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=_get_nccl_options("tp_cp", nccl_comm_cfgs))
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP_WITH_CP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks

    # Build the pipeline model-parallel groups
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, "pipeline model parallel group is already initialized"
    for ranks in rank_generator.get_ranks("pp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=_get_nccl_options("pp", nccl_comm_cfgs))
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert _TENSOR_AND_DATA_PARALLEL_GROUP is None, "Tensor + data parallel group is already initialized"
    for ranks in rank_generator.get_ranks("tp-cp-dp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=_get_nccl_options("tp_cp_dp", nccl_comm_cfgs))
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
    for ranks in rank_generator.get_ranks("tp-dp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=_get_nccl_options("tp_dp", nccl_comm_cfgs))
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP = group


def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is not None


def is_unitialized() -> bool:
    """Check if parallel state has been initialized

    Deprecated. Use is_initialized instead.

    """
    warnings.warn("is_unitialized is deprecated, use is_initialized instead", DeprecationWarning)
    return not is_initialized()


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or _PIPELINE_MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_tp_group(check_initialized=True, with_context_parallel=False):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert _TENSOR_MODEL_PARALLEL_GROUP is not None, "tensor model parallel group is not initialized"
    if with_context_parallel:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP_WITH_CP is not None
        ), "tensor model parallel group with context parallel combined is not initialized"
        return _TENSOR_MODEL_PARALLEL_GROUP_WITH_CP
    else:
        assert _TENSOR_MODEL_PARALLEL_GROUP is not None, "tensor model parallel group is not initialized"
        return _TENSOR_MODEL_PARALLEL_GROUP


def get_pp_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_dp_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), "data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
        return _DATA_PARALLEL_GROUP


def get_cp_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    if _CONTEXT_PARALLEL_GROUP is None:
        return None
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GROUP


def get_cp_extra_group(check_initialized=True):
    if check_initialized:
        assert _CONTEXT_PARALLEL_EXTRA_GROUP is not None, "context parallel extra group is not initialized"
    return _CONTEXT_PARALLEL_EXTRA_GROUP


def get_tp_world_size(with_context_parallel=False):
    """Return world size for the tensor model parallel group."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    try:
        group = get_tp_group(with_context_parallel=with_context_parallel)
    except AssertionError:
        return 1
    if group is None:
        return 1
    return torch.distributed.get_world_size(group=group)


def get_pp_world_size():
    """Return world size for the pipeline model parallel group."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    try:
        group = get_pp_group()
    except AssertionError:
        return 1
    if group is None:
        return 1
    return torch.distributed.get_world_size(group=group)


def get_tp_rank(with_context_parallel=False):
    """Return my rank for the tensor model parallel group."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    try:
        group = get_tp_group(with_context_parallel=with_context_parallel)
    except AssertionError:
        return 0
    if group is None:
        return 0
    return torch.distributed.get_rank(group=group)


def get_pp_rank():
    """Return my rank for the pipeline model parallel group."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    try:
        group = get_pp_group()
    except AssertionError:
        return 0
    if group is None:
        return 0
    return torch.distributed.get_rank(group=group)


def is_pipeline_first_stage():
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    return get_pp_rank() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    return get_pp_rank() == (get_pp_world_size() - 1)


def get_tensor_model_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None, "Tensor model parallel group is not initialized"
    if with_context_parallel:
        assert (
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Tensor model parallel group with context parallel combined is not initialized"
        return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]


def get_tensor_model_parallel_ranks(with_context_parallel=False):
    """Return all global ranks for the tensor model parallel group."""
    if with_context_parallel:
        assert (
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Tensor model parallel group with context parallel combined is not initialized"
        return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP
    else:
        assert _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None, "Tensor model parallel group is not initialized"
        return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS


def get_tensor_model_parallel_last_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None, "Tensor model parallel group is not initialized"
    if with_context_parallel:
        assert (
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Tensor model parallel group with context parallel combined is not initialized"
        return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP[-1]
    else:
        return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[-1]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pp_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pp_rank()
    world_size = get_pp_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pp_rank()
    world_size = get_pp_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_dp_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            group = get_dp_group(with_context_parallel=with_context_parallel)
        except AssertionError:
            return 1
        if group is None:
            return 1
        return torch.distributed.get_world_size(group=group)
    else:
        return 1


def get_dp_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            group = get_dp_group(with_context_parallel=with_context_parallel)
        except AssertionError:
            return 0
        if group is None:
            return 0
        return torch.distributed.get_rank(group=group)
    else:
        return 0


def get_cp_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        group = get_cp_group()
        if group is None:
            return 1
        return torch.distributed.get_world_size(group=group)
    else:
        return 1


def get_cp_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        group = get_cp_group()
        if group is None:
            return 0
        return torch.distributed.get_rank(group=group)
    else:
        return 0


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP_WITH_CP
    _TENSOR_MODEL_PARALLEL_GROUP_WITH_CP = None
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP
    _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS_WITH_CP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None
    global _DATA_PARALLEL_GLOBAL_RANKS
    _DATA_PARALLEL_GLOBAL_RANKS = None
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _CONTEXT_PARALLEL_EXTRA_GROUP
    _CONTEXT_PARALLEL_EXTRA_GROUP = None
    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
