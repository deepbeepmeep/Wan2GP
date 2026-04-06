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

import os
from datetime import timedelta

import torch

from inference.common import parse_config

from .parallel_state import initialize_model_parallel, model_parallel_is_initialized
from inference.utils import print_rank_0


def initialize_distributed():
    """Initialize torch.distributed and core model parallel."""
    config = parse_config()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print_rank_0("> torch distributed already initialized, skipping initialization ...")
    else:
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        if rank == 0:
            print_rank_0("> initializing torch distributed ...")
        # Manually set the device ids.
        if device_count > 0:
            device = rank % device_count
            torch.cuda.set_device(device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=config.engine_config.distributed_backend,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=config.engine_config.distributed_timeout_minutes),
        )

    # Set the tp, pp and dp communicators.
    if device_count > 0:
        if model_parallel_is_initialized():
            return
        initialize_model_parallel(
            tp_size=config.engine_config.tp_size,
            pp_size=config.engine_config.pp_size,
            cp_size=config.engine_config.cp_size,
            nccl_communicator_config_path=None,
            distributed_timeout_minutes=config.engine_config.distributed_timeout_minutes,
            order="tp-cp-pp-dp",
        )
