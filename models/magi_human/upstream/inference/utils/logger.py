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

import logging
import os
from typing import Callable

import torch
import torch.distributed as dist


class GlobalLogger:
    _logger = None
    _rank = 0  # default rank=0 (single-node scenario)

    @classmethod
    def _init_rank(cls):
        """Initialize rank information (distributed/single-node)."""
        if dist.is_available() and dist.is_initialized():
            cls._rank = dist.get_rank()
        else:
            cls._rank = int(os.getenv("RANK", 0))

    @classmethod
    def get_logger(cls, name=__name__, level=logging.INFO):
        if cls._logger is None:
            cls._init_rank()
            cls._logger = logging.getLogger("infra_logger")
            cls._logger.setLevel(level)
            cls._logger.propagate = False
            cls._logger.handlers.clear()

            formatter = logging.Formatter("[%(asctime)s - %(levelname)s] [Rank %(rank)s] %(message)s")

            class RankInjectHandler(logging.StreamHandler):
                def emit(self, record):
                    record.rank = cls._rank
                    super().emit(record)

            handler = RankInjectHandler()
            handler.setFormatter(formatter)
            cls._logger.addHandler(handler)

        return cls._logger


infra_logger = GlobalLogger.get_logger()


def print_per_rank(message, *args, **kwargs):
    infra_logger.info(message, *args, **kwargs)


def print_rank_0(message, *args, **kwargs):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            infra_logger.info(message, *args, **kwargs)
    else:
        infra_logger.info(message, *args, **kwargs)


def print_rank_last(message, *args, **kwargs):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            infra_logger.info(message, *args, **kwargs)
    else:
        infra_logger.info(message, *args, **kwargs)


def print_mem_info_rank_0(prefix: str = ""):
    "Print the allocated and reserved GPU memory on device 0."
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()

    allocated = round(allocated / 1024 / 1024 / 1024, 2)
    reserved = round(reserved / 1024 / 1024 / 1024, 2)
    max_allocated = round(max_allocated / 1024 / 1024 / 1024, 2)
    max_reserved = round(max_reserved / 1024 / 1024 / 1024, 2)

    print_rank_0(
        prefix
        + f" GPU 0 memory allocated: {allocated} GB, max_allocated: {max_allocated} GB, reserved: {reserved} GB, max_reserved: {max_reserved} GB"
    )


def print_model_size(model: torch.nn.Module, prefix: str = "", print_func: Callable[[str], None] = print):
    model_size_gb = sum([p.nelement() * p.element_size() for p in model.parameters()]) / (1024**3)
    parameter_count = sum([p.nelement() for p in model.parameters()])
    print_func(f"{prefix} Model size: {model_size_gb:.2f} GB, parameter count: {parameter_count}")
