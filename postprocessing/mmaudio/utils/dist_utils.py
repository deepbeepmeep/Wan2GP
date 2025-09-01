import os
from logging import Logger

from .logger import TensorboardLogger

local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1


def info_if_rank_zero(logger: Logger, msg: str):
    if local_rank == 0:
        logger.info(msg)


def string_if_rank_zero(logger: TensorboardLogger, tag: str, msg: str):
    if local_rank == 0:
        logger.log_string(tag, msg)
