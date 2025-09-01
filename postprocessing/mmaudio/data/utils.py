import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from tensordict import MemoryMappedTensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from ..utils.dist_utils import local_rank, world_size

scratch_path = Path(os.environ['SLURM_SCRATCH'] if 'SLURM_SCRATCH' in os.environ else '/dev/shm')
shm_path = Path('/dev/shm')

log = logging.getLogger()


def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def local_scatter_torch(obj: Optional[Any]):
    if world_size == 1:
        # Just one worker. Do nothing.
        return obj

    array = [obj] * world_size
    target_array = [None]
    if local_rank == 0:
        dist.scatter_object_list(target_array, scatter_object_input_list=array, src=0)
    else:
        dist.scatter_object_list(target_array, scatter_object_input_list=None, src=0)
    return target_array[0]


class ShardDataset(Dataset):

    def __init__(self, root):
        self.root = root
        self.shards = sorted(os.listdir(root))

    def __len__(self):
        return len(self.shards)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.root, self.shards[idx]), weights_only=True)


def get_tmp_dir(in_memory: bool) -> Path:
    return shm_path if in_memory else scratch_path


def load_shards_and_share(data_path: Union[str, Path], ids: list[int],
                          in_memory: bool) -> MemoryMappedTensor:
    if local_rank == 0:
        with tempfile.NamedTemporaryFile(prefix='shared-tensor-', dir=get_tmp_dir(in_memory)) as f:
            log.info(f'Loading shards from {data_path} into {f.name}...')
            data = load_shards(data_path, ids=ids, tmp_file_path=f.name)
            data = share_tensor_to_all(data)
            torch.distributed.barrier()
            f.close()  # why does the context manager not close the file for me?
    else:
        log.info('Waiting for the data to be shared with me...')
        data = share_tensor_to_all(None)
        torch.distributed.barrier()

    return data


def load_shards(
    data_path: Union[str, Path],
    ids: list[int],
    *,
    tmp_file_path: str,
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:

    id_set = set(ids)
    shards = sorted(os.listdir(data_path))
    log.info(f'Found {len(shards)} shards in {data_path}.')
    first_shard = torch.load(os.path.join(data_path, shards[0]), weights_only=True)

    log.info(f'Rank {local_rank} created file {tmp_file_path}')
    first_item = next(iter(first_shard.values()))
    log.info(f'First item shape: {first_item.shape}')
    mm_tensor = MemoryMappedTensor.empty(shape=(len(ids), *first_item.shape),
                                         dtype=torch.float32,
                                         filename=tmp_file_path,
                                         existsok=True)
    total_count = 0
    used_index = set()
    id_indexing = {i: idx for idx, i in enumerate(ids)}
    # faster with no workers; otherwise we need to set_sharing_strategy('file_system')
    loader = DataLoader(ShardDataset(data_path), batch_size=1, num_workers=0)
    for data in tqdm(loader, desc='Loading shards'):
        for i, v in data.items():
            if i not in id_set:
                continue

            # tensor_index = ids.index(i)
            tensor_index = id_indexing[i]
            if tensor_index in used_index:
                raise ValueError(f'Duplicate id {i} found in {data_path}.')
            used_index.add(tensor_index)
            mm_tensor[tensor_index] = v
            total_count += 1

    assert total_count == len(ids), f'Expected {len(ids)} tensors, got {total_count}.'
    log.info(f'Loaded {total_count} tensors from {data_path}.')

    return mm_tensor


def share_tensor_to_all(x: Optional[MemoryMappedTensor]) -> MemoryMappedTensor:
    """
    x: the tensor to be shared; None if local_rank != 0
    return: the shared tensor
    """

    # there is no need to share your stuff with anyone if you are alone; must be in memory
    if world_size == 1:
        return x

    if local_rank == 0:
        assert x is not None, 'x must not be None if local_rank == 0'
    else:
        assert x is None, 'x must be None if local_rank != 0'

    if local_rank == 0:
        filename = x.filename
        meta_information = (filename, x.shape, x.dtype)
    else:
        meta_information = None

    filename, data_shape, data_type = local_scatter_torch(meta_information)
    if local_rank == 0:
        data = x
    else:
        data = MemoryMappedTensor.from_filename(filename=filename,
                                                dtype=data_type,
                                                shape=data_shape)

    return data
