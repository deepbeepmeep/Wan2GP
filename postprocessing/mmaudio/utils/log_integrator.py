"""
Integrate numerical values for some iterations
Typically used for loss computation / logging to tensorboard
Call finalize and create a new Integrator when you want to display/log
"""
from typing import Callable, Union

import torch

from .logger import TensorboardLogger
from .tensor_utils import distribute_into_histogram


class Integrator:

    def __init__(self, logger: TensorboardLogger, distributed: bool = True):
        self.values = {}
        self.counts = {}
        self.hooks = []  # List is used here to maintain insertion order

        # for binned tensors
        self.binned_tensors = {}
        self.binned_tensor_indices = {}

        self.logger = logger

        self.distributed = distributed
        self.local_rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_scalar(self, key: str, x: Union[torch.Tensor, int, float]):
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.dtype in [torch.long, torch.int, torch.bool]:
                x = x.float()

        if key not in self.values:
            self.counts[key] = 1
            self.values[key] = x
        else:
            self.counts[key] += 1
            self.values[key] += x

    def add_dict(self, tensor_dict: dict[str, torch.Tensor]):
        for k, v in tensor_dict.items():
            self.add_scalar(k, v)

    def add_binned_tensor(self, key: str, x: torch.Tensor, indices: torch.Tensor):
        if key not in self.binned_tensors:
            self.binned_tensors[key] = [x.detach().flatten()]
            self.binned_tensor_indices[key] = [indices.detach().flatten()]
        else:
            self.binned_tensors[key].append(x.detach().flatten())
            self.binned_tensor_indices[key].append(indices.detach().flatten())

    def add_hook(self, hook: Callable[[torch.Tensor], tuple[str, torch.Tensor]]):
        """
        Adds a custom hook, i.e. compute new metrics using values in the dict
        The hook takes the dict as argument, and returns a (k, v) tuple
        e.g. for computing IoU
        """
        self.hooks.append(hook)

    def reset_except_hooks(self):
        self.values = {}
        self.counts = {}

    # Average and output the metrics
    def finalize(self, prefix: str, it: int, ignore_timer: bool = False) -> None:

        for hook in self.hooks:
            k, v = hook(self.values)
            self.add_scalar(k, v)

        # for the metrics
        outputs = {}
        for k, v in self.values.items():
            avg = v / self.counts[k]
            if self.distributed:
                # Inplace operation
                if isinstance(avg, torch.Tensor):
                    avg = avg.cuda()
                else:
                    avg = torch.tensor(avg).cuda()
                torch.distributed.reduce(avg, dst=0)

                if self.local_rank == 0:
                    avg = (avg / self.world_size).cpu().item()
                    outputs[k] = avg
            else:
                # Simple does it
                outputs[k] = avg

        if (not self.distributed) or (self.local_rank == 0):
            self.logger.log_metrics(prefix, outputs, it, ignore_timer=ignore_timer)

        # for the binned tensors
        for k, v in self.binned_tensors.items():
            x = torch.cat(v, dim=0)
            indices = torch.cat(self.binned_tensor_indices[k], dim=0)
            hist, count = distribute_into_histogram(x, indices)

            if self.distributed:
                torch.distributed.reduce(hist, dst=0)
                torch.distributed.reduce(count, dst=0)
                if self.local_rank == 0:
                    hist = hist / count
            else:
                hist = hist / count

            if (not self.distributed) or (self.local_rank == 0):
                self.logger.log_histogram(f'{prefix}/{k}', hist, it)
