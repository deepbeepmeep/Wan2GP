import torch


def broadcast(obj, cp_group=None):
    return obj


def robust_broadcast(tensor, src=0, pg=None):
    return tensor


def split_inputs_cp(tensor, seq_dim=1, cp_group=None):
    return tensor


def cat_outputs_cp_with_grad(tensor, seq_dim=1, cp_group=None):
    if cp_group is None:
        return tensor
    chunks = [torch.empty_like(tensor) for _ in range(cp_group.size())]
    torch.distributed.all_gather(chunks, tensor, group=cp_group)
    return torch.cat(chunks, dim=seq_dim)
