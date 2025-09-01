from typing import Union

import torch
from einops import rearrange
from torch import Tensor

# Ref: https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
# Ref: https://github.com/lucidrains/rotary-embedding-torch


def compute_rope_rotations(length: int,
                           dim: int,
                           theta: int,
                           *,
                           freq_scaling: float = 1.0,
                           device: Union[torch.device, str] = 'cpu') -> Tensor:
    assert dim % 2 == 0

    with torch.amp.autocast(device_type='cuda', enabled=False):
        pos = torch.arange(length, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freqs *= freq_scaling

        rot = torch.einsum('..., f -> ... f', pos, freqs)
        rot = torch.stack([torch.cos(rot), -torch.sin(rot), torch.sin(rot), torch.cos(rot)], dim=-1)
        rot = rearrange(rot, 'n d (i j) -> 1 n d i j', i=2, j=2)
        return rot


def apply_rope(x: Tensor, rot: Tensor) -> tuple[Tensor, Tensor]:
    with torch.amp.autocast(device_type='cuda', enabled=False):
        _x = x.float()
        _x = _x.view(*_x.shape[:-1], -1, 1, 2)
        x_out = rot[..., 0] * _x[..., 0] + rot[..., 1] * _x[..., 1]
        return x_out.reshape(*x.shape).to(dtype=x.dtype)
