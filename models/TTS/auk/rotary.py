"""AuK's interleaved rotary positions, without the x-transformers dependency."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('inv_freq', 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))
        self._lock_dtype = torch.float32

    def forward(self, length, device):
        positions = torch.arange(length, device=device, dtype=torch.float32)
        frequencies = positions[:, None] * self.inv_freq.to(device=device, dtype=torch.float32)[None, :]
        return frequencies.cos(), frequencies.sin()


def apply_rotary_pos_emb(tensor, rope):
    # Disposable B,H,T,D queries/keys; keep the arithmetic in FP32 and use one scratch.
    cos, sin = rope
    scratch = torch.empty_like(tensor, dtype=torch.float32)
    first, second = tensor[..., ::2], tensor[..., 1::2]
    torch.mul(first, cos, out=scratch[..., ::2])
    torch.mul(second, cos, out=scratch[..., 1::2])
    scratch[..., ::2].addcmul_(second, sin, value=-1)
    scratch[..., 1::2].addcmul_(first, sin)
    tensor.copy_(scratch)
    return tensor
