import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward_list(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        x = x_list[0]
        x_list.clear()
        gate, value = x.chunk(2, -1)
        F.silu(gate, inplace=True).mul_(value)
        return gate.contiguous()
