# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn


def get_mlp(mlp_type: Optional[str] = "normal"):
    if mlp_type == "normal":
        return MLP
    elif mlp_type == "swiglu":
        return SwiGLUMLP


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
    ):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")
        self.proj_out = nn.Linear(dim * expand_ratio, dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=False)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=False)
        self.dim = dim
        self.hidden_dim = hidden_dim

    def _forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        gate = F.silu(self.proj_in_gate(x))
        gate.mul_(self.proj_in(x))
        return self.proj_out(gate)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = x.shape[-2]
        if seq_len <= 1024:
            return self._forward(x)
        chunk_size = max(128, min(seq_len, seq_len * self.dim // max(2 * self.hidden_dim, 1)))
        for start in range(0, seq_len, chunk_size):
            chunk_out = self._forward(x.narrow(-2, start, min(chunk_size, seq_len - start)))
            x.narrow(-2, start, chunk_out.shape[-2]).copy_(chunk_out)
        return x
