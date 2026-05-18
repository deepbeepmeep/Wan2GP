# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional
import torch
from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def _forward_impl(self, x: Tensor | list[Tensor]) -> Tensor:
        if isinstance(x, list):
            x_list = x
            x = x_list[0]
            x_list.clear()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _inference_chunk_size(self, x: Tensor) -> int | None:
        if self.training or torch.is_grad_enabled() or x.ndim < 3:
            return None
        token_count = x.shape[-2]
        if token_count <= 1:
            return None
        expand_ratio = max(1, (self.fc1.out_features + self.fc1.in_features - 1) // self.fc1.in_features)
        chunk_size = max(1, token_count // expand_ratio)
        return None if chunk_size >= token_count else chunk_size

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x_list = x
            x = x_list[0]
            x_list.clear()
        chunk_size = self._inference_chunk_size(x)
        if chunk_size is None:
            x_list = [x]
            del x
            return self._forward_impl(x_list)
        output_shape = (*x.shape[:-1], self.fc2.out_features)
        output = x.new_empty(output_shape)
        x_list = [x]
        del x
        x = x_list[0]
        x_list.clear()
        for start in range(0, x.shape[-2], chunk_size):
            end = min(start + chunk_size, x.shape[-2])
            output[..., start:end, :] = self._forward_impl(x[..., start:end, :])
        return output
