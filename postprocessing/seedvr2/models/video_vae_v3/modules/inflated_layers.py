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

from functools import partial
from typing import Literal, Optional
from torch import Tensor
from torch.nn import Conv3d

from .inflated_lib import (
    MemoryState,
    extend_head,
    inflate_bias,
    inflate_weight,
    modify_state_dict,
)

_inflation_mode_t = Literal["none", "tail", "replicate"]
_memory_device_t = Optional[Literal["cpu", "same"]]


class InflatedCausalConv3d(Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t,
        memory_device: _memory_device_t = "same",
        **kwargs,
    ):
        self.inflation_mode = inflation_mode
        self.memory = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.memory_device = memory_device
        self.padding = (0, *self.padding[1:])  # Remove temporal pad to keep causal.

    def set_memory_device(self, memory_device: _memory_device_t):
        self.memory_device = memory_device

    def forward(self, input: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        mem_size = self.stride[0] - self.kernel_size[0]
        if (self.memory is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=self.memory)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        memory = (
            input[:, :, mem_size:].detach()
            if (mem_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )
        if (
            memory_state != MemoryState.DISABLED
            and not self.training
            and (self.memory_device is not None)
        ):
            self.memory = memory
        return super().forward(input)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.inflation_mode != "none":
            state_dict = modify_state_dict(
                self,
                state_dict,
                prefix,
                inflate_weight_fn=partial(inflate_weight, position="tail"),
                inflate_bias_fn=partial(inflate_bias, position="tail"),
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            (strict and self.inflation_mode == "none"),
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def init_causal_conv3d(
    *args,
    inflation_mode: _inflation_mode_t,
    **kwargs,
):
    """
    Initialize a Causal-3D convolution layer.
    Parameters:
        inflation_mode: Listed as below. It's compatible with all the 3D-VAE checkpoints we have.
            - none: No inflation will be conducted.
                    The loading logic of state dict will fall back to default.
            - tail / replicate: Refer to the definition of `InflatedCausalConv3d`.
    """
    return InflatedCausalConv3d(*args, inflation_mode=inflation_mode, **kwargs)
