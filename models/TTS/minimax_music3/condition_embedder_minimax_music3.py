# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""WanGP-local port of Diffusers' MiniMax Music 3 condition encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class MiniMaxMusic3ConditionEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        condition_hidden_dim: int = 4096,
        num_condition_layers: int = 8,
        out_dim: int = 2048,
        input_sampling_rate: int = 24000,
        input_hop_length: int = 960,
        output_sampling_rate: int = 44100,
        output_hop_length: int = 512,
    ):
        super().__init__()
        self.layer_weight_logits = nn.Parameter(torch.zeros(num_condition_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv1d(condition_hidden_dim, out_dim, kernel_size=3, padding=1)
        self._interrupt_check = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = hidden_states.shape
        num_layers = self.config.num_condition_layers
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, num_layers, self.config.condition_hidden_dim, num_frames)
        layer_weights = torch.softmax(self.layer_weight_logits, dim=0).to(hidden_states.dtype)
        hidden_states = torch.einsum("blht,l->bht", hidden_states, layer_weights)
        hidden_states = self.layer_scale.to(hidden_states.dtype) * hidden_states
        hidden_states = self.proj(hidden_states)
        if self._interrupt_check is not None and self._interrupt_check():
            raise InterruptedError("MiniMax Music 3 generation interrupted")
        latent_length = max(1, int(num_frames * self.config.output_sampling_rate / self.config.input_sampling_rate * self.config.input_hop_length / self.config.output_hop_length))
        hidden_states = F.interpolate(hidden_states, size=latent_length, mode="nearest")
        return hidden_states.transpose(1, 2)
