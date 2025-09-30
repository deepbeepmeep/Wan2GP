"""
Original Euler sampler - direct timestep-based implementation
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union

from .base_sampler import BaseSampler
from .sampler_registry import SAMPLER_REGISTRY


@SAMPLER_REGISTRY.register
class EulerSampler(BaseSampler):
    """
    Original Euler sampler - direct timestep-based implementation.
    Fast and simple, good baseline sampler.
    """
    
    @property
    def name(self) -> str:
        return "euler"
    
    @property
    def display_name(self) -> str:
        return "Euler"
    
    @property
    def description(self) -> str:
        return "Original Euler method with direct timestep calculation. Fast and reliable baseline sampler."
    
    def get_default_steps(self) -> int:
        return 50
    
    def setup_timesteps(self, num_steps: int, device: Union[str, torch.device], shift: float, 
                       num_timesteps: int = 1000, use_timestep_transform: bool = False, 
                       timestep_transform=None, **kwargs) -> Tuple[torch.Tensor, None]:
        """Setup timesteps for Euler sampling (original implementation)"""
        
        # Original logic from any2video.py lines 389-395
        timesteps = list(np.linspace(num_timesteps, 1, num_steps, dtype=np.float32))
        timesteps.append(0.)
        timesteps = [torch.tensor([t], device=device) for t in timesteps]
        
        if use_timestep_transform and timestep_transform is not None:
            timesteps = [timestep_transform(t, shift=shift, num_timesteps=num_timesteps) for t in timesteps][:-1]
        
        timesteps = torch.tensor(timesteps)
        
        return timesteps, None  # No scheduler object needed
    
    def step(self, context) -> torch.Tensor:
        """Perform Euler step using elegant context pattern"""
        
        # Extract only the parameters Euler needs
        model_output = context.model_output
        sample = context.sample
        timestep_index = context.timestep_index
        total_timesteps = context.total_timesteps
        num_timesteps = context.num_timesteps
        
        # Original Euler logic
        if timestep_index == len(total_timesteps) - 1:
            dt = total_timesteps[timestep_index]
        else:
            dt = total_timesteps[timestep_index] - total_timesteps[timestep_index + 1]
        
        dt = dt.item() / num_timesteps
        prev_sample = sample - model_output * dt
        
        return prev_sample
