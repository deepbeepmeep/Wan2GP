"""
CausVid FlowMatch scheduler - specialized flow matching
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

import torch
from typing import Tuple, Optional, Union

from .base_sampler import BaseSampler
from .sampler_registry import SAMPLER_REGISTRY

# Import existing scheduler class
from shared.utils.basic_flowmatch import FlowMatchScheduler


@SAMPLER_REGISTRY.register
class CausVidSampler(BaseSampler):
    """
    CausVid FlowMatch scheduler - specialized flow matching.
    Uses predefined timestep schedule for consistent results.
    """
    
    @property
    def name(self) -> str:
        return "causvid"
    
    @property
    def display_name(self) -> str:
        return "CausVid"
    
    @property
    def description(self) -> str:
        return "CausVid flow matching scheduler with predefined timestep schedule. Consistent results."
    
    def get_default_steps(self) -> int:
        return 9  # Uses predefined schedule
    
    def setup_timesteps(self, num_steps: int, device: Union[str, torch.device], shift: float,
                       num_timesteps: int = 1000, **kwargs) -> Tuple[torch.Tensor, FlowMatchScheduler]:
        """Setup CausVid scheduler"""
        
        # Original logic from any2video.py lines 396-400
        scheduler = FlowMatchScheduler(
            num_inference_steps=num_steps, 
            shift=shift, 
            sigma_min=0, 
            extra_one_step=True
        )
        
        # Predefined timestep schedule
        timesteps = torch.tensor([1000, 934, 862, 756, 603, 410, 250, 140, 74])[:num_steps].to(device)
        scheduler.timesteps = timesteps
        scheduler.sigmas = torch.cat([scheduler.timesteps / 1000, torch.tensor([0.], device=device)])
        
        return timesteps, scheduler
    
    def step(self, context) -> torch.Tensor:
        """Perform CausVid step using elegant context pattern"""
        
        # Extract only the parameters CausVid needs
        model_output = context.model_output
        timestep = context.timestep
        sample = context.sample
        scheduler = context.scheduler
        
        if scheduler is None:
            raise ValueError("CausVid sampler requires scheduler object")
        
        # CausVid uses FlowMatchScheduler with simple interface
        result = scheduler.step(model_output, timestep, sample)
        return result[0]  # FlowMatchScheduler returns list
