"""
Euler + Simple scheduler combination
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

import torch
from typing import Tuple, Optional, Union

from .base_sampler import BaseSampler
from .sampler_registry import SAMPLER_REGISTRY
from shared.schedulers import BaseFlowScheduler


class SimpleScheduler(BaseFlowScheduler):
    """
    Simple linear scheduler for diffusion models.
    Provides a straightforward linear interpolation between noise and clean states.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        super().__init__(num_train_timesteps, shift)
        
        # Create base sigma schedule (linear from 1.0 to 0.003/1.002)
        sigma_max = 1.0
        sigma_min = 0.003 / 1.002
        self.base_sigmas = torch.linspace(sigma_max, sigma_min, num_train_timesteps, dtype=torch.float32)
        
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, shift: Optional[float] = None):
        """Set the timesteps for inference using simple linear schedule."""
        # Use base class implementation
        super().set_timesteps(num_inference_steps, device, shift)


# @SAMPLER_REGISTRY.register  # Disabled - not exposed to users
class EulerSimpleSampler(BaseSampler):
    """
    Euler + Simple scheduler combination.
    Uses a straightforward linear noise schedule with flow matching.
    Similar to the original Euler but with a simplified approach.
    """
    
    @property
    def name(self) -> str:
        return "euler_simple"
    
    @property
    def display_name(self) -> str:
        return "Euler + Simple"
    
    @property
    def description(self) -> str:
        return "Euler sampler with simple linear noise schedule. Straightforward denoising progression."
    
    def get_default_steps(self) -> int:
        return 50
    
    def get_recommended_shift(self) -> float:
        return 3.0
    
    def setup_timesteps(self, num_steps: int, device: Union[str, torch.device], shift: float,
                       num_timesteps: int = 1000, **kwargs) -> Tuple[torch.Tensor, SimpleScheduler]:
        """Setup Simple scheduler"""
        
        scheduler = SimpleScheduler(
            num_train_timesteps=num_timesteps,
            shift=shift
        )
        scheduler.set_timesteps(num_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps
        
        return timesteps, scheduler
    
    def step(self, context) -> torch.Tensor:
        """Perform Simple scheduler step using elegant context pattern"""
        
        # Extract only the parameters Simple scheduler needs
        model_output = context.model_output
        timestep = context.timestep
        sample = context.sample
        scheduler = context.scheduler
        
        if scheduler is None:
            raise ValueError("Euler+Simple sampler requires scheduler object")
        
        result = scheduler.step(model_output, timestep, sample)
        return result.prev_sample if hasattr(result, 'prev_sample') else result[0]
