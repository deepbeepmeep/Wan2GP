"""
Euler + Beta scheduler combination
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

import torch
import math
from typing import Tuple, Optional, Union

from .base_sampler import BaseSampler
from .sampler_registry import SAMPLER_REGISTRY
from shared.schedulers import BaseFlowScheduler


class BetaScheduler(BaseFlowScheduler):
    """
    Beta noise scheduler for diffusion models.
    Uses a beta distribution for noise scheduling, providing smooth transitions.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, beta_schedule: str = "linear", shift: float = 1.0):
        super().__init__(num_train_timesteps, shift)
        self.beta_schedule = beta_schedule
        
        # Create base sigma schedule
        sigma_max = 1.0
        sigma_min = 0.003 / 1.002
        
        # Apply beta-inspired curve to the sigma schedule
        if beta_schedule == "linear":
            base_sigmas = torch.linspace(sigma_max, sigma_min, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            t = torch.linspace(0, 1, num_train_timesteps, dtype=torch.float32)
            base_sigmas = sigma_max * (1 - t**0.5) + sigma_min * t**0.5
        elif beta_schedule == "cosine":
            t = torch.linspace(0, 1, num_train_timesteps, dtype=torch.float32)
            base_sigmas = sigma_min + (sigma_max - sigma_min) * 0.5 * (1 + torch.cos(math.pi * t))
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        self.base_sigmas = base_sigmas
        
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, shift: Optional[float] = None):
        """Set the timesteps for inference using beta schedule."""
        # Use base class implementation
        super().set_timesteps(num_inference_steps, device, shift)


# @SAMPLER_REGISTRY.register  # Disabled - not exposed to users
class EulerBetaSampler(BaseSampler):
    """
    Euler + Beta scheduler combination.
    Uses beta-inspired noise schedules (linear, scaled_linear, cosine) with flow matching.
    Good for fine-tuning the denoising progression curve.
    """
    
    @property
    def name(self) -> str:
        return "euler_beta"
    
    @property
    def display_name(self) -> str:
        return "Euler + Beta"
    
    @property
    def description(self) -> str:
        return "Euler sampler with Beta noise schedules. Offers linear, scaled_linear, and cosine denoising curves."
    
    def get_default_steps(self) -> int:
        return 50
    
    def get_recommended_shift(self) -> float:
        return 3.0
    
    def setup_timesteps(self, num_steps: int, device: Union[str, torch.device], shift: float,
                       num_timesteps: int = 1000, beta_schedule: str = "linear", **kwargs) -> Tuple[torch.Tensor, BetaScheduler]:
        """Setup Beta scheduler"""
        
        scheduler = BetaScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            shift=shift
        )
        scheduler.set_timesteps(num_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps
        
        return timesteps, scheduler
    
    def step(self, context) -> torch.Tensor:
        """Perform Beta scheduler step using elegant context pattern"""
        
        # Extract only the parameters Beta scheduler needs
        model_output = context.model_output
        timestep = context.timestep
        sample = context.sample
        scheduler = context.scheduler
        
        if scheduler is None:
            raise ValueError("Euler+Beta sampler requires scheduler object")
        
        result = scheduler.step(model_output, timestep, sample)
        return result.prev_sample if hasattr(result, 'prev_sample') else result[0]
