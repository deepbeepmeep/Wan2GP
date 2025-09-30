"""
LCM + LTX scheduler combination
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union

from .base_sampler import BaseSampler
from .sampler_registry import SAMPLER_REGISTRY
from shared.schedulers import BaseFlowScheduler


class LCMScheduler(BaseFlowScheduler):
    """
    Simplified LCM scheduler for fast inference.
    Optimized for 2-8 steps with flow matching.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, num_inference_steps: int = 4, shift: float = 1.0):
        super().__init__(num_train_timesteps, shift)
        self.num_inference_steps = num_inference_steps
        
        # Create simple linear schedule for LCM (optimized for few steps)
        self.base_sigmas = torch.linspace(1.0, 0.003 / 1.002, num_train_timesteps, dtype=torch.float32)
        
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, **kwargs):
        """Set timesteps for LCM inference (optimized for few steps)"""
        # For LCM, use a simple linear schedule optimized for very few steps
        self.num_inference_steps = num_inference_steps
        self.sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * self.num_train_timesteps
        
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)
        self._step_index = None


@SAMPLER_REGISTRY.register
class LCMLTXSampler(BaseSampler):
    """
    LCM + LTX scheduler combination.
    Latent Consistency Model approach optimized for very fast inference (2-8 steps).
    Best for quick generation when speed is more important than maximum quality.
    """
    
    @property
    def name(self) -> str:
        return "lcm_ltx"
    
    @property
    def display_name(self) -> str:
        return "LCM + LTX"
    
    @property
    def description(self) -> str:
        return "Latent Consistency Model for ultra-fast inference. Optimized for 2-8 steps."
    
    def get_default_steps(self) -> int:
        return 4  # LCM works best with very few steps
    
    def get_recommended_shift(self) -> float:
        return 1.0  # LCM typically uses less shift
    
    def setup_timesteps(self, num_steps: int, device: Union[str, torch.device], shift: float,
                       num_timesteps: int = 1000, **kwargs) -> Tuple[torch.Tensor, LCMScheduler]:
        """Setup LCM scheduler"""
        
        # LCM works best with few steps
        effective_steps = min(num_steps, 8)
        
        scheduler = LCMScheduler(
            num_train_timesteps=num_timesteps,
            num_inference_steps=effective_steps,
            shift=shift
        )
        scheduler.set_timesteps(effective_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps
        
        return timesteps, scheduler
    
    def step(self, context) -> torch.Tensor:
        """Perform LCM step using elegant context pattern"""
        
        # Extract only the parameters LCM needs
        model_output = context.model_output
        timestep = context.timestep
        sample = context.sample
        scheduler = context.scheduler
        
        if scheduler is None:
            raise ValueError("LCM+LTX sampler requires scheduler object")
        
        # LCM uses BaseFlowScheduler which accepts any extra parameters gracefully
        result = scheduler.step(model_output, timestep, sample)
        return result.prev_sample if hasattr(result, 'prev_sample') else result[0]
