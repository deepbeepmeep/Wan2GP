"""
UniPC multistep scheduler - high-quality flow matching sampler
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

import torch
from typing import Tuple, Optional, Union

from .base_sampler import BaseSampler
from .sampler_registry import SAMPLER_REGISTRY

# Import existing scheduler class
from shared.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


@SAMPLER_REGISTRY.register
class UniPCSampler(BaseSampler):
    """
    UniPC multistep scheduler - high-quality flow matching sampler.
    Good balance of speed and quality.
    """
    
    @property
    def name(self) -> str:
        return "unipc"
    
    @property
    def display_name(self) -> str:
        return "UniPC"
    
    @property
    def description(self) -> str:
        return "UniPC multistep scheduler with flow matching. Good balance of speed and quality."
    
    def get_default_steps(self) -> int:
        return 50
    
    def setup_timesteps(self, num_steps: int, device: Union[str, torch.device], shift: float,
                       num_timesteps: int = 1000, **kwargs) -> Tuple[torch.Tensor, FlowUniPCMultistepScheduler]:
        """Setup UniPC scheduler"""
        
        # Original logic from any2video.py lines 401-405
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_timesteps, 
            shift=1, 
            use_dynamic_shifting=False
        )
        scheduler.set_timesteps(num_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps
        
        return timesteps, scheduler
    
    def step(self, context) -> torch.Tensor:
        """Perform UniPC step using elegant context pattern"""
        
        # Extract only the parameters UniPC needs
        model_output = context.model_output
        timestep = context.timestep
        sample = context.sample
        scheduler = context.scheduler
        
        if scheduler is None:
            raise ValueError("UniPC sampler requires scheduler object")
        
        # UniPC only needs basic parameters - no filtering needed!
        scheduler_kwargs = {}
        if context.generator is not None:
            scheduler_kwargs['generator'] = context.generator
        
        # Use the scheduler's step method
        result = scheduler.step(model_output, timestep, sample, **scheduler_kwargs)
        return result.prev_sample if hasattr(result, 'prev_sample') else result[0]
