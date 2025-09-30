"""
DPM++ solver - advanced multistep scheduler
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

import torch
from typing import Tuple, Optional, Union

from .base_sampler import BaseSampler
from .sampler_registry import SAMPLER_REGISTRY

# Import existing scheduler classes
from shared.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps


@SAMPLER_REGISTRY.register
class DPMPlusPlusSampler(BaseSampler):
    """
    DPM++ solver - advanced multistep scheduler.
    High quality results with good convergence properties.
    """
    
    @property
    def name(self) -> str:
        return "dpm++"
    
    @property
    def display_name(self) -> str:
        return "DPM++"
    
    @property
    def description(self) -> str:
        return "DPM++ multistep solver with advanced convergence properties. High quality results."
    
    def get_default_steps(self) -> int:
        return 50
    
    def setup_timesteps(self, num_steps: int, device: Union[str, torch.device], shift: float,
                       num_timesteps: int = 1000, **kwargs) -> Tuple[torch.Tensor, FlowDPMSolverMultistepScheduler]:
        """Setup DPM++ scheduler"""
        
        # Original logic from any2video.py lines 407-416
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=num_timesteps,
            shift=1,
            use_dynamic_shifting=False
        )
        
        sampling_sigmas = get_sampling_sigmas(num_steps, shift)
        timesteps, _ = retrieve_timesteps(
            scheduler,
            device=device,
            sigmas=sampling_sigmas
        )
        
        return timesteps, scheduler
    
    def step(self, context) -> torch.Tensor:
        """Perform DPM++ step using elegant context pattern"""
        
        # Extract only the parameters DPM++ needs
        model_output = context.model_output
        timestep = context.timestep
        sample = context.sample
        scheduler = context.scheduler
        
        if scheduler is None:
            raise ValueError("DPM++ sampler requires scheduler object")
        
        # DPM++ only needs basic parameters - no filtering needed!
        scheduler_kwargs = {}
        if context.generator is not None:
            scheduler_kwargs['generator'] = context.generator
        
        # Use the scheduler's step method
        result = scheduler.step(model_output, timestep, sample, **scheduler_kwargs)
        return result.prev_sample if hasattr(result, 'prev_sample') else result[0]
