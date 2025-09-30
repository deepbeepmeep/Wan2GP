"""
Base flow scheduler with common flow matching logic.
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

import torch
from typing import Union, Optional
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class BaseFlowScheduler(SchedulerMixin):
    """
    Base class for flow matching schedulers.
    Contains common logic for flow matching step and timestep handling.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self._step_index = None
        
        # Subclasses should set these in their __init__ or set_timesteps
        self.sigmas = None
        self.timesteps = None
        
    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, **kwargs) -> SchedulerOutput:
        """
        Perform one step of flow matching.
        Common implementation for all flow-based schedulers.
        """
        if self._step_index is None:
            self._init_step_index(timestep)
            
        # Get current and next sigma for flow matching
        sigma = self.sigmas[self._step_index]
        if self._step_index + 1 < len(self.sigmas):
            sigma_next = self.sigmas[self._step_index + 1]
        else:
            sigma_next = torch.zeros_like(sigma)
        
        # Flow matching step: x_{t-1} = x_t + v * (sigma_next - sigma)
        # Reshape sigma difference to match sample dimensions
        sigma_diff = (sigma_next - sigma)
        while len(sigma_diff.shape) < len(sample.shape):
            sigma_diff = sigma_diff.unsqueeze(-1)
        prev_sample = sample + model_output * sigma_diff
        
        self._step_index += 1
        
        return SchedulerOutput(prev_sample=prev_sample)
        
    def _init_step_index(self, timestep):
        """
        Initialize step index based on current timestep.
        Common implementation for all schedulers.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) > 0:
            self._step_index = indices[0].item()
        else:
            # Find closest timestep if exact match not found
            diffs = torch.abs(self.timesteps - timestep)
            self._step_index = torch.argmin(diffs).item()
            
    def _create_sigma_schedule(self, num_inference_steps: int, device: Union[str, torch.device] = None, 
                              shift: Optional[float] = None) -> torch.Tensor:
        """
        Create sigma schedule from base sigmas.
        Common logic for sampling and device placement.
        """
        # Sample from the base sigma schedule
        indices = torch.linspace(0, len(self.base_sigmas) - 1, num_inference_steps).round().long()
        sigmas = self.base_sigmas[indices]
        
        # Apply shift if provided
        if shift is None:
            shift = self.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        
        # Add final sigma (zero for complete denoising)
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        
        # Create timesteps from sigmas
        timesteps = sigmas[:-1] * self.num_train_timesteps
        
        if device is not None:
            timesteps = timesteps.to(device)
            sigmas = sigmas.to(device)
            
        return timesteps, sigmas
        
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, shift: Optional[float] = None):
        """
        Set the timesteps for inference.
        Subclasses should override this to create their specific base_sigmas first.
        """
        self.num_inference_steps = num_inference_steps
        
        if not hasattr(self, 'base_sigmas') or self.base_sigmas is None:
            raise NotImplementedError("Subclasses must define base_sigmas before calling set_timesteps")
            
        self.timesteps, self.sigmas = self._create_sigma_schedule(num_inference_steps, device, shift)
        self._step_index = None
