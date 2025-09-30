"""
SamplerContext - Flexible parameter container for unified sampler interface.
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

from typing import Any, Dict, Optional
import torch


class SamplerContext:
    """
    Context object containing all possible parameters that samplers might need.
    Each sampler extracts only the parameters it requires.
    """
    
    def __init__(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        timestep_index: int,
        total_timesteps: torch.Tensor,
        scheduler: Optional[Any] = None,
        num_timesteps: int = 1000,
        generator: Optional[torch.Generator] = None,
        **extra_kwargs
    ):
        """
        Initialize sampler context with all possible parameters.
        
        Args:
            model_output: The model's predicted noise/velocity
            timestep: Current timestep
            sample: Current sample (latents)
            timestep_index: Index in the timestep sequence
            total_timesteps: Full timestep sequence
            scheduler: Optional scheduler object (for external schedulers)
            num_timesteps: Total number of timesteps (for Euler)
            generator: Random generator (for some schedulers)
            **extra_kwargs: Any additional parameters
        """
        self.model_output = model_output
        self.timestep = timestep
        self.sample = sample
        self.timestep_index = timestep_index
        self.total_timesteps = total_timesteps
        self.scheduler = scheduler
        self.num_timesteps = num_timesteps
        self.generator = generator
        
        # Store any additional parameters
        self.extra = extra_kwargs
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get an extra parameter with optional default."""
        return self.extra.get(key, default)
        
    def has(self, key: str) -> bool:
        """Check if an extra parameter exists."""
        return key in self.extra
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for **kwargs unpacking if needed."""
        result = {
            'model_output': self.model_output,
            'timestep': self.timestep,
            'sample': self.sample,
            'timestep_index': self.timestep_index,
            'total_timesteps': self.total_timesteps,
            'scheduler': self.scheduler,
            'num_timesteps': self.num_timesteps,
        }
        if self.generator is not None:
            result['generator'] = self.generator
        result.update(self.extra)
        return result
