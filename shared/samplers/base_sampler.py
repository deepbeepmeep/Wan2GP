"""
Base sampler interface for unified sampler architecture
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, Union
import torch

from .sampler_context import SamplerContext


class BaseSampler(ABC):
    """
    Base interface for all samplers in Wan2GP.
    
    This provides a unified interface for all sampling methods, making it easy
    to add new samplers and maintain consistency across the codebase.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique sampler identifier used internally"""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in UI"""
        pass
    
    @property
    def description(self) -> str:
        """Optional description of the sampler"""
        return ""
    
    @abstractmethod
    def setup_timesteps(self, 
                       num_steps: int, 
                       device: Union[str, torch.device], 
                       shift: float,
                       num_timesteps: int = 1000,
                       **kwargs) -> Tuple[torch.Tensor, Optional[object]]:
        """
        Setup timesteps and scheduler for sampling.
        
        Args:
            num_steps: Number of inference steps
            device: Device to place tensors on
            shift: Shift parameter for flow matching
            num_timesteps: Total training timesteps
            **kwargs: Additional sampler-specific parameters
            
        Returns:
            Tuple of (timesteps, scheduler_object_or_None)
        """
        pass
    
    @abstractmethod
    def step(self, context: SamplerContext) -> torch.Tensor:
        """
        Perform one sampling step using elegant context pattern.
        
        Each sampler extracts only the parameters it needs from the context.
        This eliminates parameter mismatches and provides a truly unified interface.
        
        Args:
            context: SamplerContext containing all possible parameters
            
        Returns:
            Updated sample for next timestep
        """
        pass
    
    def supports_guidance(self) -> bool:
        """Whether this sampler supports classifier-free guidance"""
        return True
    
    def get_default_steps(self) -> int:
        """Default number of inference steps for this sampler"""
        return 50
    
    def get_recommended_shift(self) -> float:
        """Recommended shift parameter for this sampler"""
        return 3.0
