"""
Sampler registry system for unified sampler management
Copyright 2024-2025 The Wan2GP Team. All rights reserved.
"""

from typing import Dict, List, Tuple, Type
from .base_sampler import BaseSampler


class SamplerRegistry:
    """
    Central registry for all samplers in Wan2GP.
    
    This registry automatically manages all available samplers and provides
    a clean interface for accessing them throughout the application.
    """
    
    def __init__(self):
        self._samplers: Dict[str, BaseSampler] = {}
        self._sampler_classes: Dict[str, Type[BaseSampler]] = {}
    
    def register(self, sampler_class: Type[BaseSampler]):
        """
        Register a sampler class.
        
        This is typically used as a decorator:
        @SAMPLER_REGISTRY.register
        class MySampler(BaseSampler):
            ...
        """
        # Create instance to get metadata
        instance = sampler_class()
        
        # Store both class and instance
        self._sampler_classes[instance.name] = sampler_class
        self._samplers[instance.name] = instance
        
        return sampler_class
    
    def get_sampler(self, name: str) -> BaseSampler:
        """Get sampler instance by name"""
        if name not in self._samplers:
            raise ValueError(f"Unknown sampler: {name}. Available: {list(self._samplers.keys())}")
        return self._samplers[name]
    
    def get_sampler_class(self, name: str) -> Type[BaseSampler]:
        """Get sampler class by name"""
        if name not in self._sampler_classes:
            raise ValueError(f"Unknown sampler: {name}. Available: {list(self._sampler_classes.keys())}")
        return self._sampler_classes[name]
    
    def get_ui_choices(self) -> List[Tuple[str, str]]:
        """
        Get choices for UI dropdown.
        
        Returns:
            List of (display_name, internal_name) tuples
        """
        choices = []
        for sampler in self._samplers.values():
            choices.append((sampler.display_name, sampler.name))
        
        # Sort by display name for better UX
        choices.sort(key=lambda x: x[0])
        return choices
    
    def list_samplers(self) -> List[str]:
        """Get list of all registered sampler names"""
        return list(self._samplers.keys())
    
    def get_sampler_info(self, name: str) -> Dict[str, str]:
        """Get detailed info about a sampler"""
        sampler = self.get_sampler(name)
        return {
            "name": sampler.name,
            "display_name": sampler.display_name,
            "description": sampler.description,
            "default_steps": str(sampler.get_default_steps()),
            "recommended_shift": str(sampler.get_recommended_shift()),
            "supports_guidance": str(sampler.supports_guidance()),
        }


# Global registry instance
SAMPLER_REGISTRY = SamplerRegistry()


def register_sampler(sampler_class: Type[BaseSampler]):
    """Convenience function for registering samplers"""
    return SAMPLER_REGISTRY.register(sampler_class)
