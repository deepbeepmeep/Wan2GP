"""
Unified Sampler System for Wan2GP
Copyright 2024-2025 The Wan2GP Team. All rights reserved.

This module provides a unified interface for all samplers in Wan2GP, making it easy
to add new samplers and maintain consistency across the codebase.

Usage:
    from shared.samplers import get_sampler, get_sampler_choices
    
    # Get UI choices for dropdown
    choices = get_sampler_choices()
    
    # Get a specific sampler
    sampler = get_sampler("euler_beta")
    
    # Use the sampler
    timesteps, scheduler = sampler.setup_timesteps(50, device, shift)
    result = sampler.step(model_output, timestep, sample, i, timesteps, scheduler)
"""

from .sampler_registry import SAMPLER_REGISTRY, register_sampler
from .base_sampler import BaseSampler
from .sampler_context import SamplerContext

# Import all individual samplers to trigger registration
from . import euler_sampler
from . import unipc_sampler
from . import dpm_plus_plus_sampler
from . import causvid_sampler
from . import lcm_ltx_sampler

# Import disabled samplers (not registered, but available in codebase)
from . import euler_beta_sampler
from . import euler_simple_sampler


def get_sampler(name: str) -> BaseSampler:
    """
    Get a sampler by name.
    
    Args:
        name: Sampler name (e.g., "euler", "euler_beta", "lcm_ltx")
        
    Returns:
        BaseSampler instance
        
    Raises:
        ValueError: If sampler name is not found
    """
    return SAMPLER_REGISTRY.get_sampler(name)


def get_sampler_choices():
    """
    Get sampler choices for UI dropdown.
    
    Returns:
        List of (display_name, internal_name) tuples
    """
    return SAMPLER_REGISTRY.get_ui_choices()


def list_samplers():
    """Get list of all available sampler names"""
    return SAMPLER_REGISTRY.list_samplers()


def get_sampler_info(name: str):
    """Get detailed information about a sampler"""
    return SAMPLER_REGISTRY.get_sampler_info(name)


def print_available_samplers():
    """Print all available samplers with their info"""
    print("Available Samplers:")
    print("=" * 50)
    
    for name in sorted(list_samplers()):
        info = get_sampler_info(name)
        print(f"• {info['display_name']} ({info['name']})")
        if info['description']:
            print(f"  {info['description']}")
        print(f"  Default steps: {info['default_steps']}, Recommended shift: {info['recommended_shift']}")
        print()


# Export main interface
__all__ = [
    'get_sampler',
    'get_sampler_choices', 
    'list_samplers',
    'get_sampler_info',
    'print_available_samplers',
    'BaseSampler',
    'SamplerContext',
    'register_sampler',
    'SAMPLER_REGISTRY'
]


# Sampler system ready - no debug output by default
