"""
Seed Management Utilities

Centralized seed management for reproducible generation with support for:
- Main seed randomization and setting
- Subseed variation (seed interpolation)
- PyTorch Generator creation
- Cross-platform reproducibility
"""

import random
import os
import torch
import numpy as np


def set_seed(seed):
    """
    Set all random seeds for reproducible results across Python, NumPy, and PyTorch.
    
    Args:
        seed: Integer seed value, or -1/None to generate random seed
        
    Returns:
        The seed value that was set (generated if input was -1/None)
    """
    seed = random.randint(0, 99999999) if seed is None or seed < 0 else seed
    
    # Set all random number generators
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed


def initialize_subseed(subseed, subseed_strength):
    """
    Initialize subseed value for variation generation.
    Only randomizes if subseed=-1 AND subseed_strength > 0.
    
    Args:
        subseed: Integer subseed value, or -1 for random
        subseed_strength: Float 0.0-1.0, strength of variation
        
    Returns:
        Tuple of (subseed, original_subseed) where:
        - subseed: The actual subseed to use (randomized if needed)
        - original_subseed: The input subseed value (for tracking -1)
    """
    original_subseed = subseed
    
    # Only randomize if it will actually be used
    if subseed < 0 and subseed_strength > 0:
        subseed = random.randint(0, 99999999)
    
    return subseed, original_subseed


def regenerate_subseed(original_subseed, subseed_strength):
    """
    Generate a new random subseed for repeat generations.
    Only generates if original_subseed was -1 AND subseed_strength > 0.
    
    Args:
        original_subseed: The original subseed value (before any randomization)
        subseed_strength: Float 0.0-1.0, strength of variation
        
    Returns:
        New random subseed, or original_subseed if not applicable
    """
    if original_subseed < 0 and subseed_strength > 0:
        return random.randint(0, 99999999)
    return original_subseed


def create_generator(seed, device):
    """
    Create a PyTorch Generator with the specified seed.
    
    Args:
        seed: Integer seed value
        device: torch.device or string ('cpu', 'cuda', etc.)
        
    Returns:
        torch.Generator initialized with the seed
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def create_subseed_generator(subseed, subseed_strength, device):
    """
    Create a PyTorch Generator for subseed variation if enabled.
    
    Args:
        subseed: Integer subseed value
        subseed_strength: Float 0.0-1.0, strength of variation
        device: torch.device or string ('cpu', 'cuda', etc.)
        
    Returns:
        torch.Generator if variation is enabled (subseed_strength > 0 and subseed >= 0),
        None otherwise
    """
    if subseed_strength > 0 and subseed >= 0:
        return create_generator(subseed, device)
    return None


def apply_subseed_variation(latents, subseed_generator, subseed_strength):
    """
    Apply subseed variation to latents using linear interpolation.
    
    Generates a second set of latents using the subseed generator and blends them:
        result = latents * (1 - strength) + sub_latents * strength
    
    Args:
        latents: torch.Tensor - base latents from main seed
        subseed_generator: torch.Generator or None - generator for variation
        subseed_strength: Float 0.0-1.0, strength of variation
        
    Returns:
        torch.Tensor - latents with variation applied, or original latents if
        subseed_generator is None or subseed_strength is 0
    """
    if subseed_generator is not None and subseed_strength > 0:
        # Generate variation latents
        sub_latents = torch.randn(
            *latents.shape,
            dtype=latents.dtype,
            device=latents.device,
            generator=subseed_generator
        )
        
        # Linear interpolation between main and variation
        latents = latents * (1.0 - subseed_strength) + sub_latents * subseed_strength
    
    return latents

