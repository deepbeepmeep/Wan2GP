"""
Shared utilities for two-pass hires fix implementation across different models.

This module provides common functionality for implementing high-resolution refinement
(hires fix) across different diffusion models like Qwen, Z-Image, Flux, etc.

The hires fix process:
1. Generate at base resolution (Pass 1)
2. Extract and upscale latents
3. Add noise based on denoise strength
4. Filter LoRAs by phase
5. Generate at higher resolution (Pass 2)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from source.core.log import headless_logger
from source.runtime.wgp_bridge import parse_loras_multipliers


class HiresFixHelper:
    """Shared utilities for two-pass hires fix across diffusion models."""

    @staticmethod
    def parse_config(hires_config: Dict) -> Tuple[float, int, float, str]:
        """
        Extract hires fix parameters from config dictionary.

        Args:
            hires_config: Dictionary containing hires configuration

        Returns:
            Tuple of (scale, steps, denoise, upscale_method)
        """
        scale = float(hires_config.get("scale", 1.5))
        steps = int(hires_config.get("steps", 12))
        denoise = float(hires_config.get("denoise", 0.5))
        upscale_method = hires_config.get("upscale_method", "bicubic")
        return scale, steps, denoise, upscale_method

    @staticmethod
    def upscale_latents(
        latents: torch.Tensor,
        scale_factor: float,
        method: str = "bicubic"
    ) -> torch.Tensor:
        """
        Upscale latents using interpolation.

        Args:
            latents: Input latent tensor (B, C, H, W)
            scale_factor: Scaling factor (e.g., 1.5 for 1.5x upscale)
            method: Interpolation method ("nearest", "bilinear", "bicubic")

        Returns:
            Upscaled latent tensor
        """
        align_corners = False if method in ["bilinear", "bicubic"] else None

        upscaled = F.interpolate(
            latents,
            scale_factor=scale_factor,
            mode=method,
            align_corners=align_corners
        )

        return upscaled

    @staticmethod
    def add_denoise_noise(
        latents: torch.Tensor,
        denoise_strength: float,
        generator: torch.Generator
    ) -> torch.Tensor:
        """
        Blend latents with noise based on denoise strength.

        The formula is: result = latents * (1 - denoise) + noise * denoise

        Args:
            latents: Input latent tensor
            denoise_strength: Strength of denoising (0.0 = no noise, 1.0 = full noise)
            generator: Random generator for reproducible noise

        Returns:
            Latents blended with noise
        """
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype
        )
        noised_latents = latents * (1 - denoise_strength) + noise * denoise_strength
        return noised_latents

    @staticmethod
    def filter_loras_for_phase(
        lora_names: List[str],
        phase_multipliers: List[str],
        phase_index: int,
        num_phases: int,
        num_steps: int
    ) -> Tuple[Optional[object], List[str], int]:
        """
        Extract phase-specific LoRA multipliers and build loras_slists.

        Args:
            lora_names: List of LoRA names
            phase_multipliers: List of phase multiplier strings (e.g., ["0.85;0.0", "1.1;1.1"])
            phase_index: Which phase to extract (0-indexed)
            num_phases: Total number of phases
            num_steps: Number of inference steps for this phase

        Returns:
            Tuple of (loras_slists, phase_values, active_count)
            - loras_slists: Schedule for LoRA application (or None if error)
            - phase_values: List of multiplier values for this phase
            - active_count: Number of active (non-zero) LoRAs
        """
        try:
            from source.core.params.phase_multiplier_utils import extract_phase_values

            # Extract multipliers for the specified phase
            phase_values = extract_phase_values(
                multipliers=phase_multipliers,
                phase_index=phase_index,
                num_phases=num_phases
            )

            # Build loras_slists for this phase
            _, loras_slists, errors = parse_loras_multipliers(
                " ".join(phase_values),
                len(lora_names),
                num_steps,
                nb_phases=1  # Single phase for this pass
            )

            if errors:
                headless_logger.warning(f"Errors building loras_slists: {errors}")
                return None, phase_values, 0

            # Count active LoRAs (non-zero multipliers)
            active_count = sum(1 for v in phase_values if float(v) != 0)

            return loras_slists, phase_values, active_count

        except (ValueError, TypeError, RuntimeError) as e:
            headless_logger.error(f"Error filtering LoRAs for phase {phase_index}: {e}", exc_info=True)
            return None, [], 0

    @staticmethod
    def print_pass2_lora_summary(
        lora_names: List[str],
        phase_values: List[str],
        active_count: int
    ):
        """
        Print a formatted summary of Pass 2 LoRA configuration.

        Args:
            lora_names: List of LoRA names
            phase_values: List of multiplier values for Pass 2
            active_count: Number of active LoRAs
        """
        disabled_count = len(lora_names) - active_count

        headless_logger.essential("=" * 60)
        headless_logger.essential("Pass 2 LoRA Configuration:")
        headless_logger.essential("=" * 60)
        headless_logger.essential(f"Total LoRAs: {len(lora_names)}")
        headless_logger.essential(f"Active: {active_count} | Disabled: {disabled_count}")

        for i, (name, mult) in enumerate(zip(lora_names, phase_values), 1):
            mult_float = float(mult)
            if mult_float == 0:
                status = "disabled"
                suffix = "(disabled)"
            else:
                status = "active"
                suffix = ""

            headless_logger.essential(f"   [{status}] LoRA {i}: {name} @ {mult} {suffix}")

        headless_logger.essential("=" * 60)
