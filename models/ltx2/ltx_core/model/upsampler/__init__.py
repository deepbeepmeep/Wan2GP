"""Latent upsampler model components."""

from .model import LatentUpsampler, upsample_video
from .model_configurator import LatentUpsamplerConfigurator

__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "upsample_video",
]
