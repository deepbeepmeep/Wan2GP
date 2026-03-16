from typing import Protocol

import torch

from ...ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ...ltx_core.components.protocols import DiffusionStepProtocol
from ...ltx_core.tools import VideoLatentTools
from ...ltx_core.types import LatentState
from .constants import VIDEO_LATENT_CHANNELS, VIDEO_SCALE_FACTORS


class PipelineComponents:
    """
    Container class for pipeline components used throughout the LTX pipelines.
    Attributes:
        dtype (torch.dtype): Default torch dtype for tensors in the pipeline.
        device (torch.device): Target device to place tensors and modules on.
        video_scale_factors (SpatioTemporalScaleFactors): Scale factors (T, H, W) for VAE latent space.
        video_latent_channels (int): Number of channels in the video latent representation.
        video_patchifier (VideoLatentPatchifier): Patchifier instance for video latents.
        audio_patchifier (AudioPatchifier): Patchifier instance for audio latents.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.dtype = dtype
        self.device = device

        self.video_scale_factors = VIDEO_SCALE_FACTORS
        self.video_latent_channels = VIDEO_LATENT_CHANNELS

        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)


class DenoisingFunc(Protocol):
    """
    Protocol for a denoising function used in the LTX pipeline.
    Args:
        video_state (LatentState): The current latent state for video.
        audio_state (LatentState): The current latent state for audio.
        sigmas (torch.Tensor): A 1D tensor of sigma values for each diffusion step.
        step_index (int): Index of the current denoising step.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The denoised video and audio tensors.
    """

    def __call__(
        self, video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...


class DenoisingLoopFunc(Protocol):
    """
    Protocol for a denoising loop function used in the LTX pipeline.
    Args:
        sigmas (torch.Tensor): A 1D tensor of sigma values for each diffusion step.
        video_state (LatentState): The current latent state for video.
        audio_state (LatentState): The current latent state for audio.
        stepper (DiffusionStepProtocol): The diffusion step protocol to use.
    Returns:
        tuple[LatentState, LatentState]: The denoised video and audio latent states.
    """

    def __call__(
        self,
        sigmas: torch.Tensor,
        video_state: LatentState,
        audio_state: LatentState,
        stepper: DiffusionStepProtocol,
        preview_tools: VideoLatentTools | None = None,
    ) -> tuple[LatentState | None, LatentState | None]: ...
