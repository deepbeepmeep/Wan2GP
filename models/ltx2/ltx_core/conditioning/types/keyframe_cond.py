import torch

from ...components.patchifiers import get_pixel_coords
from ..item import ConditioningItem
from ...tools import VideoLatentTools
from ...types import LatentState, VideoLatentShape


class VideoConditionByKeyframeIndex(ConditioningItem):
    """
    Conditions video generation on keyframe latents at a specific frame index.
    Appends keyframe tokens to the latent state with positions offset by frame_idx,
    and sets denoise strength according to the strength parameter.
    """

    def __init__(self, keyframes: torch.Tensor, frame_idx: int, strength: float):
        self.keyframes = keyframes
        self.frame_idx = frame_idx
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.keyframes)
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(self.keyframes.shape),
            device=self.keyframes.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix if self.frame_idx == 0 else False,
        )
        remove_prepend = False
        if self.frame_idx < 0:
            self.frame_idx  = - self.frame_idx 
            remove_prepend = True
        
        positions[:, 0, ...] += self.frame_idx
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] /= latent_tools.fps

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.keyframes.device,
            dtype=self.keyframes.dtype,
        )
        if remove_prepend:
            latent_frame_tokens = self.keyframes.shape[-1] * self.keyframes.shape[-2]
            tokens = tokens[:, latent_frame_tokens:]
            denoise_mask = denoise_mask[:, latent_frame_tokens:]
            positions = positions[:, :, latent_frame_tokens:]

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
        )
