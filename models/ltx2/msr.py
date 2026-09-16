"""LTX-2.5 MSR slot conditioning, sharing WanGP's LTX sampling and VAE paths.

Checkpoint convention: https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .ltx_core.components.patchifiers import get_pixel_coords
from .ltx_core.conditioning.item import ConditioningItem
from .ltx_core.types import LatentState, VideoLatentShape


@torch.inference_mode()
def load_slot_embeddings(path):
    from mmgp import offload

    model = torch.nn.Module()
    model.register_buffer("frequencies", torch.empty(16, device="meta"))
    model.net = torch.nn.Sequential(torch.nn.Linear(33, 256, device="meta"), torch.nn.SiLU(), torch.nn.Linear(256, 128, device="meta"))
    offload.load_model_data(model, path, writable_tensors=False, default_dtype=torch.bfloat16)
    slots = {key: value.to(device="cpu", dtype=torch.float32) for key, value in model.state_dict().items()}
    # Upstream evaluates the small Fourier MLP in FP32. Keep the resulting table
    # on CPU; no additional model weights or caches need GPU residency.
    scaled = torch.arange(1, 6, device="cpu", dtype=torch.float32).unsqueeze(1) / 16.0
    phases = scaled * slots["frequencies"]
    features = torch.cat((scaled, phases.sin(), phases.cos()), dim=1)
    hidden = F.silu(F.linear(features, slots["net.0.weight"], slots["net.0.bias"]))
    return F.linear(hidden, slots["net.2.weight"], slots["net.2.bias"])


class MSRSlotConditioning(ConditioningItem):
    def __init__(self, latent, frame_offset, strength):
        self.latent = latent
        self.frame_offset = frame_offset
        self.strength = strength

    def apply_to(self, latent_state, latent_tools):
        tokens = latent_tools.patchifier.patchify(self.latent)
        bounds = latent_tools.patchifier.get_patch_grid_bounds(VideoLatentShape.from_torch_shape(self.latent.shape), device=self.latent.device)
        positions = get_pixel_coords(bounds, latent_tools.scale_factors, causal_fix=True).float()
        positions[:, 0] = (positions[:, 0] + self.frame_offset) / latent_tools.fps
        mask = tokens.new_full((*tokens.shape[:2], 1), 1.0 - self.strength)
        # LTX attention is noncausal: append with the trained coordinates, as the
        # upstream ComfyUI guide does, so shared clear_conditioning removes refs.
        return LatentState(
            latent=torch.cat((latent_state.latent, tokens), dim=1),
            clean_latent=torch.cat((latent_state.clean_latent, tokens), dim=1),
            denoise_mask=torch.cat((latent_state.denoise_mask, mask), dim=1),
            positions=torch.cat((latent_state.positions, positions), dim=2),
            keyframes_mask=None if latent_state.keyframes_mask is None else torch.cat((latent_state.keyframes_mask, torch.zeros_like(mask)), dim=1),
        )


@dataclass
class MSRReferenceImages:
    images: list
    slot_embeddings: torch.Tensor
    frame_count: int
    strength: float

    @torch.inference_mode()
    def encode(self, height, width, video_encoder, dtype, device, tiling_config):
        from .ltx_core.model.video_vae import encode_video
        from .ltx_pipelines.utils.media_io import load_video_conditioning
        from shared.utils.phase_progress import check_abort

        if self.frame_count not in (25, 33):
            raise ValueError("LTX2.5 MSR reference length must be 25 or 33 frames.")
        conditions = []
        for index, image in enumerate(self.images):
            check_abort()
            video = load_video_conditioning(image[None], height, width, 1, dtype, device)
            video = video.expand(-1, -1, self.frame_count, -1, -1)
            latent = encode_video(video, video_encoder, tiling_config)
            del video
            embedding = self.slot_embeddings[index].to(device=latent.device, dtype=latent.dtype)
            latent.add_(embedding.view(1, -1, 1, 1, 1))
            conditions.append(MSRSlotConditioning(latent, index - len(self.images), self.strength))
        return conditions
