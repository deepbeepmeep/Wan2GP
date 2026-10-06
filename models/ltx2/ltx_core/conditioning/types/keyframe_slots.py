# SPDX-FileCopyrightText: Copyright (c) 2025 Lightricks. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

from ...components.patchifiers import get_pixel_coords
from ...tools import VideoLatentTools
from ...types import GeneratedKeyframeLayout, LatentState
from ..item import ConditioningItem


class VideoGeneratedKeyframeSlots(ConditioningItem):
    """Appends empty, fully denoised single-pixel-frame slots at target pixel frames.

    Each slot is one latent frame of tokens whose temporal span is exactly ``[t, t + 1)``. Slots are marked in
    ``keyframes_mask`` so they receive the learned keyframe embedding (checkpoints with
    ``use_keyframes_abs_pos_embedding``). The model generates their content, which ``clear_conditioning`` extracts
    for keyframe-aware decoding. Must be the last conditioning item so the recorded token range stays valid.
    """

    def __init__(self, pixel_frame_indices: list[int]):
        self.pixel_frame_indices = tuple(int(index) for index in pixel_frame_indices)

    def apply_to(self, latent_state: LatentState, latent_tools: VideoLatentTools) -> LatentState:
        frame_shape = latent_tools.target_shape._replace(frames=1)
        tokens_per_keyframe = latent_tools.patchifier.get_token_count(frame_shape)
        latent = latent_state.latent
        positions = get_pixel_coords(latent_coords=latent_tools.patchifier.get_patch_grid_bounds(output_shape=frame_shape, device=latent.device), scale_factors=latent_tools.scale_factors).to(dtype=torch.float32)
        slot_positions = []
        for index in self.pixel_frame_indices:
            slot = positions.clone()
            slot[:, 0, ..., 0] = index
            slot[:, 0, ..., 1] = index + 1
            slot[:, 0] /= latent_tools.fps
            slot_positions.append(slot)
        new_tokens = tokens_per_keyframe * len(self.pixel_frame_indices)
        slot_tokens = latent.new_zeros((latent.shape[0], new_tokens, latent.shape[2]))
        return LatentState(
            latent=torch.cat([latent, slot_tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, latent_state.denoise_mask.new_ones((latent.shape[0], new_tokens, 1))], dim=1),
            positions=torch.cat([latent_state.positions, *slot_positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, torch.zeros_like(slot_tokens)], dim=1),
            keyframes_mask=torch.cat([latent_state.keyframes_mask, latent_state.keyframes_mask.new_ones((latent.shape[0], new_tokens, 1))], dim=1),
            generated_keyframe_layout=GeneratedKeyframeLayout(self.pixel_frame_indices, tokens_per_keyframe, latent.shape[1]),
        )
