# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SCAIL model integration for WanGP

import torch
from typing import Optional


def build_scail_pose_tokens(model, pose_latents: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Build SCAIL pose tokens to be concatenated after image tokens.

    Upstream SCAIL treats pose as an additional token sequence (typically at half spatial resolution),
    embedded by a dedicated Conv3d (`pose_patch_embedding`) and concatenated to the main token stream.

    Args:
        model: WanModel instance with `pose_patch_embedding` (Conv3d).
        pose_latents: VAE-encoded pose video (B, 16, T, H, W).
        target_dtype: Optional dtype to cast pose_latents before embedding.

    Returns:
        Pose tokens tensor (B, S_pose, dim).
    """
    if target_dtype is not None and pose_latents.dtype != target_dtype:
        pose_latents = pose_latents.to(dtype=target_dtype)

    # Upstream uses an all-ones pose mask concatenated to pose latents (16 + 4 = 20 channels).
    pose_mask = torch.ones(
        pose_latents.shape[0],
        4,
        *pose_latents.shape[2:],
        device=pose_latents.device,
        dtype=pose_latents.dtype,
    )
    pose_input = torch.cat([pose_latents, pose_mask], dim=1)
    pose_embed = model.pose_patch_embedding(pose_input)  # (B, dim, T', H', W')
    return pose_embed.flatten(2).transpose(1, 2)


def after_patch_embedding_scail(
    model,
    x: torch.Tensor,
    pose_latents: torch.Tensor,
    mask: Optional[torch.Tensor] = None
):
    """
    SCAIL-specific pose embedding injection.

    Unlike Animate, SCAIL doesn't need motion_encoder or face_encoder.
    It concatenates the mask with pose_latents (16 + 4 = 20 channels)
    and adds the pose embeddings to the latent representation.

    Args:
        model: WanModel instance with pose_patch_embedding
        x: Main latent tensor after patch embedding (B, seq_len, dim)
        pose_latents: VAE-encoded pose video (B, 16, T, H, W)
        mask: Conditioning mask (B, 4, T, H, W) to concatenate with pose_latents

    Returns:
        Modified x tensor with pose conditioning
    """
    # Concatenate pose_latents with mask to match in_dim=20
    if mask is not None:
        # pose_latents: (B, 16, T, H, W), mask: (B, 4, T, H, W) -> (B, 20, T, H, W)
        pose_input = torch.cat([pose_latents, mask], dim=1)
    else:
        # Fallback: pad with zeros if no mask available
        pad = torch.zeros(
            pose_latents.shape[0], 4, *pose_latents.shape[2:],
            device=pose_latents.device, dtype=pose_latents.dtype
        )
        pose_input = torch.cat([pose_latents, pad], dim=1)

    # Embed pose latents through Conv3d
    # Output shape: (B, dim, T', H', W') - same as x
    pose_embed = model.pose_patch_embedding(pose_input)

    # Add pose embeddings to x (both have shape B, dim, T', H', W')
    # SCAIL doesn't have a reference frame in the latent input, so add to all positions
    x = x + pose_embed

    return x
