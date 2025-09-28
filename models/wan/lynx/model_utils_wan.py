# Copyright (c) 2025 The Wan Team and The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on September 15, 2025.
#
# Original file was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/diffusers/blob/v0.30.3/LICENSE and https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

import os
import math
import torch

from typing import List, Optional, Tuple, Union

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

from diffusers import FlowMatchEulerDiscreteScheduler

""" Utility functions for Wan2.1-T2I model """


def cal_mean_and_std(vae, device, dtype):
    # https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py#L483

    # latents_mean
    latents_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1).to(device, dtype)

    # latents_std
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).to(device, dtype)

    return latents_mean, latents_std


def encode_video(vae, video):
    # Input video is formatted as [B, F, C, H, W]
    video = video.to(vae.device, dtype=vae.dtype)
    video = video.unsqueeze(0) if len(video.shape) == 4 else video

    # VAE takes [B, C, F, H, W]
    video = video.permute(0, 2, 1, 3, 4)
    latent_dist = vae.encode(video).latent_dist

    mean, std = cal_mean_and_std(vae, vae.device, vae.dtype)

    # NOTE: this implementation is weird but I still borrow it from the official
    # codes. A standard normailization should be (x - mean) / std, but the std 
    # here is computed by `1 / vae.config.latents_std` from cal_mean_and_std().

    # Sample latent with scaling
    video_latent = (latent_dist.sample() - mean) * std

    # Return as [B, C, F, H, W] as required by Wan
    return video_latent


def decode_latent(vae, latent):
    # Input latent is formatted as [B, C, F, H, W]
    latent = latent.to(vae.device, dtype=vae.dtype)
    latent = latent.unsqueeze(0) if len(latent.shape) == 4 else latent

    mean, std = cal_mean_and_std(vae, latent.device, latent.dtype)
    latent = latent / std + mean

    # VAE takes [B, C, F, H, W]
    frames = vae.decode(latent).sample

    # Return as [B, F, C, H, W]
    frames = frames.permute(0, 2, 1, 3, 4).float()

    return frames


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
    prompt_attention_mask=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")
    
    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), prompt_attention_mask.to(device)).last_hidden_state

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, prompt_attention_mask


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds[0]


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def prepare_sigmas(
    scheduler: FlowMatchEulerDiscreteScheduler,
    sigmas: torch.Tensor,
    batch_size: int,
    num_train_timesteps: int,
    flow_weighting_scheme: str = "none",
    flow_logit_mean: float = 0.0,
    flow_logit_std: float = 1.0,
    flow_mode_scale: float = 1.29,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        weights = compute_density_for_timestep_sampling(
            weighting_scheme=flow_weighting_scheme,
            batch_size=batch_size,
            logit_mean=flow_logit_mean,
            logit_std=flow_logit_std,
            mode_scale=flow_mode_scale,
            device=device,
            generator=generator,
        )
        indices = (weights * num_train_timesteps).long()
    else:
        raise ValueError(f"Unsupported scheduler type {type(scheduler)}")

    return sigmas[indices]


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u


def expand_tensor_dims(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    assert len(tensor.shape) <= ndim
    return tensor.reshape(tensor.shape + (1,) * (ndim - len(tensor.shape)))


def flow_match_xt(x0: torch.Tensor, n: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Forward process of flow matching."""
    return (1.0 - t) * x0 + t * n


def flow_match_target(n: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    r"""Loss target for flow matching."""
    return n - x0


def prepare_loss_weights(
    scheduler: FlowMatchEulerDiscreteScheduler,
    sigmas: Optional[torch.Tensor] = None,
    flow_weighting_scheme: str = "none",

) -> torch.Tensor:

    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)

    from diffusers.training_utils import compute_loss_weighting_for_sd3

    return compute_loss_weighting_for_sd3(sigmas=sigmas, weighting_scheme=flow_weighting_scheme)