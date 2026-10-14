"""Spatial phase-2 fusion of LTX video and audio predictions.

One global scheduler/noise trajectory; tile target and guide tokens together.
Geometry and Gaussian overlap weights reuse the shared tiled-fusion planner.
"""

import math
from dataclasses import replace

import torch
from tqdm import tqdm

from shared.utils.tiled_fusion import TiledFusionPlan
from ...ltx_core.types import LatentStateRuntimeCache


def spatially_tiled_denoising_func(denoise_fn, output_shape, components, interrupt_check=None, callback=None):
    scale = components.video_scale_factors
    height, width = output_shape.height // scale.height, output_shape.width // scale.width
    frames = (output_shape.frames - 1) // scale.time + 1
    # H3-style 2×2 grid with at least 25% overlap, in latent cells.
    tile_height, tile_width = math.ceil(height / 1.75), math.ceil(width / 1.75)
    tiles = []
    weight_sum = None

    def prepare(state):
        nonlocal weight_sum
        if tiles:
            return
        if state.generated_keyframe_layout is not None:
            raise ValueError("Phase-2 spatial tiling does not support generated keyframe slots.")
        device = state.latent.device
        plan = TiledFusionPlan.build(frames, height, width, tile_height, tile_width, frames, device=device, overlap=0.25)
        # Guide positions can be downscaled spatially or negative in time.
        # Select by spatial center so target, layout and appearance crops align.
        centers = state.positions[0, 1:].float().mean(dim=-1)
        ys, xs = centers[0] / scale.height, centers[1] / scale.width
        weight_sum = torch.zeros((1, state.latent.shape[1], 1), device=device, dtype=torch.float32)
        for _, _, y, x in plan.tiles(0):
            indices = torch.where((ys >= y) & (ys < y + plan.tile_height) & (xs >= x) & (xs < x + plan.tile_width))[0]
            rows = (ys[indices] - y).long().clamp_(0, plan.tile_height - 1)
            cols = (xs[indices] - x).long().clamp_(0, plan.tile_width - 1)
            weights = plan.weights[0, rows, cols].view(1, -1, 1)
            weight_sum.index_add_(1, indices, weights)
            positions = state.positions[:, :, indices].clone()
            positions[:, 1].sub_(y * scale.height)
            positions[:, 2].sub_(x * scale.width)
            attention = None if state.attention_mask is None else state.attention_mask[..., indices, :][..., indices]
            tiles.append((indices, weights, positions, attention, LatentStateRuntimeCache()))
        if torch.any(weight_sum == 0):
            raise ValueError("Phase-2 tile grid does not cover every conditioning token.")

    def crop(state, tile):
        indices, _, positions, attention, cache = tile
        return replace(state, latent=state.latent[:, indices], clean_latent=state.clean_latent[:, indices], denoise_mask=state.denoise_mask[:, indices], positions=positions, attention_mask=attention, keyframes_mask=None if state.keyframes_mask is None else state.keyframes_mask[:, indices], runtime_cache=cache)

    def prewarm(video_state, audio_state, sigmas):
        prepare(video_state)
        prewarm_fn = getattr(denoise_fn, "_prewarm", None)
        if prewarm_fn is not None:
            prewarm_fn(crop(video_state, tiles[0]), audio_state, sigmas)

    def step(video_state, audio_state, sigmas, step_index):
        prepare(video_state)
        video_sum = torch.zeros_like(video_state.latent, dtype=torch.float32)
        audio_sum = None
        if step_index == 0 and callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(sigmas) - 1, pass_no=2)
        for tile in tqdm(tiles, desc=f"Phase 2 - Step {step_index + 1}/{len(sigmas) - 1}", unit="tiles", leave=False):
            if interrupt_check is not None and interrupt_check():
                return None, None
            video, audio = denoise_fn(crop(video_state, tile), audio_state, sigmas, step_index)
            if video is None or (audio_state is not None and audio is None):
                return None, None
            indices, weights = tile[:2]
            video_sum.index_add_(1, indices, video.float().mul_(weights))
            if audio is not None:
                if audio_sum is None:
                    audio_sum = audio.float().clone()
                else:
                    audio_sum.add_(audio)
            del video, audio
        video_sum.div_(weight_sum)
        if audio_sum is not None:
            audio_sum.div_(len(tiles))
            audio_sum = audio_sum.to(audio_state.latent.dtype)
        return video_sum.to(video_state.latent.dtype), audio_sum

    def cleanup():
        nonlocal weight_sum
        tiles.clear()
        weight_sum = None
        cleanup_fn = getattr(denoise_fn, "_cleanup", None)
        if cleanup_fn is not None:
            cleanup_fn()

    step._prewarm = prewarm
    step._cleanup = cleanup
    return step
