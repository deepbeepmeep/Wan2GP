# Few-step pi-Flow (DX policy, grid mode) sampling of the distilled Kandinsky 6 checkpoints, port of kandinskylab/kandinsky-6
# kandinsky/core/algo/piflow_math.py and piflow_sampler.py (commit bf676db, MIT License, see LICENSE in this folder).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch


def shift_timesteps(t, shift):
    """Raw flow-matching time to the shifted DiT time."""
    return shift * t / (1 + (shift - 1) * t)


def segment_schedule(num_steps, final_step_size_scale, eps):
    """Raw-time segment of each step: equal segments, the last one scaled by final_step_size_scale."""
    final_step_size_scale = max(float(final_step_size_scale), eps)
    base_segment_size = 1.0 / (float(num_steps) - (1.0 - final_step_size_scale))
    return [base_segment_size * (final_step_size_scale if step == num_steps - 1 else 1.0) for step in range(num_steps)]


class DXPolicy:
    """Network-free velocity over one segment, interpolated from the grid of x0 predictions of the DiT."""

    def __init__(self, grid_prediction, x_t_src, sigma_t_src, segment_size, shift, eps):
        self.shift, self.eps = shift, eps
        self.sigma_t_src = sigma_t_src
        self.raw_t_src = self._unwarp_t(sigma_t_src)
        self.raw_t_dst = (self.raw_t_src - segment_size).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)
        # x_t - sigma * u in FP32 with one buffer: fl(x_t - fl(sigma * u)) == fl(-fl(sigma * u) + x_t)
        self.x0_grid = grid_prediction.float().mul_(sigma_t_src.unsqueeze(1)).neg_().add_(x_t_src.unsqueeze(1))

    def _unwarp_t(self, sigma_t):
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    def _interpolate(self, t):
        grid = self.x0_grid
        t = t.clamp(min=0, max=1) * (grid.size(1) - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=grid.size(1) - 2)
        t1 = t0 + 1
        values = torch.gather(grid, dim=1, index=torch.stack([t0, t1], dim=1).expand(-1, -1, *grid.shape[2:]))
        return (t1 - t) * values[:, 0] + (t - t0) * values[:, 1]

    def velocity(self, x_t, sigma_t):
        raw_t = self._unwarp_t(sigma_t)
        x_0 = self._interpolate((raw_t - self.raw_t_dst) / self.segment_size)
        return (x_t - x_0) / sigma_t.clamp(min=self.eps)


def policy_rollout(x_t, sigma_t, raw_t_start, raw_t_end, total_substeps, policy):
    """Integrates the policy velocity from raw_t_start to raw_t_end (tensors broadcastable to x_t)."""
    delta_raw_t = raw_t_start - raw_t_end
    num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps
    raw_t = raw_t_start
    for substep_id in range(int(num_substeps.max().item())):
        velocity = policy.velocity(x_t, sigma_t)
        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = shift_timesteps(raw_t_minus, policy.shift)
        x_t_minus = x_t + velocity * (sigma_t_minus - sigma_t)
        del velocity
        active = num_substeps > substep_id
        x_t = torch.where(active, x_t_minus, x_t)
        sigma_t = torch.where(active, sigma_t_minus, sigma_t)
        raw_t = torch.where(active, raw_t_minus, raw_t)
    return x_t
