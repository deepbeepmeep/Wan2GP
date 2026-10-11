"""Per-step tiled latent fusion geometry.

One latent canvas is denoised through overlapping spatial tiles and temporal windows. At every
step each tile is evaluated separately and the tile predictions are merged back with Gaussian
weights, so all tiles follow one coupled trajectory and no seams form. Blending clean-latent (x0)
predictions and then taking one Euler step on the canvas is identical to blending the stepped
tiles, because every tile starts from the same canvas crop and the weights are normalized per cell.

Canvas tensors use the channels-last layout ``(B, F, H, W, C)`` in latent units.

Grid planning, grid cycling, temporal windows, Gaussian weights and first-latent skipping are
ported from ComfyUI-LTXVideo (``tiled_fusion_plan.py`` / ``tiled_fusion_sampler.py``,
https://github.com/Lightricks/ComfyUI-LTXVideo, LTX-2 Community License Agreement).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FRAME_STEP = 8
DEFAULT_OVERLAP = 0.5
DEFAULT_BLEND_VAR = 0.05
MAX_GRID_SHIFT = 4


def snap_frames(n: int, step: int = FRAME_STEP) -> int:
    """Nearest ``step * k + 1`` length. Ties keep the shorter clip."""
    n = max(1, int(n))
    below = ((n - 1) // step) * step + 1
    above = below + step
    return above if above - n < n - below else below


def plan_grid(canvas: int, tile: int, min_overlap: int) -> list[int]:
    """Tile origins covering ``[0, canvas)`` with pinned endpoints and at least ``min_overlap``."""
    if canvas <= tile:
        return [0]
    min_overlap = min(max(0, min_overlap), tile - 1)
    n = 2
    while n < canvas and (n * tile - canvas) / (n - 1) < min_overlap:
        n += 1
    step = (canvas - tile) / (n - 1)
    return [round(i * step) for i in range(n)]


def shifted(base: list[int], canvas: int, tile: int, offset: int) -> list[int]:
    """Shift interior origins by ``offset``; endpoints stay pinned so coverage holds."""
    return sorted({v if i in (0, len(base) - 1) else max(1, min(canvas - tile - 1, v + offset)) for i, v in enumerate(base)})


def gaussian_weight(n: int, var: float, device: torch.device) -> torch.Tensor:
    x = torch.arange(n, dtype=torch.float32, device=device)
    return torch.exp(-((x - (n - 1) / 2) ** 2) / (n * n * 2 * var))


@dataclass
class TiledFusionPlan:
    frames: int
    height: int
    width: int
    tile_frames: int
    tile_height: int
    tile_width: int
    windows: list[int]
    skips: list[int]
    grids: list[list[tuple[int, int]]]
    time_weight: torch.Tensor
    weights: torch.Tensor
    weight_sums: list[torch.Tensor]

    @classmethod
    def build(cls, frames: int, height: int, width: int, tile_height: int, tile_width: int, tile_frames: int, *, device: torch.device, overlap: float = DEFAULT_OVERLAP, blend_var: float = DEFAULT_BLEND_VAR, grid_cycle: int = 1) -> "TiledFusionPlan":
        """All sizes are latent units. ``tile_frames <= 0`` or ``>= frames`` keeps one temporal extent."""
        tile_height, tile_width = min(tile_height, height), min(tile_width, width)
        min_overlap_y, min_overlap_x = max(2, int(tile_height * overlap)), max(2, int(tile_width * overlap))
        ys, xs = plan_grid(height, tile_height, min_overlap_y), plan_grid(width, tile_width, min_overlap_x)
        overlap_y = tile_height - (ys[1] - ys[0]) if len(ys) > 1 else tile_height
        overlap_x = tile_width - (xs[1] - xs[0]) if len(xs) > 1 else tile_width
        offset = min(MAX_GRID_SHIFT, max(0, min(overlap_x - min_overlap_x, overlap_y - min_overlap_y)))
        offsets = [(0, 0)] if grid_cycle > 1 and offset < 1 else [(0, 0), (offset, offset), (offset, 0), (0, offset)][: max(1, grid_cycle)]
        grids = [[(y, x) for y in shifted(ys, height, tile_height, oy) for x in shifted(xs, width, tile_width, ox)] for ox, oy in offsets]

        temporal = 0 < tile_frames < frames
        if temporal:
            windows = plan_grid(frames, tile_frames, max(0, int(tile_frames * overlap)))
            skips = [int(w > 0 and windows[w] < windows[w - 1] + tile_frames) for w in range(len(windows))]
        else:
            tile_frames, windows, skips = frames, [0], [0]
        time_weight = gaussian_weight(tile_frames, blend_var, device) if len(windows) > 1 else torch.ones(tile_frames, device=device)
        weights = time_weight[:, None, None] * gaussian_weight(tile_height, blend_var, device)[:, None] * gaussian_weight(tile_width, blend_var, device)[None, :]

        weight_sums = []
        for grid in grids:
            weight_sum = torch.zeros((frames, height, width), dtype=torch.float32, device=device)
            for t0, skip in zip(windows, skips):
                for y0, x0 in grid:
                    weight_sum[t0 + skip : t0 + tile_frames, y0 : y0 + tile_height, x0 : x0 + tile_width] += weights[skip:]
            weight_sums.append(weight_sum.clamp_min_(1e-12).unsqueeze(-1))
        return cls(frames, height, width, tile_frames, tile_height, tile_width, windows, skips, grids, time_weight, weights, weight_sums)

    @property
    def tiles_per_step(self) -> int:
        return len(self.windows) * len(self.grids[0])

    def tiles(self, step: int) -> list[tuple[int, int, int, int]]:
        """``(window_index, t0, y0, x0)`` for every tile evaluated at ``step``."""
        return [(window, t0, y0, x0) for window, t0 in enumerate(self.windows) for y0, x0 in self.grids[step % len(self.grids)]]

    def crop(self, canvas: torch.Tensor, t0: int, y0: int, x0: int) -> torch.Tensor:
        return canvas[:, t0 : t0 + self.tile_frames, y0 : y0 + self.tile_height, x0 : x0 + self.tile_width]

    def accumulate(self, accumulator: torch.Tensor, tile: torch.Tensor, window: int, y0: int, x0: int) -> None:
        """Add a weighted ``(B, tile_frames, tile_height, tile_width, C)`` tile prediction."""
        t0, skip = self.windows[window], self.skips[window]
        accumulator[:, t0 + skip : t0 + self.tile_frames, y0 : y0 + self.tile_height, x0 : x0 + self.tile_width] += tile[:, skip:].float() * self.weights[skip:, :, :, None]

    def normalize(self, accumulator: torch.Tensor, step: int) -> torch.Tensor:
        return accumulator.div_(self.weight_sums[step % len(self.grids)])

    def blend_windows(self, window_latents: list[torch.Tensor]) -> torch.Tensor:
        """Merge full-width ``(B, tile_frames, H, W, C)`` window latents into one canvas with the temporal weights."""
        time_weight, first = self.time_weight, window_latents[0]
        canvas = torch.zeros((first.shape[0], self.frames, *first.shape[2:]), dtype=torch.float32, device=first.device)
        weight_sum = torch.zeros(self.frames, dtype=torch.float32, device=first.device)
        for latent, t0, skip in zip(window_latents, self.windows, self.skips):
            canvas[:, t0 + skip : t0 + self.tile_frames] += latent[:, skip:].float() * time_weight[skip:, None, None, None]
            weight_sum[t0 + skip : t0 + self.tile_frames] += time_weight[skip:]
        return canvas.div_(weight_sum[None, :, None, None, None])
