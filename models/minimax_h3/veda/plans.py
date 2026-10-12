# SPDX-License-Identifier: MIT
# Adapted from veda-sparse/Veda-on-ComfyUI; see NOTICE and LICENSE.
"""Tile plans: which tile shape every (layer, head) uses on one geometry.

A plan is searched offline for one target token grid (T, H, W) and ships
inside the predictor bundle. Plans are only exact for the grid they were
searched on; any other grid can still use a plan (tile shapes pad any grid),
it is just outside what the predictor was trained on. `PlanTable.select`
therefore says how it chose, so the UI can tell the user.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from . import tiling


@dataclasses.dataclass(frozen=True)
class HeadGroup:
    """Heads of one layer that share a tile shape."""

    shape: tiling.TileShape
    heads: torch.Tensor  # [H'] int64


@dataclasses.dataclass
class TilePlan:
    """Per-(layer, head) tile shapes for one geometry.

    Attributes:
        name: Geometry name, e.g. '16x9_t37' (aspect and latent frames).
        grid: Target token grid (T, H, W) the plan was searched on.
        shapes: Distinct shapes referenced by `head_shape`.
        head_shape: [num_layers][num_heads] index into `shapes`.
    """

    name: str
    grid: tuple[int, int, int]
    shapes: list[tiling.TileShape]
    head_shape: list[list[int]]

    def __post_init__(self):
        self._groups: dict[tuple[int, str], list[HeadGroup]] = {}

    @property
    def num_layers(self) -> int:
        return len(self.head_shape)

    @property
    def num_heads(self) -> int:
        return len(self.head_shape[0]) if self.head_shape else 0

    def head_groups(self, layer: int,
                    device: torch.device | str) -> list[HeadGroup]:
        """Groups of heads sharing a shape (cached per layer and device)."""
        key = (layer, str(device))
        if key not in self._groups:
            row = torch.tensor(self.head_shape[layer], device="cpu")
            self._groups[key] = [
                HeadGroup(self.shapes[i],
                          torch.nonzero(row == i).view(-1).to(device))
                for i in sorted(set(self.head_shape[layer]))
            ]
        return self._groups[key]

    def transposed(self) -> TilePlan:
        """The H<->W mirrored plan (for portrait / landscape fallback)."""
        t, h, w = self.grid
        return TilePlan(self.name + '_T', (t, w, h),
                        [s.transposed() for s in self.shapes],
                        [list(r) for r in self.head_shape])

    @classmethod
    def from_json(cls, data: dict) -> TilePlan:
        return cls(name=data['geometry'], grid=tuple(data['grid']),
                   shapes=[tiling.TileShape.parse(s) for s in data['shapes']],
                   head_shape=[list(r) for r in data['head_shape']])


@dataclasses.dataclass(frozen=True)
class PlanChoice:
    """The plan picked for a grid and how it was picked.

    Attributes:
        plan: The plan to use.
        exact: The plan was searched on exactly this grid.
        how: Human-readable description, shown to the user.
    """

    plan: TilePlan
    exact: bool
    how: str


FPS = 24


def frames_from_latent_t(latent_t: int) -> int:
    """H3's video frame count for a latent length (17n + 5 frames)."""
    return 17 * ((latent_t - 2) // 5) + 5 if latent_t >= 2 else 1


def describe_grid(grid) -> str:
    """'1344x768 · 5.2 s' for a token grid (T, H, W)."""
    seconds = frames_from_latent_t(grid[0]) / FPS
    return f'{grid[2] * 32}x{grid[1] * 32} · {seconds:.1f} s'


def _aspect(grid) -> float:
    return grid[2] / grid[1]


def padding_fraction(plan: TilePlan, grid) -> float:
    """Share of the tiled slots that are padding on `grid`.

    Every head group pads the grid up to its own tile shape, so the cost
    is the mean over the shapes the plan actually uses.
    """
    real = grid[0] * grid[1] * grid[2]
    if not real:
        return 0.0
    slots = [s.num_tiles(grid) * tiling.TILE_SIZE for s in plan.shapes]
    return 1.0 - real / (sum(slots) / len(slots))


def tiling_period(plan: TilePlan) -> tuple[int, int, int]:
    """Token grid multiple that every shape of `plan` divides exactly."""
    period = [1, 1, 1]
    for shape in plan.shapes:
        for axis, size in enumerate((shape.t, shape.h, shape.w)):
            period[axis] = period[axis] * size // math.gcd(period[axis], size)
    return tuple(period)


def suggest_grid(plan: TilePlan, grid) -> tuple[int, int, int]:
    """Nearest token grid at or above `grid` that `plan` tiles exactly."""
    period = tiling_period(plan)
    return tuple(-(-g // p) * p for g, p in zip(grid, period))


def describe_request(grid) -> str:
    """'1344x768, 124 frames (5.2 s)' for a token grid, as the H3 nodes
    ask for it: pixels and a frame count, not latents."""
    frames = frames_from_latent_t(grid[0])
    return (f'{grid[2] * 32}x{grid[1] * 32}, {frames} frames '
            f'({frames / FPS:.1f} s)')


def describe_latent(grid) -> str:
    """'latent 48x84 x 72, tokens 24x42 x 72' for a token grid."""
    return (f'latent {grid[1] * 2}x{grid[2] * 2} x {grid[0]}, '
            f'tokens {grid[1]}x{grid[2]} x {grid[0]}')


class PlanTable:
    """All plans of a bundle; picks the plan for a target grid."""

    def __init__(self, plans: list[TilePlan]):
        if not plans:
            raise ValueError('a predictor bundle needs at least one plan')
        self.plans = {p.name: p for p in plans}

    def select(self, grid: tuple[int, int, int]) -> PlanChoice:
        """Plan for a target token grid (T, H, W): nearest aspect ratio,
        then nearest duration (latent frames), then least padding.

        Plans of the H<->W transposed geometry are candidates too (they
        only win when no plan has a closer aspect ratio). Never fails: a
        plan of another size still tiles any grid, it is just outside what
        the predictor was trained on, which `exact` and `how` report.
        """
        grid = tuple(int(g) for g in grid)
        candidates = list(self.plans.values())
        candidates += [p.transposed() for p in candidates]
        target = math.log(_aspect(grid))

        def cost(p):
            padding = sum(s.num_tiles(grid) for s in p.shapes)
            return (round(abs(math.log(_aspect(p.grid)) - target), 3),
                    abs(p.grid[0] - grid[0]), padding)

        plan = min(candidates, key=cost)
        exact = plan.grid == grid
        if exact:
            how = f'trained for this size ({describe_grid(grid)})'
        else:
            how = (f'nearest trained size: {describe_grid(plan.grid)} '
                   f'(this video: {describe_grid(grid)})')
        choice = PlanChoice(plan, exact, how)
        return choice

    def summary(self) -> str:
        """e.g. '16:9, 9:16, 1:1, 4:3 x 5.2 / 10.1 / 14.4 s'."""
        aspects, lengths = [], set()
        for name in sorted(self.plans):
            aspect, _, t = name.partition('_t')
            label = aspect.replace('x', ':')
            if label not in aspects:
                aspects.append(label)
            if t.isdigit():
                lengths.add(int(t))
        seconds = ' / '.join(f'{frames_from_latent_t(t) / FPS:.1f}'
                             for t in sorted(lengths))
        return f'{", ".join(aspects)} x {seconds} s'
