# SPDX-License-Identifier: MIT
# Adapted from veda-sparse/Veda-on-ComfyUI; see NOTICE and LICENSE.
"""Block masks from tile scores: the Veda selection rules.

Same rules as Miowtion's `miowtion/veda/mask.py` (the tile search and the
predictor training use exactly these), restated for this repo:

  1. Global tiles are dense in both directions: every query tile sees every
     global key tile, and global query tiles see every tile.
  2. Only the video -> video quadrant is sparse (top-k per query tile).
  3. Ratio budgets are equal-kernel-cost: with n_ideal = ceil(real tokens /
     128), a query tile keeps ratio * n_ideal^2 / n_cols key tiles, so a
     padded tiling never gets a larger budget than an unpadded one.
  4. The fractional part of a budget is spread by a Bresenham pattern over
     the query tile id, so the mean kept count equals the budget exactly.
  5. A query tile's own tile is forced in and consumes budget.
  6. Empty key tiles are never selected.
  7. The video columns split into a reference block [0, n_ref_tiles)
     (tiled conditions: keyframes, guide frames, reference images / videos)
     and the generated block [n_ref_tiles, n_video_tiles) (the target
     video). Each block runs its own top-k with its own budget; one pooled
     top-k would starve the reference block, since spatial neighbours in
     the target always win.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from . import tiling

_NEG_INF = float('-inf')
_POS_INF = float('inf')


@dataclasses.dataclass(frozen=True)
class Budget:
    """Keep budget of one column block: a keep ratio or a tile count.

    Attributes:
        ratio: Keep ratio of the equal-cost budget (1 - sparsity); >= 1
            keeps everything.
        tiles: Absolute key tiles kept per query tile (overrides ratio).
    """

    ratio: float | None = None
    tiles: float | None = None

    def __post_init__(self):
        if (self.ratio is None) == (self.tiles is None):
            raise ValueError('set exactly one of ratio / tiles')
        value = self.ratio if self.tiles is None else self.tiles
        if value <= 0:
            raise ValueError(f'budget must be positive: {self}')

    @classmethod
    def sparsity(cls, percent: float) -> Budget:
        """Skip `percent` % of the key tiles (keep ratio 1 - percent / 100)."""
        if not 0.0 <= percent < 100.0:
            raise ValueError(f'sparsity must be in [0, 100): {percent}')
        return cls(ratio=1.0 - percent / 100.0)

    @property
    def keeps_all(self) -> bool:
        return self.ratio is not None and self.ratio >= 1.0

    def per_row(self, real_tokens: int, n_cols: int) -> float:
        if self.tiles is not None:
            return float(self.tiles)
        n_ideal = math.ceil(real_tokens / tiling.TILE_SIZE)
        return self.ratio * n_ideal * n_ideal / n_cols

    def describe(self) -> str:
        if self.tiles is not None:
            return f'{self.tiles:g} tiles'
        return f'{100.0 * (1.0 - min(self.ratio, 1.0)):g}% sparse'


def split_budget(budget: float, n_cols: int) -> tuple[int, int, float]:
    """(k_lo, k_hi, frac) with 1 <= k_lo <= k_hi <= n_cols."""
    if budget >= n_cols:
        return n_cols, n_cols, 0.0
    k_lo = min(max(1, math.floor(budget)), n_cols)
    # Rounded so that e.g. 2.3 - 2 gives 0.3 and not 0.2999999999999998,
    # which would drop one extra tile every 1 / frac rows.
    frac = round(min(max(budget - k_lo, 0.0), 1.0), 12)
    return k_lo, min(k_lo + 1, n_cols), frac


def _bresenham(n_rows: int, frac: float, device: str) -> torch.Tensor:
    # Build on the target device without a synchronizing host transfer.
    ramp = torch.floor(torch.arange(n_rows + 1, dtype=torch.float64, device=device) * frac)
    return (ramp[1:] > ramp[:-1]).to(device)


def bresenham_extra(n_rows: int, frac: float,
                    device: torch.device | str) -> torch.Tensor:
    """[n_rows] bool: row r keeps k_hi iff floor((r+1)f) > floor(r f).

    """
    return _bresenham(n_rows, float(frac), str(torch.device(device)))


@dataclasses.dataclass(frozen=True)
class ColumnBlock:
    """A sparse column block of the video quadrant."""

    start: int
    stop: int
    budget: Budget
    real_tokens: int


def column_blocks(layout: tiling.TileLayout, generated: Budget,
                  reference: Budget) -> list[ColumnBlock]:
    """The generated (target video) block, preceded by the reference block
    when reference spans are tiled."""
    target = ColumnBlock(layout.n_ref_tiles, layout.n_video_tiles, generated,
                         layout.target_tokens)
    if layout.n_ref_tiles == 0:
        return [target]
    return [ColumnBlock(0, layout.n_ref_tiles, reference, layout.ref_tokens),
            target]


@torch.no_grad()
def select(scores: torch.Tensor, layout: tiling.TileLayout,
           blocks: list[ColumnBlock]
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the video quadrant (rules 2-7).

    Args:
        scores: [H', n_video, n_video] block scores of every video query
            tile against every video key tile (any monotone score).
        layout: Tile layout.
        blocks: Column blocks from column_blocks().

    Returns:
        (index, keep): [H', n_video, K] int64 key tile ids and [H', n_video,
        K] bool, which entries are kept (blocks concatenated).
    """
    heads, n_video = scores.shape[0], layout.n_video_tiles
    device = scores.device
    rows = torch.arange(n_video, device=device)
    indices, keeps = [], []
    for block in blocks:
        n_cols = block.stop - block.start
        if block.budget.keeps_all:
            idx = torch.arange(block.start, block.stop, device=device)
            indices.append(idx.expand(heads, n_video, n_cols))
            keeps.append(layout.kv_ok[block.start:block.stop].expand(
                heads, n_video, n_cols))
            continue
        s = scores[:, :, block.start:block.stop].float().clone()
        s.masked_fill_(~layout.kv_ok[None, None, block.start:block.stop],
                       _NEG_INF)
        # A query tile always keeps its own key tile. Written as a gather /
        # where / scatter over every row instead of indexing the rows inside
        # the block: that indexing needs nonzero, whose data-dependent size
        # synchronizes the host once per call.
        own = (rows >= block.start) & (rows < block.stop)
        col = (rows - block.start).clamp(0, n_cols - 1)
        col = col[None, :, None].expand(heads, n_video, 1)
        s.scatter_(-1, col, torch.where(own[None, :, None], _POS_INF,
                                        s.gather(-1, col)))
        budget = block.budget.per_row(block.real_tokens, n_cols)
        k_lo, k_hi, frac = split_budget(budget, n_cols)
        vals, idx = torch.topk(s, k_hi, dim=-1, sorted=True)
        extra = bresenham_extra(n_video, frac, device)
        allowed = k_lo + extra.to(torch.long)
        keep = ((torch.arange(k_hi, device=device)[None, None, :]
                 < allowed[None, :, None]) & (vals > _NEG_INF))
        indices.append(idx + block.start)
        keeps.append(keep)
    return torch.cat(indices, -1), torch.cat(keeps, -1)


@torch.no_grad()
def block_mask(index: torch.Tensor, keep: torch.Tensor,
               layout: tiling.TileLayout) -> torch.Tensor:
    """Dense block mask of a selection (rules 1-7), the kernels' input.

    Args:
        index: [H', n_video, K] from select().
        keep: [H', n_video, K] from select().
        layout: Tile layout.

    Returns:
        [H', n_tiles, n_tiles] bool, rows are query tiles. Empty key tiles
        are never set.
    """
    heads = index.shape[0]
    n, n_video = layout.n_tiles, layout.n_video_tiles
    mask = torch.zeros(heads, n, n, dtype=torch.bool, device=index.device)
    mask[:, :n_video].scatter_(2, index, keep)
    mask[:, :, n_video:] = layout.kv_ok[n_video:]
    mask[:, n_video:, :] = layout.kv_ok
    return mask


def tile_index_list(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A block mask as the per-row list of kept key tiles.

    Kernels that walk only the kept tiles want the tile ids, not a dense
    mask. Rows keep different numbers of tiles (the budget is spread by
    Bresenham), so the list is padded to the widest row and paired with a
    count; entries past the count are never read.

    Args:
        mask: [H', n_tiles, n_tiles] bool from block_mask().

    Returns:
        (index [H', n_tiles, max_kept] int32, ascending tile ids;
         count [H', n_tiles] int32).
    """
    count = mask.sum(-1, dtype=torch.int32)
    widest = mask.shape[-1]  # global queries keep every key tile; avoid a device synchronization
    # A stable descending sort of the mask puts the kept tiles first and
    # leaves them in ascending tile order, which is what the kernels walk.
    order = torch.argsort(mask.to(torch.int8), dim=-1, descending=True,
                          stable=True)
    return order[..., :widest].to(torch.int32).contiguous(), count
