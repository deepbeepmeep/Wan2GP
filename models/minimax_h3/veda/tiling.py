# SPDX-License-Identifier: MIT
# Adapted from veda-sparse/Veda-on-ComfyUI; see NOTICE and LICENSE.
"""Tile permutations: every 128 permuted rows form one 3D tile of a span.

The rules below are the contract the released predictors and tile plans
were trained against (ported from Miowtion, `miowtion/veda/tiling.py`, MIT),
so they must not drift:

  * A tile shape (t, h, w) with t * h * w = 128 cuts a span's (T, H, W)
    token grid into boxes. Box order is (h-block, w-block, t-block), outer to
    inner; rows inside a box are t, h, w row-major, then the real rows are
    stably compacted to the front. A tile's real rows are therefore always a
    prefix of length `valid_count[tile]`, and kernels / pooling rely on that.
  * Every other row (text, audio, untiled conditions) is "global": global
    rows are chunked into 128-row tiles in sequence order, after the tiles
    of all spans.

ComfyUI packs the H3 sequence without padding rows, so unlike Miowtion there
is no `used < seq_len` tail here.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence

import torch

TILE_SIZE = 128


@dataclasses.dataclass(frozen=True, order=True)
class TileShape:
    """A (t, h, w) box of TILE_SIZE tokens."""

    t: int
    h: int
    w: int

    def __post_init__(self):
        if self.t * self.h * self.w != TILE_SIZE:
            raise ValueError(f'tile {self} does not hold {TILE_SIZE} tokens')

    def __str__(self) -> str:
        return f'{self.t}x{self.h}x{self.w}'

    @classmethod
    def parse(cls, text: str) -> TileShape:
        t, h, w = (int(v) for v in text.split('x'))
        return cls(t, h, w)

    def transposed(self) -> TileShape:
        """Swaps the h and w extents."""
        return TileShape(self.t, self.w, self.h)

    def padded_grid(self, grid: Sequence[int]) -> tuple[int, int, int]:
        return tuple(-(-g // s) * s
                     for g, s in zip(grid, (self.t, self.h, self.w)))

    def num_tiles(self, grid: Sequence[int]) -> int:
        tp, hp, wp = self.padded_grid(grid)
        return tp * hp * wp // TILE_SIZE

    def aspect_spread(self) -> float:
        """max / min extent; 1.0 for the most cubic shape."""
        return max(self.t, self.h, self.w) / min(self.t, self.h, self.w)


def all_shapes() -> list[TileShape]:
    """All power-of-two (t, h, w) triples with product TILE_SIZE."""
    exponent = TILE_SIZE.bit_length() - 1
    shapes = []
    for i, j in itertools.product(range(exponent + 1), repeat=2):
        if i + j <= exponent:
            shapes.append(TileShape(2**i, 2**j, 2**(exponent - i - j)))
    return sorted(shapes)


def least_padding_shape(grid: Sequence[int]) -> TileShape:
    """Shape with the least padding on `grid` (Miowtion's rule for tiled
    conditions); ties go to the most cubic, then the lexicographic first."""
    fitting = [s for s in all_shapes()
               if s.t <= grid[0] and s.h <= grid[1] and s.w <= grid[2]]
    return min(fitting or all_shapes(),
               key=lambda s: (s.num_tiles(grid), s.aspect_spread(),
                              (s.t, s.h, s.w)))


@dataclasses.dataclass(frozen=True)
class TiledSpan:
    """Packed rows [start, start + T*H*W), T/H/W row-major, tiled by shape."""

    start: int
    grid: tuple[int, int, int]
    shape: TileShape

    @property
    def num_rows(self) -> int:
        t, h, w = self.grid
        return t * h * w


def span_tiles(span: TiledSpan) -> torch.Tensor:
    """[n_tiles, 128] int64 packed row ids of one span, -1 on padding."""
    t, h, w = span.grid
    s = span.shape
    tp, hp, wp = s.padded_grid(span.grid)
    grid = torch.full((tp, hp, wp), -1, dtype=torch.long, device="cpu")
    grid[:t, :h, :w] = span.start + torch.arange(t * h * w, device="cpu").view(t, h, w)
    tiles = grid.view(tp // s.t, s.t, hp // s.h, s.h, wp // s.w, s.w)
    tiles = tiles.permute(2, 4, 0, 1, 3, 5).reshape(-1, TILE_SIZE)
    order = torch.argsort((tiles < 0).to(torch.int8), dim=1, stable=True)
    return torch.gather(tiles, 1, order)


@dataclasses.dataclass
class TileLayout:
    """One permutation of the packed sequence and its derived constants.

    Built once per (geometry, tile shape, device) and reused on every call:
    recomputing these (nonzero, sums) would synchronize the device on the
    hot path.

    Attributes:
        perm: [N] int64 packed row of every permuted slot, -1 on padding;
            N = n_tiles * 128.
        valid_count: [n_tiles] int32 real rows per tile (a prefix).
        n_video_tiles: Tiles of all tiled spans (reference spans, then the
            target span).
        n_ref_tiles: Leading tiles that belong to reference spans; tiles
            [n_ref_tiles, n_video_tiles) belong to the target.
        n_global_tiles: Trailing tiles of global rows.
        ref_tokens: Real rows of the reference spans.
        target_tokens: Real rows of the target span.
        seq_len: Packed length S; `scatter_index` sends padding to row S.
        gather_index: [N] perm with padding redirected to row 0.
        scatter_index: [N] perm with padding redirected to row S.
        pad_slots: [P] slots of perm that are padding.
        partial_tiles: [Q] tiles with 0 < valid_count < 128.
        partial_video_tiles: The partial tiles among the video tiles.
        kv_ok: [n_tiles] bool, the tile has at least one real row.
        full_tile: [n_tiles] bool, valid_count == 128.
        slot_valid: [N] int32, 1 where the slot holds a real row.
    """

    perm: torch.Tensor
    valid_count: torch.Tensor
    n_video_tiles: int
    n_ref_tiles: int
    n_global_tiles: int
    ref_tokens: int
    target_tokens: int
    seq_len: int
    gather_index: torch.Tensor
    scatter_index: torch.Tensor
    pad_slots: torch.Tensor
    partial_tiles: torch.Tensor
    partial_video_tiles: torch.Tensor
    kv_ok: torch.Tensor
    full_tile: torch.Tensor
    slot_valid: torch.Tensor

    @property
    def n_tiles(self) -> int:
        return self.n_video_tiles + self.n_global_tiles

    @property
    def num_slots(self) -> int:
        return self.n_tiles * TILE_SIZE


def build_tile_layout(spans: Sequence[TiledSpan], seq_len: int,
                      device: torch.device | str = 'cpu') -> TileLayout:
    """Builds the permutation; rows outside every span are global.

    Args:
        spans: Tiled spans in packed order; the last one is the target.
        seq_len: Packed sequence length S.
        device: Device of the derived tensors.

    Returns:
        The tile layout.

    Raises:
        ValueError: If spans overlap, are out of order or exceed seq_len.
    """
    if not spans:
        raise ValueError('at least the target span must be tiled')
    covered = torch.zeros(seq_len, dtype=torch.bool, device="cpu")
    tiles = []
    prev_stop = 0
    for span in spans:
        if span.start < prev_stop or span.start + span.num_rows > seq_len:
            raise ValueError(f'span {span} overlaps or exceeds {seq_len}')
        prev_stop = span.start + span.num_rows
        covered[span.start:prev_stop] = True
        tiles.append(span_tiles(span))
    n_ref_tiles = sum(t.shape[0] for t in tiles[:-1])
    video = torch.cat(tiles)
    global_rows = torch.nonzero(~covered).view(-1)
    n_global = -(-global_rows.numel() // TILE_SIZE)
    global_tiles = torch.full((n_global * TILE_SIZE,), -1, dtype=torch.long, device="cpu")
    global_tiles[:global_rows.numel()] = global_rows
    perm = torch.cat([video.view(-1), global_tiles])
    valid_count = (perm.view(-1, TILE_SIZE) >= 0).sum(1).to(torch.int32)
    partial = torch.nonzero((valid_count > 0)
                            & (valid_count < TILE_SIZE)).view(-1)
    return TileLayout(
        perm=perm.to(device),
        valid_count=valid_count.to(device),
        n_video_tiles=video.shape[0],
        n_ref_tiles=n_ref_tiles,
        n_global_tiles=n_global,
        ref_tokens=sum(s.num_rows for s in spans[:-1]),
        target_tokens=spans[-1].num_rows,
        seq_len=seq_len,
        gather_index=perm.clamp(min=0).to(device),
        scatter_index=torch.where(perm < 0, seq_len, perm).to(device),
        pad_slots=torch.nonzero(perm < 0).view(-1).to(device),
        partial_tiles=partial.to(device),
        partial_video_tiles=partial[partial < video.shape[0]].to(device),
        kv_ok=(valid_count > 0).to(device),
        full_tile=(valid_count == TILE_SIZE).to(device),
        slot_valid=(perm >= 0).to(torch.int32).to(device),
    )


def gather_tiles(x: torch.Tensor, layout: TileLayout,
                 heads: torch.Tensor) -> torch.Tensor:
    """Permutes rows of x into tile order, zeroing padding slots.

    Args:
        x: [S, H, D] packed tensor (any strides).
        layout: Tile layout of this head group.
        heads: [H'] int64 head indices on x's device.

    Returns:
        [N, H', D] contiguous, tile order (the kernel's seq-major layout).
    """
    out = x[layout.gather_index[:, None], heads[None, :]]
    if layout.pad_slots.numel():
        out.index_fill_(0, layout.pad_slots, 0)
    return out


def scatter_tiles_(out: torch.Tensor, tiled: torch.Tensor,
                   layout: TileLayout, heads: torch.Tensor) -> None:
    """Writes tile-ordered rows back into out[S + 1, H, D] in place.

    Padding slots land in the spare row out[S], so the inverse permutation
    is a single index write without masking.
    """
    out[layout.scatter_index[:, None], heads[None, :]] = tiled
