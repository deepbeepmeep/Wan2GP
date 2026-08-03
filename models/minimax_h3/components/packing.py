# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 native packed-token layouts for WanGP single-GPU inference.

The geometry follows MiniMax's raw-checkpoint SGLang implementation. WanGP omits
the isolated alignment tail because it cannot attend to live rows and would only
increase attention work.
"""

from dataclasses import dataclass

import numpy as np
import torch


MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999

_INTERP = 32
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0


@dataclass
class MiniMaxH3PackedSequence:
    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int


@dataclass
class MiniMaxH3PreparedReference:
    kind: str
    has_audio: bool = False
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0

    @property
    def num_video_rows(self):
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self):
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS


def patchify_video_latents(latent, patch_size):
    patch_t, patch_h, patch_w = patch_size
    batch, channels, full_t, full_h, full_w = latent.shape
    t, h, w = full_t // patch_t, full_h // patch_h, full_w // patch_w
    packed = latent.reshape(batch, channels, t, patch_t, h, patch_h, w, patch_w)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(batch * t * h * w, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(rows, num_latent_frames, latent_height, latent_width, channels, patch_size):
    patch_t, patch_h, patch_w = patch_size
    t, h, w = num_latent_frames // patch_t, latent_height // patch_h, latent_width // patch_w
    packed = rows.reshape(-1, t, h, w, channels, patch_t, patch_h, patch_w)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def unpack_audio_tokens(rows, num_audio_latents):
    return rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, rows.shape[-1]).permute(0, 2, 1).contiguous()


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return torch.from_numpy(np.linspace(left, left + ratio, dim // patch, endpoint=False) * _INTERP).to(torch.float64)


def _video_t_grid(length, origin):
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[index % len(_FRAME_PER_TOKEN)] for index in range(length)],
        dtype=torch.float64,
    )
    return origin + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


def _keyframe_t_span(length):
    spans = np.ones(length, dtype=np.float64) * _FRAME_RESCALE
    for index, frames in enumerate(_FRAME_PER_TOKEN):
        spans[index::len(_FRAME_PER_TOKEN)] *= frames
    return float(spans.sum())


def _reference_t_span(length):
    return sum(_FRAME_RESCALE * _FRAME_PER_TOKEN[index % len(_FRAME_PER_TOKEN)] for index in range(length))


def _frame_grid(latent_height, latent_width, patch_h, patch_w):
    sqrt_area = np.sqrt(latent_height * latent_width)
    height = _axis_from_sqrt_area(latent_height, patch_h, sqrt_area)
    width = _axis_from_sqrt_area(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height, width, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width


def _fill_audio_positions(position_ids, rows, length, origin, width_grid):
    time = origin + torch.arange(length, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        (torch.full((length,), float(width_grid[0]), dtype=torch.float64),
         torch.full((length,), float(width_grid[-1]), dtype=torch.float64))
    )


def build_packed_sequence(text_token_tags, num_latent_frames, latent_height, latent_width, num_audio_latents,
                          patch_size, keyframe_anchors=()):
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    text_len = int(text_token_tags.shape[0])
    condition_rows = len(keyframe_anchors) * rows_per_frame
    audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    video_rows = num_latent_frames * rows_per_frame
    sequence_length = text_len + condition_rows + audio_rows + video_rows
    condition_start = text_len
    audio_start = condition_start + condition_rows
    video_start = audio_start + audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    frame_grid, width_grid = _frame_grid(latent_height, latent_width, patch_h, patch_w)
    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(text_len)
        elif anchor == "last":
            anchor_time = float(text_len) + _keyframe_t_span(num_latent_frames) - _FRAME_RESCALE
        else:
            raise ValueError(f"Unknown MiniMax H3 keyframe anchor {anchor!r}")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, float(text_len), width_grid)
    target = position_ids[video_start:].view(num_latent_frames, rows_per_frame, 3)
    target[:, :, 0] = _video_t_grid(num_latent_frames, float(text_len))[:, None]
    target[:, :, 1:] = frame_grid[None]

    text_indices = torch.arange(text_len)
    audio_indices = torch.arange(audio_start, video_start)
    video_indices = torch.cat((torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)))
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG
    return MiniMaxH3PackedSequence(sequence_length, position_ids, token_tags, video_indices, audio_indices,
                                   text_indices, condition_rows, 0)


def build_ref2va_packed_sequence(text_token_tags, references, num_latent_frames, latent_height, latent_width,
                                 num_audio_latents, patch_size):
    _, patch_h, patch_w = patch_size
    text_len = int(text_token_tags.shape[0])
    target_frame_grid, target_width_grid = _frame_grid(latent_height, latent_width, patch_h, patch_w)
    target_video_rows = num_latent_frames * target_frame_grid.shape[0]
    target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    condition_video_rows = sum(ref.num_video_rows for ref in references if ref.kind != "audio")
    condition_audio_rows = sum(ref.num_audio_rows for ref in references)
    sequence_length = text_len + condition_video_rows + condition_audio_rows + target_audio_rows + target_video_rows
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)

    video_indices, audio_indices = [], []
    cursor, time_cursor = text_len, float(text_len)
    for ref in references:
        if ref.kind == "image":
            rows = slice(cursor, cursor + ref.num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_grid(ref.latent_height, ref.latent_width, patch_h, patch_w)
            position_ids[rows, 0] = time_cursor
            position_ids[rows, 1:] = frame_grid
            time_cursor += 1.0
        elif ref.kind == "audio":
            rows = slice(cursor, cursor + ref.num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, ref.num_audio_latents, time_cursor, target_width_grid)
            time_cursor += float(ref.num_audio_latents)
        elif ref.kind == "video":
            audio_rows = slice(cursor, cursor + ref.num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + ref.num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))
            frame_grid, width_grid = _frame_grid(ref.latent_height, ref.latent_width, patch_h, patch_w)
            _fill_audio_positions(position_ids, audio_rows, ref.num_audio_latents, time_cursor, width_grid)
            video_grid = position_ids[video_rows].view(ref.num_latent_frames, frame_grid.shape[0], 3)
            video_grid[:, :, 0] = _video_t_grid(ref.num_latent_frames, time_cursor)[:, None]
            video_grid[:, :, 1:] = frame_grid[None]
            time_cursor += max(float(ref.num_audio_latents), _reference_t_span(ref.num_latent_frames))
        else:
            raise ValueError(f"Unknown MiniMax H3 reference kind {ref.kind!r}")

    audio_start = cursor
    video_start = audio_start + target_audio_rows
    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, time_cursor,
                          target_width_grid)
    target = position_ids[video_start:].view(num_latent_frames, target_frame_grid.shape[0], 3)
    target[:, :, 0] = _video_t_grid(num_latent_frames, time_cursor)[:, None]
    target[:, :, 1:] = target_frame_grid[None]

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(text_len)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG
    return MiniMaxH3PackedSequence(sequence_length, position_ids, token_tags, video_indices, audio_indices,
                                   text_indices, condition_video_rows, condition_audio_rows)


def build_row_timesteps(layout, video_timestep, audio_timestep, condition_video_timestep,
                        condition_audio_timestep):
    timesteps = torch.full((layout.sequence_length,), video_timestep, dtype=torch.float32,
                           device=layout.token_tags.device)
    timesteps[layout.video_indices[:layout.num_condition_video_rows]] = condition_video_timestep
    timesteps[layout.audio_indices[layout.num_condition_audio_rows:]] = audio_timestep
    timesteps[layout.audio_indices[:layout.num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(timesteps, sorted=True, return_inverse=True)


__all__ = ["MINIMAX_H3_AUDIO_TAG", "MINIMAX_H3_KEYFRAME_NOISE_AUG", "MINIMAX_H3_TEXT_TAG",
           "MINIMAX_H3_VIDEO_TAG", "MiniMaxH3PackedSequence", "MiniMaxH3PreparedReference",
           "build_packed_sequence", "build_ref2va_packed_sequence", "build_row_timesteps",
           "patchify_video_latents", "unpack_audio_tokens", "unpatchify_video_tokens"]
