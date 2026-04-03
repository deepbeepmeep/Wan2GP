# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, Optional, TYPE_CHECKING

import torch
from einops import rearrange
from inference.common import DataProxyConfig, Modality, VarlenHandler
from inference.model.dit.dit_module import FFAHandler
from torch.nn import functional as F
from unfoldNd import UnfoldNd

if TYPE_CHECKING:
    from inference.pipeline.video_generate import EvalInput


def calc_local_qk_range(num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field):
    token_per_frame = num_video_tokens // num_frames
    total_tokens = num_video_tokens + num_audio_and_txt_tokens

    q_range_list = []
    k_range_list = []

    for i in range(num_frames):
        local_q_range = torch.tensor([i * token_per_frame, (i + 1) * token_per_frame])
        local_k_range = torch.tensor(
            [(i - frame_receptive_field) * token_per_frame, (i + frame_receptive_field + 1) * token_per_frame]
        )

        q_range_list.append(local_q_range)
        k_range_list.append(local_k_range)
    local_q_range = torch.stack(q_range_list, dim=0)
    local_k_range = torch.stack(k_range_list, dim=0)

    local_k_range[local_k_range < 0] = 0
    local_k_range[local_k_range > num_video_tokens] = num_video_tokens

    video_q_range = torch.tensor([[0, num_video_tokens]])
    video_k_range = torch.tensor([[num_video_tokens, num_video_tokens + num_audio_and_txt_tokens]])

    at_q_ranges = torch.tensor([[num_video_tokens, total_tokens]])
    at_k_ranges = torch.tensor([[0, total_tokens]])

    q_ranges = torch.cat([local_q_range, video_q_range, at_q_ranges], dim=0).to(torch.int32).to("cuda", non_blocking=True)
    k_ranges = torch.cat([local_k_range, video_k_range, at_k_ranges], dim=0).to(torch.int32).to("cuda", non_blocking=True)

    return (q_ranges, k_ranges)


def calc_local_attn_ffa_handler(num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field):
    q_ranges, k_ranges = calc_local_qk_range(num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field)
    max_seqlen_q = num_video_tokens + num_audio_and_txt_tokens
    max_seqlen_k = num_video_tokens + num_audio_and_txt_tokens
    attn_type_map = torch.zeros([q_ranges.shape[0]], device="cuda", dtype=torch.int32)
    softmax_scale = None

    ffa_handler = FFAHandler(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        attn_type_map=attn_type_map,
        softmax_scale=softmax_scale,
    )
    return ffa_handler


def get_coords(
    shape: list[int],
    ref_feat_shape: list[int],
    offset_thw: list[int] = [0, 0, 0],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """
    Generate feature-grid coordinates and corresponding original/reference size metadata.
    Args:
        feat_shape: [T, H, W] original feature-map shape
        ref_feat_shape: [T_ref, H_ref, W_ref] reference feature-map shape
        device: device for coordinate tensors
    Returns:
        coords: tensor shape (T*H*W, 9), containing (t, h, w, T, H, W, ref_T, ref_H, ref_W)
    """
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape

    # Generate index ranges
    offset_t, offset_h, offset_w = offset_thw
    time_rng = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    height_rng = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    width_rng = torch.arange(ori_w, device=device, dtype=dtype) + offset_w

    # Use meshgrid to generate a 3D grid (T, H, W)
    time_grid, height_grid, width_grid = torch.meshgrid(time_rng, height_rng, width_rng, indexing="ij")

    # Stack and flatten
    coords_grid = torch.stack([time_grid, height_grid, width_grid], dim=-1)
    coords_flat = coords_grid.reshape(-1, 3)

    # Build and expand size metadata
    meta = torch.tensor([ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], device=device, dtype=dtype)
    meta_expanded = meta.expand(coords_flat.size(0), -1)

    # Merge and return
    return torch.cat([coords_flat, meta_expanded], dim=-1)


@dataclass
class SingleData:
    video_x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: int
    txt_feat: torch.Tensor
    txt_feat_len: int
    t: int
    h: int
    w: int
    patch_size: int
    t_patch_size: int
    spatial_rope_interpolation: Literal["inter", "extra"]
    ref_audio_offset: int
    text_offset: int
    coords_style: Literal["v1", "v2"] = "v1"

    def __post_init__(self):
        self.video_token_num = self.video_x_t.shape[0]

        self.audio_x_t = self.audio_x_t[: self.audio_feat_len]
        self.txt_feat = self.txt_feat[: self.txt_feat_len]

        self.video_channel = self.video_x_t.shape[-1]
        self.audio_channel = self.audio_x_t.shape[-1]
        self.txt_channel = self.txt_feat.shape[-1]

    @property
    def device(self):
        return self.video_x_t.device

    @property
    def default_dtype(self):
        return self.video_x_t.dtype

    @property
    def total_token_num(self):
        return self.video_token_num + self.audio_feat_len + self.txt_feat_len

    @property
    def token_sequence(self):
        tensors_to_concat = [self.video_x_t, self.audio_x_t, self.txt_feat]
        max_channel = max(tensor.shape[-1] for tensor in tensors_to_concat)

        padded_tensors = [F.pad(t, (0, max_channel - t.shape[-1])) for t in tensors_to_concat]
        ret_val = torch.cat(padded_tensors, dim=0)
        return ret_val

    @property
    def modality_mapping(self):
        v_map = torch.full((self.video_token_num,), Modality.VIDEO, dtype=torch.int64, device=self.device)
        a_map = torch.full((self.audio_feat_len,), Modality.AUDIO, dtype=torch.int64, device=self.device)
        t_map = torch.full((self.txt_feat_len,), Modality.TEXT, dtype=torch.int64, device=self.device)

        modality_mapping = torch.cat([v_map, a_map, t_map], dim=0)
        return modality_mapping

    def default_coords(self, shape, ref_feat_shape, offset_thw=[0, 0, 0]):
        return get_coords(
            shape=shape, ref_feat_shape=ref_feat_shape, offset_thw=offset_thw, device=self.device, dtype=self.default_dtype
        )

    @property
    def coords_mapping(self):
        if self.spatial_rope_interpolation == "inter":
            video_ref_feat_shape = (self.t // self.t_patch_size, 32, 32)
        else:
            video_ref_feat_shape = (self.t // self.t_patch_size, self.h // self.patch_size, self.w // self.patch_size)

        video_coords = self.default_coords(
            shape=(self.t // self.t_patch_size, self.h // self.patch_size, self.w // self.patch_size),
            ref_feat_shape=video_ref_feat_shape,
        )

        if self.coords_style == "v1":
            audio_coords = self.default_coords(
                shape=(self.audio_feat_len, 1, 1), ref_feat_shape=(self.t // self.t_patch_size, 1, 1)
            )

            text_coords = self.default_coords(
                shape=(self.txt_feat_len, 1, 1), ref_feat_shape=(2, 1, 1), offset_thw=[self.text_offset, 0, 0]
            )

        elif self.coords_style == "v2":
            magic_audio_ref_t = (self.audio_feat_len - 1) // 4 + 1
            audio_coords = self.default_coords(
                shape=(self.audio_feat_len, 1, 1), ref_feat_shape=(magic_audio_ref_t // self.t_patch_size, 1, 1)
            )

            text_coords = self.default_coords(
                shape=(self.txt_feat_len, 1, 1), ref_feat_shape=(1, 1, 1), offset_thw=[-self.txt_feat_len, 0, 0]
            )

        coords_mapping = torch.cat([video_coords, audio_coords, text_coords], dim=0)
        return coords_mapping

    def depack_token_sequence(self, token_sequence):
        video_x_t = token_sequence[: self.video_token_num, : self.video_channel]
        video_x_t = rearrange(
            video_x_t,
            "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
            H=self.h // self.patch_size,
            W=self.w // self.patch_size,
            pT=self.t_patch_size,
            pH=self.patch_size,
            pW=self.patch_size,
        ).contiguous()

        audio_x_t = token_sequence[self.video_token_num : self.video_token_num + self.audio_feat_len, : self.audio_channel]
        return video_x_t, audio_x_t


@dataclass
class SimplePackedData:
    items: list[SingleData]

    @property
    def token_sequence(self):
        return torch.cat([item.token_sequence for item in self.items], dim=0)

    @property
    def modality_mapping(self):
        return torch.cat([item.modality_mapping for item in self.items], dim=0)

    @property
    def coords_mapping(self):
        return torch.cat([item.coords_mapping for item in self.items], dim=0)

    @property
    def total_token_num(self):
        return sum([item.total_token_num for item in self.items])

    def __getitem__(self, index):
        return self.items[index]

    @property
    def cu_seqlen(self):
        cu_seqlen = torch.cumsum(torch.tensor([item.total_token_num for item in self.items]), dim=0)
        cu_seqlen = torch.nn.functional.pad(cu_seqlen, (1, 0))
        return cu_seqlen

    @property
    def max_seqlen(self):
        return torch.tensor(max([item.total_token_num for item in self.items]))

    def depack_token_sequence(self, token_sequence):
        video_x_t_list = []
        audio_x_t_list = []

        token_sequence_list = torch.split(token_sequence, [item.total_token_num for item in self.items], dim=0)
        for item, token_sequence in zip(self.items, token_sequence_list):
            video_x_t, audio_x_t = item.depack_token_sequence(token_sequence)
            video_x_t_list.append(video_x_t)
            audio_x_t_list.append(audio_x_t)
        return torch.stack(video_x_t_list, dim=0), torch.stack(audio_x_t_list, dim=0)


class MagiDataProxy:
    def __init__(self, config: DataProxyConfig):
        self.patch_size = config.patch_size
        self.t_patch_size = config.t_patch_size
        self.frame_receptive_field = config.frame_receptive_field
        self.spatial_rope_interpolation = 'extra'
        self.ref_audio_offset = config.ref_audio_offset
        self.text_offset = config.text_offset
        self.unfold = UnfoldNd(
            kernel_size=(self.t_patch_size, self.patch_size, self.patch_size),
            stride=(self.t_patch_size, self.patch_size, self.patch_size),
        )
        self.coords_style = config.coords_style

        self._saved_data: dict[str, Any] = {}

    def saved_for_output(self, **kwargs):
        """
        Store intermediate data used by process_output.
        Supports keyword-argument style calls: saved_for_output(a=1, b=2)
        Can be called multiple times to accumulate data

        Args:
            **kwargs: key-value pairs to store
        """
        # Directly update dict; supports accumulation across calls
        self._saved_data.update(kwargs)

    def get_saved_data(self, key: str):
        """
        Get stored data
        """
        return self._saved_data[key]

    def img2tokens(self, x_t: torch.Tensor):
        x_t_unfolded = self.unfold(x_t)
        # Transpose dimensions from (N, col_dim, num_tokens) -> (N, num_tokens, col_dim)
        x_t = rearrange(x_t_unfolded, "N col_dim num_tokens -> N num_tokens col_dim").contiguous()
        return x_t

    def process_input(self, transported_data: "EvalInput"):
        # init img2col module

        batch_size, _, t, h, w = transported_data.x_t.shape
        # 1. Process video features while keeping the batch dimension
        x_t = self.img2tokens(transported_data.x_t)

        # 2. Process audio features while keeping the batch dimension
        # Assume transported_data.audio_x_t shape is already (N, num_tokens, col_dim)
        audio_x_t = transported_data.audio_x_t.contiguous()

        # Here we assume text_in shape is (N, num_tokens, col_dim)
        text_in = transported_data.txt_feat.contiguous()

        simple_packed_data = SimplePackedData(items=[])
        for i in range(batch_size):
            single_data = SingleData(
                video_x_t=x_t[i],
                audio_x_t=audio_x_t[i],
                audio_feat_len=transported_data.audio_feat_len[i],
                txt_feat=text_in[i],
                txt_feat_len=transported_data.txt_feat_len[i],
                t=t,
                h=h,
                w=w,
                patch_size=self.patch_size,
                t_patch_size=self.t_patch_size,
                spatial_rope_interpolation=self.spatial_rope_interpolation,
                ref_audio_offset=self.ref_audio_offset,
                text_offset=self.text_offset,
                coords_style=self.coords_style,
            )
            simple_packed_data.items.append(single_data)

        if self.frame_receptive_field != -1:
            assert batch_size == 1, "local attention only supports batch size 1"

            local_attn_handler = calc_local_attn_ffa_handler(
                num_video_tokens=simple_packed_data[0].video_token_num,
                num_audio_and_txt_tokens=simple_packed_data[0].audio_feat_len + simple_packed_data[0].txt_feat_len,
                num_frames=t,
                frame_receptive_field=self.frame_receptive_field,
            )
            if isinstance(local_attn_handler.max_seqlen_k, torch.Tensor):
                local_attn_handler.max_seqlen_k = local_attn_handler.max_seqlen_k.item()
            if isinstance(local_attn_handler.max_seqlen_q, torch.Tensor):
                local_attn_handler.max_seqlen_q = local_attn_handler.max_seqlen_q.item()
        else:
            local_attn_handler = None

        varlen_handler = VarlenHandler(
            cu_seqlens_q=simple_packed_data.cu_seqlen.to(torch.int32).cuda(),
            cu_seqlens_k=simple_packed_data.cu_seqlen.to(torch.int32).cuda(),
            max_seqlen_q=simple_packed_data.max_seqlen.to(torch.int32).cuda(),
            max_seqlen_k=simple_packed_data.max_seqlen.to(torch.int32).cuda(),
        )

        self.saved_for_output(simple_packed_data=simple_packed_data)

        x = simple_packed_data.token_sequence
        coords_mapping = simple_packed_data.coords_mapping
        modality_mapping = simple_packed_data.modality_mapping

        return (x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler)

    def process_output(self, x: torch.Tensor):
        # Inserting operations in between may corrupt parallel-runtime data and cause latent errors

        simple_packed_data: SimplePackedData = self.get_saved_data("simple_packed_data")
        x_video, x_audio = simple_packed_data.depack_token_sequence(x)

        return (x_video, x_audio)
