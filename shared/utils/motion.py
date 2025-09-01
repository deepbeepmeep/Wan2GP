# Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
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

import copy
import json
import os, io
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def get_tracks_inference(tracks, height, width, quant_multi: Optional[int] = 8, **kwargs):
    if isinstance(tracks, str):
        tracks = torch.load(tracks)

    tracks_np = unzip_to_array(tracks)

    tracks = process_tracks(
        tracks_np, (width, height), quant_multi=quant_multi, **kwargs
    )

    return tracks


def unzip_to_array(
    data: bytes, key: Union[str, List[str]] = "array"
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    bytes_io = io.BytesIO(data)

    if isinstance(key, str):
        # Load the NPZ data from the BytesIO object
        with np.load(bytes_io) as data:
            return data[key]
    else:
        get = {}
        with np.load(bytes_io) as data:
            for k in key:
                get[k] = data[k]
        return get


def process_tracks(tracks_np: np.ndarray, frame_size: Tuple[int, int], quant_multi: int = 8, **kwargs):
    # tracks: shape [t, h, w, 3] => samples align with 24 fps, model trained with 16 fps.
    # frame_size: tuple (W, H)

    tracks = torch.from_numpy(tracks_np).float() / quant_multi
    if tracks.shape[1] == 121:
        tracks = torch.permute(tracks, (1, 0, 2, 3))
    tracks, visibles = tracks[..., :2], tracks[..., 2:3]
    short_edge = min(*frame_size)

    tracks = tracks - torch.tensor([*frame_size]).type_as(tracks) / 2
    tracks = tracks / short_edge * 2

    visibles = visibles * 2 - 1

    trange = torch.linspace(-1, 1, tracks.shape[0]).view(-1, 1, 1, 1).expand(*visibles.shape)
    
    out_ = torch.cat([trange, tracks, visibles], dim=-1).view(121, -1, 4)
    out_0 = out_[:1]
    out_l = out_[1:] # 121 => 120 | 1
    out_l = torch.repeat_interleave(out_l, 2, dim=0)[1::3]  # 120 => 240 => 80
    return torch.cat([out_0, out_l], dim=0)
