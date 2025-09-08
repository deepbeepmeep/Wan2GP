"""Mask Mod for Image2Video"""

from math import floor
import torch
from torch import Tensor


from functools import lru_cache
from typing import Optional, List

import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
)


# @lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask

def generate_temporal_head_mask_mod(context_length: int = 226, prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, mul: int = 2):
    
    def round_to_multiple(idx):
        return floor(idx / 128) * 128
        
    real_length = num_frames * token_per_frame + prompt_length
    def temporal_mask_mod(b, h, q_idx, kv_idx):
        real_mask = (kv_idx < real_length) & (q_idx < real_length)
        fake_mask = (kv_idx >= real_length) & (q_idx >= real_length)
        
        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = (torch.abs(q_idx - kv_idx) < two_frame)

        text_column_mask = (num_frames * token_per_frame <= kv_idx) & (kv_idx < real_length)
        text_row_mask = (num_frames * token_per_frame <= q_idx) & (q_idx < real_length)

        video_mask = temporal_head_mask | text_column_mask | text_row_mask
        real_mask = real_mask & video_mask
        
        return real_mask | fake_mask
    
    return temporal_mask_mod
