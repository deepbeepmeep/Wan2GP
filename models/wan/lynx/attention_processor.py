# Copyright (c) 2025 The Wan Team and The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on September 15, 2025.
#
# Original file was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/diffusers/blob/v0.30.3/LICENSE and https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.


from typing import Optional, List
import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm

def setup_lynx_attention_layers(blocks, lynx_full, dim):
    if lynx_full:
        lynx_cross_dim = 5120
        lynx_layers = len(blocks)
    else:
        lynx_cross_dim = 2048 
        lynx_layers = 20
    for i, block in enumerate(blocks):
        if i < lynx_layers:
            block.cross_attn.to_k_ip = nn.Linear(lynx_cross_dim, dim , bias=lynx_full)
            block.cross_attn.to_v_ip = nn.Linear(lynx_cross_dim, dim , bias=lynx_full)
        else:
            block.cross_attn.to_k_ip = None
            block.cross_attn.to_v_ip = None
        if lynx_full:
            block.cross_attn.registers = nn.Parameter(torch.randn(1, 16, lynx_cross_dim) / dim**0.5)
            block.cross_attn.norm_rms_k = None
            block.self_attn.to_k_ref = nn.Linear(dim, dim, bias=True)
            block.self_attn.to_v_ref = nn.Linear(dim, dim, bias=True)            
        else:
            block.cross_attn.registers = None
            block.cross_attn.norm_rms_k = RMSNorm(dim, eps=1e-5, elementwise_affine=False)



