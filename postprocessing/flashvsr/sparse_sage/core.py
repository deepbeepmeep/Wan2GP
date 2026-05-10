"""
https://github.com/jt-zhang/Sparse_SageAttention_API

Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .quant_per_block import per_block_int8
from .sparse_int8_attn import forward as sparse_sageattn_fwd
import torch


def sparse_sageattn(qkv_list, mask_id = None, is_causal=False, tensor_layout="HND"):
    q, k, v = qkv_list
    qkv_list.clear()
    if mask_id is None:
        mask_id = torch.ones((q.shape[0], q.shape[1], (q.shape[2] + 128 - 1)//128, (q.shape[3] + 64 - 1)//64), dtype=torch.int8, device=q.device) # TODO

    output_dtype = q.dtype
    if output_dtype == torch.bfloat16 or output_dtype == torch.float32:
        v = v.to(torch.float16)
    
    seq_dim = 1 if tensor_layout == "NHD" else 2
    km = k.mean(dim=seq_dim, keepdim=True)
    # km = torch.zeros((k.size(0), k.size(1), 1, k.size(3)), dtype=torch.float16, device=k.device)  # Placeholder for mean, not used in quantization

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, tensor_layout=tensor_layout)
    del q, k, km
    
    o = sparse_sageattn_fwd(
        q_int8, k_int8, mask_id, v, q_scale, k_scale, 
        is_causal=is_causal, tensor_layout=tensor_layout, output_dtype=output_dtype
    )
    del q_int8, k_int8, q_scale, k_scale, v
    return o

    
# flops = 4 * q.size(0) * q.size(1) * q.size(2)**2 * q.size(3)  / (2 if is_causal else 1)
