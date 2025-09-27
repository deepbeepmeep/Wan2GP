from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from .attn_mask import RadialAttention, MaskMap

def fill_radial_cache(radial_cache, nb_layers, lat_t, lat_h, lat_w):
    MaskMap._log_mask = None

    for i in range(nb_layers):
        radial_cache[i] =  WanSparseAttnProcessor2_0(i, lat_t, lat_h, lat_w)

class WanSparseAttnProcessor2_0: 
    mask_map = None
    dense_timestep = 0
    dense_block = 0
    decay_factor = 1.0
    sparse_type = "radial"  # default to radial attention, can be changed to "dense" for dense attention
    use_sage_attention = True
    
    def __init__(self, layer_idx,  lat_t, lat_h, lat_w):
        self.layer_idx = layer_idx
        self.mask_map = MaskMap(video_token_num=lat_t * lat_h * lat_w // 4 , num_frame=lat_t)        
    def __call__(
        self,
        qkv_list,
        timestep_no = 0,
    ) -> torch.Tensor:
        query, key, value = qkv_list

        batch_size = query.shape[0]
        # transform (batch_size, seq_len, num_heads, head_dim) to (seq_len * batch_size, num_heads, head_dim)
        query = rearrange(query, "b s h d -> (b s) h d")
        key = rearrange(key, "b s h d -> (b s) h d")
        value = rearrange(value, "b s h d -> (b s) h d")
        if timestep_no < self.dense_timestep or self.layer_idx < self.dense_block or self.sparse_type == "dense":
            hidden_states = RadialAttention(
                query=query, key=key, value=value, mask_map=self.mask_map, sparsity_type="dense", block_size=128, decay_factor=self.decay_factor, model_type="wan", pre_defined_mask=None, use_sage_attention=self.use_sage_attention
            )
        else:
            # apply radial attention
            hidden_states = RadialAttention(
                query=query, key=key, value=value, mask_map=self.mask_map, sparsity_type="radial", block_size=128, decay_factor=self.decay_factor, model_type="wan", pre_defined_mask=None, use_sage_attention=self.use_sage_attention
            )
        # transform back to (batch_size, num_heads, seq_len, head_dim)
        hidden_states = rearrange(hidden_states, "(b s) h d -> b s h d", b=batch_size)

        return hidden_states
