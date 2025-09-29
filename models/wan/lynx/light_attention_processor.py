# Copyright (c) 2025 The Wan Team and The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on September 15, 2025.
#
# Original file was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/diffusers/blob/v0.30.3/LICENSE and https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.normalization import RMSNorm


def register_ip_adapter_wan(
    model,
    hidden_size=5120,
    cross_attention_dim=2048,
    dtype=torch.float32,
    init_method="zero",
    layers=None
):
    attn_procs = {}
    transformer_sd = model.state_dict()

    if layers is None:
        layers = list(range(0, len(model.blocks)))
    elif isinstance(layers, int): # Only interval provided
        layers = list(range(0, len(model.blocks), layers))

    for i, block in enumerate(model.blocks):
        if i not in layers:
            continue

        name = f"blocks.{i}.attn2.processor"
        attn_procs[name] = IPAWanAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim
        )

        if init_method == "zero":
            torch.nn.init.zeros_(attn_procs[name].to_k_ip.weight)
            torch.nn.init.zeros_(attn_procs[name].to_v_ip.weight)
        elif init_method == "clone":
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": transformer_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": transformer_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name].load_state_dict(weights)
        else:
            raise ValueError(f"{init_method} is not supported.")

        block.attn2.processor = attn_procs[name]

    ip_layers = torch.nn.ModuleList(attn_procs.values())
    ip_layers.to(model.device, dtype=dtype)

    return model, ip_layers


class IPAWanAttnProcessor2_0(torch.nn.Module):
    def __init__(
        self, 
        hidden_size, 
        cross_attention_dim=None, 
        scale=1.0, 
        bias=False,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("IPAWanAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=bias)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=bias)

        torch.nn.init.zeros_(self.to_k_ip.weight)
        torch.nn.init.zeros_(self.to_v_ip.weight)
        if bias:
            torch.nn.init.zeros_(self.to_k_ip.bias)
            torch.nn.init.zeros_(self.to_v_ip.bias)

        self.norm_rms_k = RMSNorm(hidden_size, eps=1e-5, elementwise_affine=False)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        image_embed: torch.Tensor = None,
        ip_scale: float = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # =============================================================
        batch_size = image_embed.size(0)

        ip_hidden_states = image_embed
        ip_query = query # attn.to_q(hidden_states.clone())
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        if attn.norm_q is not None:
            ip_query = attn.norm_q(ip_query)
        ip_key = self.norm_rms_k(ip_key)

        ip_inner_dim = ip_key.shape[-1]
        ip_head_dim = ip_inner_dim // attn.heads

        ip_query = ip_query.view(batch_size, -1, attn.heads, ip_head_dim).transpose(1, 2)
        ip_key = ip_key.view(batch_size, -1, attn.heads, ip_head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, ip_head_dim).transpose(1, 2)

        ip_hidden_states = F.scaled_dot_product_attention(
            ip_query, ip_key, ip_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * ip_head_dim)
        ip_hidden_states = ip_hidden_states.to(ip_query.dtype)
        # ===========================================================================

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb_inner(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb_inner(query, rotary_emb)
            key = apply_rotary_emb_inner(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # Add IPA residual
        ip_scale = ip_scale or self.scale
        hidden_states = hidden_states + ip_scale * ip_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states