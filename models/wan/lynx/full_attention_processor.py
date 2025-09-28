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
from diffusers.models.attention_processor import Attention
from modules.common.flash_attention import flash_attention
from modules.common.navit_utils import vector_to_list, list_to_vector, merge_token_lists

def new_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **cross_attention_kwargs,
) -> torch.Tensor:
    return self.processor(
        self,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
Attention.forward = new_forward

class WanIPAttnProcessor(nn.Module):
    def __init__(
        self,
        cross_attention_dim: int,
        dim: int,
        n_registers: int,
        bias: bool,
    ):
        super().__init__()
        self.to_k_ip = nn.Linear(cross_attention_dim, dim, bias=bias)
        self.to_v_ip = nn.Linear(cross_attention_dim, dim, bias=bias)
        if n_registers > 0:
            self.registers = nn.Parameter(torch.randn(1, n_registers, cross_attention_dim) / dim**0.5)
        else:
            self.registers = None
    
    def forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        q_lens: List[int] = None,
        kv_lens: List[int] = None,
        ip_lens: List[int] = None,
        ip_scale: float = 1.0,
    ):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if ip_hidden_states is not None:

            if self.registers is not None:
                ip_hidden_states_list = vector_to_list(ip_hidden_states, ip_lens, 1)
                ip_hidden_states_list = merge_token_lists(ip_hidden_states_list, [self.registers] * len(ip_hidden_states_list), 1)
                ip_hidden_states, ip_lens = list_to_vector(ip_hidden_states_list, 1)

            ip_query = query
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            if attn.norm_q is not None:
                ip_query = attn.norm_q(ip_query)
            if attn.norm_k is not None:
                ip_key = attn.norm_k(ip_key)
            ip_query = ip_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ip_key = ip_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ip_value = ip_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ip_hidden_states = flash_attention(
                ip_query, ip_key, ip_value,
                q_lens=q_lens,
                kv_lens=ip_lens
            )
            ip_hidden_states = ip_hidden_states.transpose(1, 2).flatten(2, 3)
            ip_hidden_states = ip_hidden_states.type_as(ip_query)


        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        hidden_states = flash_attention(
            query, key, value,
            q_lens=q_lens,
            kv_lens=kv_lens,
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if ip_hidden_states is not None:
            hidden_states = hidden_states + ip_scale * ip_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def register_ip_adapter(
    model,
    cross_attention_dim=None,
    n_registers=0,
    init_method="zero",
    dtype=torch.float32,
):
    attn_procs = {}
    transformer_sd = model.state_dict()
    for layer_idx, block in enumerate(model.blocks):
        name = f"blocks.{layer_idx}.attn2.processor"
        layer_name = name.split(".processor")[0]
        dim = transformer_sd[layer_name + ".to_k.weight"].shape[1]
        attn_procs[name] = WanIPAttnProcessor(
            cross_attention_dim=dim if cross_attention_dim is None else cross_attention_dim,
            dim=dim,
            n_registers=n_registers,
            bias=True,
        )
        if init_method == "zero":
            torch.nn.init.zeros_(attn_procs[name].to_k_ip.weight)
            torch.nn.init.zeros_(attn_procs[name].to_k_ip.bias)
            torch.nn.init.zeros_(attn_procs[name].to_v_ip.weight)
            torch.nn.init.zeros_(attn_procs[name].to_v_ip.bias)
        elif init_method == "clone":
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": transformer_sd[layer_name + ".to_k.weight"],
                "to_k_ip.bias": transformer_sd[layer_name + ".to_k.bias"],
                "to_v_ip.weight": transformer_sd[layer_name + ".to_v.weight"],
                "to_v_ip.bias": transformer_sd[layer_name + ".to_v.bias"],
            }
            attn_procs[name].load_state_dict(weights)
        elif init_method == "random":
            pass
        else:
            raise ValueError(f"{init_method} is not supported.")
        block.attn2.processor = attn_procs[name]
    ip_layers = torch.nn.ModuleList(attn_procs.values())
    ip_layers.to(device=model.device, dtype=dtype)
    return model, ip_layers


class WanRefAttnProcessor(nn.Module):
    def __init__(
        self,
        dim: int,
        bias: bool,
    ):
        super().__init__()
        self.to_k_ref = nn.Linear(dim, dim, bias=bias)
        self.to_v_ref = nn.Linear(dim, dim, bias=bias)
    
    def forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        q_lens: List[int] = None,
        kv_lens: List[int] = None,
        ref_feature: Optional[tuple] = None,
        ref_scale: float = 1.0,
    ):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if ref_feature is None:
            ref_hidden_states = None
        else:
            ref_hidden_states, ref_lens = ref_feature
            ref_query = query
            ref_key = self.to_k_ref(ref_hidden_states)
            ref_value = self.to_v_ref(ref_hidden_states)
            if attn.norm_q is not None:
                ref_query = attn.norm_q(ref_query)
            if attn.norm_k is not None:
                ref_key = attn.norm_k(ref_key)
            ref_query = ref_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ref_key = ref_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ref_value = ref_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ref_hidden_states = flash_attention(
                ref_query, ref_key, ref_value,
                q_lens=q_lens,
                kv_lens=ref_lens
            )
            ref_hidden_states = ref_hidden_states.transpose(1, 2).flatten(2, 3)
            ref_hidden_states = ref_hidden_states.type_as(ref_query)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        
        hidden_states = flash_attention(
            query, key, value,
            q_lens=q_lens,
            kv_lens=kv_lens,
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if ref_hidden_states is not None:
            hidden_states = hidden_states + ref_scale * ref_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def register_ref_adapter(
    model,
    init_method="zero",
    dtype=torch.float32,
):
    attn_procs = {}
    transformer_sd = model.state_dict()
    for layer_idx, block in enumerate(model.blocks):
        name = f"blocks.{layer_idx}.attn1.processor"
        layer_name = name.split(".processor")[0]
        dim = transformer_sd[layer_name + ".to_k.weight"].shape[0]
        attn_procs[name] = WanRefAttnProcessor(
            dim=dim,
            bias=True,
        )
        if init_method == "zero":
            torch.nn.init.zeros_(attn_procs[name].to_k_ref.weight)
            torch.nn.init.zeros_(attn_procs[name].to_k_ref.bias)
            torch.nn.init.zeros_(attn_procs[name].to_v_ref.weight)
            torch.nn.init.zeros_(attn_procs[name].to_v_ref.bias)
        elif init_method == "clone":
            weights = {
                "to_k_ref.weight": transformer_sd[layer_name + ".to_k.weight"],
                "to_k_ref.bias": transformer_sd[layer_name + ".to_k.bias"],
                "to_v_ref.weight": transformer_sd[layer_name + ".to_v.weight"],
                "to_v_ref.bias": transformer_sd[layer_name + ".to_v.bias"],
            }
            attn_procs[name].load_state_dict(weights)
        elif init_method == "random":
            pass
        else:
            raise ValueError(f"{init_method} is not supported.")
        block.attn1.processor = attn_procs[name]
    ref_layers = torch.nn.ModuleList(attn_procs.values())
    ref_layers.to(device=model.device, dtype=dtype)
    return model, ref_layers
