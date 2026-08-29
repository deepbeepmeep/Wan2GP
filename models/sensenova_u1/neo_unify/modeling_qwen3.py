from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

import copy
import math
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers import Qwen3Config

from shared.attention import pay_attention

from .transformers_compat import causal_mask_kwargs, model_input_compat, tied_weights_keys


def _take_tensor(tensor_list):
    tensor = tensor_list[0]
    tensor_list.clear()
    return tensor


def _linear_disposable(linear, tensor_list):
    return linear(_take_tensor(tensor_list))


def _attention_mask_for_pay_attention(attention_mask, q_len, num_heads):
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        if attention_mask.shape[1] == num_heads and attention_mask.shape[2] == q_len:
            return attention_mask.transpose(1, 2)
        if attention_mask.shape[2] in (1, num_heads) and attention_mask.shape[1] == q_len:
            return attention_mask
        if attention_mask.shape[1] == 1 and attention_mask.shape[2] == q_len:
            return attention_mask.transpose(1, 2)
    if attention_mask.dim() == 3:
        return attention_mask[:, :, None, :]
    if attention_mask.dim() == 2:
        return attention_mask[:, None, None, :]
    return attention_mask


def _pay_attention_gqa(qkv_list, dropout_p=0.0, softmax_scale=None, causal=False, attention_mask=None):
    q, k, v = qkv_list
    qkv_list.clear()
    batch, q_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(q.dtype)
    if num_heads == num_kv_heads:
        qkv_list = [q, k, v]
        del q, k, v
        return pay_attention(qkv_list, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, attention_mask=attention_mask, recycle_q=True)

    groups = num_heads // num_kv_heads
    k_len = k.shape[1]
    q = q.view(batch, q_len, num_kv_heads, groups, head_dim).permute(0, 2, 1, 3, 4).reshape(batch * num_kv_heads, q_len, groups, head_dim)
    k = k.permute(0, 2, 1, 3).reshape(batch * num_kv_heads, k_len, 1, head_dim).expand(-1, -1, groups, -1)
    v = v.permute(0, 2, 1, 3).reshape(batch * num_kv_heads, k_len, 1, head_dim).expand(-1, -1, groups, -1)
    if attention_mask is not None:
        if attention_mask.shape[2] == 1:
            attention_mask = attention_mask[:, None].expand(batch, num_kv_heads, q_len, 1, k_len).reshape(batch * num_kv_heads, q_len, 1, k_len)
        else:
            attention_mask = attention_mask.view(batch, q_len, num_kv_heads, groups, k_len).permute(0, 2, 1, 3, 4).reshape(batch * num_kv_heads, q_len, groups, k_len)
    qkv_list = [q, k, v]
    del q, k, v
    output = pay_attention(qkv_list, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, attention_mask=attention_mask, force_attention="sdpa", recycle_q=True)
    return output.view(batch, num_kv_heads, q_len, groups, head_dim).permute(0, 2, 1, 3, 4).reshape(batch, q_len, num_heads, head_dim)


def _shared_attention(qkv_list, dropout_p: float = 0.0, softmax_scale=None, causal: bool = False):
    return _pay_attention_gqa(qkv_list, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)


def create_block_causal_mask(index: torch.Tensor):
    """
    index: (L)
    return: (1, 1, L, L) block-wise causal attention mask
    """
    L = index.size(0)
    idx_i = index.unsqueeze(1).expand(L, L)
    idx_j = index.unsqueeze(0).expand(L, L)

    arange = torch.arange(L, device=index.device)
    mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))

    return torch.zeros_like(mask, dtype=torch.float32).masked_fill_(~mask, float("-inf"))[None, None]


def visualize_mask(mask: torch.Tensor, i: int = 0, j: int = 12):
    """
    mask: (1,1, L, L)
    """
    submask = torch.where(mask[0, 0, :, :] == 0, torch.tensor(1.0), torch.tensor(0.0))
    submask = mask[i:j, i:j].int().cpu().numpy()
    for row in submask:
        print(" ".join(map(str, row)))


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = F.rms_norm(hidden_states.float(), (hidden_states.shape[-1],), eps=self.variance_epsilon)
        return hidden_states.to(input_dtype).mul_(self.weight)

    def forward_list(self, hidden_states_list) -> torch.Tensor:
        return self.forward(_take_tensor(hidden_states_list))

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x_list):
        x = _take_tensor(x_list)
        seq_len = x.shape[-2]
        if seq_len <= 512:
            hidden_states = self.act_fn(self.gate_proj(x))
            hidden_states.mul_(self.up_proj(x))
            del x
            hidden_states_list = [hidden_states]
            del hidden_states
            return _linear_disposable(self.down_proj, hidden_states_list)
        chunk_size = max(128, min(seq_len, seq_len * self.hidden_size // self.intermediate_size))
        output = x.new_empty(*x.shape[:-1], self.hidden_size)
        for start in range(0, seq_len, chunk_size):
            chunk = x.narrow(-2, start, min(chunk_size, seq_len - start))
            chunk_hidden_states = self.act_fn(self.gate_proj(chunk))
            chunk_hidden_states.mul_(self.up_proj(chunk))
            chunk_hidden_states_list = [chunk_hidden_states]
            del chunk_hidden_states
            chunk_output = _linear_disposable(self.down_proj, chunk_hidden_states_list)
            output.narrow(-2, start, chunk_output.shape[-2]).copy_(chunk_output)
            del chunk_output
        return output


def _apply_rotary_inplace(x, cos, sin, scratch, unsqueeze_dim):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    scratch = scratch[tuple(slice(0, size) for size in x1.shape)]
    scratch.copy_(x1)
    x1.mul_(cos).addcmul_(x2, sin, value=-1)
    x2.mul_(cos).addcmul_(scratch, sin)
    return x


def apply_rotary_pos_emb(qk_list, cos, sin, scratch, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    q, k = qk_list
    qk_list.clear()
    return _apply_rotary_inplace(q, cos, sin, scratch, unsqueeze_dim), _apply_rotary_inplace(k, cos, sin, scratch, unsqueeze_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    attention_mask = _attention_mask_for_pay_attention(attention_mask, query.shape[1], query.shape[2])
    qkv_list = [query, key, value]
    del query, key, value
    return _pay_attention_gqa(qkv_list, dropout_p=dropout, softmax_scale=scaling, attention_mask=attention_mask), None


def _compute_default_rope_parameters(config, device=None, **_kwargs):
    """Default RoPE frequencies, inlined to avoid breakage across transformers versions.

    transformers <=4.x exposes this as ``ROPE_INIT_FUNCTIONS["default"]``, but
    5.x dropped the ``"default"`` key from that table. Having a local copy keeps
    ``Qwen3RotaryEmbedding`` working on both.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    compute_default_rope_parameters = staticmethod(_compute_default_rope_parameters)

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type == "default" or self.rope_type is None:
            base_rope_init_fn = _compute_default_rope_parameters
        else:
            base_rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        def _rope_init_fn_keep_freq_range(cfg: Qwen3Config, dev=None):
            inv_freq, attention_scaling = base_rope_init_fn(cfg, dev)

            cfg2 = copy.deepcopy(cfg)
            head_dim = getattr(cfg2, "head_dim", None)
            if head_dim is None:
                head_dim = cfg2.hidden_size // cfg2.num_attention_heads
                setattr(cfg2, "head_dim", head_dim)
            cfg2.head_dim = int(head_dim) * 2

            inv_freq_full, _ = base_rope_init_fn(cfg2, dev)
            inv_freq = inv_freq_full[::2]

            return inv_freq, attention_scaling

        self.rope_init_fn = _rope_init_fn_keep_freq_range

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def reset_inv_freq(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, torch.device("cpu"))
        self.inv_freq = inv_freq
        self.original_inv_freq = inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            cos = freqs.cos().mul_(self.attention_scaling)
            sin = freqs.sin().mul_(self.attention_scaling)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_proj_mot_gen = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )

        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_mot_gen = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_mot_gen = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.o_proj_mot_gen = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.q_norm_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)

        self.k_norm = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.k_norm_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_hw = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.k_norm_hw_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        t_config = copy.deepcopy(config)
        t_config.head_dim = config.head_dim // 2
        self.rotary_emb = Qwen3RotaryEmbedding(config=t_config)

        hw_config = copy.deepcopy(config)
        hw_config.head_dim = config.head_dim // 4
        hw_config.rope_theta = config.rope_theta_hw
        if isinstance(getattr(hw_config, "rope_parameters", None), dict):
            hw_config.rope_parameters = {**hw_config.rope_parameters, "rope_theta": config.rope_theta_hw}
        hw_config.max_position_embeddings = config.max_position_embeddings_hw
        self.rotary_emb_hw = Qwen3RotaryEmbedding(config=hw_config)
    
    def forward_und(
        self,
        hidden_states_list,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = _take_tensor(hidden_states_list)
        assert self.config._attn_implementation == "eager"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        hidden_states_list = [hidden_states]
        del hidden_states
        value_states = _linear_disposable(self.v_proj, hidden_states_list).view(hidden_shape)

        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t.copy_(self.q_norm(query_states_t))
        query_states_hw.copy_(self.q_norm_hw(query_states_hw))
        query_states_t = query_states_t.transpose(1, 2)
        query_states_hw = query_states_hw.transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t.copy_(self.k_norm(key_states_t))
        key_states_hw.copy_(self.k_norm_hw(key_states_hw))
        key_states_t = key_states_t.transpose(1, 2)
        key_states_hw = key_states_hw.transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        rope_scratch = torch.empty_like(query_states_t[..., :query_states_t.shape[-1] // 2])
        qk_list = [query_states_t, key_states_t]
        del query_states_t, key_states_t
        query_states_t, key_states_t = apply_rotary_pos_emb(qk_list, cos_t, sin_t, rope_scratch)
        qk_list = [query_states_h, key_states_h]
        del query_states_h, key_states_h
        query_states_h, key_states_h = apply_rotary_pos_emb(qk_list, cos_h, sin_h, rope_scratch)
        qk_list = [query_states_w, key_states_w]
        del query_states_w, key_states_w
        query_states_w, key_states_w = apply_rotary_pos_emb(qk_list, cos_w, sin_w, rope_scratch)

        del query_states_t, query_states_hw, query_states_h, query_states_w
        del key_states_t, key_states_hw, key_states_h, key_states_w
        del cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, rope_scratch

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            update_cache = kwargs.get("update_cache", True)
            if update_cache:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
            else:
                # only use the past key values but do not append the current one
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values

                if past_k is not None:
                    key_states   = torch.cat([past_k, key_states], dim=2)   # concat on seq_len
                    value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output_list = [attn_output]
        del attn_output
        attn_output = _linear_disposable(self.o_proj, attn_output_list)
        return attn_output, attn_weights

    # def forward_gen(
    #     self,
    #     hidden_states: torch.Tensor,
    #     indexes: Optional[torch.LongTensor],
    #     attention_mask: Optional[torch.Tensor],
    #     past_key_values: Optional[Cache] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     assert self.config._attn_implementation == "eager"
    #     input_shape = hidden_states.shape[:-1]
    #     hidden_shape = (*input_shape, -1, self.head_dim)

    #     query_states = self.q_proj_mot_gen(hidden_states).view(hidden_shape)
    #     query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
    #     query_states_t = self.q_norm_mot_gen(query_states_t).transpose(1, 2)
    #     query_states_hw = self.q_norm_hw_mot_gen(query_states_hw).transpose(1, 2)
    #     query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

    #     key_states = self.k_proj_mot_gen(hidden_states).view(hidden_shape)
    #     key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
    #     key_states_t = self.k_norm_mot_gen(key_states_t).transpose(1, 2)
    #     key_states_hw = self.k_norm_hw_mot_gen(key_states_hw).transpose(1, 2)
    #     key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

    #     value_states = self.v_proj_mot_gen(hidden_states).view(hidden_shape).transpose(1, 2)

    #     cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
    #     query_states_t, key_states_t = apply_rotary_pos_emb(query_states_t, key_states_t, cos_t, sin_t)

    #     cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
    #     query_states_h, key_states_h = apply_rotary_pos_emb(query_states_h, key_states_h, cos_h, sin_h)

    #     cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
    #     query_states_w, key_states_w = apply_rotary_pos_emb(query_states_w, key_states_w, cos_w, sin_w)

    #     query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
    #     key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)


    #     if past_key_values is not None:
    #         # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #         # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #         # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    #         update_cache = kwargs.get("update_cache", True)
    #         if update_cache:
    #             key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
    #         else:
    #             # only use the past key values but do not append the current one
    #             layer = past_key_values.layers[self.layer_idx]
    #             past_k, past_v = layer.keys, layer.values

    #             if past_k is not None:
    #                 key_states   = torch.cat([past_k, key_states], dim=2)   # concat on seq_len
    #                 value_states = torch.cat([past_v, value_states], dim=2)

    #     attention_interface: Callable = eager_attention_forward
    #     if self.config._attn_implementation != "eager":
    #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    #     attn_output, attn_weights = attention_interface(
    #         self,
    #         query_states,
    #         key_states,
    #         value_states,
    #         attention_mask,
    #         dropout=0.0 if not self.training else self.attention_dropout,
    #         scaling=self.scaling,
    #         sliding_window=self.sliding_window,  # diff with Llama
    #         **kwargs,
    #     )

    #     attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    #     attn_output = self.o_proj_mot_gen(attn_output)
    #     return attn_output, attn_weights

    def forward_gen(
        self,
        hidden_states_list,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = _take_tensor(hidden_states_list)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # -----------------------------
        # Build q / k / v for current tokens
        # Internal layout before flash:
        #   q/k/v: [B, H, S, D]
        # Flash layout:
        #   q/k/v: [B, S, H, D]
        # -----------------------------
        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states = self.q_proj_mot_gen(hidden_states).view(hidden_shape)
        key_states = self.k_proj_mot_gen(hidden_states).view(hidden_shape)
        hidden_states_list = [hidden_states]
        del hidden_states
        value_states = _linear_disposable(self.v_proj_mot_gen, hidden_states_list).view(hidden_shape)

        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t.copy_(self.q_norm_mot_gen(query_states_t))
        query_states_hw.copy_(self.q_norm_hw_mot_gen(query_states_hw))
        query_states_t = query_states_t.transpose(1, 2)   # [B,H,S,D/2]
        query_states_hw = query_states_hw.transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t.copy_(self.k_norm_mot_gen(key_states_t))
        key_states_hw.copy_(self.k_norm_hw_mot_gen(key_states_hw))
        key_states_t = key_states_t.transpose(1, 2)       # [B,H,S,D/2]
        key_states_hw = key_states_hw.transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        rope_scratch = torch.empty_like(query_states_t[..., :query_states_t.shape[-1] // 2])
        qk_list = [query_states_t, key_states_t]
        del query_states_t, key_states_t
        query_states_t, key_states_t = apply_rotary_pos_emb(qk_list, cos_t, sin_t, rope_scratch)
        qk_list = [query_states_h, key_states_h]
        del query_states_h, key_states_h
        query_states_h, key_states_h = apply_rotary_pos_emb(qk_list, cos_h, sin_h, rope_scratch)
        qk_list = [query_states_w, key_states_w]
        del query_states_w, key_states_w
        query_states_w, key_states_w = apply_rotary_pos_emb(qk_list, cos_w, sin_w, rope_scratch)

        del query_states_t, query_states_hw, query_states_h, query_states_w
        del key_states_t, key_states_hw, key_states_h, key_states_w
        del cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, rope_scratch

        update_cache = kwargs.get("update_cache", True)

        # ------------------------------------------------------------------
        # Flash path:
        # Only use when there is no explicit dense mask.
        # This is exactly the t2i denoising use case:
        #   current image tokens attend to [prefix + current image tokens]
        #   fully bidirectional inside current block => causal=False
        # ------------------------------------------------------------------
        if attention_mask is None:
            q = query_states
            k_cur = key_states
            v_cur = value_states
            del query_states

            if past_key_values is not None:
                if update_cache:
                    # Rare path, keep compatibility.
                    # past_key_values.update expects [B,H,S,D]
                    key_states, value_states = past_key_values.update(
                        key_states.transpose(1, 2), value_states.transpose(1, 2), self.layer_idx, cache_kwargs=None
                    )
                    k = key_states.transpose(1, 2).contiguous()
                    v = value_states.transpose(1, 2).contiguous()
                else:
                    # Optimized path:
                    # use preallocated flash_k_cache / flash_v_cache
                    layer = past_key_values.layers[self.layer_idx]

                    if (
                        hasattr(layer, "flash_k_cache")
                        and layer.flash_k_cache is not None
                        and hasattr(layer, "flash_v_cache")
                        and layer.flash_v_cache is not None
                    ):
                        prefix_len = layer.flash_prefix_len
                        cur_len = k_cur.shape[1]

                        # overwrite current segment in-place
                        layer.flash_k_cache[:, prefix_len:prefix_len + cur_len].copy_(k_cur)
                        layer.flash_v_cache[:, prefix_len:prefix_len + cur_len].copy_(v_cur)

                        k = layer.flash_k_cache[:, :prefix_len + cur_len]
                        v = layer.flash_v_cache[:, :prefix_len + cur_len]
                    else:
                        # Low-memory mode: retain only compact prefix K/V and
                        # assemble the active layer's attention inputs on demand.
                        layer = past_key_values.layers[self.layer_idx]
                        past_k, past_v = layer.keys, layer.values

                        if past_k is not None:
                            past_k = past_k.transpose(1, 2).contiguous()
                            past_v = past_v.transpose(1, 2).contiguous()
                            k = torch.cat([past_k, k_cur], dim=1)
                            v = torch.cat([past_v, v_cur], dim=1)
                            del past_k, past_v
                        else:
                            k = k_cur
                            v = v_cur
            else:
                k = k_cur
                v = v_cur
            del key_states, value_states, k_cur, v_cur

            # sanity checks
            assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
            assert q.shape[0] == k.shape[0] == v.shape[0], (q.shape, k.shape, v.shape)
            assert k.shape[1] == v.shape[1], (k.shape, v.shape)
            assert k.shape[2] == v.shape[2], (k.shape, v.shape)
            assert q.shape[3] == k.shape[3] == v.shape[3], (q.shape, k.shape, v.shape)

            qkv_list = [q, k, v]
            del q, k, v
            attn_output = _shared_attention(
                qkv_list,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                softmax_scale=self.scaling,
                causal=False,
            )  # [B, S_q, H_q, D]

            attn_output = attn_output.reshape(*input_shape, -1)
            attn_output_list = [attn_output]
            del attn_output
            attn_output = _linear_disposable(self.o_proj_mot_gen, attn_output_list)
            return attn_output, None

        # ------------------------------------------------------------------
        # Original eager fallback path
        # ------------------------------------------------------------------
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if past_key_values is not None:
            if update_cache:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
            else:
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values
                if past_k is not None:
                    key_states = torch.cat([past_k, key_states], dim=2)
                    value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        del query_states, key_states, value_states

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output_list = [attn_output]
        del attn_output
        attn_output = _linear_disposable(self.o_proj_mot_gen, attn_output_list)
        return attn_output, attn_weights

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(hidden_states, indexes, attention_mask, past_key_values, cache_position, **kwargs)
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(hidden_states, indexes, attention_mask, past_key_values, cache_position, **kwargs)

        # Mixed und/gen path: mirrors forward_und / forward_gen per token type.
        # (Fixed per issue #207: the time-dim qk-norm, the `.view(hidden_shape)` before
        # chunking, and the transpose on the time chunk were previously missing.)
        # Note: Remove this raise once fully tested.
        raise NotImplementedError(
            "The mixed und/gen forward path is not yet validated (issue #207): known "
            "issues are fixed, but it has no parity test and no production caller. "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )

        assert self.config._attn_implementation == "eager"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = hidden_states.new_zeros((*input_shape, self.config.num_attention_heads*self.head_dim))
        if exist_non_image_gen_tokens:
            query_states[~image_gen_indicators] = self.q_proj(hidden_states[~image_gen_indicators])
        if exist_image_gen_tokens:
            query_states[image_gen_indicators] = self.q_proj_mot_gen(hidden_states[image_gen_indicators])
        query_states = query_states.view(hidden_shape)  # [B, S, H, D]
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)

        _query_states_t = query_states_t.new_zeros(query_states_t.shape)
        if exist_non_image_gen_tokens:
            _query_states_t[~image_gen_indicators] = self.q_norm(query_states_t[~image_gen_indicators])
        if exist_image_gen_tokens:
            _query_states_t[image_gen_indicators] = self.q_norm_mot_gen(query_states_t[image_gen_indicators])
        query_states_t = _query_states_t.transpose(1, 2)  # [B, H, S, D/2]

        _query_states_hw = query_states_hw.new_zeros(query_states_hw.shape)
        if exist_non_image_gen_tokens:
            _query_states_hw[~image_gen_indicators] = self.q_norm_hw(query_states_hw[~image_gen_indicators])
        if exist_image_gen_tokens:
            _query_states_hw[image_gen_indicators] = self.q_norm_hw_mot_gen(query_states_hw[image_gen_indicators])
        query_states_hw = _query_states_hw.transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = hidden_states.new_zeros((*input_shape, self.config.num_key_value_heads*self.head_dim))
        if exist_non_image_gen_tokens:
            key_states[~image_gen_indicators] = self.k_proj(hidden_states[~image_gen_indicators])
        if exist_image_gen_tokens:
            key_states[image_gen_indicators] = self.k_proj_mot_gen(hidden_states[image_gen_indicators])
        key_states = key_states.view(hidden_shape)  # [B, S, H_kv, D]
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)

        _key_states_t = key_states_t.new_zeros(key_states_t.shape)
        if exist_non_image_gen_tokens:
            _key_states_t[~image_gen_indicators] = self.k_norm(key_states_t[~image_gen_indicators])
        if exist_image_gen_tokens:
            _key_states_t[image_gen_indicators] = self.k_norm_mot_gen(key_states_t[image_gen_indicators])
        key_states_t = _key_states_t.transpose(1, 2)  # [B, H_kv, S, D/2]

        _key_states_hw = key_states_hw.new_zeros(key_states_hw.shape)
        if exist_non_image_gen_tokens:
            _key_states_hw[~image_gen_indicators] = self.k_norm_hw(key_states_hw[~image_gen_indicators])
        if exist_image_gen_tokens:
            _key_states_hw[image_gen_indicators] = self.k_norm_hw_mot_gen(key_states_hw[image_gen_indicators])
        key_states_hw = _key_states_hw.transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = hidden_states.new_zeros((*input_shape, self.config.num_key_value_heads*self.head_dim))
        if exist_non_image_gen_tokens:
            value_states[~image_gen_indicators] = self.v_proj(hidden_states[~image_gen_indicators])
        if exist_image_gen_tokens:
            value_states[image_gen_indicators] = self.v_proj_mot_gen(hidden_states[image_gen_indicators])
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        rope_scratch = torch.empty_like(query_states_t[..., :query_states_t.shape[-1] // 2])
        qk_list = [query_states_t, key_states_t]
        del query_states_t, key_states_t
        query_states_t, key_states_t = apply_rotary_pos_emb(qk_list, cos_t, sin_t, rope_scratch)
        qk_list = [query_states_h, key_states_h]
        del query_states_h, key_states_h
        query_states_h, key_states_h = apply_rotary_pos_emb(qk_list, cos_h, sin_h, rope_scratch)
        qk_list = [query_states_w, key_states_w]
        del query_states_w, key_states_w
        query_states_w, key_states_w = apply_rotary_pos_emb(qk_list, cos_w, sin_w, rope_scratch)

        query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)


        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            update_cache = kwargs.get("update_cache", True)
            if update_cache:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
            else:
                # only use the past key values but do not append the current one
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values

                if past_k is not None:
                    key_states   = torch.cat([past_k, key_states], dim=2)   # concat on seq_len
                    value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )
        del query_states, key_states, value_states

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        _attn_output = attn_output.new_zeros((*input_shape, self.config.hidden_size))
        if exist_non_image_gen_tokens:
            _attn_output[~image_gen_indicators] = self.o_proj(attn_output[~image_gen_indicators])
        if exist_image_gen_tokens:
            _attn_output[image_gen_indicators] = self.o_proj_mot_gen(attn_output[image_gen_indicators])

        attn_output = _attn_output
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.mlp_mot_gen = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward_und(
        self,
        hidden_states_list,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = _take_tensor(hidden_states_list)
        hidden_states = self.input_layernorm(residual)
        hidden_states_list = [hidden_states]
        del hidden_states
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states_list,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual.add_(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states_list = [hidden_states]
        del hidden_states
        hidden_states = self.mlp(hidden_states_list)
        hidden_states = residual.add_(hidden_states)
        return hidden_states

    def forward_gen(
        self,
        hidden_states_list,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = _take_tensor(hidden_states_list)
        hidden_states = self.input_layernorm_mot_gen(residual)
        hidden_states_list = [hidden_states]
        del hidden_states
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states_list,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual.add_(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_mot_gen(hidden_states)
        hidden_states_list = [hidden_states]
        del hidden_states
        hidden_states = self.mlp_mot_gen(hidden_states_list)
        hidden_states = residual.add_(hidden_states)
        return hidden_states

    def forward_gen_branches(self, hidden_states, branches):
        outputs = []
        for branch in branches:
            branch_hidden_states = [hidden_states.pop(0)]
            outputs.append(self.forward_gen(
                branch_hidden_states,
                image_gen_indicators=None,
                exist_non_image_gen_tokens=False,
                exist_image_gen_tokens=True,
                indexes=branch["indexes"],
                attention_mask=branch["attention_mask"][self.attention_type],
                past_key_values=branch["past_key_values"],
                use_cache=True,
                update_cache=False,
            ))
        return outputs

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        branches=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if branches is not None:
            return self.forward_gen_branches(hidden_states, branches)
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(hidden_states, image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, indexes, attention_mask, position_ids, past_key_values, use_cache, cache_position, **kwargs)
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(hidden_states, image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, indexes, attention_mask, position_ids, past_key_values, use_cache, cache_position, **kwargs)

        # Mixed und/gen path — see the NOTE in Qwen3Attention.forward for caveats.
        raise NotImplementedError(
            "Mixed und/gen decoder-layer forward is not yet validated (issue #207). "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )

        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            _hidden_states[~image_gen_indicators] = self.input_layernorm(hidden_states[~image_gen_indicators])
        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.input_layernorm_mot_gen(hidden_states[image_gen_indicators])
        hidden_states = _hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            _hidden_states[~image_gen_indicators] = self.mlp([self.post_attention_layernorm(hidden_states[~image_gen_indicators])])

        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.mlp_mot_gen([self.post_attention_layernorm_mot_gen(hidden_states[image_gen_indicators])])

        hidden_states = _hidden_states
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.current_index = -1

        # Initialize weights and apply final processing
        self.post_init()

    @model_input_compat
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_gen_indicators: Optional[torch.Tensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_embeds_list=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        
        # assert position_ids is not None
        # assert cache_position is not None
        # assert past_key_values is not None 

        if image_gen_indicators is None:
            exist_non_image_gen_tokens = True
            exist_image_gen_tokens = False
        else:
            # Normalize once before the decoder loop. Leaving these as CUDA
            # scalar tensors makes every layer's Python branch read them back
            # to the host, serializing the compute stream and defeating weight
            # prefetch overlap.
            exist_non_image_gen_tokens = bool((~image_gen_indicators).any().item())
            exist_image_gen_tokens = bool(image_gen_indicators.any().item())
        
        if sum(value is not None for value in (input_ids, inputs_embeds, inputs_embeds_list)) != 1:
            raise ValueError("You must specify exactly one of input_ids, inputs_embeds, or inputs_embeds_list")

        inputs_embeds_owned = inputs_embeds_list is not None or input_ids is not None
        if inputs_embeds_list is not None:
            inputs_embeds = _take_tensor(inputs_embeds_list)
        elif inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            if input_ids is not None:
                mask_kwargs = causal_mask_kwargs(
                    create_causal_mask,
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                self.current_index += 1
                indexes = torch.LongTensor([[self.current_index], [0], [0]]).to(input_ids.device)
            else:
                causal_mask_mapping = {
                    "full_attention": create_block_causal_mask(indexes[0]),
                }
                self.current_index = indexes[0].max()
        else:
            self.current_index = indexes[0].max()
            # raise NotImplementedError('not isinstance(causal_mask_mapping := attention_mask, dict)')

            # The sliding window alternating layers are not always activated depending on the config
            # if self.has_sliding_layers:
            #     causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds if inputs_embeds_owned else inputs_embeds.clone()
        del inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states_list = [hidden_states]
            del hidden_states
            hidden_states = decoder_layer(
                hidden_states_list,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if getattr(self, "_interrupt", False):
                raise InterruptedError("SenseNova-U1 generation aborted")
        if not exist_image_gen_tokens:
            hidden_states_list = [hidden_states]
            del hidden_states
            hidden_states = self.norm.forward_list(hidden_states_list)
        elif not exist_non_image_gen_tokens:
            hidden_states_list = [hidden_states]
            del hidden_states
            hidden_states = self.norm_mot_gen.forward_list(hidden_states_list)
        else:
            _hidden_states = hidden_states.new_zeros(hidden_states.shape)
            _hidden_states[~image_gen_indicators] = self.norm(hidden_states[~image_gen_indicators])
            _hidden_states[image_gen_indicators] = self.norm_mot_gen(hidden_states[image_gen_indicators])
            hidden_states = _hidden_states
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def forward_gen_branches(self, input_embeds_list, branches):
        input_embeds = _take_tensor(input_embeds_list)
        hidden_states = [input_embeds]
        hidden_states.extend(input_embeds.clone() for _ in branches[1:])
        del input_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, image_gen_indicators=None, exist_non_image_gen_tokens=False, exist_image_gen_tokens=True, branches=branches)
            if getattr(self, "_interrupt", False):
                raise InterruptedError("SenseNova-U1 generation aborted")
        outputs = []
        while hidden_states:
            hidden_states_list = [hidden_states.pop(0)]
            outputs.append(self.norm_mot_gen.forward_list(hidden_states_list))
        return outputs


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = tied_weights_keys("lm_head.weight", "model.embed_tokens.weight")
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_embeds_list=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            inputs_embeds_list=inputs_embeds_list,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        past_key_values = outputs.past_key_values
        output_hidden_states = outputs.hidden_states
        output_attentions = outputs.attentions
        hidden_states = outputs.last_hidden_state
        del outputs
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
        hidden_states_list = [hidden_states]
        del hidden_states
        logits = _linear_disposable(self.lm_head, hidden_states_list)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=output_hidden_states,
            attentions=output_attentions,
        )


class Qwen3ForSequenceClassification(GenericForSequenceClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
