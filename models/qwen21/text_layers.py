# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
# Adapted from transformers v5.17.0, modeling_qwen3_vl.py for inference.
from __future__ import annotations
import torch
import torch.nn as nn
from types import SimpleNamespace
from shared.attention import pay_attention
from shared.utils.phase_progress import check_abort
from .vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids, get_vision_attention_seqlens
ACT2FN = {'silu': nn.functional.silu, 'gelu_pytorch_tanh': lambda x: nn.functional.gelu(x, approximate='tanh')}
BaseModelOutputWithDeepstackFeatures = SimpleNamespace

def shared_attention(module, query, key, value, attention_mask=None, scaling=None, **kwargs):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    return (pay_attention([query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)], attention_mask=attention_mask, causal=module.is_causal and attention_mask is None, softmax_scale=scaling), None)

class Qwen3VLVisionMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

class Qwen3VLVisionPatchEmbed(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

class Qwen3VLVisionRotaryEmbedding(nn.Module):

    def __init__(self, config: Qwen3VLVisionConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = self.config.rope_parameters['rope_type']
        rope_init_fn: Callable = self.compute_axial_rope_parameters
        if self.rope_type != 'axial':
            raise ValueError(f'{self.__class__.__name__} supports only axial rope, but requested {self.rope_type}')
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(config: Qwen3VLVisionConfig, device=None, **kwargs) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters['rope_theta']
        dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
        spatial_dim = dim // 2
        attention_factor = 1.0
        inv_freq = 1.0 / base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float, device='cpu') / spatial_dim)
        return (inv_freq.to(device), attention_factor)

    def forward(self, x, position_ids):
        position_ids_expanded = position_ids[..., None].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = position_ids_expanded * self.inv_freq.to(x.device).float()
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling
        cos = self.recomposition_frequencies(cos)
        sin = self.recomposition_frequencies(sin)
        return (cos, sin)

    def recomposition_frequencies(self, freq):
        freq_h, freq_w = (freq[:, 0], freq[:, 1])
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)

class Qwen3VLVisionPatchMerger(nn.Module):

    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size ** 2
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-06)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = (q.float(), k.float())
    cos, sin = (cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float())
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return (q_embed, k_embed)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Qwen3VLVisionAttention(nn.Module):

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** (-0.5)
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None=None, max_seqlen: int | None=None, **kwargs) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_interface = shared_attention
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]
        attn_outputs = [attention_interface(self, q, k, v, attention_mask=None, scaling=self.scaling, dropout=0.0 if not self.training else self.attention_dropout, is_causal=False, **kwargs)[0] for q, k, v in zip(*splits)]
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen3VLVisionBlock(nn.Module):

    def __init__(self, config, attn_implementation: str='sdpa') -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-06)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-06)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None=None, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class Qwen3VLTextRotaryEmbedding(nn.Module):

    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = self.config.rope_parameters['rope_type']
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != 'default':
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get('mrope_section', [24, 20, 20])

    @staticmethod
    def compute_default_rope_parameters(config: Qwen3VLTextConfig, device=None, **kwargs) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters['rope_theta']
        dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float, device='cpu') / dim)
        return (inv_freq.to(device), attention_factor)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq.to(x.device)[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling
        sin = self.recomposition_frequencies(sin)
        cos = self.recomposition_frequencies(cos)
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

    def recomposition_frequencies(self, freq):
        freqs_thw = freq[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)

class Qwen3VLTextRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps: float=1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device='cpu'))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = nn.functional.rms_norm(hidden_states.float(), (hidden_states.shape[-1],), eps=self.variance_epsilon)
        return self.weight * normalized.to(hidden_states.dtype)

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)

class Qwen3VLTextAttention(nn.Module):

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, 'layer_types') else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor | None, past_key_values: Cache | None=None, **kwargs: Unpack[FlashAttentionKwargs]) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        attention_interface = shared_attention
        attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)

class Qwen3VLTextMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Qwen3VLTextDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values: Cache | None=None, use_cache: bool | None=False, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3VLVisionModel(nn.Module):
    config: Qwen3VLVisionConfig
    input_modalities = ('image', 'video')
    _can_record_outputs = {'hidden_states': Qwen3VLVisionBlock, 'attentions': Qwen3VLVisionAttention}

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        self.interpolation_align_corners = True
        self.interpolation_mode = 'bilinear'
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList([Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True) for _ in range(len(config.deepstack_visual_indexes))])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> tuple | BaseModelOutputWithDeepstackFeatures:
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(grid_thw, num_grid_per_side=self.num_grid_per_side, mode=self.interpolation_mode, align_corners=self.interpolation_align_corners, spatial_merge_size=self.config.spatial_merge_size, kwargs=kwargs)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            check_abort()
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, position_embeddings=position_embeddings, **kwargs)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithDeepstackFeatures(last_hidden_state=hidden_states, pooler_output=merged_hidden_states, deepstack_features=deepstack_feature_lists)
