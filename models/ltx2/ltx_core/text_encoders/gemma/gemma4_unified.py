"""Inference-only Gemma 4 text model used by LTX 2.5.

The architecture is derived from Hugging Face Transformers' Gemma 4 Unified
implementation (Apache-2.0), trimmed to the causal text-encoder path WanGP
actually uses. Keeping it local lets WanGP remain on its Transformers 4.x
runtime while loading the official LTX 2.5 Gemma 4 checkpoint.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


class Gemma4UnifiedTextConfig(PretrainedConfig):
    model_type = "gemma4_unified_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 262_144,
        hidden_size: int = 2304,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        hidden_activation: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 262_144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = 0,
        eos_token_id: int | list[int] | None = 1,
        bos_token_id: int | None = 2,
        tie_word_embeddings: bool = True,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int = 1024,
        layer_types: list[str] | None = None,
        final_logit_softcapping: float | None = None,
        use_bidirectional_attention: str = "vision",
        num_global_key_value_heads: int | None = None,
        global_head_dim: int = 512,
        attention_k_eq_v: bool = False,
        num_kv_shared_layers: int = 0,
        use_double_wide_mlp: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters or {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        }
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types or [
            "sliding_attention" if (index + 1) % 6 else "full_attention"
            for index in range(num_hidden_layers)
        ]
        self.final_logit_softcapping = final_logit_softcapping
        self.use_bidirectional_attention = use_bidirectional_attention
        self.num_global_key_value_heads = num_global_key_value_heads or num_key_value_heads
        self.global_head_dim = global_head_dim
        self.attention_k_eq_v = attention_k_eq_v
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_double_wide_mlp = use_double_wide_mlp
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, bos_token_id=bos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)


class Gemma4UnifiedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states.float()
        output = output * torch.pow(output.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            output = output * self.weight.float()
        return output.type_as(hidden_states)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return x * cos + _rotate_half(x) * sin


def _repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return hidden_states
    batch, heads, tokens, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None].expand(batch, heads, repetitions, tokens, head_dim)
    return hidden_states.reshape(batch, heads * repetitions, tokens, head_dim)


class Gemma4UnifiedTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4UnifiedTextConfig):
        super().__init__()
        for layer_type in set(config.layer_types):
            params = config.rope_parameters[layer_type]
            head_dim = config.global_head_dim if layer_type == "full_attention" else config.head_dim
            if params["rope_type"] == "proportional":
                rotary_angles = int(params.get("partial_rotary_factor", 1.0) * head_dim // 2)
                inv_freq = 1.0 / (
                    params["rope_theta"]
                    ** (torch.arange(0, 2 * rotary_angles, 2, dtype=torch.float32) / head_dim)
                )
                if rotary_angles < head_dim // 2:
                    inv_freq = torch.cat((inv_freq, torch.zeros(head_dim // 2 - rotary_angles)))
                inv_freq /= params.get("factor", 1.0)
            else:
                inv_freq = 1.0 / (
                    params["rope_theta"] ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
                )
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq").to(x.device)
        freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        return embeddings.cos().to(x.dtype), embeddings.sin().to(x.dtype)


class Gemma4UnifiedTextAttention(nn.Module):
    def __init__(self, config: Gemma4UnifiedTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim
        self.num_attention_heads = config.num_attention_heads
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        num_key_value_heads = config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.q_norm = Gemma4UnifiedRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.k_norm = Gemma4UnifiedRMSNorm(self.head_dim, config.rms_norm_eps)
        self.v_norm = Gemma4UnifiedRMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)
        self.v_proj = None if self.use_alternative_attention else nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        shape = (batch, tokens, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(shape))
        query = _apply_rotary(query, cos, sin).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape)
        value = key if self.v_proj is None else self.v_proj(hidden_states).view(shape)
        key = _apply_rotary(self.k_norm(key), cos, sin).transpose(1, 2)
        value = self.v_norm(value).transpose(1, 2)
        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, scale=1.0)
        output = output.transpose(1, 2).reshape(batch, tokens, -1)
        return self.o_proj(output)


class Gemma4UnifiedTextMLP(nn.Module):
    def __init__(self, config: Gemma4UnifiedTextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Gemma4UnifiedTextDecoderLayer(nn.Module):
    def __init__(self, config: Gemma4UnifiedTextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4UnifiedTextAttention(config, layer_idx)
        self.mlp = Gemma4UnifiedTextMLP(config)
        self.input_layernorm = Gemma4UnifiedRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4UnifiedRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4UnifiedRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4UnifiedRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), position_embeddings, attention_mask)
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        return hidden_states * self.layer_scalar


class Gemma4UnifiedTextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None, embed_scale: float):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma4UnifiedPreTrainedModel(PreTrainedModel):
    config_class = Gemma4UnifiedTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Gemma4UnifiedTextDecoderLayer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Gemma4UnifiedTextModel(Gemma4UnifiedPreTrainedModel):
    def __init__(self, config: Gemma4UnifiedTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Gemma4UnifiedTextScaledWordEmbedding(config.vocab_size, config.hidden_size, self.padding_idx, math.sqrt(config.hidden_size))
        self.layers = nn.ModuleList(Gemma4UnifiedTextDecoderLayer(config, index) for index in range(config.num_hidden_layers))
        self.norm = Gemma4UnifiedRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Gemma4UnifiedTextRotaryEmbedding(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _attention_masks(self, attention_mask: torch.Tensor, tokens: int) -> dict[str, torch.Tensor]:
        positions = torch.arange(tokens, device=attention_mask.device)
        if self.config.use_bidirectional_attention == "all":
            causal = torch.ones((tokens, tokens), dtype=torch.bool, device=attention_mask.device)
        else:
            causal = positions[:, None] >= positions[None, :]
        key_mask = attention_mask[:, None, None, :].bool()
        full_mask = causal[None, None] & key_mask
        sliding = positions[:, None] - positions[None, :] < self.config.sliding_window
        return {"full_attention": full_mask, "sliding_attention": full_mask & sliding[None, None]}

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, tokens, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch, tokens), dtype=torch.bool, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = torch.arange(tokens, device=inputs_embeds.device).unsqueeze(0)
        masks = self._attention_masks(attention_mask, tokens)
        position_embeddings = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for index, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[index]
            hidden_states = layer(hidden_states, position_embeddings[layer_type], masks[layer_type])
            if output_hidden_states and index + 1 < len(self.layers):
                all_hidden_states += (hidden_states,)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if return_dict is False:
            return tuple(value for value in (hidden_states, None, all_hidden_states, None) if value is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None, hidden_states=all_hidden_states, attentions=None)


class Gemma4UnifiedForCausalLM(Gemma4UnifiedPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Gemma4UnifiedTextConfig):
        super().__init__(config)
        self.model = Gemma4UnifiedTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        if self.config.final_logit_softcapping is not None:
            logits = torch.tanh(logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
        if return_dict is False:
            return (logits, outputs.hidden_states)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None, hidden_states=outputs.hidden_states, attentions=None)


def register_gemma4_config() -> None:
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if Gemma4UnifiedTextConfig.model_type not in CONFIG_MAPPING:
        AutoConfig.register(Gemma4UnifiedTextConfig.model_type, Gemma4UnifiedTextConfig)


__all__ = ["Gemma4UnifiedForCausalLM", "Gemma4UnifiedTextConfig", "register_gemma4_config"]
