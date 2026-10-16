# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from torch import nn
from mmgp import runtime
from shared import attention_kit
from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from shared.prompt_enhancer.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding
from shared.utils.phase_progress import GenerationAborted, text_encoding_progress


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        width = self.heads * self.head_dim
        self.q_proj, self.gate_proj = (nn.Linear(config.hidden_size, width, bias=False) for _ in range(2))
        self.k_proj, self.v_proj = (nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False) for _ in range(2))
        self.o_proj = nn.Linear(width, config.hidden_size, bias=False)
        self.q_norm, self.k_norm = (Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps) for _ in range(2))

    def forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        gate = self.gate_proj(hidden_states).sigmoid()
        cosine, sine = position_embeddings
        rotary_dim = cosine.shape[-1]
        def normalize(q, k, group):
            for tensor, norm in ((q, self.q_norm), (k, self.k_norm)):
                if tensor is not None:
                    def rows(part, first, last):
                        part = norm(part)
                        rotated, tail = part[..., :rotary_dim], part[..., rotary_dim:]
                        a, b = rotated.chunk(2, -1)
                        rotated = rotated * cosine[:, first:last, None] + torch.cat((-b, a), -1) * sine[:, first:last, None]
                        return torch.cat((rotated, tail), -1)
                    attention_kit.rows_(tensor, rows)
        inputs, hidden_states = [hidden_states], None
        result = attention_kit.qkv_attention(inputs, self.q_proj, self.k_proj, self.v_proj, self.heads, self.head_dim, normalize,
                                             kv_heads=self.kv_heads, attention_mask=attention_mask, force_attention='sdpa')
        return self.o_proj(result.flatten(2) * gate), None


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        config._prompt_enhancer_safe_legacy = True
        self._interrupt = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        for layer in self.layers:
            if layer.layer_type == 'full_attention':
                layer.self_attn = Attention(config)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        width = config.num_attention_heads * config.head_dim
        self.split_linear_modules_map = {'q_gate_proj': {'mapped_modules': ['q_proj', 'gate_proj'], 'split_sizes': [width, width]}}

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        embeddings = self.rotary_emb(hidden, positions)
        mask = torch.ones((1, input_ids.shape[1], 1, input_ids.shape[1]), device=hidden.device, dtype=torch.bool).transpose(1, 2).tril().transpose(1, 2)
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=embeddings, attention_mask=mask if layer.layer_type == 'full_attention' else None)
            if self._interrupt:
                raise GenerationAborted
        return self.norm(hidden)


def encode_prompt(encoder, tokenizer, prompt):
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(formatted, truncation=True, max_length=4096)
    ids = torch.tensor([inputs.input_ids], device=runtime.execution_device(encoder), dtype=torch.long)
    with text_encoding_progress(encoder.layers):
        return encoder(ids)
