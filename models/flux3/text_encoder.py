# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
from torch import nn
from mmgp import runtime
from shared import attention_kit
from shared.utils.phase_progress import GenerationAborted, text_encoding_progress
from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig
from models.ideogram4.qwen3_vl_transformers import Qwen3VLTextAttention, Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding, _apply_rotary_inplace

OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


class TextAttention(Qwen3VLTextAttention):
    def forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        values = hidden_states if isinstance(hidden_states, list) else [hidden_states]
        hidden_states = None
        cos, sin = position_embeddings

        def normalize(q, k, group):
            for tensor, norm in ((q, self.q_norm), (k, self.k_norm)):
                if tensor is not None:
                    def apply(part, first, last):
                        return _apply_rotary_inplace(norm(part), cos[:, first:last], sin[:, first:last], 2)
                    attention_kit.rows_(tensor, apply)

        result = attention_kit.qkv_attention(values, self.q_proj, self.k_proj, self.v_proj, self.config.num_attention_heads, self.head_dim, normalize, kv_heads=self.config.num_key_value_heads, attention_mask=attention_mask)
        return self.o_proj(result.flatten(2)), None


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        # The released conditioner reads states before layers 5,9,...,33. The final
        # four layers, final norm, visual tower and LM head never contribute.
        for index in range(max(OUTPUT_LAYERS)):
            layer = Qwen3VLTextDecoderLayer(config, index)
            layer.self_attn = TextAttention(config, index)
            self.layers.append(layer)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self._interrupt = False

    def forward(self, input_ids, mask):
        model = self.language_model
        h = model.embed_tokens(input_ids)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        frequencies = model.rotary_emb(h, positions)
        length = h.shape[1]
        indexes = torch.arange(length, device=h.device)
        causal_mask = (indexes[:, None] >= indexes[None, :])[None, :, None, :] & mask[:, None, None, :].bool()
        output = h.new_empty((length, len(OUTPUT_LAYERS), h.shape[-1]))
        with text_encoding_progress(model.layers):
            for index, layer in enumerate(model.layers, 1):
                h = layer(h, position_embeddings=frequencies, attention_mask=causal_mask)
                if self._interrupt:
                    raise GenerationAborted
                if index in OUTPUT_LAYERS:
                    output[:, OUTPUT_LAYERS.index(index)].copy_(h[0])
        return output.flatten(1)


def encode_prompt(model, tokenizer, prompt):
    prompt = '\n'.join(line for line in prompt.splitlines() if not line.lstrip().startswith('#'))
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    with torch.device('cpu'):
        real = len(tokenizer(text, truncation=True, max_length=4096)['input_ids'])
        length = min(math.ceil(real / 80) * 80, 4096)
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=length)
    device = runtime.execution_device(model)
    return model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
