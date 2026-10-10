"""Qwen2.5-Omni feature prefill, reusing shared Qwen layers in PyTorch mode.

No decode cache, CUDA graphs, Triton MLP/norm or FlashAttention are used.
Shared INT8 linear kernels remain available.
"""

import torch
import torch.nn.functional as F
from types import SimpleNamespace
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen2_5OmniThinkerConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder, Qwen2_5OmniThinkerForConditionalGeneration

from shared.llm_engines.nanovllm.models.qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLTextRotaryEmbedding, clear_qwen25_vl_runtime_caches
from shared.llm_engines.nanovllm.layers.embed_head import VocabParallelEmbedding
from shared.llm_engines.nanovllm.layers.layernorm import RMSNorm


class Qwen2_5OmniPrefillAttention(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, query, key, value):
        # A single unpadded conditioning sequence needs no cache, paging or host-side length reads.
        query, key, value = [tensor.transpose(0, 1).unsqueeze(0) for tensor in (query, key, value)]
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=self.scale, enable_gqa=True)
        return output.squeeze(0).transpose(0, 1)


class Qwen2_5OmniFeatureTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2_5_VLDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        for layer in self.layers:
            attention = layer.self_attn.attn
            layer.self_attn.attn = Qwen2_5OmniPrefillAttention(attention.scale)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLTextRotaryEmbedding(config)


class Qwen2_5OmniFeatureEncoder(nn.Module):
    _no_split_modules = ['Qwen2_5_VLDecoderLayer', 'Qwen2_5OmniAudioEncoderLayer']
    get_audio_features = Qwen2_5OmniThinkerForConditionalGeneration.get_audio_features

    @classmethod
    def from_config(cls, config):
        thinker = dict(config['thinker_config'])
        text = dict(thinker['text_config'])
        rope_scaling = text.pop('rope_scaling')
        # Transformers 4.54 validates this Omni field as ordinary text RoPE. Its own
        # default constructor installs MRoPE after validation; preserve that ordering.
        thinker['text_config'] = text
        thinker = Qwen2_5OmniThinkerConfig(**thinker)
        thinker.text_config.rope_scaling = rope_scaling
        return cls(SimpleNamespace(thinker_config=thinker))

    def __init__(self, config):
        super().__init__()
        self.config = config.thinker_config
        self.config.audio_config._attn_implementation = 'sdpa'
        self.audio_tower = Qwen2_5OmniAudioEncoder(self.config.audio_config)
        self.model = Qwen2_5OmniFeatureTextModel(self.config.text_config)
        self._interrupt_check = lambda: False
        for layer in self.model.layers:
            layer.input_layernorm.use_triton_rmsnorm = False
            layer.post_attention_layernorm.use_triton_rmsnorm = False
            layer.mlp.act_fn.use_triton = False
        self.model.norm.use_triton_rmsnorm = False

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, layer_weights, layer_scale, input_features=None, feature_attention_mask=None):
        hidden = self.model.embed_tokens(input_ids)
        if input_features is not None:
            with sdpa_kernel(SDPBackend.MATH):
                audio = self.get_audio_features(input_features.to(self.audio_tower.dtype), feature_attention_mask)
            hidden = hidden.masked_scatter((input_ids == self.config.audio_token_id).unsqueeze(-1), audio.to(hidden.dtype))
        # Audio-only Thinker uses the ordinary cumulative token positions on all MRoPE axes.
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        shape = hidden.shape
        hidden = hidden.flatten(0, 1)
        position_embeddings = self.model.rotary_emb(hidden, positions.flatten())
        weights = layer_weights.to(device=hidden.device, dtype=torch.float32).softmax(0)
        scale = layer_scale.to(device=hidden.device, dtype=torch.float32)
        fused = torch.zeros_like(hidden, dtype=torch.float32)
        try:
            for index, layer in enumerate(self.model.layers):
                hidden = layer(hidden, position_embeddings)
                if index == len(self.model.layers) - 1:
                    hidden = self.model.norm(hidden)
                fused.add_(F.layer_norm(hidden, [hidden.shape[-1]]).float() * weights[index])
                if self._interrupt_check():
                    raise InterruptedError('AuK conditioning interrupted')
            return fused.mul_(scale).view(shape)
        finally:
            clear_qwen25_vl_runtime_caches(hidden.device)
