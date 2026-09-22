"""Prefill-only Qwen3-VL encoder using locally vendored 5.17 model sources."""
from types import SimpleNamespace

import torch
from torch import nn

from shared.utils.phase_progress import check_abort
from .positions import Qwen3VLModel
from .text_layers import Qwen3VLVisionModel, Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding, Qwen3VLTextRMSNorm


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3VLTextDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

    def forward(self, input_ids, position_ids, visual_mask, deepstack, image_embeds=None):
        hidden = self.embed_tokens(input_ids)
        if image_embeds is not None:
            hidden = hidden.masked_scatter(visual_mask.unsqueeze(-1), image_embeds.to(hidden.dtype))
        positions = self.rotary_emb(hidden, position_ids)
        for i, layer in enumerate(self.layers):
            check_abort()
            hidden = layer(hidden, position_embeddings=positions)
            if i < len(deepstack):
                hidden = hidden.clone()
                hidden[visual_mask] += deepstack[i].to(hidden.dtype)
        # Qwen Image 2.1 consumes the final layer BEFORE the final RMSNorm.
        return self.norm(hidden)


class Qwen3VLForConditionalGeneration(nn.Module, Qwen3VLModel):
    def __init__(self, config):
        super().__init__()
        text = dict(config["text_config"])
        vision = dict(config["vision_config"])
        if "rope_parameters" not in text:
            text["rope_parameters"] = {**text["rope_scaling"], "rope_theta": text["rope_theta"]}
        vision["rope_parameters"] = {"rope_type": "axial", "rope_theta": 10000.0}
        vision["num_attention_heads"] = vision["num_heads"]
        text = SimpleNamespace(**text)
        vision = SimpleNamespace(**vision)
        self.config = SimpleNamespace(**{**config, "text_config": text, "vision_config": vision})
        self.model = nn.Module()
        self.model.visual = Qwen3VLVisionModel(vision)
        self.model.language_model = TextModel(text)

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, **kwargs):
        image_mask = input_ids == self.config.image_token_id
        types = image_mask.to(torch.int32)
        positions, _ = self.get_rope_index(input_ids, types, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        deepstack = []
        image_embeds = None
        if pixel_values is not None:
            vision = self.model.visual(pixel_values, image_grid_thw)
            image_embeds = vision.pooler_output
            deepstack = vision.deepstack_features
        check_abort()
        hidden = self.model.language_model(input_ids, positions, image_mask, deepstack, image_embeds)
        return SimpleNamespace(hidden_states=(hidden,))
