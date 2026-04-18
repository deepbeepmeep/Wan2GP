from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from .configuration_acestep_v15 import AceStepConfig
from .modeling_acestep_v15_turbo import (
    AceStepAudioTokenizer as _BaseAceStepAudioTokenizer,
    AceStepConditionEncoder as _BaseAceStepConditionEncoder,
    AceStepConditionGenerationModel as _BaseAceStepConditionGenerationModel,
    AceStepDiTModel as _BaseAceStepDiTModel,
    AceStepPreTrainedModel,
    AceStepTimbreEncoder as _BaseAceStepTimbreEncoder,
    AudioTokenDetokenizer,
    BaseModelOutput,
    FlashAttentionKwargs,
    Unpack,
    can_return_tuple,
    create_4d_mask,
)


class AceStepTimbreEncoder(_BaseAceStepTimbreEncoder):
    @can_return_tuple
    def forward(
        self,
        refer_audio_acoustic_hidden_states_packed: Optional[torch.FloatTensor] = None,
        refer_audio_order_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput:
        inputs_embeds = refer_audio_acoustic_hidden_states_packed
        inputs_embeds = self.embed_tokens(inputs_embeds)
        inputs_embeds = torch.cat([self.special_token.expand(inputs_embeds.shape[0], 1, -1), inputs_embeds], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    torch.ones(attention_mask.shape[0], 1, device=attention_mask.device, dtype=attention_mask.dtype),
                    attention_mask,
                ],
                dim=1,
            )
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)

        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        is_flash_attn = self.config._attn_implementation == "flash_attention_2"
        full_attn_mask = None
        sliding_attn_mask = None

        if is_flash_attn:
            full_attn_mask = attention_mask
            sliding_attn_mask = attention_mask if self.config.use_sliding_window else None
        else:
            full_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False,
            )
            if self.config.use_sliding_window:
                sliding_attn_mask = create_4d_mask(
                    seq_len=seq_len,
                    dtype=dtype,
                    device=device,
                    attention_mask=attention_mask,
                    sliding_window=self.config.sliding_window,
                    is_sliding_window=True,
                    is_causal=False,
                )

        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
        }
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_module in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states[:, 0, :]
        timbre_embs_unpack, timbre_embs_mask = self.unpack_timbre_embeddings(hidden_states, refer_audio_order_mask)
        return timbre_embs_unpack, timbre_embs_mask


class AceStepAudioTokenizer(_BaseAceStepAudioTokenizer):
    @can_return_tuple
    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput:
        hidden_states = self.audio_acoustic_proj(hidden_states)
        hidden_states = self.attention_pooler(hidden_states)
        input_dtype = hidden_states.dtype
        quantized, indices = self.quantizer(hidden_states.float())
        quantized = quantized.to(input_dtype)
        return quantized, indices


class AceStepDiTModel(_BaseAceStepDiTModel):
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        condition_dim = getattr(config, "encoder_hidden_size", None) or config.hidden_size
        self.condition_embedder = nn.Linear(condition_dim, config.hidden_size, bias=True)


class AceStepConditionEncoder(_BaseAceStepConditionEncoder):
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        self.timbre_encoder = AceStepTimbreEncoder(config)


class AceStepConditionGenerationModel(_BaseAceStepConditionGenerationModel):
    def __init__(self, config: AceStepConfig):
        AceStepPreTrainedModel.__init__(self, config)
        self.config = config
        self.decoder = AceStepDiTModel(config)
        encoder_config = copy.deepcopy(config)
        encoder_hidden_size = getattr(config, "encoder_hidden_size", None) or config.hidden_size
        encoder_config.hidden_size = encoder_hidden_size
        encoder_config.intermediate_size = getattr(config, "encoder_intermediate_size", None) or config.intermediate_size
        encoder_config.num_attention_heads = getattr(config, "encoder_num_attention_heads", None) or config.num_attention_heads
        encoder_config.num_key_value_heads = getattr(config, "encoder_num_key_value_heads", None) or config.num_key_value_heads
        self.encoder = AceStepConditionEncoder(encoder_config)
        self.tokenizer = AceStepAudioTokenizer(encoder_config)
        self.detokenizer = AudioTokenDetokenizer(encoder_config)
        self.null_condition_emb = nn.Parameter(torch.randn(1, 1, encoder_config.hidden_size))
        self.post_init()
