import os
import copy
import logging

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config
from transformers import Wav2Vec2Model as Wav2Vec2Model_base
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2SamePadLayer, Wav2Vec2PositionalConvEmbedding
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from safetensors.torch import load_file

from .torch_utils import linear_interpolation


def _Wav2Vec2PositionalConvEmbedding_init_hack_(self, config):
        super(Wav2Vec2PositionalConvEmbedding, self).__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

Wav2Vec2PositionalConvEmbedding.__init__ = _Wav2Vec2PositionalConvEmbedding_init_hack_


class Wav2Vec2ModelWrapper(nn.Module):
    def __init__(self, config_path, device='cpu', prefix='wav2vec2.'):
        super(Wav2Vec2ModelWrapper, self).__init__()

        config, model_kwargs = Wav2Vec2Config.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                force_download=False,
                local_files_only=True,
            )

        model_path = os.path.join(config_path, 'model.safetensors')
        if not os.path.exists(model_path):
            model_path = os.path.join(config_path, 'pytorch_model.bin')

        if model_path.endswith(".safetensors"):
            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(model_path, map_location="cpu")

        config.name_or_path = config_path
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        try:
            config = Wav2Vec2Mode._autoset_attn_implementation(config, use_flash_attention_2=False)
        except TypeError:
            config = Wav2Vec2Mode._autoset_attn_implementation(config)

        # init model
        with torch.device('meta'):
            model = Wav2Vec2Mode(config)

        # load checkpoint
        logging.info(f'loading {model_path}')
        if prefix is not None:
            state_dict = {i.replace(prefix, ''):state_dict[i] for i in state_dict}
        
        model.tie_weights()
        m, u = model.load_state_dict(state_dict, assign=True, strict=False)
            
        model.tie_weights()
        model.eval()

        self.model = model
    
    @property
    def feature_extractor(self):
        return self.model.feature_extractor

    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.model(
            input_values,
            seq_len,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return self.model.feature_extract(
            input_values,
            seq_len
        )

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return self.model.encode(
            extract_features,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# the implementation of Wav2Vec2Model is borrowed from
# https://github.com/huggingface/transformers/blob/HEAD/src/transformers/models/wav2vec2/modeling_wav2vec2.py
# initialize our encoder with the pre-trained wav2vec 2.0 weights.
class Wav2Vec2Mode(Wav2Vec2Model_base):
    def __init__(self, config: Wav2Vec2Config):
        config.attn_implementation = "eager"
        super().__init__(config)


    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config._attn_implementation = "eager"
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
            

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
