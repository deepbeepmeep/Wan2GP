"""Hugging Face SheetSage2 model with a shared MERT-v2 encoder."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BartConfig, PreTrainedModel
from transformers.models.bart.modeling_bart import BartDecoder
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.utils import ModelOutput

from .configuration_mert2 import MERT2Config
from .modeling_mert2 import MERT2Model
from .configuration_sheetsage2 import SheetSage2Config
from .tokenization_sheetsage2 import SheetSage2Tokenizer


PROJECTIONS = ("query_proj", "key_proj", "value_proj", "out_proj")
@dataclass
class SheetSage2EncoderOutput(ModelOutput):
    """Frame features. Block states exclude the separately returned input state."""

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    backbone_last_hidden_state: Optional[torch.FloatTensor] = None
    mixed_hidden_state: Optional[torch.FloatTensor] = None
    input_hidden_state: Optional[torch.FloatTensor] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    feature_attention_mask: Optional[torch.BoolTensor] = None

    @property
    def last_hidden_state(self):
        return self.encoder_last_hidden_state


@dataclass
class SheetSage2Output(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    backbone_last_hidden_state: Optional[torch.FloatTensor] = None
    mixed_hidden_state: Optional[torch.FloatTensor] = None
    input_hidden_state: Optional[torch.FloatTensor] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    feature_attention_mask: Optional[torch.BoolTensor] = None


class AttentionAdapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.lora_A = nn.Linear(width, rank, bias=False)
        self.lora_B = nn.Linear(rank, width, bias=False)


class EncoderAdapters(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(config.backbone_config["num_hidden_layers"]):
            layer = nn.Module()
            layer.attn = nn.Module()
            for name in PROJECTIONS:
                setattr(layer.attn, name, AttentionAdapter(config.backbone_config["hidden_size"], config.lora_rank))
            self.layers.append(layer)


class SheetSage2Model(PreTrainedModel):
    """Audio-to-symbolic model, with optional frame and decoder representations.

    WanGP loads the merged checkpoint through MMGP.
    ``forward`` returns raw vocabulary logits; grammar-masked generation scores
    are available separately through ``generate(output_scores=True)``.
    """

    config_class = SheetSage2Config
    base_model_prefix = ""
    main_input_name = "input_values"
    _supports_sdpa = True
    _tied_weights_keys = ["decoder.embed_tokens.weight", "output_projection.weight"]
    _no_split_modules = ["ConformerBlock", "BartDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.hparams = SimpleNamespace(input_audio_length=config.input_audio_length, time_hz=config.time_hz)
        self.max_output_seq_len = config.max_output_seq_len
        self.tokenizer = SheetSage2Tokenizer(
            config.input_audio_length, config.time_hz, config.tokenizer_schema_version,
            expected_fingerprint=config.tokenizer_fingerprint,
        )
        if self.tokenizer.n_tokens != config.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match the model.")
        if config.weights_format == "adapter":
            self.encoder = None
            self.adapter = EncoderAdapters(config)
        else:
            ec = MERT2Config(**config.backbone_config)
            ec._attn_implementation = config.encoder_attn_implementation
            self.encoder = MERT2Model(ec)
            self.adapter = None
        self.layer_weight = nn.Parameter(torch.zeros(config.backbone_config["num_hidden_layers"] + 1))
        self.encoder_projection = nn.Linear(config.backbone_config["hidden_size"], config.hidden_size)
        dc = BartConfig(
            vocab_size=config.vocab_size, d_model=config.hidden_size,
            decoder_layers=config.decoder_layers, decoder_attention_heads=config.num_attention_heads,
            decoder_ffn_dim=config.intermediate_size, max_position_embeddings=config.max_output_seq_len,
            dropout=config.decoder_dropout, attention_dropout=config.decoder_dropout,
            activation_dropout=config.decoder_dropout, activation_function="gelu",
            pad_token_id=config.pad_token_id, bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id, is_encoder_decoder=True, use_cache=True,
        )
        dc._attn_implementation = "sdpa"
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.decoder = BartDecoder(dc, embed_tokens=self.token_embedding)
        self.decoder.gradient_checkpointing_disable()
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self.lora_merged = config.weights_format == "merged"
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value
        self.decoder.embed_tokens.weight = value.weight

    def tie_weights(self):
        super().tie_weights()
        if hasattr(self, "decoder") and hasattr(self, "token_embedding"):
            self.decoder.embed_tokens.weight = self.token_embedding.weight

    def get_output_embeddings(self):
        return self.output_projection

    def set_output_embeddings(self, value):
        self.output_projection = value

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @torch.no_grad()
    def merge_lora(self):
        if self.lora_merged:
            return self
        if self.encoder is None or self.adapter is None:
            raise RuntimeError("Load both the MERT-v2 parent and adapters before merging.")
        scale = self.config.lora_alpha / self.config.lora_rank
        with torch.autocast("cpu", enabled=False):
            for layer, adapter in zip(self.encoder.layers, self.adapter.layers):
                for name in PROJECTIONS:
                    projection = getattr(layer.attn, name)
                    update = getattr(adapter.attn, name)
                    values = (projection.weight, update.lora_A.weight, update.lora_B.weight)
                    if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in values):
                        raise ValueError("Merge adapters on CPU with float32 parameters.")
                    projection.weight.add_((update.lora_B.weight @ update.lora_A.weight) * scale)
        self.adapter = None
        self.lora_merged = True
        self.config.weights_format = "merged"
        return self



    def _prepare_audio(self, input_values, attention_mask=None):
        if self.encoder is None or not self.lora_merged:
            raise RuntimeError("Load the model with from_pretrained before inference.")
        if input_values.ndim != 2 or input_values.shape[0] < 1 or not input_values.is_floating_point():
            raise ValueError("input_values must be floating-point [batch, samples].")
        if not torch.isfinite(input_values).all():
            raise ValueError("Audio must contain only finite samples.")
        batch, samples = input_values.shape
        minimum = self.encoder.config.minimum_input_samples
        window = round(self.config.input_audio_length * self.config.sampling_rate)
        if samples < minimum:
            raise ValueError(f"Each waveform must contain at least {minimum} samples.")
        if samples > window:
            raise ValueError("Audio exceeds one model window; use transcribe for whole songs.")
        if attention_mask is None:
            lengths = torch.full((batch,), samples, dtype=torch.long, device=input_values.device)
        else:
            if attention_mask.shape != input_values.shape:
                raise ValueError("attention_mask must have the same shape as input_values.")
            mask = attention_mask.to(device=input_values.device)
            if not ((mask == 0) | (mask == 1)).all():
                raise ValueError("attention_mask must contain zeros and ones.")
            mask = mask.bool()
            lengths = mask.sum(1)
            expected = torch.arange(samples, device=input_values.device)[None] < lengths[:, None]
            if not torch.equal(mask, expected) or (lengths < minimum).any():
                raise ValueError("attention_mask must identify a nonempty right-padded waveform.")
            input_values = input_values.masked_fill(~mask, 0)
        # The encoder attends to the complete fixed window, including this silence.
        input_values = F.pad(input_values.float(), (0, window - samples))
        stride = self.encoder.config.inputs_to_logits_ratio
        if window % stride:
            input_values = F.pad(input_values, (0, stride - window % stride))
        return input_values, lengths

    def get_audio_features(self, input_values, attention_mask=None, output_hidden_states=None, return_dict=True):
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        waveform, lengths = self._prepare_audio(input_values, attention_mask)
        mel = self.encoder.feature_extractor(waveform)
        weight_dtype = self.encoder.subsampling_module[0].convnext_layers[0].depthwise_block[1].weight.dtype
        hidden = self.encoder.subsampling_module(mel.to(dtype=weight_dtype))
        input_hidden = hidden if output_hidden_states else None
        weights = torch.softmax(self.layer_weight, dim=0)
        mixed = hidden * weights[0]
        positions = self.encoder.embed_positions(hidden)
        states = [] if output_hidden_states else None
        for weight, layer in zip(weights[1:], self.encoder.layers):
            hidden = layer(hidden, positions)
            mixed = mixed + hidden * weight
            if states is not None:
                states.append(hidden)
        memory = self.encoder_projection(mixed)
        stride = self.encoder.config.inputs_to_logits_ratio
        frame_mask = torch.arange(memory.shape[1], device=memory.device)[None] < ((lengths + stride - 1) // stride)[:, None]
        output = SheetSage2EncoderOutput(
            encoder_last_hidden_state=memory, backbone_last_hidden_state=hidden,
            mixed_hidden_state=mixed, input_hidden_state=input_hidden,
            backbone_hidden_states=tuple(states) if states is not None else None,
            feature_attention_mask=frame_mask,
        )
        return output if return_dict else output.to_tuple()

    def encode(self, audio):
        return self.get_audio_features(audio).last_hidden_state

    def _decode(self, memory, decoder_input_ids, use_cache=False, past_key_values=None, output_hidden_states=False):
        attention_mask = None if past_key_values is not None else decoder_input_ids != self.tokenizer.pad_token
        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        output = self.decoder(
            input_ids=decoder_input_ids, attention_mask=attention_mask,
            encoder_hidden_states=memory, encoder_attention_mask=None,
            past_key_values=past_key_values, use_cache=use_cache,
            output_hidden_states=output_hidden_states, return_dict=True,
        )
        return self.output_projection(output.last_hidden_state), output

    def decode(self, memory, decoder_input_ids, use_cache=False, past_key_values=None):
        logits, output = self._decode(memory, decoder_input_ids, use_cache, past_key_values)
        return logits, output.past_key_values

    def forward(self, input_values=None, decoder_input_ids=None, attention_mask=None,
                encoder_outputs=None, past_key_values=None, use_cache=None,
                output_hidden_states=None, return_dict=None):
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if decoder_input_ids is None:
            raise ValueError("decoder_input_ids is required for logits; use generate to transcribe audio.")
        if encoder_outputs is None:
            if input_values is None:
                raise ValueError("Provide input_values or encoder_outputs.")
            encoder_outputs = self.get_audio_features(input_values, attention_mask, output_hidden_states)
        if torch.is_tensor(encoder_outputs):
            encoder_outputs = SheetSage2EncoderOutput(encoder_last_hidden_state=encoder_outputs)
        logits, decoded = self._decode(encoder_outputs.last_hidden_state, decoder_input_ids,
                                       use_cache, past_key_values, output_hidden_states)
        output = SheetSage2Output(
            logits=logits, past_key_values=decoded.past_key_values,
            decoder_hidden_states=decoded.hidden_states,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            backbone_last_hidden_state=encoder_outputs.backbone_last_hidden_state if output_hidden_states else None,
            mixed_hidden_state=encoder_outputs.mixed_hidden_state if output_hidden_states else None,
            input_hidden_state=encoder_outputs.input_hidden_state if output_hidden_states else None,
            backbone_hidden_states=encoder_outputs.backbone_hidden_states if output_hidden_states else None,
            feature_attention_mask=encoder_outputs.feature_attention_mask,
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        return output if return_dict else output.to_tuple()

    def generate(self, input_values, **kwargs):
        """Generate grammar-constrained symbolic tokens with autoregressive caching."""
        from .generation_sheetsage2 import generate
        return generate(self, input_values, **kwargs)

    def transcribe(self, audio, output_dir=None, *, melody_only=False, **kwargs):
        """Return transcription in memory; set output_dir to also save files.

        Accepts a path, encoded audio bytes, binary stream, or waveform with
        sampling_rate. Returns ABC text, MIDI bytes, timed events, and optional
        per-window CPU tensors. With output_dir, optional tensors are saved
        instead of retained in memory. Set melody_only=True to retain both vocal
        and instrumental melodies while omitting chords from ABC and playback;
        raw predicted annotations remain available. The default keeps full
        transcription. See the model card for rendering options.
        """
        from .pipeline_sheetsage2 import transcribe
        return transcribe(self, audio, output_dir=output_dir, melody_only=melody_only, **kwargs)


SheetSage2ForConditionalGeneration = SheetSage2Model
SheetSage2Model.register_for_auto_class("AutoModel")
