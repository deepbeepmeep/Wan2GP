"""Standalone MERT2 waveform-to-representation inference."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import AmplitudeToDB, MelScale, Spectrogram
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_mert2 import MERT2Config


@dataclass
class MERT2ModelOutput(ModelOutput):
    """Frame representations and their validity mask.

    ``hidden_states`` contains one tensor per Conformer block, without an input
    embedding. ``feature_attention_mask`` is True at valid output frames.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    feature_attention_mask: Optional[torch.BoolTensor] = None


class MERT2MelFrontend(nn.Module):
    """Power log-mel features with fixed, checkpoint-specific normalization."""

    def __init__(self, config):
        super().__init__()
        # Filter construction requires real tensors even during meta loading.
        with torch.device("cpu"):
            self.register_buffer("mel_mean", torch.zeros(config.num_mel_bins, dtype=torch.float32))
            self.register_buffer("mel_std", torch.ones(config.num_mel_bins, dtype=torch.float32))
            self.spectrogram = Spectrogram(
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                power=2.0,
            ).float()
            self.mel_scale = MelScale(
                n_mels=config.num_mel_bins,
                sample_rate=config.sampling_rate,
                n_stft=config.n_fft // 2 + 1,
            ).float()
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=None)

    def _apply(self, fn, recurse=True):
        # Preserve the original float32 values, including on model.half().
        saved = [(module, name, value) for module in self.modules() for name, value in module._buffers.items() if value is not None]
        super()._apply(fn, recurse=recurse)
        for module, name, original in saved:
            current = module._buffers[name]
            if original.is_floating_point():
                module._buffers[name] = (
                    current.float() if original.is_meta else original.to(device=current.device, dtype=torch.float32)
                )
        return self

    @torch.no_grad()
    def forward(self, waveform):
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = self.spectrogram(waveform.float())
            mel = self.amplitude_to_db(self.mel_scale(spectrum))
            mel = mel[..., :-1].transpose(-1, -2)
            return (mel - self.mel_mean) / self.mel_std.clamp_min(1e-5)


class Transpose(nn.Module):
    def forward(self, hidden_states):
        return hidden_states.transpose(1, 2)


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, hidden_states):
        magnitude = torch.norm(hidden_states, p=2, dim=1, keepdim=True)
        normalized = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)
        return self.weight * (hidden_states * normalized) + self.bias + hidden_states


class ConvNextLayer(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.depthwise_block = nn.Sequential(
            Transpose(), nn.Conv1d(dim, dim, 7, padding=3, groups=dim), Transpose()
        )
        self.pointwise_block = nn.Sequential(
            nn.LayerNorm(dim, eps=eps),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            GlobalResponseNorm(4 * dim),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, hidden_states):
        return hidden_states + self.pointwise_block(self.depthwise_block(hidden_states))


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, depth, eps):
        super().__init__()
        self.resampling_layer = (
            nn.Sequential(
                nn.LayerNorm(in_channels, eps=eps),
                Transpose(),
                nn.Conv1d(in_channels, out_channels, 2, stride=stride),
                Transpose(),
            )
            if in_channels != out_channels or stride > 1 else nn.Identity()
        )
        self.convnext_layers = nn.Sequential(*[ConvNextLayer(out_channels, eps) for _ in range(depth)])

    def forward(self, hidden_states):
        return self.convnext_layers(self.resampling_layer(hidden_states))


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.base = config.rotary_embedding_base
        with torch.device("cpu"):
            inverse_frequency = 1.0 / (
                self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
        self.register_buffer("inv_freq", inverse_frequency, persistent=False)
        self._sequence_length = 0
        self._cache_device = None
        self._cos = None
        self._sin = None

    def _apply(self, fn, recurse=True):
        original = self.inv_freq
        super()._apply(fn, recurse=recurse)
        self.inv_freq = original.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    def forward(self, hidden_states):
        length = hidden_states.shape[1]
        if self._cos is None or self._cache_device != hidden_states.device or length > self._sequence_length:
            positions = torch.arange(length, device=hidden_states.device, dtype=self.inv_freq.dtype)
            frequencies = torch.einsum("i,j->ij", positions, self.inv_freq)
            angles = torch.cat((frequencies, frequencies), dim=-1)
            self._cos = angles.cos()[:, None, None, :]
            self._sin = angles.sin()[:, None, None, :]
            self._sequence_length = length
            self._cache_device = hidden_states.device
        return (
            self._cos[:length].to(hidden_states.dtype).permute(1, 0, 2, 3),
            self._sin[:length].to(hidden_states.dtype).permute(1, 0, 2, 3),
        )


def rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, position_embeddings):
        batch, time, width = hidden_states.shape
        shape = (batch, time, self.num_heads, self.head_dim)
        query = self.query_proj(hidden_states).reshape(shape)
        key = self.key_proj(hidden_states).reshape(shape)
        value = self.value_proj(hidden_states).reshape(shape)
        cos, sin = position_embeddings
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        from shared.attention import pay_attention
        attended = pay_attention([query, key, value], force_attention="sdpa")
        return self.out_proj(attended.reshape(batch, time, width))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.w_2(F.gelu(self.w_1(hidden_states)))


class ConvolutionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        kernel = config.conv_depthwise_kernel_size
        self.layer_norm = nn.LayerNorm(width, eps=config.layer_norm_eps)
        self.conv_block = nn.Sequential(
            Transpose(),
            nn.Conv1d(width, 2 * width, 1, bias=False),
            nn.GLU(dim=1),
            nn.Conv1d(width, width, kernel, padding=(kernel - 1) // 2, groups=width, bias=False),
            nn.Sequential(Transpose(), nn.LayerNorm(width, eps=config.layer_norm_eps), Transpose()),
            nn.GELU(),
            nn.Conv1d(width, width, 1, bias=False),
            Transpose(),
        )

    def forward(self, hidden_states):
        return self.conv_block(self.layer_norm(hidden_states))


class ConformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, eps = config.hidden_size, config.layer_norm_eps
        self.ffn1_layer_norm = nn.LayerNorm(width, eps=eps)
        self.ffn1 = FeedForward(config)
        self.attn_layer_norm = nn.LayerNorm(width, eps=eps)
        self.attn = SelfAttention(config)
        self.conv_module = ConvolutionModule(config)
        self.ffn2_layer_norm = nn.LayerNorm(width, eps=eps)
        self.ffn2 = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(width, eps=eps)

    def forward(self, hidden_states, position_embeddings):
        hidden_states = hidden_states + 0.5 * self.ffn1(self.ffn1_layer_norm(hidden_states))
        hidden_states = self.attn(self.attn_layer_norm(hidden_states), position_embeddings) + hidden_states
        hidden_states = self.conv_module(hidden_states) + hidden_states
        hidden_states = hidden_states + 0.5 * self.ffn2(self.ffn2_layer_norm(hidden_states))
        return self.final_layer_norm(hidden_states)


class MERT2Model(PreTrainedModel):
    """Encode mono waveforms into 25 Hz MERT2 frame representations.

    Inputs are floating-point mono waveforms sampled at ``config.sampling_rate``.
    Do not standardize their amplitude. An optional waveform attention mask must
    contain a prefix of ones followed by zeros. Without a mask, every input
    sample, including any caller-provided silence or padding, is processed.
    """

    config_class = MERT2Config
    base_model_prefix = ""
    main_input_name = "input_values"
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _no_split_modules = ["ConvNextBlock", "ConformerBlock"]

    def __init__(self, config):
        super().__init__(config)
        if config._attn_implementation not in {"sdpa", "flash_attention_2"}:
            raise ValueError("MERT2 supports attn_implementation='sdpa' or 'flash_attention_2'.")
        self.feature_extractor = MERT2MelFrontend(config)
        channels = [config.num_mel_bins] + config.subsampling_channels
        self.subsampling_module = nn.Sequential(*[
            ConvNextBlock(channels[i], channels[i + 1], (1, 2, 2)[i], config.subsampling_depths[i], config.subsampling_layer_norm_eps)
            for i in range(3)
        ])
        self.layers = nn.ModuleList([ConformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.embed_positions = RotaryEmbedding(config)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _get_feat_extract_output_lengths(self, input_lengths):
        return input_lengths // self.config.inputs_to_logits_ratio

    def _encode(self, input_values, output_hidden_states):
        mel = self.feature_extractor(input_values)
        input_dtype = self.subsampling_module[0].convnext_layers[0].depthwise_block[1].weight.dtype
        hidden = self.subsampling_module(mel.to(dtype=input_dtype))
        positions = self.embed_positions(hidden)
        states = [] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(hidden, positions)
            if states is not None:
                states.append(hidden)
        return hidden, tuple(states) if states is not None else None

    def forward(self, input_values, attention_mask=None, output_hidden_states=None, return_dict=None):
        """Return final frames, optionally all block states, and a frame mask.

        Different valid waveform lengths are encoded separately so convolution
        and global normalization never incorporate another sample's padding.
        Outputs are zero-padded to the longest valid feature sequence.
        """
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if input_values.ndim != 2 or not input_values.is_floating_point() or input_values.shape[0] == 0:
            raise ValueError("input_values must be a nonempty floating-point tensor of shape [batch, samples].")
        if input_values.shape[1] < self.config.minimum_input_samples:
            raise ValueError(f"Each waveform must contain at least {self.config.minimum_input_samples} samples.")
        if not torch.isfinite(input_values).all():
            raise ValueError("input_values must contain only finite samples.")
        batch, samples = input_values.shape
        if attention_mask is None:
            hidden, states = self._encode(input_values, output_hidden_states)
            feature_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            if attention_mask.shape != input_values.shape:
                raise ValueError("attention_mask must have the same shape as input_values.")
            attention_mask = attention_mask.to(device=input_values.device)
            if not ((attention_mask == 0) | (attention_mask == 1)).all():
                raise ValueError("attention_mask must contain only zeros and ones.")
            mask = attention_mask.bool()
            lengths = mask.sum(dim=1)
            expected = torch.arange(samples, device=input_values.device)[None, :] < lengths[:, None]
            if not torch.equal(mask, expected):
                raise ValueError("attention_mask must be right padded: valid samples followed by padding.")
            if (lengths < self.config.minimum_input_samples).any():
                raise ValueError(f"Each waveform must contain at least {self.config.minimum_input_samples} valid samples.")
            feature_lengths = self._get_feat_extract_output_lengths(lengths)
            maximum = int(feature_lengths.max().item())
            feature_mask = torch.arange(maximum, device=input_values.device)[None, :] < feature_lengths[:, None]
            hidden, state_values = None, None
            for length in torch.unique(lengths, sorted=True).tolist():
                indices = torch.where(lengths == length)[0]
                group_hidden, group_states = self._encode(input_values.index_select(0, indices)[:, :length], output_hidden_states)
                padding = (0, 0, 0, maximum - group_hidden.shape[1])
                if hidden is None:
                    hidden = group_hidden.new_zeros(batch, maximum, self.config.hidden_size)
                    if output_hidden_states:
                        state_values = [torch.zeros_like(hidden) for _ in self.layers]
                hidden = hidden.index_copy(0, indices, F.pad(group_hidden, padding))
                if state_values is not None:
                    for i, value in enumerate(group_states):
                        state_values[i] = state_values[i].index_copy(0, indices, F.pad(value, padding))
            states = tuple(state_values) if state_values is not None else None
        output = MERT2ModelOutput(last_hidden_state=hidden, hidden_states=states, feature_attention_mask=feature_mask)
        return output if return_dict else output.to_tuple()


MERT2Model.register_for_auto_class("AutoModel")
