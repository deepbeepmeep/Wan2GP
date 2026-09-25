"""Configuration for MERT2 music representation models."""

from transformers import PretrainedConfig


class MERT2Config(PretrainedConfig):
    """Architecture shared by the 30-second and full-song MERT2 encoders."""

    model_type = "mert2"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_mel_bins=128,
        sampling_rate=24000,
        n_fft=2048,
        win_length=2048,
        hop_length=240,
        subsampling_channels=None,
        subsampling_depths=None,
        conv_depthwise_kernel_size=31,
        rotary_embedding_base=10000,
        layer_norm_eps=1e-5,
        subsampling_layer_norm_eps=1e-6,
        initializer_range=0.02,
        variant="30s",
        context_seconds=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_mel_bins = int(num_mel_bins)
        self.sampling_rate = int(sampling_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.subsampling_channels = list(subsampling_channels or [num_mel_bins, 512, hidden_size])
        self.subsampling_depths = list(subsampling_depths or [3, 4, 5])
        self.conv_depthwise_kernel_size = int(conv_depthwise_kernel_size)
        self.rotary_embedding_base = int(rotary_embedding_base)
        self.layer_norm_eps = float(layer_norm_eps)
        self.subsampling_layer_norm_eps = float(subsampling_layer_norm_eps)
        self.initializer_range = float(initializer_range)
        self.variant = str(variant)
        self.context_seconds = float(context_seconds if context_seconds is not None else (360 if variant == "fs" else 30))
        self.inputs_to_logits_ratio = self.hop_length * 4
        self.frame_rate = self.sampling_rate / self.inputs_to_logits_ratio
        self.minimum_input_samples = self.n_fft // 2 + 1

        if min(self.hidden_size, self.intermediate_size, self.num_hidden_layers, self.num_attention_heads) <= 0:
            raise ValueError("Encoder dimensions and layer counts must be positive.")
        if self.hidden_size % self.num_attention_heads or (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("hidden_size must divide into an even head dimension.")
        if len(self.subsampling_channels) != 3 or len(self.subsampling_depths) != 3:
            raise ValueError("The subsampler must contain three channel widths and three depths.")
        if self.subsampling_channels[0] != self.num_mel_bins or self.subsampling_channels[-1] != self.hidden_size:
            raise ValueError("Subsampling widths must start at num_mel_bins and end at hidden_size.")
        if min(*self.subsampling_channels, *self.subsampling_depths, self.num_mel_bins, self.sampling_rate, self.hop_length) <= 0:
            raise ValueError("Frontend dimensions and sampling parameters must be positive.")
        if self.n_fft < self.win_length or self.win_length <= 0 or self.n_fft % 2:
            raise ValueError("n_fft must be even and at least win_length > 0.")
        if self.conv_depthwise_kernel_size <= 0 or self.conv_depthwise_kernel_size % 2 != 1:
            raise ValueError("conv_depthwise_kernel_size must be positive and odd.")
        if self.rotary_embedding_base <= 0 or min(self.layer_norm_eps, self.subsampling_layer_norm_eps) <= 0:
            raise ValueError("Rotary base and normalization epsilons must be positive.")
        if self.variant not in {"30s", "fs"}:
            raise ValueError("variant must be '30s' or 'fs'.")


MERT2Config.register_for_auto_class()
