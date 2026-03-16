from typing import List

from transformers.configuration_utils import PretrainedConfig


class HeartCodecConfig(PretrainedConfig):
    model_type = "heartcodec"

    def __init__(
        self,
        dim: int = 512,
        codebook_size: int = 8192,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = False,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        norm_type: str = "ada_norm_single",
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
        num_bands: int = 1,
        sample_rate: int = 48000,
        causal: bool = True,
        num_samples: int = 2,
        downsample_factors: List[int] = [3, 4, 4, 4, 5],
        downsample_kernel_sizes: List[int] = [6, 8, 8, 8, 10],
        upsample_factors: List[int] = [5, 4, 4, 4, 3],
        upsample_kernel_sizes: List[int] = [10, 8, 8, 8, 6],
        latent_hidden_dim: int = 128,
        default_kernel_size: int = 7,
        delay_kernel_size: int = 5,
        init_channel: int = 64,
        res_kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.use_cosine_sim = use_cosine_sim
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers

        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.num_layers_2 = num_layers_2
        self.out_channels = out_channels

        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.causal = causal
        self.num_samples = num_samples
        self.downsample_factors = downsample_factors
        self.downsample_kernel_sizes = downsample_kernel_sizes
        self.upsample_factors = upsample_factors
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.latent_hidden_dim = latent_hidden_dim
        self.default_kernel_size = default_kernel_size
        self.delay_kernel_size = delay_kernel_size
        self.init_channel = init_channel
        self.res_kernel_size = res_kernel_size
