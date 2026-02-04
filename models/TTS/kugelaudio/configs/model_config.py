"""Configuration classes for KugelAudio models."""

from typing import Optional, List, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.utils import logging

logger = logging.get_logger(__name__)


class KugelAudioAcousticTokenizerConfig(PretrainedConfig):
    """Configuration for the acoustic tokenizer.
    
    The acoustic tokenizer converts continuous speech latents back to audio waveforms.
    It uses a hierarchical convolutional architecture with multiple upsampling stages.
    """
    
    model_type = "kugelaudio_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = "gaussian",
        # Common settings
        mixer_layer: str = "depthwise_conv",
        conv_norm: str = "none",
        pad_mode: str = "constant",
        disable_last_norm: bool = True,
        layernorm: str = "RMSNorm",
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # Encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        # Decoder specific
        decoder_n_filters: int = 32,
        decoder_ratios: Optional[List[int]] = None,
        decoder_depths: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type

        # Common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # Encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios if encoder_ratios is not None else [8, 5, 5, 4, 2, 2]
        self.encoder_depths = encoder_depths

        # Decoder specific parameters
        self.decoder_ratios = decoder_ratios if decoder_ratios is not None else self.encoder_ratios
        self.decoder_n_filters = decoder_n_filters
        self.decoder_depths = decoder_depths


class KugelAudioSemanticTokenizerConfig(PretrainedConfig):
    """Configuration for the semantic tokenizer.
    
    The semantic tokenizer extracts semantic features from audio for conditioning.
    """
    
    model_type = "kugelaudio_semantic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0,
        std_dist_type: str = "none",
        # Common settings
        mixer_layer: str = "depthwise_conv",
        conv_norm: str = "none",
        pad_mode: str = "constant",
        disable_last_norm: bool = True,
        layernorm: str = "RMSNorm",
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # Encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type

        # Common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # Encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios if encoder_ratios is not None else [8, 5, 5, 4, 2, 2]
        self.encoder_depths = encoder_depths


class KugelAudioDiffusionHeadConfig(PretrainedConfig):
    """Configuration for the diffusion prediction head.
    
    The diffusion head predicts speech latents from text-conditioned hidden states
    using a denoising diffusion process.
    """
    
    model_type = "kugelaudio_diffusion_head"

    def __init__(
        self,
        hidden_size: int = 768,
        head_layers: int = 4,
        head_ffn_ratio: float = 3.0,
        rms_norm_eps: float = 1e-5,
        latent_size: int = 64,
        speech_vae_dim: Optional[int] = None,
        prediction_type: str = "v_prediction",
        diffusion_type: str = "ddpm",
        ddpm_num_steps: int = 1000,
        ddpm_num_inference_steps: int = 20,
        ddpm_beta_schedule: str = "cosine",
        ddpm_algorithm_type: str = "sde-dpmsolver++",
        ddpm_batch_mul: int = 4,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.head_layers = head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.rms_norm_eps = rms_norm_eps
        self.latent_size = latent_size
        self.speech_vae_dim = speech_vae_dim
        self.prediction_type = prediction_type
        self.diffusion_type = diffusion_type
        self.ddpm_num_steps = ddpm_num_steps
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        self.ddpm_beta_schedule = ddpm_beta_schedule
        self.ddpm_algorithm_type = ddpm_algorithm_type
        self.ddpm_batch_mul = ddpm_batch_mul

        super().__init__(**kwargs)


class KugelAudioConfig(PretrainedConfig):
    """Main configuration for KugelAudio TTS model.
    
    This configuration combines:
    - A language model backbone (Qwen2) for text understanding
    - An acoustic tokenizer for audio encoding/decoding
    - A semantic tokenizer for semantic feature extraction
    - A diffusion head for speech latent prediction
    
    Example:
        >>> from kugelaudio import KugelAudioConfig
        >>> config = KugelAudioConfig.from_pretrained("kugelaudio/kugelaudio-0-open")
    """
    
    model_type = "kugelaudio"
    is_composition = True
    
    sub_configs = {
        "acoustic_tokenizer_config": KugelAudioAcousticTokenizerConfig,
        "semantic_tokenizer_config": KugelAudioSemanticTokenizerConfig,
        "decoder_config": Qwen2Config,
        "diffusion_head_config": KugelAudioDiffusionHeadConfig,
    }
    
    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        decoder_config=None,
        diffusion_head_config=None,
        **kwargs,
    ):
        # Disable auto attention implementation selection
        kwargs["_attn_implementation_autoset"] = False

        # Initialize acoustic tokenizer config
        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"]()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "kugelaudio_acoustic_tokenizer"
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"](**acoustic_tokenizer_config)
        elif isinstance(acoustic_tokenizer_config, KugelAudioAcousticTokenizerConfig):
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        # Initialize semantic tokenizer config
        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"]()
        elif isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = "kugelaudio_semantic_tokenizer"
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"](**semantic_tokenizer_config)
        elif isinstance(semantic_tokenizer_config, KugelAudioSemanticTokenizerConfig):
            self.semantic_tokenizer_config = semantic_tokenizer_config

        # Initialize decoder (language model) config
        if decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"]()
        elif isinstance(decoder_config, dict):
            if decoder_config.get("model_type", "") == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                raise ValueError(
                    f"Unsupported decoder model type: {decoder_config.get('model_type', '')}"
                )
        elif isinstance(decoder_config, Qwen2Config):
            self.decoder_config = decoder_config

        # Initialize diffusion head config
        if diffusion_head_config is None:
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"]()
        elif isinstance(diffusion_head_config, dict):
            diffusion_head_config["model_type"] = "kugelaudio_diffusion_head"
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"](**diffusion_head_config)
        elif isinstance(diffusion_head_config, KugelAudioDiffusionHeadConfig):
            self.diffusion_head_config = diffusion_head_config

        # Derived parameters
        self.acoustic_vae_dim = self.acoustic_tokenizer_config.vae_dim
        self.semantic_vae_dim = self.semantic_tokenizer_config.vae_dim

        super().__init__(**kwargs)


# Aliases for backwards compatibility
AcousticTokenizerConfig = KugelAudioAcousticTokenizerConfig
SemanticTokenizerConfig = KugelAudioSemanticTokenizerConfig
DiffusionHeadConfig = KugelAudioDiffusionHeadConfig


__all__ = [
    "KugelAudioAcousticTokenizerConfig",
    "KugelAudioSemanticTokenizerConfig", 
    "KugelAudioDiffusionHeadConfig",
    "KugelAudioConfig",
    # Aliases
    "AcousticTokenizerConfig",
    "SemanticTokenizerConfig",
    "DiffusionHeadConfig",
]
