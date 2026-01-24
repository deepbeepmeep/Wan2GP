from transformers.configuration_utils import PretrainedConfig


class HeartMuLaConfig(PretrainedConfig):
    model_type = "heartmula"

    def __init__(
        self,
        backbone_flavor: str = "llama-3B",
        decoder_flavor: str = "llama-300M",
        text_vocab_size: int = 128256,
        audio_vocab_size: int = 8197,
        audio_num_codebooks: int = 8,
        muq_dim: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_flavor = backbone_flavor
        self.decoder_flavor = decoder_flavor
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        self.muq_dim = muq_dim
