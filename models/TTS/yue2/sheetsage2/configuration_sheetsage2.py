"""Configuration for SheetSage2 audio-to-symbolic transcription."""

from transformers import PretrainedConfig

from .configuration_mert2 import MERT2Config


class SheetSage2Config(PretrainedConfig):
    model_type = "sheetsage2"

    def __init__(
        self,
        vocab_size=31678,
        hidden_size=512,
        decoder_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        decoder_dropout=0.1,
        input_audio_length=300.0,
        max_output_seq_len=5120,
        time_hz=100,
        sampling_rate=24000,
        lora_rank=64,
        lora_alpha=128,
        weights_format="adapter",
        backbone_config=None,
        encoder_attn_implementation="sdpa",
        base_model_name_or_path="m-a-p/MERT-v2-FullSong",
        base_model_revision="d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
        base_model_sha256="e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d",
        tokenizer_schema_version="v1",
        tokenizer_fingerprint="5ba3325af0344c7f",
        **kwargs,
    ):
        for name, default in (
            ("is_encoder_decoder", True), ("tie_word_embeddings", True),
            ("pad_token_id", 0), ("bos_token_id", 1), ("eos_token_id", 2),
            ("decoder_start_token_id", 1),
            ("use_cache", True),
        ):
            kwargs.setdefault(name, default)
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.decoder_layers = int(decoder_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.decoder_dropout = float(decoder_dropout)
        self.input_audio_length = float(input_audio_length)
        self.max_output_seq_len = int(max_output_seq_len)
        self.time_hz = int(time_hz)
        self.sampling_rate = int(sampling_rate)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.weights_format = str(weights_format)
        self.backbone_config = dict(backbone_config or MERT2Config(variant="fs").to_dict())
        self.backbone_config.pop("_name_or_path", None)
        self.backbone_config.pop("auto_map", None)
        self.encoder_attn_implementation = str(encoder_attn_implementation)
        self.base_model_name_or_path = str(base_model_name_or_path)
        self.base_model_revision = str(base_model_revision)
        self.base_model_sha256 = str(base_model_sha256)
        self.tokenizer_schema_version = str(tokenizer_schema_version)
        self.tokenizer_fingerprint = str(tokenizer_fingerprint)
        self.architectures = ["SheetSage2Model"]
        self.auto_map = {
            "AutoConfig": "configuration_sheetsage2.SheetSage2Config",
            "AutoModel": "modeling_sheetsage2.SheetSage2Model",
            "AutoModelForSeq2SeqLM": "modeling_sheetsage2.SheetSage2Model",
            "AutoProcessor": "processing_sheetsage2.SheetSage2Processor",
        }
        if self.weights_format not in {"adapter", "merged"}:
            raise ValueError("weights_format must be 'adapter' or 'merged'.")
        if self.encoder_attn_implementation not in {"sdpa", "flash_attention_2"}:
            raise ValueError("Select encoder attention 'sdpa' or 'flash_attention_2'.")
        if min(self.vocab_size, self.hidden_size, self.decoder_layers, self.num_attention_heads,
               self.intermediate_size, self.input_audio_length, self.max_output_seq_len,
               self.time_hz, self.sampling_rate, self.lora_rank, self.lora_alpha) <= 0:
            raise ValueError("Model dimensions and timing parameters must be positive.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if not 0 <= self.decoder_dropout < 1:
            raise ValueError("decoder_dropout must lie in [0, 1).")
        if self.sampling_rate != self.backbone_config["sampling_rate"]:
            raise ValueError("Processor and encoder sampling rates must match.")


SheetSage2Config.register_for_auto_class()
