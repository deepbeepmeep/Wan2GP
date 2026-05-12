from transformers.models.auto import AutoConfig, AutoModel

from .configuration_higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerConfig
from .modeling_higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel, HiggsAudioV2TokenizerPreTrainedModel


AutoConfig.register("higgs_audio_v2_tokenizer", HiggsAudioV2TokenizerConfig)
AutoModel.register(HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel)


__all__ = [
    "HiggsAudioV2TokenizerConfig",
    "HiggsAudioV2TokenizerModel",
    "HiggsAudioV2TokenizerPreTrainedModel",
]
