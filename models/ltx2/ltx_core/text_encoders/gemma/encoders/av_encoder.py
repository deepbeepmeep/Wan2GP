from typing import NamedTuple

import torch
from transformers.models.gemma3 import Gemma3ForConditionalGeneration

from ....loader.sd_ops import SDOps
from ....model.model_protocol import ModelConfigurator
from ..embeddings_connector import (
    Embeddings1DConnector,
    Embeddings1DConnectorConfigurator,
)
from .base_encoder import GemmaTextEncoderModelBase
from ..feature_extractor import GemmaFeaturesExtractorProjLinear
from ..tokenizer import LTXVGemmaTokenizer


class AVGemmaEncoderOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor
    attention_mask: torch.Tensor


class GemmaTextEncoderModel(GemmaTextEncoderModelBase):
    """
    Gemma text encoder that exposes raw embeddings without post-processing modules.
    """

    def __init__(
        self,
        tokenizer: LTXVGemmaTokenizer | None = None,
        model: Gemma3ForConditionalGeneration | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            feature_extractor_linear=None,
            tokenizer=tokenizer,
            model=model,
            dtype=dtype,
        )


class GemmaTextEncoderModelConfigurator(ModelConfigurator[GemmaTextEncoderModel]):
    @classmethod
    def from_config(cls: type["GemmaTextEncoderModel"], _config: dict) -> "GemmaTextEncoderModel":
        return GemmaTextEncoderModel()


class AVGemmaTextEncoderModel(GemmaTextEncoderModelBase):
    """
    AVGemma Text Encoder Model.
    This class combines the tokenizer, Gemma model, feature extractor from base class and a
    video and audio embeddings connectors to provide a preprocessing for audio-visual pipeline.
    """

    def __init__(
        self,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        embeddings_connector: Embeddings1DConnector,
        audio_embeddings_connector: Embeddings1DConnector,
        tokenizer: LTXVGemmaTokenizer | None = None,
        model: Gemma3ForConditionalGeneration | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            feature_extractor_linear=feature_extractor_linear,
            tokenizer=tokenizer,
            model=model,
            dtype=dtype,
        )
        self.embeddings_connector = embeddings_connector.to(dtype=dtype)
        self.audio_embeddings_connector = audio_embeddings_connector.to(dtype=dtype)

    def _run_connectors(
        self, encoded_input: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        connector_attention_mask = self._convert_to_additive_mask(attention_mask, encoded_input.dtype)

        encoded, encoded_connector_attention_mask = self.embeddings_connector(
            encoded_input,
            connector_attention_mask,
        )

        # restore the mask values to int64
        attention_mask = (encoded_connector_attention_mask < 0.000001).to(torch.int64)
        attention_mask = attention_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
        encoded = encoded * attention_mask

        encoded_for_audio, _ = self.audio_embeddings_connector(encoded_input, connector_attention_mask)

        return encoded, encoded_for_audio, attention_mask.squeeze(-1)

    def forward(self, text: str, padding_side: str = "left") -> AVGemmaEncoderOutput:
        encoded_inputs, attention_mask = self._preprocess_text(text, padding_side)
        video_encoding, audio_encoding, attention_mask = self._run_connectors(encoded_inputs, attention_mask)
        return AVGemmaEncoderOutput(video_encoding, audio_encoding, attention_mask)


class AVGemmaTextEncoderModelConfigurator(ModelConfigurator[AVGemmaTextEncoderModel]):
    @classmethod
    def from_config(cls: type["AVGemmaTextEncoderModel"], config: dict) -> "AVGemmaTextEncoderModel":
        feature_extractor_linear = GemmaFeaturesExtractorProjLinear.from_config(config)
        embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        audio_embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        return AVGemmaTextEncoderModel(
            feature_extractor_linear=feature_extractor_linear,
            embeddings_connector=embeddings_connector,
            audio_embeddings_connector=audio_embeddings_connector,
        )


class GemmaTextEmbeddingsConnectorModel(torch.nn.Module):
    def __init__(
        self,
        video_embeddings_connector: Embeddings1DConnector,
        audio_embeddings_connector: Embeddings1DConnector,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.video_embeddings_connector = video_embeddings_connector.to(dtype=dtype)
        self.audio_embeddings_connector = audio_embeddings_connector.to(dtype=dtype)


class GemmaTextEmbeddingsConnectorModelConfigurator(ModelConfigurator[GemmaTextEmbeddingsConnectorModel]):
    @classmethod
    def from_config(cls: type["GemmaTextEmbeddingsConnectorModel"], config: dict) -> "GemmaTextEmbeddingsConnectorModel":
        video_embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        audio_embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        return GemmaTextEmbeddingsConnectorModel(
            video_embeddings_connector=video_embeddings_connector,
            audio_embeddings_connector=audio_embeddings_connector,
        )


GEMMA_TEXT_ENCODER_KEY_OPS = SDOps("GEMMA_TEXT_ENCODER_KEY_OPS")

AV_GEMMA_TEXT_ENCODER_KEY_OPS = (
    SDOps("AV_GEMMA_TEXT_ENCODER_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_matching(prefix="diffusion_model.audio_embeddings_connector.")
    .with_matching(prefix="diffusion_model.video_embeddings_connector.")
    .with_replacement("text_embedding_projection.", "feature_extractor_linear.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
    .with_replacement("diffusion_model.video_embeddings_connector.", "embeddings_connector.")
    .with_replacement("diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
)

TEXT_EMBEDDING_PROJECTION_KEY_OPS = (
    SDOps("TEXT_EMBEDDING_PROJECTION_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.")
    .with_replacement("text_embedding_projection.", "")
)

TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS = (
    SDOps("TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_matching(prefix="diffusion_model.audio_embeddings_connector.")
    .with_matching(prefix="diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "video_embeddings_connector.")
    .with_replacement("diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
    .with_replacement("diffusion_model.video_embeddings_connector.", "video_embeddings_connector.")
)
