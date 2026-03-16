from typing import NamedTuple

import torch
from transformers import Gemma3ForConditionalGeneration

from ....loader.sd_ops import SDOps
from ....model.model_protocol import ModelConfigurator
from ..embeddings_connector import (
    Embeddings1DConnector,
    Embeddings1DConnectorConfigurator,
)
from .base_encoder import GemmaTextEncoderModelBase
from ..feature_extractor import GemmaFeaturesExtractorProjLinear
from ..tokenizer import LTXVGemmaTokenizer


class VideoGemmaEncoderOutput(NamedTuple):
    video_encoding: torch.Tensor
    attention_mask: torch.Tensor


class VideoGemmaTextEncoderModel(GemmaTextEncoderModelBase):
    """
    Video Gemma Text Encoder Model.
    This class combines the tokenizer, Gemma model, feature extractor from base class and a
    video embeddings connector to provide a preprocessing for video only pipeline.
    """

    def __init__(
        self,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        embeddings_connector: Embeddings1DConnector,
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

    def _run_connector(
        self, encoded_input: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        connector_attention_mask = self._convert_to_additive_mask(attention_mask, encoded_input.dtype)

        encoded, encoded_connector_attention_mask = self.embeddings_connector(
            encoded_input,
            connector_attention_mask,
        )

        # restore the mask values to int64
        attention_mask = (encoded_connector_attention_mask < 0.000001).to(torch.int64)
        attention_mask = attention_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
        encoded = encoded * attention_mask

        return encoded, attention_mask.squeeze(-1)

    def forward(self, text: str, padding_side: str = "left") -> VideoGemmaEncoderOutput:
        encoded_inputs, attention_mask = self._preprocess_text(text, padding_side)
        video_encoding, attention_mask = self._run_connector(encoded_inputs, attention_mask)
        return VideoGemmaEncoderOutput(video_encoding, attention_mask)


class VideoGemmaTextEncoderModelConfigurator(ModelConfigurator[VideoGemmaTextEncoderModel]):
    @classmethod
    def from_config(cls: type["VideoGemmaTextEncoderModel"], config: dict) -> "VideoGemmaTextEncoderModel":
        feature_extractor_linear = GemmaFeaturesExtractorProjLinear.from_config(config)
        embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        return VideoGemmaTextEncoderModel(
            feature_extractor_linear=feature_extractor_linear,
            embeddings_connector=embeddings_connector,
        )


VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS = (
    SDOps("VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.")
    .with_matching(prefix="model.diffusion_model.embeddings_connector.")
    .with_replacement("text_embedding_projection.", "feature_extractor_linear.")
    .with_replacement("model.diffusion_model.embeddings_connector.", "embeddings_connector.")
)
