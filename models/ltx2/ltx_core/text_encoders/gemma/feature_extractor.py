import torch

from ...model.model_protocol import ModelConfigurator


class GemmaFeaturesExtractorProjLinear(torch.nn.Module, ModelConfigurator["GemmaFeaturesExtractorProjLinear"]):
    """
    Feature extractor module for Gemma models.
    This module applies a single linear projection to the input tensor.
    It expects a flattened feature tensor of shape (batch_size, 3840*49).
    The linear layer maps this to a (batch_size, 3840) embedding.
    Attributes:
        aggregate_embed (torch.nn.Linear): Linear projection layer.
    """

    def __init__(self) -> None:
        """
        Initialize the GemmaFeaturesExtractorProjLinear module.
        The input dimension is expected to be 3840 * 49, and the output is 3840.
        """
        super().__init__()
        self.aggregate_embed = torch.nn.Linear(3840 * 49, 3840, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feature extractor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3840 * 49).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3840).
        """
        return self.aggregate_embed(x)

    @classmethod
    def from_config(cls: type["GemmaFeaturesExtractorProjLinear"], _config: dict) -> "GemmaFeaturesExtractorProjLinear":
        return cls()
