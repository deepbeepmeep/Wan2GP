"""KugelAudio configuration classes."""

from .model_config import (
    KugelAudioConfig,
    KugelAudioAcousticTokenizerConfig,
    KugelAudioSemanticTokenizerConfig,
    KugelAudioDiffusionHeadConfig,
    # Aliases
    AcousticTokenizerConfig,
    SemanticTokenizerConfig,
    DiffusionHeadConfig,
)

__all__ = [
    "KugelAudioConfig",
    "KugelAudioAcousticTokenizerConfig",
    "KugelAudioSemanticTokenizerConfig",
    "KugelAudioDiffusionHeadConfig",
    "AcousticTokenizerConfig",
    "SemanticTokenizerConfig",
    "DiffusionHeadConfig",
]
