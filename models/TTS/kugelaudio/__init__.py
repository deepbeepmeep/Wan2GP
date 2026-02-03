"""KugelAudio model components (vendored for WanGP)."""

__version__ = "0.1.0"

from .configs import (
    KugelAudioAcousticTokenizerConfig,
    KugelAudioConfig,
    KugelAudioDiffusionHeadConfig,
    KugelAudioSemanticTokenizerConfig,
)
from .models import (
    KugelAudioAcousticTokenizerModel,
    KugelAudioDiffusionHead,
    KugelAudioForConditionalGeneration,
    KugelAudioForConditionalGenerationInference,
    KugelAudioModel,
    KugelAudioPreTrainedModel,
    KugelAudioSemanticTokenizerModel,
)
from .processors import KugelAudioProcessor
from .schedule import DPMSolverMultistepScheduler


__all__ = [
    # Version
    "__version__",
    # Configs
    "KugelAudioConfig",
    "KugelAudioAcousticTokenizerConfig",
    "KugelAudioSemanticTokenizerConfig",
    "KugelAudioDiffusionHeadConfig",
    # Models
    "KugelAudioModel",
    "KugelAudioPreTrainedModel",
    "KugelAudioForConditionalGeneration",
    "KugelAudioForConditionalGenerationInference",
    "KugelAudioAcousticTokenizerModel",
    "KugelAudioSemanticTokenizerModel",
    "KugelAudioDiffusionHead",
    # Scheduler
    "DPMSolverMultistepScheduler",
    # Processors
    "KugelAudioProcessor",
    # Processors
    "KugelAudioProcessor",
]
