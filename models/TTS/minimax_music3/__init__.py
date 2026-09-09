"""MiniMax Music 3 integration for WanGP."""

from .condition_embedder_minimax_music3 import MiniMaxMusic3ConditionEncoder
from .minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
from .minimax_music3_vocoder import MiniMaxMusic3Vocoder
from .transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel

__all__ = [
    "MiniMaxMusic3ConditionEncoder",
    "MiniMaxMusic3RVQDepthDecoder",
    "MiniMaxMusic3Transformer1DModel",
    "MiniMaxMusic3Vocoder",
]
