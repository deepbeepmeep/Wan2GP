"""Text tokenizer wrapper for KugelAudio based on Qwen2."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


class KugelAudioTextTokenizer:
    """Wrapper around a fast Qwen2 tokenizer with speech special tokens.

    This avoids relying on transformers' Qwen2TokenizerFast class which is
    missing in older versions, while still using the correct tokenizer.json.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, base_tokenizer):
        self._tok = base_tokenizer
        self._add_speech_special_tokens()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Ensure we use the fast tokenizer files if available.
        kwargs.setdefault("use_fast", True)
        base = AutoTokenizer.from_pretrained(*args, **kwargs)
        return cls(base)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tok, name)

    def _add_speech_special_tokens(self):
        """Add KugelAudio-specific special tokens for speech."""
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",  # Speech start (reusing vision tokens for compatibility)
                "<|vision_end|>",    # Speech end
                "<|vision_pad|>",    # Speech diffusion pad
            ]
        }
        self._tok.add_special_tokens(special_tokens)

        # Cache special token IDs
        self._speech_start_id = self._tok.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self._tok.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self._tok.convert_tokens_to_ids("<|vision_pad|>")
        self._eos_id = self._tok.eos_token_id
        self._pad_id = self._tok.convert_tokens_to_ids("<|image_pad|>")

    @property
    def eos_id(self) -> int:
        """End of sequence token ID."""
        return self._eos_id

    @property
    def speech_start_id(self) -> int:
        """Speech start token ID."""
        return self._speech_start_id

    @property
    def speech_end_id(self) -> int:
        """Speech end token ID."""
        return self._speech_end_id

    @property
    def speech_diffusion_id(self) -> int:
        """Speech diffusion placeholder token ID."""
        return self._speech_diffusion_id

    @property
    def pad_id(self) -> int:
        """Padding token ID (returns -100 for loss masking)."""
        return self._pad_id
