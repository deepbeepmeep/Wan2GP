"""Main processor for KugelAudio combining text and audio processing."""

import json
import math
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)
from transformers.utils import TensorType, cached_file, logging

from .audio_processor import AudioNormalizer, AudioProcessor

logger = logging.get_logger(__name__)


class KugelAudioProcessor:
    """Combined processor for KugelAudio text and audio.

    Wraps a text tokenizer and audio processor into a single interface
    for preparing inputs for KugelAudio models.

    Example:
        >>> processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")
        >>> inputs = processor(text="Hello world", voice_prompt=voice_audio)
    """

    def __init__(
        self,
        tokenizer=None,
        audio_processor: Optional[AudioProcessor] = None,
        speech_compression_ratio: int = 3200,
        db_normalize: bool = True,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor or AudioProcessor()
        self.speech_compression_ratio = speech_compression_ratio
        self.db_normalize = db_normalize
        self.audio_normalizer = AudioNormalizer() if db_normalize else None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load processor from pretrained model.

        Args:
            pretrained_model_name_or_path: Model ID or local path

        Returns:
            KugelAudioProcessor instance
        """
        from .text_tokenizer import KugelAudioTextTokenizer

        # Try to load config
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        config = None

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path, "preprocessor_config.json", **kwargs
                )
                with open(config_file, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Using defaults.")
                config = {
                    "speech_compression_ratio": 3200,
                    "db_normalize": True,
                }

        # Extract parameters
        speech_compression_ratio = config.get("speech_compression_ratio", 3200)
        db_normalize = config.get("db_normalize", True)

        # Load tokenizer
        lm_name = config.get("language_model_pretrained_name") or kwargs.pop(
            "language_model_pretrained_name", "Qwen/Qwen2.5-1.5B"
        )
        logger.info(f"Loading tokenizer from {lm_name}")
        tokenizer = KugelAudioTextTokenizer.from_pretrained(lm_name, **kwargs)

        # Load audio processor
        if "audio_processor" in config:
            audio_config = config["audio_processor"]
            audio_processor = AudioProcessor(
                sampling_rate=audio_config.get("sampling_rate", 24000),
                normalize_audio=audio_config.get("normalize_audio", True),
                target_dB_FS=audio_config.get("target_dB_FS", -25),
            )
        else:
            audio_processor = AudioProcessor()

        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_compression_ratio=speech_compression_ratio,
            db_normalize=db_normalize,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """Save processor to directory."""
        os.makedirs(save_directory, exist_ok=True)

        config = {
            "processor_class": "KugelAudioProcessor",
            "speech_compression_ratio": self.speech_compression_ratio,
            "db_normalize": self.db_normalize,
            "audio_processor": {
                "feature_extractor_type": "AudioProcessor",
                "sampling_rate": getattr(self.audio_processor, "sampling_rate", 24000),
                "normalize_audio": getattr(self.audio_processor, "normalize_audio", True),
                "target_dB_FS": getattr(self.audio_processor, "target_dB_FS", -25),
            },
        }

        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Processor saved to {config_path}")

    def __call__(
        self,
        text: Optional[str] = None,
        voice_prompt: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """Process text and optional voice prompt.

        Args:
            text: Input text to synthesize
            voice_prompt: Voice prompt audio for speaker identity (raw audio tensor or path)
            padding: Padding strategy
            truncation: Truncation strategy
            max_length: Maximum sequence length
            return_tensors: Return format

        Returns:
            BatchEncoding with processed inputs including speech_input_mask for voice cloning
        """
        if text is None:
            raise ValueError("Text input is required")

        # Special token IDs
        speech_start_id = 151652  # <|vision_start|> repurposed for speech
        speech_diffusion_id = 151654  # VAE token used as placeholder

        # Format text with proper template
        # Add speaker prefix if not present (use Speaker 0 to match training format)
        formatted_text = text.strip()
        if not formatted_text.startswith("Speaker"):
            formatted_text = f"Speaker 0: {formatted_text}"

        # Build the full prompt template matching the training format
        system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"

        # Start building tokens and speech_input_mask
        full_tokens = []
        speech_input_mask = []
        voice_audio = None

        # System prompt tokens
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=False)
        full_tokens.extend(system_tokens)
        speech_input_mask.extend([False] * len(system_tokens))

        # Process voice prompt if provided
        if voice_prompt is not None:
            # Load audio if it's a path
            if isinstance(voice_prompt, str):
                voice_audio = self.audio_processor._load_from_path(voice_prompt)
                if self.db_normalize and self.audio_normalizer:
                    voice_audio = self.audio_normalizer(voice_audio)
            elif isinstance(voice_prompt, np.ndarray):
                voice_audio = voice_prompt.astype(np.float32)
            elif isinstance(voice_prompt, torch.Tensor):
                voice_audio = voice_prompt.cpu().numpy()
                if voice_audio.ndim > 1:
                    voice_audio = voice_audio.squeeze()
                voice_audio = voice_audio.astype(np.float32)

            # Voice input section with placeholder tokens
            voice_input_tokens = self.tokenizer.encode(" Voice input:\n", add_special_tokens=False)
            full_tokens.extend(voice_input_tokens)
            speech_input_mask.extend([False] * len(voice_input_tokens))

            # Speaker prefix for voice
            speaker_prefix = self.tokenizer.encode(" Speaker 0:", add_special_tokens=False)
            full_tokens.extend(speaker_prefix)
            speech_input_mask.extend([False] * len(speaker_prefix))

            # Calculate number of VAE tokens needed based on audio length
            # compression ratio is typically 3200 samples per token at 24kHz
            num_voice_tokens = math.ceil(len(voice_audio) / self.speech_compression_ratio)

            # Add placeholder VAE tokens that will be replaced with speech embeddings
            full_tokens.extend([speech_diffusion_id] * num_voice_tokens)
            speech_input_mask.extend([True] * num_voice_tokens)  # These positions get speech embeddings

            # Newline after voice
            newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
            full_tokens.extend(newline_tokens)
            speech_input_mask.extend([False] * len(newline_tokens))

        # Text input section
        text_input_tokens = self.tokenizer.encode(" Text input:\n", add_special_tokens=False)
        full_tokens.extend(text_input_tokens)
        speech_input_mask.extend([False] * len(text_input_tokens))

        # Speaker text
        speaker_text_tokens = self.tokenizer.encode(f" {formatted_text}\n", add_special_tokens=False)
        full_tokens.extend(speaker_text_tokens)
        speech_input_mask.extend([False] * len(speaker_text_tokens))

        # Speech output section
        speech_output_tokens = self.tokenizer.encode(" Speech output:\n", add_special_tokens=False)
        full_tokens.extend(speech_output_tokens)
        speech_input_mask.extend([False] * len(speech_output_tokens))

        # Add speech_start token
        full_tokens.append(speech_start_id)
        speech_input_mask.append(False)

        result = BatchEncoding()
        result["text_ids"] = full_tokens
        result["speech_input_mask"] = speech_input_mask

        if return_tensors == "pt":
            result["text_ids"] = torch.tensor([full_tokens], dtype=torch.long)
            result["speech_input_mask"] = torch.tensor([speech_input_mask], dtype=torch.bool)

        # Include processed voice audio for the model to encode
        if voice_audio is not None:
            if return_tensors == "pt":
                result["speech_tensors"] = torch.tensor(voice_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                # Create speech_masks (all True for the voice frames)
                num_frames = math.ceil(len(voice_audio) / self.speech_compression_ratio)
                result["speech_masks"] = torch.ones(1, num_frames, dtype=torch.bool)
            else:
                result["speech_tensors"] = voice_audio
                num_frames = math.ceil(len(voice_audio) / self.speech_compression_ratio)
                result["speech_masks"] = [True] * num_frames

        return result

    def process_with_cached_prompt(
        self,
        text: str,
        cached_prompt: Dict[str, Any],
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        **kwargs,
    ) -> BatchEncoding:
        """Process text with pre-computed voice prompt cache.

        Args:
            text: Input text to synthesize
            cached_prompt: Pre-computed KV cache from voice prompt
            return_tensors: Return format

        Returns:
            BatchEncoding ready for generation
        """
        script_tokens = self.tokenizer.encode(text.strip() + "\n", add_special_tokens=False)

        lm_length = cached_prompt["lm"]["last_hidden_state"].size(1)
        tts_lm_length = cached_prompt["tts_lm"]["last_hidden_state"].size(1)

        # Create pseudo input IDs
        input_ids = [self.tokenizer.pad_id] * lm_length
        tts_lm_input_ids = [self.tokenizer.pad_id] * tts_lm_length
        speech_input_mask = [False] * tts_lm_length

        result = BatchEncoding()

        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([input_ids], dtype=torch.long)
            result["tts_lm_input_ids"] = torch.tensor([tts_lm_input_ids], dtype=torch.long)
            result["tts_text_ids"] = torch.tensor([script_tokens], dtype=torch.long)
            result["attention_mask"] = torch.ones(1, lm_length, dtype=torch.long)
            result["tts_lm_attention_mask"] = torch.ones(1, tts_lm_length, dtype=torch.long)
            result["speech_input_mask"] = torch.tensor([speech_input_mask], dtype=torch.bool)
        else:
            result["input_ids"] = [input_ids]
            result["tts_lm_input_ids"] = [tts_lm_input_ids]
            result["tts_text_ids"] = [script_tokens]
            result["attention_mask"] = [[1] * lm_length]
            result["tts_lm_attention_mask"] = [[1] * tts_lm_length]
            result["speech_input_mask"] = [speech_input_mask]

        return result

    def prepare_speech_inputs(
        self,
        speech_inputs: List[np.ndarray],
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Prepare speech inputs for model.

        Args:
            speech_inputs: List of speech arrays
            return_tensors: Return format
            device: Device to place tensors
            dtype: Data type for tensors

        Returns:
            Dictionary with padded speeches and masks
        """
        if not speech_inputs:
            return {"padded_speeches": None, "speech_masks": None}

        # Calculate sequence lengths
        seq_lens = [math.ceil(s.shape[0] / self.speech_compression_ratio) for s in speech_inputs]
        max_speech_len = max(s.shape[0] for s in speech_inputs)

        # Pad speeches
        padded = np.zeros((len(speech_inputs), max_speech_len), dtype=np.float32)
        masks = np.zeros((len(speech_inputs), max(seq_lens)), dtype=np.bool_)

        for i, (speech, seq_len) in enumerate(zip(speech_inputs, seq_lens)):
            padded[i, : len(speech)] = speech
            masks[i, :seq_len] = True

        result = {"padded_speeches": padded, "speech_masks": masks}

        if return_tensors == "pt":
            result["padded_speeches"] = torch.tensor(
                padded, device=device, dtype=dtype or torch.float32
            )
            result["speech_masks"] = torch.tensor(masks, device=device, dtype=torch.bool)

        return result

    def batch_decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.decode(*args, **kwargs)

    def save_audio(self, audio, output_path: str = "output.wav", **kwargs) -> List[str]:
        """Save generated audio to file."""
        return self.audio_processor.save_audio(audio, output_path, **kwargs)

    @property
    def model_input_names(self) -> List[str]:
        """Return list of model input names."""
        tokenizer_names = getattr(self.tokenizer, "model_input_names", [])
        audio_names = getattr(self.audio_processor, "model_input_names", [])
        return list(
            dict.fromkeys(tokenizer_names + audio_names + ["speech_inputs", "speech_input_mask"])
        )
