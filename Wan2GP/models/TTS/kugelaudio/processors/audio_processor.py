"""Audio processing utilities for KugelAudio."""

import os
from typing import Optional, Union, List, Dict, Any

import numpy as np
import torch

from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AudioNormalizer:
    """Normalize audio to target dB FS level.
    
    This ensures consistent input levels for the model while
    maintaining audio quality and avoiding clipping.
    """
    
    def __init__(self, target_dB_FS: float = -25, eps: float = 1e-6):
        self.target_dB_FS = target_dB_FS
        self.eps = eps
    
    def normalize_db(self, audio: np.ndarray) -> tuple:
        """Adjust audio to target dB FS level."""
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        return audio * scalar, rms, scalar
    
    def avoid_clipping(self, audio: np.ndarray) -> tuple:
        """Scale down if necessary to avoid clipping."""
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            scalar = max_val + self.eps
            return audio / scalar, scalar
        return audio, 1.0
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio: adjust dB FS then avoid clipping."""
        audio, _, _ = self.normalize_db(audio)
        audio, _ = self.avoid_clipping(audio)
        return audio


class AudioProcessor(FeatureExtractionMixin):
    """Processor for audio preprocessing and postprocessing.
    
    Handles:
    - Audio format conversion (stereo to mono)
    - Normalization
    - Loading from various file formats
    - Saving to WAV files
    
    Example:
        >>> processor = AudioProcessor(sampling_rate=24000)
        >>> audio = processor("path/to/audio.wav")
        >>> processor.save_audio(generated_audio, "output.wav")
    """
    
    model_input_names = ["input_features"]
    
    def __init__(
        self,
        sampling_rate: int = 24000,
        normalize_audio: bool = True,
        target_dB_FS: float = -25,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.normalizer = AudioNormalizer(target_dB_FS, eps) if normalize_audio else None
        
        self.feature_extractor_dict = {
            "sampling_rate": sampling_rate,
            "normalize_audio": normalize_audio,
            "target_dB_FS": target_dB_FS,
            "eps": eps,
        }
    
    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo to mono if needed."""
        if len(audio.shape) == 1:
            return audio
        elif len(audio.shape) == 2:
            if audio.shape[0] == 2:
                return np.mean(audio, axis=0)
            elif audio.shape[1] == 2:
                return np.mean(audio, axis=1)
            elif audio.shape[0] == 1:
                return audio.squeeze(0)
            elif audio.shape[1] == 1:
                return audio.squeeze(1)
            else:
                raise ValueError(f"Unexpected audio shape: {audio.shape}")
        else:
            raise ValueError(f"Audio should be 1D or 2D, got shape: {audio.shape}")
    
    def _process_single(self, audio: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Process a single audio array."""
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)
        
        audio = self._ensure_mono(audio)
        
        if self.normalize_audio and self.normalizer:
            audio = self.normalizer(audio)
        
        return audio
    
    def _load_from_path(self, audio_path: str) -> np.ndarray:
        """Load audio from file path."""
        ext = os.path.splitext(audio_path)[1].lower()
        
        if ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
            import librosa
            audio, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            return audio
        elif ext == ".pt":
            tensor = torch.load(audio_path, map_location="cpu", weights_only=True).squeeze()
            return tensor.numpy().astype(np.float32)
        elif ext == ".npy":
            return np.load(audio_path).astype(np.float32)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    def __call__(
        self,
        audio: Union[str, np.ndarray, List[float], List[np.ndarray], List[str]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process audio input(s).
        
        Args:
            audio: Audio input - path, array, or list of either
            sampling_rate: Input sampling rate (for validation)
            return_tensors: Return format ("pt" for PyTorch, "np" for NumPy)
            
        Returns:
            Dictionary with processed audio
        """
        if audio is None:
            raise ValueError("Audio input is required")
        
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            logger.warning(
                f"Input sampling rate ({sampling_rate}) differs from expected ({self.sampling_rate}). "
                "Please resample your audio."
            )
        
        # Handle different input types
        if isinstance(audio, str):
            audio = self._load_from_path(audio)
            is_batched = False
        elif isinstance(audio, list):
            if all(isinstance(item, str) for item in audio):
                audio = [self._load_from_path(p) for p in audio]
                is_batched = True
            else:
                is_batched = isinstance(audio[0], (np.ndarray, list))
        else:
            is_batched = False
        
        # Process
        if is_batched:
            processed = [self._process_single(a) for a in audio]
        else:
            processed = [self._process_single(audio)]
        
        # Convert to tensors
        if return_tensors == "pt":
            if len(processed) == 1:
                features = torch.from_numpy(processed[0]).unsqueeze(0).unsqueeze(1)
            else:
                features = torch.stack([torch.from_numpy(a) for a in processed]).unsqueeze(1)
        elif return_tensors == "np":
            if len(processed) == 1:
                features = processed[0][np.newaxis, np.newaxis, :]
            else:
                features = np.stack(processed)[:, np.newaxis, :]
        else:
            features = processed[0] if len(processed) == 1 else processed
        
        return {"audio": features}
    
    def save_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray, List],
        output_path: str = "output.wav",
        sampling_rate: Optional[int] = None,
        normalize: bool = False,
        batch_prefix: str = "audio_",
    ) -> List[str]:
        """Save audio to WAV file(s).
        
        Args:
            audio: Audio data to save
            output_path: Output path (directory for batched audio)
            sampling_rate: Sampling rate (defaults to processor's rate)
            normalize: Whether to normalize before saving
            batch_prefix: Prefix for batch files
            
        Returns:
            List of saved file paths
        """
        import soundfile as sf
        
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.float().detach().cpu().numpy()
        elif isinstance(audio, list):
            if all(isinstance(a, torch.Tensor) for a in audio):
                audio_np = [a.float().detach().cpu().numpy() for a in audio]
            else:
                audio_np = audio
        else:
            audio_np = audio
        
        saved_paths = []
        
        if isinstance(audio_np, list):
            os.makedirs(output_path, exist_ok=True)
            for i, item in enumerate(audio_np):
                item = self._prepare_for_save(item, normalize)
                path = os.path.join(output_path, f"{batch_prefix}{i}.wav")
                sf.write(path, item, sampling_rate)
                saved_paths.append(path)
        elif len(audio_np.shape) >= 3 and audio_np.shape[0] > 1:
            os.makedirs(output_path, exist_ok=True)
            for i in range(audio_np.shape[0]):
                item = audio_np[i].squeeze()
                item = self._prepare_for_save(item, normalize)
                path = os.path.join(output_path, f"{batch_prefix}{i}.wav")
                sf.write(path, item, sampling_rate)
                saved_paths.append(path)
        else:
            item = self._prepare_for_save(audio_np.squeeze(), normalize)
            sf.write(output_path, item, sampling_rate)
            saved_paths.append(output_path)
        
        return saved_paths
    
    def _prepare_for_save(self, audio: np.ndarray, normalize: bool) -> np.ndarray:
        """Prepare audio for saving."""
        if len(audio.shape) > 1 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        
        if normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        
        return audio
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.feature_extractor_dict
