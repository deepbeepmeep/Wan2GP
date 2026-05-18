from __future__ import annotations

"""
IndexTTS2 local audiotools compatibility layer.

This file provides a minimal API subset used by IndexTTS2 and is based on
behavior from the upstream Descript projects:
- https://github.com/descriptinc/audiotools (tag: 0.7.4)
- https://github.com/descriptinc/descript-audio-codec

Upstream package metadata credits authors Prem Seetharaman and Lucas Gestin.
Upstream audiotools is licensed under MIT (see upstream LICENSE file).
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


_AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
}


def find_audio(path: Union[str, Path]) -> list[Path]:
    input_path = Path(path)
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in _AUDIO_EXTENSIONS else []
    if not input_path.exists():
        return []
    audio_files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS]
    return sorted(audio_files)


@dataclass
class STFTParams:
    window_length: int
    hop_length: int
    match_stride: bool = False
    window_type: Optional[str] = None


def _get_window(window_type: Optional[str], window_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    name = (window_type or "hann").lower()
    if name in {"hann", "hann_window"}:
        return torch.hann_window(window_length, device=device, dtype=dtype)
    if name in {"hamming", "hamming_window"}:
        return torch.hamming_window(window_length, device=device, dtype=dtype)
    return torch.hann_window(window_length, device=device, dtype=dtype)


class AudioSignal:
    def __init__(
        self,
        data: Union[str, Path, torch.Tensor],
        sample_rate: Optional[int] = None,
        stft_params: Optional[STFTParams] = None,
    ):
        if isinstance(data, (str, Path)):
            waveform, sr = torchaudio.load(str(data))
            self.audio_data = waveform.unsqueeze(0)
            self.sample_rate = int(sr)
        else:
            tensor = torch.as_tensor(data)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 3:
                raise ValueError(f"Expected 1D/2D/3D audio tensor, got shape {tuple(tensor.shape)}")
            if sample_rate is None:
                raise ValueError("sample_rate is required when constructing AudioSignal from tensors")
            self.audio_data = tensor
            self.sample_rate = int(sample_rate)

        if not torch.is_floating_point(self.audio_data):
            self.audio_data = self.audio_data.float()

        default_hop = max(1, int(self.sample_rate * 0.01))
        default_win = max(16, default_hop * 4)
        self.stft_params = stft_params or STFTParams(window_length=default_win, hop_length=default_hop)
        self.magnitude = None

    @property
    def device(self) -> torch.device:
        return self.audio_data.device

    @property
    def shape(self):
        return self.audio_data.shape

    @property
    def signal_length(self) -> int:
        return int(self.audio_data.shape[-1])

    @property
    def signal_duration(self) -> float:
        return float(self.signal_length) / float(self.sample_rate)

    def clone(self) -> "AudioSignal":
        return AudioSignal(self.audio_data.clone(), self.sample_rate, stft_params=self.stft_params)

    def __getitem__(self, item) -> "AudioSignal":
        tensor = self.audio_data.__getitem__(item)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim != 3:
            raise ValueError(f"Unsupported indexed shape {tuple(tensor.shape)}")
        return AudioSignal(tensor, self.sample_rate, stft_params=self.stft_params)

    def to(self, device: Union[str, torch.device]) -> "AudioSignal":
        self.audio_data = self.audio_data.to(device)
        return self

    def zero_pad(self, left: int, right: int) -> "AudioSignal":
        padded = F.pad(self.audio_data, (int(left), int(right)))
        return AudioSignal(padded, self.sample_rate, stft_params=self.stft_params)

    def resample(self, target_sample_rate: int) -> "AudioSignal":
        target_sample_rate = int(target_sample_rate)
        if target_sample_rate == self.sample_rate:
            return self
        batch, channels, samples = self.audio_data.shape
        flat = self.audio_data.reshape(batch * channels, samples)
        resampled = torchaudio.functional.resample(flat, self.sample_rate, target_sample_rate)
        self.audio_data = resampled.reshape(batch, channels, -1)
        self.sample_rate = target_sample_rate
        self.magnitude = None
        return self

    def ffmpeg_resample(self, target_sample_rate: int) -> "AudioSignal":
        return self.resample(target_sample_rate)

    def loudness(self) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(self.audio_data.float() ** 2) + 1e-12)
        db = 20.0 * torch.log10(rms.clamp_min(1e-7))
        return db.detach().cpu()

    def ffmpeg_loudness(self) -> torch.Tensor:
        return self.loudness()

    def normalize(self, target_db: Union[float, torch.Tensor]) -> "AudioSignal":
        target = float(torch.as_tensor(target_db).detach().cpu().item())
        current = float(self.loudness().item())
        gain = 10 ** ((target - current) / 20.0)
        self.audio_data = self.audio_data * gain
        self.magnitude = None
        return self

    def ensure_max_of_audio(self, max_value: float = 0.99) -> "AudioSignal":
        peak = self.audio_data.abs().amax()
        if torch.isfinite(peak) and peak.item() > max_value:
            self.audio_data = self.audio_data * (max_value / peak)
        return self

    def stft(
        self,
        window_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_type: Optional[str] = None,
    ) -> torch.Tensor:
        params = self.stft_params
        win_length = int(window_length or params.window_length)
        hop = int(hop_length or params.hop_length)
        window = _get_window(window_type or params.window_type, win_length, self.device, self.audio_data.dtype)
        batch, channels, samples = self.audio_data.shape
        flat = self.audio_data.reshape(batch * channels, samples)
        spec = torch.stft(
            flat,
            n_fft=win_length,
            hop_length=hop,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        spec = spec.reshape(batch, channels, spec.shape[-2], spec.shape[-1])
        self.magnitude = spec.abs()
        return spec

    def mel_spectrogram(
        self,
        n_mels: int,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        window_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_type: Optional[str] = None,
    ) -> torch.Tensor:
        params = self.stft_params
        win_length = int(window_length or params.window_length)
        hop = int(hop_length or params.hop_length)
        window_fn = torch.hann_window
        if (window_type or params.window_type or "").lower() in {"hamming", "hamming_window"}:
            window_fn = torch.hamming_window
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop,
            f_min=float(mel_fmin),
            f_max=None if mel_fmax is None else float(mel_fmax),
            n_mels=int(n_mels),
            power=2.0,
            center=True,
            window_fn=window_fn,
        ).to(self.device)
        batch, channels, samples = self.audio_data.shape
        flat = self.audio_data.reshape(batch * channels, samples)
        mels = mel(flat)
        return mels.reshape(batch, channels, mels.shape[-2], mels.shape[-1])

    def write(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform = self.audio_data[0].detach().cpu()
        torchaudio.save(str(out_path), waveform, self.sample_rate)
        return out_path

    @classmethod
    def load_from_file_with_ffmpeg(cls, path: Union[str, Path]) -> "AudioSignal":
        return cls(path)


class BaseModel(nn.Module):
    INTERN = []
    EXTERN = []

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @classmethod
    def _extract_state_dict(cls, checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "generator", "weights"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint
        raise RuntimeError("Unsupported checkpoint format for BaseModel.load")

    @classmethod
    def _clean_state_dict(cls, state_dict):
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            cleaned[key] = value
        return cleaned

    @classmethod
    def load(cls, path: Union[str, Path]):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = cls._clean_state_dict(cls._extract_state_dict(checkpoint))
        model = cls()
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model


class Accelerator:
    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare(self, *objects):
        if len(objects) == 1:
            return objects[0]
        return objects

    @staticmethod
    def unwrap_model(model):
        return model


ml = SimpleNamespace(BaseModel=BaseModel, Accelerator=Accelerator)


__all__ = [
    "AudioSignal",
    "STFTParams",
    "BaseModel",
    "Accelerator",
    "ml",
    "find_audio",
]
