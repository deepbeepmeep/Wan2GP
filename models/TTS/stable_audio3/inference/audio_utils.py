import numpy as np
import torch
from torch import nn
from ..data.utils import PadCrop

from torchaudio import transforms as T

def numpy_audio_to_tensor(audio):
    """Convert a numpy audio array to a float32 torch tensor with shape [channels, samples]."""
    if audio.dtype == np.float32:
        t = torch.from_numpy(audio)
    elif audio.dtype == np.int16:
        t = torch.from_numpy(audio).float().div(32767)
    elif audio.dtype == np.int32:
        t = torch.from_numpy(audio).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio dtype: {audio.dtype}")
    if t.dim() == 1:
        t = t.unsqueeze(0)       # [n] -> [1, n]
    elif t.dim() == 2:
        t = t.transpose(0, 1)   # [n, channels] -> [channels, n]
    return t


def preprocess_audio(sr_audio, sample_rate, sample_size, half=False):
    """Normalize a (sr, numpy_or_tensor) tuple into (sample_rate, tensor) ready for the model."""
    in_sr, audio = sr_audio

    if isinstance(audio, np.ndarray):
        audio = numpy_audio_to_tensor(audio)
    elif audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1)  # [n, 2] -> [2, n]

    if half:
        audio = audio.to(torch.float16)

    if in_sr != sample_rate:
        resample_tf = T.Resample(in_sr, sample_rate).to(audio.device).to(audio.dtype)
        audio = resample_tf(audio)

    if audio.shape[-1] > sample_size:
        audio = audio[:, :sample_size]

    return (sample_rate, audio)


def set_audio_channels(audio, target_channels):
    # Add channel dim if it's missing
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
        
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio

def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    
    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = T.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)

    audio = PadCrop(target_length, randomize=False)(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)

    return audio


class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output
