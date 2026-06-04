import math
import random
import torch

from torch import nn
from typing import Optional, Tuple

from torchaudio import transforms as T

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

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True, pad: bool = True):

        super().__init__()

        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.pad = pad

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int, torch.Tensor]:
        
        n_channels, n_samples = source.shape
        
        # Calculate bounds and offset
        upper_bound = max(0, n_samples - self.n_samples)
        offset = 0
        if self.randomize and n_samples > self.n_samples:
            offset = random.randint(0, upper_bound)

        # Calculate normalized times
        norm_denom = upper_bound + self.n_samples
        t_start = offset / norm_denom
        t_end = (offset + self.n_samples) / norm_denom

        # Calculate timing info
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Optimize for different cases
        if n_samples >= self.n_samples:
            # No padding needed - use view (zero-copy)
            chunk = source[:, offset:offset + self.n_samples]
            # Create full mask efficiently
            padding_mask = torch.ones(self.n_samples, dtype=source.dtype, device=source.device)
        elif not self.pad:
            # No padding mode - return audio at natural length
            chunk = source
            padding_mask = torch.ones(n_samples, dtype=source.dtype, device=source.device)
        else:
            # Padding needed - create chunk and fill in-place
            chunk = torch.zeros(n_channels, self.n_samples, dtype=source.dtype, device=source.device)
            chunk[:, :n_samples] = source  # Use in-place assignment

            # Create padding mask in-place
            padding_mask = torch.zeros(self.n_samples, dtype=source.dtype, device=source.device)
            padding_mask[:n_samples] = 1
        
        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

def strip_trailing_silence(audio, sample_rate, threshold_db=-60, min_silence_duration=0.1):
    """Strip silence from the end of an audio tensor.

    Args:
        audio: tensor [channels, samples]
        sample_rate: audio sample rate
        threshold_db: dB threshold below which audio is considered silent
        min_silence_duration: minimum trailing silence duration in seconds to strip
    Returns:
        Truncated audio tensor [channels, trimmed_samples], or original if no significant trailing silence
    """
    n_samples = audio.shape[-1]
    hop_length = max(1, int(sample_rate * 0.01))  # 10ms frames
    min_silence_samples = int(sample_rate * min_silence_duration)
    n_frames = n_samples // hop_length

    if n_frames == 0:
        return audio

    # Work in float32 for precision
    audio_f = audio.float()

    # Reshape into frames and compute max absolute amplitude per frame across channels
    trimmed = audio_f[:, :n_frames * hop_length]
    frames = trimmed.reshape(audio_f.shape[0], n_frames, hop_length)
    frame_peak = frames.abs().amax(dim=(0, 2))  # [n_frames] - max across channels and samples
    frame_db = 20 * torch.log10(frame_peak + 1e-10)

    # Find last frame above threshold
    above_thresh = (frame_db > threshold_db).nonzero(as_tuple=True)[0]

    if len(above_thresh) == 0:
        # Entire audio is silent
        return audio[:, :0]

    last_active_frame = above_thresh[-1].item()
    content_end = min((last_active_frame + 1) * hop_length, n_samples)

    # Only strip if trailing silence is long enough
    if (n_samples - content_end) < min_silence_samples:
        return audio

    return audio[:, :content_end]


class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal

class VolumeNorm(nn.Module):
    "Volume normalization and augmentation of a signal [LUFS standard]"
    def __init__(self, params=[-16, 2], sample_rate=16000, energy_threshold=1e-6):
        super().__init__()
        self.loudness = T.Loudness(sample_rate)
        self.value = params[0]
        self.gain_range = [-params[1], params[1]]
        self.energy_threshold = energy_threshold

    def __call__(self, signal):
        """
        signal: torch.Tensor [channels, time]
        """
        # avoid do normalisation for silence
        energy = torch.mean(signal**2)
        if energy < self.energy_threshold:
            return signal
        
        input_loudness = self.loudness(signal)
        # Generate a random target loudness within the specified range
        target_loudness = self.value + (torch.rand(1).item() * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0])
        delta_loudness = target_loudness - input_loudness
        gain = torch.pow(10.0, delta_loudness / 20.0)
        output = gain * signal

        # Check for potentially clipped samples
        if torch.max(torch.abs(output)) >= 1.0:
            output = self.declip(output)

        return output

    def declip(self, signal):
        """
        Declip the signal by scaling down if any samples are clipped
        """
        max_val = torch.max(torch.abs(signal))
        if max_val > 1.0:
            signal = signal / max_val
            signal *= 0.95
        return signal


def create_padding_mask_from_lengths(
    valid_lengths: torch.Tensor,
    total_seq_len: int,
) -> torch.Tensor:
    """
    Create a boolean padding mask from per-batch valid sequence lengths.

    Args:
        valid_lengths: Tensor of shape (batch_size,) with valid length per sample
        total_seq_len: Total sequence length of the latent

    Returns:
        Boolean tensor of shape (batch_size, total_seq_len) where True = valid, False = padding
    """
    device = valid_lengths.device
    positions = torch.arange(total_seq_len, device=device).unsqueeze(0)  # (1, T)
    padding_mask = positions < valid_lengths.unsqueeze(1)  # (B, T)
    return padding_mask


def compute_effective_seq_len_from_conditioning(
    conditioning: list,
    sample_rate: int,
    downsampling_ratio: int = 1,
    device: str = "cuda"
) -> Optional[torch.Tensor]:
    """
    Compute effective sequence lengths from seconds_total in conditioning dicts.

    Args:
        conditioning: List of conditioning dicts, one per batch element
        sample_rate: Audio sample rate
        downsampling_ratio: Pretransform downsampling ratio (1 if no pretransform)
        device: Device to place the tensor on

    Returns:
        Tensor of shape (batch_size,) with effective sequence lengths in latent space,
        or None if seconds_total is not present in conditioning
    """
    if conditioning is None:
        return None

    # Check if seconds_total is present in any conditioning dict
    if not any("seconds_total" in c for c in conditioning):
        return None

    effective_lengths = []
    for cond_dict in conditioning:
        if "seconds_total" in cond_dict:
            seconds = cond_dict["seconds_total"]
            # Convert seconds to latent sequence length
            audio_samples = int(seconds * sample_rate)
            latent_length = math.ceil(audio_samples / downsampling_ratio)
            effective_lengths.append(latent_length)
        else:
            # If seconds_total not present for this item, use None as marker
            effective_lengths.append(None)

    # If any item is missing seconds_total, return None to fall back to full length
    if any(l is None for l in effective_lengths):
        return None

    return torch.tensor(effective_lengths, dtype=torch.float32, device=device)

