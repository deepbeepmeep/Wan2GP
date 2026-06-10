import math
import torch
import typing as tp


class IdentityDistributionShift:
    """No-op distribution shift â€” returns timesteps unchanged."""
    def shift(self, t: torch.Tensor, seq_len):
        return t


class FluxDistributionShift:
    """Flux/SD3/Self-Flow timestep shift: t_shifted = alpha * t / (1 + (alpha-1) * t).

    Convention: t=0 is data, t=1 is noise.
    alpha > 1 shifts timesteps toward noise, appropriate for longer sequences
    where the critical structure-from-noise transition happens at higher noise levels.

    Can be used in two ways:
    - Constant alpha: set alpha_min == alpha_max. This is how the Self-Flow paper
      (BFL, 2025) uses it, with alpha chosen per modality/autoencoder.
      Reference values from the paper: audio sampleshift=6.93, trainshift=1.0;
      video sampleshift=15.0, trainshift=2.95; images sampleshift=1.78-6.93.
    - Seq_len-dependent alpha: set different alpha_min/alpha_max. Alpha is
      interpolated log-linearly in seq_len space (power-law), following the
      SD3 derivation where alpha âˆ sqrt(seq_len).

    Args:
        min_length: Minimum sequence length (alpha = alpha_min here)
        max_length: Maximum sequence length (alpha = alpha_max here)
        alpha_min: Shift factor at min_length (1.0 = no shift)
        alpha_max: Shift factor at max_length (1.0 = no shift)
    """
    def __init__(self, min_length=256, max_length=4096,
                 alpha_min=1.0, alpha_max=1.0):
        self.min_length = min_length
        self.max_length = max_length
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        # Precompute for log-linear interpolation
        self.log_alpha_min = math.log(max(alpha_min, 1e-8))
        self.log_alpha_max = math.log(max(alpha_max, 1e-8))
        self.log_min_seq = math.log(min_length)
        self.log_max_seq = math.log(max_length)
        if self.log_max_seq == self.log_min_seq:
            self.log_max_seq += 1e-8  # prevent division by zero for constant alpha

    def get_alpha(self, seq_len: tp.Union[int, torch.Tensor]):
        """Compute alpha via log-linear interpolation in seq_len."""
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.float().clamp(self.min_length, self.max_length)
            log_seq = torch.log(seq_len)
            frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
            log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
            return torch.exp(log_alpha)
        else:
            seq_len = max(min(seq_len, self.max_length), self.min_length)
            log_seq = math.log(seq_len)
            frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
            log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
            return math.exp(log_alpha)

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        """Shift timesteps based on sequence length.

        Args:
            t: Timesteps tensor of shape (batch_size,) or (steps,)
            seq_len: Either a scalar int (same shift for all elements) or
                     tensor of shape (batch_size,) for per-element shifts
        Returns:
            Shifted timesteps. If seq_len is a tensor and t is 1D with different size,
            returns shape (batch_size, steps) for per-element schedules.
        """
        alpha = self.get_alpha(seq_len)

        if isinstance(seq_len, torch.Tensor):
            alpha = alpha.to(t.device)
            if t.dim() == 1 and alpha.dim() == 1 and t.shape[0] != alpha.shape[0]:
                t = t.unsqueeze(0)
                alpha = alpha.unsqueeze(1)

        return alpha * t / (1 + (alpha - 1.0) * t)


class DistributionShift:
    def __init__(self, base_shift=0.5, max_shift=1.15, max_length=4096, min_length=256, use_sine=False):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.max_length = max_length
        self.min_length = min_length
        self.use_sine = use_sine

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        """
        Shift timesteps based on sequence length to adjust noise schedule.

        Args:
            t: Timesteps tensor of shape (batch_size,) or (steps,)
            seq_len: Either a scalar int (same shift for all elements) or
                     tensor of shape (batch_size,) for per-element shifts
        Returns:
            Shifted timesteps. If seq_len is a tensor and t is 1D with different size,
            returns shape (batch_size, steps) for per-element schedules.
        """
        if isinstance(seq_len, torch.Tensor):
            # Per-element sequence lengths
            # Ensure seq_len is on the same device as t
            seq_len = seq_len.to(t.device)
            seq_len_clamped = seq_len.float().clamp(self.min_length, self.max_length)
            # Handle broadcasting when t and seq_len have different sizes
            if t.dim() == 1 and seq_len_clamped.dim() == 1 and t.shape[0] != seq_len_clamped.shape[0]:
                # t: (steps,) -> (1, steps), seq_len: (batch,) -> (batch, 1)
                # Result: (batch, steps)
                t = t.unsqueeze(0)
                seq_len_clamped = seq_len_clamped.unsqueeze(1)
            sigma = 1.0
            mu = - (self.base_shift + (self.max_shift - self.base_shift) * (seq_len_clamped - self.min_length) / (self.max_length - self.min_length))
            t_out = 1 - torch.exp(mu) / (torch.exp(mu) + (1 / (1 - t) - 1) ** sigma)
            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)
        else:
            # Scalar path (original behavior)
            seq_len = min(max(seq_len, self.min_length), self.max_length)
            sigma = 1.0
            mu = - (self.base_shift + (self.max_shift - self.base_shift) * (seq_len - self.min_length) / (self.max_length - self.min_length))
            t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)

            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)

        return t_out


class LogSNRShift:
    """Adaptive log-SNR distribution shift.

    Maps tâˆˆ[0,1] to log-SNR-spaced values while preserving order (0â†’0, 1â†’1).
    Equivalent to applying: logsnr = linspace(logsnr_end, logsnr_start, N)
    then t = sigmoid(-logsnr), which spaces steps uniformly in log-SNR.

    logsnr_start (the high-t bound) scales with sequence length following
    the "-1 per doubling" rule:
        logsnr_start = anchor_logsnr - rate * logâ‚‚(seq_len / anchor_length)

    This captures the empirical finding that the critical log-SNR point
    (where structure emerges from noise) drops by ~rate for each doubling
    of sequence length. logsnr_end (the low-t bound) is fixed because
    low-t refinement is purely local.
    """

    def __init__(self, anchor_length=2000, anchor_logsnr=-6.2,
                 rate=1.0, logsnr_end=2.0):
        self.anchor_length = anchor_length
        self.anchor_logsnr = anchor_logsnr
        self.rate = rate
        self.logsnr_end = logsnr_end

    def get_logsnr_start(self, seq_len):
        """Compute adaptive logsnr_start: drops by `rate` per doubling of seq_len."""
        if isinstance(seq_len, torch.Tensor):
            log2_ratio = torch.log2(seq_len.float() / self.anchor_length)
            return self.anchor_logsnr - self.rate * log2_ratio
        else:
            log2_ratio = math.log2(seq_len / self.anchor_length)
            return self.anchor_logsnr - self.rate * log2_ratio

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        """Transform tâˆˆ[0,1] to log-SNR-spaced t with adaptive bounds.

        Maps through: logsnr = logsnr_end - t * (logsnr_end - logsnr_start)
                      t_out = sigmoid(-logsnr)

        Preserves order: 0â†’~0, 1â†’~1, with exact endpoint preservation.

        Args:
            t: Timesteps tensor of shape (batch_size,) or (steps,)
            seq_len: Either a scalar int or tensor of shape (batch_size,)
        Returns:
            Log-SNR-spaced timesteps in [0, 1].
        """
        t_original = t
        logsnr_start = self.get_logsnr_start(seq_len)

        if isinstance(seq_len, torch.Tensor):
            logsnr_start = logsnr_start.to(t.device)
            if t.dim() == 1 and logsnr_start.dim() == 1 and t.shape[0] != logsnr_start.shape[0]:
                t = t.unsqueeze(0)
                logsnr_start = logsnr_start.unsqueeze(1)

        # Map t through log-SNR space (monotonically: low t â†’ high logsnr â†’ low t_out)
        logsnr = self.logsnr_end - t * (self.logsnr_end - logsnr_start)
        t_out = torch.sigmoid(-logsnr)

        # Preserve exact endpoints
        t_out = torch.where(t_original <= 0, torch.zeros_like(t_out), t_out)
        t_out = torch.where(t_original >= 1, torch.ones_like(t_out), t_out)

        return t_out

