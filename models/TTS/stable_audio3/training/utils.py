from pytorch_lightning.loggers import WandbLogger, CometLogger
from ..interface.aeiou import pca_point_cloud

import math
import wandb
import torch
import torch.nn.functional as F
import os
import typing as tp

def get_rank():
    """Get rank of current process."""

    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0

    return torch.distributed.get_rank()

class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0.,
                 last_epoch=-1):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]

def create_optimizer_from_config(optimizer_config, parameters):
    """Create optimizer from config.

    Args:
        parameters (iterable): parameters to optimize.
        optimizer_config (dict): optimizer config.

    Returns:
        torch.optim.Optimizer: optimizer.
    """

    optimizer_type = optimizer_config["type"]

    if optimizer_type == "FusedAdam":
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(parameters, **optimizer_config["config"])
    elif optimizer_type == "CAdamW":
        from stable_audio_tools.training.optims import CAdamW
        optimizer = CAdamW(parameters, **optimizer_config["config"])
    elif optimizer_type == "CLion":
        from stable_audio_tools.training.optims import CLion
        optimizer = CLion(parameters, **optimizer_config["config"])
    elif optimizer_type == "AdamW8bit":
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(parameters, **optimizer_config["config"])
    elif optimizer_type == "MuonAdamW":
        from stable_audio_tools.training.optims import MuonAdamW
        optimizer = MuonAdamW(parameters, **optimizer_config["config"])
    else:
        optimizer_fn = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_fn(parameters, **optimizer_config["config"])
    return optimizer

def create_scheduler_from_config(scheduler_config, optimizer):
    """Create scheduler from config.

    Args:
        scheduler_config (dict): scheduler config.
        optimizer (torch.optim.Optimizer): optimizer.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: scheduler.
    """
    if scheduler_config["type"] == "InverseLR":
        scheduler_fn = InverseLR
    else:
        scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    scheduler = scheduler_fn(optimizer, **scheduler_config["config"])
    return scheduler

def logger_project_name(logger) -> str:
    if isinstance(logger, WandbLogger):
        return logger.experiment.project
    elif isinstance(logger, CometLogger):
        return logger.name

def log_metric(logger, key, value, step=None):
    from pytorch_lightning.loggers import WandbLogger, CometLogger
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: value})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_metrics({key: value}, step=step)

def log_audio(logger, key, audio_path, sample_rate, caption=None, step=None):
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Audio(audio_path, sample_rate=sample_rate, caption=caption)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_audio(audio_path, file_name=key, sample_rate=sample_rate, step=step)

def log_image(logger, key, img_data, step=None):
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Image(img_data)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_image(img_data, name=key, step=step)

def log_point_cloud(logger, key, tokens, caption=None):
    if isinstance(logger, WandbLogger):
        point_cloud = pca_point_cloud(tokens)
        logger.experiment.log({key: point_cloud})
    elif isinstance(logger, CometLogger):
        point_cloud = pca_point_cloud(tokens, rgb_float=True, output_type="points")
        #logger.experiment.log_points_3d(scene_name=key, points=point_cloud)


def compute_per_elem_trim(conditioning, sample_rate, margin_seconds=5.0):
    """Compute per-element trim lengths from seconds_total in conditioning dicts.

    Returns a list of trim lengths (in audio samples) or None if no seconds_total found.
    """
    if not any("seconds_total" in c for c in conditioning):
        return None
    margin_samples = int(margin_seconds * sample_rate)
    per_elem_trim = []
    for c in conditioning:
        if "seconds_total" in c:
            per_elem_trim.append(int(c["seconds_total"] * sample_rate) + margin_samples)
        else:
            per_elem_trim.append(None)
    return per_elem_trim

def trim_and_concat(x, per_elem_trim):
    """Per-element trim and concatenate along time axis.

    Trims each batch element to its own length (from seconds_total + margin),
    removing trailing padding silence before concatenation.

    Args:
        x: (b, d, n) tensor or list of (d, n) tensors
        per_elem_trim: list of trim lengths per element, or None for no trimming
    """
    items = [x[i] for i in range(x.shape[0])] if isinstance(x, torch.Tensor) and x.dim() == 3 else x
    if per_elem_trim is None:
        return torch.cat(items, dim=-1)
    parts = []
    for i, elem in enumerate(items):
        if per_elem_trim[i] is not None:
            parts.append(elem[..., :min(per_elem_trim[i], elem.shape[-1])])
        else:
            parts.append(elem)
    return torch.cat(parts, dim=-1)


class StaggeredLogger:
    """Accumulates log values over N steps and flushes averaged metrics.

    Avoids per-step CUDA syncs (.item()) and prevents PL from caching
    stale values between flush intervals by logging directly to the
    experiment logger.

    Usage in a PL module::

        # In __init__:
        self._staggered_logger = StaggeredLogger(every_n_steps=10)

        # At end of training_step (pass detached tensors or Python scalars):
        self._staggered_logger.log(log_dict, self)
    """

    def __init__(self, every_n_steps: int = 10):
        self.every_n_steps = every_n_steps
        self._accum: dict = {}
        self._counts: dict = {}
        self._steps: int = 0

    def log(self, log_dict: dict, module) -> None:
        """Accumulate *log_dict* and flush every *every_n_steps* calls.

        NaN/inf values are handled at flush time: if an accumulated sum
        is non-finite (due to a NaN step poisoning the window), that
        metric is simply omitted from the flush.  This avoids the
        expensive per-step CUDA syncs that an eager ``isfinite`` check
        would require.

        Args:
            log_dict: Mapping of metric names to detached GPU tensors or
                Python scalars.  Tensors are kept on-device until flush.
            module: The ``pl.LightningModule`` instance (used for
                ``log_dict``, ``logger``, and ``global_step``).
        """
        for k, v in log_dict.items():
            val = v.detach() if torch.is_tensor(v) else v
            if k not in self._accum:
                self._accum[k] = val
                self._counts[k] = 1
            else:
                self._accum[k] = self._accum[k] + val
                self._counts[k] += 1
        self._steps += 1

        if self._steps >= self.every_n_steps:
            flushed = {}
            for k, v in self._accum.items():
                c = self._counts[k]
                if c > 0:
                    val = v.item() / c if torch.is_tensor(v) else v / c
                    # Skip non-finite values (NaN from a bad step poisoned this window)
                    if math.isfinite(val):
                        flushed[k] = val
            # Update progress bar only (logger=False avoids PL caching stale values)
            if flushed:
                module.log_dict(flushed, prog_bar=True, on_step=True, logger=False)
                # Log directly to experiment (bypasses PL metric caching)
                if module.logger is not None:
                    module.logger.log_metrics(flushed, step=module.global_step)
            self._accum = {}
            self._counts = {}
            self._steps = 0


def resize_padding_mask(padding_mask: torch.Tensor, target_length: int) -> torch.Tensor:
    """Resize a padding mask to target_length using ceiling-based length scaling.

    Unlike F.interpolate(mode="nearest"), this ensures any target position
    that partially overlaps valid audio is marked valid (rounds up).

    Args:
        padding_mask: (B, T) or (T,) tensor where 1/True = valid, 0/False = padding
        target_length: desired output length
    Returns:
        Resized boolean mask with same ndim as input
    """
    if padding_mask.ndim == 1:
        valid_length = padding_mask.sum()
        source_length = padding_mask.shape[0]
        valid_target_length = torch.ceil(
            valid_length.float() * target_length / source_length
        ).long().clamp(max=target_length)
        positions = torch.arange(target_length, device=padding_mask.device)
        return positions < valid_target_length
    else:
        valid_lengths = padding_mask.sum(dim=-1)  # (B,)
        source_length = padding_mask.shape[-1]
        valid_target_lengths = torch.ceil(
            valid_lengths.float() * target_length / source_length
        ).long().clamp(max=target_length)
        positions = torch.arange(target_length, device=padding_mask.device).unsqueeze(0)
        return positions < valid_target_lengths.unsqueeze(1)


def create_augmented_padding_mask(
    padding_mask: torch.Tensor,
    silence_extension_scale_seconds: float = 0.0,
    sample_rate: int = 44100,
    downsampling_ratio: int = 1,
) -> torch.Tensor:
    """
    Augment padding mask by randomly extending the valid region to include silence.

    This helps the model learn to handle silence at the end of audio by randomly
    including some of the padded (silent) region as valid.

    Uses an exponential distribution so most extensions are small (within ~scale seconds)
    but can occasionally extend to the full sequence length.

    Args:
        padding_mask: Boolean tensor of shape (batch, seq_len) where True = valid audio
        silence_extension_scale_seconds: Scale (mean) of exponential distribution in seconds.
            ~63% of samples within 1x scale, ~86% within 2x, ~95% within 3x.
        sample_rate: Audio sample rate (used to convert seconds to tokens)
        downsampling_ratio: Downsampling ratio from pretransform (to convert samples to latent tokens)

    Returns:
        Augmented padding mask of shape (batch, seq_len) where True = valid region (audio + extended silence)
    """
    if silence_extension_scale_seconds <= 0:
        return padding_mask

    batch_size, seq_len = padding_mask.shape
    device = padding_mask.device

    # Find the last valid position for each batch element
    valid_lengths = padding_mask.sum(dim=-1)  # (batch,)

    # Calculate scale in tokens for exponential distribution
    scale_tokens = silence_extension_scale_seconds * sample_rate / downsampling_ratio

    # Sample from exponential distribution (biased toward smaller values)
    # Exponential: ~63% within 1x scale, ~86% within 2x, ~95% within 3x
    random_extensions = torch.empty(batch_size, device=device).exponential_(lambd=1.0/scale_tokens)
    random_extensions = random_extensions.long()

    # Extend valid lengths, clamping to sequence length (allows extension to full padding)
    augmented_lengths = torch.clamp(valid_lengths + random_extensions, max=seq_len)

    # Create augmented mask: True where position < augmented_length
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
    augmented_mask = positions < augmented_lengths.unsqueeze(1)  # (batch, seq_len)

    return augmented_mask


def compute_normalized_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_normalization: str = "none",
    loss_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute MSE normalized by detached target magnitude.

    Prevents high-magnitude latents from dominating the loss,
    ensuring quiet passages receive equal learning signal.

    Normalization is only applied to valid (non-padded) regions.
    Padded regions retain raw MSE for compatibility with mask_loss_weight.

    Args:
        pred: [B, C, T] predicted tensor
        target: [B, C, T] target tensor
        loss_mask: [B, T] boolean, True = valid (non-padded), or None
        loss_normalization: One of "none", "timestep", "sample", "sample_channel"
        loss_norm_eps: Epsilon for numerical stability

    Returns:
        Per-element MSE [B, C, T], normalized for signal, raw for padding
    """
    mse = (pred - target) ** 2  # [B, C, T]

    if loss_normalization == "none":
        return mse

    if loss_mask is None:
        mask_expanded = torch.ones(pred.shape[0], 1, pred.shape[2], device=pred.device, dtype=torch.bool)
    else:
        mask_expanded = loss_mask.unsqueeze(1)  # [B, 1, T]

    with torch.no_grad():
        if loss_normalization == "timestep":
            # Var across channels at each timestep [B, 1, T]
            mag_sq = torch.mean((target - torch.mean(target, dim=1, keepdim=True)) ** 2, dim=1, keepdim=True) + loss_norm_eps
        else:
            # For sample/sample_channel modes, exclude padding via NaN masking
            masked_targets = torch.where(mask_expanded, target, float('nan'))

            if loss_normalization == "sample":
                sample_mean = torch.nanmean(masked_targets, dim=(1, 2), keepdim=True)
                mag_sq = torch.nanmean((masked_targets - sample_mean) ** 2, dim=(1, 2), keepdim=True) + loss_norm_eps
            elif loss_normalization == "sample_channel":
                channel_mean = torch.nanmean(masked_targets, dim=2, keepdim=True)
                mag_sq = torch.nanmean((masked_targets - channel_mean) ** 2, dim=2, keepdim=True) + loss_norm_eps
            else:
                raise ValueError(f"Unknown loss normalization mode: {loss_normalization}")

            # Handle edge case where entire sample is padding (all NaN -> NaN variance)
            mag_sq = torch.where(torch.isnan(mag_sq), torch.ones_like(mag_sq), mag_sq)

    # Only normalize signal regions; keep padding MSE raw for mask_loss_weight compatibility
    normalized_mse = mse / mag_sq
    return torch.where(mask_expanded, normalized_mse, mse)


def compute_masked_loss(
    loss_full: torch.Tensor,
    loss_mask: torch.Tensor,
    mask_padding_attention: bool,
    mask_loss_weight: float = 0.0,
) -> tuple:
    """
    Compute loss with separate signal and padding contributions.

    Args:
        loss_full: Full loss tensor of shape (B, C, T)
        loss_mask: Boolean mask of shape (B, T) where True = signal, False = padding
        mask_padding_attention: If True, only compute loss on signal region
        mask_loss_weight: Weight for padding loss when mask_padding_attention is False

    Returns:
        Tuple of (loss, signal_mean, padding_mean) where:
            - loss: The final scalar loss
            - signal_mean: Mean loss on signal region (for logging)
            - padding_mean: Mean loss on padding region (for logging)
    """
    # Compute separate signal and padding losses
    signal = torch.where(loss_mask.unsqueeze(1), loss_full, 0.0)
    signal_sum = signal.sum(dim=(1, 2))
    n_channels = loss_full.shape[1]
    signal_count = loss_mask.sum(dim=1) * n_channels

    padding = torch.where(~loss_mask.unsqueeze(1), loss_full, 0.0)
    padding_sum = padding.sum(dim=(1, 2))
    padding_count = (~loss_mask).sum(dim=1) * n_channels

    if mask_padding_attention:
        # When attention masking is on, only compute loss on signal region
        # (padding saw no attention, so loss there is meaningless)
        per_sample_loss = signal_sum / (signal_count + 1e-8)
        loss = per_sample_loss.mean()
    else:
        # Weighted loss: signal + mask_loss_weight * padding
        w = mask_loss_weight
        denom = signal_count + w * padding_count + 1e-8
        per_sample_loss = (signal_sum + w * padding_sum) / denom
        loss = per_sample_loss.mean()

    # Compute separate means for logging
    signal_mean = signal.sum() / (signal_count.sum() + 1e-8)
    padding_mean = padding.sum() / (padding_count.sum() + 1e-8)

    return loss, signal_mean.detach(), padding_mean.detach()


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean over masked (valid) positions only.

    Args:
        values: Tensor of shape (B, C, T) or (B, T)
        mask: Boolean mask of shape (B, T) where True = valid, False = padding

    Returns:
        Scalar mean over valid positions only
    """
    if values.ndim == 2:
        # (B, T) -> add channel dim
        values = values.unsqueeze(1)

    # Expand mask to match values shape: (B, T) -> (B, 1, T)
    mask_expanded = mask.unsqueeze(1)

    # Zero out padding positions
    masked_values = torch.where(mask_expanded, values, torch.zeros_like(values))

    # Count valid positions (per sample, times channels)
    n_channels = values.shape[1]
    valid_count = mask.sum(dim=1) * n_channels  # (B,)

    # Sum over all dimensions and divide by valid count
    per_sample_sum = masked_values.sum(dim=(1, 2))  # (B,)
    per_sample_mean = per_sample_sum / (valid_count + 1e-8)

    return per_sample_mean.mean()


def masked_sum(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: tp.List[int],
) -> torch.Tensor:
    """
    Compute sum over masked (valid) positions only.

    Args:
        values: Tensor of shape (B, C, T) or (B, T)
        mask: Boolean mask of shape (B, T) where True = valid, False = padding
        dim: Dimensions to sum over (should include the time dimension)

    Returns:
        Sum over valid positions, reduced over specified dims
    """
    if values.ndim == 2:
        # (B, T) -> add channel dim for consistency
        mask_expanded = mask
    else:
        # (B, C, T) -> expand mask
        mask_expanded = mask.unsqueeze(1)

    # Zero out padding positions
    masked_values = torch.where(mask_expanded, values, torch.zeros_like(values))

    return masked_values.sum(dim=tuple(dim))
