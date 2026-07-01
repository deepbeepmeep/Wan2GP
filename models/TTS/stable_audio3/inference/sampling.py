import torch
import typing as tp
from tqdm import trange, tqdm
import torch.distributions as dist

from ..data.utils import create_padding_mask_from_lengths, compute_effective_seq_len_from_conditioning


def build_schedule(
    steps: int,
    sigma_max: float = 1.0,
    dist_shift = None,
    effective_seq_len: tp.Union[int, torch.Tensor, None] = None,
    fallback_seq_len: tp.Optional[int] = None,
    include_endpoint: bool = True,
    device: tp.Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build a timestep schedule for diffusion sampling.

    Returns a 1D tensor of shape (N,) where N = steps+1 (if include_endpoint)
    or steps (if not), OR a 2D tensor of shape (batch_size, N) when
    effective_seq_len is a tensor and dist_shift produces per-element schedules.

    Args:
        steps: Number of sampling steps.
        sigma_max: Starting noise level (1.0 for full generation, <1.0 for variations).
        dist_shift: Optional distribution shift object (FluxDistributionShift,
            DistributionShift, LogSNRShift, etc.). Applied to warp the linear schedule.
        effective_seq_len: Sequence length for dist_shift. Scalar int or
            tensor of shape (batch_size,) for per-element schedules.
        fallback_seq_len: Fallback when effective_seq_len is None (typically x.shape[-1]).
        include_endpoint: If True, schedule includes 0 as final value (RF samplers).
            If False, excludes 0 (v-diffusion DDIM).
        device: Device for the output tensor.
    """
    n_points = steps + 1 if include_endpoint else steps

    if include_endpoint:
        t = torch.linspace(sigma_max, 0, n_points, device=device)
    else:
        t = torch.linspace(sigma_max, 0, n_points + 1, device=device)[:-1]

    if dist_shift is not None:
        seq_len = effective_seq_len if effective_seq_len is not None else fallback_seq_len
        if isinstance(seq_len, torch.Tensor):
            # Clamp per-element sequence lengths to avoid zeros causing log/NaN issues
            seq_len = torch.clamp(seq_len, min=1)
        elif seq_len is not None:
            # Clamp scalar sequence length to at least 1
            seq_len = max(int(seq_len), 1)
        t = dist_shift.shift(t, seq_len)

        # Ensure the first timestep remains aligned with sigma_max after shifting.
        # This keeps the schedule consistent with the initialization in sample_diffusion(),
        # which mixes init_data using sigma_max.
        if isinstance(t, torch.Tensor):
            sigma_max_tensor = t.new_tensor(sigma_max)
            if t.ndim == 1:
                t[0] = sigma_max_tensor
            else:
                # For batched/per-element schedules, enforce sigma_max at the first time index.
                t[..., 0] = sigma_max_tensor

    return t


def sample_timesteps_logsnr(batch_size, mean_logsnr=-1.2, std_logsnr=2.0):
    """
    Sample timesteps for diffusion training by sampling logSNR values and converting to t.

    Args:
        batch_size (int): Number of timesteps to sample
        mean_logsnr (float): Mean of the logSNR Gaussian distribution
        std_logsnr (float): Standard deviation of the logSNR Gaussian distribution

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing timestep values t in [0, 1]
    """
    # Sample logSNR from Gaussian distribution
    logsnr = torch.randn(batch_size) * std_logsnr + mean_logsnr

    # Convert logSNR to timesteps using the logistic function
    # Since logSNR = ln((1-t)/t), we can solve for t:
    # t = 1 / (1 + exp(logsnr))
    t = torch.sigmoid(-logsnr)

    # Clamp values to ensure numerical stability
    t = t.clamp(1e-4, 1 - 1e-4)

    return t

def sample_timesteps_logsnr_uniform(batch_size, min_logsnr=-6, max_logsnr=5.0):
    """
    Sample timesteps for diffusion training by sampling logSNR values and converting to t.

    Args:
        batch_size (int): Number of timesteps to sample
        min_logsnr (float): Minimum logSNR value
        max_logsnr (float): Maximum logSNR value

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing timestep values t in [0, 1]
    """
    # Sample logSNR from uniform distribution
    logsnr = torch.rand(batch_size) * (max_logsnr - min_logsnr) + min_logsnr

    # Convert logSNR to timesteps using the logistic function
    # Since logSNR = ln((1-t)/t), we can solve for t:
    # t = 1 / (1 + exp(logsnr))
    t = torch.sigmoid(-logsnr)

    # Clamp values to ensure numerical stability
    t = t.clamp(1e-4, 1 - 1e-4)

    return t

def truncated_logistic_normal_rescaled(shape, left_trunc=0.075, right_trunc=1):
    """

    shape: shape of the output tensor
    left_trunc: left truncation point, fraction of probability to be discarded
    right_trunc: right truncation boundary, should be 1 (never seen at test time)
    """

    # Step 1: Sample from the logistic normal distribution (sigmoid of normal)
    logits = torch.randn(shape)

    # Step 2: Apply the CDF transformation of the normal distribution
    normal_dist = dist.Normal(0, 1)
    cdf_values = normal_dist.cdf(logits)

    # Step 3: Define the truncation bounds on the CDF
    lower_bound = normal_dist.cdf(torch.logit(torch.tensor(left_trunc)))
    upper_bound = normal_dist.cdf(torch.logit(torch.tensor(right_trunc)))

    # Step 4: Rescale linear CDF values into the truncated region (between lower_bound and upper_bound)
    truncated_cdf_values = lower_bound + (upper_bound - lower_bound) * cdf_values

    # Step 5: Map back to logistic-normal space using inverse CDF
    truncated_samples = torch.sigmoid(normal_dist.icdf(truncated_cdf_values))

    # Step 6: Rescale values so that min is 0 and max is just below 1
    rescaled_samples = (truncated_samples - left_trunc) / (right_trunc - left_trunc)

    return rescaled_samples

def sample_discrete_euler(model, x, sigmas, callback=None, disable_tqdm=False, abort_fn=None, **extra_args):
    """Draws samples from a model given starting noise. Euler method

    Args:
        sigmas: Pre-computed schedule tensor. Shape (steps+1,) for global schedule
            or (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        if callable(abort_fn) and abort_fn():
            raise InterruptedError("Stable Audio 3 generation aborted.")
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr_tensor = t[:, i].to(x.dtype)  # (batch_size,)
            t_prev = t[:, i + 1].to(x.dtype)  # (batch_size,)
            dt = t_prev - t_curr_tensor  # (batch_size,)
            # Reshape for broadcasting with x: (batch_size,) -> (batch_size, 1, 1)
            dt_broadcast = dt.view(-1, 1, 1)
        else:
            # Global schedule: t has shape (steps+1,)
            t_curr = t[i]
            t_prev = t[i + 1]
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
            dt = t_prev - t_curr
            dt_broadcast = dt

        v = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr_tensor[:, None, None] * v
            callback({'x': x, 't': t_curr_tensor, 'sigma': t_curr_tensor, 'i': i, 'denoised': denoised})

        x = x + dt_broadcast * v
        if callable(abort_fn) and abort_fn():
            raise InterruptedError("Stable Audio 3 generation aborted.")

    # If we are on the last timestep, output the denoised data
    return x

def sample_rk4(model, x, sigmas, callback=None, disable_tqdm=False, abort_fn=None, **extra_args):
    """Draws samples from a model given starting noise. 4th-order Runge-Kutta

    Args:
        sigmas: Pre-computed schedule tensor of shape (steps+1,).
            Per-element schedules not supported for RK4.
    """
    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    t = sigmas

    t = t.to(x.device)

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm)):
        if callable(abort_fn) and abort_fn():
            raise InterruptedError("Stable Audio 3 generation aborted.")
        # Broadcast the current timestep to the correct shape
        t_curr_tensor = t_curr * ts
        dt = t_prev - t_curr  # we solve backwards in our formulation

        k1 = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr * k1
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i, 'denoised': denoised})

        k2 = model(x + dt / 2 * k1, (t_curr + dt / 2) * ts, **extra_args)
        k3 = model(x + dt / 2 * k2, (t_curr + dt / 2) * ts, **extra_args)

        # Clamp t_prev to avoid evaluating model at exactly t=0
        # (models aren't trained at t=0 and may return garbage/NaN)
        t_prev_eval = t_prev.clamp(min=1e-5)
        k4 = model(x + dt * k3, t_prev_eval * ts, **extra_args)

        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if callable(abort_fn) and abort_fn():
            raise InterruptedError("Stable Audio 3 generation aborted.")

    # If we are on the last timestep, output the denoised data
    return x

def sample_flow_dpmpp(model, x, sigmas, callback=None, disable_tqdm=False, abort_fn=None, **extra_args):
    """Draws samples from a model given starting noise. DPM-Solver++ for RF models

    Args:
        sigmas: Pre-computed schedule tensor. Shape (steps+1,) for global schedule
            or (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    old_denoised = None

    # Clamp t to avoid numerical issues with log(0) and division by zero
    # This prevents inf/-inf values that can cause NaN propagation
    log_snr = lambda t: ((1-t).clamp(min=1e-10) / t.clamp(min=1e-10)).log()

    for i in trange(num_steps, disable=disable_tqdm):
        if callable(abort_fn) and abort_fn():
            raise InterruptedError("Stable Audio 3 generation aborted.")
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr = t[:, i]  # (batch_size,)
            t_next = t[:, i + 1]  # (batch_size,)
            t_prev = t[:, i - 1] if i > 0 else None
            # Reshape for broadcasting with x: (batch_size,) -> (batch_size, 1, 1)
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
            t_curr_tensor = t_curr  # already (batch_size,)
        else:
            # Global schedule: t has shape (steps+1,)
            t_curr = t[i]
            t_next = t[i + 1]
            t_prev = t[i - 1] if i > 0 else None
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next
            t_curr_tensor = t_curr.expand(x.shape[0])

        model_output = model(x, t_curr_tensor, **extra_args)
        denoised = x - t_curr_broadcast * model_output

        if callback is not None:
            callback({'x': x, 'i': i, 't': t_curr, 'sigma': t_curr, 'denoised': denoised})

        alpha_t = 1 - t_next_broadcast

        # For rectified flow, compute the DPM++ coefficient directly without log_snr
        # to avoid numerical issues at t=0 or t=1
        # The formula is: (-h).expm1() = (t_next - t_curr) / [(1 - t_next) * t_curr]
        # Note: t_next < t_curr, so this is negative
        # We'll compute this directly instead of going through log_snr
        dt = t_next_broadcast - t_curr_broadcast
        # Clamp to avoid division by zero when t_curr or t_next are at boundaries
        dpmpp_coeff = dt / ((1 - t_next_broadcast).clamp(min=1e-10) * t_curr_broadcast.clamp(min=1e-10))

        # Check if this is the first step or the last step (t_next == 0)
        is_first_step = old_denoised is None
        is_last_step = (t_next_broadcast == 0).all() if per_element_schedule else (t_next == 0)

        if is_first_step or is_last_step:
            # First-order update using the directly computed coefficient
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised
        else:
            # Second-order update with Richardson extrapolation
            if per_element_schedule:
                t_prev_broadcast = t_prev.view(-1, 1, 1)
            else:
                t_prev_broadcast = t_prev
            # Compute r = h_last / h in log-SNR space for second-order correction
            # h = log_snr(t_next) - log_snr(t_curr), h_last = log_snr(t_curr) - log_snr(t_prev)
            h = log_snr(t_next_broadcast) - log_snr(t_curr_broadcast)
            h_last = log_snr(t_curr_broadcast) - log_snr(t_prev_broadcast)
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised_d

        old_denoised = denoised
        if callable(abort_fn) and abort_fn():
            raise InterruptedError("Stable Audio 3 generation aborted.")
    return x

def sample_flow_pingpong(model, x, sigmas, callback=None, disable_tqdm=False, abort_fn=None, **extra_args):
    """Draws samples from a model given starting noise. Ping-pong sampling for distilled models

    Args:
        sigmas: Pre-computed schedule tensor. Shape (steps+1,) for global schedule
            or (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    for i in trange(num_steps, disable=disable_tqdm):
        if callable(abort_fn) and abort_fn():
            raise InterruptedError("Stable Audio 3 generation aborted.")
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr = t[:, i].to(x.dtype)  # (batch_size,)
            t_next = t[:, i + 1].to(x.dtype)  # (batch_size,)
            # Reshape for broadcasting with x: (batch_size,) -> (batch_size, 1, 1)
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
        else:
            # Global schedule: t has shape (steps+1,)
            t_curr = t[i].to(x.dtype)
            t_next = t[i + 1].to(x.dtype)
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next

        # Model forward
        if per_element_schedule:
            t_curr_tensor = t_curr  # already (batch_size,)
        else:
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)

        denoised = x - t_curr_broadcast * model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 't': t_curr, 'sigma': t_curr, 'sigma_hat': t_curr, 'denoised': denoised})

        x = (1 - t_next_broadcast) * denoised + t_next_broadcast * torch.randn_like(x)
        if callable(abort_fn) and abort_fn():
            raise InterruptedError("Stable Audio 3 generation aborted.")

    return x



@torch.no_grad()
def sample_diffusion(
    model,
    noise: torch.Tensor,
    cond_inputs: dict,
    diffusion_objective: str,
    steps: int,
    cfg_scale: float = 1.0,
    # Varlen support
    conditioning: tp.Optional[tp.List[dict]] = None,
    sample_rate: int = 44100,
    pretransform = None,
    mask_padding_attention: bool = False,
    use_effective_length_for_schedule: bool = False,
    headroom_seconds: float = 5.0,
    padding_mask: tp.Optional[torch.Tensor] = None,
    # Timestep schedule
    dist_shift = None,
    # Sampler options
    sampler_type: str = None,
    batch_cfg: bool = True,
    rescale_cfg: bool = False,
    # CFG options
    apg_scale: float = 1.0,
    # Init data (variation / img2img)
    init_data: tp.Optional[torch.Tensor] = None,
    init_noise_level: float = 1.0,
    # Other
    callback = None,
    disable_tqdm: bool = False,
    abort_fn = None,
    decode: bool = True,
    chunked_decode: tp.Optional[bool] = None,
    **sampler_kwargs
) -> torch.Tensor:
    """
    Unified sampling function for diffusion models. Handles all diffusion objectives,
    varlen support (padding_mask + effective_seq_len), timestep scheduling, and init_data
    for variation/img2img.

    Args:
        model: The diffusion model backbone (model.model, not the wrapper)
        noise: Initial noise tensor of shape (B, C, T)
        cond_inputs: Pre-processed conditioning inputs dict (merged positive + negative)
        diffusion_objective: One of "v", "rectified_flow", "rf_denoiser"
        steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale
        conditioning: List of conditioning dicts (for computing varlen from seconds_total)
        sample_rate: Audio sample rate
        pretransform: Optional pretransform for decoding latents and computing downsampling_ratio
        mask_padding_attention: Whether to create padding_mask for attention
        use_effective_length_for_schedule: Whether to use effective_seq_len for dist_shift
        padding_mask: Optional pre-computed padding mask (B, T). If provided, skips
            internal mask computation. Use this to ensure consistency with training masks.
        headroom_seconds: Extra seconds beyond seconds_total for valid region
        dist_shift: Distribution shift object for warping the timestep schedule, or None
        sampler_type: Sampler type. For RF: "euler", "rk4", "dpmpp", "pingpong".
            For v-diffusion: "v-ddim", "v-ddim-cfgpp", or k-diffusion types like "dpmpp-2m-sde".
        batch_cfg: Whether to use batched CFG
        rescale_cfg: Whether to use rescaled CFG
        apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG
        init_data: Optional pre-encoded latent tensor for variation/img2img (shape: B, C, T)
        init_noise_level: Noise level (sigma_max) when using init_data. 1.0 = full noise (no variation).
        callback: Optional callback for progress reporting
        disable_tqdm: Whether to disable progress bar
        decode: Whether to decode latents using pretransform
        **sampler_kwargs: Additional kwargs passed to sampler

    Returns:
        Generated samples (decoded audio if decode=True, else latents)
    """
    device = noise.device
    batch_size = noise.shape[0]
    latent_seq_len = noise.shape[-1]

    # Compute downsampling ratio
    downsampling_ratio = pretransform.downsampling_ratio if pretransform is not None else 1

    # Default sampler_type per objective
    if sampler_type is None:
        sampler_type = "pingpong" if diffusion_objective == "rf_denoiser" else "euler"


    # Compute effective_seq_len for dist_shift if enabled
    effective_seq_len = None
    if use_effective_length_for_schedule and conditioning is not None:
        effective_seq_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )

    # Create padding_mask for attention if enabled (skip if pre-computed mask provided)
    if padding_mask is None and mask_padding_attention and conditioning is not None:
        raw_effective_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )
        if raw_effective_len is not None:
            headroom_tokens = int(headroom_seconds * sample_rate / downsampling_ratio)
            valid_lengths = (raw_effective_len + headroom_tokens).clamp(max=latent_seq_len).long()
            padding_mask = create_padding_mask_from_lengths(valid_lengths, latent_seq_len)

    # Determine sigma_max for schedule
    sigma_max = init_noise_level if init_data is not None else 1.0

    # Mix init_data with noise for variation/img2img
    # For k-diffusion v-diffusion samplers, init_data is passed through to sample_k
    # which handles mixing internally with its own sigma scaling
    k_diff_sampler_types = {"k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2",
                            "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m"}

    if init_data is not None:
        noise = init_data * (1 - sigma_max) + noise * sigma_max

    # Build common sampler kwargs (conditioning + model-level params only).
    # disable_tqdm and callback are passed explicitly to samplers that use them,
    # not included here, to avoid leaking into model forward() calls.
    common_kwargs = {
        **cond_inputs,
        "cfg_scale": cfg_scale,
        "batch_cfg": batch_cfg,
        "rescale_cfg": rescale_cfg,
        "padding_mask": padding_mask,
        "apg_scale": apg_scale,
        **sampler_kwargs
    }


    if diffusion_objective in ["rectified_flow", "rf_denoiser"]:
        # Remove v-diffusion-specific kwargs that don't apply to RF
        common_kwargs.pop("sigma_min", None)
        common_kwargs.pop("sigma_max", None)
        common_kwargs.pop("rho", None)

        # Build schedule
        sigmas = build_schedule(
            steps=steps, sigma_max=sigma_max,
            dist_shift=dist_shift, effective_seq_len=effective_seq_len,
            fallback_seq_len=latent_seq_len, include_endpoint=True, device=device
        )

        # Route to sampler
        if sampler_type == "euler":
            sampled = sample_discrete_euler(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, abort_fn=abort_fn, **common_kwargs)
        elif sampler_type == "rk4":
            sampled = sample_rk4(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, abort_fn=abort_fn, **common_kwargs)
        elif sampler_type == "dpmpp":
            sampled = sample_flow_dpmpp(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, abort_fn=abort_fn, **common_kwargs)
        elif sampler_type == "pingpong":
            sampled = sample_flow_pingpong(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, abort_fn=abort_fn, **common_kwargs)
        else:
            raise ValueError(f"Unknown sampler_type for {diffusion_objective}: {sampler_type}")

    else:
        raise ValueError(f"Unknown diffusion_objective: {diffusion_objective}")

    # Decode if requested
    if decode and pretransform is not None:
        sampled = sampled.to(next(pretransform.parameters()).dtype)
        sampled = pretransform.decode(sampled, chunked=chunked_decode)

        # Zero out audio beyond valid region (padding positions decode to garbage)
        if padding_mask is not None:
            audio_mask = padding_mask.unsqueeze(1).repeat_interleave(downsampling_ratio, dim=-1)
            # Trim or pad to match sampled length
            if audio_mask.shape[-1] > sampled.shape[-1]:
                audio_mask = audio_mask[..., :sampled.shape[-1]]
            elif audio_mask.shape[-1] < sampled.shape[-1]:
                audio_mask = torch.nn.functional.pad(audio_mask, (0, sampled.shape[-1] - audio_mask.shape[-1]), value=False)
            sampled = sampled * audio_mask.to(sampled.dtype)

    return sampled
