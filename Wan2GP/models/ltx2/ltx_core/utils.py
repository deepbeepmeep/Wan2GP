from typing import Any

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6, in_place = False) -> torch.Tensor:
    # deepbeepmeep RMS Norm
    dtype = x.dtype
    y = x.float()
    y.pow_(2)
    y = y.mean(dim=-1, keepdim=True)
    y += eps
    y.rsqrt_()
    if in_place:
        x *=  y
    else:
        x = x * y.to(dtype)
    if weight is not None:
        x *= weight
    return x

    # return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def check_config_value(config: dict, key: str, expected: Any) -> None:  # noqa: ANN401
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoised version to velocity.
    Returns:
        Velocity
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.
    Returns:
        Denoised sample
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)
