import torch


class MomentumBuffer:
    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,  # [B, C, H, W]
    v1: torch.Tensor,  # [B, C, H, W]
    dims=[-1, -2],
):
    dtype = v0.dtype
    device_type = v0.device.type
    if device_type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()

    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device_type), v0_orthogonal.to(dtype).to(
        device_type
    )


def apg_forward(
    pred_cond: torch.Tensor,  # [B, C, H, W]
    pred_uncond: torch.Tensor,  # [B, C, H, W]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims=[-1, -2],
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


def cfg_forward(cond_output, uncond_output, cfg_strength):
    return uncond_output + cfg_strength * (cond_output - uncond_output)


def cfg_double_condition_forward(
    cond_output,
    uncond_output,
    only_text_cond_output,
    guidance_scale_text,
    guidance_scale_lyric,
):
    return (
        (1 - guidance_scale_text) * uncond_output
        + (guidance_scale_text - guidance_scale_lyric) * only_text_cond_output
        + guidance_scale_lyric * cond_output
    )


def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star


def cfg_zero_star(
    noise_pred_with_cond,
    noise_pred_uncond,
    guidance_scale,
    i,
    zero_steps=1,
    use_zero_init=True,
):
    bsz = noise_pred_with_cond.shape[0]
    positive_flat = noise_pred_with_cond.view(bsz, -1)
    negative_flat = noise_pred_uncond.view(bsz, -1)
    alpha = optimized_scale(positive_flat, negative_flat)
    alpha = alpha.view(bsz, 1, 1, 1)
    if (i <= zero_steps) and use_zero_init:
        noise_pred = noise_pred_with_cond * 0.0
    else:
        noise_pred = noise_pred_uncond * alpha + guidance_scale * (
            noise_pred_with_cond - noise_pred_uncond * alpha
        )
    return noise_pred
