# Flow Euler sampler with classifier-free guidance, guidance interval and CFG rescale, ported from microsoft/TRELLIS.2
# (trellis2/pipelines/samplers/flow_euler.py, classifier_free_guidance_mixin.py, guidance_interval_mixin.py, MIT License).
# The arithmetic keeps upstream's operation order on FP32 tensors and Python float timesteps.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch


def timesteps(steps, rescale_t):
    t_seq = np.linspace(1, 0, steps + 1)
    return (rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)).tolist()


def dense_std(x):
    return x.std(dim=list(range(1, x.ndim)), keepdim=True)


def sparse_std(x):
    """Upstream VarLenTensor.std of one sample's (N, C) features: sqrt(E[x^2] - E[x]^2) over all of them."""
    lengths = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
    mean = torch.segment_reduce(x.mean(dim=1, keepdim=True), "mean", lengths=lengths)
    mean2 = torch.segment_reduce((x ** 2).mean(dim=1, keepdim=True), "mean", lengths=lengths)
    return (mean2 - mean ** 2).sqrt()


def guided_velocity(x_t, t, pred_pos, pred_neg, strength, rescale, sigma_min, std):
    pred = strength * pred_pos + (1 - strength) * pred_neg
    if rescale > 0:
        x_0_pos = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * pred_pos
        x_0_cfg = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * pred
        x_0_rescaled = x_0_cfg * (std(x_0_pos) / std(x_0_cfg))
        x_0 = rescale * x_0_rescaled + (1 - rescale) * x_0_cfg
        pred = ((1 - sigma_min) * x_t - x_0) / (sigma_min + (1 - sigma_min) * t)
    return pred


def sample(model, noise, cond, neg_cond, steps, rescale_t, guidance_strength, guidance_rescale, guidance_interval, sigma_min, std, on_step=None):
    """FlowEulerGuidanceIntervalSampler.sample. model(x_t, t_tensor, contexts) returns one velocity per context; with guidance active,
    the conditional and unconditional branches run in one joint pass. on_step(step) is called after each step."""
    x = noise
    t_seq = timesteps(steps, rescale_t)
    for step, (t, t_prev) in enumerate(zip(t_seq[:-1], t_seq[1:])):
        strength = guidance_strength if guidance_interval[0] <= t <= guidance_interval[1] else 1
        t_tensor = torch.tensor([1000 * t], device=x.device, dtype=torch.float32)
        if strength == 1:
            pred = model(x, t_tensor, [cond])[0]
        elif strength == 0:
            pred = model(x, t_tensor, [neg_cond])[0]
        else:
            pred_pos, pred_neg = model(x, t_tensor, [cond, neg_cond])
            pred = guided_velocity(x, t, pred_pos, pred_neg, strength, guidance_rescale, sigma_min, std)
            del pred_pos, pred_neg
        x = x - (t - t_prev) * pred
        del pred
        if on_step is not None:
            on_step(step)
    return x
