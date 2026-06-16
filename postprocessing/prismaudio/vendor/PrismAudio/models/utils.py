import torch
from safetensors.torch import load_file
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
#from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from torch.nn.utils import remove_weight_norm

def load_ckpt_state_dict(ckpt_path, prefix=None):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    # 过滤特定前缀的state_dict
    filtered_state_dict = {k.replace(f'{prefix}',''): v for k, v in state_dict.items() if k.startswith(prefix)} if prefix is not None else state_dict

    return filtered_state_dict

def remove_weight_norm_from_model(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            print(f"Removing weight norm from {module}")
            remove_weight_norm(module)

    return model

# Sampling functions copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/utils.py under MIT license
# License can be found in LICENSES/LICENSE_META.txt

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """

    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)

    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def next_power_of_two(n):
    return 2 ** (n - 1).bit_length()

def next_multiple_of_64(n):
    return ((n + 63) // 64) * 64


# mask construction helpers

def mask_from_start_end_indices(
    seq_len: int,
    start: Tensor,
    end: Tensor
):
    assert start.shape == end.shape
    device = start.device

    seq = torch.arange(seq_len, device = device, dtype = torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), seq_len)
    seq = seq.expand(*start.shape, seq_len)

    mask = seq >= start[..., None].long()
    mask &= seq < end[..., None].long()
    return mask

def mask_from_frac_lengths(
    seq_len: int,
    frac_lengths: Tensor
):
    device = frac_lengths.device

    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.zeros_like(frac_lengths, device = device).float().uniform_(0, 1)
    start = (max_start * rand).clamp(min = 0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

def _build_spline(video_feat, video_t, target_t):
    # 三次样条插值核心实现
    coeffs = natural_cubic_spline_coeffs(video_t, video_feat.permute(0,2,1))
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(target_t).permute(0,2,1)

def resample(video_feat, audio_latent):
    """
    9s
    video_feat: [B, 72, D]
    audio_latent: [B, D', 194] or int
    """
    B, Tv, D = video_feat.shape
    
    if isinstance(audio_latent, torch.Tensor):
        # audio_latent is a tensor
        if audio_latent.shape[1] != 64:
            Ta = audio_latent.shape[1]
        else:
            Ta = audio_latent.shape[2]
    elif isinstance(audio_latent, int):
        # audio_latent is an int
        Ta = audio_latent
    else:
        raise TypeError("audio_latent must be either a tensor or an int")
    
    # 构建时间戳 (关键改进点)
    video_time = torch.linspace(0, 9, Tv, device=video_feat.device)
    audio_time = torch.linspace(0, 9, Ta, device=video_feat.device)
    
    # 三维化处理 (Batch, Feature, Time)
    video_feat = video_feat.permute(0, 2, 1)  # [B, D, Tv]
    
    # 三次样条插值
    aligned_video = _build_spline(video_feat, video_time, audio_time)  # [B, D, Ta]
    return aligned_video.permute(0, 2, 1)  # [B, Ta, D]

import os
enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"

def compile(function, *args, **kwargs):
    
    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function

    return function