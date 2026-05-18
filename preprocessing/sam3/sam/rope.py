# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Adapted from:
1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
2. https://github.com/naver-ai/rope-vit
3. https://github.com/lucidrains/rotary-embedding-torch
"""

from typing import Optional

import torch
from einops import rearrange, repeat
from torch import broadcast_tensors, nn


def init_t_xy(end_x: int, end_y: int, scale: float = 1.0, offset: int = 0, device=None):
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device=None,
):
    freqs_x = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    freqs_y = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )

    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def compute_axial_cis_real(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device=None,
):
    freqs_x = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    freqs_y = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )

    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    return torch.cat([freqs_x.cos(), freqs_y.cos()], dim=-1), torch.cat([freqs_x.sin(), freqs_y.sin()], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor = None,
    freqs_cis: torch.Tensor = None,
    repeat_freqs_k: bool = False,
):
    if isinstance(xq, list):
        qk_list = xq
        xq, xk = qk_list
        qk_list.clear()

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def _apply_rotary_enc_real_inplace(x: torch.Tensor, freqs_cis_real: torch.Tensor, freqs_cis_imag: torch.Tensor, repeat_freqs: bool = False):
    seq_len = freqs_cis_real.shape[0]
    if repeat_freqs:
        repeat_count = x.shape[-2] // seq_len
        x_pair = x.view(*x.shape[:-2], repeat_count, seq_len, -1, 2)
    else:
        x_pair = x.view(*x.shape[:-1], -1, 2)
    x_real = x_pair[..., 0]
    x_imag = x_pair[..., 1]
    freqs_cis_real = reshape_for_broadcast(freqs_cis_real, x_real)
    freqs_cis_imag = reshape_for_broadcast(freqs_cis_imag, x_imag)
    target_dtype = x.dtype
    if freqs_cis_real.device != x.device or freqs_cis_real.dtype != target_dtype:
        freqs_cis_real = freqs_cis_real.to(device=x.device, dtype=target_dtype)
    if freqs_cis_imag.device != x.device or freqs_cis_imag.dtype != target_dtype:
        freqs_cis_imag = freqs_cis_imag.to(device=x.device, dtype=target_dtype)
    x_real_orig = x_real.clone()
    x_real.mul_(freqs_cis_real).addcmul_(x_imag, freqs_cis_imag, value=-1)
    x_imag.mul_(freqs_cis_real).addcmul_(x_real_orig, freqs_cis_imag)
    return x


def apply_rotary_enc_real(
    xq: torch.Tensor,
    xk: torch.Tensor = None,
    freqs_cis_real: torch.Tensor = None,
    freqs_cis_imag: torch.Tensor = None,
    repeat_freqs_k: bool = False,
):
    if isinstance(xq, list):
        qk_list = xq
        xq, xk = qk_list
        qk_list.clear()

    assert xk is not None
    _apply_rotary_enc_real_inplace(xq, freqs_cis_real, freqs_cis_imag)
    if xk.shape[-2] != 0:
        _apply_rotary_enc_real_inplace(xk, freqs_cis_real, freqs_cis_imag, repeat_freqs=repeat_freqs_k)
    return xq, xk


# rotary embedding helper functions
def broadcat(tensors, dim=-1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


def rotate_half(x: torch.Tensor):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingVE(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        pt_seq_len: Optional[int] = None,
        theta: float = 10000.0,
        offset: int = 1,  # specific to VE
    ):
        super().__init__()

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        scale = 1.0
        if pt_seq_len is not None:
            scale = pt_seq_len / seq_len

        # offset of +1 following VE - even though for the
        # attention op only differences matter
        t = torch.arange(seq_len) * scale + offset

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs[None, :, :], freqs[:, None, :]), dim=-1)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t: torch.Tensor):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin
