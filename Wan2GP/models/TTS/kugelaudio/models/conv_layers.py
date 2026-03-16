"""Convolutional layers for KugelAudio tokenizers.

This module provides the building blocks for the acoustic and semantic tokenizers,
including streaming-capable convolutions and normalization layers.
"""

import math
import typing as tp
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import logging

logger = logging.get_logger(__name__)


# Normalization modules
class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        x = nn.functional.layer_norm(
            x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps
        ).type_as(x) 
        x = x.transpose(1, 2)  # b t ... -> b ... t
        return x
    

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            weight_shape = (dim,) if weight_shape is None else weight_shape
            self.weight = nn.Parameter(torch.ones(weight_shape))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class ConvRMSNorm(RMSNorm):
    """Convolution-friendly RMSNorm."""
    
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__(dim, eps, elementwise_affine, weight_shape)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        output = output.transpose(1, 2)  # b t ... -> b ... t
        return output


# Convolutional layers and utilities
CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                'time_layer_norm', 'layer_norm', 'time_group_norm'])


def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return nn.utils.weight_norm(module)
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm(module)
    else:
        return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """Return the proper normalization module."""
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                padding_total: int = 0) -> int:
    """Calculate extra padding needed for convolution to have the same output length."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Pad 1D input with handling for small inputs in reflect mode."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv."""
    
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv."""
    
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    """Conv1d with built-in handling of asymmetric or causal padding and normalization."""
    
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, dilation: int = 1,
                groups: int = 1, bias: bool = True, causal: bool = False,
                norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                pad_mode: str = 'reflect'):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                            dilation=dilation, groups=groups, bias=bias, causal=causal,
                            norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode
        
        # Store configuration
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # For non-streaming mode, calculate padding
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)
                  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (non-streaming)."""
        B, C, T = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        padding_total = self.padding_total
        
        # Compute extra padding for stride alignment
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        if self.causal:
            # Left padding for causal
            if self.pad_mode == 'constant':
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode, value=0)
            else:
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Symmetric padding for non-causal
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
            
        output = self.conv(x)
        return output


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in handling of asymmetric or causal padding and normalization."""
    
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, causal: bool = False,
                norm: str = 'none', trim_right_ratio: float = 1.,
                norm_kwargs: tp.Dict[str, tp.Any] = {}, bias: bool = True):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                        causal=causal, norm=norm, norm_kwargs=norm_kwargs, bias=bias)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

        # Store configuration
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # For transposed convolution, padding calculation is different
        self.padding_total = kernel_size - stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (non-streaming)."""
        kernel_size = self.kernel_size
        stride = self.stride
        padding_total = self.padding_total

        y = self.convtr(x)
        
        # Remove the padding from output
        if self.causal:
            # Trim right side for causal
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Symmetric unpadding for non-causal
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
            
        return y


__all__ = [
    "ConvLayerNorm",
    "RMSNorm",
    "ConvRMSNorm",
    "NormConv1d",
    "NormConvTranspose1d",
    "SConv1d",
    "SConvTranspose1d",
    "pad1d",
    "unpad1d",
    "get_extra_padding_for_conv1d",
]
