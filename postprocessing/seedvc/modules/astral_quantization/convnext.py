import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvNextV2LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[None, :, None] * x + self.bias[None, :, None]
        return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class InterpolationLayer(nn.Module):
    def __init__(self, ):  # this is a default of 1 / 50 * (44100 / 512) / 4
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, target_len: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = F.interpolate(x, size=target_len, mode='linear')
        return x

class ConvNeXtV2Stage(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        intermediate_dim: int = 2048,
        num_blocks: int = 1,
        dilation: int = 1,
        downsample_layer_indices: List[int] = None,
        downsample_factors: List[int] = None,
        upsample_layer_indices: List[int] = None,
        upsample_factors: List[int] = None,
        interpolation_layer_indices: List[int] = None,
        input_dim: int = None,
        output_dim: int = None,
        gin_channels: int = 0,
    ):
        super().__init__()
        # maybe downsample layers
        if downsample_layer_indices is not None:
            assert downsample_factors is not None
            self.downsample_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvNextV2LayerNorm(dim, data_format="channels_first"),
                        nn.Conv1d(
                            dim, dim, kernel_size=downsample_factor, stride=downsample_factor
                        ),
                    ) for _, downsample_factor in zip(downsample_layer_indices, downsample_factors)
                ]
            )
            self.downsample_layer_indices = downsample_layer_indices
        else:
            self.downsample_blocks = nn.ModuleList()
            self.downsample_layer_indices = []

        # maybe upsample layers
        if upsample_layer_indices is not None:
            assert upsample_factors is not None
            self.upsample_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvNextV2LayerNorm(dim, data_format="channels_first"),
                        nn.ConvTranspose1d(
                            dim, dim, kernel_size=upsample_factor, stride=upsample_factor
                        ),
                    ) for _, upsample_factor in zip(upsample_layer_indices, upsample_factors)
                ]
            )
            self.upsample_layer_indices = upsample_layer_indices
        else:
            self.upsample_blocks = nn.ModuleList()
            self.upsample_layer_indices = []

        # maybe interpolation layers
        if interpolation_layer_indices is not None:
            self.interpolation_blocks = nn.ModuleList(
                [
                    InterpolationLayer()
                    for _ in interpolation_layer_indices
                ]
            )
            self.interpolation_layer_indices = interpolation_layer_indices
        else:
            self.interpolation_blocks = nn.ModuleList()
            self.interpolation_layer_indices = []

        # main blocks
        self.blocks = nn.ModuleList(
            [
                ConvNeXtV2Block(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    dilation=dilation,
                )
                for _ in range(num_blocks)
            ]
        )
        # maybe input and output projections
        if input_dim is not None and input_dim != dim:
            self.input_projection = nn.Conv1d(input_dim, dim, kernel_size=1)
        else:
            self.input_projection = nn.Identity()
        if output_dim is not None and output_dim != dim:
            self.output_projection = nn.Conv1d(dim, output_dim, kernel_size=1)
        else:
            self.output_projection = nn.Identity()

        if gin_channels > 0:
            self.gin = nn.Conv1d(gin_channels, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.input_projection(x)  # B, D, T
        if hasattr(self, 'gin'):
            g = kwargs['g']
            x = x + self.gin(g)
        # pad to a multiple of cumprod(downsample_factors)
        if len(self.downsample_blocks) > 0:
            downsample_factor = 1
            for factor in self.downsample_blocks:
                downsample_factor *= factor[1].stride[0]
            pad_len = downsample_factor - x.size(-1) % downsample_factor
            if pad_len > 0:
                x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)

        # main blocks
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx in self.downsample_layer_indices:
                x = self.downsample_blocks[self.downsample_layer_indices.index(layer_idx)](x)
            if layer_idx in self.upsample_layer_indices:
                x = self.upsample_blocks[self.upsample_layer_indices.index(layer_idx)](x)
            if layer_idx in self.interpolation_layer_indices:
                x = self.interpolation_blocks[self.interpolation_layer_indices.index(layer_idx)](x, target_len=kwargs['target_len'])
            x = block(x)
        x = self.output_projection(x)
        return x

    def setup_caches(self, *args, **kwargs):
        pass


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = ConvNextV2LayerNorm(dim, data_format="channels_first")
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # b n d -> b d n
        return residual + x