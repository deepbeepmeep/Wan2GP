# Modified from timm library:
# https://github.com/huggingface/pytorch-image-models/blob/648aaa41233ba83eb38faf5ba9d415d574823241/timm/layers/mlp.py#L13

from functools import partial

import torch
import torch.nn as nn

from .modulate_layers import modulate_
from ..utils.helpers import to_2tuple


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(
            in_channels, hidden_channels, bias=bias[0], **factory_kwargs
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_channels, **factory_kwargs)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = linear_layer(
            hidden_channels, out_features, bias=bias[1], **factory_kwargs
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def apply_(self, x, divide = 4):
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        chunk_size = int(x_shape[1]/divide)
        x_chunks = torch.split(x, chunk_size)
        for i, x_chunk  in enumerate(x_chunks):
            mlp_chunk = self.fc1(x_chunk)
            mlp_chunk = self.act(mlp_chunk)
            mlp_chunk = self.drop1(mlp_chunk)
            mlp_chunk = self.norm(mlp_chunk)
            mlp_chunk = self.fc2(mlp_chunk)
            x_chunk[...] = self.drop2(mlp_chunk)
        return x        

# 
class MLPEmbedder(nn.Module):
    """copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py"""
    def __init__(self, in_dim: int, hidden_dim: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True, **factory_kwargs)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(
        self, hidden_size, patch_size, out_channels, act_layer, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Just use LayerNorm for the final layer
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        if isinstance(patch_size, int):
            self.linear = nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                **factory_kwargs
            )
        else:
            self.linear = nn.Linear(
                hidden_size,
                patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
                bias=True,
            )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Here we don't distinguish between the modulate types. Just use the simple one.
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate_(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x
