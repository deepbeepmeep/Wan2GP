import math
import torch
from torch import nn
from torch.nn.utils import weight_norm


def get_activation(activation, channels=None) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    elif activation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, depthwise=False, bias=True):
        super().__init__()

        self.dilation = dilation

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation("elu"),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=padding, groups=1 if not depthwise else out_channels, bias=bias),
            get_activation("elu"),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, bias=bias)
        )

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=16.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class ExpoFourierFeatures(nn.Module):
    def __init__(self, dim, min_freq=0.5, max_freq=10000.0):
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq

    @torch.amp.autocast("cuda",enabled=False)
    def forward(self, t):
        """
        t: [B] tensor.
        """
        in_dtype = t.dtype
        t = t.float()

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2

        # Calculate frequencies (safely in FP32)
        ramp = torch.linspace(0, 1, half_dim, device=t.device, dtype=torch.float32)
        log_min = math.log(self.min_freq)
        log_max = math.log(self.max_freq)

        freqs = torch.exp(ramp * (log_max - log_min) + log_min)

        # Calculate arguments (safely in FP32)
        args = t * freqs * 2 * math.pi

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return embedding.to(in_dtype)



