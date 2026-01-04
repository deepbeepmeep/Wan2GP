from typing import List, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .group_modules import MainToGroupDistributor, GroupResBlock, upsample_groups, GConv2d, downsample_groups


class UpsampleBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale_factor: int = 2):
        super().__init__()
        self.out_conv = ResBlock(in_dim, out_dim)
        self.scale_factor = scale_factor

    def forward(self, in_g: torch.Tensor, skip_f: torch.Tensor) -> torch.Tensor:
        g = F.interpolate(in_g,
                      scale_factor=self.scale_factor,
                      mode='bilinear')
        g = self.out_conv(g)
        g = g + skip_f
        return g

class MaskUpsampleBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale_factor: int = 2):
        super().__init__()
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(in_dim, out_dim)
        self.scale_factor = scale_factor

    def forward(self, in_g: torch.Tensor, skip_f: torch.Tensor) -> torch.Tensor:
        g = upsample_groups(in_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g
    

class DecoderFeatureProcessor(nn.Module):
    def __init__(self, decoder_dims: List[int], out_dims: List[int]):
        super().__init__()
        self.transforms = nn.ModuleList([
            nn.Conv2d(d_dim, p_dim, kernel_size=1) for d_dim, p_dim in zip(decoder_dims, out_dims)
        ])

    def forward(self, multi_scale_features: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        outputs = [func(x) for x, func in zip(multi_scale_features, self.transforms)]
        return outputs


# @torch.jit.script
def _recurrent_update(h: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # h: batch_size * num_objects * hidden_dim * h * w
    # values: batch_size * num_objects * (hidden_dim*3) * h * w
    dim = values.shape[2] // 3
    forget_gate = torch.sigmoid(values[:, :, :dim])
    update_gate = torch.sigmoid(values[:, :, dim:dim * 2])
    new_value = torch.tanh(values[:, :, dim * 2:])
    new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value
    return new_h


class SensoryUpdater_fullscale(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims: List[int], mid_dim: int, sensory_dim: int):
        super().__init__()
        self.g16_conv = GConv2d(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2d(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2d(g_dims[2], mid_dim, kernel_size=1)
        self.g2_conv = GConv2d(g_dims[3], mid_dim, kernel_size=1)
        self.g1_conv = GConv2d(g_dims[4], mid_dim, kernel_size=1)

        self.transform = GConv2d(mid_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4)) + \
            self.g2_conv(downsample_groups(g[3], ratio=1/8)) + \
            self.g1_conv(downsample_groups(g[4], ratio=1/16))

        with torch.amp.autocast("cuda"):
            g = g.float()
            h = h.float()
            values = self.transform(torch.cat([g, h], dim=2))
            new_h = _recurrent_update(h, values)

        return new_h

class SensoryUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims: List[int], mid_dim: int, sensory_dim: int):
        super().__init__()
        self.g16_conv = GConv2d(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2d(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2d(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2d(mid_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        with torch.amp.autocast("cuda"):
            g = g.float()
            h = h.float()
            values = self.transform(torch.cat([g, h], dim=2))
            new_h = _recurrent_update(h, values)

        return new_h


class SensoryDeepUpdater(nn.Module):
    def __init__(self, f_dim: int, sensory_dim: int):
        super().__init__()
        self.transform = GConv2d(f_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda"):
            g = g.float()
            h = h.float()
            values = self.transform(torch.cat([g, h], dim=2))
            new_h = _recurrent_update(h, values)

        return new_h

  
class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))

        g = self.downsample(g)

        return out_g + g