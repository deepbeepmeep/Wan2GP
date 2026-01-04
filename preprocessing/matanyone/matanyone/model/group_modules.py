from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .channel_attn import CAResBlock

def interpolate_groups(g: torch.Tensor, ratio: float, mode: str,
                       align_corners: bool) -> torch.Tensor:
    batch_size, num_objects = g.shape[:2]
    g = F.interpolate(g.flatten(start_dim=0, end_dim=1),
                      scale_factor=ratio,
                      mode=mode,
                      align_corners=align_corners)
    g = g.view(batch_size, num_objects, *g.shape[1:])
    return g


def upsample_groups(g: torch.Tensor,
                    ratio: float = 2,
                    mode: str = 'bilinear',
                    align_corners: bool = False) -> torch.Tensor:
    return interpolate_groups(g, ratio, mode, align_corners)


def downsample_groups(g: torch.Tensor,
                      ratio: float = 1 / 2,
                      mode: str = 'area',
                      align_corners: bool = None) -> torch.Tensor:
    return interpolate_groups(g, ratio, mode, align_corners)


class GConv2d(nn.Conv2d):
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])


class GroupResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = nn.Identity()
        else:
            self.downsample = GConv2d(in_dim, out_dim, kernel_size=1)

        self.conv1 = GConv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = GConv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))

        g = self.downsample(g)

        return out_g + g


class MainToGroupDistributor(nn.Module):
    def __init__(self,
                 x_transform: Optional[nn.Module] = None,
                 g_transform: Optional[nn.Module] = None,
                 method: str = 'cat',
                 reverse_order: bool = False):
        super().__init__()

        self.x_transform = x_transform
        self.g_transform = g_transform
        self.method = method
        self.reverse_order = reverse_order

    def forward(self, x: torch.Tensor, g: torch.Tensor, skip_expand: bool = False) -> torch.Tensor:
        num_objects = g.shape[1]

        if self.x_transform is not None:
            x = self.x_transform(x)

        if self.g_transform is not None:
            g = self.g_transform(g)

        if not skip_expand:
            x = x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
        if self.method == 'cat':
            if self.reverse_order:
                g = torch.cat([g, x], 2)
            else:
                g = torch.cat([x, g], 2)
        elif self.method == 'add':
            g = x + g
        elif self.method == 'mulcat':
            g = torch.cat([x * g, g], dim=2)
        elif self.method == 'muladd':
            g = x * g + g
        else:
            raise NotImplementedError

        return g


class GroupFeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim: int, g_in_dim: int, out_dim: int):
        super().__init__()

        x_transform = nn.Conv2d(x_in_dim, out_dim, kernel_size=1)
        g_transform = GConv2d(g_in_dim, out_dim, kernel_size=1)

        self.distributor = MainToGroupDistributor(x_transform=x_transform,
                                                  g_transform=g_transform,
                                                  method='add')
        self.block1 = CAResBlock(out_dim, out_dim)
        self.block2 = CAResBlock(out_dim, out_dim)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects = g.shape[:2]

        g = self.distributor(x, g)

        g = g.flatten(start_dim=0, end_dim=1)

        g = self.block1(g)
        g = self.block2(g)

        g = g.view(batch_size, num_objects, *g.shape[1:])

        return g