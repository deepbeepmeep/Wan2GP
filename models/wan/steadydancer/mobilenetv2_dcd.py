# Modify from https://github.com/liyunsheng13/dcd/blob/main/models/imagenet/mobilenetv2_dcd.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.


class DYModule(nn.Module):
    def __init__(self, inp, oup, fc_squeeze=8):
        super(DYModule, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        if inp < oup:
            self.mul = 4
            reduction = 8
            self.avg_pool = nn.AdaptiveAvgPool2d(2)
        else:
            self.mul = 1
            reduction = 2
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dim = min((inp * self.mul) // reduction, oup // reduction)
        while self.dim ** 2 > inp * self.mul * 2:
            reduction *= 2
            self.dim = min((inp * self.mul) // reduction, oup // reduction)
        if self.dim < 4:
            self.dim = 4

        squeeze = max(inp * self.mul, self.dim ** 2) // fc_squeeze
        if squeeze < 4:
            squeeze = 4
        self.conv_q = nn.Conv2d(inp, self.dim, 1, 1, 0, bias=False)

        self.fc = nn.Sequential(
            nn.Linear(inp * self.mul, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(squeeze, oup, bias=False)
        self.hs = Hsigmoid()
        self.conv_p = nn.Conv2d(self.dim, oup, 1, 1, 0, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=self.dim)
        # self.bn2 = nn.BatchNorm1d(self.dim)
        self.bn2 = nn.GroupNorm(num_groups=4, num_channels=self.dim)

    def forward(self, x):
        r = self.conv(x)

        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c * self.mul)
        y = self.fc(y)
        dy_phi = self.fc_phi(y).view(b, self.dim, self.dim)
        dy_scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)
        r = dy_scale.expand_as(r) * r

        x = self.conv_q(x)
        x = self.bn1(x)
        x = x.view(b, -1, h * w)
        x = self.bn2(torch.matmul(dy_phi, x)) + x
        x = x.view(b, -1, h, w)
        x = self.conv_p(x)
        return x + r


class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
