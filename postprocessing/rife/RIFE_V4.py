"""
MIT License

Copyright (c) 2024 Hzwer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid):
    dtype = tenInput.dtype
    tenInput = tenInput.to(torch.float)
    tenFlow = tenFlow.to(torch.float)

    tenFlow = torch.cat(
        [tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1
    )
    g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
    padding_mode = "border"
    if tenInput.device.type == "mps":
        padding_mode = "zeros"
        g = g.clamp(-1, 1)
    return F.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    ).to(dtype)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        x = self.cnn3(x)
        return x


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class IFNet(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.scaleList = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

    def forward(self, img0, img1, timestep, tenFlow_div, backwarp_tenGrid, f0, f1):
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)

        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        feat = None

        for i in range(5):
            if flow is None:
                flow, mask, feat = self.blocks[i](
                    torch.cat((img0, img1, f0, f1, timestep), 1),
                    None,
                    scale=self.scaleList[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
                wf1 = warp(f1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
                fd, m0, feat = self.blocks[i](
                    torch.cat(
                        (
                            warped_img0,
                            warped_img1,
                            wf0,
                            wf1,
                            timestep,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=self.scaleList[i],
                )
                mask = m0
                flow = flow + fd
            warped_img0 = warp(img0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
            warped_img1 = warp(img1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
        mask = torch.sigmoid(mask)
        return warped_img0 * mask + warped_img1 * (1 - mask)


class Model:
    def __init__(self):
        self.flownet = IFNet()
        self.encode = Head()
        self.pad_mod = 64
        self.supports_timestep = True
        self._grid_cache = {}
        self.device = None

    def train(self):
        self.flownet.train()
        self.encode.train()

    def eval(self):
        self.flownet.eval()
        self.encode.eval()

    def to(self, device):
        self.flownet.to(device)
        self.encode.to(device)

    def _get_grid(self, height, width, device):
        key = (height, width, device.type, device.index)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        tenFlow_div = torch.tensor(
            [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
            dtype=torch.float32,
            device=device,
        )
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device)
            .view(1, 1, 1, width)
            .expand(1, 1, height, width)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=device)
            .view(1, 1, height, 1)
            .expand(1, 1, height, width)
        )
        backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)
        self._grid_cache[key] = (tenFlow_div, backwarp_tenGrid)
        return tenFlow_div, backwarp_tenGrid

    def load_model(self, path, rank=0, device="cuda"):
        self.device = device
        state_dict = torch.load(path, map_location=device)
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "flownet" in state_dict:
                state_dict = state_dict["flownet"]
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }
        head_state = {
            k.replace("encode.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encode.")
        }
        if head_state:
            self.encode.load_state_dict(head_state, strict=True)
        flow_state = {
            k: v for k, v in state_dict.items() if not k.startswith("encode.")
        }
        self.flownet.load_state_dict(flow_state, strict=False)
        self.to(device)

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        if scale != 1.0:
            self.flownet.scaleList = [
                16 / scale,
                8 / scale,
                4 / scale,
                2 / scale,
                1 / scale,
            ]
        f0 = self.encode(img0)
        f1 = self.encode(img1)
        height = img0.shape[2]
        width = img0.shape[3]
        tenFlow_div, backwarp_tenGrid = self._get_grid(height, width, img0.device)
        timestep_tensor = torch.full(
            (1, 1, height, width),
            float(timestep),
            dtype=img0.dtype,
            device=img0.device,
        )
        return self.flownet(
            img0, img1, timestep_tensor, tenFlow_div, backwarp_tenGrid, f0, f1
        )
