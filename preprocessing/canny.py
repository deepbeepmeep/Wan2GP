# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image


norm_layer = nn.InstanceNorm2d

def convert_to_torch(image):
    if isinstance(image, Image.Image):
        image = torch.from_numpy(np.array(image)).float()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    elif isinstance(image, np.ndarray):
        image = torch.from_numpy(image.copy()).float()
    else:
        raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
    return image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ContourInference(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(ContourInference, self).__init__()

        # Initial convolution block
        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True)
        ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features,
                                   out_features,
                                   3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class CannyAnnotator:
    def __init__(self, cfg, device=None):
        input_nc = cfg.get('INPUT_NC', 3)
        output_nc = cfg.get('OUTPUT_NC', 1)
        n_residual_blocks = cfg.get('N_RESIDUAL_BLOCKS', 3)
        sigmoid = cfg.get('SIGMOID', True)
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = ContourInference(input_nc, output_nc, n_residual_blocks,
                                      sigmoid)
        self.model.load_state_dict(torch.load(pretrained_model, weights_only=True))
        self.model = self.model.eval().requires_grad_(False).to(self.device)

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        is_batch = False if len(image.shape) == 3 else True
        image = convert_to_torch(image)
        if len(image.shape) == 3:
            image = rearrange(image, 'h w c -> 1 c h w')
        image = image.float().div(255).to(self.device)
        contour_map = self.model(image)
        contour_map = (contour_map.squeeze(dim=1) * 255.0).clip(
            0, 255).cpu().numpy().astype(np.uint8)
        contour_map = contour_map[..., None].repeat(3, -1)
        contour_map = 255 - contour_map  #.where( image >= 127.5,0,1)
        contour_map[ contour_map > 127.5] = 255
        contour_map[ contour_map <= 127.5] = 0
        if not is_batch:
            contour_map = contour_map.squeeze()
        return contour_map


class CannyVideoAnnotator(CannyAnnotator):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames