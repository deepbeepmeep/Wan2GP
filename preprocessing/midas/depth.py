# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
from einops import rearrange
from PIL import Image
import cv2



def convert_to_numpy(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
    return image

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image, (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img, k


def resize_image_ori(h, w, image, k):
    img = cv2.resize(
        image, (w, h),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

class DepthAnnotator:
    def __init__(self, cfg, device=None):
        from .api import MiDaSInference
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = MiDaSInference(model_type='dpt_hybrid', model_path=pretrained_model).to(self.device)
        self.a = cfg.get('A', np.pi * 2.0)
        self.bg_th = cfg.get('BG_TH', 0.1)

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        image = convert_to_numpy(image)
        image_depth = image
        h, w, c = image.shape
        image_depth, k = resize_image(image_depth,
                                      1024 if min(h, w) > 1024 else min(h, w))
        image_depth = torch.from_numpy(image_depth).float().to(self.device)
        image_depth = image_depth / 127.5 - 1.0
        image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
        depth = self.model(image_depth)[0]

        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)
        depth_pt = depth_pt.cpu().numpy()
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)
        depth_image = depth_image[..., None].repeat(3, 2)

        depth_image = resize_image_ori(h, w, depth_image, k)
        return depth_image


class DepthVideoAnnotator(DepthAnnotator):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames