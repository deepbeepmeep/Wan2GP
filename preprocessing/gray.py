# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np
from PIL import Image
import torch

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

class GrayAnnotator:
    def __init__(self, cfg):
        pass
    def forward(self, image):
        image = convert_to_numpy(image)
        gray_map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_map[..., None].repeat(3, axis=2)


class GrayVideoAnnotator(GrayAnnotator):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames
