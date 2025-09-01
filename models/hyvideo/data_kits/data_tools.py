import os
import cv2
import torch
import numpy as np
import imageio
import torchvision
from einops import rearrange


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, quality=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x,0,1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, quality=quality)

def pad_image(crop_img, size, color=(255, 255, 255), resize_ratio=1):
    crop_h, crop_w = crop_img.shape[:2]
    target_w, target_h = size
    scale_h, scale_w = target_h / crop_h, target_w / crop_w
    if scale_w > scale_h:
        resize_h = int(target_h*resize_ratio)
        resize_w = int(crop_w / crop_h * resize_h)
    else:
        resize_w = int(target_w*resize_ratio)
        resize_h = int(crop_h / crop_w * resize_w)
    crop_img = cv2.resize(crop_img, (resize_w, resize_h))
    pad_left = (target_w - resize_w) // 2
    pad_top = (target_h - resize_h) // 2
    pad_right = target_w - resize_w - pad_left
    pad_bottom = target_h - resize_h - pad_top
    crop_img = cv2.copyMakeBorder(crop_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return crop_img