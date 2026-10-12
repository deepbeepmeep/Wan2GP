# Wan2.1-Fun Control-Camera conditioning. SimpleAdapter and ResidualBlock are ported from VideoX-Fun
# (https://github.com/aigc-apps/VideoX-Fun, videox_fun/models/wan_transformer3d.py), Copyright Alibaba PAI, Apache License 2.0.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn as nn

from ..vista4d.camera import get_plucker_embedding
from .trajectory import build_camera_path

# Scene depth of the preset moves: 6 units make a truck or a pedestal move cover 1.5 units, the amplitude of VideoX-Fun's basic poses.
FUN_CAMERA_SCENE_DEPTH = 6.0


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


class SimpleAdapter(nn.Module):
    """Plucker rays (6 channels x 4 frames per latent frame, pixel resolution) to patch tokens added after the patch embedding."""

    def __init__(self, in_dim, out_dim, kernel_size, stride, downscale_factor=8, num_residual_blocks=1):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        self.conv = nn.Conv2d(in_dim * downscale_factor * downscale_factor, out_dim, kernel_size=kernel_size, stride=stride, padding=0)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(out_dim) for _ in range(num_residual_blocks)])

    def forward(self, x):
        bs, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(bs * f, c, h, w)
        x = self.residual_blocks(self.conv(self.pixel_unshuffle(x)))
        return x.view(bs, f, *x.shape[1:]).permute(0, 2, 1, 3, 4)


def fun_camera_intrinsics(height, width):
    """VideoX-Fun's basic poses: fx = 0.532 x 1280 and fy = 0.946 x 720 on 16:9 frames, cropped to the output aspect ratio."""
    focal = max(0.946026558 * height, 0.532139961 * width)
    return [focal, focal, width / 2, height / 2]


def prepare_fun_camera(pipeline, model_mode, custom_path, custom_settings, lat_frames, height, width):
    """Adapter features [1, dim, lat_frames, h, w] of the camera path, computed once: they do not depend on the denoised latents."""
    frame_step = pipeline.vae_stride[0]
    c2w, intrinsics = build_camera_path(model_mode, custom_path, custom_settings, (lat_frames - 1) * frame_step + 1, height, width, fun_camera_intrinsics(height, width), FUN_CAMERA_SCENE_DEPTH)
    c2w, intrinsics = torch.from_numpy(c2w).to(pipeline.device)[None], torch.from_numpy(intrinsics).to(pipeline.device)[None]
    features = []
    for latent_no in range(lat_frames):
        # VideoX-Fun repeats the first frame 4 times, then packs 4 consecutive frames per latent frame (channel = ray channel * 4 + frame)
        frames = [0] * frame_step if latent_no == 0 else list(range(frame_step * (latent_no - 1) + 1, frame_step * latent_no + 1))
        plucker = get_plucker_embedding(intrinsics[:, frames], c2w[:, frames], height, width).permute(0, 4, 1, 2, 3)
        features.append(pipeline.model.control_adapter(plucker.reshape(1, -1, 1, height, width)))
    return torch.cat(features, dim=2)
