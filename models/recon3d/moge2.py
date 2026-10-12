# MoGe-2 monocular geometry estimation (Wang et al., microsoft/MoGe, MIT License, see LICENSE_MoGe): inference port of moge/model/v2.py,
# moge/model/modules (DINOv2Encoder, ConvStack, MLP) and moge/utils/geometry_torch.py for WanGP (modified: the DINOv2 backbone of
# dinov2.py, the configured layer types only, the forward reports its progress per block).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import contextlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
from torch import nn

from .dinov2 import DinoV2

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "moge2.json")
FP16_AUTOCAST = torch.float16  # upstream infer(use_fp16=True): the forward runs under FP16 autocast with the FP32 weights


def _conv3(channels_in, channels_out):
    return nn.Conv2d(channels_in, channels_out, 3, padding=1, padding_mode="replicate")


def _resampler(kind, channels_in, channels_out):
    """2x upsampling of ConvStack: a transposed 2x2 convolution or a bilinear resize, then a 3x3 convolution."""
    first = nn.ConvTranspose2d(channels_in, channels_out, 2, stride=2) if kind == "conv_transpose" else nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    return nn.Sequential(first, _conv3(channels_out if kind == "conv_transpose" else channels_in, channels_out))


class ResidualConvBlock(nn.Module):
    """x + conv(relu(conv(relu(x)))), 3x3 convolutions without norms (res_block_in_norm / hidden_norm "none")."""

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Identity(), nn.ReLU(), _conv3(channels, channels), nn.Identity(), nn.ReLU(), _conv3(channels, channels))

    def forward(self, x):
        return x + self.layers(x)


class ConvStack(nn.Module):
    """Multi-scale decoder: per level an optional 1x1 input projection added to the upsampled running features, residual blocks (ReLU,
    3x3 convolution, twice, without norms) and an optional 1x1 output projection (in FP32 for the points and normal heads)."""

    def __init__(self, dim_in, dim_out, dim_res_blocks, num_res_blocks, resamplers, fp32_output=False):
        super().__init__()
        self.input_blocks = nn.ModuleList([nn.Identity() if d_in is None else nn.Conv2d(d_in, d, 1) for d_in, d in zip(dim_in, dim_res_blocks)])
        self.resamplers = nn.ModuleList([_resampler(kind, d, d_next) for kind, d, d_next in zip(resamplers, dim_res_blocks, dim_res_blocks[1:])])
        self.res_blocks = nn.ModuleList([nn.Sequential(*[ResidualConvBlock(d) for _ in range(count)]) for d, count in zip(dim_res_blocks, num_res_blocks)])
        # the neck has no output projection: no list of Identity modules (MMGP would plan it as an empty block tower)
        self.output_blocks = None if all(d_out is None for d_out in dim_out) else nn.ModuleList(
            [nn.Identity() if d_out is None else nn.Conv2d(d, d_out, 1) for d_out, d in zip(dim_out, dim_res_blocks)])
        self.fp32_output = fp32_output

    def forward(self, features):
        outputs = []
        for level, (input_block, res_block) in enumerate(zip(self.input_blocks, self.res_blocks)):
            feature = input_block(features[level])
            x = feature if level == 0 else x + feature
            x = res_block(x)
            if self.output_blocks is None:
                outputs.append(x)
            else:
                with torch.autocast(x.device.type, dtype=torch.float32) if self.fp32_output else contextlib.nullcontext():
                    outputs.append(self.output_blocks[level](x))
            if level < len(self.resamplers):
                x = self.resamplers[level](x)
        return outputs


def normalized_view_plane_uv(width, height, aspect_ratio, dtype, device):
    """UV of the pixel centres, the image corners at (+-width, +-height) / diagonal."""
    span_x, span_y = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5, 1 / (1 + aspect_ratio ** 2) ** 0.5
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    return torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1)


class MoGe2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.Module()
        self.encoder.backbone = DinoV2(**config["backbone"])
        self.encoder.output_projections = nn.ModuleList([nn.Conv2d(config["backbone"]["dim"], config["dim_out"], 1) for _ in config["intermediate_layers"]])
        self.encoder.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.encoder.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.neck = ConvStack(**config["neck"])
        self.points_head = ConvStack(**config["points_head"], fp32_output=True)
        self.normal_head = ConvStack(**config["normal_head"], fp32_output=True)
        self.mask_head = ConvStack(**config["mask_head"])
        dims = config["scale_head"]
        self.scale_head = nn.Sequential(*[layer for d_in, d_out in zip(dims[:-2], dims[1:-1]) for layer in (nn.Linear(d_in, d_out), nn.ReLU())],
                                        nn.Linear(dims[-2], dims[-1]))

    def transformer_blocks(self):
        return [self.encoder.backbone.blocks]

    @staticmethod
    def from_config():
        with open(CONFIG_PATH, encoding="utf-8") as reader:
            return MoGe2(json.load(reader))

    def block_count(self):
        return len(self.encoder.backbone.blocks) + 1

    def forward(self, image, num_tokens, progress=None):
        """image (1, 3, H, W) RGB in 0..1, FP32: affine point map (1, H, W, 3), unit normals, mask probability and metric scale."""
        batch, _, height, width = image.shape
        aspect_ratio = width / height
        rows, cols = round((num_tokens / aspect_ratio) ** 0.5), round((num_tokens * aspect_ratio) ** 0.5)
        encoder = self.encoder
        with torch.autocast(image.device.type, dtype=FP16_AUTOCAST):
            pixels = F.interpolate(image, (rows * 14, cols * 14), mode="bilinear", align_corners=False, antialias=True)
            pixels = (pixels - encoder.image_mean) / encoder.image_std
            layers = encoder.backbone(pixels, set(self.config["intermediate_layers"]), progress)
            del pixels
            layers = [encoder.backbone.norm(layer) for layer in layers]
            cls_token = layers[-1][:, 0]
            features = torch.stack([projection(layer[:, 1:].permute(0, 2, 1).unflatten(2, (rows, cols)).contiguous())
                                    for projection, layer in zip(encoder.output_projections, layers)], dim=1).sum(dim=1)
            del layers
            features = [features, None, None, None, None]
            for level in range(5):
                uv = normalized_view_plane_uv(cols * 2 ** level, rows * 2 ** level, aspect_ratio, image.dtype, image.device)
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch, -1, -1, -1)
                features[level] = uv if features[level] is None else torch.cat([features[level], uv], dim=1)
            features = self.neck(features)
            points, normal, mask = (head(features)[-1] for head in (self.points_head, self.normal_head, self.mask_head))
            metric_scale = self.scale_head(cls_token)
            del features
            points, normal, mask = (F.interpolate(v, (height, width), mode="bilinear", align_corners=False, antialias=False) for v in (points, normal, mask))
        if progress is not None:
            progress()
        points = points.permute(0, 2, 3, 1)
        xy, z = points.split([2, 1], dim=-1)  # remap_output "exp"
        z = torch.exp(z)
        points = torch.cat([xy * z, z], dim=-1)
        normal = F.normalize(normal.permute(0, 2, 3, 1), dim=-1)
        return {"points": points.float(), "normal": normal.float(), "mask": mask.squeeze(1).sigmoid().float(), "metric_scale": metric_scale.squeeze(1).float().exp()}


def _solve_focal_shift(uv, xyz):
    """Upstream solve_optimal_focal_shift: min |focal * xy / (z + shift) - uv| over shift (Levenberg-Marquardt) and focal (closed form)."""
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def residuals(shift):
        xy_proj = xy / (z + shift)[:, None]
        focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (focal * xy_proj - uv).ravel()

    shift = least_squares(residuals, x0=0, ftol=1e-3, method="lm")["x"].squeeze().astype(np.float32)
    xy_proj = xy / (z + shift)[:, None]
    return shift, (xy_proj * uv).sum() / np.square(xy_proj).sum()


def recover_focal_shift(points, mask):
    """Focal length (relative to half the image diagonal) and z shift turning the affine point map (H, W, 3) into camera space, solved
    on a 64 x 64 nearest-neighbour subsampling of the valid points."""
    height, width = points.shape[:2]
    uv = normalized_view_plane_uv(width, height, width / height, points.dtype, points.device)
    points_lr = F.interpolate(points.permute(2, 0, 1)[None], (64, 64), mode="nearest")[0].permute(1, 2, 0)
    uv_lr = F.interpolate(uv.permute(2, 0, 1)[None], (64, 64), mode="nearest")[0].permute(1, 2, 0)
    mask_lr = F.interpolate(mask[None, None].float(), (64, 64), mode="nearest")[0, 0] > 0
    uv_lr, points_lr = uv_lr[mask_lr].cpu().numpy(), points_lr[mask_lr].cpu().numpy()
    if len(uv_lr) < 2:
        return 1.0, 0.0
    shift, focal = _solve_focal_shift(uv_lr, points_lr)
    return float(focal), float(shift)


def point_map(output, image_size):
    """Upstream MoGeModel.infer (force_projection, apply_mask): metric camera-space points (H, W, 3) in OpenCV axes, depth, normals,
    validity mask and pixel intrinsics (OpenCV convention, pixel centres at integer coordinates)."""
    points, mask = output["points"][0], output["mask"][0] > 0.5
    height, width = image_size
    aspect_ratio = width / height
    focal, shift = recover_focal_shift(points, mask)
    fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
    depth = points[..., 2] + shift
    mask &= depth > 0
    u = (torch.arange(width, device=depth.device, dtype=torch.float32) + 0.5) / width
    v = (torch.arange(height, device=depth.device, dtype=torch.float32) + 0.5) / height
    points = torch.stack([(u[None, :] - 0.5) / fx * depth, (v[:, None] - 0.5) / fy * depth, depth], dim=-1)
    scale = output["metric_scale"][0]
    points, depth = points * scale, depth * scale
    intrinsics = torch.tensor([[fx * width, 0, width / 2 - 0.5], [0, fy * height, height / 2 - 0.5], [0, 0, 1]], dtype=torch.float32, device=depth.device)
    return points, depth, output["normal"][0], mask, intrinsics
