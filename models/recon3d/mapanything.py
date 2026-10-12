# MapAnything metric multi-view reconstruction (Keetha et al., facebookresearch/map-anything, Apache-2.0, see LICENSE_MapAnything; its
# modules come from castacks/UniCeption, BSD-3-Clause, see LICENSE_UniCeption): inference port for WanGP of mapanything/models/
# mapanything/model.py (image-only inputs) with the UniCeption DINOv2 encoder, alternating-attention transformer, pose, MLP and adaptor
# heads, and of mapanything/utils (image preprocessing, output post-processing). Modified: no optional geometric inputs (their
# encoders are not loaded), the DPT feature head is a reimplementation of the original DPT design (Ranftl et al., MIT) rather than the
# CroCo-derived (CC BY-NC-SA) UniCeption code, attention goes through shared.attention_kit, the dense and pose heads run view by view.
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .dinov2 import DinoV2, ViTBlock

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "mapanything.json")
IMAGE_MEAN, IMAGE_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(features, features, 3, padding=1), nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(F.relu(x)))) + x


class FusionBlock(nn.Module):
    """DPT feature fusion: the coarser path plus the residual-refined skip features, refined, upsampled 2x (bilinear, corners aligned)
    and projected 1x1."""

    def __init__(self, features, with_skip=True):
        super().__init__()
        if with_skip:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = F.interpolate(self.resConfUnit2(x), scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPTFeature(nn.Module):
    """Reassembles four token maps at strides 4, 8, 16 and 32 of the image (from the patch grid at stride 14) and fuses them coarse to
    fine into features at 8 times the patch grid."""

    def __init__(self, input_dim, layer_dims, feature_dim):
        super().__init__()
        reassemble = [[nn.Conv2d(input_dim, layer_dims[0], 1), nn.ConvTranspose2d(layer_dims[0], layer_dims[0], 4, stride=4)],
                      [nn.Conv2d(input_dim, layer_dims[1], 1), nn.ConvTranspose2d(layer_dims[1], layer_dims[1], 2, stride=2)],
                      [nn.Conv2d(input_dim, layer_dims[2], 1)],
                      [nn.Conv2d(input_dim, layer_dims[3], 1), nn.Conv2d(layer_dims[3], layer_dims[3], 3, stride=2, padding=1)]]
        self.input_process = nn.ModuleList([nn.Sequential(nn.Sequential(*layers), nn.Conv2d(dim, feature_dim, 3, padding=1, bias=False))
                                            for layers, dim in zip(reassemble, layer_dims)])
        self.scratch = nn.Module()
        for index in range(1, 5):
            setattr(self.scratch, f"refinenet{index}", FusionBlock(feature_dim, with_skip=index < 4))

    def forward(self, features):
        layers = [process(feature) for process, feature in zip(self.input_process, features)]
        scratch = self.scratch
        path = scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path = scratch.refinenet3(path, layers[2])
        path = scratch.refinenet2(path, layers[1])
        return scratch.refinenet1(path, layers[0])


class DPTRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim // 2, 3, padding=1)
        self.conv2 = nn.Sequential(nn.Conv2d(input_dim // 2, input_dim // 2, 3, padding=1), nn.ReLU(), nn.Conv2d(input_dim // 2, output_dim, 1))

    def forward(self, x, size):
        return self.conv2(F.interpolate(self.conv1(x), size=size, mode="bilinear", align_corners=True))


class ResConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res_conv1, self.res_conv2, self.res_conv3 = (nn.Conv2d(channels, channels, 1) for _ in range(3))

    def forward(self, x):
        return x + F.relu(self.res_conv3(F.relu(self.res_conv2(F.relu(self.res_conv1(x))))))


class PoseHead(nn.Module):
    """Reloc3r-style pose regression from the final view features: (translation, quaternion x y z w) per view."""

    def __init__(self, input_dim, patch_size, num_resconv_block):
        super().__init__()
        channels = 4 * patch_size ** 2
        self.proj = nn.Conv2d(input_dim, channels, 1)
        self.res_conv = nn.ModuleList([ResConvBlock(channels) for _ in range(num_resconv_block)])
        self.more_mlps = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels), nn.ReLU())
        self.fc_t, self.fc_rot = nn.Linear(channels, 3), nn.Linear(channels, 4)

    def forward(self, x):
        x = self.proj(x)
        for block in self.res_conv:
            x = block(x)
        x = self.more_mlps(x.mean(dim=(2, 3)))
        return torch.cat([self.fc_t(x), self.fc_rot(x)], dim=1)


class ScaleHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_mlp_layers):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_mlp_layers)])
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, token):
        x = self.proj(token)
        for layer in self.mlp:
            x = layer(x)
        return self.output_proj(x)


class MapAnything(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder, sharing = config["encoder"], config["info_sharing"]
        dim = encoder["dim"]
        self.encoder = nn.Module()
        self.encoder.model = DinoV2(**encoder, norm=False)
        self.fusion_norm_layer = nn.LayerNorm(dim, eps=1e-6)
        self.scale_token = nn.Parameter(torch.empty(dim))
        self.info_sharing = nn.Module()
        self.info_sharing.self_attention_blocks = nn.ModuleList([ViTBlock(dim, sharing["heads"], swiglu=True) for _ in range(sharing["depth"])])
        self.info_sharing.norm = nn.LayerNorm(dim, eps=1e-6)
        self.info_sharing.register_buffer("view_pos_table", torch.empty(1, dim))
        dpt = config["dpt"]
        self.dense_head = nn.Sequential(DPTFeature(dim, dpt["layer_dims"], dpt["feature_dim"]), DPTRegression(dpt["feature_dim"], dpt["output_dim"]))
        self.pose_head = PoseHead(dim, 14, **config["pose_head"])
        self.scale_head = ScaleHead(dim, **config["scale_head"])

    def transformer_blocks(self):
        return [self.encoder.model.blocks, self.info_sharing.self_attention_blocks]

    @staticmethod
    def from_config():
        with open(CONFIG_PATH, encoding="utf-8") as reader:
            return MapAnything(json.load(reader))

    def block_count(self, views):
        return len(self.encoder.model.blocks) + len(self.info_sharing.self_attention_blocks) + views

    def _share(self, features, registers, progress):
        """Alternating attention over the views (V, C, h, w), each followed by its register (class) token, and the global scale token:
        even blocks attend over all tokens, odd blocks within each view (the scale token waits). Returns the view features after the
        intermediate blocks and the final norm, (V, C, h, w) each, and the final scale token (1, C)."""
        views, dim, rows, cols = features.shape
        per_view = rows * cols + 1
        sharing = self.info_sharing
        x = torch.cat([features.flatten(2), registers[:, :, None]], dim=2).transpose(1, 2).reshape(1, views * per_view, dim)
        x = torch.cat([x, self.scale_token[None, None].to(x.dtype)], dim=1)
        x[:, :per_view] += sharing.view_pos_table[0]
        take, outputs = set(self.config["info_sharing"]["indices"]), []
        for index, block in enumerate(sharing.self_attention_blocks):
            if index % 2 == 0:
                x = block(x)
            else:
                x = torch.cat([block(x[:, :-1].reshape(views, per_view, dim)).reshape(1, views * per_view, dim), x[:, -1:]], dim=1)
            if index in take:
                outputs.append(sharing.norm(x))
            progress()
        outputs.append(sharing.norm(x))
        del x
        scale_token = outputs[-1][0, -1:]
        maps = [output[0, :-1].reshape(views, per_view, dim)[:, :-1].transpose(1, 2).reshape(views, dim, rows, cols) for output in outputs]
        return maps, scale_token

    def forward(self, images, progress):
        """images (V, 3, H, W) normalized: per view ray directions (V, H, W, 3), depth along the rays (V, H, W, 1), confidence and
        non-ambiguous mask (V, H, W), camera translations (V, 3) and quaternions (V, 4) in the frame of view 0, the metric scale (1,)."""
        views, _, height, width = images.shape
        rows, cols = height // 14, width // 14
        with torch.autocast(images.device.type, dtype=torch.bfloat16):
            tokens = self.encoder.model(images, {len(self.encoder.model.blocks) - 1}, progress)[0]
            registers = tokens[:, 0]
            features = tokens[:, 1:].transpose(1, 2).reshape(views, -1, rows, cols)
            del tokens
            with torch.autocast(images.device.type, enabled=False):
                features = self.fusion_norm_layer(features.permute(0, 2, 3, 1).float()).permute(0, 3, 1, 2).contiguous()
            maps, scale_token = self._share(features, registers, progress)
        with torch.autocast(images.device.type, enabled=False):
            maps = [features] + maps
            del features
            dense, pose = [], []
            for view in range(views):  # upstream memory-efficient inference: the dense head (the memory bottleneck) by mini-batches
                view_maps = [layer[view:view + 1] for layer in maps]
                dense.append(self.dense_head[1](self.dense_head[0](view_maps), (height, width)))
                pose.append(self.pose_head(view_maps[-1]))
                progress()
            del maps
            dense, pose = torch.cat(dense), torch.cat(pose)
            scale = self.scale_head(scale_token).exp().clamp(min=1e-8).view(1)
            rays, depth, confidence, mask = dense.split([3, 1, 1, 1], dim=1)
            return {"rays": F.normalize(rays, dim=1, eps=1e-8).permute(0, 2, 3, 1), "depth": depth.exp().clamp(min=0).permute(0, 2, 3, 1),
                    "confidence": 1 + confidence.exp()[:, 0], "mask": mask.sigmoid()[:, 0] > 0.5, "translations": pose[:, :3],
                    "quaternions": F.normalize(pose[:, 3:], dim=1, eps=1e-8), "scale": scale}


def preprocess(images, resolution_set):
    """Upstream load_images ("fixed_mapping"): every image is resized (Lanczos down, bicubic up) to cover the 518 px resolution closest to
    the average aspect ratio, centre cropped to it and normalized like DINOv2. Returns (V, 3, H, W) FP32 on the CPU and the RGB images
    (V, H, W, 3) in 0..1."""
    aspect_ratio = sum(image.width / image.height for image in images) / len(images)
    width, height = resolution_set[min(resolution_set, key=lambda key: abs(float(key) - aspect_ratio))]
    tensors = []
    for image in images:
        scale = max(width / image.width, height / image.height) + 1e-8
        size = int(np.floor(image.width * scale)), int(np.floor(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC)
        left, top = (size[0] - width) // 2, (size[1] - height) // 2
        tensors.append(torch.from_numpy(np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.float32) / 255))
    colors = torch.stack(tensors)
    mean, std = torch.tensor(IMAGE_MEAN, device="cpu"), torch.tensor(IMAGE_STD, device="cpu")
    return ((colors - mean) / std).permute(0, 3, 1, 2).contiguous(), colors


def quaternion_to_matrix(quaternions):
    x, y, z, w = F.normalize(quaternions, dim=1).unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=1).view(-1, 3, 3)


def intrinsics_from_rays(rays):
    """Upstream recover_pinhole_intrinsics_from_ray_directions (standard resolution path): per axis least squares x = c + f * d_x / d_z
    on a grid of about 50 x 50 samples, pixel centres at integer coordinates. rays (V, H, W, 3) -> (V, 3, 3)."""
    views, height, width, _ = rays.shape
    rows = torch.arange(0, height, max(1, height // 50), device=rays.device)
    cols = torch.arange(0, width, max(1, width // 50), device=rays.device)
    sampled = rays[:, rows[:, None], cols[None, :]].reshape(views, -1, 3)
    intrinsics = torch.zeros(views, 3, 3, device=rays.device)
    intrinsics[:, 2, 2] = 1
    for axis, coords in ((0, cols.float()[None, :].expand(len(rows), -1)), (1, rows.float()[:, None].expand(-1, len(cols)))):
        design = torch.stack([torch.ones_like(sampled[..., 0]), sampled[..., axis] / sampled[..., 2]], dim=2)
        target = coords.reshape(1, -1, 1).expand(views, -1, -1)
        solution = torch.linalg.solve(design.transpose(1, 2) @ design, design.transpose(1, 2) @ target)[..., 0]
        intrinsics[:, axis, 2], intrinsics[:, axis, axis] = solution[:, 0], solution[:, 1]
    return intrinsics


def reconstruct(output, confidence_percentile):
    """Upstream infer() post-processing: metric world points of every pixel (V, H, W, 3) in the frame of view 0 (OpenCV axes), camera
    to world poses (V, 4, 4), pixel intrinsics and the validity mask (non-ambiguous, optional confidence percentile, without the
    pixels that are both depth and normal edges)."""
    from .geometry import depth_edges, normal_edges, points_to_normals

    scale = output["scale"]
    rotations = quaternion_to_matrix(output["quaternions"])
    camera_points = output["rays"] * output["depth"] * scale
    points = torch.einsum("vij,vhwj->vhwi", rotations, camera_points) + (output["translations"] * scale)[:, None, None]
    poses = torch.eye(4, device=points.device).repeat(len(points), 1, 1)
    poses[:, :3, :3], poses[:, :3, 3] = rotations, output["translations"] * scale
    mask = output["mask"]
    if confidence_percentile > 0:
        confidence = output["confidence"]
        threshold = torch.quantile(confidence.reshape(len(confidence), -1), confidence_percentile / 100, dim=1)
        mask = mask & (confidence > threshold[:, None, None])
    for view in range(len(points)):
        if mask[view].any():
            normals, normals_mask = points_to_normals(points[view], mask[view])
            edges = depth_edges(camera_points[view, ..., 2], mask[view], rtol=0.03) & normal_edges(normals, normals_mask, tolerance=5.0)
            mask[view] &= ~edges
    return points, poses, intrinsics_from_rays(output["rays"]), mask
