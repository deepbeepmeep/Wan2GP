# WorldMirror 2.0 prediction heads, ported from Tencent HY-World 2.0 (hyworld2/worldrecon/hyworldmirror/models: heads/camera_head.py,
# inspired by VGGT; heads/dense_head.py, inspired by Depth-Anything-V2; models/rasterization.py GaussianSplatRenderer.gs_head;
# utils/grid.py). Copyright (C) 2026 Tencent, licensed under the Tencent HY-WORLD 2.0 Community License Agreement (LICENSE in this
# folder). Modified by DeepBeepMeep for WanGP (2026-10-08): inference only, FP32 like upstream (its heads run with autocast disabled),
# camera-head attention through shared.attention_kit (SDPA, FP32), the gsplat rasterizer and training paths removed, positional
# embeddings cached per size, in-place additions where the input is no longer read.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import Block, Mlp


class CameraHead(nn.Module):
    """Iterative refinement of a 9-value camera code per view (translation, quaternion x y z w, vertical and horizontal fields of view) from
    the camera tokens of the last layer; the views attend to each other."""

    def __init__(self, dim, depth, heads):
        super().__init__()
        self.refine_net = nn.Sequential(*[Block(dim, heads, compute_dtype=torch.float32) for _ in range(depth)])
        self.token_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)
        self.init_token = nn.Parameter(torch.empty(1, 1, 9))
        self.param_embed = nn.Linear(9, dim)
        self.adapt_norm_gen = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.adapt_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.param_predictor = Mlp(dim, dim // 2, 9)

    def forward(self, tokens, steps):
        """tokens: (1, views, dim) FP32 -> activated camera codes (1, views, 9)."""
        tokens = self.token_norm(tokens)
        prediction = None
        for _ in range(steps):
            embedded = self.param_embed(self.init_token.expand(*tokens.shape[:2], -1) if prediction is None else prediction)
            shift, scale, gate = self.adapt_norm_gen(embedded).chunk(3, dim=-1)
            x = gate * (self.adapt_norm(tokens) * (1 + scale) + shift) + tokens
            for block in self.refine_net:
                block(x, force_attention="sdpa")
            hidden = self.param_predictor.fc1(self.out_norm(x))
            delta = self.param_predictor.fc2(0.5 * hidden * (1 + torch.erf(hidden * 2 ** -0.5)))  # upstream MlpFP32 (erf GELU)
            prediction = delta if prediction is None else prediction + delta
        return torch.cat([prediction[..., :7], F.relu(prediction[..., 7:])], dim=-1)


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1)

    def forward(self, x):
        x = F.relu_(x)  # upstream's nn.ReLU(inplace=True) rewrites its input: the residual adds relu(x)
        return self.conv2(F.relu_(self.conv1(x))).add_(x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, has_residual=True):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, 1)
        self.resConfUnit1 = ResidualConvUnit(features) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, x, skip=None, size=None):
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        x = F.interpolate(x, size=size or (2 * x.shape[-2], 2 * x.shape[-1]), mode="bilinear", align_corners=True)
        return self.out_conv(x)


def uv_embedding(height, width, channels, aspect_ratio, device):
    """create_uv_grid + position_grid_to_embed (sinusoids computed in FP64, rounded to FP32) times the 0.1 ratio: (channels, height, width)
    FP32, computed by rows of positions (each position is independent)."""
    diagonal = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x, span_y = aspect_ratio / diagonal, 1.0 / diagonal
    xs = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, steps=width, dtype=torch.float32, device=device)
    ys = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, steps=height, dtype=torch.float32, device=device)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2)
    omega = torch.arange(channels // 4, dtype=torch.float64, device=device)
    omega /= channels / 4.0
    omega = 1.0 / 100 ** omega
    table = torch.empty(height * width, channels, dtype=torch.float32, device=device)
    for start in range(0, len(grid), 65536):
        rows = grid[start:start + 65536]
        out_x, out_y = torch.einsum("m,d->md", rows[:, 0], omega), torch.einsum("m,d->md", rows[:, 1], omega)
        table[start:start + 65536] = torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)
    return (table * 0.1).T.reshape(channels, height, width)


class DPTHead(nn.Module):
    """Dense head over the 4 intermediate layers: per-pixel outputs at the input resolution. With merge_image (the Gaussian head), it also
    returns the 128-channel fused features plus the image features that the Gaussian parameter convolution reads."""

    def __init__(self, dim_in, output_dim, patch_size, features, out_channels, merge_image=False):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, channels, 1) for channels in out_channels])
        self.resize_layers = nn.ModuleList([nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, 4), nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2),
                                            nn.Identity(), nn.Conv2d(out_channels[3], out_channels[3], 3, 2, 1)])
        self.scratch = nn.Module()
        for index, channels in enumerate(out_channels):
            setattr(self.scratch, f"layer{index + 1}_rn", nn.Conv2d(channels, features, 3, 1, 1, bias=False))
        for index in range(1, 5):
            setattr(self.scratch, f"refinenet{index}", FeatureFusionBlock(features, has_residual=index < 4))
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, 1, 1)
        self.scratch.output_conv2 = nn.Sequential(nn.Conv2d(features // 2, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(32, output_dim, 1))
        self.input_merger = nn.Sequential(nn.Conv2d(3, features // 2, 7, 1, 3), nn.ReLU()) if merge_image else None

    def forward(self, token_list, images, frames, patch_start, positions):
        """token_list: 4 x (views, tokens, dim_in); images: the (n, 3, H, W) views of `frames` (a slice) in [0, 1]; positions(h, w, c,
        aspect) returns the cached uv embedding. Returns the raw output channels (n, output_dim, H, W) [and the merged features]."""
        height, width = images.shape[-2:]
        patches_h, patches_w = height // self.patch_size, width // self.patch_size
        features = []
        for project, resize, tokens in zip(self.projects, self.resize_layers, token_list):
            x = self.norm(tokens[frames, patch_start:])
            x = project(x.transpose(1, 2).reshape(len(x), -1, patches_h, patches_w))
            features.append(resize(x.add_(positions(patches_h, patches_w, x.shape[1], width / height))))
            del x
        scratch = self.scratch
        layers = [getattr(scratch, f"layer{index + 1}_rn")(feature) for index, feature in enumerate(features)]
        del features
        out = scratch.refinenet4(layers[3], size=layers[2].shape[2:])
        out = scratch.refinenet3(out, layers[2], size=layers[1].shape[2:])
        out = scratch.refinenet2(out, layers[1], size=layers[0].shape[2:])
        out = scratch.refinenet1(out, layers[0])
        del layers
        fused = F.interpolate(scratch.output_conv1(out), size=(height, width), mode="bilinear", align_corners=True)
        del out
        fused.add_(positions(height, width, fused.shape[1], width / height))
        raw = scratch.output_conv2(fused)
        if self.input_merger is None:
            return raw
        return raw, fused.add_(self.input_merger(images))


class GaussianParameterHead(nn.Module):
    """GaussianSplatRenderer.gs_head: 12 raw channels per pixel (quaternion 4, log scales 3, opacity 1, SH DC residual 3, merge weight 1)."""

    def __init__(self, features):
        super().__init__()
        self.gs_head = nn.Sequential(nn.Conv2d(features // 2, features, 3, padding=1, bias=False), nn.ReLU(True), nn.Conv2d(features, 12, 1))


class WorldMirrorHeads(nn.Module):
    def __init__(self, dim_in, patch_size, features, gs_features, out_channels, camera_depth, camera_heads, camera_steps):
        super().__init__()
        self.camera_steps = camera_steps
        self.cam_head = CameraHead(dim_in, camera_depth, camera_heads)
        self.depth_head = DPTHead(dim_in, 3, patch_size, features, out_channels)
        self.norm_head = DPTHead(dim_in, 4, patch_size, features, out_channels)
        self.gs_head = DPTHead(dim_in, 3, patch_size, gs_features, out_channels, merge_image=True)
        self.gs_renderer = GaussianParameterHead(gs_features)
