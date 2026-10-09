# Uni3C camera control (PCDController) for Wan2.1 I2V 14B, ported from https://github.com/alibaba-damo-academy/Uni3C
# (src/models/pcd_controller.py, src/models/controlnet.py, src/camera.py, src/pointcloud.py), Copyright Alibaba DAMO Academy,
# Apache License 2.0. The start image is lifted with Depth Anything 3 Metric and rendered with Vista4D's point z-buffer renderer
# instead of Depth Pro and PyTorch3D; human-motion (SMPL) conditioning is not ported.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.utils import files_locator as fl
from shared.utils.phase_progress import control_video_encoding
from ..modules.model import WanSelfAttention
from ..modules.posemb_layers import get_nd_rotary_pos_embed
from ..vista4d.preprocess import _render_point_chunks_torch
from .trajectory import build_camera_path

UNI3C_FOCAL = 0.866  # focal length / longest side (60 degrees field of view) assumed for the start image, Depth Anything 3 Metric gives none
UNI3C_SUPERSAMPLING = 2  # points lifted per pixel side, so that 1-pixel splats leave no cracks when the camera moves closer


class Uni3CSelfAttention(WanSelfAttention):
    pass  # same layers and attention path, without the main blocks' radial / chipmunk attentions


class Uni3CBlock(nn.Module):
    """Uni3C SingleAttentionBlock: CogVideoX-style LayerNormZero modulated by the main model time embedding."""

    def __init__(self, dim, ffn_dim, num_heads, time_embed_dim, out_dim, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm1_modulation = nn.Linear(time_embed_dim, 3 * dim)
        self.self_attn = Uni3CSelfAttention(dim, num_heads, qk_norm=True, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2_modulation = nn.Linear(time_embed_dim, 3 * dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.proj_out = nn.Linear(dim, out_dim)

    def forward(self, x, temb, freqs):
        """x [1, tokens, dim] is updated in place."""
        temb = F.silu(temb).to(self.norm1_modulation.weight.dtype)
        shift, scale, gate = self.norm1_modulation(temb).unsqueeze(1).chunk(3, dim=-1)
        y = self.norm1(x.float()).mul_(1 + scale).add_(shift).to(x.dtype)
        x += self.self_attn([y], None, freqs)[0].mul_(gate)
        shift, scale, gate = self.norm2_modulation(temb).unsqueeze(1).chunk(3, dim=-1)
        for part in torch.split(x, max(1, x.shape[1] // 4), dim=1):  # rows are independent: the FFN by quarters of the tokens
            y = self.norm2(part.float()).mul_(1 + scale).add_(shift).to(x.dtype)
            part += self.ffn(y).mul_(gate)
        return x


class Uni3CControlNet(nn.Module):
    """Point cloud renders, render holes and camera rays to residuals added after the first num_layers blocks of the main model."""

    def __init__(self, in_dim, out_dim, conv_out_dim=5120, time_embed_dim=5120, dim=1024, ffn_dim=4096, num_heads=16, num_layers=20, add_channels=7, mid_channels=256, patch_size=(1, 2, 2)):
        super().__init__()
        self.head_dim = dim // num_heads
        self.patch_embedding = nn.Conv3d(in_dim, conv_out_dim, kernel_size=patch_size, stride=patch_size)
        self.mask_proj = nn.Sequential(nn.Conv3d(add_channels, mid_channels, kernel_size=(4, 8, 8), stride=(4, 8, 8)), nn.GroupNorm(mid_channels // 8, mid_channels), nn.SiLU())
        self.mask_zero_proj = nn.Conv3d(mid_channels, conv_out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_in = nn.Linear(conv_out_dim, dim)
        self.blocks = nn.ModuleList([Uni3CBlock(dim, ffn_dim, num_heads, time_embed_dim, out_dim) for _ in range(num_layers)])

    def fp32_layers(self):
        """Layers kept in float32 as upstream, which runs them outside its bfloat16 autocast (embeddings) or in float32 under it (norms)."""
        return [self.patch_embedding, self.mask_proj[0], self.mask_proj[1], self.mask_zero_proj] + [norm for block in self.blocks for norm in (block.norm1, block.norm2)]

    def condition_features(self, chunks):
        """Render holes + camera rays, [1, 7, 4, H, W] per latent frame, to [1, mid_channels, frames, H / 8, W / 8] (group norm over all)."""
        x = torch.cat([self.mask_proj[0](chunk.to(self.mask_proj[0].weight.dtype)) for chunk in chunks], dim=2)
        return self.mask_proj[2](self.mask_proj[1](x))

    def embed(self, x, condition_features):
        """x [1, 36, F, H, W]: noisy latents, i2v mask and render latents. Returns [1, F * H * W / 4, dim], one latent frame at a time."""
        tokens = None
        for frame_no in range(x.shape[2]):
            frame = self.patch_embedding(x[:, :, frame_no:frame_no + 1].to(self.patch_embedding.weight.dtype))
            frame += self.mask_zero_proj(condition_features[:, :, frame_no:frame_no + 1])
            frame = self.proj_in(frame.flatten(2).transpose(1, 2).to(self.proj_in.weight.dtype))
            if tokens is None:
                tokens = frame.new_empty(1, frame.shape[1] * x.shape[2], frame.shape[2])
            tokens[:, frame_no * frame.shape[1]:(frame_no + 1) * frame.shape[1]] = frame
        return tokens


def _boundary_mask(disparity, beta=10.0, threshold=0.35):
    """Uni3C get_boundaries_mask: depth edges (kornia unnormalized Sobel, replicate padding) whose points are dropped."""
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
    sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=disparity.device)
    padded = F.pad(disparity[None, None], (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(padded, sobel[None, None]), F.conv2d(padded, sobel.t()[None, None])
    return (torch.exp(-beta * torch.sqrt(gx ** 2 + gy ** 2)) < threshold)[0, 0]


def _lift_points(image, depth, intrinsics):
    """Colored points of the start image in its camera frame. image [3, H, W] in [-1, 1], depth [H, W]; Uni3C contracts the depths
    beyond 8 x the median and drops depth edges."""
    height, width = depth.shape
    mid_depth = depth.median() * 8
    depth = torch.where(depth > mid_depth, 2 * mid_depth - mid_depth ** 2 / (depth + 1e-6), depth)
    keep = ~_boundary_mask(1 / (depth + 1e-7))
    size, scale = (height * UNI3C_SUPERSAMPLING, width * UNI3C_SUPERSAMPLING), UNI3C_SUPERSAMPLING
    depth = F.interpolate(depth[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
    colors = F.interpolate(image[None].float(), size=size, mode="nearest")[0].permute(1, 2, 0)  # each point keeps its pixel color
    keep = F.interpolate(keep[None, None].float(), size=size, mode="nearest")[0, 0] > 0.5
    ys, xs = torch.meshgrid(torch.arange(size[0], device=depth.device, dtype=torch.float32), torch.arange(size[1], device=depth.device, dtype=torch.float32), indexing="ij")
    fx, fy, cx, cy = intrinsics.tolist()
    points = torch.stack((((xs + 0.5) / scale - cx) / fx * depth, ((ys + 0.5) / scale - cy) / fy * depth, depth), dim=-1)
    return points[keep], colors[keep].add_(1).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)


def _render(image, points, colors, c2w, intrinsics):
    """Render video [3, F, H, W] in [-1, 1] (holes at 0 as upstream's background) and holes mask [F, H, W]; frame 0 is the start image."""
    height, width = image.shape[-2:]
    w2c = torch.linalg.inv(c2w)
    pixel_centers = intrinsics - intrinsics.new_tensor([0, 0, 0.5, 0.5])  # the z-buffer renderer rounds to integer pixel centers
    frames, holes = [], []
    for frame_no in range(c2w.shape[0]):
        frame, covered, _ = _render_point_chunks_torch([(points, colors, False)], w2c[frame_no], pixel_centers[frame_no], height, width)
        frames.append(torch.from_numpy(frame))
        holes.append(torch.from_numpy(~covered))
    video = torch.stack(frames).to(image.device).permute(3, 0, 1, 2).float().div_(127.5).sub_(1)
    holes = torch.stack(holes).to(image.device).float()
    video.mul_(1 - holes)
    video[:, 0], holes[0] = image, 0
    return video, holes


def _camera_center_normalization(w2c, camera_scale=2.0):
    """Uni3C (from SEVA): first view as identity, cameras centered and scaled so that the first one is at camera_scale."""
    w2c = torch.linalg.inv(w2c[:1]) @ w2c
    c2w = torch.linalg.inv(w2c)
    distances = torch.norm(c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid = distances <= torch.clamp(torch.quantile(distances, 0.97) * 10, max=1e6)
    c2w[:, :3, 3] -= c2w[valid, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)
    first_distance = torch.norm(c2w[0, :3, 3])
    w2c[:, :3, 3] *= camera_scale if torch.isclose(first_distance, torch.zeros_like(first_distance), atol=1e-5) else camera_scale / first_distance
    return w2c


def _camera_rays(K, w2c, height, width):
    """Uni3C batch_sample_rays + embed_rays (x offset by -0.5 and y by +0.5 as upstream): [6, n, H, W] = (direction, origin x direction)."""
    c2w = torch.linalg.inv(w2c)[:, :3]
    x = torch.arange(width, device=w2c.device, dtype=torch.float32) - 0.5
    y = torch.arange(height, device=w2c.device, dtype=torch.float32) + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1).transpose(0, 1).reshape(1, height * width, 2)
    points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    rays_d = F.normalize(points @ torch.linalg.inv(K).transpose(-1, -2) @ c2w[:, :, :3].transpose(-1, -2), dim=-1)
    rays_o = c2w[:, None, :, 3].expand_as(rays_d)
    return torch.cat([rays_d, torch.cross(rays_o, rays_d, dim=-1)], dim=-1).view(-1, height, width, 6).permute(3, 0, 1, 2)


def _condition_chunks(holes, K, w2c, lat_frames, frame_step):
    """Per latent frame: holes and rays of its frames, padded at the start with frame_step - 1 empty frames as upstream."""
    height, width = holes.shape[-2:]
    for latent_no in range(lat_frames):
        frames = slice(0, 1) if latent_no == 0 else slice(frame_step * (latent_no - 1) + 1, frame_step * latent_no + 1)
        count = frames.stop - frames.start
        chunk = torch.cat([holes[None, frames], _camera_rays(K.expand(count, -1, -1), w2c[frames], height, width)])[None]
        yield F.pad(chunk, (0, 0, 0, 0, frame_step - count, 0))


def _estimate_metric_depth(image, focal):
    """Depth Anything 3 Metric on the start image; metric depth = focal * output / 300 (focal in pixels of the processed image)."""
    from preprocessing.depth_anything_v3.depth import DA3_METRIC_BF16_MODEL, _infer_da3_depth_prediction, _load_da3
    frame = image.permute(1, 2, 0).add(1).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
    with torch.device(image.device):  # the Depth Anything 3 code builds some of its tensors on the default device
        model = _load_da3(fl.locate_file(DA3_METRIC_BF16_MODEL), image.device, model_name="da3metric-large")
        with torch.inference_mode():
            depth = _infer_da3_depth_prediction(model, frame[None], [0], max(frame.shape[:2]))[0]
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    return torch.from_numpy(depth).to(image.device).mul_(focal / 300)


def prepare_uni3c(pipeline, image, model_mode, custom_path, custom_settings, lat_frames, height, width, tile_size):
    """Uni3C conditions of a start image [3, H, W] in [-1, 1] along a camera path: render latents, holes + rays features and RoPE."""
    device, frame_step = pipeline.device, pipeline.vae_stride[0]
    image = image.to(device=device, dtype=torch.float32)
    focal = UNI3C_FOCAL * max(height, width)
    default_intrinsics = [focal, focal, width / 2, height / 2]
    if custom_path:
        _, path_intrinsics = build_camera_path(model_mode, custom_path, custom_settings, (lat_frames - 1) * frame_step + 1, height, width, default_intrinsics, 1.0)
        focal = float(path_intrinsics[0, 0])
    depth = _estimate_metric_depth(image, focal)
    if pipeline._interrupt:
        return None
    c2w, intrinsics = build_camera_path(model_mode, custom_path, custom_settings, (lat_frames - 1) * frame_step + 1, height, width, default_intrinsics, float(depth.median()))
    c2w, intrinsics = torch.from_numpy(c2w).to(device), torch.from_numpy(intrinsics).to(device)
    points, colors = _lift_points(image, depth, intrinsics[0])
    video, holes = _render(image, points, colors, c2w, intrinsics)
    points = colors = depth = None
    with control_video_encoding():
        render_latents = pipeline.vae.encode([video.to(pipeline.VAE_dtype)], tile_size=tile_size)[0].unsqueeze(0)
    video = None
    fx, fy, cx, cy = intrinsics[0].tolist()
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device)[None]  # upstream embeds the rays with the first intrinsics
    w2c = _camera_center_normalization(torch.linalg.inv(c2w))
    condition_features = pipeline.model.uni3c.condition_features(_condition_chunks(holes, K, w2c, lat_frames, frame_step))
    head_dim = pipeline.model.uni3c.head_dim
    freqs = get_nd_rotary_pos_embed((lat_frames, height // 16, width // 16), rope_dim_list=[head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)], head_dim=head_dim, device=device)
    return {"render_latents": render_latents, "condition_features": condition_features, "freqs": freqs}
