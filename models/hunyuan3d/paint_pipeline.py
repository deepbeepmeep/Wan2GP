# Hunyuan3D 2.1 PBR texturing, ported from Tencent Hunyuan3D 2.1 hy3dpaint/{textureGenPipeline.py, utils/pipeline_utils.py,
# utils/multiview_utils.py, hunyuanpaintpbr/pipeline.py, hunyuanpaintpbr/unet/modules.py (position voxel indices),
# DifferentiableRenderer/{MeshRender.py, camera_utils.py}} (Copyright (C) 2025 Tencent; Tencent Hunyuan 3D 2.1 Community License, see
# models/hunyuan3d/LICENSE and Notice.txt).
# Modified by DeepBeepMeep for WanGP: rasterization by shared.mesh3d (pure PyTorch) instead of the custom_rasterizer CUDA extension, with
# the clip coordinates scaled to reproduce its pixel convention; vertex and texture inpainting by shared.mesh3d (vectorized version of
# the mesh_inpaint_processor C++ extension, OpenCV Navier-Stokes); decimation and UV unwrapping by shared.mesh3d (fast-simplification and
# xatlas instead of pymeshlab and trimesh); the albedo and metallic-roughness views baked in one pass per view, blended view by view;
# the multiview UNet run with sequential guidance branches; a GLB with a PBR metallic-roughness material written directly (no Blender);
# xatlas run in a child process while the views are generated (it holds the GIL), the texture channels inpainted in parallel threads,
# OpenCV inpainting limited to the charts and a band around them (push-pull fill of the atlas background on the GPU).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from shared.mesh3d.cleanup import BackgroundUnwrap, inpaint_texture, inpaint_vertex_colors  # BackgroundUnwrap: xatlas in a child process
from shared.mesh3d.raster import interpolate, rasterize
from shared.utils.phase_progress import PhaseProgress, check_abort
from .paint_unet import Branch, PaintContext
from .unipc import UniPCSampler

CAMERA_DISTANCE, ORTHO_SCALE, SCALE_FACTOR = 1.45, 1.2, 1.15
RENDER_SIZE = 2048
BACKGROUND_MARGIN = 16  # bake texels around the UV charts inpainted by OpenCV; the atlas background beyond is filled by push-pull
BAKE_ANGLE_THRESHOLD, BAKE_EXP, DEPTH_THRESHOLD = 75, 4, 3e-3
# Upstream sets its boundary erosion (2 px at 512) while the view selection renders at 1024, and keeps it for the baking at 2048:
# a 9x9 kernel.
UNRELIABLE_KERNEL = int((2 / 512) * 1024) * 2 + 1
CANDIDATE_VIEWS = [(0, 0, 1), (0, 90, 0.1), (0, 180, 0.5), (0, 270, 0.1), (90, 0, 0.05), (-90, 180, 0.05)]  # (elevation, azimuth, weight)
CANDIDATE_VIEWS += [(elevation, azim, 0.01) for azim in range(0, 360, 30) for elevation in (20, -20)]


def get_mv_matrix(elev, azim, camera_distance=CAMERA_DISTANCE):
    """Upstream camera_utils.get_mv_matrix: world-to-camera matrix (float32 numpy) of an orbit camera, z up."""
    elev, azim = -elev, azim + 90
    elev_rad, azim_rad = math.radians(elev), math.radians(azim)
    camera_position = np.array([camera_distance * math.cos(elev_rad) * math.cos(azim_rad), camera_distance * math.cos(elev_rad) * math.sin(azim_rad),
                                camera_distance * math.sin(elev_rad)])
    lookat = -camera_position / np.linalg.norm(-camera_position)
    right = np.cross(lookat, np.array([0, 0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, lookat)
    up = up / np.linalg.norm(up)
    c2w = np.concatenate([np.stack([right, up, -lookat], axis=-1), camera_position[:, None]], axis=-1)
    w2c = np.zeros((4, 4))
    w2c[:3, :3] = np.transpose(c2w[:3, :3], (1, 0))
    w2c[:3, 3:] = -np.matmul(np.transpose(c2w[:3, :3], (1, 0)), c2w[:3, 3:])
    w2c[3, 3] = 1.0
    return w2c.astype(np.float32)


def orthographic_matrix(scale=ORTHO_SCALE, near=0.1, far=100):
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0], matrix[1, 1] = 2 / scale, 2 / scale
    matrix[2, 2], matrix[2, 3] = -2 / (far - near), -(far + near) / (far - near)
    return matrix


def transform_pos(matrix, pos):
    matrix = torch.from_numpy(matrix).to(pos.device)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1], device=pos.device)], dim=1) if pos.shape[-1] == 3 else pos
    return torch.matmul(posw, matrix.t())


class MeshRenderer:
    """The parts of upstream MeshRender used by the texturing pipeline, on mesh3d's rasterizer: z-up renderer coordinates, mesh
    normalized into a sphere of diameter 1.15, orthographic cameras of scale 1.2 at distance 1.45, face-normal shading."""

    def __init__(self, vertices, faces, uvs, texture_size, device):
        """vertices [V, 3] (glTF axes), faces [F, 3], uvs [V, 2] in upstream's renderer convention (v up from the texture's top row), or
        None for a renderer of views only (no back-projection)."""
        self.device, self.texture_size = device, texture_size
        vtx_pos = vertices.to(device=device, dtype=torch.float32).clone()
        vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]
        vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]
        max_bb, min_bb = vtx_pos.max(0)[0], vtx_pos.min(0)[0]
        center = (max_bb + min_bb) / 2
        scale = torch.norm(vtx_pos - center, dim=1).max() * 2.0
        self.vtx_pos = (vtx_pos - center) * (SCALE_FACTOR / float(scale))
        self.pos_idx = faces.to(device=device, dtype=torch.int64)
        self.proj = orthographic_matrix()
        if uvs is not None:
            self.vtx_uv = uvs.to(device=device, dtype=torch.float32)
            self._extract_textiles()

    @staticmethod
    def _rasterize(clip, faces, resolution):
        """custom_rasterizer.rasterize semantics (NDC -1 / +1 on the first / last pixel centres, row 0 at NDC y = -1): 1-based face ids
        (0 = empty) [H, W] and barycentrics [H, W, 3]."""
        height, width = resolution
        clip = clip.clone()
        clip[:, 0] *= (width - 1) / width
        clip[:, 1] *= (height - 1) / height
        face_id, bary, _ = rasterize(clip, faces, (height, width))
        return face_id[0] + 1, bary[0]

    def _view(self, elev, azim):
        r_mv = get_mv_matrix(elev, azim)
        pos_camera = transform_pos(r_mv, self.vtx_pos)
        return r_mv, pos_camera, transform_pos(self.proj, pos_camera)

    def _face_normals(self, positions):
        v = positions[self.pos_idx]
        return F.normalize(torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1), dim=-1)

    def _face_shading(self, findices, face_normals):
        normal = torch.zeros(*findices.shape, 3, device=self.device)
        covered = findices > 0
        normal[covered] = face_normals[(findices[covered] - 1).long()]
        return normal

    def render_normal(self, elev, azim, resolution=RENDER_SIZE):
        """World-space face normals (use_abs_coor), white background, mapped to [0, 1] as a uint8 RGB image (upstream truncates)."""
        _, _, pos_clip = self._view(elev, azim)
        findices, _ = self._rasterize(pos_clip, self.pos_idx, (resolution, resolution))
        normal = self._face_shading(findices, self._face_normals(self.vtx_pos))
        visible = findices.clamp(0, 1)[..., None].float()
        background = torch.ones(3, dtype=torch.float32, device=self.device)
        image = ((normal * visible + background * (1 - visible)) + 1) * 0.5
        return (image.cpu().numpy() * 255).astype(np.uint8)

    def render_position(self, elev, azim, resolution=RENDER_SIZE):
        _, _, pos_clip = self._view(elev, azim)
        findices, bary = self._rasterize(pos_clip, self.pos_idx, (resolution, resolution))
        position = interpolate((0.5 - self.vtx_pos / SCALE_FACTOR).contiguous(), self.pos_idx, findices - 1, bary)
        visible = findices.clamp(0, 1)[..., None].float()
        background = torch.ones(3, dtype=torch.float32, device=self.device)
        return ((position * visible + background * (1 - visible)).cpu().numpy() * 255).astype(np.uint8)

    def viewed_faces(self, elev, azim, resolution=1024):
        _, _, pos_clip = self._view(elev, azim)
        findices, _ = self._rasterize(pos_clip, self.pos_idx, (resolution, resolution))
        return findices

    def face_areas(self):
        v = self.vtx_pos[self.pos_idx]
        return torch.norm(torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1), dim=-1) * 0.5

    def _extract_textiles(self):
        """Texel positions of the UV layout (upstream extract_textiles): world position (homogeneous) and (row, column) of every covered
        texel."""
        size = self.texture_size
        clip = torch.cat([self.vtx_uv, torch.zeros_like(self.vtx_uv[:, :1]), torch.ones_like(self.vtx_uv[:, :1])], dim=1) * 2 - 1
        findices, bary = self._rasterize(clip, self.pos_idx, (size, size))
        mask = (findices > 0).reshape(-1)
        position = interpolate(self.vtx_pos, self.pos_idx, findices - 1, bary).reshape(-1, 3)[mask]
        del bary
        self.tex_position = torch.cat([position, torch.ones_like(position[:, :1])], dim=-1)
        self.tex_grid = torch.nonzero(mask.view(size, size))  # row-major (row, column), the order of upstream's masked meshgrid

    def back_project(self, images, elev, azim):
        """Upstream back_project ("back_sample", face shading) of several images of the same view at once: per image the texture
        [T * T, C] of the sampled colours, the texel weights cos_map [T * T] and the texel indices they were written to."""
        resolution = images[0].shape[:2]
        r_mv, pos_camera, pos_clip = self._view(elev, azim)
        pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]
        findices, bary = self._rasterize(pos_clip, self.pos_idx, resolution)
        visible_mask = findices.clamp(0, 1)[..., None].float()
        normal = self._face_shading(findices, self._face_normals(pos_camera))
        depth = interpolate(pos_camera[:, 2:3].contiguous(), self.pos_idx, findices - 1, bary)
        del bary
        visible = visible_mask > 0
        depth_max, depth_min = depth[visible].max(), depth[visible].min()
        depth_image = (depth - depth_min) / (depth_max - depth_min) * visible_mask
        sketch = cv2.Canny((depth_image.cpu().numpy() * 255).astype(np.uint8), 30, 80)
        sketch_image = (torch.from_numpy(sketch).to(self.device).float() / 255.0)[..., None]
        del depth_image
        lookat = torch.tensor([[0, 0, -1]], device=self.device)
        cos_image = F.cosine_similarity(lookat, normal.view(-1, 3)).view(*normal.shape[:2], 1)
        del normal
        cos_image[cos_image < np.cos(BAKE_ANGLE_THRESHOLD / 180 * np.pi)] = 0
        kernel_size = UNRELIABLE_KERNEL
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=self.device)
        visible_mask = 1.0 - (F.conv2d(1.0 - visible_mask.permute(2, 0, 1)[None], kernel, padding=kernel_size // 2) > 0).float()
        visible_mask = visible_mask[0].permute(1, 2, 0)
        sketch_image = (F.conv2d(sketch_image.permute(2, 0, 1)[None], kernel, padding=kernel_size // 2) > 0).float()[0].permute(1, 2, 0)
        visible_mask = visible_mask * (sketch_image < 0.5)
        del sketch_image
        cos_image[visible_mask == 0] = 0

        proj = self.proj
        img_proj = torch.from_numpy(np.array(((proj[0, 0], 0, 0, 0), (0, proj[1, 1], 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))).to(self.tex_position)
        w2c = torch.from_numpy(r_mv).to(self.tex_position)
        v_proj = self.tex_position @ w2c.T @ img_proj
        inner_mask = (v_proj[:, 0] <= 1.0) & (v_proj[:, 0] >= -1.0) & (v_proj[:, 1] <= 1.0) & (v_proj[:, 1] >= -1.0)
        inner_valid_idx = torch.where(inner_mask)[0].long()
        img_x = torch.clamp(((v_proj[:, 0].clamp(-1, 1) * 0.5 + 0.5) * (resolution[0])).long(), 0, resolution[0] - 1)
        img_y = torch.clamp(((v_proj[:, 1].clamp(-1, 1) * 0.5 + 0.5) * (resolution[1])).long(), 0, resolution[1] - 1)
        indices = img_y * resolution[0] + img_x
        sampled_z, sampled_m = depth.reshape(-1)[indices], visible_mask.reshape(-1)[indices]
        sampled_w = cos_image.reshape(-1)[indices]
        valid_idx = torch.where((torch.abs(v_proj[:, 2] - sampled_z) < DEPTH_THRESHOLD) & (sampled_m * sampled_w > 0))[0]
        valid_idx = valid_idx[torch.isin(valid_idx, inner_valid_idx)].to(inner_valid_idx)
        indices, sampled_w = indices[valid_idx], sampled_w[valid_idx]
        wx = ((v_proj[:, 0] * 0.5 + 0.5) * resolution[0] - img_x)[valid_idx].reshape(-1, 1)
        wy = ((v_proj[:, 1] * 0.5 + 0.5) * resolution[1] - img_y)[valid_idx].reshape(-1, 1)
        img_x, img_y = img_x[valid_idx], img_y[valid_idx]
        del v_proj
        img_x_r, img_y_r = torch.clamp(img_x + 1, 0, resolution[0] - 1), torch.clamp(img_y + 1, 0, resolution[1] - 1)
        indices_lr, indices_rl, indices_rr = img_y * resolution[0] + img_x_r, img_y_r * resolution[0] + img_x, img_y_r * resolution[0] + img_x_r
        tex_indices = self.tex_grid[valid_idx, 0] * self.texture_size + self.tex_grid[valid_idx, 1]
        textures = []
        for image in images:
            rgb = image.reshape(-1, image.shape[-1])
            sampled = (rgb[indices] * (1 - wx) + rgb[indices_lr] * wx) * (1 - wy) + (rgb[indices_rl] * (1 - wx) + rgb[indices_rr] * wx) * wy
            texture = torch.zeros(self.texture_size * self.texture_size, image.shape[-1], device=self.device)
            texture[tex_indices, :] = sampled
            textures.append(texture)
        cos_map = torch.zeros(self.texture_size * self.texture_size, device=self.device)
        cos_map[tex_indices] = sampled_w
        return textures, cos_map


def select_views(renderer, max_views):
    """Upstream bake_view_selection: the 6 canonical views, then greedily the candidate view showing the most still unseen surface
    (more than 1% of the area), up to max_views."""
    areas = torch.cat([torch.zeros(1, device=renderer.device), renderer.face_areas()]).cpu().numpy()
    ratios = areas / areas.sum()
    viewed = [set(torch.unique(renderer.viewed_faces(elev, azim)).cpu().numpy()) for elev, azim, _ in CANDIDATE_VIEWS]  # sorted, as np.unique
    selected = list(range(6))
    seen = set().union(*(viewed[i] for i in selected))
    for _ in range(max_views - len(selected)):
        best, best_gain = -1, 0
        for index in range(len(CANDIDATE_VIEWS)):
            if index in selected:
                continue
            gain = ratios[list(viewed[index] - seen)].sum()
            if gain > best_gain:
                best, best_gain = index, gain
        if best_gain <= 0.01:
            break
        selected.append(best)
        seen |= viewed[best]
    return [CANDIDATE_VIEWS[i] for i in selected]


def voxel_indices(position_maps, grid_resolution, voxel_resolution):
    """Upstream compute_discrete_voxel_indice, in place on position_maps (1, views, 3, H, W) FP16 as upstream (its background becomes 0
    for the following resolutions)."""
    position = position_maps
    _, views, _, height, width = position.shape
    valid_mask = (position != 1).all(dim=2, keepdim=True).expand_as(position)
    position[valid_mask == False] = 0  # noqa: E712
    cells = lambda x: x.reshape(1, views, 3, grid_resolution, height // grid_resolution, grid_resolution, width // grid_resolution).permute(0, 1, 3, 5, 2, 4, 6)
    grid_position = cells(position).sum(dim=(-2, -1))
    count_masked = cells(valid_mask).sum(dim=(-2, -1))
    grid_position = grid_position / count_masked.clamp(min=1)
    grid_position[count_masked < (height // grid_resolution) * (width // grid_resolution) // (4 * 4)] = 0
    grid_position = grid_position.permute(0, 1, 4, 2, 3).clamp(0, 1)
    return torch.round(grid_position * (voxel_resolution - 1)).long()


def rope_tables(position_maps, latent_size, head_dim=64):
    """3D RoPE cos/sin (1, tokens, 1, head_dim) of the multiview attention for each sequence length (views x tokens of a level),
    upstream calc_multires_voxel_idxs and RotaryEmbedding.get_3d_rotary_pos_embed."""
    tables = {}
    for grid_resolution, voxel_resolution in zip([latent_size, latent_size // 2, latent_size // 4, latent_size // 8], [latent_size * 8, latent_size * 4, latent_size * 2, latent_size]):
        index = voxel_indices(position_maps, grid_resolution, voxel_resolution)
        index = index.permute(0, 1, 3, 4, 2).reshape(-1, 3)
        dim_xy, dim_z = head_dim // 8 * 3, head_dim // 8 * 2
        grid = torch.arange(voxel_resolution, dtype=torch.float32, device=index.device)

        def axis(dim):
            freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=index.device)[: dim // 2] / dim)) / 1.0
            freqs = torch.outer(grid, freqs)
            return freqs.cos().repeat_interleave(2, dim=1).float(), freqs.sin().repeat_interleave(2, dim=1).float()

        (xy_cos, xy_sin), (z_cos, z_sin) = axis(dim_xy), axis(dim_z)
        cos = torch.cat((xy_cos[index[:, 0]], xy_cos[index[:, 1]], z_cos[index[:, 2]]), dim=-1)
        sin = torch.cat((xy_sin[index[:, 0]], xy_sin[index[:, 1]], z_sin[index[:, 2]]), dim=-1)
        tables[index.shape[0]] = (cos[None, :, None], sin[None, :, None])
    return tables


def to_tensor_views(images, device):
    """Upstream convert_pil_list_to_tensor: uint8 RGB images -> (1, views, 3, H, W) FP16 in [0, 1]."""
    return torch.stack([torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1) for image in images])[None].half().to(device)


def encode_images(vae, images, generator):
    """Upstream encode_images: images (1, views, 3, H, W) in [0, 1] -> VAE latents (1, views, 4, h, w) sampled from the posterior, one
    image at a time through the encoder, with one draw for all views as upstream's batched call."""
    flat = (images[0] - 0.5) * 2.0
    moments = torch.cat([vae.quant_conv(vae.encoder(flat[i:i + 1])) for i in range(flat.shape[0])])
    mean, logvar = torch.chunk(moments, 2, dim=1)
    std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
    noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
    return ((mean + std * noise) * vae.config.scaling_factor)[None]


def decode_latents(vae, latents):
    """VAE decode and upstream image postprocessing ((x / 2 + 0.5) clamped, rounded to uint8), one image at a time."""
    images = []
    with PhaseProgress(len(latents), "Decoding Texture Views", "views") as progress:
        for latent in latents:
            image = vae._decode(latent[None] / vae.config.scaling_factor, return_dict=False)[0]
            images.append(((image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()[0] * 255).round().astype("uint8"))
            progress.advance()
    return images


def multiview_diffusion(unet, unet_dual, vae, dino_features, reference, normal_maps, position_maps, view_size, steps, guidance_scale, seed, callback, device):
    """Upstream multiviewDiffusionNet.forward_one + HunyuanPaintPipeline: albedo and metallic-roughness images of every view."""
    dtype = torch.float16
    views = len(normal_maps)
    generator = torch.Generator(device=device).manual_seed(seed)  # upstream's global generator after seed_everything(seed)
    ref = torch.from_numpy(np.array(reference) / 255.0)[None].permute(0, 3, 1, 2)[None].to(device=device, dtype=dtype)
    ref_latents = encode_images(vae, ref, generator)
    embeds_normal = encode_images(vae, to_tensor_views(normal_maps, device), generator)
    position_images = to_tensor_views(position_maps, device)
    embeds_position = encode_images(vae, position_images, generator)
    latent_size = view_size // 8
    rope = rope_tables(position_images, latent_size)
    del position_images
    references = unet_dual(ref_latents[0], unet.learned_text_clip_ref[None].to(device))
    del ref_latents
    dino_tokens = unet.project_dino(dino_features)
    zero_tokens = unet.project_dino(torch.zeros_like(dino_features))
    zero_branch = dict()  # the unconditional and reference branches share the zero-image DINO tokens and their keys/values
    branches = [Branch(0.0, zero_tokens, zero_branch), Branch(1.0, zero_tokens, zero_branch), Branch(1.0, dino_tokens)]
    ctx = PaintContext(views, references, rope)
    conditions = torch.cat([embeds_normal, embeds_position], dim=2)[0].repeat(2, 1, 1, 1)  # materials x views
    del embeds_normal, embeds_position
    sampler = UniPCSampler()
    timesteps = sampler.set_timesteps(steps, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((2 * views, 4, latent_size, latent_size), generator=generator, device=device, dtype=dtype)
    view_scale = torch.ones((2 * views, 1, 1, 1), dtype=dtype, device=device)
    title = "Generating Texture Views"
    if callback is not None:
        callback(-1, None, True, override_num_inference_steps=steps, progress_title=title)
    for i, t in enumerate(tqdm(timesteps, desc=title)):
        uncond, ref_pred, full = unet(torch.cat([latents, conditions], dim=1), t, ctx, branches)
        noise_pred = uncond + guidance_scale * view_scale * (ref_pred - uncond)
        noise_pred += guidance_scale * view_scale * (full - ref_pred)
        del uncond, ref_pred, full
        latents = sampler.step(noise_pred, latents)
        del noise_pred
        if callback is not None:
            callback(i, None, False, override_num_inference_steps=steps, progress_title=title)
        check_abort()
    del ctx, branches, references, conditions
    images = decode_latents(vae, latents)
    return images[:views], images[views:]


def bake_views(renderer, albedo_views, mr_views, cameras):
    """Upstream bake_from_multiview + fast_bake_texture for albedo and metallic-roughness together: cos^4 weighted blend of the
    back-projected views, a view skipped when it adds less than 1% of new texels. Returns the two float textures [T, T, 3] and the mask
    of painted texels (uint8 0/255)."""
    size = renderer.texture_size
    merged = [torch.zeros(size * size, 3, device=renderer.device) for _ in range(2)]
    trust = torch.zeros(size * size, device=renderer.device)
    with PhaseProgress(len(cameras), "Baking Texture", "views") as progress:
        for albedo, mr, (elev, azim, weight) in zip(albedo_views, mr_views, cameras):
            images = [torch.from_numpy(np.array(image)).to(renderer.device).double().div_(255.0).float() for image in (albedo, mr)]  # float64 / 255 as upstream's numpy
            textures, cos_map = renderer.back_project(images, elev, azim)
            del images
            cos_map = weight * (cos_map ** BAKE_EXP)
            view_sum = (cos_map > 0).sum()
            if not ((cos_map > 0) * (trust > 0)).sum() / view_sum > 0.99:  # a view without texels (0 / 0) is not skipped, as upstream
                for target, texture in zip(merged, textures):
                    target += texture * cos_map[:, None]
                trust += cos_map
            del textures, cos_map
            progress.advance()
    textures = [(target / torch.clamp(trust, min=1e-8)[:, None]).view(size, size, 3) for target in merged]
    mask = ((trust > 1e-8).view(size, size).cpu().numpy().astype(np.float32) * 255).astype(np.uint8)
    return textures, mask


def push_pull(image, known):
    """Fills the texels of image (uint8 [H, W, C], H and W powers of 2) where known is False from the mean of the known texels of the
    smallest pyramid cell holding some, upsampled bilinearly coarse to fine. Returns uint8 [H, W, C] on known's device."""
    known = known.float()[None, None]
    values = torch.from_numpy(image).to(known.device).permute(2, 0, 1)[None].float() * known
    levels = []
    while values.shape[-1] > 1:
        levels.append((values, known))
        values, known = F.avg_pool2d(values, 2), F.avg_pool2d(known, 2)
    filled = values / known.clamp_min(1e-12)
    for values, known in reversed(levels):
        coarse = F.interpolate(filled, size=values.shape[-2:], mode="bilinear", align_corners=False)
        filled = torch.where(known > 0, values / known.clamp_min(1e-12), coarse)
        del coarse
    return filled[0].permute(1, 2, 0).round_().clamp_(0, 255).to(torch.uint8)


def inpaint(renderer, textures, mask, channels):
    """Upstream uv_inpaint of textures sharing the painted mask: vertex-colour inpainting at the uv corners, then OpenCV Navier-Stokes
    inpainting (radius 3) of the texels still unpainted. Unlike upstream, OpenCV only fills the charts and a band of BACKGROUND_MARGIN
    texels around them: the atlas background beyond, which the mesh never shows (only mipmaps read it), is filled on the GPU by push-pull
    first. Upstream's whole-image inpainting cost seconds per channel; the texels of the charts are the same but for about 0.04% of them
    (unseen texels near the edge of the band). OpenCV inpaints each channel independently and on one thread, releasing the GIL: the
    channels kept (channels[i] of textures[i]) run in parallel threads. textures float [T, T, 3] in [0, 1]; returns uint8
    [T, T, len(channels[i])] for each texture."""
    size, margin = renderer.texture_size, BACKGROUND_MARGIN
    near = torch.zeros(size * size, device=renderer.device)
    near[renderer.tex_grid[:, 0] * size + renderer.tex_grid[:, 1]] = 1
    near = F.max_pool2d(near.view(1, 1, size, size), 2 * margin + 1, stride=1, padding=margin)[0, 0] > 0
    jobs = []
    for texture, kept in zip(textures, channels):
        texture, mask_t = inpaint_vertex_colors(texture.cpu(), torch.from_numpy(mask), renderer.vtx_pos.cpu(), renderer.vtx_uv.cpu(), renderer.pos_idx.cpu())
        texture_uint8 = (texture.numpy() * 255).astype(np.uint8)[..., list(kept)]
        painted = mask_t.to(renderer.device) > 0
        far = ~painted & ~near
        filled = torch.from_numpy(texture_uint8).to(renderer.device)
        filled[far] = push_pull(texture_uint8, painted)[far]
        holes = (~painted & near).to(torch.uint8).cpu() * 255
        jobs += [(filled[..., c:c + 1].cpu(), holes) for c in range(len(kept))]
        del filled, painted, far
    with ThreadPoolExecutor(len(jobs)) as pool:
        filled = list(pool.map(lambda job: inpaint_texture(*job, radius=3, method="ns").numpy(), jobs))
    ends = np.cumsum([len(kept) for kept in channels])
    return [np.concatenate(filled[end - len(kept):end], axis=2) for end, kept in zip(ends, channels)]
