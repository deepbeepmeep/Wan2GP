# WorldMirror 2.0 reconstruction pipeline (images and/or a video to Gaussian splats or a point cloud, with cameras), inference port of
# Tencent HY-World 2.0 (hyworld2/worldrecon/pipeline.py and hyworldmirror/utils/inference_utils.py). Copyright (C) 2026 Tencent, licensed
# under the Tencent HY-WORLD 2.0 Community License Agreement (LICENSE in this folder). Modified by DeepBeepMeep for WanGP (2026-10-08):
# single GPU without FSDP, inputs from WanGP (view images and a reference video), heads run by chunks of views with only their small
# outputs kept, filtering and voxel merging on the GPU, outputs in glTF axes for the 3D gallery (INRIA splat PLY, point PLY and cameras).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import math
import os

import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload
from PIL import Image

from shared.mesh3d.render import perspective, splat_preview
from shared.mesh3d.splats import points_to_splats, splats_opencv_to_gltf, splats_to_ply, write_points_ply
from shared.utils import files_locator as fl
from shared.utils.phase_progress import PhaseProgress, generation_progress, set_phase_status
from . import geometry
from .heads import WorldMirrorHeads, uv_embedding
from .settings import DEFAULT_CUSTOM_SETTINGS
from .transformer import VisualGeometryTransformer, qkv_split_map
from .video_frames import select_frames

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "worldmirror_2.json")
HEADS_FILE = "worldmirror/worldmirror_2_heads.safetensors"
PREVIEW_SIZE = 512
HEAD_CHUNK_PIXELS = 1 << 20  # views per dense-head chunk: about 1 Mpixel (a full-resolution view costs ~1 GB in the Gaussian head)


def _rgb(image):
    """_handle_alpha_channel: RGBA composited over white, then RGB."""
    if image.mode == "RGBA":
        image = Image.alpha_composite(Image.new("RGBA", image.size, (255, 255, 255, 255)), image)
    return image.convert("RGB")


def _resize_dims(width, height, target, patch_size=14):
    """_calculate_resize_dims: the long side becomes target, the short side the nearest multiple of the patch size."""
    if width >= height:
        return target, round(height * (target / width) / patch_size) * patch_size
    return round(width * (target / height) / patch_size) * patch_size, target


def prepare_views(images, max_resolution, device):
    """compute_adaptive_target_size + prepare_images_to_tensor: the first view's long side, at most max_resolution, floored to a multiple
    of 14, sets the size; views are resized (bicubic) to it, and views of another aspect ratio are resized to cover it and centre-cropped
    (upstream rejects them). Returns (views, 3, H, W) FP32 in [0, 1]."""
    first = _rgb(images[0])
    target = max(min(max(first.size), max_resolution) // 14 * 14, 28)
    width, height = _resize_dims(*first.size, target)
    tensors = []
    for image in images:
        image = _rgb(image)
        size = _resize_dims(*image.size, target)
        if size != (width, height):
            scale = max(width / image.width, height / image.height)
            size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.BICUBIC)
        if size != (width, height):
            left, top = (size[0] - width) // 2, (size[1] - height) // 2
            image = image.crop((left, top, left + width, top + height))
        tensors.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
    return torch.stack(tensors).to(device).float().div_(255)


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False):
        self.device = torch.device("cuda")
        with open(CONFIG_PATH, encoding="utf-8") as reader:
            self.config = json.load(reader)
        with init_empty_weights():
            self.transformer = VisualGeometryTransformer(**self.config["trunk"])
            self.heads = WorldMirrorHeads(**self.config["heads"])
        # FP32 checkpoint: as under the upstream BF16 autocast, the matmul / convolution weights of the trunk become BF16 and its tokens,
        # norms and layer scales stay FP32 (locked); the heads run in FP32 like upstream (autocast disabled there)
        self.transformer.lock_fp32_modules()
        offload.load_model_data(self.transformer, model_filename[0], writable_tensors=False, default_dtype=None, fp32_dtype=torch.bfloat16,
                                fused_split_map=qkv_split_map(self.config["trunk"]["embed_dim"]))
        self.transformer.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.transformer, model_type, model_filename[0], torch.bfloat16, None)
        offload.load_model_data(self.heads, fl.locate_file(HEADS_FILE), writable_tensors=False, default_dtype=None,
                                fused_split_map=qkv_split_map(self.config["heads"]["dim_in"]))
        self.heads._convertWeightsFloatTo = None
        self.heads.eval().requires_grad_(False)
        self._interrupt = False

    def pipe(self):
        return {"transformer": self.transformer, "heads": self.heads}

    def _dense_predictions(self, token_list, views, gaussians):
        """Depth, depth-mask logits and normals of every view, plus the Gaussian depth, its mask logits and the 12 raw Gaussian channels,
        computed by chunks of views so that only these outputs are kept."""
        heads, patch_start = self.heads, self.transformer.patch_start_idx
        height, width = views.shape[-2:]
        cache = {}

        def positions(rows, columns, channels, aspect):
            if (rows, columns, channels) not in cache:
                cache[rows, columns, channels] = uv_embedding(rows, columns, channels, aspect, views.device)
            return cache[rows, columns, channels]

        outputs = {name: [] for name in ("depth", "depth_mask", "normals", "gs_depth", "gs_mask", "gs_raw")}
        step = max(1, HEAD_CHUNK_PIXELS // (height * width))
        title = "Predicting Depth, Normals and Gaussians" if gaussians else "Predicting Depth and Normals"
        with PhaseProgress(len(views), title, "views") as progress:
            for start in range(0, len(views), step):
                frames = slice(start, min(start + step, len(views)))
                images = views[frames]
                raw = heads.depth_head(token_list, images, frames, patch_start, positions)
                outputs["depth"].append(raw[:, 0].exp())
                outputs["depth_mask"].append(raw[:, 2])
                raw = heads.norm_head(token_list, images, frames, patch_start, positions)[:, :3]
                outputs["normals"].append(raw / raw.norm(dim=1, keepdim=True))
                if gaussians:
                    raw, merged = heads.gs_head(token_list, images, frames, patch_start, positions)
                    outputs["gs_depth"].append(raw[:, 0].exp())
                    outputs["gs_mask"].append(raw[:, 2])
                    outputs["gs_raw"].append(heads.gs_renderer.gs_head(merged))
                    del merged
                del raw
                progress.advance(len(images))
        return {name: torch.cat(values) if values else None for name, values in outputs.items()}

    def _preview(self, splats, camera_to_world, intrinsics, depth, height, width):
        """Thumbnail seen from the first input view (square, covering its whole field of view)."""
        fov = 2 * math.degrees(math.atan(max(width / (2 * float(intrinsics[0, 0, 0])), height / (2 * float(intrinsics[0, 1, 1])))))
        near = 0.01 * float(depth[0].median())
        projection = perspective(fov, 1.0, near, 1e4 * near, device=depth.device)
        view = torch.linalg.inv(geometry.opencv_to_gltf_cameras(camera_to_world[:1])[0])
        return splat_preview(splats, PREVIEW_SIZE, cameras=(view, projection))

    @generation_progress
    @torch.inference_mode()
    def generate(self, original_input_ref_images, video_guide, model_mode, custom_settings, callback=None, set_progress_status=None, **kwargs):
        """original_input_ref_images: the View Images (PIL); video_guide: the Reference Video file path or None."""
        settings = DEFAULT_CUSTOM_SETTINGS | custom_settings
        post = self.config["postprocess"]
        max_views, gaussians = int(settings["max_views"]), model_mode == "gaussians"
        chosen = [int(round(index)) for index in np.linspace(0, len(original_input_ref_images) - 1, min(max_views, len(original_input_ref_images)))]
        images = [original_input_ref_images[index] for index in chosen]
        sources = [f"image {index + 1}" for index in chosen]
        if video_guide is not None:
            set_phase_status("Selecting Video Frames")
            frames, indices = select_frames(video_guide, max_views - len(images))
            images += frames
            sources += [f"video frame {index}" for index in indices]
        set_phase_status("Preparing Views")
        views = prepare_views(images, min(int(settings["max_resolution"]), post["max_resolution"]), self.device)
        del images
        count, _, height, width = views.shape
        print(f"[WorldMirror] {count} views at {width}x{height}")

        with PhaseProgress(self.transformer.layer_count, "Reconstructing Scene", "layers") as progress:
            token_list = self.transformer(views, progress)
        set_phase_status("Estimating Cameras")
        codes = self.heads.cam_head(token_list[-1][None, :, 0], self.heads.camera_steps)[0]
        camera_to_world, intrinsics = geometry.camera_matrices(codes, height, width)
        dense = self._dense_predictions(token_list, views, gaussians)
        del token_list

        set_phase_status("Filtering Points")
        keep, gs_keep = geometry.filter_masks(dense["depth"], dense["normals"], dense["gs_depth"], dense["gs_mask" if gaussians else "depth_mask"],
                                              post["sky_threshold"], settings["filtering"] == "sky_edges", settings["filtering"] != "none",
                                              post["edge_normal_degrees"], post["edge_depth_tolerance"])
        keep &= dense["depth"] > 1e-8
        points = geometry.unproject(dense["depth"], camera_to_world, intrinsics)[keep]
        colors = (views.permute(0, 2, 3, 1) * 255).to(torch.uint8)[keep]
        points, colors = geometry.merge_points(points, colors, post["voxel_size"], post["max_points"])
        points = points * points.new_tensor(geometry.OPENCV_TO_GLTF)
        colors = colors.float() / 255
        if gaussians:
            set_phase_status("Merging Gaussians")
            attributes = geometry.gaussian_attributes(dense["gs_raw"], views, dense["gs_depth"], camera_to_world, intrinsics)
            dense["gs_raw"] = None
            gs_keep = gs_keep.reshape(-1)
            means, rotations, scales, sh, opacities = geometry.merge_gaussians(*(value[gs_keep] for value in attributes), post["voxel_size"],
                                                                                 post["max_gaussians"], post["scale_quantile"])
            del attributes
            splats = splats_opencv_to_gltf({"means": means, "rotations": rotations, "scales": scales, "opacities": opacities, "sh_dc": sh,
                                            "sh_rest": None, "sh_degree": 0})
        else:
            set_phase_status("Sizing Points")
            splats = points_to_splats(points, colors)
            # isolated points get huge nearest-neighbour scales: cap them like upstream's 98 % scale filter of the Gaussians
            splats["scales"] = splats["scales"].clamp_max(torch.quantile(splats["scales"][:, 0], post["scale_quantile"]))
        print(f"[WorldMirror] {len(splats['means'])} {'Gaussians' if gaussians else 'points'}, point cloud {len(points)} points")

        set_phase_status("Rendering Preview")
        preview = self._preview(splats, camera_to_world, intrinsics, dense["depth"], height, width)
        set_phase_status("Exporting Splats")
        ply = splats_to_ply(splats["means"], splats["scales"], splats["rotations"], splats["opacities"], splats["sh_dc"])
        cameras_gltf = geometry.opencv_to_gltf_cameras(camera_to_world)
        cameras = [{"view": index, "source": source, "width": width, "height": height, "camera_to_world": cameras_gltf[index].tolist(),
                    "intrinsics": intrinsics[index].tolist(), "fov_x_degrees": math.degrees(float(codes[index, 8])), "fov_y_degrees": math.degrees(float(codes[index, 7]))}
                   for index, source in enumerate(sources)]
        side_files = {"_points.ply": write_points_ply(points, colors), "_cameras.json": json.dumps(cameras, indent=1).encode("utf-8")}
        return {"x": preview[:, None], "splats": [ply], "side_files": side_files}
