# 3D reconstruction family: MoGe-2 (one photo), MapAnything and Depth Anything 3 (photos and/or video frames) predict per-pixel 3D points
# and cameras; the points are returned as small isotropic Gaussian splats (or, for MoGe-2, as a textured mesh of the point map), with the
# plain point cloud and the cameras as side files. Axes: glTF (+Y up, +Z towards the viewer), origin at the first camera.
import cv2
import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload

from shared.mesh3d import Mesh, mesh_to_glb, points_to_splats, render, splat_preview, splats_to_ply, submesh, write_points_ply
from shared.utils import files_locator as fl
from shared.utils.phase_progress import PhaseProgress, check_abort, generation_progress, set_phase_status
from shared.utils.utils import get_video_frame, get_video_info
from . import geometry, mapanything, moge2
from .dinov2 import QKV_SPLIT_MAP
from .settings import DEFAULT_CUSTOM_SETTINGS

PREVIEW_SIZE = 512
DA3_MODEL_NAME = "da3-large"
MOGE2_EDGE_LTOL = 0.01  # the MoGe demo's depth edge removal


def _load(module, filename, **kwargs):
    """Checkpoint-native dtypes (FP32), read-only mmap; MMGP keeps them (no conversion to the transformer dtype)."""
    offload.load_model_data(module, filename, writable_tensors=False, default_dtype=None, **kwargs)
    module._convertWeightsFloatTo = None
    return module.eval().requires_grad_(False)


def _bf16_block_linears(model):
    """The linears of the transformer blocks in BF16, as autocast runs them: the dtype in which their INT8 versions compute (applied to the
    placeholders before loading, an FP32 checkpoint keeps its weights)."""
    for blocks in model.transformer_blocks():
        for module in blocks.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(torch.bfloat16)


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False):
        self.device = torch.device("cuda")
        self.architecture = base_model_type
        if base_model_type == "moge2":
            with init_empty_weights():
                model = moge2.MoGe2.from_config()
            model = _load(model, model_filename[0], fused_split_map=QKV_SPLIT_MAP)
        elif base_model_type == "mapanything":
            with init_empty_weights():
                model = mapanything.MapAnything.from_config()
            model = _load(model, model_filename[0], fused_split_map=QKV_SPLIT_MAP, pre_load_callback=_bf16_block_linears)
        else:
            from preprocessing.depth_anything_v3.api import DepthAnything3
            from preprocessing.depth_anything_v3.depth import DA3_BF16_MODEL

            # WanGP's depth checkpoint (DA3-LARGE-1.1, BF16), the model in BF16 as in preprocessing/depth_anything_v3/depth.py; built
            # initialized because the LayerNorms of the unused auxiliary ray heads are absent from every DA3 release
            model = _load(DepthAnything3(model_name=DA3_MODEL_NAME).to(torch.bfloat16), fl.locate_file(DA3_BF16_MODEL), ignore_missing_keys=True)
        if save_quantized and base_model_type != "da3_points":  # da3_points reuses the depth preprocessor file: no transformer to quantize
            from wgp import save_quantized_model
            _bf16_block_linears(model)
            save_quantized_model(model, model_type, model_filename[0], torch.bfloat16, None)
        if base_model_type != "da3_points":
            for module in model.modules():  # checkpoint-native mixed dtypes (FP32; INT8 blocks with BF16 biases): MMGP keeps each dtype
                module._lock_dtype = None
        self.transformer = model
        self._interrupt = False

    def _views(self, images, video, max_frames):
        """RGB PIL views: the photos, then max_frames frames evenly spaced over the video; and their names for the cameras."""
        views, sources = [image.convert("RGB") for image in images], [f"photo {index + 1}" for index in range(len(images))]
        if video is not None:
            set_phase_status("Loading Video Frames")
            frame_count = get_video_info(video)[3]
            for frame in np.unique(np.linspace(0, frame_count - 1, min(max_frames, frame_count)).round().astype(int)).tolist():
                views.append(get_video_frame(video, frame, return_last_if_missing=True).convert("RGB"))
                sources.append(f"frame {frame}")
                check_abort()
        if not views:
            raise ValueError("Add at least one photo or a video to reconstruct.")
        return views, sources

    def _moge2(self, image, settings):
        """Camera-space points of one photo (1, H, W, 3), RGB, mask, normals and intrinsics (MoGe demo pipeline)."""
        pixels = np.array(image)
        larger = max(pixels.shape[:2])
        if larger > settings["max_size"]:
            scale = settings["max_size"] / larger
            pixels = cv2.resize(pixels, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        colors = torch.from_numpy(pixels).to(self.device).float().div_(255)
        low, high = self.transformer.config["num_tokens_range"]
        num_tokens = int(low + settings["detail"] / 9 * (high - low))
        with PhaseProgress(self.transformer.block_count(), "Reconstructing Geometry", "blocks") as progress:
            output = self.transformer(colors.permute(2, 0, 1)[None].contiguous(), num_tokens, progress.advance)
        points, depth, normals, mask, intrinsics = moge2.point_map(output, colors.shape[:2])
        if settings["edges"] == "remove":
            mask &= ~geometry.disparity_edges(depth, mask, MOGE2_EDGE_LTOL)
        return points[None], colors[None], mask[None], normals, intrinsics[None], torch.eye(4, device=self.device)[None]

    def _mapanything(self, images, settings):
        set_phase_status("Preprocessing Images")
        pixels, colors = mapanything.preprocess(images, self.transformer.config["resolution_set"])
        with PhaseProgress(self.transformer.block_count(len(images)), "Reconstructing Geometry", "blocks") as progress:
            output = self.transformer(pixels.to(self.device), progress.advance)
        points, poses, intrinsics, mask = mapanything.reconstruct(output, settings["confidence"])
        return points, colors.to(self.device), mask, intrinsics, poses

    def _da3(self, images, settings):
        """WanGP's Depth Anything 3 (preprocessing/depth_anything_v3) on the views: depth, confidence and cameras back-projected as its
        GLB export does (integer pixel centres, adaptive confidence threshold)."""
        model = self.transformer
        set_phase_status("Preprocessing Images")
        pixels, _, _ = model.input_processor(images, None, None, settings["process_res"], "upper_bound_resize")
        pixels = pixels.to(self.device)[None]
        blocks = model.model.backbone.pretrained.blocks
        with PhaseProgress(len(blocks), "Reconstructing Geometry", "blocks") as progress:
            handles = [block.register_forward_hook(lambda *_: progress.advance()) for block in blocks]
            try:
                output = model(pixels, export_feat_layers=[])
            finally:
                for handle in handles:
                    handle.remove()
        depth, confidence, intrinsics = output.depth[0].float(), output.depth_conf[0].float(), output.intrinsics[0].float()
        world_to_camera = torch.eye(4, device=self.device).repeat(len(depth), 1, 1)
        world_to_camera[:, :3] = output.extrinsics[0, :, :3].float()
        poses = torch.linalg.inv(world_to_camera)  # exact inverse, as the DA3 exporter (the BF16 rotations are orthonormal to about 1e-2)
        views, height, width = depth.shape
        rows, cols = torch.meshgrid(torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing="ij")
        pixel = torch.stack([cols, rows, torch.ones_like(rows)], dim=-1).float()
        rays = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(intrinsics), pixel)
        points = torch.einsum("vij,vhwj->vhwi", poses[:, :3, :3], rays * depth[..., None]) + poses[:, None, None, :3, 3]
        mask = torch.isfinite(depth) & (depth > 0)
        if settings["confidence"] > 0:
            values = confidence.cpu().numpy()
            threshold = min(max(1.05, np.percentile(values, settings["confidence"])), np.percentile(values, 90))
            mask &= confidence >= threshold
        mean, std = (torch.tensor(v, device=self.device).view(1, 3, 1, 1) for v in ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        colors = (pixels[0] * std + mean).clamp_(0, 1).permute(0, 2, 3, 1)
        return points, colors, mask, intrinsics, poses

    def _splats(self, points, colors, mask, cameras, max_points, seed, preview_camera=None):
        set_phase_status("Filtering Points")
        points, colors = points[mask], colors[mask]
        if len(points) < 2:
            raise ValueError("No 3D points could be reconstructed from these inputs.")
        keep = geometry.subsample(len(points), max_points, seed, points.device)
        if keep is not None:
            points, colors = points[keep], colors[keep]
        points = geometry.opencv_to_gltf(points)
        set_phase_status("Creating Splats")
        splats = points_to_splats(points, colors)
        check_abort()
        set_phase_status("Rendering Preview")
        image = splat_preview(splats, PREVIEW_SIZE, cameras=None if preview_camera is None else preview_camera(points))
        set_phase_status("Exporting Splats")
        ply = splats_to_ply(splats["means"], splats["scales"], splats["rotations"], splats["opacities"], splats["sh_dc"])
        return {"x": image[:, None], "splats": [ply], "side_files": {"_points.ply": write_points_ply(points, colors), "_cameras.json": cameras}}

    def _mesh(self, points, colors, mask, normals, cameras, preview_camera):
        """MoGe-2 mesh: the point map triangulated over its valid pixels, textured with the photo."""
        set_phase_status("Building Mesh")
        height, width = mask.shape
        faces = geometry.grid_faces(mask)
        if len(faces) == 0:
            raise ValueError("No surface could be reconstructed from this photo.")
        rows, cols = torch.meshgrid(torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing="ij")
        uvs = torch.stack([(cols + 0.5) / width, (rows + 0.5) / height], dim=-1).reshape(-1, 2)
        mesh = Mesh(vertices=geometry.opencv_to_gltf(points.reshape(-1, 3)), faces=faces, normals=geometry.opencv_to_gltf(normals.reshape(-1, 3)),
                    uvs=uvs, base_color=colors.mul(255).round_().to(torch.uint8))
        mesh = submesh(mesh, torch.ones(len(faces), dtype=torch.bool, device=self.device))
        set_phase_status("Rendering Preview")
        image = render(mesh, preview_camera(mesh.vertices), PREVIEW_SIZE, "shaded", supersample=2)[0].permute(2, 0, 1) * 2 - 1
        set_phase_status("Exporting Mesh")
        return {"x": image[:, None], "meshes": [mesh_to_glb(mesh, image_format="jpeg")], "side_files": {"_cameras.json": cameras}}

    @generation_progress
    @torch.inference_mode()
    def generate(self, seed, model_mode, custom_settings, original_input_ref_images, video_guide=None, callback=None, set_progress_status=None, **kwargs):
        settings = DEFAULT_CUSTOM_SETTINGS[self.architecture] | custom_settings
        views, sources = self._views(original_input_ref_images, video_guide, settings.get("max_frames"))
        if self.architecture == "moge2":
            points, colors, mask, normals, intrinsics, poses = self._moge2(views[0], settings)
        else:
            points, colors, mask, intrinsics, poses = (self._mapanything if self.architecture == "mapanything" else self._da3)(views, settings)
            anchor = torch.linalg.inv(poses[0])  # world = frame of the first camera
            points = points @ anchor[:3, :3].T + anchor[:3, 3]
            poses = anchor @ poses
        check_abort()
        cameras = geometry.cameras_json(intrinsics, poses, tuple(mask.shape[1:]), sources)
        preview_camera = None  # multi-view scenes: the canonical 3/4 view of the toolkit; one photo: its own camera turned about the scene
        if self.architecture == "moge2":
            preview_camera = lambda gltf_points: geometry.photo_view(gltf_points, intrinsics[0], tuple(mask.shape[1:]))
            if model_mode == "mesh":
                return self._mesh(points[0], colors[0], mask[0], normals, cameras, preview_camera)
        return self._splats(points, colors, mask, cameras, int(settings["max_points"]), seed, preview_camera)
