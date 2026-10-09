# TRELLIS.2 image-to-3D pipeline, inference port of microsoft/TRELLIS.2 (trellis2/pipelines/trellis2_image_to_3d.py, MIT License, see
# LICENSE in this folder): sparse structure flow, shape flow (512, 1024 or a 512 -> 1024/1536 cascade), texture flow, then the
# shape and texture decoders, mesh extraction and a PBR GLB.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from mmgp import offload
from PIL import Image
from tqdm import tqdm

from shared.utils import files_locator as fl
from shared.utils.phase_progress import PhaseProgress, check_abort, generation_progress, set_phase_status, text_encoding_progress
from shared.utils.text_encoder_cache import TextEncoderCache
from . import sampler
from .dinov3 import DINOv3ViT, image_to_pixels
from .dit import TrellisFlowDiT, fused_split_map
from .export import SurfaceJob, bake_textures, export_glb, remesh_surface, render_preview, to_gltf_mesh, vertex_colors
from .fdg_mesh import decode_fdg, flexible_dual_grid_to_mesh
from .settings import DEFAULT_CUSTOM_SETTINGS
from .sparse import SparseTensor
from .vae import SparseStructureDecoder, SparseUnetDecoder

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "trellis2.json")
# checkpoint prefixes without digits, so that MMGP detects the 30 blocks of each flow (quantization, offloading)
SHAPE_STAGES = ("ss_flow", "shape_lowres", "shape_highres")
TEXTURE_STAGES = ("tex_lowres", "tex_highres")
SS_DECODER_FILE = "trellis2/trellis2_ss_decoder_fp16.safetensors"
SHAPE_DECODER_FILE = "trellis2/trellis2_shape_decoder_fp16.safetensors"
TEXTURE_DECODER_FILE = "trellis2/trellis2_texture_decoder_fp16.safetensors"
IMAGE_ENCODER_FILE = "dinov3_vitl16/dinov3_vitl16_lvd1689m.safetensors"  # Meta's weights, shared folder, DINOv3 License beside them
PIPELINES = {0: "512", 1: "1024", 2: "1024_cascade", 3: "1536_cascade"}
PREVIEW_SIZE = 512


def _keep_checkpoint_dtypes(model):
    """MMGP keeps the checkpoint dtypes (no FP32 conversion) and does not cast the FP32 tensors given to the model and to its top-level
    layers to the model dtype: upstream feeds FP32 latents, timesteps and features to the layers it computes in FP32."""
    model._convertWeightsFloatTo = None
    for module in [model, *model.children()]:
        if not isinstance(module, nn.ModuleList) and not hasattr(module, "_lock_dtype"):
            module._lock_dtype = None
    return model.eval().requires_grad_(False)


def _load_module(module, filename):
    """Checkpoint-native dtypes (default_dtype None), read-only mmap."""
    offload.load_model_data(module, filename, writable_tensors=False, default_dtype=None)
    return _keep_checkpoint_dtypes(module)


def preprocess_image(image, remove_background):
    """Upstream Trellis2ImageTo3DPipeline.preprocess_image: images larger than 1024 are downscaled, the alpha channel (or a background
    removal mask) crops a square around the object, and the object is premultiplied over black."""
    has_alpha = image.mode == "RGBA" and not np.all(np.array(image)[:, :, 3] == 255)
    scale = min(1, 1024 / max(image.size))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    output = image if has_alpha else remove_background(image.convert("RGB"))
    alpha = np.array(output)[:, :, 3]
    points = np.argwhere(alpha > 0.8 * 255)
    if len(points) == 0:
        raise ValueError("No object was found in the reference image: use an image of one object, or an RGBA image with its alpha mask.")
    bbox = np.min(points[:, 1]), np.min(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    output = np.array(output.crop((center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2))).astype(np.float32) / 255
    return Image.fromarray(((output[:, :, :3] * output[:, :, 3:4]) * 255).astype(np.uint8))


def _rembg(image):
    set_phase_status("Removing Background")
    from rembg import remove
    from shared.utils.utils import new_rembg_session

    return remove(image, session=new_rembg_session("u2net"))


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False):
        self.device = torch.device("cuda")
        with open(CONFIG_PATH, encoding="utf-8") as reader:
            self.config = json.load(reader)
        dit = self.config["dit"]
        self.flows = {}
        for filename, stages, submodel_no in ((model_filename[0], SHAPE_STAGES, 1), (model_filename[1], TEXTURE_STAGES, 2)):
            with init_empty_weights():
                container = nn.ModuleDict({name: TrellisFlowDiT(**dit, **self.config["stages"][name]) for name in stages})
            # BF16 checkpoint (INT8 linears compute in BF16); the fused q/k/v and k/v projections are split at load for the attention kit
            offload.load_model_data(container, filename, writable_tensors=False, default_dtype=None, fp32_dtype=torch.bfloat16, fused_split_map=fused_split_map(dit["model_channels"]))
            container.eval().requires_grad_(False)
            if save_quantized:
                from wgp import save_quantized_model
                save_quantized_model(container, model_type, filename, torch.bfloat16, None, submodel_no=submodel_no)
            for name, flow in container.items():
                self.flows[name] = _keep_checkpoint_dtypes(flow)
        self.transformer = self.flows["ss_flow"]
        with init_empty_weights():
            ss_decoder = SparseStructureDecoder(**self.config["ss_decoder"])
            shape_decoder = SparseUnetDecoder(**self.config["shape_decoder"])
            texture_decoder = SparseUnetDecoder(**self.config["texture_decoder"])
            image_encoder = DINOv3ViT(**self.config["image_encoder"])
        ss_decoder.lock_checkpoint_fp32_modules()
        self.ss_decoder = _load_module(ss_decoder, fl.locate_file(SS_DECODER_FILE))
        self.shape_decoder = _load_module(shape_decoder, fl.locate_file(SHAPE_DECODER_FILE))
        self.texture_decoder = _load_module(texture_decoder, fl.locate_file(TEXTURE_DECODER_FILE))
        self.image_encoder = _load_module(image_encoder, fl.locate_file(IMAGE_ENCODER_FILE))
        self.cache = TextEncoderCache()
        self._prepared = (None, None)  # (source image digest, preprocessed image): background removal of the last reference
        self._interrupt = False

    def pipe(self):
        return {"transformer": self.flows["ss_flow"], "transformer2": self.flows["shape_lowres"], "transformer3": self.flows["shape_highres"],
                "transformer4": self.flows["tex_lowres"], "transformer5": self.flows["tex_highres"], "ss_decoder": self.ss_decoder,
                "shape_decoder": self.shape_decoder, "texture_decoder": self.texture_decoder, "image_encoder": self.image_encoder}

    def _prepare_image(self, image):
        digest = hashlib.sha1(image.tobytes() + repr((image.mode, image.size)).encode()).hexdigest()
        if self._prepared[0] != digest:
            set_phase_status("Preparing Image")
            self._prepared = (digest, preprocess_image(image, _rembg))
            check_abort()
        return self._prepared[1]

    def _encode_image(self, image, sizes):
        """DINOv3 features (1, tokens, 1024) FP32 of the preprocessed image at each size, cached on the CPU by image content and size."""
        digest = hashlib.sha1(image.tobytes()).hexdigest()

        def encode(batch):
            with text_encoding_progress(self.image_encoder.layer, title="Encoding Image"):
                return [self.image_encoder(image_to_pixels(image, batch[0], self.device))]

        features = self.cache.encode(encode, list(sizes), device=self.device, cache_keys=[(digest, size) for size in sizes])
        return {size: feature.to(torch.bfloat16) for size, feature in zip(sizes, features)}

    def _denoise(self, flow, noise, cond, coords, params, title, callback, guidance=None, concat=None, dense=False):
        """Euler sampling of one flow over the tokens of one sample (dense structure grid or sparse voxels)."""
        steps = params["steps"]
        guidance = params["guidance_strength"] if guidance is None else guidance
        # upstream computes the dense grid phases once on the CPU (a model buffer), the sparse ones on the device
        phases = flow.phases(coords).to(self.device)
        channels = noise.shape[1]

        def model(x, t, contexts):
            tokens = x.view(1, channels, -1).transpose(1, 2).contiguous() if dense else x[None]
            if concat is not None:
                tokens = torch.cat([tokens, concat[None]], dim=-1)
            outputs = flow(tokens, t, contexts, phases)
            return [out.transpose(1, 2).reshape(noise.shape) if dense else out[0] for out in outputs]

        bar = tqdm(total=steps, desc=title)

        def on_step(step):
            bar.update(1)
            if callback is not None:
                callback(step, None, False, override_num_inference_steps=steps, progress_title=title)
            check_abort()

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=steps, progress_title=title)
        try:
            return sampler.sample(model, noise, cond, torch.zeros_like(cond), steps, params["rescale_t"], guidance, params["guidance_rescale"],
                                  params["guidance_interval"], self.config["sigma_min"], sampler.dense_std if dense else sampler.sparse_std, on_step)
        finally:
            bar.close()

    def _normalization(self, name):
        stats = self.config[name]
        return (torch.tensor(stats["mean"], dtype=torch.float32, device=self.device)[None],
                torch.tensor(stats["std"], dtype=torch.float32, device=self.device)[None])

    def _decode(self, decoder, x, title, subdivisions=None, coords_only=False):
        with PhaseProgress(decoder.block_count(coords_only), title, "blocks") as progress:
            return decoder([x], subdivisions, coords_only, progress=progress.advance)

    def _structure(self, cond, generator, steps, guidance, resolution, title, callback):
        flow = self.flows["ss_flow"]
        grid = flow.resolution
        noise = torch.randn(1, flow.in_channels, grid, grid, grid, generator=generator, device="cpu").to(self.device)
        axis = torch.arange(grid, device="cpu")
        coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        params = dict(self.config["samplers"]["structure"], steps=steps)
        latent = self._denoise(flow, noise, cond, coords, params, title, callback, guidance=guidance, dense=True)
        set_phase_status("Decoding Structure")
        occupied = self.ss_decoder(latent) > 0
        if resolution != occupied.shape[2]:
            ratio = occupied.shape[2] // resolution
            occupied = F.max_pool3d(occupied.float(), ratio, ratio, 0) > 0.5
        return torch.argwhere(occupied[0, 0]).int()

    def _shape(self, flow_name, cond, coords, generator, steps, guidance, title, callback):
        flow = self.flows[flow_name]
        noise = torch.randn(coords.shape[0], flow.in_channels, generator=generator, device="cpu").to(self.device)
        params = dict(self.config["samplers"]["shape"], steps=steps)
        slat = self._denoise(flow, noise, cond, coords, params, title, callback, guidance=guidance)
        mean, std = self._normalization("shape_normalization")
        return slat * std + mean

    @generation_progress
    @torch.inference_mode()
    def generate(self, seed, sampling_steps, guide_scale, alt_guide_scale, batch_size, model_mode, custom_settings, original_input_ref_images,
                 callback=None, set_progress_status=None, **kwargs):
        settings = DEFAULT_CUSTOM_SETTINGS | custom_settings
        pipeline = PIPELINES[int(model_mode)]
        max_faces, texture_size, max_tokens = int(settings["max_faces"]), int(settings["texture_size"]), int(settings["max_tokens"])
        texture_mode, keep_alpha = settings["texture_mode"], settings["alpha_mode"] == "blend"
        image = self._prepare_image(original_input_ref_images[0])
        conds = self._encode_image(image, (512,) if pipeline == "512" else (512, 1024))
        previews, meshes = [], []
        for sample_no in range(batch_size):
            suffix = f" {sample_no + 1}/{batch_size}" if batch_size > 1 else ""
            generator = torch.Generator(device="cpu").manual_seed(seed + sample_no)
            coords = self._structure(conds[512], generator, sampling_steps, guide_scale, 64 if pipeline == "1024" else 32, "Denoising Structure" + suffix, callback)
            if pipeline == "512":
                shape_flow, texture_flow, cond, resolution = "shape_lowres", "tex_lowres", conds[512], 512
                shape_slat = self._shape(shape_flow, cond, coords, generator, sampling_steps, guide_scale, "Denoising Shape" + suffix, callback)
            elif pipeline == "1024":
                shape_flow, texture_flow, cond, resolution = "shape_highres", "tex_highres", conds[1024], 1024
                shape_slat = self._shape(shape_flow, cond, coords, generator, sampling_steps, guide_scale, "Denoising Shape" + suffix, callback)
            else:
                texture_flow, cond = "tex_highres", conds[1024]
                coarse = self._shape("shape_lowres", conds[512], coords, generator, sampling_steps, guide_scale, "Denoising Coarse Shape" + suffix, callback)
                fine_coords = self._decode(self.shape_decoder, SparseTensor(coarse, coords), "Upsampling Shape", coords_only=True)
                del coarse
                resolution = 1536 if pipeline == "1536_cascade" else 1024
                while True:
                    coords = ((fine_coords + 0.5) / 512 * (resolution // 16)).int().unique(dim=0)
                    if coords.shape[0] < max_tokens or resolution == 1024:
                        break
                    resolution -= 128
                del fine_coords
                shape_slat = self._shape("shape_highres", cond, coords, generator, sampling_steps, guide_scale, "Denoising Shape" + suffix, callback)
            # the surface first: its decimation and UV unwrapping run on the CPU while the texture flow and decoder run on the GPU
            fields, subdivisions = self._decode(self.shape_decoder, SparseTensor(shape_slat, coords), "Decoding Shape" + suffix)
            set_phase_status("Extracting Surface")
            mesh = list(flexible_dual_grid_to_mesh(fields, *decode_fdg(fields.feats, self.config["shape_decoder"]["voxel_margin"]), resolution))
            del fields
            vertices, faces, offset = remesh_surface(mesh, resolution)
            surface = SurfaceJob(vertices, faces, max_faces, texture_size if texture_mode == "texture" else 0)
            del vertices, faces
            try:
                voxels = None
                if texture_mode != "none":
                    flow = self.flows[texture_flow]
                    mean, std = self._normalization("shape_normalization")
                    concat = (shape_slat - mean) / std
                    noise = torch.randn(coords.shape[0], flow.in_channels - concat.shape[1], generator=generator, device="cpu").to(self.device)
                    params = dict(self.config["samplers"]["texture"], steps=sampling_steps)
                    texture_slat = self._denoise(flow, noise, cond, coords, params, "Denoising Texture" + suffix, callback, guidance=alt_guide_scale, concat=concat)
                    del concat, noise
                    mean, std = self._normalization("texture_normalization")
                    voxels, _ = self._decode(self.texture_decoder, SparseTensor(texture_slat * std + mean, coords), "Decoding Texture" + suffix, subdivisions)
                    voxels.feats.mul_(0.5).add_(0.5)
                    del texture_slat
                del shape_slat, subdivisions
                set_phase_status("Simplifying Mesh")
                surface_mesh = surface.result()
            finally:
                surface.close()
            vertices, faces = surface_mesh.vertices.to(self.device), surface_mesh.faces.to(self.device)
            if voxels is None:
                mesh = to_gltf_mesh(vertices, faces)
            elif texture_mode == "vertex":
                colors = vertex_colors(vertices, faces, voxels, voxels.feats, resolution, offset)
                mesh = to_gltf_mesh(vertices, faces, colors=colors if keep_alpha else colors[:, :3])
            else:
                vertices, faces, uvs, base_color, metallic_roughness = bake_textures(surface_mesh, voxels, voxels.feats, resolution, texture_size, offset)
                # the official exporter stores the alpha attribute in the base color but writes an OPAQUE material
                mesh = to_gltf_mesh(vertices, faces, uvs, base_color if keep_alpha else base_color[..., :3].contiguous(), metallic_roughness)
            del voxels, vertices, faces, surface_mesh
            set_phase_status("Exporting Mesh")
            meshes.append(export_glb(mesh))
            previews.append(render_preview(mesh, PREVIEW_SIZE))
            del mesh
            check_abort()
        return {"x": torch.stack(previews, dim=1), "meshes": meshes}
