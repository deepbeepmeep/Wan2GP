# AnimateAnyMesh for WanGP: a static mesh and a motion prompt to an animated mesh (16 frames of vertex trajectories), from
# JarrentWu1031/AnimateAnyMesh test_drive.py and diffusion/rf_diffusion.py (Apache License 2.0, see LICENSE in this folder).
# Modified by DeepBeepMeep for WanGP: model loading through MMGP, the shared CLIP ViT-L/14 text tower, the WanGP generation contract
# (morph-animated GLB bytes and previews), mesh I/O through the shared mesh toolkit instead of Blender, the midpoint ODE solver
# written out instead of torchdiffeq, optional simplification of very dense meshes.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import os

import torch
from accelerate import init_empty_weights
from mmgp import offload
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from shared.mesh3d import Mesh, animated_mesh_to_glb, decimate, load_mesh, preview
from shared.utils import files_locator as fl
from shared.utils.phase_progress import check_abort, generation_progress, set_phase_status, text_encoding_progress
from shared.utils.text_encoder_cache import TextEncoderCache
from .dymesh_vae import DyMeshVAE, farthest_point_order
from .settings import DEFAULT_CUSTOM_SETTINGS
from .transformer import DyMeshMMDiT, fused_split_map

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "animate_any_mesh.json")
VAE_FILE = "animate_any_mesh/animate_any_mesh_vae_fp32.safetensors"
TEXT_ENCODER_FOLDER = "clip_vit_large_patch14"
PREVIEW_SIZE = 512
NEAREST = 4  # simplified-mesh vertices whose motion a vertex of a dense mesh follows
DISTANCE_CHUNK = 2048


def to_model_frame(points):
    """glTF axes (+Y up, +Z front) to the Z-up axes AnimateAnyMesh works in, those of upstream's Blender glTF import (x, -z, y)."""
    return torch.stack([points[..., 0], -points[..., 2], points[..., 1]], dim=-1)


def from_model_frame(points):
    return torch.stack([points[..., 0], points[..., 2], -points[..., 1]], dim=-1)


def _load(module, filename):
    """FP32 checkpoint, read-only mmap; MMGP keeps FP32 (no conversion of the float weights)."""
    offload.load_model_data(module, filename, writable_tensors=False, default_dtype=None)
    module._convertWeightsFloatTo = None
    return module.eval().requires_grad_(False)


def _welded(vertices, faces):
    """Upstream merge_identical_vertices_with_indices: unique positions in sorted order, the index of each vertex's position, and
    the faces on the unique positions without the degenerate ones (winding kept)."""
    welded, inverse = torch.unique(vertices, dim=0, return_inverse=True)
    faces = inverse[faces]
    return welded, inverse, faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]


def follow_motion(points, sources, trajectories):
    """Trajectories (frames, len(points), 3) of points that follow the trajectories (frames, len(sources), 3) of the nearby source
    points (rest positions): inverse-distance weighted displacements of the NEAREST closest sources."""
    displacements = trajectories - sources[None]
    moved = []
    for chunk in points.split(DISTANCE_CHUNK):
        distances, index = torch.cdist(chunk, sources).topk(NEAREST, dim=1, largest=False)
        weights = 1 / distances.clamp_min(1e-12)
        weights = weights / weights.sum(1, keepdim=True)
        moved.append(chunk[None] + (displacements[:, index] * weights[None, :, :, None]).sum(2))
    return torch.cat(moved, dim=1)


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, text_encoder_filename, save_quantized=False):
        self.device = torch.device("cuda")
        with open(CONFIG_PATH, encoding="utf-8") as reader:
            self.config = json.load(reader)
        width = self.config["transformer"]["width"]
        with init_empty_weights():
            transformer = DyMeshMMDiT(**self.config["transformer"])
            vae = DyMeshVAE(**self.config["vae"])
        offload.load_model_data(transformer, model_filename[0], writable_tensors=False, default_dtype=None, fused_split_map=fused_split_map(width))
        transformer._convertWeightsFloatTo = None
        self.transformer = transformer.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.transformer, model_type, model_filename[0], torch.float32, None)
        self.vae = _load(vae, fl.locate_file(VAE_FILE))
        self.text_encoder = offload.fast_load_transformers_model(text_encoder_filename, modelClass=CLIPTextModel, ignore_unused_weights=True, writable_tensors=False,
                                                                 forcedConfigPath=fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "text_config.json")), default_dtype=None)
        self.text_encoder._convertWeightsFloatTo = None
        self.text_encoder.eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(fl.locate_folder(TEXT_ENCODER_FOLDER))
        self.cache = TextEncoderCache()
        self._interrupt = False

    def pipe(self):
        return {"transformer": self.transformer, "vae": self.vae, "text_encoder": self.text_encoder}

    def _encode_prompt(self, prompt):
        """CLIP last hidden states (1, 77, 768) of the motion prompt, padded to 77 tokens without attention mask, as upstream."""
        def encode(prompts):
            ids = self.tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(self.device)
            with text_encoding_progress(self.text_encoder.text_model.encoder.layers, next_status="Encoding Mesh"):
                return [self.text_encoder(ids)[0][0]]

        return self.cache.encode(encode, [prompt], device=self.device)[0][None]

    def _prepare_mesh(self, mesh, max_vertices):
        """Model input of the mesh: welded vertices in the model frame (simplified past max_vertices), normalized as upstream into
        [-1, 1] around the bounding box center, with the faces, the denormalization and the map back to the mesh vertices."""
        welded, inverse, faces = _welded(to_model_frame(mesh.vertices.to(self.device).float()), mesh.faces.to(self.device))
        dense = None
        if len(welded) > max_vertices:
            set_phase_status("Simplifying Mesh")
            simplified = decimate(Mesh(welded, faces), 2 * max_vertices)
            dense, (vertices, _, faces) = welded, _welded(simplified.vertices, simplified.faces)
            check_abort()
        else:
            vertices = welded
        faces = torch.unique(faces.sort(dim=1).values, dim=0)  # upstream keeps each triangle once, its corners sorted: adjacency only
        center = (vertices.amax(0) + vertices.amin(0)) / 2
        vertices = vertices - center
        scale = vertices.abs().max() + 1e-8
        return vertices / scale, faces, (center, scale), inverse, dense

    def _sample(self, model, noise, f0, text_states, steps, guidance_scale, callback):
        """Upstream rf_sample: torchdiffeq odeint midpoint method on the time grid linspace(0, 1, steps + 1); the static-shape channels
        are reset to f0 before every evaluation and in the result."""
        channels = f0.shape[-1]
        times = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        flow = lambda t, x: model(torch.cat([f0, x[..., channels:]], dim=-1), t.view(1), text_states, guidance_scale)
        x = noise
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=steps, progress_title="Denoising Motion")
        for step in tqdm(range(steps), desc="Denoising Motion"):
            t0, dt = times[step], times[step + 1] - times[step]
            half_dt = 0.5 * dt
            x = x + dt * flow(t0 + half_dt, x + flow(t0, x) * half_dt)
            if callback is not None:
                callback(step, None, False, override_num_inference_steps=steps, progress_title="Denoising Motion")
            check_abort()
        return torch.cat([f0, x[..., channels:]], dim=-1)

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, input_custom, seed, sampling_steps, guide_scale, batch_size, custom_settings, callback=None, set_progress_status=None, **kwargs):
        settings = DEFAULT_CUSTOM_SETTINGS | custom_settings
        stats, frames = self.config["normalization"], self.config["vae"]["T"]
        set_phase_status("Loading Mesh")
        mesh = load_mesh(input_custom, device="cpu")
        text_states = self._encode_prompt(input_prompt)
        vertices, faces, (center, scale), inverse, dense = self._prepare_mesh(mesh, int(settings["max_vertices"]))
        count = len(vertices)
        trajectories = int(settings["trajectories"]) or max(self.config["min_trajectories"], count // self.config["trajectory_ratio"])
        set_phase_status("Encoding Mesh")
        points = torch.zeros((1, max(self.config["max_length"], count), 3), device=self.device)  # upstream zero padding to max_length
        points[0, :count] = vertices
        embedding = self.vae.embed_points(points, faces)
        order = farthest_point_order(embedding[0, :count], trajectories)
        check_abort()
        previews, meshes = [], []
        for sample_no in range(batch_size):
            # upstream draws, on the device, the farthest point sampling replacements then the noise (torch.manual_seed(seed))
            generator = torch.Generator(device=self.device).manual_seed(seed + sample_no)
            f0 = (self.vae.encode(embedding, order, count, generator) - stats["f0_mean"]) / stats["f0_std"]
            noise = torch.randn((1, trajectories, 2 * f0.shape[-1]), generator=generator, device=self.device)
            latent = self._sample(self.transformer, noise, f0, text_states, sampling_steps, guide_scale, callback)
            del noise, f0
            set_phase_status("Decoding Motion")
            latent = torch.cat([latent[..., :self.vae.latent_dim] * stats["f0_std"] + stats["f0_mean"], latent[..., self.vae.latent_dim:] * stats["ft_std"] + stats["ft_mean"]], dim=-1)
            motion = self.vae.decode(latent, embedding, points[:, :count])[0] * scale + center
            del latent
            if dense is not None:
                motion = follow_motion(dense, vertices * scale + center, motion)
            motion = from_model_frame(motion[:, inverse]).cpu()
            set_phase_status("Exporting Animated Mesh")
            fps = float(settings["fps"])
            clip = torch.cat([motion, motion.flip(0)[1:]]) if settings["playback"] == "pingpong" else motion
            meshes.append(animated_mesh_to_glb(mesh, clip, fps))
            previews.append(preview(mesh.replace(vertices=motion[frames // 2], normals=None).to(self.device), PREVIEW_SIZE).cpu())
            del motion, clip
            check_abort()
        return {"x": torch.stack(previews, dim=1), "meshes": meshes}
