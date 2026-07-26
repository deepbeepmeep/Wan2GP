from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from postprocessing.prismaudio import PRISMAUDIO_LATENT_DOWNSAMPLING, PRISMAUDIO_SAMPLE_RATE
from shared.utils import offload_registry


_CLIP_FPS = 4
_VIDEOPRISM_MIN_FRAMES = 8
_VIDEOPRISM_MAX_FRAMES = 36
_CLIP_SIZE = 288
_SYNC_FPS = 25
_SYNC_SIZE = 224
_DEFAULT_CAPTION = "Generate a realistic soundtrack synchronized with the visible video. Emphasize foreground sounds from visible motion, environmental ambience, impacts, and spatial movement. Avoid unrelated speech or music unless clearly present in the video."
_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
_PRISMAUDIO_DIR = os.path.join(_VENDOR_DIR, "PrismAudio")
_MODEL_CONFIG_PATH = os.path.join(_PRISMAUDIO_DIR, "configs", "model_configs", "prismaudio.json")


def _install_vendor_path() -> None:
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


_REQUIRED_MODULES = {
    "einshape": "einshape",
    "jax": "jax",
    "flax": "flax",
    "sentencepiece": "sentencepiece",
}


@dataclass(frozen=True)
class PrismAudioPaths:
    model: str
    vae: str
    synchformer: str
    t5_model_dir: str
    videoprism: str
    videoprism_tokenizer: str


def requirement_message() -> str | None:
    _install_vendor_path()
    missing = [package for module, package in _REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None]
    if missing:
        return "PrismAudio Python dependencies are missing. Run pip install -r requirements.txt to install: " + ", ".join(missing)
    return None


def _abort_requested(abort_callback) -> bool:
    return callable(abort_callback) and abort_callback()


def _report_progress(progress_callback, phase: str, current_step: int | None = None, total_steps: int | None = None) -> None:
    if callable(progress_callback):
        progress_callback(phase, current_step, total_steps)


def _check_abort(abort_callback) -> bool:
    return _abort_requested(abort_callback)


def _autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.amp.autocast(device.type, dtype=dtype) if device.type == "cuda" else nullcontext()


def _caption_text(prompt: str) -> str:
    return str(prompt or "").strip() or _DEFAULT_CAPTION


def _pad_to_square(video_tensor: torch.Tensor) -> torch.Tensor:
    video_tensor = video_tensor.to(torch.float32).div_(255.0)
    _, _, height, width = video_tensor.shape
    max_side = max(height, width)
    pad_h = max_side - height
    pad_w = max_side - width
    padded = F.pad(video_tensor, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode="constant", value=0)
    return F.interpolate(padded, size=(_CLIP_SIZE, _CLIP_SIZE), mode="bilinear", align_corners=False).clamp_(0, 1)


def _sample_discrete_euler(model, x: torch.Tensor, steps: int, sigma_max: float = 1.0, *, abort_callback=None, progress_callback=None, **extra_args) -> torch.Tensor | None:
    t = torch.linspace(float(sigma_max), 0, int(steps) + 1, device=x.device, dtype=x.dtype)
    _report_progress(progress_callback, "Denoising", 0, int(steps))
    for step_no, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), total=int(steps)), start=1):
        if _abort_requested(abort_callback):
            return None
        t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
        x = x + (t_prev - t_curr) * model(x, t_curr_tensor, **extra_args)
        _report_progress(progress_callback, "Denoising", step_no, int(steps))
    return x


class PrismAudioRuntime:
    def __init__(self) -> None:
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths: PrismAudioPaths | None = None
        self.profile = None
        self.feature_extractor = None
        self.diffusion = None
        self.model_config = None
        self.sync_transform = None
        self.tokenizer = None
        self.offloadobj = None

    def load(self, paths: PrismAudioPaths, *, init_pipe, profile, progress_callback=None, verbose_level: int = 1) -> None:
        if self.diffusion is not None and self.feature_extractor is not None and self.paths == paths and self.profile == profile:
            return
        missing = requirement_message()
        if missing:
            raise ImportError(missing)
        self.release()
        _report_progress(progress_callback, "PrismAudio Loading Models")
        _install_vendor_path()

        from mmgp import offload
        from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils
        from PrismAudio.models import create_model_from_config
        from torchvision.transforms import v2

        with open(_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
            self.model_config = json.load(f)

        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.feature_extractor = FeaturesUtils(
            vae_ckpt=None,
            vae_config=None,
            enable_conditions=True,
            synchformer_ckpt=None,
            t5_model_path=paths.t5_model_dir,
            videoprism_ckpt_path=paths.videoprism,
            videoprism_tokenizer_path=paths.videoprism_tokenizer,
        ).eval().requires_grad_(False)
        offload.load_model_data(self.feature_extractor.synchformer, paths.synchformer, writable_tensors=False, default_dtype=self.dtype, ignore_unused_weights=True, verboseLevel=-1)
        self.feature_extractor.synchformer.eval().requires_grad_(False)

        self.diffusion = create_model_from_config(self.model_config).eval().requires_grad_(False)
        offload.load_model_data(self.diffusion, paths.model, writable_tensors=False, default_dtype=self.dtype, verboseLevel=-1, ignore_missing_keys=True)
        offload.load_model_data(self.diffusion.pretransform, paths.vae, modelPrefix="autoencoder", writable_tensors=False, default_dtype=self.dtype, verboseLevel=-1)
        self.diffusion.pretransform.eval().requires_grad_(False)

        pipe = {
            "prismaudio_t5": self.feature_extractor.t5,
            "prismaudio_synchformer": self.feature_extractor.synchformer,
            "prismaudio_conditioner": self.diffusion.conditioner,
            "transformer": self.diffusion.model,
            "prismaudio_vae": self.diffusion.pretransform,
        }
        kwargs: dict[str, Any] = {}
        profile_no = init_pipe(pipe, kwargs, profile) if callable(init_pipe) else int(profile)
        self.offloadobj = offload.profile(pipe, profile_no=profile_no, quantizeTransformer=False, convertWeightsFloatTo=self.dtype, verboseLevel=verbose_level, **kwargs)
        offload_registry.register_offloadobj("PrismAudio", self.offloadobj, self.release)
        self.paths = paths
        self.profile = profile

    def extract_video_frames(self, video_path: str, duration: float | None, *, abort_callback=None, progress_callback=None):
        if _check_abort(abort_callback):
            return None
        _report_progress(progress_callback, "Extracting Video Frames")
        from shared.utils.video_decode import decode_video_frame_indices_ffmpeg, decode_video_frames_ffmpeg, probe_video_stream_metadata

        metadata = probe_video_stream_metadata(video_path)
        if metadata is None:
            raise RuntimeError(f"Unable to probe video metadata for PrismAudio: {video_path}")
        if duration is None or duration <= 0:
            duration = float(metadata.get("duration") or 0.0)
            if duration <= 0:
                fps = float(metadata.get("fps_float") or metadata.get("fps") or 0.0)
                frames_count = int(metadata.get("frame_count") or 0)
                duration = frames_count / fps
        clip_expected = max(_VIDEOPRISM_MIN_FRAMES, min(_VIDEOPRISM_MAX_FRAMES, int(round(_CLIP_FPS * float(duration)))))
        sync_expected = max(16, int(_SYNC_FPS * float(duration)))
        frame_count = int(metadata.get("frame_count") or 0)
        if frame_count > 0:
            clip_indices = np.linspace(0, frame_count - 1, clip_expected).round().astype(np.int64).tolist()
            clip_chunk = decode_video_frame_indices_ffmpeg(video_path, clip_indices, bridge="torch")
        else:
            clip_chunk = decode_video_frames_ffmpeg(video_path, 0, clip_expected, target_fps=_CLIP_FPS, bridge="torch")
        sync_chunk = decode_video_frames_ffmpeg(video_path, 0, sync_expected, target_fps=_SYNC_FPS, bridge="torch")
        if clip_chunk.shape[0] == 0:
            raise RuntimeError("PrismAudio clip video stream returned no frames")
        if sync_chunk.shape[0] == 0:
            raise RuntimeError("PrismAudio sync video stream returned no frames")
        clip_chunk = clip_chunk.permute(0, 3, 1, 2).contiguous()
        sync_chunk = sync_chunk.permute(0, 3, 1, 2).contiguous()
        clip_chunk = clip_chunk[:clip_expected]
        if clip_chunk.shape[0] < clip_expected:
            clip_chunk = torch.cat([clip_chunk, clip_chunk[-1:].repeat(clip_expected - clip_chunk.shape[0], 1, 1, 1)], dim=0)
        clip_chunk = _pad_to_square(clip_chunk).permute(0, 2, 3, 1).cpu().numpy()

        sync_chunk = sync_chunk[:sync_expected]
        if sync_chunk.shape[0] < sync_expected:
            sync_chunk = torch.cat([sync_chunk, sync_chunk[-1:].repeat(sync_expected - sync_chunk.shape[0], 1, 1, 1)], dim=0)
        sync_chunk = self.sync_transform(sync_chunk)
        return clip_chunk, sync_chunk, float(duration)

    @torch.inference_mode()
    def extract_features(self, clip_chunk, sync_chunk: torch.Tensor, caption: str, *, abort_callback=None, progress_callback=None) -> dict[str, torch.Tensor] | None:
        if _check_abort(abort_callback):
            return None
        _report_progress(progress_callback, "Extracting Text Features")
        inputs = {key: value.to(self.device) for key, value in self.feature_extractor.t5tokenizer([caption], padding=True, truncation=False, return_tensors="pt").items()}
        inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
        with _autocast_context(self.device, self.dtype):
            text_features = self.feature_extractor.t5(**inputs).last_hidden_state[0].cpu()
        if _check_abort(abort_callback):
            return None

        _report_progress(progress_callback, "Extracting VideoPrism Features")
        clip_input = torch.from_numpy(clip_chunk).unsqueeze(0)
        video_feat, frame_embed, _, text_feat = self.feature_extractor.encode_video_and_text_with_videoprism(clip_input, [caption])
        if _check_abort(abort_callback):
            return None

        _report_progress(progress_callback, "Extracting Sync Features")
        sync_input = sync_chunk.unsqueeze(0).to(self.device)
        with _autocast_context(self.device, self.dtype):
            sync_features = self.feature_extractor.encode_video_with_sync(sync_input)[0].cpu()
        return {
            "text_features": text_features,
            "global_video_features": torch.tensor(np.array(video_feat)).squeeze(0).cpu(),
            "video_features": torch.tensor(np.array(frame_embed)).squeeze(0).cpu(),
            "global_text_features": torch.tensor(np.array(text_feat)).squeeze(0).cpu(),
            "sync_features": sync_features,
        }

    @torch.inference_mode()
    def run_diffusion(self, info: dict[str, torch.Tensor], duration: float, caption: str, *, seed: int, steps: int, cfg_scale: float, abort_callback=None, progress_callback=None) -> torch.Tensor | None:
        if _check_abort(abort_callback):
            return None
        _report_progress(progress_callback, "Preparing Diffusion")
        latent_length = max(1, round(PRISMAUDIO_SAMPLE_RATE * float(duration) / PRISMAUDIO_LATENT_DOWNSAMPLING))
        meta = dict(info)
        meta["id"] = "demo"
        meta["relpath"] = "demo.npz"
        meta["path"] = "demo.npz"
        meta["caption_cot"] = caption
        meta["video_exist"] = torch.tensor(True)
        meta_on_device = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in meta.items()}
        metadata = (meta_on_device,)

        with _autocast_context(self.device, self.dtype):
            conditioning = self.diffusion.conditioner(metadata, self.device)
        cond_inputs = self.diffusion.get_conditioning_inputs(conditioning)
        if cond_inputs["sync_cond"].shape[1] != latent_length:
            sync_cond = cond_inputs["sync_cond"].transpose(1, 2)
            cond_inputs["sync_cond"] = F.interpolate(sync_cond, size=latent_length, mode="linear", align_corners=False).transpose(1, 2)

        generator = None
        seed = int(seed)
        if seed >= 0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        noise = torch.randn([1, self.diffusion.io_channels, latent_length], device=self.device, dtype=self.dtype, generator=generator)
        with _autocast_context(self.device, self.dtype):
            sampled = _sample_discrete_euler(self.diffusion.model, noise, int(steps), **cond_inputs, cfg_scale=float(cfg_scale), batch_cfg=True, abort_callback=abort_callback, progress_callback=progress_callback)
            if sampled is None:
                return None
            if self.diffusion.pretransform is not None:
                _report_progress(progress_callback, "Decoding Audio")
                sampled = self.diffusion.pretransform.decode(sampled)
        audio = sampled.float()
        peak = torch.max(torch.abs(audio)).clamp_min(1e-8)
        return audio.div(peak).clamp(-1, 1).detach().cpu()

    def generate(self, video_path: str, output_path: str, *, prompt: str, seed: int, duration: float | None, steps: int, cfg_scale: float, abort_callback=None, progress_callback=None) -> str | None:
        caption = _caption_text(prompt)
        _report_progress(progress_callback, "Preparing Video")
        frame_data = self.extract_video_frames(video_path, duration, abort_callback=abort_callback, progress_callback=progress_callback)
        if frame_data is None:
            return None
        clip_chunk, sync_chunk, resolved_duration = frame_data
        info = self.extract_features(clip_chunk, sync_chunk, caption, abort_callback=abort_callback, progress_callback=progress_callback)
        if info is None:
            return None
        audio = self.run_diffusion(info, resolved_duration, caption, seed=seed, steps=steps, cfg_scale=cfg_scale, abort_callback=abort_callback, progress_callback=progress_callback)
        if audio is None:
            return None
        _report_progress(progress_callback, "Saving Audio")
        from shared.utils.audio_video import write_wav_file

        write_wav_file(output_path, audio[0], PRISMAUDIO_SAMPLE_RATE)
        return output_path

    def release(self) -> None:
        if self.offloadobj is not None:
            offload_registry.unregister_offloadobj("PrismAudio", self.offloadobj)
            self.offloadobj.release()
            self.offloadobj = None
        self.paths = None
        self.profile = None
        self.feature_extractor = None
        self.diffusion = None
        self.model_config = None
        self.sync_transform = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_RUNTIME = PrismAudioRuntime()


def generate_audio(
    paths: PrismAudioPaths,
    video_path: str,
    output_path: str | None,
    *,
    prompt: str = "",
    seed: int = -1,
    duration: float | None = None,
    steps: int = 24,
    cfg_scale: float = 5.0,
    persistent_models: bool = False,
    init_pipe=None,
    profile=4,
    abort_callback=None,
    progress_callback=None,
    verbose_level: int = 1,
) -> str | None:
    if output_path is None:
        handle = tempfile.NamedTemporaryFile(prefix="prismaudio_", suffix=".wav", delete=False)
        output_path = handle.name
        handle.close()
    try:
        _RUNTIME.load(paths, init_pipe=init_pipe, profile=profile, progress_callback=progress_callback, verbose_level=verbose_level)
        return _RUNTIME.generate(video_path, output_path, prompt=prompt, seed=seed, duration=duration, steps=steps, cfg_scale=cfg_scale, abort_callback=abort_callback, progress_callback=progress_callback)
    finally:
        if persistent_models:
            offload_registry.unload_vram(["PrismAudio"])
        else:
            release_models()


def release_models() -> None:
    _RUNTIME.release()
