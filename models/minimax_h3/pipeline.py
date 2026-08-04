"""WanGP inference pipeline for MiniMax H3."""

import functools
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as audio_F
from PIL import Image
from tqdm import tqdm

from mmgp import offload
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.frame_scheduler import normalize_frame_count
from .interrupt import GenerationInterrupted
from .spectrum import MiniMaxH3Spectrum
from .transformer import VISUAL_COND_TIMESTEP, pack_audio, patchify_video, unpack_audio


FPS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_LATENT_FPS = 40


def _return_none_on_interrupt(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except GenerationInterrupted:
            return None

    return wrapped


def video_latent_frames(frame_count):
    frame_count = normalize_frame_count(max(5, int(frame_count)), 5, 17, 5)
    return 2 + ((frame_count - 5) // 17) * 5


def _as_video(tensor):
    if tensor is None:
        return None
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a CTHW tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _qwen_frames(video):
    return video.permute(1, 2, 3, 0).add(1.0).mul_(0.5).clamp_(0.0, 1.0)


def _fit_audio_samples(audio, sample_count):
    return F.pad(audio[..., :sample_count], (0, max(0, sample_count - audio.shape[-1])))


def _resolve_canvas(width, height, short_edge, max_pixels=None):
    ratio = width / height
    if not 0.25 <= ratio <= 4.0:
        raise ValueError(f"MiniMax H3 references must have an aspect ratio from 1:4 to 4:1, got {width}x{height}")
    if ratio >= 1.0:
        target_w, target_h = short_edge * ratio, float(short_edge)
    else:
        target_w, target_h = float(short_edge), short_edge / ratio
    if max_pixels is not None and target_w * target_h > max_pixels:
        scale = math.sqrt(max_pixels / (target_w * target_h))
        target_w, target_h = target_w * scale, target_h * scale
    return max(32, round(target_h / 32) * 32), max(32, round(target_w / 32) * 32)


def _to_pil(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if torch.is_tensor(frame):
        if frame.ndim == 4:
            frame = frame[:, 0]
        array = frame.permute(1, 2, 0).add(1.0).mul_(127.5).round_().clamp_(0, 255).byte().cpu().numpy()
        return Image.fromarray(array)
    return Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")


def _pil_to_video(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(1)


def _prepare_reference_video(source, max_frames):
    if isinstance(source, str):
        from shared.utils.utils import get_resampled_video_transparent

        frames = get_resampled_video_transparent(source, 0, max_frames, FPS, bridge="torch")
        if not torch.is_tensor(frames):
            frames = torch.from_numpy(frames)
        video = frames.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
    else:
        video = _as_video(source)
    video = video[:, :max_frames]
    target_h, target_w = _resolve_canvas(video.shape[-1], video.shape[-2], 768, 768 * 1344)
    if video.shape[-2:] != (target_h, target_w):
        frames = []
        for index in range(video.shape[1]):
            frame = _to_pil(video[:, index:index + 1]).resize((target_w, target_h), Image.Resampling.LANCZOS)
            frames.append(_pil_to_video(frame))
        video = torch.cat(frames, dim=1)
    return video


class MiniMaxH3Pipeline:
    def __init__(self, transformer, text_encoder, video_vae, audio_vae, reference_mode=False, dtype=torch.bfloat16):
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.vae = video_vae
        self.video_encoder = torch.nn.ModuleDict({"encoder": video_vae.encoder, "quant_conv": video_vae.quant_conv})
        self.video_decoder = torch.nn.ModuleDict({"post_quant_conv": video_vae.post_quant_conv, "decoder": video_vae.decoder})
        if torch.cuda.is_available() and torch.cuda.get_device_properties(None).total_memory >= 10 * 1024**3:
            self.video_decoder._budget = 0
        self.audio_vae = audio_vae
        self.reference_mode = bool(reference_mode)
        self.dtype = dtype
        self.text_encoder_cache = TextEncoderCache()
        self._interrupt = False

    @property
    def _interrupt(self):
        return getattr(self, "_abort", False)

    @_interrupt.setter
    def _interrupt(self, value):
        self._abort = bool(value)
        for name in ("transformer", "text_encoder", "vae", "audio_vae"):
            if hasattr(self, name):
                getattr(self, name)._interrupt = self._abort

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else next(self.transformer.parameters()).device)

    def get_trans_lora(self):
        return self.transformer, None

    def _check_abort(self):
        if self._interrupt:
            raise GenerationInterrupted

    def _set_interrupt_state(self):
        self.transformer._interrupt = self._interrupt
        self.text_encoder._interrupt = self._interrupt
        self.vae._interrupt = self._interrupt
        self.audio_vae._interrupt = self._interrupt

    @staticmethod
    def _update_tensor_digest(digest, tensor):
        tensor = tensor.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        tensor = tensor.contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(memoryview(tensor.view(torch.uint8).reshape(-1).numpy()))

    def _prompt_cache_key(self, prompt, presentation):
        digest = hashlib.sha256()
        for item in presentation:
            digest.update(item["type"].encode())
            if "frames" in item:
                self._update_tensor_digest(digest, item["frames"])
            if "timestamps" in item:
                digest.update(repr(tuple(item["timestamps"])).encode())
        return "minimax_h3", str(self.dtype), prompt, digest.digest()

    def _encode_prompt(self, prompt, presentation):
        def encode_fn(prompts):
            return [self.text_encoder.encode(prompts[0], presentation, self.device, self.dtype)]

        cache_key = self._prompt_cache_key(prompt, presentation)
        return self.text_encoder_cache.encode(encode_fn, prompt, device=self.device, cache_keys=cache_key)[0]

    def _configure_tiling(self, _tile_size):
        self.vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256)

    def _encode_video(self, video):
        self._check_abort()
        return self.vae.encode_condition(video.unsqueeze(0).to(device=self.device, dtype=self.vae._model_dtype)).cpu()

    def _waveform(self, waveform, sample_rate):
        if waveform is None:
            return None
        audio = torch.as_tensor(waveform, dtype=torch.float32)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.transpose(0, 1)
        else:
            raise ValueError(f"Expected mono or sample-major audio, got shape {tuple(audio.shape)}")
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] != 2:
            audio = audio[:2]
        sample_rate = int(sample_rate or AUDIO_SAMPLE_RATE)
        if sample_rate != AUDIO_SAMPLE_RATE:
            audio = audio_F.resample(audio, sample_rate, AUDIO_SAMPLE_RATE)
        return audio.unsqueeze(0)

    def _load_audio_reference(self, path):
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return self._waveform(audio, sample_rate)

    def _encode_audio(self, waveform):
        self._check_abort()
        return self.audio_vae.encode(waveform.to(device=self.device, dtype=torch.float32)).cpu()

    def _add_image_condition(self, image, frame_index, presentation, visual_latents, keyframes):
        video = _as_video(image)
        if video is None:
            return
        video = video[:, :1]
        presentation.append({"type": "image", "frames": _qwen_frames(video.clone())})
        visual_latents.append(self._encode_video(video))
        keyframes.append({"resolved_frame_index": frame_index})

    def _add_image_reference(self, image, presentation, visual_latents, refs):
        if image is None:
            return
        image = _to_pil(image)
        target_h, target_w = _resolve_canvas(*image.size, 2048)
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        video = _pil_to_video(image)
        latent = self._encode_video(video)
        presentation.append({"type": "image", "frames": _qwen_frames(video.clone())})
        visual_latents.append(latent)
        refs.append({"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1]})

    def _add_video_reference(self, video, soundtrack, presentation, visual_latents, audio_latents, refs):
        video = _as_video(video)
        if video is None:
            return
        if video.shape[1] < 5:
            video = torch.cat((video, video[:, -1:].repeat(1, 5 - video.shape[1], 1, 1)), dim=1)
        valid_frames = ((video.shape[1] - 5) // 17) * 17 + 5
        video = video[:, :valid_frames]
        latent = self._encode_video(video)
        audio_latent = self._encode_audio(soundtrack) if soundtrack is not None else None
        if audio_latent is not None:
            presentation.append({"type": "audio"})
            audio_latents.append(audio_latent)
        sample_indices = list(range(0, video.shape[1], FPS // 2))
        presentation.append({"type": "video", "frames": _qwen_frames(video[:, sample_indices].clone()),
                             "timestamps": [index / FPS for index in sample_indices]})
        visual_latents.append(latent)
        refs.append({"kind": "video_audio" if audio_latent is not None else "video", "latent_t": latent.shape[2],
                     "latent_h": latent.shape[-2], "latent_w": latent.shape[-1],
                     "ref_audio_t": 0 if audio_latent is None else audio_latent.shape[-1]})

    def _add_audio_reference(self, waveform, presentation, audio_latents, refs):
        if waveform is None:
            return
        latent = self._encode_audio(waveform)
        presentation.append({"type": "audio"})
        audio_latents.append(latent)
        refs.append({"kind": "audio", "ref_audio_t": latent.shape[-1]})

    def _prepare_condition_rows(self, visual_latents, audio_latents, generator):
        video_rows = []
        for latent in visual_latents:
            rows = patchify_video(latent.float(), self.transformer.patch_size)
            noise = patchify_video(torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu"), self.transformer.patch_size)
            video_rows.append(rows.mul(VISUAL_COND_TIMESTEP).add_(noise, alpha=1.0 - VISUAL_COND_TIMESTEP))
        return (torch.cat(video_rows) if video_rows else None,
                torch.cat([pack_audio(latent.float()) for latent in audio_latents]) if audio_latents else None)

    @_return_none_on_interrupt
    @torch.inference_mode()
    def generate(self, input_prompt, image_start=None, image_end=None, input_frames=None, input_ref_images=None,
                 input_video=None, video_guide=None, video_guide2=None, input_waveform=None, input_waveform_sample_rate=None,
                 audio_guide=None, audio_guide2=None, prefix_frames_count=0,
                 frame_num=124, height=768, width=1344, shift=12.0, sampling_steps=30, seed=0,
                 callback=None, VAE_tile_size=None, audio_prompt_type="", video_prompt_type="", fps=24,
                 sample_solver="euler",
                 set_progress_status=None, **kwargs):
        if int(fps) != FPS:
            raise ValueError(f"MiniMax H3 requires {FPS} fps")
        self._set_interrupt_state()
        self._check_abort()
        self._configure_tiling(VAE_tile_size)
        if sample_solver != "euler":
            raise ValueError("MiniMax H3 uses its released dual-schedule Euler sampler")
        if int(sampling_steps) < 1:
            raise ValueError("MiniMax H3 requires at least one inference step")
        frame_num = normalize_frame_count(int(frame_num), 5, 17, 5)
        if not 4.0 <= frame_num / FPS <= 481 / FPS:
            raise ValueError("MiniMax H3 generates between 4 and 20 seconds at 24 fps")
        prefix_frames_count = int(prefix_frames_count or 0)
        continuation = _as_video(input_video) if input_video is not None and prefix_frames_count > 0 else None
        sliding_prefix = continuation if self.reference_mode and prefix_frames_count >= 5 else None
        prefix_count = min(int(prefix_frames_count or 0), sliding_prefix.shape[1]) if sliding_prefix is not None else 0
        target_frames = frame_num - prefix_count
        aligned_target_frames = normalize_frame_count(target_frames, 5, 17, 5)
        if target_frames <= 0:
            raise ValueError("Sliding-window overlap leaves no frames for H3 to generate")

        presentation, visual_latents, audio_latents, refs, keyframes = [], [], [], [], []
        if self.reference_mode:
            if image_start is not None or image_end is not None:
                raise ValueError("First/last-frame conditioning requires the FL2VA checkpoint")
            for image in input_ref_images or []:
                self._add_image_reference(image, presentation, visual_latents, refs)
        else:
            if input_ref_images or input_frames is not None or "A" in (audio_prompt_type or ""):
                raise ValueError("Image, video, and audio references require the Ref2VA checkpoint")
            if image_start is None and continuation is not None:
                image_start = continuation[:, min(prefix_frames_count, continuation.shape[1]) - 1:][:, :1]
            if image_start is not None:
                self._add_image_condition(image_start, 0, presentation, visual_latents, keyframes)
            if image_end is not None:
                self._add_image_condition(image_end, aligned_target_frames - 1, presentation, visual_latents, keyframes)

        waveform = self._waveform(input_waveform, input_waveform_sample_rate)
        video_sources = []
        if self.reference_mode and "V" in (video_prompt_type or ""):
            video_sources.append(video_guide if video_guide is not None else input_frames)
            if "+" in (video_prompt_type or ""):
                video_sources.append(video_guide2)
        soundtrack_sources = (audio_guide, audio_guide2) if "K" in (audio_prompt_type or "") else (None, None)
        for index, source in enumerate(video_sources):
            video = _prepare_reference_video(source, 15 * FPS)
            soundtrack = self._load_audio_reference(soundtrack_sources[index]) if soundtrack_sources[index] is not None else None
            self._add_video_reference(video, soundtrack, presentation, visual_latents, audio_latents, refs)
        if self.reference_mode and "A" in (audio_prompt_type or ""):
            self._add_audio_reference(self._load_audio_reference(audio_guide) if audio_guide is not None else waveform, presentation, audio_latents, refs)
        if self.reference_mode and "B" in (audio_prompt_type or ""):
            self._add_audio_reference(self._load_audio_reference(audio_guide2), presentation, audio_latents, refs)
        if sliding_prefix is not None:
            prefix_audio = waveform if not any(flag in (audio_prompt_type or "") for flag in "ABK") else None
            self._add_video_reference(sliding_prefix[:, :prefix_count], prefix_audio, presentation, visual_latents, audio_latents, refs)
        if self.reference_mode:
            if not refs:
                raise ValueError("The Ref2VA checkpoint requires at least one image, video, or audio reference")
            visual_ref_count = sum(ref["kind"] in ("image", "video", "video_audio") for ref in refs)
            audio_ref_count = sum(ref["kind"] in ("audio", "video_audio") for ref in refs)
            if audio_ref_count > visual_ref_count:
                raise ValueError(f"MiniMax H3 requires at least as many image and video references as audio references (found {visual_ref_count} visual and {audio_ref_count} audio)")
            if len(refs) > 12 or sum(ref["kind"] == "image" for ref in refs) > 9 or sum(ref["kind"] in ("video", "video_audio") for ref in refs) > 2 or sum(ref["kind"] in ("audio", "video_audio") for ref in refs) > 2:
                raise ValueError("WanGP supports at most 12 MiniMax H3 references: 9 images, 2 videos, and 2 audio clips")

        if set_progress_status is not None:
            set_progress_status("Encoding H3 prompt and references")
        context, text_tags = self._encode_prompt(input_prompt, presentation)
        self._check_abort()
        context = self.transformer.preprocess_text_embeds(context)

        latent_t = video_latent_frames(aligned_target_frames)
        latent_h, latent_w = math.ceil(height / 16), math.ceil(width / 16)
        audio_t = round(aligned_target_frames / FPS * AUDIO_LATENT_FPS)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        cond_video_rows, cond_audio_rows = self._prepare_condition_rows(visual_latents, audio_latents, generator)
        video = torch.randn((1, 24, latent_t, latent_h, latent_w), generator=generator, dtype=torch.float32, device="cpu").to(self.device)
        audio = unpack_audio(torch.randn((audio_t * 2, 32), generator=generator, dtype=torch.float32, device="cpu")).to(self.device)
        payload = {"keyframes": keyframes or None, "refs": refs or None, "cond_video_rows": cond_video_rows,
                   "cond_audio_rows": cond_audio_rows, "frame_count": aligned_target_frames, "text_token_tags": text_tags}

        base_sigmas = torch.linspace(1.0, 0.0, int(sampling_steps) + 1, dtype=torch.float32)
        sigmas_video = torch.unique_consecutive(float(shift) * base_sigmas / (1.0 + (float(shift) - 1.0) * base_sigmas)).to(self.device)
        sigmas_audio = torch.unique_consecutive(3.0 * base_sigmas / (1.0 + 2.0 * base_sigmas)).to(self.device)
        if sigmas_video.shape != sigmas_audio.shape:
            raise ValueError("The selected H3 flow shift collapses a different number of video and audio schedule points")
        model_steps = sigmas_video.numel() - 1
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=model_steps)
        cache = self.transformer.cache
        spectrum = MiniMaxH3Spectrum(cache, sigmas_video[:-1]) if cache is not None and cache.cache_type == "spectrum" else None
        try:
            for step in tqdm(range(model_steps), desc="H3 denoising"):
                self._set_interrupt_state()
                self._check_abort()
                offload.set_step_no_for_lora(self.transformer, step)
                if spectrum is not None:
                    spectrum.begin_step(step)
                video_velocity, audio_velocity = self.transformer(video, audio, sigmas_video[step:step + 1], sigmas_audio[step:step + 1], context, payload, spectrum=spectrum)
                if spectrum is not None:
                    spectrum.finish_step()
                video_sigma_from_t = 1.0 - (1.0 - sigmas_video[step])
                video_ratio = sigmas_video[step + 1] / sigmas_video[step]
                video_velocity.mul_(video_sigma_from_t).add_(video)
                video.mul_(video_ratio).add_(video_velocity, alpha=1.0 - video_ratio)
                audio_sigma_from_t = 1.0 - (1.0 - sigmas_audio[step])
                audio_ratio = sigmas_audio[step + 1] / sigmas_audio[step]
                audio_velocity.mul_(audio_sigma_from_t).add_(audio)
                audio.mul_(audio_ratio).add_(audio_velocity, alpha=1.0 - audio_ratio)
                video_velocity = audio_velocity = None
                if callback is not None:
                    callback(step, video[0].detach().cpu(), False)
        finally:
            if spectrum is not None:
                spectrum.reset()

        if set_progress_status is not None:
            set_progress_status("Decoding H3 video and stereo audio")
        self._check_abort()
        context = payload = presentation = visual_latents = audio_latents = refs = keyframes = None
        decoded_video = self.vae.decode(video.to(self.vae._model_dtype)).clamp_(-1.0, 1.0)[0, :, :target_frames].cpu()
        video = None
        decoded_audio = self.audio_vae.decode(audio)[0]
        audio = None
        target_samples = round(target_frames / FPS * AUDIO_SAMPLE_RATE)
        decoded_audio = _fit_audio_samples(decoded_audio, target_samples)

        if sliding_prefix is not None:
            decoded_video = torch.cat((sliding_prefix[:, :prefix_count].to(decoded_video), decoded_video), dim=1)
            if waveform is not None:
                prefix_samples = round(prefix_count / FPS * AUDIO_SAMPLE_RATE)
                prefix_audio = _fit_audio_samples(waveform[0].to(decoded_audio), prefix_samples)
            else:
                prefix_samples = round(prefix_count / FPS * AUDIO_SAMPLE_RATE)
                prefix_audio = torch.zeros((2, prefix_samples), dtype=decoded_audio.dtype, device=decoded_audio.device)
            decoded_audio = torch.cat((prefix_audio, decoded_audio), dim=1)

        total_samples = round(frame_num / FPS * AUDIO_SAMPLE_RATE)
        decoded_audio = decoded_audio[:, :total_samples].transpose(0, 1).float().cpu().numpy()
        return {"x": decoded_video, "audio": decoded_audio, "audio_sampling_rate": AUDIO_SAMPLE_RATE}


__all__ = ["MiniMaxH3Pipeline", "video_latent_frames"]
