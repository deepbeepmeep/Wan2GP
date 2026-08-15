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
from shared.utils.frame_scheduler import floor_frame_count, normalize_frame_count, normalize_overlap
from .first_block_cache import MiniMaxH3FirstBlockCache
from .interrupt import GenerationInterrupted
from .spectrum import MiniMaxH3Spectrum
from .transformer import VISUAL_COND_TIMESTEP, pack_audio, patchify_video, unpack_audio


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


def _build_frozen_control_video(input_frames, input_video, frame_num, prefix_frames_count):
    control = _as_video(input_frames)
    if control is None:
        raise ValueError("MiniMax H3 audio-from-control-video mode requires a Control Video")
    prefix_count = min(int(prefix_frames_count), input_video.shape[1]) if input_video is not None else 0
    output_frames = min(int(frame_num), prefix_count + control.shape[1])
    if output_frames < 5:
        raise ValueError("MiniMax H3 audio-from-control-video mode requires at least 5 Control Video frames")
    output_frames = floor_frame_count(output_frames, 5, 17, 5)
    pieces = []
    if prefix_count:
        pieces.append(input_video[:, :min(prefix_count, output_frames)])
    remaining = output_frames - sum(piece.shape[1] for piece in pieces)
    if remaining:
        control = control[:, -remaining:] if pieces and control.shape[1] > remaining else control[:, :remaining]
        pieces.append(control)
    return torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]


def _qwen_frames(video):
    return video.permute(1, 2, 3, 0).add(1.0).mul_(0.5).clamp_(0.0, 1.0)


def _fit_audio_samples(audio, sample_count):
    return F.pad(audio[..., :sample_count], (0, max(0, sample_count - audio.shape[-1])))


def _resize_video_mask(mask, latent_shape, clip_length, temporal_ratio):
    latent_t, latent_h, latent_w = latent_shape
    mask = mask[:1].unsqueeze(0).float()
    pad_frames = (-mask.shape[2]) % clip_length
    if pad_frames:
        mask = F.pad(mask, (0, 0, 0, 0, 0, pad_frames), mode="replicate")
    offsets = torch.cat((torch.zeros(1, dtype=torch.long, device=mask.device),
                         torch.arange(1, clip_length, temporal_ratio, device=mask.device)))
    starts = torch.arange(0, mask.shape[2], clip_length, device=mask.device)
    frame_indices = (starts[:, None] + offsets[None]).flatten()[:latent_t]
    mask = mask.index_select(2, frame_indices)
    return F.interpolate(mask, size=(latent_t, latent_h, latent_w), mode="nearest").ge(0.5).float()


def _reinject_video_source(video, source, noise, editable_mask, sigma, buffer):
    torch.lerp(source, noise, sigma, out=buffer)
    if editable_mask is None:
        video.copy_(buffer)
    else:
        video.lerp_(buffer, 1.0 - editable_mask)


def _res_multistep_coefficients(sigmas):
    """Precompute deterministic second-order RES weights (arXiv:2308.02157)."""
    values = [float(sigma) for sigma in sigmas]
    coefficients = []
    old_sigma_down = None
    for index, (sigma, sigma_next) in enumerate(zip(values, values[1:])):
        if old_sigma_down is None or sigma_next == 0.0:
            ratio = sigma_next / sigma
            coefficients.append((ratio, 1.0 - ratio, 0.0))
        else:
            t = -math.log(sigma)
            h = -math.log(sigma_next) - t
            c2 = (-math.log(values[index - 1]) + math.log(old_sigma_down)) / h
            phi1 = math.expm1(-h) / -h
            phi2 = (phi1 - 1.0) / -h
            coefficients.append((math.exp(-h), h * (phi1 - phi2 / c2), h * phi2 / c2))
        old_sigma_down = sigma_next
    return coefficients


def _res_multistep_update(sample, denoised, old_denoised, coefficients):
    sample_coefficient, denoised_coefficient, old_denoised_coefficient = coefficients
    sample.mul_(sample_coefficient).add_(denoised, alpha=denoised_coefficient)
    if old_denoised_coefficient:
        sample.add_(old_denoised, alpha=old_denoised_coefficient)


def _resolve_canvas(width, height, short_edge, max_pixels=None):
    ratio = width / height
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

    def _encode_video(self, video, keep_all_latents=False):
        self._check_abort()
        return self.vae.encode_condition(video.unsqueeze(0).to(device=self.device, dtype=self.vae._model_dtype), keep_all_latents=keep_all_latents).cpu()

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

    def _add_image_condition(self, image, frame_index, presentation, visual_latents, keyframes, anchor=None):
        video = _as_video(image)
        if video is None:
            return
        video = video[:, :1]
        latent = self._encode_video(video)
        presentation.append({"type": "image", "frames": _qwen_frames(video.clone())})
        visual_latents.append(latent)
        keyframe = {"anchor": anchor or ("first" if frame_index == 0 else "last"), "latent_frame_count": latent.shape[2]}
        if anchor == "frame":
            keyframe["frame_index"] = int(frame_index)
        keyframes.append(keyframe)

    def _add_video_history(self, video, visual_latents, keyframes):
        latent = self._encode_video(video, keep_all_latents=True)
        visual_latents.append(latent)
        keyframes.append({"anchor": "history", "latent_frame_count": latent.shape[2]})

    def _add_audio_condition(self, latent, anchor, audio_latents, audio_keyframes):
        audio_latents.append(latent)
        audio_keyframes.append({"anchor": anchor, "latent_frame_count": latent.shape[-1]})

    def _add_image_reference(self, image, target_width, target_height, image_refs_relative_size, presentation, visual_latents, refs):
        if image is None:
            return
        image = _to_pil(image)
        ratio = image.width / image.height
        pixel_budget = target_width * target_height * image_refs_relative_size / 100
        target_h, target_w = _resolve_canvas(*image.size, math.sqrt(pixel_budget / max(ratio, 1 / ratio)))
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        video = _pil_to_video(image)
        latent = self._encode_video(video)
        presentation.append({"type": "image", "frames": _qwen_frames(video.clone())})
        visual_latents.append(latent)
        refs.append({"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1]})

    def _add_video_reference(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs):
        video = _as_video(video)
        if video is None:
            return
        if video.shape[1] < 5 or (video.shape[1] - 5) % 17:
            raise ValueError(f"MiniMax H3 reference videos must contain 17n+5 preprocessed frames, got {video.shape[1]}")
        latent = self._encode_video(video)
        audio_latent = self._encode_audio(soundtrack) if soundtrack is not None else None
        if audio_latent is not None:
            presentation.append({"type": "audio"})
            audio_latents.append(audio_latent)
        sample_indices, cursor = [], 0.0
        while round(cursor) < video.shape[1]:
            if not sample_indices or round(cursor) > sample_indices[-1]:
                sample_indices.append(round(cursor))
            cursor += fps / 2
        presentation.append({"type": "video", "frames": _qwen_frames(video[:, sample_indices].clone()),
                             "timestamps": [index / fps for index in sample_indices]})
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
    def generate(self, input_prompt, image_start=None, image_end=None, input_frames=None, input_frames2=None, input_ref_images=None,
                 frames_to_inject=None, frames_relative_positions_list=None, image_refs_relative_size=100,
                 input_masks=None, denoising_strength=1.0, masking_strength=1.0,
                 input_video=None, input_waveform=None, input_waveform_sample_rate=None,
                 audio_guide=None, audio_guide2=None, prefix_frames_count=0,
                 frame_num=124, height=768, width=1344, shift=12.0, sampling_steps=30, seed=0,
                 callback=None, VAE_tile_size=None, audio_prompt_type="", video_prompt_type="", fps=24,
                 sample_solver="euler",
                 set_progress_status=None, **kwargs):
        fps = float(fps)
        if fps <= 0:
            raise ValueError("MiniMax H3 requires a positive output frame rate")
        self._set_interrupt_state()
        self._check_abort()
        self._configure_tiling(VAE_tile_size)
        if sample_solver not in ("euler", "res_multistep"):
            raise ValueError(f"Unsupported MiniMax H3 sampler {sample_solver!r}")
        if int(sampling_steps) < 1:
            raise ValueError("MiniMax H3 requires at least one inference step")
        frame_num = normalize_frame_count(int(frame_num), 5, 17, 5)
        audio_from_control_video = not self.reference_mode and "2" in (audio_prompt_type or "")
        prefix_frames_count, overlap_error = normalize_overlap(int(prefix_frames_count or 0), 17, 1)
        if overlap_error:
            raise ValueError(overlap_error)
        continuation = _as_video(input_video) if input_video is not None and prefix_frames_count > 0 else None
        continuation_count = min(prefix_frames_count, continuation.shape[1]) if continuation is not None and image_start is None else 0
        if continuation_count and continuation_count < prefix_frames_count:
            continuation_count = floor_frame_count(continuation_count, 1, 17, 1)
        frozen_control_video = (_build_frozen_control_video(input_frames, continuation, frame_num, continuation_count)
                                if audio_from_control_video else None)
        if frozen_control_video is not None:
            frame_num = frozen_control_video.shape[1]
        history_frames = continuation[:, -continuation_count:-1] if continuation_count > 1 else None
        history_count = 0 if history_frames is None else history_frames.shape[1]
        target_frames = frame_num - history_count
        aligned_target_frames = normalize_frame_count(target_frames, 5, 17, 5)
        if target_frames <= 0:
            raise ValueError("Sliding-window overlap leaves no frames for H3 to generate")
        control_video = not self.reference_mode and "G" in (video_prompt_type or "")
        video_to_video = control_video and not audio_from_control_video and (float(denoising_strength) < 1.0 or input_masks is not None)

        waveform = self._waveform(input_waveform, input_waveform_sample_rate)
        history_waveform = None
        target_audio_condition = None
        target_video_condition = None
        frozen_target_video = None if frozen_control_video is None else frozen_control_video[:, history_count:]
        presentation, visual_latents, audio_latents, refs, keyframes, audio_keyframes = [], [], [], [], [], []
        if not self.reference_mode and (input_ref_images or input_frames is not None and not (control_video or audio_from_control_video) or input_frames2 is not None):
            raise ValueError("Image, video, and audio references require the Ref2VA checkpoint")
        if continuation_count:
            if history_frames is not None:
                self._add_video_history(history_frames, visual_latents, keyframes)
            self._add_image_condition(continuation[:, -1:], 0, presentation, visual_latents, keyframes)
            if waveform is not None:
                overlap_samples = round(continuation_count / fps * AUDIO_SAMPLE_RATE)
                continuation_waveform = _fit_audio_samples(waveform[0], overlap_samples).unsqueeze(0)
                history_samples = round(history_count / fps * AUDIO_SAMPLE_RATE)
                history_waveform = continuation_waveform[..., :history_samples] if history_samples else None
                continuation_audio = self._encode_audio(continuation_waveform)
                boundary_latents = (continuation_audio.shape[-1] if not history_count else
                                    min(continuation_audio.shape[-1], max(1, round(AUDIO_LATENT_FPS / fps))))
                history_latents = continuation_audio.shape[-1] - boundary_latents
                if history_latents:
                    self._add_audio_condition(continuation_audio[..., :history_latents], "history", audio_latents, audio_keyframes)
                self._add_audio_condition(continuation_audio[..., history_latents:], "first", audio_latents, audio_keyframes)
        elif image_start is not None and not audio_from_control_video:
            self._add_image_condition(image_start, 0, presentation, visual_latents, keyframes)
        if image_end is not None and not audio_from_control_video:
            self._add_image_condition(image_end, aligned_target_frames - 1, presentation, visual_latents, keyframes)
        for image, frame_index in zip(frames_to_inject or (), frames_relative_positions_list or ()):
            frame_index = int(frame_index) - history_count
            if 0 <= frame_index < target_frames:
                image = _to_pil(image)
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                self._add_image_condition(_pil_to_video(image), frame_index, presentation, visual_latents, keyframes, anchor="frame")
        if frozen_target_video is not None:
            target_video_condition = self._encode_video(frozen_target_video, keep_all_latents=True)
        if self.reference_mode:
            for image in input_ref_images or []:
                self._add_image_reference(image, width, height, image_refs_relative_size, presentation, visual_latents, refs)

        video_sources = []
        if self.reference_mode and "V" in (video_prompt_type or ""):
            video_sources.append(input_frames)
            if "+" in (video_prompt_type or ""):
                video_sources.append(input_frames2)
        video_sources = [_as_video(source) for source in video_sources]
        total_reference_duration = sum(video.shape[1] for video in video_sources) / fps
        if total_reference_duration > 15:
            raise ValueError(f"MiniMax H3 reference videos must total at most 15 seconds (found {total_reference_duration:.2f}s)")
        soundtrack_sources = (audio_guide, audio_guide2) if "K" in (audio_prompt_type or "") else (None, None)
        for index, source in enumerate(video_sources):
            soundtrack = self._load_audio_reference(soundtrack_sources[index]) if soundtrack_sources[index] is not None else None
            self._add_video_reference(source, soundtrack, fps, presentation, visual_latents, audio_latents, refs)
        if self.reference_mode and "A" in (audio_prompt_type or ""):
            self._add_audio_reference(self._load_audio_reference(audio_guide) if audio_guide is not None else waveform, presentation, audio_latents, refs)
        if self.reference_mode and "B" in (audio_prompt_type or ""):
            self._add_audio_reference(self._load_audio_reference(audio_guide2), presentation, audio_latents, refs)
        if not self.reference_mode and any(flag in (audio_prompt_type or "") for flag in "AK") and waveform is not None:
            condition_start = round(history_count / fps * AUDIO_SAMPLE_RATE)
            condition_samples = round(target_frames / fps * AUDIO_SAMPLE_RATE)
            condition_waveform = waveform[..., condition_start:condition_start + condition_samples]
            if condition_waveform.shape[-1]:
                target_audio_condition = self._encode_audio(condition_waveform)
        if self.reference_mode:
            visual_ref_count = sum(ref["kind"] in ("image", "video", "video_audio") for ref in refs)
            audio_ref_count = sum(ref["kind"] in ("audio", "video_audio") for ref in refs)
            if audio_ref_count > visual_ref_count:
                raise ValueError(f"MiniMax H3 requires at least as many image and video references as audio references (found {visual_ref_count} visual and {audio_ref_count} audio)")
            if len(refs) > 12 or sum(ref["kind"] == "image" for ref in refs) > 9 or sum(ref["kind"] in ("video", "video_audio") for ref in refs) > 2 or sum(ref["kind"] in ("audio", "video_audio") for ref in refs) > 2:
                raise ValueError("WanGP supports at most 12 MiniMax H3 references: 9 images, 2 videos, and 2 audio clips")

        source_latents = editable_mask = None
        if video_to_video:
            if set_progress_status is not None:
                set_progress_status("Encoding H3 control video")
            self._check_abort()
            source_video = _as_video(input_frames)[:, history_count:history_count + aligned_target_frames]
            source_latents = self.vae.encode(source_video.unsqueeze(0).to(device=self.device, dtype=self.vae._model_dtype)).cpu()
            self._check_abort()
            if input_masks is not None:
                source_mask = input_masks[:, history_count:history_count + source_video.shape[1]]
                editable_mask = _resize_video_mask(source_mask, source_latents.shape[-3:], self.vae.config.clip_length, self.vae.temporal_compression_ratio)
            source_video = source_mask = None

        if set_progress_status is not None:
            set_progress_status("Encoding H3 prompt and references")
        context, text_tags = self._encode_prompt(input_prompt, presentation)
        self._check_abort()
        context = self.transformer.preprocess_text_embeds(context)

        latent_t = video_latent_frames(aligned_target_frames)
        latent_h, latent_w = math.ceil(height / 16), math.ceil(width / 16)
        audio_t = round(aligned_target_frames / fps * AUDIO_LATENT_FPS)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        cond_video_rows, cond_audio_rows = self._prepare_condition_rows(visual_latents, audio_latents, generator)
        target_video_condition_frames = 0 if target_video_condition is None else target_video_condition.shape[2]
        video = (torch.randn((1, 24, latent_t, latent_h, latent_w), generator=generator, dtype=torch.float32, device="cpu")
                 if target_video_condition is None else target_video_condition).to(self.device)
        target_video_condition = None
        if source_latents is not None:
            source_latents = source_latents[:, :, :latent_t].to(device=video.device, dtype=video.dtype)
            source_noise = video[:, :, :source_latents.shape[2]].clone()
            source_buffer = torch.empty_like(source_latents)
            editable_mask = editable_mask[:, :, :source_latents.shape[2]].to(video) if editable_mask is not None else None
        else:
            source_noise = source_buffer = None
        audio = unpack_audio(torch.randn((audio_t * 2, 32), generator=generator, dtype=torch.float32, device="cpu")).to(self.device)
        target_audio_condition_latents = 0 if target_audio_condition is None else min(audio_t, target_audio_condition.shape[-1])
        if target_audio_condition_latents:
            audio[..., :target_audio_condition_latents].copy_(target_audio_condition[..., :target_audio_condition_latents].to(audio))
        target_audio_condition = None
        payload = {"keyframes": keyframes or None, "audio_keyframes": audio_keyframes or None,
                   "refs": refs or None, "cond_video_rows": cond_video_rows,
                   "cond_audio_rows": cond_audio_rows, "frame_count": aligned_target_frames, "text_token_tags": text_tags,
                   "fps": fps, "target_audio_condition_latents": target_audio_condition_latents,
                   "target_video_condition_frames": target_video_condition_frames}

        base_sigmas = torch.linspace(1.0, 0.0, int(sampling_steps) + 1, dtype=torch.float32)
        sigmas_video = torch.unique_consecutive(float(shift) * base_sigmas / (1.0 + (float(shift) - 1.0) * base_sigmas))
        sigmas_audio = torch.unique_consecutive(3.0 * base_sigmas / (1.0 + 2.0 * base_sigmas))
        if sigmas_video.shape != sigmas_audio.shape:
            raise ValueError("The selected H3 flow shift collapses a different number of video and audio schedule points")
        res_coefficients = _res_multistep_coefficients(sigmas_video) if sample_solver == "res_multistep" else None
        sigmas_video, sigmas_audio = sigmas_video.to(self.device), sigmas_audio.to(self.device)
        model_steps = sigmas_video.numel() - 1
        denoising_start_step = int(round(model_steps * (1.0 - float(denoising_strength)), 4))
        mask_end_step = min(model_steps, denoising_start_step + math.ceil(model_steps * float(masking_strength))) if editable_mask is not None else 0
        cache = self.transformer.cache
        spectrum = MiniMaxH3Spectrum(cache, sigmas_video[:-1], sample_solver) if cache is not None and cache.cache_type == "spectrum" else None
        first_block_cache = MiniMaxH3FirstBlockCache(cache) if cache is not None and cache.cache_type == "first_block" else None
        offline_spectrum = spectrum is not None and spectrum.full_anchor_cache
        audio_scale = float(shift) / 3.0

        def denoise_pass(description, denoising_extra=""):
            nonlocal video, audio
            old_video_denoised = old_audio_denoised = None
            for step in tqdm(range(model_steps), desc=description):
                self._set_interrupt_state()
                self._check_abort()
                offload.set_step_no_for_lora(self.transformer, step)
                if spectrum is not None:
                    spectrum.begin_step(step)
                if first_block_cache is not None:
                    first_block_cache.begin_step(step)
                audio_tail = audio[..., target_audio_condition_latents:]
                if res_coefficients is not None and audio_tail.shape[-1]:
                    # RES carries generated audio on the video schedule; H3 still receives its native audio state.
                    audio_tail.mul_(sigmas_audio[step] / sigmas_video[step])
                video_velocity, audio_velocity = self.transformer(video, audio, sigmas_video[step:step + 1], sigmas_audio[step:step + 1], context, payload, spectrum=spectrum, first_block_cache=first_block_cache)
                if spectrum is not None:
                    spectrum.finish_step()
                if res_coefficients is None:
                    video_ratio = sigmas_video[step + 1] / sigmas_video[step]
                    if not target_video_condition_frames:
                        video_velocity.mul_(sigmas_video[step]).add_(video)
                        video.mul_(video_ratio).add_(video_velocity, alpha=1.0 - video_ratio)
                    audio_ratio = sigmas_audio[step + 1] / sigmas_audio[step]
                    if audio_tail.shape[-1]:
                        audio_velocity_tail = audio_velocity[..., target_audio_condition_latents:]
                        audio_velocity_tail.mul_(sigmas_audio[step]).add_(audio_tail)
                        audio_tail.mul_(audio_ratio).add_(audio_velocity_tail, alpha=1.0 - audio_ratio)
                else:
                    coefficients = res_coefficients[step]
                    if not target_video_condition_frames:
                        video_denoised = video_velocity.mul_(sigmas_video[step]).add_(video)
                        _res_multistep_update(video, video_denoised, old_video_denoised, coefficients)
                        old_video_denoised = video_denoised
                    if audio_tail.shape[-1]:
                        audio_velocity_tail = audio_velocity[..., target_audio_condition_latents:]
                        audio_denoised = audio_velocity_tail.mul_(sigmas_audio[step]).add_(audio_tail).mul_(audio_scale)
                        audio_tail.mul_(sigmas_video[step] / sigmas_audio[step])
                        _res_multistep_update(audio_tail, audio_denoised, old_audio_denoised, coefficients)
                        old_audio_denoised = audio_denoised
                if source_latents is not None and (step < denoising_start_step or step < mask_end_step):
                    source_video = video[:, :, :source_latents.shape[2]]
                    source_mask = None if step < denoising_start_step else editable_mask
                    _reinject_video_source(source_video, source_latents, source_noise, source_mask, sigmas_video[step + 1], source_buffer)
                video_velocity = audio_velocity = None
                if callback is not None:
                    preview = video[0].detach().cpu() if not offline_spectrum or spectrum.replaying else None
                    callback(step, preview, False, denoising_extra=denoising_extra) if denoising_extra else callback(step, preview, False)

        initial_video = initial_audio = None
        try:
            if offline_spectrum:
                initial_video = video.detach().to(device="cpu", copy=True, non_blocking=False)
                initial_audio = audio.detach().to(device="cpu", copy=True, non_blocking=False)
                if set_progress_status is not None:
                    set_progress_status("Denoising")
                if callback is not None:
                    callback(-1, None, True, override_num_inference_steps=model_steps)
                denoise_pass("H3 Spectrum anchor capture")
                spectrum.complete_capture(self._check_abort)
                spectrum.start_replay()
                video = initial_video.to(self.device)
                audio = initial_audio.to(self.device)
                if set_progress_status is not None:
                    set_progress_status("Spectrum smoothing replay")
                if callback is not None:
                    callback(-1, None, True, override_num_inference_steps=model_steps, denoising_extra="Spectrum smoothing replay")
                denoise_pass("H3 Spectrum replay", "Spectrum smoothing replay")
            else:
                if callback is not None:
                    callback(-1, None, True, override_num_inference_steps=model_steps)
                denoise_pass("H3 denoising")
        finally:
            if spectrum is not None:
                spectrum.reset()
            if first_block_cache is not None:
                first_block_cache.reset()

        if res_coefficients is not None:
            audio[..., target_audio_condition_latents:].div_(audio_scale)
        initial_video = initial_audio = None

        if set_progress_status is not None:
            set_progress_status("Decoding H3 stereo audio" if frozen_target_video is not None else "VAE Decoding of Video and Audio")
        self._check_abort()
        context = payload = presentation = visual_latents = audio_latents = refs = keyframes = audio_keyframes = source_latents = source_noise = source_buffer = editable_mask = None
        decoded_video = (self.vae.decode(video.to(self.vae._model_dtype)).clamp_(-1.0, 1.0)[0, :, :target_frames].cpu()
                         if frozen_target_video is None else frozen_target_video[:, :target_frames].cpu())
        video = None
        decoded_audio = self.audio_vae.decode(audio)[0]
        audio = None
        target_samples = round(target_frames / fps * AUDIO_SAMPLE_RATE)
        decoded_audio = _fit_audio_samples(decoded_audio, target_samples)

        output_prefix = history_frames
        output_prefix_count = history_count
        if output_prefix is not None:
            decoded_video = torch.cat((output_prefix.to(decoded_video), decoded_video), dim=1)
            if history_waveform is not None:
                prefix_samples = round(output_prefix_count / fps * AUDIO_SAMPLE_RATE)
                prefix_audio = _fit_audio_samples(history_waveform[0].to(decoded_audio), prefix_samples)
            else:
                prefix_samples = round(output_prefix_count / fps * AUDIO_SAMPLE_RATE)
                prefix_audio = torch.zeros((2, prefix_samples), dtype=decoded_audio.dtype, device=decoded_audio.device)
            decoded_audio = torch.cat((prefix_audio, decoded_audio), dim=1)

        total_samples = round(frame_num / fps * AUDIO_SAMPLE_RATE)
        decoded_audio = decoded_audio[:, :total_samples].transpose(0, 1).float().cpu().numpy()
        return {"x": decoded_video, "audio": decoded_audio, "audio_sampling_rate": AUDIO_SAMPLE_RATE}


__all__ = ["MiniMaxH3Pipeline", "video_latent_frames"]
