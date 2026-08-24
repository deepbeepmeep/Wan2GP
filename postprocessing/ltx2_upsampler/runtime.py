from __future__ import annotations

import gc

import numpy as np
import torch
from tqdm import tqdm

from shared.utils import offload_registry
from shared.utils.audio_video import slice_audio_window


RUNTIME_NAME = "LTX-2 Video Upsampler"
DEFAULT_WINDOW_FRAMES = 81
MAX_WINDOW_FRAMES = 481
WINDOW_OVERLAP_FRAMES = 17
TEMPORAL_STRIDE = 8
MODEL_TYPES = {"ltx23": "ltx2_22B", "ltx25": "ltx2_25_22B_distilled"}
FAKE_MODEL_TYPES = {"ltx23": "ltx2_upsampler_23", "ltx25": "ltx2_upsampler_25"}
LORA_KEYS = {"ltx23": ("ltx2_lora_distilled_1_1", "ltx2_lora_pixel_spatial_upscaler"), "ltx25": ("ltx2_lora_pixel_spatial_upscaler",)}
LORA_MULTIPLIERS = {"ltx23": (0.5, 1.0), "ltx25": (1.0,)}
AUDIO_SAMPLE_RATE = 16000


def lora_urls(wgp, variant: str) -> tuple[str, ...]:
    model_def = wgp.get_model_def(MODEL_TYPES[variant])
    return tuple(model_def[key] for key in LORA_KEYS[variant])


def _report(progress_callback, phase: str, current: int | None = None, total: int | None = None):
    if callable(progress_callback):
        progress_callback(phase, current, total)


def window_starts(frame_count: int, window_size: int = DEFAULT_WINDOW_FRAMES, window_overlap: int = WINDOW_OVERLAP_FRAMES) -> tuple[int, ...]:
    if window_size > MAX_WINDOW_FRAMES:
        raise ValueError(f"LTX upsampler window size cannot exceed {MAX_WINDOW_FRAMES} frames")
    if window_overlap >= window_size:
        raise ValueError("LTX upsampler window overlap must be smaller than its window size")
    if frame_count <= window_size:
        return (0,)
    return tuple(range(0, frame_count - window_overlap, window_size - window_overlap))


def pad_window(video: torch.Tensor) -> torch.Tensor:
    frame_count = int(video.shape[1])
    padded_frame_count = max(TEMPORAL_STRIDE + 1, ((frame_count - 1 + TEMPORAL_STRIDE - 1) // TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 1)
    if padded_frame_count == frame_count:
        return video
    return torch.cat((video, video[:, -1:].expand(-1, padded_frame_count - frame_count, -1, -1)), dim=1)


def crossfade_frames(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.0, torch.pi, previous.shape[1], device=previous.device, dtype=torch.float32).cos_().mul_(-0.5).add_(0.5).view(1, -1, 1, 1)
    return previous.float().lerp_(current.float(), weights).round_().to(previous.dtype)


def slice_audio_for_window(audio_waveform, audio_sample_rate: int, source_audio_path: str | None, start: int, frame_count: int, padded_frame_count: int, fps: float):
    if source_audio_path:
        audio, sample_rate = slice_audio_window(source_audio_path, start, frame_count, fps)
    else:
        sample_rate = int(audio_sample_rate or AUDIO_SAMPLE_RATE)
        if audio_waveform is None:
            audio = np.zeros((int(round(frame_count * sample_rate / fps)), 1), dtype=np.float32)
        else:
            audio = audio_waveform.detach().cpu().numpy() if torch.is_tensor(audio_waveform) else np.asarray(audio_waveform)
            if audio.ndim == 1:
                audio = audio[:, None]
            start_sample = int(round(start * sample_rate / fps))
            stop_sample = start_sample + int(round(frame_count * sample_rate / fps))
            audio = audio[start_sample:stop_sample]
    target_samples = int(round(padded_frame_count * sample_rate / fps))
    if audio.shape[0] < target_samples:
        audio = np.pad(audio, ((0, target_samples - audio.shape[0]), (0, 0)))
    return np.asarray(audio[:target_samples], dtype=np.float32), sample_rate


class LTXUpsamplerRuntime:
    def __init__(self):
        self.variant = None
        self.model = None
        self.offloadobj = None

    def load(self, variant: str) -> None:
        if self.model is not None and self.variant == variant:
            return
        self.release()
        if variant not in MODEL_TYPES:
            raise ValueError(f"Unknown LTX upsampler variant: {variant}")
        if not torch.cuda.is_available():
            raise RuntimeError("LTX video upsampling requires CUDA")
        import wgp

        self.variant = variant
        try:
            self.model, self.offloadobj = wgp.load_models(
                MODEL_TYPES[variant],
                override_profile=5,
                output_type="video",
                runtime_model_type=FAKE_MODEL_TYPES[variant],
                track_as_main=False,
            )
            from mmgp import offload

            lora_dir = wgp.get_lora_dir(MODEL_TYPES[variant])
            lora_paths = [wgp.get_lora_local_path(lora_dir, url) for url in lora_urls(wgp, variant)]
            transformer, _ = self.model.get_trans_lora()
            offload.load_loras_into_model(
                transformer,
                lora_paths,
                LORA_MULTIPLIERS[variant],
                activate_all_loras=True,
                preprocess_sd=wgp.get_loras_preprocessor(transformer, FAKE_MODEL_TYPES[variant]),
                pinnedLora=False,
                maxReservedLoras=wgp.server_config.get("max_reserved_loras", -1),
                split_linear_modules_map=getattr(transformer, "split_linear_modules_map", None),
            )
            if transformer._loras_errors:
                raise RuntimeError(f"Error while loading LTX {variant[3:]} Pixel Spatial Upscaler LoRAs: " + ", ".join(message for _, message in transformer._loras_errors))
            offload_registry.register_offloadobj(RUNTIME_NAME, self.offloadobj, self.release)
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        if self.offloadobj is not None:
            offload_registry.unregister_offloadobj(RUNTIME_NAME, self.offloadobj)
            self.offloadobj.release()
        self.variant = self.model = self.offloadobj = None
        gc.collect()

    def vae_tile_size(self, vae_config: int, output_height: int, output_width: int):
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1048576
        mixed_precision = self.model.VAE_dtype == torch.float32
        return self.model.vae.get_VAE_tile_size(vae_config, device_memory, mixed_precision, output_height=output_height, output_width=output_width)

    @torch.inference_mode()
    def upscale(
        self,
        sample: torch.Tensor,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        fps: float,
        window_size: int = DEFAULT_WINDOW_FRAMES,
        window_overlap: int = WINDOW_OVERLAP_FRAMES,
        frame_offset: int = 0,
        audio_waveform=None,
        audio_sample_rate: int = 0,
        source_audio_path: str | None = None,
        vae_tile_size=None,
        abort_callback=None,
        progress_callback=None,
    ):
        if self.model is None:
            raise RuntimeError("LTX video upsampler model is not loaded")
        frame_count = int(sample.shape[1])
        starts = window_starts(frame_count, window_size, window_overlap)
        output = None
        for window_index, start in enumerate(tqdm(starts, desc=f"LTX {self.variant[3:]} upsampler windows", unit="window"), start=1):
            if callable(abort_callback) and abort_callback():
                return None, None
            stop = min(start + window_size, frame_count)
            input_window = pad_window(sample[:, start:stop])
            audio_window, window_audio_sample_rate = slice_audio_for_window(audio_waveform, audio_sample_rate, source_audio_path, start, stop - start, int(input_window.shape[1]), fps)
            window_label = f"Window {window_index} / {len(starts)}" if len(starts) > 1 else ""

            def status_callback(phase, label=window_label):
                _report(progress_callback, f"{label} - {phase}" if label else phase)

            step_count = 8

            def step_callback(step_idx, _latent=None, _force_refresh=False, **_kwargs):
                phase = f"{window_label} - Distilled refinement" if window_label else "Distilled refinement"
                _report(progress_callback, phase, int(step_idx) + 1, step_count)

            window_output = self.model.upscale_video(
                input_window,
                prompt=prompt,
                negative_prompt=negative_prompt,
                audio_waveform=audio_window,
                audio_sample_rate=window_audio_sample_rate,
                upsampler_variant=self.variant,
                seed=seed,
                fps=fps,
                pixel_frame_offset=int(frame_offset) + start,
                VAE_tile_size=vae_tile_size,
                callback=step_callback,
                set_progress_status=status_callback,
                interrupt_check=abort_callback,
            )
            input_window = audio_window = None
            if window_output is None:
                return None, None
            window_output = window_output[:, : stop - start]
            if output is None:
                output = window_output.new_empty(window_output.shape[0], frame_count, *window_output.shape[2:])
                output[:, :stop].copy_(window_output)
            else:
                overlap = min(window_overlap, int(window_output.shape[1]))
                output[:, start:start + overlap].copy_(crossfade_frames(output[:, start:start + overlap], window_output[:, :overlap]))
                output[:, start + overlap:stop].copy_(window_output[:, overlap:])
            window_output = None
        _report(progress_callback, f"LTX {self.variant[3:]} upsampling complete")
        return output, None


RUNTIME = LTXUpsamplerRuntime()


def load_model(variant: str) -> None:
    RUNTIME.load(variant)


def upscale_video(sample: torch.Tensor, **kwargs):
    return RUNTIME.upscale(sample, **kwargs)


def release_model() -> None:
    RUNTIME.release()
