import json
import math
import os
from typing import Optional

import numpy as np
import torch
import torchaudio

from models.TTS.stable_audio3.loading_utils import load_diffusion_cond
from models.TTS.stable_audio3.model import StableAudioModel


STABLE_AUDIO3_SAMPLE_RATE = 44100
STABLE_AUDIO3_DURATION_PADDING_SEC = 6.0


def _read_text_or_file(value, label):
    if value is None:
        return ""
    if isinstance(value, str) and os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as reader:
            return reader.read()
    return str(value)


def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _audio_path(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "name"):
            if value.get(key):
                return value[key]
    if isinstance(value, (list, tuple)) and value:
        return _audio_path(value[0])
    return None


def _float_setting(custom_settings, key, default):
    if not isinstance(custom_settings, dict):
        return default
    value = custom_settings.get(key, default)
    if value is None or value == "":
        return default
    return float(value)


def _mode_from_audio_prompt_type(audio_prompt_type):
    audio_prompt_type = str(audio_prompt_type or "").upper()
    if "A" not in audio_prompt_type:
        return "text"
    if "E" in audio_prompt_type:
        return "audio_to_audio"
    if "I" in audio_prompt_type:
        return "inpaint"
    if "C" in audio_prompt_type:
        return "continue"
    return "text"


class StableAudio3MainModule(torch.nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.model = wrapper.model
        self.conditioner = wrapper.conditioner


class StableAudio3Pipeline:
    def __init__(
        self,
        transformer_weights_path,
        config_path,
        autoencoder_weights_path,
        text_encoder_weights_path,
        text_encoder_tokenizer_dir,
        *,
        model_id,
        max_duration,
        dtype=torch.bfloat16,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Stable Audio 3 support in WanGP is CUDA-only.")

        with open(config_path, "r", encoding="utf-8") as reader:
            model_config = json.load(reader)

        self.model_id = model_id
        self.max_duration = float(max_duration)
        self.dtype = dtype or torch.bfloat16
        self.device = torch.device("cuda")
        self.model = load_diffusion_cond(
            model_config,
            transformer_weights_path,
            pretransform_ckpt_path=autoencoder_weights_path,
            text_encoder_weights_path=text_encoder_weights_path,
            text_encoder_tokenizer_dir=text_encoder_tokenizer_dir,
            dtype=self.dtype,
        )
        self.core = StableAudioModel(self.model, model_config, self.device, model_half=False)
        self.main_model = StableAudio3MainModule(self.model)
        self.sample_rate = int(getattr(self.model, "sample_rate", STABLE_AUDIO3_SAMPLE_RATE))
        self._interrupt = False
        self._early_stop = False
        self._set_transformer_abort_hook()

    def _set_transformer_abort_hook(self):
        transformer = getattr(getattr(self.model.model, "model", None), "transformer", None)
        if transformer is not None:
            transformer.abort_callback = self._should_abort

    def get_trans_lora(self):
        return self.main_model, None

    def _abort_requested(self) -> bool:
        return bool(self._interrupt)

    def _early_stop_requested(self) -> bool:
        return bool(self._early_stop)

    def _should_abort(self) -> bool:
        return self._abort_requested() or self._early_stop_requested()

    def _load_audio_tuple(self, audio_value):
        path = _audio_path(audio_value)
        if path is None:
            return None
        waveform, sample_rate = torchaudio.load(path)
        return int(sample_rate), waveform

    def _audio_duration(self, audio_value):
        path = _audio_path(audio_value)
        if path is None:
            return 0.0
        try:
            info = torchaudio.info(path)
            if info.sample_rate > 0:
                return float(info.num_frames) / float(info.sample_rate)
        except Exception:
            pass
        sample_rate, waveform = self._load_audio_tuple(path)
        return float(waveform.shape[-1]) / float(sample_rate)

    def _sample_size(self, duration, duration_padding_sec):
        seconds = max(float(duration), 1.0) + max(float(duration_padding_sec), 0.0)
        downsampling_ratio = int(getattr(self.model.pretransform, "downsampling_ratio", 4096) or 4096)
        samples = int(math.ceil(seconds * self.sample_rate))
        return ((samples + downsampling_ratio - 1) // downsampling_ratio) * downsampling_ratio

    def _callback(self, callback, steps):
        if callback is not None:
            callback(step_idx=-1, override_num_inference_steps=int(steps), denoising_extra=f"0/{int(steps)} steps", progress_unit="steps")

        def _inner(payload):
            if callback is None:
                return
            step = int(payload.get("i", 0)) + 1
            callback(step_idx=step, override_num_inference_steps=int(steps), denoising_extra=f"{step}/{int(steps)} steps", progress_unit="steps")

        return _inner

    def generate(
        self,
        input_prompt: str,
        audio_guide: Optional[str] = None,
        *,
        batch_size=1,
        sample_solver=None,
        sampling_steps: int = 8,
        guide_scale: float = 1.0,
        n_prompt=None,
        seed: int = -1,
        callback=None,
        audio_prompt_type: str = "",
        audio_scale=None,
        custom_settings=None,
        duration_seconds: Optional[float] = None,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False

        prompt = _read_text_or_file(input_prompt, "Prompt").strip()
        if not prompt:
            raise ValueError("Prompt text cannot be empty for Stable Audio 3.")

        if seed is not None and int(seed) >= 0:
            _seed_everything(int(seed))

        mode = _mode_from_audio_prompt_type(audio_prompt_type)
        duration = float(duration_seconds) if duration_seconds and duration_seconds > 0 else min(30.0, self.max_duration)
        duration = max(1.0, min(duration, self.max_duration))
        steps = int(sampling_steps or 8)
        steps = max(1, steps)
        batch_size = max(1, int(batch_size or 1))
        cfg_scale = float(guide_scale if guide_scale is not None else 1.0)
        init_noise_level = 0.9 if audio_scale is None else float(audio_scale)
        inpaint_start = _float_setting(custom_settings, "inpaint_start_seconds", 0.0)
        inpaint_end = _float_setting(custom_settings, "inpaint_end_seconds", duration)
        duration_padding_sec = STABLE_AUDIO3_DURATION_PADDING_SEC

        init_audio = None
        inpaint_audio = None
        inpaint_mask_start_seconds = None
        inpaint_mask_end_seconds = None
        if mode == "audio_to_audio":
            init_audio = self._load_audio_tuple(audio_guide)
            if init_audio is None:
                raise ValueError("Stable Audio 3 audio-to-audio mode requires a source audio file.")
        elif mode in ("inpaint", "continue"):
            inpaint_audio = self._load_audio_tuple(audio_guide)
            if inpaint_audio is None:
                raise ValueError("Stable Audio 3 inpaint/continuation mode requires a source audio file.")
            if mode == "continue":
                inpaint_start = self._audio_duration(audio_guide)
                inpaint_end = duration
            inpaint_mask_start_seconds = max(0.0, min(float(inpaint_start), duration))
            inpaint_mask_end_seconds = max(inpaint_mask_start_seconds, min(float(inpaint_end), duration))

        sampler_type = str(sample_solver or "pingpong").strip().lower()

        try:
            audio = self.core.generate(
                prompt=prompt,
                negative_prompt=_read_text_or_file(n_prompt, "Negative prompt").strip() or None,
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                batch_size=batch_size,
                sample_size=self._sample_size(duration, duration_padding_sec),
                seed=int(seed) if seed is not None else -1,
                init_audio=init_audio,
                init_noise_level=max(0.0, min(float(init_noise_level), 1.0)),
                inpaint_audio=inpaint_audio,
                inpaint_mask_start_seconds=inpaint_mask_start_seconds,
                inpaint_mask_end_seconds=inpaint_mask_end_seconds,
                duration_padding_sec=duration_padding_sec,
                apg_scale=1.0,
                callback=self._callback(callback, steps),
                abort_fn=self._should_abort,
                disable_tqdm=False,
                sampler_type=sampler_type,
                chunked_decode=None,
            )
        except InterruptedError:
            return None

        if self._should_abort():
            return None
        if audio is None:
            return None
        audio = audio.to(torch.float32).cpu()
        return {"x": audio, "audio_sampling_rate": self.sample_rate}
