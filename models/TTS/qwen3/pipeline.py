from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import Qwen2TokenizerFast
from transformers.generation import StoppingCriteria, StoppingCriteriaList

from mmgp import offload
from shared.utils import files_locator as fl

from . import defs as qwen3_defs
from .core.models.configuration_qwen3_tts import Qwen3TTSConfig
from .core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from .core.models.processing_qwen3_tts import Qwen3TTSProcessor
from .inference.qwen3_tts_model import Qwen3TTSModel
from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_text_or_file(value: Optional[str], label: str) -> str:
    if value is None:
        return ""
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as handle:
            return handle.read()
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {type(value)}")
    return value


def _set_interrupt_check(module: torch.nn.Module, check_fn) -> None:
    module._interrupt_check = check_fn
    for child in module.modules():
        child._interrupt_check = check_fn


class _AbortAndProgressCriteria(StoppingCriteria):
    def __init__(
        self,
        total_seconds: int,
        seconds_per_token: float,
        abort_check,
        callback,
        early_stop_check=None,
    ):
        self.total_seconds = max(1, int(total_seconds))
        self.seconds_per_token = max(0.0, float(seconds_per_token or 0.0))
        self.abort_check = abort_check
        self.early_stop_check = early_stop_check
        self.callback = callback
        self._last_length = None
        self._generated_tokens = 0
        self._reported_seconds = 0
        self._progress = tqdm(total=self.total_seconds, desc="Qwen3 TTS", unit="s")

    def update(self, token_delta: int) -> None:
        if token_delta <= 0:
            return
        self._generated_tokens += token_delta
        if self.seconds_per_token <= 0:
            return
        generated_seconds = int(self._generated_tokens * self.seconds_per_token)
        if generated_seconds <= self._reported_seconds:
            return
        generated_seconds = min(generated_seconds, self.total_seconds)
        delta = generated_seconds - self._reported_seconds
        self._reported_seconds = generated_seconds
        self._progress.update(delta)
        if self.callback is not None:
            self.callback(
                step_idx=self._reported_seconds - 1,
                override_num_inference_steps=self.total_seconds,
                denoising_extra=f"{self._reported_seconds}s/{self.total_seconds}s",
                progress_unit="seconds",
            )

    def close(self) -> None:
        self._progress.close()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.early_stop_check is not None and self.early_stop_check():
            return True
        if self.abort_check():
            return True
        current_len = int(input_ids.shape[-1])
        if self._last_length is None:
            self._last_length = current_len
            return False
        delta = current_len - self._last_length
        self._last_length = current_len
        self.update(delta)
        return False


@dataclass
class _Qwen3Assets:
    weights_path: str
    config_path: str
    generate_config_path: Optional[str]
    text_tokenizer_dir: str
    speech_tokenizer_dir: str
    speech_tokenizer_weights: str


class Qwen3TTSPipeline:
    def __init__(
        self,
        model_weights_path: str,
        base_model_type: str,
        *,
        ckpt_root: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.base_model_type = base_model_type
        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else Path(fl.get_download_location())
        self._interrupt = False
        self._early_stop = False

        assets = self._resolve_assets(model_weights_path)
        self.model = self._load_main_model(assets)
        self.speech_tokenizer = self._load_speech_tokenizer(assets)
        self.processor = self._load_text_processor(assets)

        self.model.load_speech_tokenizer(self.speech_tokenizer)
        self.tts = Qwen3TTSModel(
            model=self.model,
            processor=self.processor,
            generate_defaults=self.model.generate_config or {},
        )
        self.sample_rate = int(self.speech_tokenizer.get_output_sample_rate())

        _set_interrupt_check(self.model, self._abort_requested)
        _set_interrupt_check(self.speech_tokenizer.model, self._abort_requested)

        self.supported_speakers = sorted(self.tts.get_supported_speakers() or [])
        self.supported_languages = sorted(self.tts.get_supported_languages() or [])

    def _abort_requested(self) -> bool:
        return bool(self._interrupt)

    def _early_stop_requested(self) -> bool:
        return bool(self._early_stop)

    def request_early_stop(self) -> None:
        self._early_stop = True

    def _get_tokens_per_second(self) -> float:
        try:
            decode_rate = self.speech_tokenizer.get_decode_upsample_rate()
            output_sr = self.speech_tokenizer.get_output_sample_rate()
            if decode_rate and output_sr:
                return float(output_sr) / float(decode_rate)
        except Exception:
            pass
        pos_per_second = getattr(self.model.config.talker_config, "position_id_per_seconds", None)
        if pos_per_second:
            return float(pos_per_second)
        return 1.0

    def _get_seconds_per_token(self) -> float:
        tokens_per_second = self._get_tokens_per_second()
        if tokens_per_second <= 0:
            return 1.0
        return 1.0 / tokens_per_second

    def _resolve_assets(self, model_weights_path: str) -> _Qwen3Assets:
        weights_path = model_weights_path
        config_path = qwen3_defs.get_qwen3_config_path(self.base_model_type)
        generate_config_path = qwen3_defs.get_qwen3_generation_config_path()
        text_tokenizer_dir = fl.locate_folder(qwen3_defs.QWEN3_TTS_TEXT_TOKENIZER_DIR)
        speech_tokenizer_dir = fl.locate_folder(qwen3_defs.QWEN3_TTS_SPEECH_TOKENIZER_DIR)
        speech_weights = fl.locate_file(os.path.join(qwen3_defs.QWEN3_TTS_SPEECH_TOKENIZER_DIR, qwen3_defs.QWEN3_TTS_SPEECH_TOKENIZER_WEIGHTS))

        return _Qwen3Assets(
            weights_path=weights_path,
            config_path=config_path,
            generate_config_path=generate_config_path,
            text_tokenizer_dir=text_tokenizer_dir,
            speech_tokenizer_dir=speech_tokenizer_dir,
            speech_tokenizer_weights=speech_weights,
        )

    def _load_main_model(self, assets: _Qwen3Assets) -> Qwen3TTSForConditionalGeneration:
        with open(assets.config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        config = Qwen3TTSConfig(**config_dict)
        with init_empty_weights():
            model = Qwen3TTSForConditionalGeneration(config)
        offload.load_model_data(
            model,
            assets.weights_path,
            default_dtype=None,
            writable_tensors=False,
        )
        model.eval()
        if assets.generate_config_path:
            with open(assets.generate_config_path, "r", encoding="utf-8") as handle:
                model.load_generate_config(json.load(handle))
        first_param = next(model.parameters(), None)
        if first_param is not None:
            model._model_dtype = first_param.dtype
        return model

    def _load_speech_tokenizer(self, assets: _Qwen3Assets) -> Qwen3TTSTokenizer:
        tokenizer = Qwen3TTSTokenizer.from_local(
            assets.speech_tokenizer_dir,
            assets.speech_tokenizer_weights,
        )
        return tokenizer

    def _load_text_processor(self, assets: _Qwen3Assets) -> Qwen3TTSProcessor:
        tokenizer = Qwen2TokenizerFast.from_pretrained(assets.text_tokenizer_dir)
        return Qwen3TTSProcessor(tokenizer=tokenizer)

    def _build_stopping_criteria(
        self,
        max_new_tokens: int,
        seconds_per_token: float,
        callback,
        total_seconds: Optional[int] = None,
    ):
        if total_seconds is None:
            total_seconds = max(1, int(math.ceil(max_new_tokens * seconds_per_token)))
        criteria = _AbortAndProgressCriteria(
            total_seconds,
            seconds_per_token,
            self._abort_requested,
            callback,
            early_stop_check=self._early_stop_requested,
        )
        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=criteria.total_seconds,
                denoising_extra=f"0s/{criteria.total_seconds}s",
                progress_unit="seconds",
            )
        return criteria, StoppingCriteriaList([criteria])

    def generate(
        self,
        input_prompt: str,
        model_mode: Optional[str],
        audio_guide: Optional[str],
        *,
        alt_prompt: Optional[str] = None,
        temperature: float = 0.9,
        seed: int = -1,
        callback=None,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False

        text = _read_text_or_file(input_prompt, "Prompt")
        if not text.strip():
            raise ValueError("Prompt text cannot be empty for Qwen3 TTS.")

        if seed is not None and int(seed) >= 0:
            _seed_everything(int(seed))

        duration_seconds = kwargs.get("duration_seconds", None)
        if duration_seconds is not None:
            try:
                duration_seconds = float(duration_seconds)
            except (TypeError, ValueError):
                duration_seconds = None

        seconds_per_token = self._get_seconds_per_token()
        total_seconds = None
        if duration_seconds is not None and duration_seconds > 0:
            tokens_per_second = self._get_tokens_per_second()
            max_new_tokens = max(1, int(math.ceil(duration_seconds * tokens_per_second)))
            total_seconds = max(1, int(math.ceil(duration_seconds)))
        else:
            max_new_tokens = kwargs.get("max_new_tokens")
            if max_new_tokens is None:
                sampling_steps = kwargs.get("sampling_steps")
                if sampling_steps:
                    max_new_tokens = int(sampling_steps)
            if max_new_tokens is None:
                max_new_tokens = self.tts.generate_defaults.get("max_new_tokens", 2048)
            max_new_tokens = int(max_new_tokens)

        criteria, stopping = self._build_stopping_criteria(
            max_new_tokens,
            seconds_per_token,
            callback,
            total_seconds=total_seconds,
        )
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": float(temperature),
            "stopping_criteria": stopping,
        }
        top_k = kwargs.get("top_k", None)
        if top_k is not None:
            try:
                gen_kwargs["top_k"] = int(top_k)
            except (TypeError, ValueError):
                pass

        try:
            if self.base_model_type == "qwen3_tts_customvoice":
                if not self.supported_speakers:
                    raise ValueError("No supported speakers found for Qwen3 CustomVoice.")
                speaker = model_mode or self.supported_speakers[0]
                language = "auto"
                wavs, sr = self.tts.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                    instruct=_read_text_or_file(alt_prompt, "Instruction"),
                    **gen_kwargs,
                )
            elif self.base_model_type == "qwen3_tts_voicedesign":
                language = (model_mode or "auto").lower()
                wavs, sr = self.tts.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=_read_text_or_file(alt_prompt, "Instruction"),
                    **gen_kwargs,
                )
            elif self.base_model_type == "qwen3_tts_base":
                if not audio_guide:
                    raise ValueError("Reference audio is required for Qwen3 Base voice clone.")
                language = (model_mode or "auto").lower()
                ref_text = _read_text_or_file(alt_prompt, "Reference transcript")
                x_vector_only_mode = not ref_text.strip()
                if x_vector_only_mode:
                    ref_text = None
                wavs, sr = self.tts.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=audio_guide,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    **gen_kwargs,
                )
            else:
                raise ValueError(f"Unknown Qwen3 TTS type: {self.base_model_type}")
        except RuntimeError as exc:
            if "Abort requested" in str(exc):
                return None
            raise
        finally:
            criteria.close()

        if self._abort_requested():
            return None

        wav = torch.from_numpy(wavs[0])
        return {"x": wav, "audio_sampling_rate": int(sr)}

    def release(self) -> None:
        for module in [self.model, getattr(self.speech_tokenizer, "model", None)]:
            if hasattr(module, "to"):
                module.to("cpu")
        self.model = None
        self.speech_tokenizer = None
