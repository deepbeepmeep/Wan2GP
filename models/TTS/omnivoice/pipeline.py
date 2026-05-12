from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from accelerate import init_empty_weights
from transformers import AutoFeatureExtractor, AutoTokenizer

from mmgp import offload
from shared.utils import files_locator as fl

from .higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel
from .modeling_omnivoice import (
    OMNIVOICE_AUTO_REF_LEAD_SILENCE_MS,
    OMNIVOICE_AUTO_REF_MAX_DURATION,
    OMNIVOICE_AUTO_REF_MID_SILENCE_MS,
    OMNIVOICE_AUTO_REF_TRAIL_SILENCE_MS,
    OMNIVOICE_AUTO_REF_TRIM_THRESHOLD,
    OmniVoice,
    OmniVoiceConfig,
    OmniVoiceGenerationConfig,
    VoiceClonePrompt,
)
from .utils.duration import RuleDurationEstimator
from .utils.voice_design import _INSTRUCT_VALID_EN, _INSTRUCT_VALID_ZH


OMNIVOICE_ASSET_DIR = "omnivoice"
OMNIVOICE_AUDIO_TOKENIZER_DIR = "omnivoice/audio_tokenizer"
OMNIVOICE_CONFIG_NAME = "config.json"
OMNIVOICE_AUDIO_TOKENIZER_WEIGHTS = "audio_tokenizer_bf16.safetensors"
OMNIVOICE_DEFAULT_VOICE_INSTRUCTION = ""
OMNIVOICE_LEGACY_DEFAULT_VOICE_INSTRUCTION = "female, warm tone, clear articulation"
OMNIVOICE_SIGNATURE_CHUNK_SIZE = 65536
OMNIVOICE_AUTO_END_TRIM_FLAG = "E"
OMNIVOICE_AUTO_SPLIT_SETTING_ID = "auto_split_every_s"
OMNIVOICE_AUTO_SPLIT_MIN_SECONDS = 5.0
OMNIVOICE_AUTO_SPLIT_MAX_SECONDS = 90.0
OMNIVOICE_TRAILING_SILENCE_WINDOW_SECONDS = 0.02
OMNIVOICE_TRAILING_SILENCE_KEEP_SECONDS = 0.20
OMNIVOICE_TRAILING_SILENCE_RELATIVE_RMS = 0.015
OMNIVOICE_TRAILING_SILENCE_MIN_RMS = 1e-4
OMNIVOICE_AUTO_SPLIT_BOUNDARY_PUNCTUATION = set(".。．｡!！?？;；:：,，、،؛؟।॥…")
OMNIVOICE_SPECIAL_TOKENS = [
    "<|denoise|>",
    "<|lang_start|>",
    "<|lang_end|>",
    "<|instruct_start|>",
    "<|instruct_end|>",
    "<|text_start|>",
    "<|text_end|>",
]
_TRANSCRIPTION_CACHE: dict[tuple, str] = {}
_WORD_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", flags=re.UNICODE)


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


def normalize_omnivoice_voice_instruction(text: str) -> str:
    if text.strip().lower() == OMNIVOICE_LEGACY_DEFAULT_VOICE_INSTRUCTION:
        return ""
    return text


def is_omnivoice_voice_instruction(text: str) -> bool:
    parts = [part.strip() for part in re.split(r"[,\uff0c]", text or "") if part.strip()]
    if not parts:
        return False
    return all(part.lower() in _INSTRUCT_VALID_EN or part in _INSTRUCT_VALID_ZH for part in parts)


def _reference_audio_signature(ref_audio) -> tuple:
    if isinstance(ref_audio, str) and os.path.isfile(ref_audio):
        path = Path(ref_audio).resolve()
        stat = path.stat()
        digest = hashlib.blake2b(digest_size=16)
        with path.open("rb") as handle:
            digest.update(handle.read(OMNIVOICE_SIGNATURE_CHUNK_SIZE))
            if stat.st_size > OMNIVOICE_SIGNATURE_CHUNK_SIZE:
                handle.seek(max(0, stat.st_size - OMNIVOICE_SIGNATURE_CHUNK_SIZE))
                digest.update(handle.read(OMNIVOICE_SIGNATURE_CHUNK_SIZE))
        return ("file", int(stat.st_size), int(stat.st_mtime_ns), digest.hexdigest())
    if isinstance(ref_audio, tuple):
        waveform, sample_rate = ref_audio
        array = waveform.detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform)
        flat = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(flat[:OMNIVOICE_SIGNATURE_CHUNK_SIZE].tobytes())
        if flat.size > OMNIVOICE_SIGNATURE_CHUNK_SIZE:
            digest.update(flat[-OMNIVOICE_SIGNATURE_CHUNK_SIZE:].tobytes())
        return ("waveform", tuple(array.shape), str(array.dtype), int(sample_rate), digest.hexdigest())
    return ("object", repr(ref_audio))


def _transcription_cache_key(ref_audio) -> tuple:
    return (
        "omnivoice_ref_asr_v1",
        OMNIVOICE_AUTO_REF_MAX_DURATION,
        OMNIVOICE_AUTO_REF_TRIM_THRESHOLD,
        OMNIVOICE_AUTO_REF_MID_SILENCE_MS,
        OMNIVOICE_AUTO_REF_LEAD_SILENCE_MS,
        OMNIVOICE_AUTO_REF_TRAIL_SILENCE_MS,
        _reference_audio_signature(ref_audio),
    )


def _is_abort_exception(exc: Exception) -> bool:
    return "Abort requested" in str(exc)


@dataclass
class _OmniVoiceAssets:
    weights_path: str
    config_path: str
    text_tokenizer_dir: str
    audio_tokenizer_dir: str
    audio_tokenizer_weights: str


class OmniVoicePipeline:
    def __init__(
        self,
        model_weights_path: str,
        *,
        ckpt_root: Optional[Path] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else Path(fl.get_download_location())
        self._interrupt = False
        self._early_stop = False
        self._whisper_model = None
        self._whisper_device = None
        self._owns_whisper_model = False
        self._active_offloadobj = None
        self._verbose_level = 0

        assets = self._resolve_assets(model_weights_path)
        self.model = self._load_main_model(assets)
        self.audio_tokenizer = self._load_audio_tokenizer(assets)
        self.text_tokenizer = AutoTokenizer.from_pretrained(assets.text_tokenizer_dir, extra_special_tokens={})
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": OMNIVOICE_SPECIAL_TOKENS})
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(assets.audio_tokenizer_dir)
        self.sample_rate = int(self.feature_extractor.sampling_rate)

        self.model.text_tokenizer = self.text_tokenizer
        self.model.feature_extractor = self.feature_extractor
        self.model.sampling_rate = self.sample_rate
        self.model.duration_estimator = RuleDurationEstimator()
        self.model.__dict__["audio_tokenizer"] = self.audio_tokenizer
        self.model._abort_callback = self._abort_requested
        self.model._transcribe_reference_callback = self._auto_transcribe_reference_audio
        self.model.eval().requires_grad_(False)
        self.audio_tokenizer.eval().requires_grad_(False)

    def set_whisper_model(self, whisper_model) -> None:
        self._whisper_model = whisper_model
        self._whisper_device = torch.device("cpu")
        self._owns_whisper_model = False

    def _abort_requested(self) -> bool:
        return bool(self._interrupt)

    def _early_stop_requested(self) -> bool:
        return bool(self._early_stop)

    def request_early_stop(self) -> None:
        self._early_stop = True

    def release(self) -> None:
        self.model._abort_callback = None
        self.model._progress_callback = None
        self.model._transcribe_reference_callback = None
        self._release_whisper_model()

    def _get_whisper_model(self, offloadobj=None):
        if self._whisper_model is None:
            raise RuntimeError("OmniVoice Whisper support requires the handler-provided MMGP-managed Whisper model.")
        if offloadobj is not None:
            offloadobj.ensure_model_loaded("whisper")
        device = next(self._whisper_model.parameters()).device
        self._whisper_device = device
        return self._whisper_model, device

    def _release_whisper_model(self) -> None:
        if not self._owns_whisper_model:
            return
        model = self._whisper_model
        self._whisper_model = None
        self._whisper_device = None
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _whisper_uses_fp16(whisper_model, whisper_device: torch.device) -> bool:
        return whisper_device.type == "cuda" and getattr(whisper_model, "_model_dtype", None) == torch.float16

    def _auto_transcribe_reference_audio(self, ref_audio, ref_wav: np.ndarray, sample_rate: int) -> str:
        signature = _transcription_cache_key(ref_audio)
        cached_text = _TRANSCRIPTION_CACHE.get(signature)
        if cached_text is not None:
            print("OmniVoice using cached Whisper transcript for reference audio.")
            return cached_text

        print("OmniVoice auto-transcribing reference audio with Whisper...")
        whisper_model, whisper_device = self._get_whisper_model(self._active_offloadobj)
        waveform = np.asarray(ref_wav, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        waveform_tensor = torch.from_numpy(waveform)
        if int(sample_rate) != 16000:
            waveform_tensor = torchaudio.functional.resample(waveform_tensor, int(sample_rate), 16000)
        try:
            result = whisper_model.transcribe(waveform_tensor.numpy().astype(np.float32), verbose=None, fp16=self._whisper_uses_fp16(whisper_model, whisper_device))
        finally:
            self._release_whisper_model()
        transcript = str(result.get("text", "") or "").strip()
        if not transcript:
            raise ValueError("Whisper could not transcribe the OmniVoice reference audio. Provide the transcript manually in the Voice instruction/reference transcript field.")
        _TRANSCRIPTION_CACHE[signature] = transcript
        return transcript

    @staticmethod
    def _words_from_text(text: str) -> list[str]:
        return _WORD_RE.findall(str(text or ""))

    @staticmethod
    def _count_words(text: str) -> int:
        return len(OmniVoicePipeline._words_from_text(text))

    @staticmethod
    def _is_compact_script_char(char: str) -> bool:
        code = ord(char)
        return (
            0x0E00 <= code <= 0x0EFF
            or 0x1000 <= code <= 0x109F
            or 0x1100 <= code <= 0x11FF
            or 0x1780 <= code <= 0x17FF
            or 0x3040 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xA000 <= code <= 0xA48F
            or 0xAC00 <= code <= 0xD7AF
            or 0xF900 <= code <= 0xFAFF
            or 0x20000 <= code <= 0x2FA1F
        )

    @staticmethod
    def _speech_units_from_text(text: str) -> list[str]:
        units = []
        current = []
        normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()

        def flush_current():
            if current:
                units.append("".join(current))
                current.clear()

        for char in normalized:
            category = unicodedata.category(char)
            if category.startswith("M"):
                if current:
                    current.append(char)
                continue
            if category[0] in ("L", "N"):
                if OmniVoicePipeline._is_compact_script_char(char):
                    flush_current()
                    units.append(char)
                else:
                    current.append(char)
            else:
                flush_current()
        flush_current()
        return units

    @staticmethod
    def _unit_similarity(expected_unit: str, whisper_unit: str) -> float:
        if expected_unit == whisper_unit:
            return 1.0
        if len(expected_unit) < 4 or len(whisper_unit) < 4:
            return 0.0
        return SequenceMatcher(None, expected_unit, whisper_unit, autojunk=False).ratio()

    @classmethod
    def _align_speech_units(cls, expected_units: list[str], whisper_units: list[str]) -> dict[int, int]:
        if not expected_units or not whisper_units:
            return {}
        matches: dict[int, int] = {}
        matcher = SequenceMatcher(None, expected_units, whisper_units, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    matches[i1 + offset] = j1 + offset
            elif tag == "replace":
                next_j = j1
                for expected_index in range(i1, i2):
                    best_j = None
                    best_score = 0.0
                    for whisper_index in range(next_j, j2):
                        score = cls._unit_similarity(expected_units[expected_index], whisper_units[whisper_index])
                        if score > best_score:
                            best_score = score
                            best_j = whisper_index
                    if best_j is not None and best_score >= 0.72:
                        matches[expected_index] = best_j
                        next_j = best_j + 1
        return matches

    @classmethod
    def _flatten_whisper_units(cls, words: list[dict]) -> tuple[list[str], list[int]]:
        units = []
        unit_word_indices = []
        for word_index, word in enumerate(words):
            for unit in cls._speech_units_from_text(str(word.get("word", "") or "")):
                units.append(unit)
                unit_word_indices.append(word_index)
        return units, unit_word_indices

    def _transcribe_generated_segment_words(self, segment_audio: torch.Tensor, offloadobj=None) -> list[dict]:
        whisper_model, whisper_device = self._get_whisper_model(offloadobj)
        waveform = segment_audio.detach().to(device="cpu", dtype=torch.float32).flatten()
        if self.sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, self.sample_rate, 16000)
        result = whisper_model.transcribe(waveform.numpy().astype(np.float32), verbose=None, fp16=self._whisper_uses_fp16(whisper_model, whisper_device), word_timestamps=True)
        words = []
        for item in list(result.get("segments", []) or []):
            for word in list(item.get("words", []) or []):
                if self._count_words(str(word.get("word", "") or "")) > 0:
                    words.append(word)
        return words

    def _detect_trailing_silence_cut_sample(self, segment_audio: torch.Tensor, min_cut_sample: int = 0) -> Optional[int]:
        waveform = segment_audio.detach().to(device="cpu", dtype=torch.float32).flatten()
        sample_count = int(waveform.numel())
        if sample_count == 0:
            return None
        peak = float(waveform.abs().max().item())
        if peak <= OMNIVOICE_TRAILING_SILENCE_MIN_RMS:
            return None
        window = max(1, int(round(self.sample_rate * OMNIVOICE_TRAILING_SILENCE_WINDOW_SECONDS)))
        keep = max(0, int(round(self.sample_rate * OMNIVOICE_TRAILING_SILENCE_KEEP_SECONDS)))
        threshold = max(OMNIVOICE_TRAILING_SILENCE_MIN_RMS, peak * OMNIVOICE_TRAILING_SILENCE_RELATIVE_RMS)
        for end in range(sample_count, 0, -window):
            start = max(0, end - window)
            rms = float(torch.sqrt(torch.mean(waveform[start:end].square())).item())
            if rms > threshold:
                cut_sample = max(min_cut_sample, min(sample_count, end + keep))
                return cut_sample if cut_sample < sample_count else None
        return None

    def _post_process_generated_segment_end(self, segment_audio: torch.Tensor, segment_text: str, segment_index: int, total_segments: int, *, offloadobj=None, verbose_level: int = 0) -> torch.Tensor:
        expected_words = self._words_from_text(segment_text)
        expected_units = self._speech_units_from_text(segment_text)
        words = self._transcribe_generated_segment_words(segment_audio, offloadobj=offloadobj)
        whisper_words = []
        for word in words:
            whisper_words.extend(self._words_from_text(str(word.get("word", "") or "")))
        whisper_units, whisper_unit_word_indices = self._flatten_whisper_units(words)
        alignment = self._align_speech_units(expected_units, whisper_units)
        final_expected_index = len(expected_units) - 1
        final_whisper_unit_index = alignment.get(final_expected_index)
        if int(verbose_level or 0) >= 2:
            print(
                f"OmniVoice auto end trim segment {segment_index + 1}/{total_segments}: "
                f"expected words ({len(expected_words)}) {json.dumps(expected_words, ensure_ascii=False)} | "
                f"Whisper words ({len(whisper_words)}) {json.dumps(whisper_words, ensure_ascii=False)} | "
                f"matched units {len(alignment)}/{len(expected_units)}"
            )
        else:
            print(
                f"OmniVoice auto end trim segment {segment_index + 1}/{total_segments}: "
                f"expected {len(expected_words)} words, Whisper found {len(whisper_words)}, "
                f"matched units {len(alignment)}/{len(expected_units)}"
            )
        cut_candidates = []
        cut_reasons = []
        if final_whisper_unit_index is not None and final_whisper_unit_index < len(whisper_unit_word_indices):
            final_word_index = whisper_unit_word_indices[final_whisper_unit_index]
            for extra_word in words[final_word_index + 1 :]:
                if self._speech_units_from_text(str(extra_word.get("word", "") or "")):
                    extra_start = extra_word.get("start", None)
                    if extra_start is not None:
                        cut_candidates.append(max(0, int(round(float(extra_start) * self.sample_rate))))
                        cut_reasons.append("trailing Whisper words")
                    break

            final_word_end = words[final_word_index].get("end", None)
            if final_word_end is not None:
                min_cut_sample = max(0, int(round((float(final_word_end) + OMNIVOICE_TRAILING_SILENCE_KEEP_SECONDS) * self.sample_rate)))
                silence_cut_sample = self._detect_trailing_silence_cut_sample(segment_audio, min_cut_sample=min_cut_sample)
                if silence_cut_sample is not None:
                    cut_candidates.append(silence_cut_sample)
                    cut_reasons.append("trailing silence")
        elif expected_units:
            last_whisper_unit_index = None
            previous_whisper_unit_index = -1
            for expected_unit_index in range(len(expected_units)):
                aligned_whisper_unit_index = alignment.get(expected_unit_index)
                if aligned_whisper_unit_index is None or aligned_whisper_unit_index <= previous_whisper_unit_index:
                    break
                previous_whisper_unit_index = aligned_whisper_unit_index
                last_whisper_unit_index = aligned_whisper_unit_index
            if last_whisper_unit_index is not None:
                enough_transcribed_units = len(whisper_units) >= max(1, int(len(expected_units) * 0.75))
                if last_whisper_unit_index < len(whisper_unit_word_indices) and len(whisper_units) > len(alignment) and enough_transcribed_units:
                    last_word_index = whisper_unit_word_indices[last_whisper_unit_index]
                    last_word_end = words[last_word_index].get("end", None)
                    if last_word_end is not None:
                        cut_candidates.append(max(0, int(round((float(last_word_end) + OMNIVOICE_TRAILING_SILENCE_KEEP_SECONDS) * self.sample_rate))))
                        cut_reasons.append("partial Whisper alignment")
            if not cut_candidates:
                print(f"OmniVoice auto end trim segment {segment_index + 1}/{total_segments}: final expected unit was not aligned; skipping end trim.")

        if not cut_candidates:
            return segment_audio

        cut_sample = max(0, min(int(segment_audio.shape[-1]), min(cut_candidates)))
        if cut_sample >= int(segment_audio.shape[-1]):
            return segment_audio
        print(
            f"OmniVoice auto end trim segment {segment_index + 1}/{total_segments}: "
            f"cut {segment_audio.shape[-1] / self.sample_rate:.2f}s -> {cut_sample / self.sample_rate:.2f}s "
            f"({', '.join(cut_reasons)})."
        )
        return segment_audio[:cut_sample]

    def _resolve_assets(self, model_weights_path: str) -> _OmniVoiceAssets:
        return _OmniVoiceAssets(
            weights_path=model_weights_path,
            config_path=fl.locate_file(os.path.join(OMNIVOICE_ASSET_DIR, OMNIVOICE_CONFIG_NAME)),
            text_tokenizer_dir=fl.locate_folder(OMNIVOICE_ASSET_DIR),
            audio_tokenizer_dir=fl.locate_folder(OMNIVOICE_AUDIO_TOKENIZER_DIR),
            audio_tokenizer_weights=fl.locate_file(os.path.join(OMNIVOICE_AUDIO_TOKENIZER_DIR, OMNIVOICE_AUDIO_TOKENIZER_WEIGHTS)),
        )

    def _load_main_model(self, assets: _OmniVoiceAssets) -> OmniVoice:
        with open(assets.config_path, "r", encoding="utf-8") as handle:
            config = OmniVoiceConfig(**json.load(handle))
        with init_empty_weights(include_buffers=True):
            model = OmniVoice(config)
        self._refresh_generated_buffers(model)
        offload.load_model_data(model, assets.weights_path, default_dtype=None, writable_tensors=False)
        first_param = next(model.parameters(), None)
        if first_param is not None:
            model._model_dtype = first_param.dtype
        return model

    @staticmethod
    def _refresh_generated_buffers(model: OmniVoice) -> None:
        rotary = getattr(getattr(model, "llm", None), "rotary_emb", None)
        if rotary is not None and getattr(rotary.inv_freq, "device", None).type == "meta":
            inv_freq, rotary.attention_scaling = rotary.rope_init_fn(rotary.config, torch.device("cpu"))
            rotary.register_buffer("inv_freq", inv_freq, persistent=False)
            rotary.original_inv_freq = inv_freq

    def _load_audio_tokenizer(self, assets: _OmniVoiceAssets) -> HiggsAudioV2TokenizerModel:
        config = HiggsAudioV2TokenizerConfig.from_pretrained(assets.audio_tokenizer_dir)
        with init_empty_weights(include_buffers=True):
            tokenizer = HiggsAudioV2TokenizerModel(config)
        offload.load_model_data(tokenizer, assets.audio_tokenizer_weights, default_dtype=None, writable_tensors=False)
        return tokenizer

    @staticmethod
    def _normalize_language(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        if len(value) == 0 or value.lower() in ("auto", "none"):
            return None
        return value

    @staticmethod
    def _normalize_audio_prompt_type(value) -> str:
        raw_mode = str(value or "").strip().upper()
        if "B" in raw_mode:
            return "AB"
        if "A" in raw_mode:
            return "A"
        return ""

    @staticmethod
    def _resolve_auto_split_seconds(custom_settings) -> Optional[float]:
        if not isinstance(custom_settings, dict):
            return None
        raw_value = custom_settings.get(OMNIVOICE_AUTO_SPLIT_SETTING_ID, None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if len(raw_value) == 0:
                return None
        if isinstance(raw_value, bool):
            return None
        value = float(raw_value)
        return value if value > 0 else None

    def _estimate_text_seconds(self, text: str, voice_clone_prompt: Optional[VoiceClonePrompt]) -> float:
        if voice_clone_prompt is None:
            ref_text = None
            num_ref_audio_tokens = None
        else:
            ref_text = voice_clone_prompt.ref_text
            num_ref_audio_tokens = int(voice_clone_prompt.ref_audio_tokens.shape[-1])
        tokens = self.model._estimate_target_tokens(text, ref_text, num_ref_audio_tokens, speed=1.0)
        return float(tokens) / float(self.audio_tokenizer.config.frame_rate)

    @staticmethod
    def _find_auto_split_index(text: str, cut_index: int) -> int:
        safe_cut = min(len(text), max(1, int(cut_index)))
        prefix = text[:safe_cut]
        min_boundary_index = max(0, safe_cut // 2)
        for index in range(len(prefix) - 1, -1, -1):
            if index >= min_boundary_index and prefix[index] == "\n":
                return index + 1
        for index in range(len(prefix) - 1, -1, -1):
            if index >= min_boundary_index and prefix[index] in OMNIVOICE_AUTO_SPLIT_BOUNDARY_PUNCTUATION:
                split_index = index + 1
                while split_index < len(text) and unicodedata.category(text[split_index]).startswith("P"):
                    split_index += 1
                return split_index
        lookahead_end = min(len(text), max(safe_cut + 1, safe_cut + max(1, safe_cut // 2)))
        for index in range(safe_cut, lookahead_end):
            if text[index] in OMNIVOICE_AUTO_SPLIT_BOUNDARY_PUNCTUATION:
                split_index = index + 1
                while split_index < len(text) and unicodedata.category(text[split_index]).startswith("P"):
                    split_index += 1
                return split_index
        for index in range(len(prefix) - 1, -1, -1):
            if index >= min_boundary_index and prefix[index].isspace():
                return index + 1
        return safe_cut

    def _auto_split_text_block(self, text: str, auto_split_seconds: float, voice_clone_prompt: Optional[VoiceClonePrompt]) -> list[str]:
        segments = []
        remaining = text.strip()
        while remaining:
            estimated_seconds = self._estimate_text_seconds(remaining, voice_clone_prompt)
            if estimated_seconds <= auto_split_seconds:
                segments.append(remaining)
                break
            cut_index = max(1, int(len(remaining) * float(auto_split_seconds) / estimated_seconds))
            split_index = self._find_auto_split_index(remaining, cut_index)
            piece = remaining[:split_index].strip()
            if len(piece) == 0:
                split_index = min(len(remaining), max(1, cut_index))
                piece = remaining[:split_index].strip()
            if len(piece) == 0:
                split_index = 1
                piece = remaining[:1]
            segments.append(piece)
            remaining = remaining[split_index:].lstrip()
        return segments

    def _split_text_sequence(self, text: str, auto_split_seconds: Optional[float] = None, voice_clone_prompt: Optional[VoiceClonePrompt] = None) -> list[str]:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\n(?:[ \t]*\n)+", "\n\n", normalized)
        manual_blocks = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
        segments = []
        for block in manual_blocks:
            if auto_split_seconds is not None and auto_split_seconds > 0:
                segments.extend(self._auto_split_text_block(block, auto_split_seconds, voice_clone_prompt))
            else:
                segments.append(block)
        return segments or ([normalized.strip()] if normalized.strip() else [])

    def _parse_two_speaker_dialogue(self, text: str, auto_split_seconds: Optional[float], speaker_prompts: dict[int, Optional[VoiceClonePrompt]]) -> list[tuple[int, str]]:
        speaker_matches = list(re.finditer(r"Speaker\s*(\d+)\s*:\s*", text, flags=re.IGNORECASE))
        if not speaker_matches:
            raise ValueError("Two-speaker mode requires prompt lines using Speaker 1: and Speaker 2:.")
        speaker_ids = sorted({int(match.group(1)) for match in speaker_matches})
        if len(speaker_ids) != 2:
            raise ValueError("Two-speaker mode requires exactly two speaker IDs. Use Speaker 1: and Speaker 2:.")
        speaker_id_to_internal = {speaker_ids[0]: 0, speaker_ids[1]: 1}
        segments: list[tuple[int, str]] = []
        for index, match in enumerate(speaker_matches):
            start = match.end()
            end = speaker_matches[index + 1].start() if index + 1 < len(speaker_matches) else len(text)
            segment_text = text[start:end].strip()
            if segment_text:
                speaker_id = speaker_id_to_internal[int(match.group(1))]
                for split_segment in self._split_text_sequence(segment_text, auto_split_seconds, speaker_prompts.get(speaker_id)):
                    segments.append((speaker_id, split_segment))
        if not segments:
            raise ValueError("No dialogue text found after Speaker tags.")
        return segments

    @staticmethod
    def _resolve_two_speaker_ref_texts(raw_ref_text: str) -> list[str]:
        normalized = str(raw_ref_text or "").replace("\r\n", "\n").replace("\r", "\n")
        matches = list(re.finditer(r"Speaker\s*(\d+)\s*:\s*", normalized, flags=re.IGNORECASE))
        if matches:
            values = {}
            for index, match in enumerate(matches):
                start = match.end()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
                values[int(match.group(1))] = normalized[start:end].strip()
            speaker_ids = sorted(values)
            return [values.get(speaker_ids[0], ""), values.get(speaker_ids[1], "")] if len(speaker_ids) >= 2 else [values.get(speaker_ids[0], ""), ""]
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        return [(lines[0] if len(lines) > 0 else ""), (lines[1] if len(lines) > 1 else "")]

    def _create_voice_clone_prompt(self, audio_path: str, ref_text: Optional[str], generation_config: OmniVoiceGenerationConfig) -> VoiceClonePrompt:
        return self.model.create_voice_clone_prompt(audio_path, ref_text=ref_text, preprocess_prompt=generation_config.preprocess_prompt)

    def _run_segment(
        self,
        *,
        text: str,
        language: Optional[str],
        voice_clone_prompt: Optional[VoiceClonePrompt],
        instruct: Optional[str],
        generation_config: OmniVoiceGenerationConfig,
    ) -> Optional[torch.Tensor]:
        if self._abort_requested() or self._early_stop_requested():
            return None
        try:
            audios = self.model.generate(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                instruct=instruct,
                generation_config=generation_config,
            )
        except RuntimeError as exc:
            if _is_abort_exception(exc):
                return None
            raise
        if self._abort_requested() or not audios:
            return None
        return torch.from_numpy(np.asarray(audios[0], dtype=np.float32)).cpu()

    def _generate_segments(
        self,
        *,
        segments: list[tuple[int, str]],
        speaker_prompts: dict[int, Optional[VoiceClonePrompt]],
        language: Optional[str],
        instruct: Optional[str],
        generation_config: OmniVoiceGenerationConfig,
        pause_seconds: float,
        duration_seconds: Optional[float],
        auto_end_trim: bool,
        callback,
        offloadobj=None,
        verbose_level: int = 0,
    ) -> Optional[dict]:
        max_total_samples = int(round(float(duration_seconds) * self.sample_rate)) if duration_seconds is not None and duration_seconds > 0 else None
        pause_samples = max(0, int(round(float(pause_seconds or 0.0) * self.sample_rate)))
        total_progress_steps = max(1, len(segments) * generation_config.num_step)
        audio_segments = []
        elapsed_samples = 0

        def _poll_early_stop(segment_index: int) -> None:
            if callback is None:
                return
            safe_segment_index = min(segment_index, len(segments))
            progress_offset = safe_segment_index * generation_config.num_step
            callback(
                step_idx=max(-1, progress_offset - 1),
                override_num_inference_steps=total_progress_steps,
                denoising_extra=f"Segment {min(safe_segment_index + 1, len(segments))}/{len(segments)}",
                progress_unit="steps",
            )

        for segment_index, (speaker_id, segment_text) in enumerate(segments):
            _poll_early_stop(segment_index)
            if self._abort_requested() or self._early_stop_requested():
                break
            if max_total_samples is not None and elapsed_samples >= max_total_samples:
                break

            progress_offset = segment_index * generation_config.num_step

            def _segment_callback(step_idx=None, override_num_inference_steps=None, denoising_extra=None, progress_unit=None):
                if callback is None:
                    return
                local_step = -1
                if step_idx is not None:
                    try:
                        local_step = int(step_idx)
                    except Exception:
                        local_step = -1
                global_step = progress_offset + max(0, local_step - 1) if local_step >= 0 else max(-1, progress_offset - 1)
                segment_info = f"Segment {segment_index + 1}/{len(segments)}"
                extra_info = f"{denoising_extra} | {segment_info}" if denoising_extra else segment_info
                callback(
                    step_idx=global_step,
                    override_num_inference_steps=total_progress_steps,
                    denoising_extra=extra_info,
                    progress_unit=progress_unit or "steps",
                )

            self.model._progress_callback = _segment_callback
            segment_audio = self._run_segment(
                text=segment_text,
                language=language,
                voice_clone_prompt=speaker_prompts.get(speaker_id),
                instruct=instruct,
                generation_config=generation_config,
            )
            if segment_audio is None:
                break
            if auto_end_trim:
                segment_audio = self._post_process_generated_segment_end(segment_audio, segment_text, segment_index, len(segments), offloadobj=offloadobj, verbose_level=verbose_level)
            _poll_early_stop(segment_index + 1)
            if max_total_samples is not None:
                samples_left = max_total_samples - elapsed_samples
                if samples_left <= 0:
                    break
                segment_audio = segment_audio[:samples_left]
            if segment_audio.numel() > 0:
                audio_segments.append(segment_audio)
                elapsed_samples += int(segment_audio.shape[-1])
            if self._abort_requested() or self._early_stop_requested():
                break
            if max_total_samples is not None and elapsed_samples >= max_total_samples:
                break
            if pause_samples > 0 and segment_index < len(segments) - 1:
                silence_samples = pause_samples
                if max_total_samples is not None:
                    silence_samples = min(silence_samples, max_total_samples - elapsed_samples)
                if silence_samples > 0:
                    audio_segments.append(torch.zeros((silence_samples,), dtype=torch.float32, device="cpu"))
                    elapsed_samples += silence_samples

        self.model._progress_callback = None
        if not audio_segments:
            return None
        output_audio = torch.cat(audio_segments, dim=-1)
        return {"x": output_audio, "audio_sampling_rate": self.sample_rate}

    def generate(
        self,
        input_prompt: str,
        model_mode: Optional[str] = None,
        audio_guide: Optional[str] = None,
        *,
        alt_prompt: Optional[str] = None,
        image_start=None,
        image_end=None,
        input_frames=None,
        input_frames2=None,
        input_ref_images=None,
        input_ref_masks=None,
        input_masks=None,
        input_masks2=None,
        input_video=None,
        input_faces=None,
        input_custom=None,
        denoising_strength=None,
        masking_strength=None,
        prefix_frames_count=None,
        frame_num=None,
        batch_size=None,
        height=None,
        width=None,
        fit_into_canvas=None,
        shift=None,
        sample_solver=None,
        sampling_steps: int = 32,
        guide_scale: float = 2.0,
        guide2_scale=None,
        guide3_scale=None,
        switch_threshold=None,
        switch2_threshold=None,
        guide_phases=None,
        model_switch_phase=None,
        embedded_guidance_scale=None,
        n_prompt=None,
        seed: int = -1,
        callback=None,
        enable_RIFLEx=None,
        VAE_tile_size=None,
        joint_pass=None,
        perturbation_switch=None,
        perturbation_layers=None,
        perturbation_start=None,
        perturbation_end=None,
        apg_switch=None,
        cfg_star_switch=None,
        cfg_zero_step=None,
        alt_guide_scale=None,
        audio_cfg_scale=None,
        input_waveform=None,
        input_waveform_sample_rate=None,
        audio_guide2: Optional[str] = None,
        audio_prompt_type: str = "",
        audio_proj=None,
        audio_scale=None,
        audio_context_lens=None,
        context_scale=None,
        control_scale_alt=None,
        alt_scale=None,
        motion_amplitude=None,
        causal_block_size=None,
        causal_attention=None,
        fps=None,
        overlapped_latents=None,
        return_latent_slice=None,
        overlap_noise=None,
        overlap_size=None,
        color_correction_strength=None,
        conditioning_latents_size=None,
        input_video_is_hdr=None,
        lora_dir=None,
        keep_frames_parsed=None,
        model_filename=None,
        model_type=None,
        loras_slists=None,
        NAG_scale=None,
        NAG_tau=None,
        NAG_alpha=None,
        speakers_bboxes=None,
        image_mode=None,
        video_prompt_type=None,
        window_no=None,
        offloadobj=None,
        set_header_text=None,
        pre_video_frame=None,
        prefix_video=None,
        original_input_ref_images=None,
        image_refs_relative_size=None,
        outpainting_dims=None,
        face_arc_embeds=None,
        custom_settings=None,
        temperature: float = 0.0,
        window_start_frame_no=None,
        input_video_strength=None,
        self_refiner_setting=None,
        self_refiner_plan=None,
        self_refiner_f_uncertainty=None,
        self_refiner_certain_percentage=None,
        duration_seconds: Optional[float] = None,
        pause_seconds: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 50,
        set_progress_status=None,
        loras_selected=None,
        frames_relative_positions_list=None,
        frames_to_inject=None,
        verbose_level: int = 0,
    ) -> Optional[dict]:
        self._interrupt = False
        self._early_stop = False
        try:
            self._verbose_level = int(verbose_level or 0)
        except (TypeError, ValueError):
            self._verbose_level = 0
        self._active_offloadobj = offloadobj

        text = _read_text_or_file(input_prompt, "Prompt")
        if not text.strip():
            raise ValueError("Prompt text cannot be empty for OmniVoice.")
        if seed is not None and int(seed) >= 0:
            _seed_everything(int(seed))

        mode = self._normalize_audio_prompt_type(audio_prompt_type)
        auto_end_trim = OMNIVOICE_AUTO_END_TRIM_FLAG in str(audio_prompt_type or "").upper()
        auto_split_seconds = self._resolve_auto_split_seconds(custom_settings)
        language = self._normalize_language(model_mode)
        guide_scale = float(guide_scale if guide_scale is not None else 2.0)
        generation_config = OmniVoiceGenerationConfig(
            num_step=max(1, int(sampling_steps or 32)),
            guidance_scale=guide_scale,
            class_temperature=max(0.0, float(temperature or 0.0)),
            audio_chunk_duration=15.0,
            audio_chunk_threshold=30.0,
        )

        try:
            duration_value = float(duration_seconds) if duration_seconds is not None else None
        except (TypeError, ValueError):
            duration_value = None
        if duration_value is not None and duration_value <= 0:
            duration_value = None

        instruction_or_ref = normalize_omnivoice_voice_instruction(_read_text_or_file(alt_prompt, "Voice instruction/reference transcript"))
        voice_clone_instruct = instruction_or_ref.strip() if is_omnivoice_voice_instruction(instruction_or_ref) else None
        if mode == "AB":
            if not audio_guide:
                raise ValueError("Speaker 1 reference audio is required for OmniVoice two-speaker mode.")
            if not audio_guide2:
                raise ValueError("Speaker 2 reference audio is required for OmniVoice two-speaker mode.")
            speaker_ref_texts = [None, None] if voice_clone_instruct else [text.strip() or None for text in self._resolve_two_speaker_ref_texts(instruction_or_ref)]
            try:
                speaker_prompts = {
                    0: self._create_voice_clone_prompt(audio_guide, speaker_ref_texts[0], generation_config),
                    1: self._create_voice_clone_prompt(audio_guide2, speaker_ref_texts[1], generation_config),
                }
            finally:
                self._release_whisper_model()
            segments = self._parse_two_speaker_dialogue(text, auto_split_seconds, speaker_prompts)
            return self._generate_segments(
                segments=segments,
                speaker_prompts=speaker_prompts,
                language=language,
                instruct=voice_clone_instruct,
                generation_config=generation_config,
                pause_seconds=pause_seconds,
                duration_seconds=duration_value,
                auto_end_trim=auto_end_trim,
                callback=callback,
                offloadobj=offloadobj,
                verbose_level=self._verbose_level,
            )

        if mode == "A":
            if not audio_guide:
                raise ValueError("Reference audio is required for OmniVoice voice cloning mode.")
            ref_text = None if voice_clone_instruct else instruction_or_ref.strip() or None
            try:
                speaker_prompts = {0: self._create_voice_clone_prompt(audio_guide, ref_text, generation_config)}
            finally:
                self._release_whisper_model()
            instruct = voice_clone_instruct
        else:
            speaker_prompts = {0: None}
            instruct = instruction_or_ref.strip() or None

        segments = [(0, segment) for segment in self._split_text_sequence(text, auto_split_seconds, speaker_prompts.get(0))]
        return self._generate_segments(
            segments=segments,
            speaker_prompts=speaker_prompts,
            language=language,
            instruct=instruct,
            generation_config=generation_config,
            pause_seconds=pause_seconds,
            duration_seconds=duration_value,
            auto_end_trim=auto_end_trim,
            callback=callback,
            offloadobj=offloadobj,
            verbose_level=self._verbose_level,
        )
