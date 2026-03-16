from __future__ import annotations

import math
import os
import random
import re
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from shared.utils import files_locator as fl

from .infer_v2 import IndexTTS2


_ABORT_ERROR = "INDEXTTS2_ABORT"
_BIGVGAN_FOLDER = "bigvgan_v2_22khz_80band_256x"
_BIGVGAN_FILES = ("config.json", "bigvgan_generator.pt")
_QWEN_EMO_FOLDER = "qwen0.6bemo4-merge"
_W2V_BERT_FOLDER = "w2v-bert-2.0"
_CONFIGS_SUBDIR = "configs"
_BASE_CFG_NAME = "config.yaml"
_RUNTIME_CFG_NAME = "config_runtime.yaml"
_SOURCE_CONFIGS_DIR = Path(__file__).resolve().parent / _CONFIGS_SUBDIR
_AUTO_SPLIT_SETTING_ID = "auto_split_every_s"
_AUTO_SPLIT_TOKENS_PER_SECOND = 6.0
_MEL_TOKENS_PER_SOUND_TOKEN = 1.72
_MIN_CG_SOUND_SECONDS = 10.0
_CG_SOUND_TOKENS_PER_TEXT_TOKEN = 29.67
# Local dev toggle: set to True to simulate FlashAttention2 unavailable.
_FORCE_NO_FLASH2 = False


def _read_text_or_file(value: Optional[str], label: str) -> str:
    if value is None:
        return ""
    text = str(value)
    if os.path.isfile(text):
        with open(text, "r", encoding="utf-8") as handle:
            return handle.read()
    if "\r\n" in text:
        return text.replace("\r\n", "\n")
    return text


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class IndexTTS2Pipeline:
    def __init__(
        self,
        ckpt_root: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        gpt_weights_path: Optional[str | Path] = None,
        show_load_logs: bool = False,
        lm_decoder_engine: Optional[str] = None,
    ) -> None:
        self._interrupt = False
        self._early_stop = False
        self.device = device or torch.device("cpu")
        self.show_load_logs = bool(show_load_logs)
        self.lm_decoder_engine = str(lm_decoder_engine or "legacy").strip().lower()
        if self.lm_decoder_engine == "cudagraph":
            self.lm_decoder_engine = "cg"
        if self.lm_decoder_engine not in ("legacy", "cg", "vllm"):
            self.lm_decoder_engine = "legacy"
        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else Path(fl.get_download_location())
        self.gpt_weights_path = Path(gpt_weights_path) if gpt_weights_path is not None else None
        self.model_dir = self._resolve_model_dir()
        cfg_path = self._resolve_config_path()
        runtime_cfg_path = self._build_runtime_cfg(cfg_path)
        use_accel_engine = self.lm_decoder_engine in ("cg", "vllm")
        allow_vllm_kernels = self.lm_decoder_engine == "vllm"

        device_str = str(self.device)
        if device_str == "cuda":
            device_str = "cuda:0"
        self.model = IndexTTS2(
            cfg_path=str(runtime_cfg_path),
            model_dir=str(self.model_dir),
            use_fp16=False,
            device=device_str,
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=use_accel_engine,
            use_torch_compile=False,
            show_load_logs=self.show_load_logs,
            lm_decoder_engine=self.lm_decoder_engine,
            force_no_flash2=bool(_FORCE_NO_FLASH2 or (not allow_vllm_kernels)),
            accel_allow_vllm_kernels=allow_vllm_kernels,
        )
        engine = self.lm_decoder_engine
        flash_status = "n/a"
        backend = "n/a"
        if engine in ("cg", "vllm"):
            if getattr(self.model.gpt, "accel_engine", None) is None:
                flash_status = "disabled"
            else:
                flash_status = "on" if getattr(self.model.gpt, "accel_flash2_available", False) else "off(sdpa)"
                backend = str(getattr(self.model.gpt, "accel_kernel_mode", "sdpa"))
        print(f"[IndexTTS2] LM Engine='{engine}', flash2={flash_status}, backend={backend}")
        self.sample_rate = 22050
        self._mark_model_dtypes()

    def _resolve_config_path(self) -> Path:
        cfg_path = _SOURCE_CONFIGS_DIR / _BASE_CFG_NAME
        if cfg_path.is_file():
            return cfg_path
        raise FileNotFoundError(
            f"IndexTTS2 source config not found: {cfg_path}. "
            f"Expected configs bundled with project source in '{_SOURCE_CONFIGS_DIR}'."
        )

    def _resolve_model_dir(self) -> Path:
        located = fl.locate_folder("index_tts2", error_if_none=False)
        if located is not None:
            return Path(located)
        fallback = self.ckpt_root / "index_tts2"
        if fallback.is_dir():
            return fallback
        raise FileNotFoundError(
            "IndexTTS2 checkpoint folder not found. Expected 'index_tts2' under the WanGP checkpoints root."
        )

    def _resolve_bigvgan_dir(self) -> Path:
        located = fl.locate_folder(_BIGVGAN_FOLDER, error_if_none=False)
        if located is None:
            local_fallback = self.model_dir / _BIGVGAN_FOLDER
            if local_fallback.is_dir():
                located = str(local_fallback)
        if located is None:
            raise FileNotFoundError(
                f"IndexTTS2 BigVGAN folder '{_BIGVGAN_FOLDER}' is missing. "
                "WanGP must download BigVGAN before model load."
            )
        bigvgan_dir = Path(located)
        missing_files = [name for name in _BIGVGAN_FILES if not (bigvgan_dir / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"IndexTTS2 BigVGAN files missing in '{bigvgan_dir}': {', '.join(missing_files)}"
            )
        return bigvgan_dir

    def _resolve_qwen_emo_dir(self) -> Path:
        located = fl.locate_folder(_QWEN_EMO_FOLDER, error_if_none=False)
        if located is None:
            local_fallback = self.ckpt_root / _QWEN_EMO_FOLDER
            if local_fallback.is_dir():
                located = str(local_fallback)
        if located is None:
            raise FileNotFoundError(
                f"IndexTTS2 Qwen emotion folder '{_QWEN_EMO_FOLDER}' is missing in checkpoints root."
            )
        qwen_dir = Path(located)
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing_files = [name for name in required if not (qwen_dir / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"IndexTTS2 Qwen emotion files missing in '{qwen_dir}': {', '.join(missing_files)}"
            )
        return qwen_dir

    def _resolve_w2v_bert_dir(self) -> Path:
        located = fl.locate_folder(_W2V_BERT_FOLDER, error_if_none=False)
        if located is None:
            local_fallback = self.ckpt_root / _W2V_BERT_FOLDER
            if local_fallback.is_dir():
                located = str(local_fallback)
        if located is None:
            raise FileNotFoundError(
                f"IndexTTS2 semantic folder '{_W2V_BERT_FOLDER}' is missing in checkpoints root."
            )
        w2v_dir = Path(located)
        required = ("config.json", "preprocessor_config.json")
        missing_files = [name for name in required if not (w2v_dir / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"IndexTTS2 semantic files missing in '{w2v_dir}': {', '.join(missing_files)}"
            )
        if not (w2v_dir / "model.safetensors").is_file() and not (w2v_dir / "model_fp16.safetensors").is_file():
            raise FileNotFoundError(
                f"IndexTTS2 semantic weights missing in '{w2v_dir}'. "
                "Expected model.safetensors and/or model_fp16.safetensors."
            )
        return w2v_dir

    def _build_runtime_cfg(self, cfg_path: Path) -> Path:
        cfg = OmegaConf.load(str(cfg_path))
        if cfg.get("vocoder") is None:
            raise ValueError("IndexTTS2 config is missing the 'vocoder' section.")
        if self.gpt_weights_path is not None and self.gpt_weights_path.is_file():
            cfg.gpt_checkpoint = str(self.gpt_weights_path.resolve())
        else:
            gpt_safetensor = self.model_dir / "gpt.safetensors"
            if gpt_safetensor.is_file():
                cfg.gpt_checkpoint = gpt_safetensor.name
        s2mel_safetensor = self.model_dir / "s2mel.safetensors"
        if s2mel_safetensor.is_file():
            cfg.s2mel_checkpoint = s2mel_safetensor.name
        cfg.vocoder.name = str(self._resolve_bigvgan_dir())
        cfg.qwen_emo_path = str(self._resolve_qwen_emo_dir().resolve())
        cfg.w2v_bert_path = str(self._resolve_w2v_bert_dir().resolve())
        runtime_cfg_path = self.model_dir / _CONFIGS_SUBDIR / _RUNTIME_CFG_NAME
        runtime_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=cfg, f=str(runtime_cfg_path))
        return runtime_cfg_path

    def _mark_model_dtypes(self) -> None:
        modules = [
            getattr(self.model, "gpt", None),
            getattr(self.model, "s2mel", None),
            getattr(self.model, "bigvgan", None),
            getattr(self.model, "semantic_model", None),
            getattr(self.model, "semantic_codec", None),
            getattr(self.model, "campplus_model", None),
            getattr(getattr(self.model, "qwen_emo", None), "model", None),
        ]
        for module in modules:
            if module is None:
                continue
            first_param = next(module.parameters(), None)
            if first_param is not None:
                module._model_dtype = first_param.dtype

    def _abort_requested(self) -> bool:
        return bool(self._interrupt)

    def _early_stop_requested(self) -> bool:
        return bool(self._early_stop)

    def request_early_stop(self) -> None:
        self._early_stop = True

    def _build_progress_callback(self, callback, total_steps: int):
        def _progress(value, desc):
            if self._abort_requested() or self._early_stop_requested():
                raise RuntimeError(_ABORT_ERROR)
            if callback is None:
                return
            try:
                fraction = float(value)
            except Exception:
                fraction = 0.0
            fraction = min(1.0, max(0.0, fraction))
            step_idx = int(round(fraction * max(total_steps - 1, 0)))
            callback(
                step_idx=step_idx,
                override_num_inference_steps=total_steps,
                denoising_extra=str(desc or "Synthesizing speech"),
                progress_unit="segments",
                read_state=True,
            )

        return _progress

    def _seed_everything(self, seed: Optional[int]) -> None:
        if seed is None or seed < 0:
            return
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_auto_split_seconds(self, kwargs: dict) -> Optional[float]:
        custom_settings = kwargs.get("custom_settings", None)
        if not isinstance(custom_settings, dict):
            return None
        raw_value = custom_settings.get(_AUTO_SPLIT_SETTING_ID, None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if len(raw_value) == 0:
                return None
        try:
            if isinstance(raw_value, bool):
                return None
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _resolve_cut_char_index(self, text: str, token_limit: Optional[int]) -> Optional[int]:
        if token_limit is None or token_limit <= 0 or len(text) == 0:
            return None
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "tokenize"):
            return None

        def _count_tokens(prefix: str) -> int:
            try:
                return len(tokenizer.tokenize(prefix))
            except Exception:
                return len(prefix.split())

        if _count_tokens(text) <= token_limit:
            return None
        low, high = 1, len(text)
        best = high
        while low <= high:
            mid = (low + high) // 2
            if _count_tokens(text[:mid]) > token_limit:
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        return min(len(text), max(1, best))

    @staticmethod
    def _find_split_index_before_cut(text: str, cut_index: int) -> int:
        safe_cut = min(len(text), max(1, int(cut_index)))
        prefix = text[:safe_cut]
        punct_idx = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"), prefix.rfind(";"), prefix.rfind("\n"))
        if punct_idx >= 0:
            return punct_idx + 1
        space_idx = prefix.rfind(" ")
        if space_idx >= 0:
            return space_idx + 1
        return safe_cut

    def _split_text_sequence(self, text: str, auto_split_tokens: Optional[int]) -> list[str]:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\n(?:[ \t]*\n)+", "\n\n", normalized)
        manual_blocks = re.split(r"\n\s*\n", normalized)
        segments = []
        for block in manual_blocks:
            remaining = block.strip()
            if len(remaining) == 0:
                continue
            if auto_split_tokens is None or auto_split_tokens <= 0:
                segments.append(remaining)
                continue
            while len(remaining) > 0:
                cut_index = self._resolve_cut_char_index(remaining, auto_split_tokens)
                if cut_index is None:
                    segments.append(remaining.strip())
                    break
                split_index = self._find_split_index_before_cut(remaining, cut_index)
                if split_index <= 0:
                    split_index = min(len(remaining), max(1, cut_index))
                piece = remaining[:split_index].strip()
                if len(piece) == 0:
                    split_index = min(len(remaining), max(1, cut_index))
                    piece = remaining[:split_index].strip()
                if len(piece) == 0:
                    split_index = 1
                    piece = remaining[:1]
                segments.append(piece)
                remaining = remaining[split_index:].lstrip()
        if len(segments) == 0 and len(normalized.strip()) > 0:
            segments.append(normalized.strip())
        return segments

    def _compute_shared_cg_settings(self, segments: list[dict], duration_seconds: float, max_mel_tokens: int, max_text_tokens_per_segment: int) -> dict:
        if not segments:
            return {}
        tokenizer = getattr(self.model, "tokenizer", None)
        token_counts = []
        for one_segment in segments:
            text = str(one_segment.get("text", "") or "").strip()
            if len(text) == 0:
                continue
            if tokenizer is not None and hasattr(tokenizer, "tokenize") and hasattr(tokenizer, "convert_tokens_to_ids"):
                try:
                    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    token_counts.append(max(1, len(tokens)))
                    continue
                except Exception:
                    pass
            token_counts.append(max(1, len(text.split())))
        if len(token_counts) == 0:
            token_counts = [max(1, int(max_text_tokens_per_segment))]
        max_segment_text_tokens = max(1, max(token_counts))
        try:
            mel_sr = float(self.model.cfg.s2mel["preprocess_params"]["sr"])
            mel_hop = float(self.model.cfg.s2mel["preprocess_params"]["spect_params"]["hop_length"])
        except Exception:
            mel_sr = 22050.0
            mel_hop = 256.0
        sound_tokens_per_second = max(1.0, (mel_sr / max(1.0, mel_hop)) / _MEL_TOKENS_PER_SOUND_TOKEN)
        min_cg_sound_tokens = int(math.ceil(_MIN_CG_SOUND_SECONDS * sound_tokens_per_second))
        duration_sound_tokens = int(math.ceil(duration_seconds * sound_tokens_per_second)) if duration_seconds > 0 else int(max_mel_tokens)
        longest_segment_estimated_sound_tokens = int(math.ceil(max_segment_text_tokens * _CG_SOUND_TOKENS_PER_TEXT_TOKEN))
        capped_segment_sound_tokens = max(1, min(int(duration_sound_tokens), int(longest_segment_estimated_sound_tokens)))
        segment_generation_tokens = max(min_cg_sound_tokens, capped_segment_sound_tokens)
        segment_generation_tokens = max(1, min(int(max_mel_tokens), int(segment_generation_tokens)))
        prompt_budget = int(getattr(self.model.gpt, "cond_num", 32) + max_segment_text_tokens + 3)
        cg_max_total_tokens = int(prompt_budget + max(1, int(segment_generation_tokens)))
        return {"cg_max_total_tokens": cg_max_total_tokens, "cg_segment_generation_tokens": int(segment_generation_tokens)}

    def _parse_segment_plan(
        self,
        text: str,
        *,
        two_speaker: bool,
        default_emotion: str,
        auto_split_tokens: Optional[int],
    ) -> list[dict]:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\n(?:[ \t]*\n)+", "\n\n", normalized)
        blocks = re.split(r"\n\s*\n", normalized)
        speaker_pattern = re.compile(r"Speaker\s*(\d+)\s*:\s*", flags=re.IGNORECASE)
        raw_chunks: list[tuple[int, str]] = []
        if two_speaker:
            for block in blocks:
                block_text = str(block or "").strip()
                if not block_text:
                    continue
                matches = list(speaker_pattern.finditer(block_text))
                if not matches:
                    raise ValueError(
                        "Two-speaker mode requires each dialogue block to include "
                        "Speaker 1: and Speaker 2: (or any two numeric speaker IDs)."
                    )
                prefix = block_text[: matches[0].start()].strip()
                if prefix:
                    raise ValueError(
                        "Two-speaker mode requires each dialogue block to start with "
                        "Speaker 1: and Speaker 2: (or any two numeric speaker IDs)."
                    )
                for idx, match in enumerate(matches):
                    start = match.end()
                    end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block_text)
                    chunk_text = block_text[start:end].strip()
                    if chunk_text:
                        raw_chunks.append((int(match.group(1)), chunk_text))
            if not raw_chunks:
                raise ValueError(
                    "Two-speaker mode requires prompt lines using Speaker 1: and Speaker 2: "
                    "(or any two numeric speaker IDs)."
                )
            speaker_ids = sorted({speaker_id for speaker_id, _ in raw_chunks})
            if len(speaker_ids) != 2:
                raise ValueError("Two-speaker mode requires exactly two speaker IDs. Use Speaker 1: and Speaker 2:.")
            speaker_id_to_slot = {speaker_ids[0]: 1, speaker_ids[1]: 2}
        else:
            speaker_id_to_slot = {1: 1}
            for block in blocks:
                block_text = block.strip()
                if block_text:
                    raw_chunks.append((1, block_text))
            if not raw_chunks:
                raise ValueError("Prompt text cannot be empty for IndexTTS2.")

        # Allow rich free-form emotion descriptions between brackets.
        emotion_pattern = re.compile(r"\[(.+?)\]", flags=re.DOTALL)
        segments: list[dict] = []
        for raw_speaker_id, chunk_text in raw_chunks:
            speaker_slot = speaker_id_to_slot.get(raw_speaker_id, 1)
            current_emotion = default_emotion
            cursor = 0
            for match in emotion_pattern.finditer(chunk_text):
                text_piece = chunk_text[cursor:match.start()].strip()
                if text_piece:
                    for split_piece in self._split_text_sequence(text_piece, auto_split_tokens):
                        split_piece = split_piece.strip()
                        if split_piece:
                            segments.append({"speaker": speaker_slot, "emotion": current_emotion, "text": split_piece})
                parsed_emotion = re.sub(r"\s+", " ", match.group(1)).strip()
                current_emotion = parsed_emotion if parsed_emotion else default_emotion
                cursor = match.end()
            tail_text = chunk_text[cursor:].strip()
            if tail_text:
                for split_piece in self._split_text_sequence(tail_text, auto_split_tokens):
                    split_piece = split_piece.strip()
                    if split_piece:
                        segments.append({"speaker": speaker_slot, "emotion": current_emotion, "text": split_piece})
        if not segments:
            raise ValueError("No dialogue text found after Speaker tags.")
        return segments

    @staticmethod
    def _to_wav_tensor(output):
        if output is None:
            return None, None
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("IndexTTS2 inference returned an unexpected output format.")
        sample_rate, wav_data = output
        wav = torch.as_tensor(np.asarray(wav_data))
        if wav.ndim == 2:
            if wav.shape[1] == 1:
                wav = wav[:, 0]
            elif wav.shape[0] == 1:
                wav = wav[0]
            else:
                wav = wav.mean(dim=1)
        wav = wav.to(torch.float32).contiguous()
        if wav.numel() > 0 and float(wav.abs().max().item()) > 1.5:
            wav = wav / 32767.0
        return _to_int(sample_rate, 22050), wav

    def generate(
        self,
        input_prompt: str,
        model_mode: Optional[str],
        audio_guide: Optional[str],
        *,
        alt_prompt: Optional[str] = None,
        audio_guide2: Optional[str] = None,
        audio_prompt_type: str = "A",
        duration_seconds: float = 0.0,
        pause_seconds: float = 0.0,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        set_progress_status=None,
        callback=None,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False
        self.model.abort_checker = lambda: self._abort_requested() or self._early_stop_requested()

        text = _read_text_or_file(input_prompt, "Prompt").strip()
        if len(text) == 0:
            raise ValueError("Prompt text cannot be empty for IndexTTS2.")

        if audio_guide is None or not os.path.isfile(str(audio_guide)):
            raise ValueError("IndexTTS2 requires one reference audio file.")

        raw_audio_prompt_type = str(audio_prompt_type or "A").upper()
        if "2" in raw_audio_prompt_type:
            audio_prompt_type = "2"
        elif "B" in raw_audio_prompt_type:
            audio_prompt_type = "AB"
        elif "A" in raw_audio_prompt_type:
            audio_prompt_type = "A"
        else:
            audio_prompt_type = "A"

        seed = kwargs.get("seed", None)
        seed = _to_int(seed, -1) if seed is not None else None
        self._seed_everything(seed)

        duration_seconds = _to_float(duration_seconds, 0.0)
        duration_seconds = max(0.0, duration_seconds)
        top_p = _to_float(top_p, 0.8)
        top_k = max(1, _to_int(top_k, 30))
        temperature = _to_float(temperature, 0.8)
        pause_seconds = _to_float(pause_seconds, 0.0)
        pause_seconds = max(0.0, min(10.0, pause_seconds))
        pause_transition_ms = int(round(pause_seconds * 1000.0))
        last_progress_status = None

        def _set_progress_phase(phase_text: str):
            nonlocal last_progress_status
            if not callable(set_progress_status):
                return
            phase = str(phase_text or "").strip()
            if len(phase) == 0 or phase == last_progress_status:
                return
            set_progress_status(phase)
            last_progress_status = phase

        max_text_tokens_per_segment = max(20, _to_int(kwargs.get("max_text_tokens_per_segment", 120), 120))
        requested_max_mel_tokens = max(0, _to_int(kwargs.get("max_mel_tokens", 0), 0))
        try:
            mel_sr = float(self.model.cfg.s2mel["preprocess_params"]["sr"])
            mel_hop = float(self.model.cfg.s2mel["preprocess_params"]["spect_params"]["hop_length"])
        except Exception:
            mel_sr = 22050.0
            mel_hop = 256.0
        mel_tokens_per_second = mel_sr / max(1.0, mel_hop)
        if requested_max_mel_tokens > 0:
            global_max_mel_tokens = requested_max_mel_tokens
        elif duration_seconds > 0:
            # Keep extra headroom so GPT can naturally emit stop tokens before final hard crop.
            global_max_mel_tokens = int(math.ceil(duration_seconds * mel_tokens_per_second * 1.75) + 256)
        else:
            # No explicit duration: use a generous budget tied to split limit.
            global_max_mel_tokens = int(math.ceil(max_text_tokens_per_segment * 24.0))
        global_max_mel_tokens = max(1500, min(12000, global_max_mel_tokens))
        auto_split_seconds = self._resolve_auto_split_seconds(kwargs)
        auto_split_tokens = (
            max(1, int(round(auto_split_seconds * _AUTO_SPLIT_TOKENS_PER_SECOND)))
            if auto_split_seconds is not None
            else None
        )
        if auto_split_tokens is not None and auto_split_tokens > 0:
            max_text_tokens_per_segment = max(20, min(max_text_tokens_per_segment, auto_split_tokens))
        default_emotion = _read_text_or_file(alt_prompt, "Default Emotion Instruction").strip()
        auto_sentence_emotion_mode = len(default_emotion) == 0 and audio_prompt_type != "AB"

        def _explicit_segment_emotion(segment):
            return str(segment.get("emotion", "") or "").strip()

        def _effective_segment_emotion_text(segment):
            explicit = _explicit_segment_emotion(segment)
            if len(explicit) > 0:
                return explicit
            if auto_sentence_emotion_mode:
                return str(segment.get("text", "") or "").strip()
            return ""

        def _emotion_progress_label(segment):
            explicit = _explicit_segment_emotion(segment)
            if len(explicit) > 0:
                return explicit
            if auto_sentence_emotion_mode:
                return "auto(text)"
            return "default"

        def _emotion_key(value):
            return re.sub(r"\s+", " ", str(value or "")).strip().lower()

        def _pause_after_ms(current_segment, next_segment):
            if pause_transition_ms <= 0 or next_segment is None:
                return 0
            curr_speaker = int(current_segment.get("speaker", 1))
            next_speaker = int(next_segment.get("speaker", 1))
            curr_emo = _emotion_key(current_segment.get("emotion", ""))
            next_emo = _emotion_key(next_segment.get("emotion", ""))
            return pause_transition_ms if (curr_speaker != next_speaker or curr_emo != next_emo) else 0

        if audio_prompt_type == "2":
            if audio_guide2 is None or not os.path.isfile(str(audio_guide2)):
                raise ValueError("Two-speaker mode requires a second speaker reference audio file.")
            segments = self._parse_segment_plan(
                text,
                two_speaker=True,
                default_emotion=default_emotion,
                auto_split_tokens=auto_split_tokens,
            )
            cg_shared_kwargs = self._compute_shared_cg_settings(
                segments,
                duration_seconds=duration_seconds,
                max_mel_tokens=global_max_mel_tokens,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
            )
            unique_emo_texts = sorted({
                one_text for one_text in (_effective_segment_emotion_text(seg) for seg in segments) if len(one_text) > 0
            })
            try:
                if hasattr(self.model, "precache_emotion_texts") and unique_emo_texts:
                    self.model.precache_emotion_texts(unique_emo_texts)
                if hasattr(self.model, "precache_reference_audio"):
                    self.model.precache_reference_audio([str(audio_guide), str(audio_guide2)], verbose=False)
                if hasattr(self.model, "precache_emotion_audio"):
                    self.model.precache_emotion_audio([str(audio_guide), str(audio_guide2)], verbose=False)
            except RuntimeError as exc:
                if _ABORT_ERROR in str(exc) or self._abort_requested() or self._early_stop_requested():
                    self.model._clear_persistent_generation_cache()
                    self.model._release_runtime_gpu_caches()
                    return None
                raise

            total_steps = max(1, len(segments))
            max_samples = int(round(22050 * duration_seconds)) if duration_seconds > 0 else 0
            estimated_total_samples = 0
            reached_duration_limit = False
            collected_segment_payloads = []
            try:
                for seg_idx, segment in enumerate(segments):
                    if self._abort_requested():
                        return None
                    if self._early_stop_requested():
                        if collected_segment_payloads:
                            break
                        return None

                    speaker_slot = int(segment["speaker"])
                    speaker_audio = str(audio_guide) if speaker_slot == 1 else str(audio_guide2)
                    segment_emo_text = _effective_segment_emotion_text(segment)
                    use_emo_text = len(segment_emo_text) > 0
                    segment_text = str(segment["text"]).strip()
                    if not segment_text:
                        continue

                    def _segment_progress(value, desc, _seg_idx=seg_idx, _speaker=speaker_slot, _emotion_label=_emotion_progress_label(segment)):
                        if self._abort_requested() or self._early_stop_requested():
                            raise RuntimeError(_ABORT_ERROR)
                        extra = f"{str(desc or 'Synthesizing speech')} | Segment {_seg_idx + 1}/{len(segments)} | Speaker {_speaker} | Emotion: {_emotion_label}"
                        _set_progress_phase(extra)
                        if callback is None:
                            return
                        step_idx = min(max(0, int(_seg_idx)), max(total_steps - 1, 0))
                        callback(
                            step_idx=step_idx,
                            override_num_inference_steps=total_steps,
                            denoising_extra=extra,
                            progress_unit="segments",
                            read_state=True,
                        )

                    self.model.gr_progress = _segment_progress
                    _segment_progress(0, "Synthesizing speech")
                    segment_output = self.model.infer(
                        spk_audio_prompt=speaker_audio,
                        text=segment_text,
                        output_path=None,
                        emo_audio_prompt=None,
                        use_emo_text=use_emo_text,
                        emo_text=segment_emo_text if use_emo_text else None,
                        interval_silence=0,
                        verbose=False,
                        max_text_tokens_per_segment=max_text_tokens_per_segment,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        max_mel_tokens=global_max_mel_tokens,
                        duration_seconds=duration_seconds,
                        cg_max_total_tokens=cg_shared_kwargs.get("cg_max_total_tokens"),
                        cg_segment_generation_tokens=cg_shared_kwargs.get("cg_segment_generation_tokens"),
                        clear_cache_after=False,
                        defer_s2mel=True,
                        defer_bigvgan=True,
                    )
                    if segment_output is None:
                        if self._abort_requested():
                            return None
                        if self._early_stop_requested() and collected_segment_payloads:
                            break
                        return None
                    if not isinstance(segment_output, dict) or "deferred_segment_payloads" not in segment_output:
                        raise RuntimeError("IndexTTS2 deferred generation returned an unexpected output format.")
                    seg_payloads = list(segment_output["deferred_segment_payloads"] or [])
                    segment_output["deferred_segment_payloads"] = []
                    if len(seg_payloads) == 0:
                        del seg_payloads, segment_output
                        gc.collect()
                        continue
                    appended_payload_count = 0
                    next_segment = segments[seg_idx + 1] if (seg_idx + 1) < len(segments) else None
                    pause_after_ms = _pause_after_ms(segment, next_segment)
                    for payload in seg_payloads:
                        est_samples = _to_int(payload.get("estimated_samples", 0), 0) if isinstance(payload, dict) else 0
                        if max_samples > 0 and est_samples > 0 and estimated_total_samples >= max_samples and collected_segment_payloads:
                            reached_duration_limit = True
                            break
                        if max_samples > 0 and est_samples > 0 and (estimated_total_samples + est_samples) > max_samples and collected_segment_payloads:
                            reached_duration_limit = True
                            break
                        collected_segment_payloads.append({"segment_payload": payload, "pause_after_ms": 0})
                        appended_payload_count += 1
                        if est_samples > 0:
                            estimated_total_samples += est_samples
                    if appended_payload_count > 0 and not reached_duration_limit:
                        collected_segment_payloads[-1]["pause_after_ms"] = pause_after_ms
                    seg_payloads.clear()
                    del seg_payloads, segment_output
                    gc.collect()
                    if reached_duration_limit:
                        break
            except RuntimeError as exc:
                if _ABORT_ERROR in str(exc) or self._abort_requested() or self._early_stop_requested():
                    if self._abort_requested():
                        return None
                    if self._early_stop_requested() and collected_segment_payloads:
                        pass
                    else:
                        return None
                else:
                    raise
            finally:
                self.model.gr_progress = None
                self.model._clear_persistent_generation_cache()
                self.model._release_runtime_gpu_caches()

            if self._abort_requested():
                return None
            if not collected_segment_payloads:
                return None
            previous_abort_checker = getattr(self.model, "abort_checker", None)
            self.model.abort_checker = lambda: self._abort_requested()
            try:
                _set_progress_phase("Preparing vocoder conditioning")
                collected_vc_targets = self.model.synthesize_from_segment_payloads(collected_segment_payloads, verbose=False)
                collected_segment_payloads.clear()
                del collected_segment_payloads
                gc.collect()
                self.model._release_runtime_gpu_caches()
                _set_progress_phase("Generating waveform")
                output = self.model.synthesize_from_vc_targets(collected_vc_targets, interval_silence=0, verbose=False)
                if isinstance(collected_vc_targets, list):
                    collected_vc_targets.clear()
                del collected_vc_targets
                gc.collect()
                self.model._release_runtime_gpu_caches()
            except RuntimeError as exc:
                if _ABORT_ERROR in str(exc) or self._abort_requested():
                    return None
                raise
            finally:
                self.model.abort_checker = previous_abort_checker
            sample_rate, wav = self._to_wav_tensor(output)
            if wav is None or wav.numel() == 0:
                return None
            if duration_seconds > 0:
                max_samples_out = int(round(sample_rate * duration_seconds))
                if max_samples_out > 0 and wav.shape[-1] > max_samples_out:
                    wav = wav[:max_samples_out]
            self.model._release_runtime_gpu_caches()
            return {"x": wav, "audio_sampling_rate": int(sample_rate)}

        emo_audio_prompt = None
        if audio_prompt_type == "AB":
            if audio_guide2 is None or not os.path.isfile(str(audio_guide2)):
                raise ValueError("IndexTTS2 emotion-reference mode requires a second reference audio file.")
            emo_audio_prompt = str(audio_guide2)

        segments = self._parse_segment_plan(
            text,
            two_speaker=False,
            default_emotion=default_emotion,
            auto_split_tokens=auto_split_tokens,
        )
        cg_shared_kwargs = self._compute_shared_cg_settings(
            segments,
            duration_seconds=duration_seconds,
            max_mel_tokens=global_max_mel_tokens,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
        )
        unique_emo_texts = sorted({
            one_text for one_text in (_effective_segment_emotion_text(seg) for seg in segments) if len(one_text) > 0
        })
        try:
            if hasattr(self.model, "precache_emotion_texts") and unique_emo_texts:
                self.model.precache_emotion_texts(unique_emo_texts)
            if hasattr(self.model, "precache_reference_audio"):
                self.model.precache_reference_audio([str(audio_guide)], verbose=False)
            if hasattr(self.model, "precache_emotion_audio"):
                emotion_refs = [str(audio_guide)]
                if emo_audio_prompt is not None:
                    emotion_refs.append(str(emo_audio_prompt))
                self.model.precache_emotion_audio(emotion_refs, verbose=False)
        except RuntimeError as exc:
            if _ABORT_ERROR in str(exc) or self._abort_requested() or self._early_stop_requested():
                self.model._clear_persistent_generation_cache()
                self.model._release_runtime_gpu_caches()
                return None
            raise

        total_steps = max(1, len(segments))
        max_samples = int(round(22050 * duration_seconds)) if duration_seconds > 0 else 0
        estimated_total_samples = 0
        reached_duration_limit = False
        collected_segment_payloads = []
        try:
            for seg_idx, segment in enumerate(segments):
                if self._abort_requested():
                    return None
                if self._early_stop_requested():
                    if collected_segment_payloads:
                        break
                    return None

                segment_emo_text = _effective_segment_emotion_text(segment)
                use_emo_text = len(segment_emo_text) > 0
                segment_text = str(segment["text"]).strip()
                if not segment_text:
                    continue

                def _segment_progress(value, desc, _seg_idx=seg_idx, _emotion_label=_emotion_progress_label(segment)):
                    if self._abort_requested() or self._early_stop_requested():
                        raise RuntimeError(_ABORT_ERROR)
                    fallback_label = "audio-ref" if emo_audio_prompt else "default"
                    extra = f"{str(desc or 'Synthesizing speech')} | Segment {_seg_idx + 1}/{len(segments)} | Emotion: {_emotion_label or fallback_label}"
                    _set_progress_phase(extra)
                    if callback is None:
                        return
                    step_idx = min(max(0, int(_seg_idx)), max(total_steps - 1, 0))
                    callback(
                        step_idx=step_idx,
                        override_num_inference_steps=total_steps,
                        denoising_extra=extra,
                        progress_unit="segments",
                        read_state=True,
                    )

                self.model.gr_progress = _segment_progress
                _segment_progress(0, "Synthesizing speech")
                segment_output = self.model.infer(
                    spk_audio_prompt=str(audio_guide),
                    text=segment_text,
                    output_path=None,
                    emo_audio_prompt=None if use_emo_text else emo_audio_prompt,
                    use_emo_text=use_emo_text,
                    emo_text=segment_emo_text if use_emo_text else None,
                    interval_silence=0,
                    verbose=False,
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    max_mel_tokens=global_max_mel_tokens,
                    duration_seconds=duration_seconds,
                    cg_max_total_tokens=cg_shared_kwargs.get("cg_max_total_tokens"),
                    cg_segment_generation_tokens=cg_shared_kwargs.get("cg_segment_generation_tokens"),
                    clear_cache_after=False,
                    defer_s2mel=True,
                    defer_bigvgan=True,
                )
                if segment_output is None:
                    if self._abort_requested():
                        return None
                    if self._early_stop_requested() and collected_segment_payloads:
                        break
                    return None
                if not isinstance(segment_output, dict) or "deferred_segment_payloads" not in segment_output:
                    raise RuntimeError("IndexTTS2 deferred generation returned an unexpected output format.")
                seg_payloads = list(segment_output["deferred_segment_payloads"] or [])
                segment_output["deferred_segment_payloads"] = []
                if len(seg_payloads) == 0:
                    del seg_payloads, segment_output
                    gc.collect()
                    continue
                appended_payload_count = 0
                next_segment = segments[seg_idx + 1] if (seg_idx + 1) < len(segments) else None
                pause_after_ms = _pause_after_ms(segment, next_segment)
                for payload in seg_payloads:
                    est_samples = _to_int(payload.get("estimated_samples", 0), 0) if isinstance(payload, dict) else 0
                    if max_samples > 0 and est_samples > 0 and estimated_total_samples >= max_samples and collected_segment_payloads:
                        reached_duration_limit = True
                        break
                    if max_samples > 0 and est_samples > 0 and (estimated_total_samples + est_samples) > max_samples and collected_segment_payloads:
                        reached_duration_limit = True
                        break
                    collected_segment_payloads.append({"segment_payload": payload, "pause_after_ms": 0})
                    appended_payload_count += 1
                    if est_samples > 0:
                        estimated_total_samples += est_samples
                if appended_payload_count > 0 and not reached_duration_limit:
                    collected_segment_payloads[-1]["pause_after_ms"] = pause_after_ms
                seg_payloads.clear()
                del seg_payloads, segment_output
                gc.collect()
                if reached_duration_limit:
                    break
        except RuntimeError as exc:
            if _ABORT_ERROR in str(exc) or self._abort_requested() or self._early_stop_requested():
                if self._abort_requested():
                    return None
                if self._early_stop_requested() and collected_segment_payloads:
                    pass
                else:
                    return None
            else:
                raise
        finally:
            self.model.gr_progress = None
            self.model._clear_persistent_generation_cache()
            self.model._release_runtime_gpu_caches()

        if self._abort_requested():
            return None
        if not collected_segment_payloads:
            return None
        previous_abort_checker = getattr(self.model, "abort_checker", None)
        self.model.abort_checker = lambda: self._abort_requested()
        try:
            _set_progress_phase("Preparing vocoder conditioning")
            collected_vc_targets = self.model.synthesize_from_segment_payloads(collected_segment_payloads, verbose=False)
            collected_segment_payloads.clear()
            del collected_segment_payloads
            gc.collect()
            self.model._release_runtime_gpu_caches()
            _set_progress_phase("Generating waveform")
            output = self.model.synthesize_from_vc_targets(collected_vc_targets, interval_silence=0, verbose=False)
            if isinstance(collected_vc_targets, list):
                collected_vc_targets.clear()
            del collected_vc_targets
            gc.collect()
            self.model._release_runtime_gpu_caches()
        except RuntimeError as exc:
            if _ABORT_ERROR in str(exc) or self._abort_requested():
                return None
            raise
        finally:
            self.model.abort_checker = previous_abort_checker
        sample_rate, wav = self._to_wav_tensor(output)
        if wav is None or wav.numel() == 0:
            return None
        if duration_seconds > 0:
            max_samples_out = int(round(sample_rate * duration_seconds))
            if max_samples_out > 0 and wav.shape[-1] > max_samples_out:
                wav = wav[:max_samples_out]
        self.model._release_runtime_gpu_caches()
        return {"x": wav, "audio_sampling_rate": int(sample_rate)}
