from __future__ import annotations

import json
import math
import os
import random
import re
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

from .. import qwen3_handler as qwen3_defs
from .core.models.configuration_qwen3_tts import Qwen3TTSConfig
from .core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from .core.models.processing_qwen3_tts import Qwen3TTSProcessor
from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
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


def _is_abort_exception(exc: Exception) -> bool:
    return "Abort requested" in str(exc)


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
        lm_decoder_engine: Optional[str] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.base_model_type = base_model_type
        self.lm_decoder_engine = str(lm_decoder_engine or "legacy").strip().lower()
        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else Path(fl.get_download_location())
        self._interrupt = False
        self._early_stop = False

        assets = self._resolve_assets(model_weights_path)
        self.model = self._load_main_model(assets)
        self.speech_tokenizer = self._load_speech_tokenizer(assets)
        self.processor = self._load_text_processor(assets)

        self.model.load_speech_tokenizer(self.speech_tokenizer)
        self.model.set_lm_decoder_engine(self.lm_decoder_engine)
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
        engine = "cg" if self.lm_decoder_engine == "cg" else "legacy"
        print(f"[Qwen3TTS] LM Engine='{engine}'")

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

    def _resolve_max_new_tokens(self, duration_seconds: Optional[float], kwargs: dict) -> int:
        if duration_seconds is not None and duration_seconds > 0:
            tokens = int(round(float(duration_seconds) * self._get_tokens_per_second()))
            return max(1, tokens)
        max_new_tokens = kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            sampling_steps = kwargs.get("sampling_steps")
            if sampling_steps:
                max_new_tokens = int(sampling_steps)
        if max_new_tokens is None:
            max_new_tokens = self.tts.generate_defaults.get("max_new_tokens", 2048)
        try:
            return max(1, int(max_new_tokens))
        except (TypeError, ValueError):
            return 2048

    def _resolve_auto_split_seconds(self, kwargs: dict) -> Optional[float]:
        custom_settings = kwargs.get("custom_settings", None)
        if not isinstance(custom_settings, dict):
            return None
        raw_value = custom_settings.get(qwen3_defs.QWEN3_TTS_AUTO_SPLIT_SETTING_ID, None)
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
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded.get("offset_mapping", None) if isinstance(encoded, dict) else None
            if offsets is None and hasattr(encoded, "offset_mapping"):
                offsets = encoded.offset_mapping
            if offsets is None or len(offsets) <= token_limit:
                return None
            cut_char = offsets[token_limit][0]
            if isinstance(cut_char, (list, tuple)):
                cut_char = cut_char[0]
            cut_char = int(cut_char)
            return min(len(text), max(1, cut_char))
        except Exception:
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                return None
            if token_ids is None or len(token_ids) <= token_limit:
                return None
            try:
                prefix = tokenizer.decode(
                    token_ids[:token_limit],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                return None
            return min(len(text), max(1, len(prefix)))

    def _find_split_index_before_cut(self, text: str, cut_index: int) -> int:
        safe_cut = min(len(text), max(1, int(cut_index)))
        prefix = text[:safe_cut]
        dot_idx = prefix.rfind(".")
        newline_idx = prefix.rfind("\n")
        best_idx = max(dot_idx, newline_idx)
        if best_idx >= 0:
            return best_idx + 1
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

    @staticmethod
    def _normalize_audio_prompt_type(value) -> str:
        mode = str(value or "A").strip().upper()
        if mode not in ("A", "AB"):
            raise ValueError(f"Unsupported audio prompt mode '{mode}'. Use 'A' (one speaker) or 'AB' (two speakers).")
        return mode

    @staticmethod
    def _parse_two_speaker_dialogue(text: str) -> list[tuple[int, str]]:
        speaker_matches = list(re.finditer(r"Speaker\s*(\d+)\s*:\s*", text, flags=re.IGNORECASE))
        if not speaker_matches:
            raise ValueError(
                "Two-speaker mode requires prompt lines using Speaker 1: and Speaker 2: "
                "(or any two numeric speaker IDs)."
            )
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
                segments.append((speaker_id_to_internal[int(match.group(1))], segment_text))
        if not segments:
            raise ValueError("No dialogue text found after Speaker tags.")
        return segments

    @staticmethod
    def _resolve_two_speaker_ref_scripts(raw_ref_text: str) -> tuple[list[Optional[str]], list[bool]]:
        normalized = str(raw_ref_text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        if not lines:
            return [None, None], [True, True]

        first = lines[0] if len(lines) > 0 else None
        second = lines[1] if len(lines) > 1 else None
        speaker_ref_texts = [first, second]
        speaker_xvector_only = [text is None for text in speaker_ref_texts]
        return speaker_ref_texts, speaker_xvector_only

    def _estimate_two_speaker_prompt_tokens(
        self,
        *,
        dialogue_segments: list[tuple[int, str]],
        speaker_prompts: dict[int, VoiceClonePromptItem],
    ) -> int:
        max_prompt_tokens = 1
        for speaker_id, segment_text in dialogue_segments:
            assistant_text = f"<|im_start|>assistant\n{segment_text}<|im_end|>\n<|im_start|>assistant\n"
            encoded = self.processor(text=[assistant_text], return_tensors="pt")
            input_token_length = int(encoded["input_ids"].shape[-1])
            prompt_item = speaker_prompts[speaker_id]
            ref_token_length = 0
            if prompt_item.ref_text:
                ref_encoded = self.processor(text=[f"<|im_start|>assistant\n{prompt_item.ref_text}<|im_end|>\n"], return_tensors="pt")
                ref_token_length = int(ref_encoded["input_ids"].shape[-1])
            ref_code_length = int(prompt_item.ref_code.shape[0]) if prompt_item.ref_code is not None else 0
            max_prompt_tokens = max(max_prompt_tokens, input_token_length + ref_token_length + ref_code_length + 64)
        return max_prompt_tokens

    def _generate_base_two_speaker(
        self,
        *,
        text: str,
        language: str,
        audio_guide: str,
        audio_guide2: str,
        speaker_ref_texts: list[Optional[str]],
        speaker_xvector_only: list[bool],
        max_new_tokens: int,
        duration_seconds: Optional[float],
        total_seconds: Optional[int],
        temperature: float,
        top_k: Optional[int],
        pause_seconds: float,
        auto_split_tokens: Optional[int],
        callback,
    ):
        dialogue_segments_raw = self._parse_two_speaker_dialogue(text)
        dialogue_segments: list[tuple[int, str]] = []
        for speaker_id, segment_text in dialogue_segments_raw:
            split_segments = self._split_text_sequence(segment_text, auto_split_tokens)
            for one_segment in split_segments:
                if len(one_segment.strip()) > 0:
                    dialogue_segments.append((speaker_id, one_segment.strip()))
        if len(dialogue_segments) == 0:
            return None
        try:
            prompt_items = self.tts.create_voice_clone_prompt(
                ref_audio=[audio_guide, audio_guide2],
                ref_text=speaker_ref_texts,
                x_vector_only_mode=speaker_xvector_only,
            )
        except RuntimeError as exc:
            if _is_abort_exception(exc):
                return None
            raise
        speaker_prompts = {0: prompt_items[0], 1: prompt_items[1]}

        keep_shared_graph = self.lm_decoder_engine == "cg"
        shared_gen_kwargs = {}
        if keep_shared_graph:
            max_prompt_tokens = self._estimate_two_speaker_prompt_tokens(
                dialogue_segments=dialogue_segments,
                speaker_prompts=speaker_prompts,
            )
            shared_gen_kwargs["cg_max_talker_tokens"] = int(max_prompt_tokens + max_new_tokens)
            shared_gen_kwargs["cg_max_subtalker_tokens"] = int(max(2, int(self.model.talker.config.num_code_groups) + 1))
            shared_gen_kwargs["release_decode_graph_on_exit"] = False

        try:
            pause_seconds = float(pause_seconds)
        except (TypeError, ValueError):
            pause_seconds = 0.5
        pause_seconds = max(0.0, min(10.0, pause_seconds))
        pause_samples_default = int(round(pause_seconds * self.sample_rate))

        tokens_per_second = self._get_tokens_per_second()
        seconds_per_token = self._get_seconds_per_token()
        max_total_samples = int(round(duration_seconds * self.sample_rate)) if duration_seconds is not None and duration_seconds > 0 else None
        max_total_seconds = float(duration_seconds) if duration_seconds is not None and duration_seconds > 0 else None
        per_segment_token_limit = int(max_new_tokens)
        if auto_split_tokens is not None and auto_split_tokens > 0:
            per_segment_token_limit = max(1, min(per_segment_token_limit, int(auto_split_tokens)))

        generated_segments = []
        elapsed_seconds = 0.0

        try:
            for segment_index, (speaker_id, segment_text) in enumerate(dialogue_segments):
                if self._abort_requested():
                    return None
                if max_total_seconds is not None and elapsed_seconds >= max_total_seconds:
                    break
                if self._early_stop_requested():
                    break

                segment_seconds_left = None
                if max_total_seconds is not None:
                    segment_seconds_left = max(0.0, max_total_seconds - elapsed_seconds)
                    if segment_seconds_left <= 0:
                        break
                    duration_cap = max(1, int(math.ceil(segment_seconds_left * tokens_per_second)))
                    segment_max_new_tokens = max(1, min(per_segment_token_limit, duration_cap))
                else:
                    segment_max_new_tokens = per_segment_token_limit

                progress_offset_seconds = int(elapsed_seconds)

                def _offset_callback(step_idx=None, override_num_inference_steps=None, denoising_extra=None, progress_unit=None):
                    if callback is None:
                        return
                    local_step = -1
                    if step_idx is not None:
                        try:
                            local_step = int(step_idx)
                        except Exception:
                            local_step = -1
                    if local_step < 0:
                        global_step = max(-1, progress_offset_seconds - 1)
                    else:
                        global_step = progress_offset_seconds + local_step
                    if total_seconds is not None:
                        progress_total = int(total_seconds)
                    else:
                        try:
                            local_total = int(override_num_inference_steps)
                        except Exception:
                            local_total = max(1, int(math.ceil(segment_max_new_tokens * seconds_per_token)))
                        progress_total = max(1, progress_offset_seconds + local_total)
                    segment_info = f"Segment {segment_index + 1}/{len(dialogue_segments)}"
                    extra_info = f"{denoising_extra} | {segment_info}" if denoising_extra else segment_info
                    callback(
                        step_idx=global_step,
                        override_num_inference_steps=progress_total,
                        denoising_extra=extra_info,
                        progress_unit=progress_unit or "seconds",
                    )

                criteria, stopping = self._build_stopping_criteria(
                    segment_max_new_tokens,
                    seconds_per_token,
                    _offset_callback,
                    total_seconds=int(math.ceil(segment_seconds_left)) if segment_seconds_left is not None else None,
                )
                prompt_item = speaker_prompts[speaker_id]
                voice_clone_prompt_dict = self.tts._prompt_items_to_voice_clone_prompt([prompt_item])
                input_ids = self.tts._tokenize_texts([self.tts._build_assistant_text(segment_text)])
                ref_ids = None
                if prompt_item.ref_text is not None and prompt_item.ref_text != "":
                    ref_ids = [self.tts._tokenize_texts([self.tts._build_ref_text(prompt_item.ref_text)])[0]]
                segment_kwargs = self.tts._merge_generate_kwargs(
                    max_new_tokens=int(segment_max_new_tokens),
                    temperature=float(temperature),
                    top_k=int(top_k) if top_k is not None else None,
                    stopping_criteria=stopping,
                    **shared_gen_kwargs,
                )

                try:
                    talker_codes_list, _ = self.model.generate(
                        input_ids=input_ids,
                        ref_ids=ref_ids,
                        voice_clone_prompt=voice_clone_prompt_dict,
                        languages=[language],
                        non_streaming_mode=False,
                        **segment_kwargs,
                    )
                except RuntimeError as exc:
                    if _is_abort_exception(exc):
                        return None
                    raise
                finally:
                    criteria.close()

                if self._abort_requested():
                    return None
                if not talker_codes_list:
                    if self._early_stop_requested() and generated_segments:
                        break
                    return None

                talker_codes = talker_codes_list[0]

                has_next = segment_index + 1 < len(dialogue_segments)
                append_pause_after = has_next and dialogue_segments[segment_index + 1][0] != speaker_id
                generated_segments.append(
                    {
                        "codes": talker_codes,
                        "append_pause_after": append_pause_after,
                    }
                )

                elapsed_seconds += float(talker_codes.shape[0]) / float(tokens_per_second)
                if append_pause_after:
                    elapsed_seconds += pause_seconds
                if max_total_seconds is not None and elapsed_seconds >= max_total_seconds:
                    break
                if self._early_stop_requested():
                    break
        finally:
            if keep_shared_graph:
                self.model.release_decode_cuda_graph()

        if self._abort_requested():
            return None
        if not generated_segments:
            return None

        wavs_all, sr = self.model.speech_tokenizer.decode([{"audio_codes": item["codes"]} for item in generated_segments])
        audio_segments = []
        elapsed_samples = 0
        for index, item in enumerate(generated_segments):
            wav = wavs_all[index]
            segment_audio = torch.from_numpy(wav)
            if max_total_samples is not None and elapsed_samples + int(segment_audio.shape[-1]) > max_total_samples:
                keep_samples = max_total_samples - elapsed_samples
                if keep_samples <= 0:
                    break
                segment_audio = segment_audio[:keep_samples]
            if int(segment_audio.shape[-1]) > 0:
                audio_segments.append(segment_audio)
                elapsed_samples += int(segment_audio.shape[-1])

            if max_total_samples is not None and elapsed_samples >= max_total_samples:
                break
            if not bool(item["append_pause_after"]):
                continue
            pause_samples = pause_samples_default
            if pause_samples <= 0:
                continue
            if max_total_samples is not None:
                pause_samples = min(pause_samples, max_total_samples - elapsed_samples)
            if pause_samples <= 0:
                continue
            audio_segments.append(torch.zeros((pause_samples,), dtype=segment_audio.dtype, device=segment_audio.device))
            elapsed_samples += pause_samples
            if max_total_samples is not None and elapsed_samples >= max_total_samples:
                break

        if not audio_segments:
            return None
        output_audio = torch.cat(audio_segments, dim=-1)
        return {"x": output_audio, "audio_sampling_rate": int(sr)}

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
        max_new_tokens = self._resolve_max_new_tokens(duration_seconds, kwargs)
        total_seconds = max(1, int(math.ceil(duration_seconds))) if duration_seconds is not None and duration_seconds > 0 else None

        top_k = kwargs.get("top_k", None)
        if top_k is not None:
            try:
                top_k = int(top_k)
            except (TypeError, ValueError):
                top_k = None
        audio_prompt_type = self._normalize_audio_prompt_type(kwargs.get("audio_prompt_type", "A"))
        audio_guide2 = kwargs.get("audio_guide2", None)
        pause_seconds = kwargs.get("pause_seconds", 0.5)
        auto_split_seconds = self._resolve_auto_split_seconds(kwargs)
        auto_split_tokens = (
            max(1, int(round(auto_split_seconds * self._get_tokens_per_second())))
            if auto_split_seconds is not None
            else None
        )

        if self.base_model_type == "qwen3_tts_base" and audio_prompt_type == "AB":
            if not audio_guide:
                raise ValueError("Speaker 1 reference audio is required for Qwen3 Base voice clone.")
            if not audio_guide2:
                raise ValueError("Speaker 2 reference audio is required for two-speaker Qwen3 Base mode.")
            language = (model_mode or "auto").lower()
            ref_text = _read_text_or_file(alt_prompt, "Reference transcript(s)")
            speaker_ref_texts, speaker_xvector_only = self._resolve_two_speaker_ref_scripts(ref_text)
            return self._generate_base_two_speaker(
                text=text,
                language=language,
                audio_guide=audio_guide,
                audio_guide2=audio_guide2,
                speaker_ref_texts=speaker_ref_texts,
                speaker_xvector_only=speaker_xvector_only,
                max_new_tokens=max_new_tokens,
                duration_seconds=duration_seconds,
                total_seconds=total_seconds,
                temperature=float(temperature),
                top_k=top_k,
                pause_seconds=pause_seconds,
                auto_split_tokens=auto_split_tokens,
                callback=callback,
            )

        text_segments = self._split_text_sequence(text, auto_split_tokens)
        if len(text_segments) > 1:
            if self.base_model_type == "qwen3_tts_customvoice":
                if not self.supported_speakers:
                    raise ValueError("No supported speakers found for Qwen3 CustomVoice.")
                segment_speaker = model_mode or self.supported_speakers[0]
                segment_language = "auto"
                segment_instruct = _read_text_or_file(alt_prompt, "Instruction")
            elif self.base_model_type == "qwen3_tts_voicedesign":
                segment_language = (model_mode or "auto").lower()
                segment_instruct = _read_text_or_file(alt_prompt, "Instruction")
            elif self.base_model_type == "qwen3_tts_base":
                if not audio_guide:
                    raise ValueError("Reference audio is required for Qwen3 Base voice clone.")
                segment_language = (model_mode or "auto").lower()
                segment_ref_text = _read_text_or_file(alt_prompt, "Reference transcript")
                segment_x_vector_only_mode = not segment_ref_text.strip()
                if segment_x_vector_only_mode:
                    segment_ref_text = None
                try:
                    segment_voice_clone_prompt_item = self.tts.create_voice_clone_prompt(
                        ref_audio=audio_guide,
                        ref_text=segment_ref_text,
                        x_vector_only_mode=segment_x_vector_only_mode,
                    )[0]
                except RuntimeError as exc:
                    if _is_abort_exception(exc):
                        return None
                    raise
            else:
                raise ValueError(f"Unknown Qwen3 TTS type: {self.base_model_type}")

            audio_segments = []
            elapsed_seconds = 0.0
            elapsed_samples = 0
            max_total_samples = int(round(duration_seconds * self.sample_rate)) if duration_seconds is not None and duration_seconds > 0 else None
            max_total_seconds = float(duration_seconds) if duration_seconds is not None and duration_seconds > 0 else None
            per_segment_token_limit = int(max_new_tokens)
            if auto_split_tokens is not None and auto_split_tokens > 0:
                per_segment_token_limit = max(1, min(per_segment_token_limit, int(auto_split_tokens)))
            sr_last = int(self.sample_rate)

            try:
                for segment_index, segment_text in enumerate(text_segments):
                    if self._abort_requested():
                        return None
                    if self._early_stop_requested():
                        break

                    segment_seconds_left = None
                    if max_total_seconds is not None:
                        segment_seconds_left = max(0.0, max_total_seconds - elapsed_seconds)
                        if segment_seconds_left <= 0:
                            break
                        duration_cap = max(1, int(math.ceil(segment_seconds_left * self._get_tokens_per_second())))
                        segment_max_new_tokens = max(1, min(per_segment_token_limit, duration_cap))
                    else:
                        segment_max_new_tokens = per_segment_token_limit

                    progress_offset_seconds = int(elapsed_seconds)

                    def _offset_callback(step_idx=None, override_num_inference_steps=None, denoising_extra=None, progress_unit=None):
                        if callback is None:
                            return
                        local_step = -1
                        if step_idx is not None:
                            try:
                                local_step = int(step_idx)
                            except Exception:
                                local_step = -1
                        if local_step < 0:
                            global_step = max(-1, progress_offset_seconds - 1)
                        else:
                            global_step = progress_offset_seconds + local_step
                        if total_seconds is not None:
                            progress_total = int(total_seconds)
                        else:
                            try:
                                local_total = int(override_num_inference_steps)
                            except Exception:
                                local_total = max(1, int(math.ceil(segment_max_new_tokens * seconds_per_token)))
                            progress_total = max(1, progress_offset_seconds + local_total)
                        segment_info = f"Segment {segment_index + 1}/{len(text_segments)}"
                        extra_info = f"{denoising_extra} | {segment_info}" if denoising_extra else segment_info
                        callback(
                            step_idx=global_step,
                            override_num_inference_steps=progress_total,
                            denoising_extra=extra_info,
                            progress_unit=progress_unit or "seconds",
                        )

                    criteria, stopping = self._build_stopping_criteria(
                        segment_max_new_tokens,
                        seconds_per_token,
                        _offset_callback,
                        total_seconds=int(math.ceil(segment_seconds_left)) if segment_seconds_left is not None else None,
                    )
                    segment_gen_kwargs = {
                        "max_new_tokens": int(segment_max_new_tokens),
                        "temperature": float(temperature),
                        "stopping_criteria": stopping,
                    }
                    if top_k is not None:
                        segment_gen_kwargs["top_k"] = top_k

                    try:
                        if self.base_model_type == "qwen3_tts_customvoice":
                            wavs, sr = self.tts.generate_custom_voice(
                                text=segment_text,
                                language=segment_language,
                                speaker=segment_speaker,
                                instruct=segment_instruct,
                                **segment_gen_kwargs,
                            )
                        elif self.base_model_type == "qwen3_tts_voicedesign":
                            wavs, sr = self.tts.generate_voice_design(
                                text=segment_text,
                                language=segment_language,
                                instruct=segment_instruct,
                                **segment_gen_kwargs,
                            )
                        else:
                            wavs, sr = self.tts.generate_voice_clone(
                                text=segment_text,
                                language=segment_language,
                                voice_clone_prompt=[segment_voice_clone_prompt_item],
                                **segment_gen_kwargs,
                            )
                    except RuntimeError as exc:
                        if _is_abort_exception(exc):
                            return None
                        raise
                    finally:
                        criteria.close()

                    if self._abort_requested():
                        return None
                    if not wavs:
                        return None
                    sr_last = int(sr)
                    segment_audio = torch.from_numpy(wavs[0])
                    if max_total_samples is not None and elapsed_samples + int(segment_audio.shape[-1]) > max_total_samples:
                        keep_samples = max_total_samples - elapsed_samples
                        if keep_samples <= 0:
                            break
                        segment_audio = segment_audio[:keep_samples]
                    if int(segment_audio.shape[-1]) > 0:
                        audio_segments.append(segment_audio)
                        elapsed_samples += int(segment_audio.shape[-1])
                        elapsed_seconds += float(segment_audio.shape[-1]) / float(sr_last)
                    if max_total_samples is not None and elapsed_samples >= max_total_samples:
                        break
            finally:
                if self.base_model_type == "qwen3_tts_base" and self.lm_decoder_engine == "cg":
                    self.model.release_decode_cuda_graph()

            if len(audio_segments) == 0:
                return None
            return {"x": torch.cat(audio_segments, dim=-1), "audio_sampling_rate": sr_last}

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
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

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
            if _is_abort_exception(exc):
                return None
            raise
        finally:
            criteria.close()

        if self._abort_requested():
            return None

        wav = torch.from_numpy(wavs[0])
        return {"x": wav, "audio_sampling_rate": int(sr)}

    def release(self) -> None:
        if self.model is not None and hasattr(self.model, "release_decode_cuda_graph"):
            self.model.release_decode_cuda_graph()
        for module in [self.model, getattr(self.speech_tokenizer, "model", None)]:
            if hasattr(module, "to"):
                module.to("cpu")
        self.model = None
        self.speech_tokenizer = None
