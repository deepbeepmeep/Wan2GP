from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import init_empty_weights

from mmgp import offload
from shared.utils import files_locator as fl

from .configs import KugelAudioConfig
from .models.kugelaudio_inference import KugelAudioForConditionalGenerationInference
from .processors import AudioProcessor, KugelAudioProcessor
from .processors.text_tokenizer import KugelAudioTextTokenizer


KUGELAUDIO_ASSET_DIR = "kugelaudio"
KUGELAUDIO_LOCAL_CONFIG_DIR = Path(__file__).parent / "configs" / "kugelaudio"
KUGELAUDIO_CONFIG_NAME = "config.json"
KUGELAUDIO_GENERATION_CONFIG_NAME = "generation_config.json"
KUGELAUDIO_TOKENIZER_DIR = "kugelaudio_text_tokenizer"
KUGELAUDIO_TOKENIZER_FILES = [
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
KUGELAUDIO_DEBUG = os.getenv("KUGELAUDIO_DEBUG", "0") not in ("0", "", "false", "False")
KUGELAUDIO_AUTO_SPLIT_SETTING_ID = "auto_split_every_s"


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


class KugelAudioPipeline:
    def __init__(
        self,
        model_weights_path: str,
        *,
        ckpt_root: Optional[Path] = None,
        device: Optional[torch.device] = None,
        lm_decoder_engine: Optional[str] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else Path(fl.get_download_location())
        self.lm_decoder_engine = str(lm_decoder_engine or "legacy").strip().lower()
        self._interrupt = False
        self._early_stop = False

        assets = self._resolve_assets()
        self.model = self._load_model(model_weights_path, assets["config_path"])
        self.processor = self._load_processor(assets["tokenizer_dir"])
        self.sample_rate = getattr(self.processor.audio_processor, "sampling_rate", 24000)

        self.generation_config = self._load_generation_config(assets["generation_config_path"])
        engine = "cg" if self.lm_decoder_engine == "cg" else "legacy"
        attn = getattr(self.model, "_attn_implementation", "sdpa")
        print(f"[KugelAudio] LM engine='{engine}', attention='{attn}'")

    def _abort_requested(self) -> bool:
        return bool(self._interrupt)

    def _early_stop_requested(self) -> bool:
        return bool(self._early_stop)

    def request_early_stop(self) -> None:
        self._early_stop = True

    def _resolve_assets(self) -> dict:
        local_config_path = KUGELAUDIO_LOCAL_CONFIG_DIR / KUGELAUDIO_CONFIG_NAME
        if local_config_path.is_file():
            config_path = str(local_config_path)
        else:
            config_path = fl.locate_file(
                os.path.join(KUGELAUDIO_ASSET_DIR, KUGELAUDIO_CONFIG_NAME),
                error_if_none=False,
            )
            if config_path is None:
                config_path = os.path.join(KUGELAUDIO_ASSET_DIR, KUGELAUDIO_CONFIG_NAME)

        local_generation_config_path = KUGELAUDIO_LOCAL_CONFIG_DIR / KUGELAUDIO_GENERATION_CONFIG_NAME
        if local_generation_config_path.is_file():
            generation_config_path = str(local_generation_config_path)
        else:
            generation_config_path = fl.locate_file(
                os.path.join(KUGELAUDIO_ASSET_DIR, KUGELAUDIO_GENERATION_CONFIG_NAME),
                error_if_none=False,
            )
            if generation_config_path is None:
                generation_config_path = os.path.join(
                    KUGELAUDIO_ASSET_DIR, KUGELAUDIO_GENERATION_CONFIG_NAME
                )

        tokenizer_sample = None
        for filename in KUGELAUDIO_TOKENIZER_FILES:
            candidate = fl.locate_file(
                os.path.join(KUGELAUDIO_TOKENIZER_DIR, filename),
                error_if_none=False,
            )
            if tokenizer_sample is None and candidate is not None:
                tokenizer_sample = candidate
        tokenizer_dir = (
            str(Path(tokenizer_sample).parent)
            if tokenizer_sample is not None
            else KUGELAUDIO_TOKENIZER_DIR
        )

        return {
            "config_path": config_path,
            "generation_config_path": generation_config_path,
            "tokenizer_dir": tokenizer_dir,
        }

    def _load_model(self, weights_path: str, config_path: str) -> KugelAudioForConditionalGenerationInference:
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        config = KugelAudioConfig(**config_dict)
        attn_implementation = "sdpa"
        if hasattr(config, "decoder_config") and config.decoder_config is not None:
            config.decoder_config._attn_implementation = attn_implementation
            config.decoder_config.attn_implementation = attn_implementation
        config._attn_implementation = attn_implementation
        config.attn_implementation = attn_implementation
        with init_empty_weights():
            model = KugelAudioForConditionalGenerationInference(config)
        offload.load_model_data(
            model,
            weights_path,
            default_dtype=None,
            writable_tensors=False,
        )
        model.set_lm_decoder_engine(self.lm_decoder_engine)
        model.eval()
        language_model = getattr(getattr(model, "model", None), "language_model", None)
        if language_model is not None and hasattr(language_model, "config"):
            language_model.config._attn_implementation = attn_implementation
            language_model.config.attn_implementation = attn_implementation
        if hasattr(model, "config"):
            model.config._attn_implementation = attn_implementation
            model.config.attn_implementation = attn_implementation
            if hasattr(model.config, "decoder_config") and model.config.decoder_config is not None:
                model.config.decoder_config._attn_implementation = attn_implementation
                model.config.decoder_config.attn_implementation = attn_implementation
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config._attn_implementation = attn_implementation
            model.generation_config.attn_implementation = attn_implementation
        model._attn_implementation = attn_implementation
        if KUGELAUDIO_DEBUG:
            try:
                print(
                    "[KugelAudio][debug] scaling/bias:",
                    model.speech_scaling_factor,
                    model.speech_bias_factor,
                )
            except Exception:
                pass
        first_param = next(model.parameters(), None)
        if first_param is not None:
            model._model_dtype = first_param.dtype
        model._input_device = (
            torch.device("cuda") if torch.cuda.is_available() else self.device
        )
        return model

    def _load_processor(self, tokenizer_dir: str) -> KugelAudioProcessor:
        try:
            return KugelAudioProcessor.from_pretrained(
                tokenizer_dir,
                language_model_pretrained_name=tokenizer_dir,
            )
        except Exception:
            tokenizer = KugelAudioTextTokenizer.from_pretrained(tokenizer_dir)
            audio_processor = AudioProcessor()
            return KugelAudioProcessor(
                tokenizer=tokenizer,
                audio_processor=audio_processor,
                speech_compression_ratio=3200,
                db_normalize=True,
            )

    def _load_generation_config(self, config_path: str) -> dict:
        if not config_path or not os.path.isfile(config_path):
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                return json.load(handle) or {}
        except Exception:
            return {}

    def _tokens_per_second(self) -> float:
        return float(self.sample_rate) / 3200.0

    def _resolve_max_new_tokens(self, duration_seconds: Optional[float], kwargs: dict) -> int:
        if duration_seconds is not None and duration_seconds > 0:
            tokens = int(round(duration_seconds * self._tokens_per_second()))
            return max(1, tokens)

        max_new_tokens = kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            sampling_steps = kwargs.get("sampling_steps")
            if sampling_steps:
                max_new_tokens = sampling_steps
        if max_new_tokens is None:
            max_new_tokens = self.generation_config.get("max_new_tokens", 2048)
        try:
            return max(1, int(max_new_tokens))
        except (TypeError, ValueError):
            return 2048

    def _resolve_auto_split_seconds(self, kwargs: dict) -> Optional[float]:
        custom_settings = kwargs.get("custom_settings", None)
        if not isinstance(custom_settings, dict):
            return None
        raw_value = custom_settings.get(KUGELAUDIO_AUTO_SPLIT_SETTING_ID, None)
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

    def generate(
        self,
        input_prompt: str,
        model_mode: Optional[str],
        audio_guide: Optional[str],
        *,
        temperature: float = 1.0,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False
        def _finish(value):
            self.model.release_decode_cuda_graph()
            return value

        text = _read_text_or_file(input_prompt, "Prompt")
        if not text.strip():
            raise ValueError("Prompt text cannot be empty for KugelAudio.")
        if "\\n" in text and "\n" not in text:
            text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
        if KUGELAUDIO_DEBUG:
            print("[KugelAudio][debug] prompt repr:", repr(text))
            print("[KugelAudio][debug] contains 'Speaker':", "Speaker" in text)

        seed = kwargs.get("seed", None)
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None
        if seed is not None and seed >= 0:
            _seed_everything(seed)

        duration_seconds = kwargs.get("duration_seconds", None)
        if duration_seconds is not None:
            try:
                duration_seconds = float(duration_seconds)
            except (TypeError, ValueError):
                duration_seconds = None

        max_new_tokens = self._resolve_max_new_tokens(duration_seconds, kwargs)
        if KUGELAUDIO_DEBUG:
            print(
                "[KugelAudio][debug] duration_seconds:",
                duration_seconds,
                "max_new_tokens:",
                max_new_tokens,
            )
        auto_split_seconds = self._resolve_auto_split_seconds(kwargs)
        auto_split_tokens = (
            max(1, int(round(auto_split_seconds * self._tokens_per_second())))
            if auto_split_seconds is not None
            else None
        )
        if KUGELAUDIO_DEBUG:
            print(
                "[KugelAudio][debug] auto_split_seconds:",
                auto_split_seconds,
                "auto_split_tokens:",
                auto_split_tokens,
            )

        cfg_scale = kwargs.get("guide_scale", None)
        if cfg_scale is None:
            cfg_scale = kwargs.get("audio_cfg_scale", None)
        try:
            cfg_scale = float(cfg_scale) if cfg_scale is not None else 3.0
        except (TypeError, ValueError):
            cfg_scale = 3.0

        do_sample = abs(float(temperature) - 1.0) >= 1e-6

        audio_guide2 = kwargs.get("audio_guide2", None)

        def _run_single(
            line_text: str,
            voice_path: Optional[str],
            *,
            extra_tail_tokens: int = 0,
            segment_duration_seconds: Optional[float] = None,
            completed_lines: Optional[int] = None,
            total_lines: Optional[int] = None,
            cumulative_offset_seconds: float = 0.0,
        ):
            inputs = self.processor(
                text=line_text,
                voice_prompt=voice_path,
                return_tensors="pt",
            )
            text_ids = inputs.get("text_ids")
            speech_input_mask = inputs.get("speech_input_mask")
            speech_tensors = inputs.get("speech_tensors")
            speech_masks = inputs.get("speech_masks")
            if KUGELAUDIO_DEBUG:
                print("[KugelAudio][debug] text_len:", int(text_ids.shape[1]) if text_ids is not None else None)
                if speech_input_mask is not None:
                    print("[KugelAudio][debug] speech_input_mask len/sum:", int(speech_input_mask.shape[1]), int(speech_input_mask.sum()))
                if speech_tensors is not None:
                    print("[KugelAudio][debug] speech_tensors shape:", tuple(speech_tensors.shape))
                if speech_masks is not None:
                    print("[KugelAudio][debug] speech_masks shape/sum:", tuple(speech_masks.shape), int(speech_masks.sum()))

            callback = kwargs.get("callback")
            max_tokens_local = max_new_tokens
            if segment_duration_seconds is not None and segment_duration_seconds > 0:
                max_tokens_local = self._resolve_max_new_tokens(segment_duration_seconds, kwargs)
            if completed_lines is not None and total_lines:
                last_line_update = {"t": 0.0}
                segment_total_seconds = None
                if segment_duration_seconds is not None and segment_duration_seconds > 0:
                    segment_total_seconds = segment_duration_seconds
                else:
                    segment_total_seconds = max_tokens_local / self._tokens_per_second()
                cumulative_offset_seconds = float(cumulative_offset_seconds or 0.0)
                def _line_callback(step_idx=None, override_num_inference_steps=None, denoising_extra=None, progress_unit=None):
                    if callback is None:
                        return
                    now = time.time()
                    if now - last_line_update["t"] < 1.0:
                        return
                    last_line_update["t"] = now
                    try:
                        step_val = 0 if step_idx is None else int(step_idx)
                    except Exception:
                        step_val = 0
                    seconds_generated = int(round((step_val + 1) / self._tokens_per_second()))
                    total_seconds = None
                    if duration_seconds is not None and duration_seconds > 0:
                        total_seconds = duration_seconds
                    else:
                        total_seconds = cumulative_offset_seconds + segment_total_seconds
                    progress_seconds = int(round(cumulative_offset_seconds + seconds_generated))
                    callback(
                        step_idx=progress_seconds,
                        override_num_inference_steps=int(round(total_seconds)),
                        denoising_extra=f"Segment {int(completed_lines) + 1}/{total_lines}",
                        progress_unit="s",
                    )
                active_callback = _line_callback
            else:
                active_callback = callback
            outputs = self.model.generate(
                text_ids=text_ids,
                speech_input_mask=speech_input_mask,
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                cfg_scale=cfg_scale,
                max_new_tokens=max_tokens_local,
                tail_tokens=max(0, int(extra_tail_tokens)),
                do_sample=do_sample,
                temperature=float(temperature),
                show_progress=True,
                abort_check=self._abort_requested,
                early_stop_check=self._early_stop_requested,
                callback=active_callback,
                release_decode_graph_on_exit=False,
            )
            if outputs is None:
                return None
            if getattr(outputs, "speech_outputs", None):
                return outputs.speech_outputs[0]
            return None

        # Multi-speaker mode: split by Speaker tags and run per segment (ComfyUI behavior)
        # Only do this when a reference voice is provided; otherwise let the native model handle speakers.
        if "Speaker" in text and (audio_guide is not None or audio_guide2 is not None):
            tag_pattern = re.compile(r"Speaker\s*(\d+)\s*:\s*", re.IGNORECASE)
            matches = list(tag_pattern.finditer(text))
            if KUGELAUDIO_DEBUG:
                print("[KugelAudio][debug] raw_text repr:", repr(text))
                head = text[:160]
                print("[KugelAudio][debug] head:", head)
                print("[KugelAudio][debug] head ords:", [ord(ch) for ch in head])
                print("[KugelAudio][debug] speaker matches:", len(matches))
                for idx, m in enumerate(matches[:10]):
                    print(
                        "[KugelAudio][debug] match",
                        idx,
                        "span",
                        m.span(),
                        "speaker",
                        m.group(1),
                    )
            parsed = []
            for i, match in enumerate(matches):
                speaker_id = int(match.group(1))
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                line_text = text[start:end].strip()
                if line_text:
                    parsed.append((speaker_id, line_text))
            speaker_id_map = None
            if parsed and audio_guide2 is not None:
                unique_ids = sorted({speaker_id for speaker_id, _ in parsed})
                if len(unique_ids) > 2:
                    raise ValueError("Two-speaker mode supports exactly two speaker IDs.")
                if len(unique_ids) >= 2:
                    speaker_id_map = {unique_ids[0]: 0, unique_ids[1]: 1}
                elif len(unique_ids) == 1:
                    speaker_id_map = {unique_ids[0]: 0}
            if KUGELAUDIO_DEBUG:
                print("[KugelAudio][debug] parsed segments:", len(parsed))
                for idx, (sid, seg) in enumerate(parsed[:6]):
                    print("[KugelAudio][debug] seg", idx, "speaker", sid, "text:", repr(seg[:120]))

            if parsed:
                expanded = []
                for parsed_idx, (speaker_id, line_text) in enumerate(parsed):
                    split_segments = self._split_text_sequence(line_text, auto_split_tokens)
                    for split_idx, one_segment in enumerate(split_segments):
                        expanded.append(
                            (
                                speaker_id,
                                one_segment,
                                split_idx == len(split_segments) - 1 and parsed_idx < len(parsed) - 1,
                            )
                        )
                if KUGELAUDIO_DEBUG:
                    print("[KugelAudio][debug] expanded segments:", len(expanded))
                audio_segments = []
                pause_seconds = kwargs.get("pause_seconds", 0.2)
                try:
                    pause_seconds = float(pause_seconds)
                except (TypeError, ValueError):
                    pause_seconds = 0.2
                pause_seconds = max(0.0, min(2.0, pause_seconds))
                pause_tokens_total = int(round(pause_seconds * self._tokens_per_second()))
                tail_tokens = min(8, pause_tokens_total)
                remaining_seconds = pause_seconds - (tail_tokens / self._tokens_per_second())
                pause_samples = int(round(max(0.0, remaining_seconds) * self.sample_rate))
                silence = (
                    torch.zeros(pause_samples, dtype=torch.float32) if pause_samples > 0 else None
                )
                elapsed_seconds = 0.0
                total_duration_seconds = duration_seconds if duration_seconds is not None and duration_seconds > 0 else None
                completed_lines = 0
                for speaker_id, line_text, append_pause_after in expanded:
                    if self._abort_requested():
                        return _finish(None)
                    duration_left = None
                    if total_duration_seconds is not None:
                        duration_left = max(0.0, total_duration_seconds - elapsed_seconds)
                        if duration_left <= 0:
                            break
                    voice_path = audio_guide
                    mapped_id = speaker_id
                    if speaker_id_map is not None:
                        mapped_id = speaker_id_map.get(speaker_id, 0)
                    if mapped_id == 1 and audio_guide2 is not None:
                        voice_path = audio_guide2
                    extra_tail_tokens = tail_tokens
                    segment = _run_single(
                        line_text,
                        voice_path,
                        extra_tail_tokens=extra_tail_tokens,
                        segment_duration_seconds=duration_left,
                        completed_lines=completed_lines,
                        total_lines=len(expanded),
                        cumulative_offset_seconds=elapsed_seconds,
                    )
                    if segment is None:
                        if self._early_stop_requested() and audio_segments:
                            audio = torch.cat(audio_segments, dim=-1)
                            return _finish({"x": audio, "audio_sampling_rate": int(self.sample_rate)})
                        return _finish(None)
                    audio_segments.append(segment)
                    completed_lines += 1
                    # No extra callback here; per-line callback handles progress.
                    elapsed_seconds += float(segment.shape[-1]) / float(self.sample_rate)
                    if silence is not None and append_pause_after:
                        audio_segments.append(silence.to(segment))
                        elapsed_seconds += float(pause_samples) / float(self.sample_rate)
                    if self._early_stop_requested():
                        break
                if audio_segments:
                    audio = torch.cat(audio_segments, dim=-1)
                    return _finish({"x": audio, "audio_sampling_rate": int(self.sample_rate)})

        voice_prompt = audio_guide
        single_segments = self._split_text_sequence(text, auto_split_tokens)
        if len(single_segments) > 1:
            audio_segments = []
            elapsed_seconds = 0.0
            total_duration_seconds = (
                duration_seconds if duration_seconds is not None and duration_seconds > 0 else None
            )
            for idx, one_segment in enumerate(single_segments):
                if self._abort_requested():
                    return _finish(None)
                duration_left = None
                if total_duration_seconds is not None:
                    duration_left = max(0.0, total_duration_seconds - elapsed_seconds)
                    if duration_left <= 0:
                        break
                segment = _run_single(
                    one_segment,
                    voice_prompt,
                    extra_tail_tokens=3,
                    segment_duration_seconds=duration_left,
                    completed_lines=idx,
                    total_lines=len(single_segments),
                    cumulative_offset_seconds=elapsed_seconds,
                )
                if segment is None:
                    if self._early_stop_requested() and audio_segments:
                        audio = torch.cat(audio_segments, dim=-1)
                        return _finish({"x": audio, "audio_sampling_rate": int(self.sample_rate)})
                    return _finish(None)
                audio_segments.append(segment)
                elapsed_seconds += float(segment.shape[-1]) / float(self.sample_rate)
                if self._early_stop_requested():
                    break
            if len(audio_segments) > 0:
                audio = torch.cat(audio_segments, dim=-1)
                return _finish({"x": audio, "audio_sampling_rate": int(self.sample_rate)})
            return _finish(None)

        # Default single-pass generation
        inputs = self.processor(
            text=text,
            voice_prompt=voice_prompt,
            return_tensors="pt",
        )
        text_ids = inputs.get("text_ids")
        speech_input_mask = inputs.get("speech_input_mask")
        speech_tensors = inputs.get("speech_tensors")
        speech_masks = inputs.get("speech_masks")
        if KUGELAUDIO_DEBUG:
            print("[KugelAudio][debug] text_len:", int(text_ids.shape[1]) if text_ids is not None else None)
            if speech_input_mask is not None:
                print("[KugelAudio][debug] speech_input_mask len/sum:", int(speech_input_mask.shape[1]), int(speech_input_mask.sum()))
            if speech_tensors is not None:
                print("[KugelAudio][debug] speech_tensors shape:", tuple(speech_tensors.shape))
            if speech_masks is not None:
                print("[KugelAudio][debug] speech_masks shape/sum:", tuple(speech_masks.shape), int(speech_masks.sum()))

        callback = kwargs.get("callback")
        if callback is not None:
            base_callback = callback
            total_seconds = None
            if duration_seconds is not None and duration_seconds > 0:
                total_seconds = duration_seconds
            else:
                total_seconds = max_new_tokens / self._tokens_per_second()

            def _seconds_callback(step_idx=None, override_num_inference_steps=None, denoising_extra=None, progress_unit=None):
                if total_seconds is None:
                    base_callback(step_idx=step_idx, override_num_inference_steps=override_num_inference_steps,
                                  denoising_extra=denoising_extra, progress_unit=progress_unit)
                    return
                try:
                    step_val = 0 if step_idx is None else int(step_idx)
                except Exception:
                    step_val = 0
                seconds_generated = int(round((step_val + 1) / self._tokens_per_second()))
                base_callback(
                    step_idx=seconds_generated,
                    override_num_inference_steps=int(round(total_seconds)),
                    denoising_extra=f"{seconds_generated}/{int(round(total_seconds))} s",
                    progress_unit="s",
                )
            callback = _seconds_callback
        outputs = self.model.generate(
            text_ids=text_ids,
            speech_input_mask=speech_input_mask,
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
            tail_tokens=3,
            do_sample=do_sample,
            temperature=float(temperature),
            show_progress=True,
            abort_check=self._abort_requested,
            early_stop_check=self._early_stop_requested,
            callback=callback,
            release_decode_graph_on_exit=False,
        )
        if outputs is None:
            return _finish(None)

        if self._abort_requested():
            return _finish(None)

        audio = None
        if getattr(outputs, "speech_outputs", None):
            audio = outputs.speech_outputs[0]
        if audio is None:
            return _finish(None)

        return _finish({"x": audio, "audio_sampling_rate": int(self.sample_rate)})

    def release(self) -> None:
        if self.model is not None and hasattr(self.model, "release_decode_cuda_graph"):
            self.model.release_decode_cuda_graph()
        if hasattr(self.model, "to"):
            self.model.to("cpu")
        self.model = None
