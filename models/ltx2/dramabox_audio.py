import os
import random
import re
import gc
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch

from shared.utils.audio_cleaning import mute_isolated_transient_noise, trim_leading_noise_before_speech, trim_leading_transient_noise, trim_trailing_transient_noise

from .ltx_audio_tts import LTXAudioTTSPipelineBase
from .ltx_core.components.schedulers import LTX2Scheduler
from .ltx_core.conditioning import AudioConditionByAppendedReferenceLatent
from .ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from .scenema_audio import _audio_tensor_to_numpy, _clean_spaces, _normalize_volume, _numpy_to_audio_tensor, _parse_speaker_options, _shorten_long_silence, _trim_leading_extra_words_tensor, _trim_silence


DRAMABOX_DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone"
DRAMABOX_FPS = 25.0
DRAMABOX_DEFAULT_STEPS = 30
DRAMABOX_DEFAULT_DURATION_MULTIPLIER = 1.1
DRAMABOX_DEFAULT_REFERENCE_SECONDS = 10.0
DRAMABOX_DEFAULT_CFG_SCALE = 2.5
DRAMABOX_DEFAULT_STG_SCALE = 1.5
DRAMABOX_REFERENCE_PEAK_DB = -4.0
DRAMABOX_STG_BLOCK = 29
DRAMABOX_TRANSIENT_SILENCE_THRESHOLD = 0.006
DRAMABOX_ISOLATED_TRANSIENT_THRESHOLD = 0.01
DRAMABOX_TRANSIENT_MAX_SECONDS = 0.18
DRAMABOX_LEADING_TRANSIENT_MAX_SECONDS = 0.30
DRAMABOX_LEADING_SPEECH_THRESHOLD = 0.03
DRAMABOX_MAX_LEADING_SECONDS = 2.0


@dataclass
class _DramaBoxSegment:
    prompt: str
    duration_s: float
    seed: int
    speaker: int = 1
    expected_text: str = ""


_LAUGH_VERBS = {
    r"\blaugh(?:s|ed|ing)?\b": 1.5,
    r"\bcackl(?:e|es|ed|ing)\b": 1.5,
    r"\bchuckl(?:e|es|ed|ing)\b": 1.0,
    r"\bgiggl(?:e|es|ed|ing)\b": 1.0,
    r"\bsnicker(?:s|ed|ing)?\b": 0.8,
    r"\bcru?el laugh\b": 1.5,
}


def _read_text_or_file(value, label: str) -> str:
    if value is None:
        return ""
    text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    if os.path.isfile(text) and os.path.splitext(text)[1].lower() in {".txt", ".xml"}:
        with open(text, "r", encoding="utf-8") as reader:
            return reader.read()
    return text


def _contextual_laugh_duration(text: str) -> float:
    short_mod = re.compile(r"^\s*(?:[a-z]+ly )?(?:briefly|shortly|once|quickly)", re.IGNORECASE)
    long_mod = re.compile(
        r"^\s*(?:[a-z]+ly )?(?:maniacally|heartily|uproariously|uncontrollably|hysterically|darkly|wickedly|evilly|loudly|long)|^\s*between phrases",
        re.IGNORECASE,
    )
    total = 0.0
    for pattern, base_duration in _LAUGH_VERBS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            context = text[match.end() : match.end() + 40]
            if short_mod.match(context):
                total += base_duration * 0.4
            elif long_mod.match(context):
                total += base_duration * 1.2
            else:
                total += base_duration

    for quoted in re.findall(r'"([^"]+)"', text) + re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text):
        for run in re.findall(r"(?:h[ae]){3,}|(?:h[ae][ \-]?){3,}", quoted, re.IGNORECASE):
            syllables = len(re.findall(r"h[ae]", run, re.IGNORECASE))
            total += 0.2 * max(syllables - 2, 0)
    return total


def _estimate_nonverbal_duration(text: str) -> float:
    patterns = {
        r"\bsighs?\b": 0.8,
        r"\bshaky breath\b": 1.0,
        r"\bbreathing deeply\b": 1.0,
        r"\bgasps?\b": 0.5,
        r"\bburps?\b": 0.5,
        r"\byawns?\b": 1.0,
        r"\bpants?\b": 0.8,
        r"\bwheezes?\b": 0.8,
        r"\bcoughs?\b": 0.8,
        r"\bsniffles?\b": 0.5,
        r"\bsnorts?\b": 0.3,
        r"\bgroans?\b": 0.8,
        r"\blong pause\b": 1.0,
        r"\bpauses? briefly\b": 0.3,
        r"\bpauses?\b": 0.5,
        r"\bsilence\b": 1.0,
        r"\blets? the .{1,20} hang\b": 1.0,
        r"\blets? .{1,20} sink in\b": 1.0,
        r"\bslams?\b": 0.5,
        r"\bclaps?\b": 0.3,
        r"\bdraws? (?:his|her|a) sword\b": 0.5,
        r"\btakes? a (?:drag|swig|sip|drink)\b": 0.5,
        r"\bwhistles?\b": 1.0,
        r"\bhums?\b": 0.8,
        r"\bmutters?\b": 1.5,
        r"\bmumbles?\b": 1.0,
        r"\bwhispers?\b": 0.0,
        r"\bclears? (?:his|her) throat\b": 0.5,
        r"\bgulps?\b": 0.5,
        r"\bswallows?\b": 0.5,
        r"\bvoice (?:breaks?|cracks?|trembles?|drops?|rises?)\b": 0.5,
        r"\bsteadies? (?:him|her)self\b": 1.0,
        r"\bcatches? (?:his|her) breath\b": 1.0,
        r"\bcomposes? (?:him|her)self\b": 0.8,
        r"\bdemeanor shifts?\b": 0.5,
        r"\bsettles? in\b": 0.5,
        r"\bleans? in\b": 0.3,
        r"\bwipes? (?:his|her) eyes\b": 0.5,
    }
    extra = 0.0
    for pattern, duration in patterns.items():
        extra += duration * len(re.findall(pattern, text, re.IGNORECASE))
    return extra + _contextual_laugh_duration(text)


def estimate_speech_duration(text: str, speed: float = 1.0) -> float:
    quotes = re.findall(r'"([^"]+)"', text)
    if not quotes:
        quotes = re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text)
        quotes = [quote for quote in quotes if len(quote.split()) > 3]
    if quotes:
        spoken = " ".join(quotes)
    elif ":" in text:
        spoken = text.split(":", 1)[1].strip()
    else:
        spoken = text

    chars_per_second = 14.0
    text_length = len(spoken)
    if text_length < 40:
        chars_per_second *= 0.6
    elif text_length < 80:
        chars_per_second *= 0.8
    chars_per_second *= speed

    duration = text_length / chars_per_second
    duration += (spoken.count(".") + spoken.count("!") + spoken.count("?")) * 0.3
    duration += _estimate_nonverbal_duration(text)
    return max(3.0, round(duration + 2.0, 1))


def _normalize_speaker_id(value) -> int:
    try:
        match = re.search(r"\d+", str(value if value is not None else "1"))
        return max(1, int(match.group(0))) if match else 1
    except Exception:
        return 1


def _has_speaker_headers(text: str) -> bool:
    return re.search(r"(?im)^\s*Speaker\s*\d+\s*(?:\{[^\n{}]*\})?\s*:", text or "") is not None


def _speaker_prefix(speaker: int, attrs: dict) -> str:
    voice = _clean_spaces(attrs.get("voice", ""))
    gender = _clean_spaces(attrs.get("gender", "")).lower()
    scene = _clean_spaces(attrs.get("scene", ""))
    parts = []
    if voice:
        parts.append(voice)
    elif gender == "female":
        parts.append("female speaker")
    elif gender == "male":
        parts.append("male speaker")
    elif speaker:
        parts.append(f"speaker {speaker}")
    if scene:
        parts.append(f"in {scene}")
    return ". ".join(parts)


def _format_dramabox_segment_prompt(text: str, speaker: int, attrs: dict) -> str:
    text = _clean_spaces(text)
    if not text:
        return ""
    prefix = _speaker_prefix(speaker, attrs)
    if '"' not in text:
        spoken = text.strip(" .")
        text = f'says, "{spoken}."'
    return _clean_spaces(f"{prefix}. {text}" if prefix else text)


def _extract_complete_quoted_speech(text: str) -> str:
    raw = str(text or "")
    if raw.count('"') < 2 or raw.count('"') % 2 != 0:
        return ""
    return _clean_spaces(" ".join(quote.strip() for quote in re.findall(r'"([^"]+)"', raw) if quote.strip()))


def _parse_dramabox_segments(text: str) -> list[tuple[int, str, str]]:
    raw = str(text or "").strip()
    if not raw:
        return []
    has_headers = _has_speaker_headers(raw)
    if not has_headers:
        return [(1, _format_dramabox_segment_prompt(line.strip(), 1, {}), _extract_complete_quoted_speech(line)) for line in raw.splitlines() if line.strip()]

    header = re.compile(r"^\s*Speaker\s*(\d+)\s*(\{[^\n{}]*\})?\s*:\s*(.*)$", re.IGNORECASE)
    speaker_attrs: dict[int, dict] = {}
    current_speaker = 1
    segments: list[tuple[int, str, str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = header.match(stripped)
        if match:
            current_speaker = _normalize_speaker_id(match.group(1))
            attrs = speaker_attrs.setdefault(current_speaker, {})
            parsed = _parse_speaker_options(match.group(2))
            if parsed:
                attrs.update(parsed)
            stripped = match.group(3).strip()
            if not stripped:
                continue
        attrs = speaker_attrs.setdefault(current_speaker, {})
        expected_text = _extract_complete_quoted_speech(stripped)
        prompt = _format_dramabox_segment_prompt(stripped, current_speaker, attrs)
        if prompt:
            segments.append((current_speaker, prompt, expected_text))
    return segments


def _scale_segment_durations(durations: list[float], duration_seconds) -> list[float]:
    try:
        target_duration = float(duration_seconds or 0.0)
    except (TypeError, ValueError):
        target_duration = 0.0
    if target_duration <= 0 or not durations:
        return durations
    if len(durations) == 1:
        return [target_duration]
    total = sum(durations)
    if total <= 0:
        return durations
    scale = target_duration / total
    return [max(1.0, round(duration * scale, 1)) for duration in durations]


def _plan_dramabox_segments(text: str, seed: int, duration_seconds, duration_multiplier: float) -> list[_DramaBoxSegment]:
    parsed = _parse_dramabox_segments(text)
    durations = [max(1.0, round(estimate_speech_duration(prompt) * float(duration_multiplier), 1)) for _, prompt, _ in parsed]
    durations = _scale_segment_durations(durations, duration_seconds)
    return [
        _DramaBoxSegment(prompt=prompt, duration_s=duration, seed=seed + index * 1000, speaker=speaker, expected_text=expected_text)
        for index, ((speaker, prompt, expected_text), duration) in enumerate(zip(parsed, durations))
    ]


def _clean_segment_audio(audio: torch.Tensor, sample_rate: int, debug: bool = False) -> torch.Tensor:
    original_device = audio.device
    original_dtype = audio.dtype
    audio_np = _audio_tensor_to_numpy(audio)
    audio_np = trim_leading_transient_noise(audio_np, sample_rate, max_transient_seconds=DRAMABOX_LEADING_TRANSIENT_MAX_SECONDS, threshold=DRAMABOX_TRANSIENT_SILENCE_THRESHOLD, debug=debug, label="DramaBox Audio")
    audio_np = trim_leading_noise_before_speech(audio_np, sample_rate, speech_threshold=DRAMABOX_LEADING_SPEECH_THRESHOLD, max_leading_seconds=DRAMABOX_MAX_LEADING_SECONDS, debug=debug, label="DramaBox Audio")
    audio_np = trim_trailing_transient_noise(audio_np, sample_rate, max_transient_seconds=DRAMABOX_TRANSIENT_MAX_SECONDS, threshold=DRAMABOX_TRANSIENT_SILENCE_THRESHOLD, debug=debug, label="DramaBox Audio")
    audio_np = _trim_silence(audio_np, sample_rate, max_silence=0.5)
    audio_np = _normalize_volume(audio_np)
    audio_np = mute_isolated_transient_noise(audio_np, sample_rate, max_transient_seconds=DRAMABOX_TRANSIENT_MAX_SECONDS, threshold=DRAMABOX_ISOLATED_TRANSIENT_THRESHOLD, debug=debug, label="DramaBox Audio")
    return _numpy_to_audio_tensor(audio_np).to(device=original_device, dtype=original_dtype).clamp_(-1.0, 1.0)


def _concatenate_dramabox_segments(chunks: list[torch.Tensor], sample_rate: int, debug: bool = False) -> torch.Tensor:
    if not chunks:
        raise ValueError("No DramaBox Audio segments were generated.")
    processed = [_audio_tensor_to_numpy(chunk) for chunk in chunks]
    audio_np = np.concatenate(processed, axis=0)
    audio_np = _shorten_long_silence(audio_np, sample_rate, max_duration=0.8, target_duration=0.35, threshold_db=-30.0)
    audio_np = trim_trailing_transient_noise(audio_np, sample_rate, max_transient_seconds=DRAMABOX_TRANSIENT_MAX_SECONDS, threshold=DRAMABOX_TRANSIENT_SILENCE_THRESHOLD, debug=debug, label="DramaBox Audio")
    return _numpy_to_audio_tensor(audio_np).clamp_(-1.0, 1.0)


def _load_dramabox_alignment_whisper():
    from shared.deepy.transcription import _load_whisper_medium

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alignment_whisper = _load_whisper_medium(device)
    alignment_heads = alignment_whisper.alignment_heads
    del alignment_whisper._buffers["alignment_heads"]
    object.__setattr__(alignment_whisper, "alignment_heads", alignment_heads)
    for module in alignment_whisper.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module._lock_dtype = torch.float32
    alignment_whisper._offload_hooks = ["transcribe"]
    alignment_whisper._model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    alignment_whisper.eval().requires_grad_(False)
    return alignment_whisper


class DramaBoxAudioPipeline(LTXAudioTTSPipelineBase):
    def __init__(
        self,
        model_weights_path: str,
        gemma_path: str,
        audio_vae_path: str,
        vocoder_path: str,
        text_projection_path: str,
        text_connector_path: str,
        config_path: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            model_weights_path=model_weights_path,
            gemma_path=gemma_path,
            audio_vae_path=audio_vae_path,
            vocoder_path=vocoder_path,
            text_projection_path=text_projection_path,
            text_connector_path=text_connector_path,
            config_path=config_path,
            device=device,
            dtype=dtype,
        )

    def _encode_fixed_reference_waveform(self, waveform: torch.Tensor, sample_rate: int, *, tail: bool = False):
        reference_seconds = DRAMABOX_DEFAULT_REFERENCE_SECONDS
        target_samples = max(1, int(round(float(reference_seconds) * sample_rate)))
        if waveform.shape[-1] > target_samples:
            waveform = waveform[:, -target_samples:] if tail else waveform[:, :target_samples]
        elif waveform.shape[-1] < target_samples:
            repeat = (target_samples // max(1, waveform.shape[-1])) + 1
            waveform = waveform.repeat(1, repeat)
            waveform = waveform[:, :target_samples]
        target_peak = 10 ** (DRAMABOX_REFERENCE_PEAK_DB / 20.0)
        return self._encode_reference_waveform(waveform, sample_rate, max_seconds=reference_seconds, normalize_peak=target_peak)

    def _encode_voice_reference(self, input_waveform, input_waveform_sample_rate, audio_guide: str | None):
        waveform, sample_rate = self._waveform_from_input(input_waveform, input_waveform_sample_rate, audio_guide)
        if waveform is None or sample_rate <= 0:
            return None
        return self._encode_fixed_reference_waveform(waveform, sample_rate)

    def _encode_generated_tail_reference(self, audio: torch.Tensor, sample_rate: int):
        channels_first = audio.detach().cpu().float()
        if channels_first.ndim == 3:
            channels_first = channels_first.squeeze(0)
        if channels_first.ndim == 1:
            channels_first = channels_first.unsqueeze(0)
        return self._encode_fixed_reference_waveform(channels_first, sample_rate, tail=True)

    @staticmethod
    def _patch_long_clip_silence_prior(audio_state):
        latent = audio_state.latent
        if latent.shape[2] <= 513:
            return audio_state
        f0, f1 = 511, 514
        span = f1 - f0
        patched = latent.clone()
        for frame in (512, 513):
            amount = (frame - f0) / span
            patched[:, :, frame, :] = (1.0 - amount) * latent[:, :, f0, :] + amount * latent[:, :, f1, :]
        return replace(audio_state, latent=patched)

    def _target_duration(self, prompt: str, duration_seconds, duration_multiplier: float) -> float:
        try:
            explicit_duration = float(duration_seconds or 0)
        except (TypeError, ValueError):
            explicit_duration = 0.0
        if explicit_duration > 0:
            return explicit_duration
        return max(1.0, round(estimate_speech_duration(prompt) * float(duration_multiplier), 1))

    def _generate_segment_audio(
        self,
        segment: _DramaBoxSegment,
        negative_prompt: str,
        cfg_scale: float,
        stg_scale: float,
        rescale_scale: float,
        sampling_steps: int,
        ref_latent=None,
        callback=None,
        set_progress_status=None,
        status_extra: str = "",
    ) -> torch.Tensor | None:
        if set_progress_status is not None:
            set_progress_status(f"Encoding Prompt | {status_extra}" if status_extra else "Encoding Prompt")
        if cfg_scale > 1.0:
            audio_context, audio_context_n = self._encode_prompts([segment.prompt, negative_prompt])
        else:
            audio_context = self._encode_prompt(segment.prompt)
            audio_context_n = None
        if self._interrupt or self._early_stop_requested():
            return None

        audio_state, audio_tools = self._build_audio_state(
            segment.duration_s,
            DRAMABOX_FPS,
            torch.empty(0, dtype=torch.float32, device=self.device),
            segment.seed,
            ref_latent=ref_latent,
            reference_conditioner=AudioConditionByAppendedReferenceLatent,
        )
        sigmas = LTX2Scheduler().execute(steps=max(1, int(sampling_steps or DRAMABOX_DEFAULT_STEPS)), latent=audio_state.latent).to(self.device)
        audio_state = self._generate_audio_euler(
            audio_context,
            sigmas,
            audio_state,
            audio_tools,
            audio_context_n=audio_context_n,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            stg_blocks=[DRAMABOX_STG_BLOCK],
            rescale_scale=rescale_scale,
            callback=callback,
            status_extra=status_extra,
            set_progress_status=set_progress_status,
        )
        if audio_state is None:
            return None
        audio_state = self._patch_long_clip_silence_prior(audio_state)
        return self._decode_audio_state(audio_state, set_progress_status=set_progress_status, status_extra=status_extra)

    def _remove_unexpected_words(
        self,
        generated_segments: list[tuple[_DramaBoxSegment, torch.Tensor]],
        sample_rate: int,
        *,
        debug_prompt: bool = False,
        set_progress_status=None,
    ) -> list[tuple[_DramaBoxSegment, torch.Tensor]]:
        if not any(segment.expected_text for segment, _ in generated_segments):
            return generated_segments
        if set_progress_status is not None:
            set_progress_status("Loading Whisper Alignment")
        for model in (self.model, self.text_encoder, self.text_embedding_projection, self.video_embeddings_connector, self.audio_embeddings_connector, self.audio_encoder, self.audio_decoder, self.vocoder):
            self._unload_managed_model(model)
        alignment_whisper = _load_dramabox_alignment_whisper()
        processed: list[tuple[_DramaBoxSegment, torch.Tensor]] = []
        try:
            for index, (segment, audio) in enumerate(generated_segments):
                if self._interrupt:
                    processed.extend(generated_segments[index:])
                    break
                if not segment.expected_text:
                    processed.append((segment, audio))
                    continue
                if set_progress_status is not None:
                    set_progress_status(f"Removing Unexpected Words | Segment {index + 1}/{len(generated_segments)}")
                trimmed = _trim_leading_extra_words_tensor(alignment_whisper, audio, sample_rate, segment.expected_text, "en", debug_prompt=debug_prompt, label="DramaBox Audio")
                processed.append((segment, _clean_segment_audio(trimmed, sample_rate, debug=debug_prompt)))
        finally:
            self._unload_managed_model(alignment_whisper)
            try:
                alignment_whisper.to("cpu")
            except Exception:
                pass
            del alignment_whisper
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return processed

    def generate(
        self,
        input_prompt: str,
        audio_guide: Optional[str] = None,
        sampling_steps: int = DRAMABOX_DEFAULT_STEPS,
        guide_scale: float = DRAMABOX_DEFAULT_CFG_SCALE,
        n_prompt=None,
        seed: int = -1,
        callback=None,
        audio_cfg_scale=None,
        input_waveform=None,
        input_waveform_sample_rate=None,
        audio_guide2: Optional[str] = None,
        audio_prompt_type: str = "",
        alt_scale=None,
        set_header_text=None,
        custom_settings=None,
        duration_seconds: Optional[float] = None,
        set_progress_status=None,
        verbose_level: int = 0,
        **kwargs
    ) -> Optional[dict]:
        self._interrupt = False
        self._early_stop = False
        prompt = _read_text_or_file(input_prompt, "Prompt").strip()
        if not prompt:
            raise ValueError("Prompt text cannot be empty for DramaBox Audio.")

        seed = random.randrange(0, 2**31) if seed is None or int(seed) < 0 else int(seed)
        duration_multiplier = self._custom_float(custom_settings, "duration_multiplier", DRAMABOX_DEFAULT_DURATION_MULTIPLIER)
        stg_scale = DRAMABOX_DEFAULT_STG_SCALE if audio_cfg_scale is None else float(audio_cfg_scale)
        rescale_scale = 0.0 if alt_scale is None else float(alt_scale)
        cfg_scale = float(guide_scale)
        debug_prompt = verbose_level > 1

        if set_progress_status is not None:
            set_progress_status("Planning Audio Segments")
        segments = _plan_dramabox_segments(prompt, seed, duration_seconds, duration_multiplier)
        if not segments:
            raise ValueError("DramaBox Audio prompt produced no segments.")

        negative_prompt = _read_text_or_file(n_prompt, "Negative prompt").strip() or DRAMABOX_DEFAULT_NEGATIVE_PROMPT
        audio_prompt_type = str(audio_prompt_type or "").upper()
        remove_unexpected_words = "0" in audio_prompt_type
        speaker_ref_latents = {}
        if "A" in audio_prompt_type or audio_guide is not None or input_waveform is not None:
            if set_progress_status is not None:
                set_progress_status("Encoding Speaker 1 Reference")
            speaker_ref_latents[1] = self._encode_voice_reference(input_waveform, input_waveform_sample_rate, audio_guide)
            if speaker_ref_latents[1] is None:
                raise ValueError("DramaBox Audio Speaker 1 reference mode requires a reference audio file.")
        if "B" in audio_prompt_type or audio_guide2 is not None:
            if set_progress_status is not None:
                set_progress_status("Encoding Speaker 2 Reference")
            speaker_ref_latents[2] = self._encode_voice_reference(None, None, audio_guide2)
            if speaker_ref_latents[2] is None:
                raise ValueError("DramaBox Audio Speaker 2 reference mode requires a second reference audio file.")

        if self._interrupt:
            return None

        duration = sum(segment.duration_s for segment in segments)
        if set_header_text is not None:
            set_header_text(f"DramaBox Audio - {len(segments)} segment{'s' if len(segments) != 1 else ''}, {duration:.1f}s")

        output_audio_sampling_rate = int(getattr(self.vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE))
        generated_segments: list[tuple[_DramaBoxSegment, torch.Tensor]] = []
        generated_ref_latents = {}
        anchored_ref_speakers = set(speaker_ref_latents)
        for index, segment in enumerate(segments):
            if self._interrupt:
                break
            if self._early_stop_requested() and generated_segments:
                break
            status_extra = f"Segment {index + 1}/{len(segments)}"
            ref_latent = speaker_ref_latents.get(segment.speaker)
            if ref_latent is None:
                ref_latent = generated_ref_latents.get(segment.speaker)
            audio = self._generate_segment_audio(
                segment,
                negative_prompt,
                cfg_scale,
                stg_scale,
                rescale_scale,
                sampling_steps,
                ref_latent=ref_latent,
                callback=callback,
                set_progress_status=set_progress_status,
                status_extra=status_extra,
            )
            if audio is None:
                if generated_segments and (self._interrupt or self._early_stop_requested()):
                    break
                return None
            if set_progress_status is not None:
                set_progress_status(f"Trimming Segment {index + 1}/{len(segments)}")
            audio = _clean_segment_audio(audio, output_audio_sampling_rate, debug=debug_prompt)
            generated_segments.append((segment, audio))
            if segment.speaker not in anchored_ref_speakers and segment.speaker not in generated_ref_latents:
                generated_ref_latents[segment.speaker] = self._encode_generated_tail_reference(audio, output_audio_sampling_rate)
            if self._early_stop_requested():
                break

        if not generated_segments:
            return None

        if remove_unexpected_words and not self._interrupt:
            generated_segments = self._remove_unexpected_words(generated_segments, output_audio_sampling_rate, debug_prompt=debug_prompt, set_progress_status=set_progress_status)

        if set_progress_status is not None:
            set_progress_status("Combining Audio Segments")
        audio = _concatenate_dramabox_segments([audio for _, audio in generated_segments], output_audio_sampling_rate, debug=debug_prompt)
        return {"x": audio, "audio_sampling_rate": output_audio_sampling_rate}
