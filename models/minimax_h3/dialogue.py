"""Dialogue planning, H3 prompt compilation, and Whisper boundary alignment."""

from __future__ import annotations

import math
import random
import re
import unicodedata
from dataclasses import dataclass, replace

import torch
import torchaudio.functional as audio_F


# Python switch for the simplified ``Speaker N:`` dialogue workflow.
H3_DIALOGUE_GENERATION = True
H3_DIALOGUE_MIN_SEGMENT_SECONDS = 4.0
H3_DIALOGUE_MAX_SEGMENT_SECONDS = 45.0
H3_DIALOGUE_MAX_TOTAL_SECONDS = 5.0 * 60.0
H3_DIALOGUE_MAX_REFERENCE_SECONDS = 15.0
H3_DIALOGUE_AUDIO_SAMPLE_RATE = 32000
H3_DIALOGUE_PAUSE_SECONDS = 0.18
H3_DIALOGUE_WORDS_PER_SECOND = 2.6
H3_DIALOGUE_BOUNDARY_PADDING_SECONDS = 0.14
H3_DIALOGUE_SILENCE_THRESHOLD = 0.012

H3_DIALOGUE_PROMPT_INFOS = """
### Dialogue script mode

Write one turn per `Speaker N:` block. Put acting, language, emotion, pace, or microphone directions in square brackets; bracketed text is not spoken.

```text
Speaker 1:
[French, calm and close to the microphone] Je savais que tu reviendrais.
Speaker 2:
[French, worried, speaking quickly] Comment pouvais-tu en etre aussi sure ?
Speaker 1:
[whispering] Parce que tu as garde la cle.
```

WanGP generates every turn as a separate H3 segment, dynamically compiles the full six-section Ref2VA prompt, removes unexpected speech before and after the requested line with Whisper, then joins the turns. Audio Reference 1 belongs to Speaker 1 and Audio Reference 2 to Speaker 2. A speaker without an uploaded reference uses their first generated turn as the voice reference for later turns.
"""

_SPEAKER_HEADER = re.compile(r"^\s*Speaker\s*(\d+)\s*(?:\{([^{}\n]*)\})?\s*:\s*", re.IGNORECASE | re.MULTILINE)
_DIRECTION = re.compile(r"\[([^\[\]\n]+)\]")
_LANGUAGES = {
    "ar": ("Arabic", "ar"), "arabic": ("Arabic", "ar"), "arabe": ("Arabic", "ar"),
    "zh": ("Chinese", "zh"), "chinese": ("Chinese", "zh"), "chinois": ("Chinese", "zh"), "mandarin": ("Chinese", "zh"),
    "nl": ("Dutch", "nl"), "dutch": ("Dutch", "nl"), "neerlandais": ("Dutch", "nl"),
    "en": ("English", "en"), "english": ("English", "en"), "anglais": ("English", "en"),
    "fr": ("French", "fr"), "french": ("French", "fr"), "francais": ("French", "fr"),
    "de": ("German", "de"), "german": ("German", "de"), "allemand": ("German", "de"),
    "hi": ("Hindi", "hi"), "hindi": ("Hindi", "hi"),
    "it": ("Italian", "it"), "italian": ("Italian", "it"), "italien": ("Italian", "it"),
    "ja": ("Japanese", "ja"), "japanese": ("Japanese", "ja"), "japonais": ("Japanese", "ja"),
    "ko": ("Korean", "ko"), "korean": ("Korean", "ko"), "coreen": ("Korean", "ko"),
    "pl": ("Polish", "pl"), "polish": ("Polish", "pl"), "polonais": ("Polish", "pl"),
    "pt": ("Portuguese", "pt"), "portuguese": ("Portuguese", "pt"), "portugais": ("Portuguese", "pt"),
    "ru": ("Russian", "ru"), "russian": ("Russian", "ru"), "russe": ("Russian", "ru"),
    "es": ("Spanish", "es"), "spanish": ("Spanish", "es"), "espagnol": ("Spanish", "es"),
    "tr": ("Turkish", "tr"), "turkish": ("Turkish", "tr"), "turc": ("Turkish", "tr"),
}


@dataclass(frozen=True)
class DialogueTurn:
    speaker: int
    text: str
    direction: str
    language_name: str
    language_code: str | None
    duration_s: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class _VoiceReference:
    waveform: torch.Tensor | None = None
    sample_rate: int | None = None


def _normalized_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z]+", " ", text).strip()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _parse_header_directions(raw: str | None) -> list[str]:
    directions = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        directions.append(item.split("=", 1)[1].strip() if "=" in item else item)
    return directions


def _extract_language(directions: list[str]) -> tuple[str, str | None, list[str]]:
    remaining = []
    language_name, language_code = "Original language", None
    for direction in directions:
        key = _normalized_key(direction)
        if language_code is None and key in _LANGUAGES:
            language_name, language_code = _LANGUAGES[key]
        else:
            remaining.append(direction)
    return language_name, language_code, remaining


def is_dialogue_prompt(prompt: str) -> bool:
    return _SPEAKER_HEADER.search(str(prompt or "")) is not None


def parse_dialogue_prompt(prompt: str) -> list[DialogueTurn]:
    raw = str(prompt or "").strip()
    headers = list(_SPEAKER_HEADER.finditer(raw))
    if not headers:
        return []
    turns = []
    speaker_languages = {}
    for index, header in enumerate(headers):
        body = raw[header.end():headers[index + 1].start() if index + 1 < len(headers) else len(raw)].strip()
        directions = _parse_header_directions(header.group(2))
        for item in _DIRECTION.findall(body):
            directions.extend(_parse_header_directions(item))
        text = _clean_text(_DIRECTION.sub(" ", body)).strip('"\'\u201c\u201d\u2018\u2019 ')
        if not text:
            raise ValueError(f"MiniMax H3 dialogue Speaker {header.group(1)} has no spoken text")
        language_name, language_code, directions = _extract_language(directions)
        speaker = max(1, int(header.group(1)))
        if language_code is None and speaker in speaker_languages:
            language_name, language_code = speaker_languages[speaker]
        elif language_code is not None:
            speaker_languages[speaker] = (language_name, language_code)
        turns.append(DialogueTurn(
            speaker=speaker,
            text=text,
            direction="; ".join(directions) or "natural, expressive delivery",
            language_name=language_name,
            language_code=language_code,
        ))
    return turns


def _natural_duration(turn: DialogueTurn) -> float:
    words = len(_normalize_words(turn.text))
    punctuation = len(re.findall(r"[.!?;:]", turn.text)) * 0.18
    return max(H3_DIALOGUE_MIN_SEGMENT_SECONDS, min(H3_DIALOGUE_MAX_SEGMENT_SECONDS, words / H3_DIALOGUE_WORDS_PER_SECOND + punctuation + 1.4))


def plan_dialogue(prompt: str, seed: int, duration_seconds: float | None) -> list[DialogueTurn]:
    turns = parse_dialogue_prompt(prompt)
    if not turns:
        return []
    base_seed = random.randrange(0, 2**31) if seed is None or int(seed) < 0 else int(seed)
    duration_limit = min(float(duration_seconds or 0.0), H3_DIALOGUE_MAX_TOTAL_SECONDS)
    planned, elapsed = [], 0.0
    for index, turn in enumerate(turns):
        pause = H3_DIALOGUE_PAUSE_SECONDS if planned else 0.0
        remaining = duration_limit - elapsed - pause
        if duration_limit > 0 and remaining < H3_DIALOGUE_MIN_SEGMENT_SECONDS:
            break
        duration = _natural_duration(turn)
        if duration_limit > 0:
            duration = min(duration, remaining)
        duration = round(duration, 2)
        planned.append(replace(turn, duration_s=duration, seed=(base_seed + index * 1000) % (2**31)))
        elapsed += pause + duration
    return planned


def build_h3_dialogue_prompt(turn: DialogueTurn, has_reference: bool) -> str:
    speaker = f"Speaker {turn.speaker} (S{turn.speaker})"
    direction = turn.direction.rstrip(". ")
    if has_reference:
        subject = f"<Audio 1> supplies the voice timbre, accent, cadence, and delivery identity for {speaker}; its words, timing, and background noise are not copied."
        summary_mode = "audio reference"
        retention = f"<Audio 1>: reference - retain only the stable vocal identity for (S{turn.speaker}); generate the requested words and performance anew."
        voice = f"using the voice identity from <Audio 1>, {direction}"
    else:
        subject = f"{speaker} is a newly created, distinct, stable voice used only for this dialogue turn."
        summary_mode = "audio generation"
        retention = "No external media is retained; create a clean, internally consistent voice and performance."
        voice = direction
    return "\n".join((
        "subject_definitions:",
        subject,
        "summary:",
        f"[{summary_mode}] Generate one clean isolated dialogue turn spoken only by {speaker}, with no extra words before or after it.",
        "retention_analysis:",
        retention,
        "detailed_description:",
        f"The hidden target video is a minimal neutral close-microphone studio shot. [Shot 1] {speaker}, {voice}, says exactly, <d>[{turn.language_name}] {turn.text}</d> The line is complete and no other dialogue is spoken.",
        "overall_soundscape:",
        "Clean close-microphone speech, subtle natural breaths, quiet neutral room tone, no overlapping voices, and no unrelated sounds.",
        "non_diegetic_music:",
        "N/A",
    ))


def load_dialogue_whisper() -> torch.nn.Module:
    from shared.deepy.transcription import _load_whisper_medium

    model = _load_whisper_medium(torch.device("cpu"))
    alignment_heads = model.alignment_heads
    del model._buffers["alignment_heads"]
    object.__setattr__(model, "alignment_heads", alignment_heads)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module._lock_dtype = torch.float32
    model._offload_hooks = ["transcribe"]
    model._model_dtype = torch.float16
    model._budget = 0
    return model.eval().requires_grad_(False)


def _normalize_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().casefold()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalized)


def _fuzzy_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    distances = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        diagonal = distances[0]
        distances[0] = row
        for column, right_char in enumerate(right, 1):
            previous = distances[column]
            distances[column] = diagonal if left_char == right_char else 1 + min(diagonal, distances[column], distances[column - 1])
            diagonal = previous
    return 1.0 - distances[-1] / max(len(left), len(right)) >= 0.55


def _align_words(transcribed: list[str], expected: list[str]) -> list[tuple[str, int | None]]:
    rows, columns, gap = len(transcribed), len(expected), -1
    scores = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        scores[row][0] = scores[row - 1][0] + gap
    for column in range(1, columns + 1):
        scores[0][column] = scores[0][column - 1] + gap
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            diagonal = scores[row - 1][column - 1] + (2 if _fuzzy_match(transcribed[row - 1], expected[column - 1]) else -1)
            scores[row][column] = max(diagonal, scores[row - 1][column] + gap, scores[row][column - 1] + gap)
    aligned = []
    row, column = rows, columns
    while row or column:
        match_score = 2 if row and column and _fuzzy_match(transcribed[row - 1], expected[column - 1]) else -1
        if row and column and scores[row][column] == scores[row - 1][column - 1] + match_score:
            aligned.append(("match" if match_score == 2 else "substitution", column - 1))
            row -= 1
            column -= 1
        elif row and scores[row][column] == scores[row - 1][column] + gap:
            aligned.append(("insertion", None))
            row -= 1
        else:
            column -= 1
    return list(reversed(aligned))


def _transcribe_words(model: torch.nn.Module, audio: torch.Tensor, sample_rate: int, language: str | None) -> list[dict]:
    mono = audio.detach().float().mean(dim=0, keepdim=True).cpu()
    if int(sample_rate) != 16000:
        mono = audio_F.resample(mono, int(sample_rate), 16000)
    result = model.transcribe(mono.squeeze(0).numpy(), language=language, word_timestamps=True,
                              fp16=getattr(model, "_model_dtype", torch.float32) == torch.float16, verbose=None)
    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            for normalized in _normalize_words(word.get("word", "")):
                words.append({"word": normalized, "start": float(word.get("start", 0.0)), "end": float(word.get("end", 0.0)),
                              "probability": float(word.get("probability", 1.0))})
    return words


def _reliable_boundary_word(word: dict) -> bool:
    duration = word["end"] - word["start"]
    return word["probability"] >= 0.35 and duration <= 0.5 + 0.15 * len(word["word"])


def _quiet_boundary(mono: torch.Tensor, sample_rate: int, start: int, stop: int, prefer_last: bool) -> int | None:
    start, stop = max(0, start), min(mono.numel(), stop)
    hop = max(1, round(sample_rate * 0.01))
    quiet = []
    for position in range(start, stop, hop):
        window = mono[position:min(stop, position + hop)]
        if window.numel() and float(window.square().mean().sqrt()) < H3_DIALOGUE_SILENCE_THRESHOLD:
            quiet.append(position)
    if not quiet:
        return None
    return quiet[-1] if prefer_last else quiet[0]


def trim_dialogue_surplus(model: torch.nn.Module, audio: torch.Tensor, sample_rate: int, expected_text: str,
                          language: str | None, verbose: bool = False) -> torch.Tensor:
    expected = _normalize_words(expected_text)
    transcribed = _transcribe_words(model, audio, sample_rate, language)
    if not expected or not transcribed:
        return audio
    alignment = _align_words([word["word"] for word in transcribed], expected)
    matching = sum(label == "match" for label, _ in alignment)
    if matching < max(1, math.ceil(len(expected) * 0.5)):
        if verbose:
            print("[MiniMax H3 Dialogue] Whisper alignment was not confident enough to trim this segment")
        return audio
    first = next((index for index, (_, expected_index) in enumerate(alignment) if expected_index == 0), None)
    last = next((index for index in range(len(alignment) - 1, -1, -1) if alignment[index][1] == len(expected) - 1), None)
    if first is None or last is None:
        return audio
    reliable_first = next((index for index in range(first, last + 1)
                           if alignment[index][0] == "match" and _reliable_boundary_word(transcribed[index])), first)
    unreliable_leading_words = reliable_first > first
    first = reliable_first
    mono = audio.detach().float().mean(dim=0).cpu()
    start_sample, stop_sample = 0, audio.shape[-1]
    leading_insertion = first > 0 and any(label == "insertion" for label, _ in alignment[:first])
    if unreliable_leading_words or leading_insertion or transcribed[first]["start"] > 0.5:
        intended_start = round(transcribed[first]["start"] * sample_rate)
        padding = round(H3_DIALOGUE_BOUNDARY_PADDING_SECONDS * sample_rate)
        search_start = max(0, intended_start - round(0.5 * sample_rate))
        start_sample = _quiet_boundary(mono, sample_rate, search_start, max(search_start + 1, intended_start - padding), True)
        start_sample = max(0, intended_start - padding) if start_sample is None else start_sample
    trailing_duration = audio.shape[-1] / sample_rate - transcribed[last]["end"]
    trailing_insertion = last + 1 < len(transcribed) and any(label == "insertion" for label, _ in alignment[last + 1:])
    if trailing_insertion or trailing_duration > 0.5:
        intended_end = round(transcribed[last]["end"] * sample_rate)
        padding = round(H3_DIALOGUE_BOUNDARY_PADDING_SECONDS * sample_rate)
        search_stop = min(audio.shape[-1], intended_end + round(0.5 * sample_rate))
        padded_end = min(search_stop, intended_end + padding)
        stop_sample = _quiet_boundary(mono, sample_rate, padded_end, search_stop, False)
        stop_sample = min(audio.shape[-1], intended_end + padding) if stop_sample is None else max(padded_end, stop_sample)
    if stop_sample <= start_sample:
        return audio
    if verbose and (start_sample or stop_sample < audio.shape[-1]):
        print(f"[MiniMax H3 Dialogue] Whisper trimmed {start_sample / sample_rate:.2f}s before and {(audio.shape[-1] - stop_sample) / sample_rate:.2f}s after the expected speech")
    return audio[..., start_sample:stop_sample]


def _generated_reference(audio: torch.Tensor, sample_rate: int) -> _VoiceReference:
    samples = round(H3_DIALOGUE_MAX_REFERENCE_SECONDS * sample_rate)
    return _VoiceReference(waveform=audio[..., -samples:].detach().float().cpu(), sample_rate=sample_rate)


def generate_dialogue(pipeline, input_prompt: str, *, audio_guide=None, audio_guide2=None, input_waveform=None,
                      input_waveform_sample_rate=None, audio_prompt_type="", duration_seconds=None,
                      sampling_steps=20, seed=0, shift=12.0, callback=None, VAE_tile_size=None, fps=24,
                      sample_solver="euler", attention_sparsity=1.0, loras_slists=None, loras_selected=None,
                      custom_settings=None, set_progress_status=None, verbose_level=0) -> dict | None:
    turns = plan_dialogue(input_prompt, seed, duration_seconds)
    if not turns:
        raise ValueError("MiniMax H3 dialogue prompt produced no Speaker N segments")
    if set_progress_status is not None:
        set_progress_status(f"Planning H3 dialogue ({len(turns)} segments)")
    prompt_type = str(audio_prompt_type or "").upper()
    reference_speakers, reference_sources = [], []
    if "A" in prompt_type:
        reference_speakers.append(1)
        reference_sources.append(audio_guide if audio_guide is not None else pipeline._waveform(input_waveform, input_waveform_sample_rate))
    if "B" in prompt_type:
        reference_speakers.append(2)
        reference_sources.append(audio_guide2)
    reference_waveforms = pipeline._prepare_audio_references(reference_sources)
    references = {speaker: _VoiceReference(waveform=waveform[0].detach().to(device="cpu", non_blocking=False), sample_rate=H3_DIALOGUE_AUDIO_SAMPLE_RATE)
                  for speaker, waveform in zip(reference_speakers, reference_waveforms)}
    duration_limit = min(float(duration_seconds or 0.0), H3_DIALOGUE_MAX_TOTAL_SECONDS)
    generated, generated_duration = [], 0.0
    for index, turn in enumerate(turns):
        pipeline._check_abort()
        if pipeline._early_stop_requested() and generated:
            break
        pause_duration = H3_DIALOGUE_PAUSE_SECONDS if generated else 0.0
        remaining = duration_limit - generated_duration - pause_duration
        if duration_limit > 0 and remaining < H3_DIALOGUE_MIN_SEGMENT_SECONDS:
            break
        if duration_limit > 0 and turn.duration_s > remaining:
            turn = replace(turn, duration_s=round(remaining, 2))
        reference = references.get(turn.speaker)

        segment_label = f"Segment {index + 1}/{len(turns)}"

        def segment_status(message, label=segment_label):
            if set_progress_status is not None:
                set_progress_status(f"{label} | {message}")

        def segment_callback(*args, label=segment_label, **kwargs):
            kwargs["status_prefix"] = label
            return callback(*args, **kwargs)

        result = pipeline.generate(
            input_prompt=build_h3_dialogue_prompt(turn, reference is not None),
            input_waveform=None if reference is None or reference.waveform is None else reference.waveform.transpose(0, 1),
            input_waveform_sample_rate=None if reference is None else reference.sample_rate,
            audio_guide=None,
            audio_prompt_type="A" if reference is not None else "",
            duration_seconds=turn.duration_s,
            sampling_steps=sampling_steps,
            seed=turn.seed,
            shift=shift,
            callback=segment_callback if callback is not None else None,
            VAE_tile_size=VAE_tile_size,
            fps=fps,
            sample_solver=sample_solver,
            attention_sparsity=attention_sparsity,
            guide_phases=1,
            loras_slists=loras_slists,
            loras_selected=loras_selected,
            custom_settings=custom_settings,
            set_progress_status=segment_status,
            dialogue_segment=True,
        )
        if result is None:
            return None
        audio = result["x"].detach().to(device="cpu", non_blocking=False)
        sample_rate = int(result["audio_sampling_rate"])
        generated.append((turn, audio))
        generated_duration += pause_duration + audio.shape[-1] / sample_rate
        references.setdefault(turn.speaker, _generated_reference(audio, sample_rate))
        if pipeline._early_stop_requested() or (duration_limit > 0 and generated_duration >= duration_limit):
            break
    if set_progress_status is not None:
        set_progress_status("Loading Whisper dialogue alignment")
    whisper = pipeline.dialogue_whisper
    aligned = []
    for index, (turn, audio) in enumerate(generated):
        pipeline._check_abort()
        if set_progress_status is not None:
            set_progress_status(f"Segment {index + 1}/{len(generated)} | Whisper dialogue alignment")
        trimmed = trim_dialogue_surplus(whisper, audio, sample_rate, turn.text, turn.language_code, verbose=verbose_level > 1)
        aligned.append(trimmed.detach().to(device="cpu", non_blocking=False))
    pause = torch.zeros((aligned[0].shape[0], round(H3_DIALOGUE_PAUSE_SECONDS * sample_rate)), dtype=aligned[0].dtype, device="cpu")
    pieces = [piece for index, audio in enumerate(aligned) for piece in ((pause,) if index else ()) + (audio,)]
    output = torch.cat(pieces, dim=-1)
    if duration_limit > 0:
        output = output[..., :round(duration_limit * sample_rate)]
    actual_duration = output.shape[-1] / sample_rate
    if set_progress_status is not None:
        set_progress_status(f"Combined H3 dialogue ({len(aligned)} segments, {actual_duration:.1f}s)")
    return {"x": output, "audio_sampling_rate": sample_rate,
            "overridden_inputs": {"resolution": "32x32", "video_length": round(actual_duration * float(fps)), "duration_seconds": round(actual_duration, 3)}}


__all__ = [
    "DialogueTurn", "H3_DIALOGUE_GENERATION", "H3_DIALOGUE_MAX_TOTAL_SECONDS", "H3_DIALOGUE_PROMPT_INFOS",
    "build_h3_dialogue_prompt", "generate_dialogue", "is_dialogue_prompt", "load_dialogue_whisper",
    "parse_dialogue_prompt", "plan_dialogue", "trim_dialogue_surplus",
]
