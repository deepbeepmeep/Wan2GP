from __future__ import annotations

import math
import os
import re
import tempfile

import numpy as np
import torch


JOYAI_CONTROL_MEMORY_SETTING = "joyai_control_memory_positions"
JOYAI_CONTROL_MEMORY_MAX_SECONDS = 60.0
JOYAI_AUDIO_SILENCE_DYNAMIC_RANGE_DB = 6.0
JOYAI_AUDIO_SILENCE_THRESHOLD_FRACTION = 0.35
JOYAI_ECHO_PROMPT_INFOS = """### JoyAI-Echo prompt format

Write one complete generation as plain text. Separate shots with one empty line. Each shot should be self-contained and repeat the important persistent details: character IDs, voices, wardrobe, objects, location, action, camera framing, dialogue, and sound design.

Use stable IDs such as `ID_A` and `ID_B` whenever a person, object, or place must persist across shots. JoyAI-Echo remembers compact moments from previous shots, but explicit prompt continuity is still important.

Add optional Joy shot options inside a shot only when needed: `[/duration=121]` for frames, `[/duration=5s]` for seconds, `[/overlap]` to reinject overlap frames from the previous shot, `[/overlap=0]` for a sharp transition, or `[/duration=5s,/nomem]` to generate a shot without recording it as future memory. If a shot has no duration marker, it uses the UI Video Length. Joy shot options are removed before text encoding. Other bracket syntax, including Prompt Relay markers, is preserved.

Only `/duration`, `/overlap`, and `/nomem` are valid Joy slash options. Any other `[/...]` command is rejected during validation.

Long example:

`[/duration=5s] ID_A is Mara, a cheerful disaster archivist in a cobalt raincoat, short black hair tucked behind one ear, silver round glasses, and a bright orange satchel marked with hand-written labels. ID_PLACE is the Midnight Ferry Terminal, a glass-and-brass station floating on dark water under violet city lights. ID_OBJECT is a tiny mechanical moon inside a cracked snow globe, glowing pale blue whenever it hears a lie. ID_A holds ID_OBJECT near her face so her mouth movement stays readable, smiles with nervous confidence, and says, "If this moon starts singing, we all pretend that was planned." The camera is a stable medium close-up with wet reflections, soft terminal announcements, distant waves, and a delicate ticking sound from ID_OBJECT.`

`ID_A is still Mara in the cobalt raincoat, silver glasses, and orange satchel, standing inside ID_PLACE, the Midnight Ferry Terminal. ID_OBJECT, the cracked snow-globe moon, is now balanced on a ticket counter between stacks of impossible ferry passes. At normal speed, ID_A taps the glass and the moon projects a miniature map of glowing canals across her hands. She whispers, "Good, it remembers the route I forgot." Keep the same violet city lights, rainy brass-and-glass station design, readable face, gentle comedic tension, terminal ambience, water sounds, and the ticking blue moon.`

`[/duration=6s] ID_B is Sol, a tall soft-spoken lighthouse engineer with warm brown skin, a moss-green work jacket, a knitted red scarf, and a tool belt full of polished copper instruments. ID_B enters ID_PLACE carrying ID_OBJECT_B, a lunchbox-sized foghorn that occasionally sneezes sparks. ID_B notices Mara's glowing snow-globe moon, raises an eyebrow, and says, "That thing owes my lighthouse an apology." The camera frames ID_B from the waist up, showing the scarf, tool belt, and foghorn clearly. Add warm dry humor, soft footsteps on wet tile, distant ferry bells, and the little foghorn's embarrassed squeak.`

`ID_A, Mara in the cobalt raincoat with silver glasses and orange satchel, and ID_B, Sol in the moss-green jacket with red scarf and copper tools, stand together at ID_PLACE. ID_OBJECT, the glowing cracked snow-globe moon, floats above ID_OBJECT_B, the tiny foghorn, and both devices argue in musical beeps while projecting a route over the dark water. Mara laughs and says, "I think our machines are flirting." Sol looks horrified but amused and replies, "Then they can buy their own ferry tickets." Keep both faces readable, both wardrobes consistent, the rainy violet terminal continuous, the blue moon glow, the foghorn sparks, lively dialogue timing, ferry bells, water ambience, and a playful magical-realism mood.`

The first Start Image or Continue Video applies to the first shot only; later shots continue from JoyAI-Echo memory. A Control Video with an audio track can also be used as artificial memory by selecting JoyAI-Echo Control Video Memory. Leave the positions field empty to auto-pick a non-silent moment from the first minute, or enter comma-separated memory positions such as `2s, 5s, 9s` or `49, 121, 217`.
"""

JOYAI_ECHO_INFOS = """JoyAI-Echo is an LTX-2.3 based multi-shot audio-video model. WanGP treats each Paragraph seberated by a blank-line block as one shot, generates them sequentially, and injects compact video/audio memory from earlier shots into later shots.

- Each remembered video point represents a very short moment: about 8 frames, roughly 320 ms at 25 fps. Generated-shot memory also keeps up to about 3.8 seconds of nearby voice/sound from the same shot.
- Optional per-shot controls use Joy slash syntax: `[/duration=5s]` changes one shot duration, `[/overlap]` reinjects the previous shot tail into the current shot, `[/overlap=0]` forces a sharp transition, and `[/nomem]` prevents that shot from being saved into future memory. Unknown `[/...]` commands are rejected.
- Optional Control Video Memory builds artificial memory before shot 1 from the first minute of a Control Video with an audio track. Leave memory positions empty to auto-pick a non-silent moment, or enter comma-separated positions; integers are frame positions, `Ns` values are seconds. Each frame position may correspond to a Person you will reference in your Text Prompt.
"""

JOYAI_ECHO_PROMPT_ENHANCER = """You are writing prompts for JoyAI-Echo, an LTX-2.3 based multi-shot audio-video model.

Return only the final prompt text. Do not return JSON, bullets, headings, commentary, or code fences.

Format:
- Write 2 to 6 cinematic shots.
- Separate shots with exactly one empty line.
- Each shot is a complete paragraph.
- Preserve any useful Prompt Relay bracket syntax the user requested.
- Add `[/duration=5s]` style markers only when the user asks for different per-shot durations; otherwise omit them and let the UI Video Length apply.
- Add `[/overlap]` only when the shot should start from overlap frames of the previous shot, or `[/overlap=0]` to force no overlap, including on shot 1 after a Continue Video.
- Add `[/nomem]` or combine it as `[/duration=5s,/nomem]` only when a shot should not be recorded into future Joy memory.
- Do not invent other `[/...]` commands. Only `/duration`, `/overlap`, and `/nomem` are supported.

Content rules:
- Use stable IDs such as ID_A, ID_B, ID_OBJECT, or ID_PLACE for recurring people, objects, and settings.
- Reintroduce the key visual identity, wardrobe, voice, location, and recurring object details in every shot where they matter.
- Include natural movement, camera framing, facial/lip visibility when there is speech, sound effects, ambience, and any spoken lines.
- Make later shots clearly reuse earlier characters, objects, or locations so JoyAI-Echo memory has something meaningful to carry forward.
- Avoid JSON, shot numbering, Markdown, and unrelated clips stitched together without continuity.
"""

_SHOT_OPTIONS_RE = re.compile(r"\[\s*/\s*([^\]]+?)\s*\]", re.IGNORECASE)
_SHOT_OPTIONS_HELP = "Supported JoyAI-Echo shot options are /duration, /overlap, and /nomem."


def _normalize_frame_count(frame_count: int, minimum: int, step: int) -> int:
    frame_count = max(int(minimum), int(frame_count))
    step = max(1, int(step))
    if step <= 1:
        return frame_count
    return int(math.ceil(max(0, frame_count - 1) / step) * step + 1)


def _parse_duration_frames(raw_value: str, fps: float, default_frames: int, minimum: int, step: int) -> int:
    raw_value = (raw_value or "").strip().lower()
    try:
        if raw_value.endswith("s"):
            seconds = float(raw_value[:-1].strip())
            frame_count = int(round(seconds * float(fps)))
        else:
            frame_count = int(raw_value)
    except Exception as exc:
        raise ValueError(f"Invalid JoyAI-Echo /duration value '{raw_value}'. Use an integer frame count or seconds like 5s.") from exc
    return _normalize_frame_count(frame_count, minimum, step)


def _normalize_overlap_frames(frame_count: int, step: int) -> int:
    frame_count = int(frame_count)
    if frame_count < 0:
        raise ValueError("JoyAI-Echo /overlap must be 0 or a positive frame count.")
    if frame_count == 0:
        return 0
    step = max(1, int(step))
    return int(((frame_count - 1 + step // 2) // step) * step + 1)


def _parse_overlap_frames(raw_value: str | None, default_overlap: int, step: int) -> int:
    if raw_value is None:
        return _normalize_overlap_frames(default_overlap, step)
    raw_value = (raw_value or "").strip()
    if not raw_value:
        raise ValueError("JoyAI-Echo /overlap value cannot be empty. Use [/overlap] or [/overlap=9].")
    try:
        return _normalize_overlap_frames(int(raw_value), step)
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("JoyAI-Echo"):
            raise
        raise ValueError(f"Invalid JoyAI-Echo /overlap value '{raw_value}'. Use an integer frame count.") from exc


def _apply_shot_options(block: str, *, default_frames: int, fps: float, minimum: int, step: int, default_overlap: int) -> tuple[str, int, bool, int | None]:
    duration = default_frames
    record_memory = True
    overlap_frames = None

    def replace_options(match):
        nonlocal duration, record_memory, overlap_frames
        for raw_option in match.group(1).split(","):
            option = raw_option.strip()
            if option.startswith("/"):
                option = option[1:].strip()
            key, separator, value = option.partition("=")
            key = key.strip().lower()
            if key == "duration":
                if not separator or not value.strip():
                    raise ValueError("JoyAI-Echo /duration requires a value, e.g. [/duration=5s].")
                duration = _parse_duration_frames(value, fps, default_frames, minimum, step)
            elif key == "nomem":
                if separator:
                    raise ValueError("JoyAI-Echo /nomem does not take a value.")
                record_memory = False
            elif key == "overlap":
                overlap_frames = _parse_overlap_frames(value if separator else None, default_overlap, step)
            else:
                raise ValueError(f"Unknown JoyAI-Echo shot option '/{key}'. {_SHOT_OPTIONS_HELP}")
        return ""

    return _SHOT_OPTIONS_RE.sub(replace_options, block), duration, record_memory, overlap_frames


def split_blank_line_shots(prompt: str, *, default_frames: int, fps: float, minimum: int, step: int, default_overlap: int = 0) -> list[tuple[str, int, bool, int | None]]:
    shots = []
    blocks = re.split(r"\n\s*\n+", (prompt or "").strip())
    for block in blocks:
        if not block.strip():
            continue
        block, duration, record_memory, overlap_frames = _apply_shot_options(block, default_frames=default_frames, fps=fps, minimum=minimum, step=step, default_overlap=default_overlap)
        shot = re.sub(r"\s*\n\s*", " ", block).strip()
        if shot:
            shots.append((shot, duration, record_memory, overlap_frames))
    return shots


def validate_joyai_prompt_options(prompt: str, *, default_frames: int, fps: float, minimum: int, step: int, default_overlap: int = 0) -> str | None:
    try:
        split_blank_line_shots(prompt, default_frames=default_frames, fps=fps, minimum=minimum, step=step, default_overlap=default_overlap)
    except Exception as exc:
        return str(exc)
    return None


def merge_shot_results(results: list[dict]) -> dict | None:
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    merged = dict(results[-1])
    merged["x"] = torch.cat([one["x"] for one in results], dim=1)
    audio_arrays = [one.get("audio") for one in results if one.get("audio") is not None]
    merged["audio"] = np.concatenate(audio_arrays, axis=0) if len(audio_arrays) == len(results) else None
    return merged


def _default_overlap_frames(model_def: dict, step: int) -> int:
    defaults = model_def.get("sliding_window_defaults", {}) if isinstance(model_def, dict) else {}
    return _normalize_overlap_frames(int(defaults.get("overlap_default", 0) or 0), step)


def _video_overlap_input(video: torch.Tensor | None, overlap_frames: int) -> torch.Tensor | None:
    if video is None or int(overlap_frames) <= 0:
        return None
    frames = min(int(overlap_frames), int(video.shape[1]))
    if frames <= 0:
        return None
    tail = video[:, -frames:].detach().cpu().contiguous()
    if tail.dtype == torch.uint8:
        return tail.float().div_(127.5).sub_(1.0)
    return tail


def _audio_overlap_input(audio, sample_rate: int | None, overlap_frames: int, fps: float):
    if audio is None or not sample_rate or int(overlap_frames) <= 0:
        return None, 0
    samples = int(round(float(overlap_frames) * float(sample_rate) / float(fps)))
    if samples <= 0:
        return None, 0
    return np.ascontiguousarray(audio[-min(samples, int(audio.shape[0])):]), int(sample_rate)


def _trim_audio_start(audio, trim_frames: int, fps: float, sample_rate: int | None):
    if audio is None or not sample_rate or int(trim_frames) <= 0:
        return audio
    samples = int(round(float(trim_frames) * float(sample_rate) / float(fps)))
    return audio[min(samples, int(audio.shape[0])):]


def _trim_memory_latents(model, memory_latents: dict | None, trim_frames: int, total_frames: int) -> dict | None:
    if not memory_latents or int(trim_frames) <= 0 or int(total_frames) <= 0:
        return memory_latents
    phase_latents = memory_latents if "phase1" in memory_latents or "phase2" in memory_latents else {"phase1": memory_latents}
    trimmed = {}
    video_trim = _pixel_to_latent_index(int(trim_frames), _latent_stride(model))
    for phase, latents in phase_latents.items():
        if not isinstance(latents, dict):
            continue
        phase_trimmed = dict(latents)
        video_latent = phase_trimmed.get("video")
        if video_latent is not None and int(video_latent.shape[2]) > 1:
            keep_from = min(video_trim, int(video_latent.shape[2]) - 1)
            phase_trimmed["video"] = video_latent[:, :, keep_from:].contiguous()
        audio_latent = phase_trimmed.get("audio")
        if audio_latent is not None and int(audio_latent.shape[2]) > 1:
            audio_trim = int(round(float(trim_frames) / float(total_frames) * float(audio_latent.shape[2])))
            keep_from = min(max(0, audio_trim), int(audio_latent.shape[2]) - 1)
            phase_trimmed["audio"] = audio_latent[:, :, keep_from:].contiguous()
        trimmed[phase] = phase_trimmed
    return trimmed if "phase1" in memory_latents or "phase2" in memory_latents else trimmed.get("phase1")


class JoyAIEchoMemoryBank:
    def __init__(self, max_size: int = 7, num_fix_frames: int = 3, audio_window_size: int = 96) -> None:
        self.max_size = int(max_size)
        self.num_fix_frames = max(0, int(num_fix_frames))
        self.audio_window_size = max(1, int(audio_window_size))
        self.entries: list[dict] = []

    def __len__(self) -> int:
        return len(self.entries)

    def _trim(self) -> None:
        if self.max_size <= 0 or len(self.entries) <= self.max_size:
            return
        fixed = self.entries[: self.num_fix_frames]
        tail = self.entries[self.num_fix_frames :]
        keep_tail = max(0, self.max_size - len(fixed))
        self.entries = fixed + tail[-keep_tail:]

    def _build_entry(self, model, phase: str, video_latent: torch.Tensor | None, audio_latent: torch.Tensor | None, audio_waveform=None, audio_sample_rate: int | None = None) -> dict | None:
        if video_latent is None:
            return None
        video_latent = video_latent.detach().cpu().contiguous()
        video_frames = int(video_latent.shape[2])
        if audio_latent is None:
            video_idx = max(0, video_frames // 2)
            return {"video": {phase: video_latent[:, :, video_idx : video_idx + 1]}, "audio": {}, "audio_lengths": {}}
        audio_latent = audio_latent.detach().cpu().contiguous()
        total_audio_frames = int(audio_latent.shape[2])
        waveform = None if audio_waveform is None or audio_sample_rate is None else _normalize_waveform(audio_waveform, channels_first=False)
        window_start, window_len = _select_audio_window_start(model, audio_latent, waveform, audio_sample_rate, self.audio_window_size)
        window_end = window_start + window_len
        video_idx = _video_idx_from_audio_window(video_frames, total_audio_frames, window_start, window_len)
        return {
            "video": {phase: video_latent[:, :, video_idx : video_idx + 1]},
            "audio": {phase: audio_latent[:, :, window_start:window_end]},
            "audio_lengths": {phase: int(window_len)},
        }

    def add_generation(self, model, memory_latents: dict | None, audio_waveform=None, audio_sample_rate: int | None = None) -> None:
        if not memory_latents:
            return
        phase_latents = memory_latents if "phase1" in memory_latents or "phase2" in memory_latents else {"phase1": memory_latents}
        entry = {"video": {}, "audio": {}, "audio_lengths": {}}
        for phase, latents in phase_latents.items():
            if not isinstance(latents, dict):
                continue
            phase_entry = self._build_entry(model, phase, latents.get("video"), latents.get("audio"), audio_waveform, audio_sample_rate)
            if phase_entry is None:
                continue
            entry["video"].update(phase_entry["video"])
            entry["audio"].update(phase_entry["audio"])
            entry["audio_lengths"].update(phase_entry["audio_lengths"])
        if entry["video"]:
            self.entries.append(entry)
            self._trim()

    def add_artificial_memory(self, memory: dict) -> None:
        phase_video_latents = memory.get("video", {}) if isinstance(memory, dict) else {}
        phase_audio_slots = memory.get("audio", {}) if isinstance(memory, dict) else {}
        if not phase_video_latents:
            return
        slots = max(int(latent.shape[2]) for latent in phase_video_latents.values() if latent is not None)
        for slot_idx in range(slots):
            entry = {"video": {}, "audio": {}, "audio_lengths": {}}
            for phase, latent in phase_video_latents.items():
                if latent is not None and slot_idx < int(latent.shape[2]):
                    entry["video"][phase] = latent.detach().cpu().contiguous()[:, :, slot_idx : slot_idx + 1]
            for phase, audio_slots in phase_audio_slots.items():
                if audio_slots is not None and slot_idx < len(audio_slots):
                    entry["audio"][phase] = audio_slots[slot_idx].detach().cpu().contiguous()
                    entry["audio_lengths"][phase] = int(audio_slots[slot_idx].shape[2])
            if entry["video"]:
                self.entries.append(entry)
        self._trim()

    def video_latent(self, phase: str = "phase1") -> torch.Tensor | None:
        latents = [entry["video"][phase] for entry in self.entries if phase in entry["video"]]
        if not latents:
            return None
        return torch.cat(latents, dim=2).contiguous()

    def audio_latent(self, phase: str = "phase1") -> torch.Tensor | None:
        latents = [entry["audio"][phase] for entry in self.entries if phase in entry["audio"]]
        if not latents:
            return None
        return torch.cat(latents, dim=2).contiguous()

    def audio_segment_lengths(self, phase: str = "phase1"):
        lengths = [entry["audio_lengths"][phase] for entry in self.entries if phase in entry["audio_lengths"]]
        if not lengths:
            return None
        return (tuple(lengths),)

    def paired_audio_memory(self, phase: str = "phase1") -> bool:
        video_slots = sum(1 for entry in self.entries if phase in entry["video"])
        audio_slots = sum(1 for entry in self.entries if phase in entry["audio"])
        return video_slots > 0 and video_slots == audio_slots


def _latent_stride(model) -> int:
    scale_factors = getattr(getattr(model.pipeline, "pipeline_components", None), "video_scale_factors", None)
    if scale_factors is not None:
        time_factor = getattr(scale_factors, "time", None)
        return int(time_factor if time_factor is not None else scale_factors[0])
    return 8


def _pixel_to_latent_index(frame_idx: int, stride: int) -> int:
    if frame_idx <= 0:
        return 0
    return (int(frame_idx) - 1) // int(stride) + 1


def _parse_control_memory_positions(raw_value: str, fps: float, *, max_seconds: float | None = None) -> list[int]:
    positions = []
    for raw_pos in re.split(r"\s*,\s*", raw_value or ""):
        if not raw_pos:
            continue
        value = raw_pos.strip().lower()
        seconds = float(value[:-1]) if value.endswith("s") else (int(value) - 1) / float(fps)
        if max_seconds is not None and seconds > float(max_seconds):
            raise ValueError(f"JoyAI-Echo Control Video Memory position '{value}' is beyond the first {int(max_seconds)} seconds.")
        frame_idx = int(round(seconds * float(fps))) if value.endswith("s") else int(value) - 1
        positions.append(max(0, frame_idx))
    return positions


def validate_control_memory_positions(raw_value: str, fps: float, *, max_seconds: float = JOYAI_CONTROL_MEMORY_MAX_SECONDS) -> str | None:
    try:
        _parse_control_memory_positions(raw_value, fps, max_seconds=max_seconds)
    except Exception as exc:
        return str(exc)
    return None


def _normalize_waveform(waveform, *, channels_first: bool, max_seconds: float | None = None, sample_rate: int | None = None) -> torch.Tensor:
    waveform = torch.as_tensor(waveform).detach().cpu().float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and not channels_first:
        waveform = waveform.T
    elif waveform.ndim == 3:
        waveform = waveform[0]
    if max_seconds is not None and sample_rate:
        waveform = waveform[:, : int(round(float(max_seconds) * float(sample_rate)))]
    return waveform.contiguous()


def _align_waveform_channels(waveform: torch.Tensor, target_channels: int) -> torch.Tensor:
    target_channels = max(1, int(target_channels))
    if waveform.shape[0] == target_channels:
        return waveform.contiguous()
    if waveform.shape[0] == 1:
        return waveform.repeat(target_channels, 1).contiguous()
    if waveform.shape[0] > target_channels:
        return waveform[:target_channels].contiguous()
    return torch.cat([waveform, waveform[-1:].repeat(target_channels - waveform.shape[0], 1)], dim=0).contiguous()


def _audio_processor(model):
    from .ltx_core.model.audio_vae import AudioProcessor

    encoder = model.audio_encoder
    return AudioProcessor(sample_rate=encoder.sample_rate, mel_bins=encoder.mel_bins, mel_hop_length=encoder.mel_hop_length, n_fft=encoder.n_fft)


def _audio_latent_downsample(model) -> int:
    return int(getattr(getattr(model.audio_encoder, "patchifier", None), "audio_latent_downsample_factor", 4))


def _encode_audio_memory(model, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
    target_channels = int(getattr(model.audio_encoder, "in_channels", waveform.shape[0]) or waveform.shape[0])
    waveform = _align_waveform_channels(waveform, target_channels).to(device="cpu", dtype=torch.float32)
    processor = _audio_processor(model).to(waveform.device)
    if processor.waveform_too_short_for_mel(waveform.unsqueeze(0), int(sample_rate)):
        return None
    mel = processor.waveform_to_mel(waveform.unsqueeze(0), int(sample_rate))
    audio_params = next(model.audio_encoder.parameters(), None)
    audio_device = audio_params.device if audio_params is not None else model.device
    audio_dtype = audio_params.dtype if audio_params is not None else model.dtype
    with torch.inference_mode():
        return model.audio_encoder(mel.to(device=audio_device, dtype=audio_dtype)).detach().cpu().contiguous()


def _max_response_mel_bounds(mel: torch.Tensor, window_size: int) -> tuple[int, int]:
    time_steps = int(mel.shape[2])
    window_size = max(1, int(window_size))
    max_start = time_steps - window_size if time_steps >= window_size else time_steps - 1
    starts = list(range(0, max_start + 1, max(1, window_size // 4)))
    if starts[-1] != max_start:
        starts.append(max_start)
    offsets = torch.arange(window_size, device=mel.device)
    scores = []
    for start in starts:
        scores.append(mel.index_select(2, (start + offsets).clamp(0, time_steps - 1).long()).float().exp().sum())
    start = int(starts[int(torch.stack(scores).argmax().item())])
    return start, min(start + window_size - 1, time_steps - 1)


def _audio_energy_mask(model, waveform: torch.Tensor, sample_rate: int, total_frames: int) -> torch.Tensor:
    total_frames = max(1, int(total_frames))
    mono = waveform.mean(dim=0).float()
    samples_per_latent = max(1, int(round(float(sample_rate) * float(model.audio_encoder.mel_hop_length) * float(_audio_latent_downsample(model)) / float(model.audio_encoder.sample_rate))))
    padded = torch.nn.functional.pad(mono, (0, max(0, total_frames * samples_per_latent - mono.shape[-1])))
    rms = padded[: total_frames * samples_per_latent].reshape(total_frames, samples_per_latent).square().mean(dim=1).sqrt()
    db = 20.0 * torch.log10(rms + 1e-8)
    floor = torch.quantile(db, 0.2)
    peak = db.max()
    if float(peak - floor) < JOYAI_AUDIO_SILENCE_DYNAMIC_RANGE_DB:
        return torch.zeros_like(db, dtype=torch.bool)
    threshold = floor + (peak - floor) * JOYAI_AUDIO_SILENCE_THRESHOLD_FRACTION
    return db >= threshold


def _nearest_nonsilent_window_start(start: int, window_len: int, non_silent: torch.Tensor | None) -> int:
    if non_silent is None or int(non_silent.numel()) == 0 or not bool(non_silent.any()):
        return max(0, int(start))
    max_start = max(0, int(non_silent.numel()) - int(window_len))
    start = max(0, min(int(start), max_start))
    for radius in range(max_start + 1):
        for candidate in (start + radius, start - radius):
            if 0 <= candidate <= max_start and bool(non_silent[candidate : candidate + int(window_len)].any()):
                return int(candidate)
    return start


def _select_audio_window_start(model, audio_latent: torch.Tensor, waveform: torch.Tensor | None, sample_rate: int | None, window_size: int, *, center_latent: int | None = None) -> tuple[int, int]:
    total_frames = int(audio_latent.shape[2])
    window_len = min(total_frames, max(1, int(window_size)))
    start = max(0, min((total_frames - window_len) // 2 if center_latent is None else int(center_latent) - window_len // 2, max(total_frames - window_len, 0)))
    if waveform is None or sample_rate is None:
        return start, window_len
    if center_latent is None:
        processor = _audio_processor(model).to(waveform.device)
        mel = processor.waveform_to_mel(waveform.unsqueeze(0), int(sample_rate))
        mel_window = max(1, window_len * _audio_latent_downsample(model) - (_audio_latent_downsample(model) - 1))
        mel_start, mel_end = _max_response_mel_bounds(mel, mel_window)
        center_time = ((mel_start + mel_end + 1) * 0.5 * float(model.audio_encoder.mel_hop_length)) / float(model.audio_encoder.sample_rate)
        duration = max(float(waveform.shape[-1]) / float(sample_rate), 1e-6)
        center_latent = int(round(max(0.0, min(center_time, duration)) / duration * float(max(total_frames - 1, 0))))
        start = max(0, min(center_latent - window_len // 2, max(total_frames - window_len, 0)))
    return _nearest_nonsilent_window_start(start, window_len, _audio_energy_mask(model, waveform, int(sample_rate), total_frames)), window_len


def _video_idx_from_audio_window(video_frames: int, audio_frames: int, window_start: int, window_len: int, *, min_idx: int = 0) -> int:
    center_ratio = 0.5 if audio_frames <= 1 else (window_start + max(window_len - 1, 0) * 0.5) / float(audio_frames - 1)
    return max(int(min_idx), min(int(round(center_ratio * float(max(video_frames - 1, 0)))), max(video_frames - 1, 0)))


def _normal_tiling_config(tile_size: int | tuple | list | None, num_frames: int | None) -> TilingConfig | None:
    from .ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig

    if isinstance(tile_size, (tuple, list)):
        tile_size = tile_size[-1] if tile_size else None
    if tile_size is None:
        return None
    tile_size = int(tile_size)
    if tile_size <= 0:
        return None
    tile_size = max(64, int(math.ceil(tile_size / 32) * 32))
    spatial_config = SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=int(math.floor((tile_size // 4) / 32) * 32))
    temporal_config = None
    if num_frames is not None and num_frames > 241:
        temporal_config = TemporalTilingConfig(tile_size_in_frames=232, tile_overlap_in_frames=88)
    return TilingConfig(spatial_config=spatial_config, temporal_config=temporal_config)


def _load_control_audio(video_path: str) -> tuple[torch.Tensor, int]:
    from shared.utils.audio_video import extract_audio_track_to_wav
    import soundfile as sf

    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="joyai_control_audio_")
    os.close(fd)
    try:
        extract_audio_track_to_wav(video_path, wav_path)
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T).contiguous(), int(sample_rate)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def _target_audio_center_for_frame(frame_idx: int, fps: float, waveform: torch.Tensor, sample_rate: int, audio_frames: int) -> int:
    duration = max(float(waveform.shape[-1]) / float(sample_rate), 1e-6)
    seconds = max(0.0, min(float(frame_idx) / float(fps), duration))
    return int(round(seconds / duration * float(max(audio_frames - 1, 0))))


def _latent_center_frame(latent_idx: int, stride: int) -> int:
    if latent_idx <= 0:
        return 0
    return int((latent_idx - 1) * stride + 1 + (stride // 2))


def _encode_control_video_slots(model, video_path: str, latent_indices: list[int], *, fps: float, height: int, width: int, two_phase: bool, VAE_tile_size=None) -> dict[str, torch.Tensor]:
    from .ltx_core.model.video_vae import encode_video as vae_encode_video
    from .ltx_pipelines.utils.helpers import cleanup_memory
    from .ltx_pipelines.utils.media_io import load_video_conditioning
    from shared.utils.video_decode import decode_video_frames_ffmpeg

    stride = _latent_stride(model)
    if not latent_indices:
        return {}
    phase_sizes = {"phase1": (height // 2, width // 2)} if two_phase else {"phase1": (height, width)}
    if two_phase:
        phase_sizes["phase2"] = (height, width)
    tiling_config = _normal_tiling_config(VAE_tile_size, None)
    phase_slots = {phase: [] for phase in phase_sizes}
    video_encoder = model.video_encoder
    for latent_idx in latent_indices:
        context_start_latent = max(0, int(latent_idx) - 2)
        start_frame = 0 if context_start_latent <= 0 else (context_start_latent - 1) * stride + 1
        end_frame = (int(latent_idx) + 1) * stride
        frames = decode_video_frames_ffmpeg(video_path, start_frame, max(1, end_frame - start_frame + 1), target_fps=fps, bridge="torch")
        if int(frames.shape[0]) == 0:
            continue
        local_idx = max(0, min(_pixel_to_latent_index(_latent_center_frame(latent_idx, stride) - start_frame, stride), max(0, int(math.ceil((int(frames.shape[0]) - 1) / stride)))))
        for phase, (phase_height, phase_width) in phase_sizes.items():
            video = load_video_conditioning(frames, height=int(phase_height), width=int(phase_width), frame_cap=None, dtype=model.dtype, device=model.device)
            encoded = vae_encode_video(video, video_encoder, tiling_config)
            if int(encoded.shape[2]) > 0:
                phase_slots[phase].append(encoded[:, :, min(local_idx, int(encoded.shape[2]) - 1) : min(local_idx, int(encoded.shape[2]) - 1) + 1].detach().cpu().contiguous())
            del video, encoded
            cleanup_memory()
        del frames
    return {phase: torch.cat(slots, dim=2).contiguous() for phase, slots in phase_slots.items() if slots}


def build_control_video_memory(model, control_video_path: str, positions_text: str, *, fps: float, height: int, width: int, two_phase: bool, VAE_tile_size=None) -> dict:
    positions = _parse_control_memory_positions(positions_text, fps, max_seconds=JOYAI_CONTROL_MEMORY_MAX_SECONDS)
    waveform, sample_rate = _load_control_audio(control_video_path)
    waveform = _normalize_waveform(waveform, channels_first=True, max_seconds=JOYAI_CONTROL_MEMORY_MAX_SECONDS, sample_rate=sample_rate)
    audio_latent = _encode_audio_memory(model, waveform, sample_rate)
    if audio_latent is None:
        raise RuntimeError("JoyAI-Echo Control Video Memory audio is too short to encode.")
    audio_frames = int(audio_latent.shape[2])
    window_starts = []
    if positions:
        for frame_idx in positions:
            center_latent = _target_audio_center_for_frame(frame_idx, fps, waveform, sample_rate, audio_frames)
            window_start, window_len = _select_audio_window_start(model, audio_latent, waveform, sample_rate, int(model.model_def.get("joyai_audio_memory_window_size", 96)), center_latent=center_latent)
            if window_start not in window_starts:
                window_starts.append(window_start)
    else:
        window_start, window_len = _select_audio_window_start(model, audio_latent, waveform, sample_rate, int(model.model_def.get("joyai_audio_memory_window_size", 96)))
        window_starts.append(window_start)
    window_len = min(audio_frames, int(model.model_def.get("joyai_audio_memory_window_size", 96)))
    stride = _latent_stride(model)
    latent_indices = []
    audio_slots = []
    for window_start in window_starts:
        video_idx = _video_idx_from_audio_window(max(1, int(math.ceil(float(waveform.shape[-1]) / float(sample_rate) * float(fps) / float(stride))) + 1), audio_frames, window_start, window_len, min_idx=1)
        if video_idx not in latent_indices:
            latent_indices.append(video_idx)
            audio_slots.append(audio_latent[:, :, window_start : window_start + window_len].detach().cpu().contiguous())
    video = _encode_control_video_slots(model, control_video_path, latent_indices, fps=fps, height=height, width=width, two_phase=two_phase, VAE_tile_size=VAE_tile_size)
    audio = {phase: list(audio_slots) for phase in video}
    print(f"[WAN2GP][JoyAI-Echo] control_memory_slots={len(latent_indices)} audio_paired={bool(audio_slots)} max_seconds={int(JOYAI_CONTROL_MEMORY_MAX_SECONDS)}", flush=True)
    return {"video": video, "audio": audio}


def generate_joyai_echo_shots(model, single_shot_generate, **call_args):
    fps = float(call_args.get("fps", model.model_def.get("fps", 25)) or 25)
    frames_minimum = int(model.model_def.get("frames_minimum", 17))
    frames_step = int(model.model_def.get("frames_steps", 8))
    source_overlap = int(call_args.get("prefix_frames_count", 0) or 0) if torch.is_tensor(call_args.get("input_video")) else 0
    default_frames = _normalize_frame_count(int(call_args.get("frame_num", 0) or frames_minimum) - max(0, source_overlap - 1), frames_minimum, frames_step)
    default_overlap = _default_overlap_frames(model.model_def, frames_step)
    shots = split_blank_line_shots(
        call_args.get("input_prompt", ""),
        default_frames=default_frames,
        fps=fps,
        minimum=frames_minimum,
        step=frames_step,
        default_overlap=default_overlap,
    )
    if not shots:
        return None
    total = len(shots)
    base_seed = int(call_args.get("seed", 0) or 0)
    set_progress_status = call_args.get("set_progress_status")
    custom_settings = call_args.get("custom_settings") if isinstance(call_args.get("custom_settings"), dict) else {}
    guide_phases = int(call_args.get("guide_phases", call_args.get("guidance_phases", 1)) or 1)
    two_phase = guide_phases > 1
    memory_bank = JoyAIEchoMemoryBank(
        max_size=int(model.model_def.get("joyai_memory_max_size", 7)),
        num_fix_frames=int(model.model_def.get("joyai_memory_num_fix_frames", 3)),
        audio_window_size=int(model.model_def.get("joyai_audio_memory_window_size", 96)),
    )
    control_memory_enabled = "1" in (call_args.get("video_prompt_type", "") or "")
    positions_text = custom_settings.get(JOYAI_CONTROL_MEMORY_SETTING, "") if control_memory_enabled else ""
    if control_memory_enabled:
        control_video_path = call_args.get("video_guide")
        if not control_video_path:
            raise RuntimeError("JoyAI-Echo Control Video Memory requires the original Control Video path.")
        target_height = int(call_args.get("height"))
        target_width = int(call_args.get("width"))
        if target_height % 64 != 0:
            target_height = int(math.ceil(target_height / 64) * 64)
        if target_width % 64 != 0:
            target_width = int(math.ceil(target_width / 64) * 64)
        artificial_memory = build_control_video_memory(
            model,
            control_video_path,
            str(positions_text),
            fps=fps,
            height=target_height,
            width=target_width,
            two_phase=two_phase,
            VAE_tile_size=call_args.get("VAE_tile_size"),
        )
        memory_bank.add_artificial_memory(artificial_memory)
    results = []
    previous_video = previous_audio = None
    previous_audio_sample_rate = None
    for shot_idx, (shot, shot_frames, record_memory, overlap_frames) in enumerate(shots):
        if model._interrupt:
            return None
        prefix = f"Shot {shot_idx + 1}/{total}"
        if set_progress_status is not None:
            set_progress_status(prefix)

        def shot_progress(status, *, _prefix=prefix):
            if set_progress_status is not None:
                set_progress_status(f"{_prefix} - {status}")

        def shot_callback(*args, _prefix=prefix, _callback=call_args.get("callback"), **kwargs):
            if _callback is not None:
                kwargs["status_prefix"] = _prefix
                return _callback(*args, **kwargs)

        audio_memory_enabled = bool(model.model_def.get("joyai_audio_memory", False))
        shot_args = dict(call_args)
        overlap_prefix = None
        overlap_audio = None
        overlap_audio_sample_rate = 0
        if overlap_frames is not None and overlap_frames > 0:
            if shot_idx == 0:
                overlap_prefix = _video_overlap_input(call_args.get("input_video"), overlap_frames)
            else:
                overlap_prefix = _video_overlap_input(previous_video, overlap_frames)
        actual_overlap = int(overlap_prefix.shape[1]) if overlap_prefix is not None else 0
        if actual_overlap > 0:
            if shot_idx == 0:
                overlap_audio, overlap_audio_sample_rate = _audio_overlap_input(call_args.get("input_waveform"), call_args.get("input_waveform_sample_rate"), actual_overlap, fps)
            else:
                overlap_audio, overlap_audio_sample_rate = _audio_overlap_input(previous_audio, previous_audio_sample_rate, actual_overlap, fps)
        if overlap_frames == 0:
            actual_overlap = 0
        frame_num = int(shot_frames + max(0, actual_overlap - 1))
        trim_frames = max(0, frame_num - int(shot_frames))
        print(f"[WAN2GP][JoyAI-Echo] shot {shot_idx + 1}/{total}: frames={shot_frames} overlap={actual_overlap} memory_slots={len(memory_bank)} record_memory={record_memory} seed={base_seed + shot_idx}", flush=True)
        shot_args.update({"input_prompt": shot, "frame_num": frame_num, "seed": base_seed + shot_idx, "set_progress_status": shot_progress, "callback": shot_callback})
        if overlap_frames is not None:
            shot_args.update({"input_video": overlap_prefix, "prefix_frames_count": actual_overlap, "input_waveform": overlap_audio, "input_waveform_sample_rate": overlap_audio_sample_rate})
        reference_context = {
            "video_latent": memory_bank.video_latent("phase1"),
            "audio_latent": memory_bank.audio_latent("phase1"),
            "audio_segment_lengths": memory_bank.audio_segment_lengths("phase1"),
            "paired_audio": audio_memory_enabled and memory_bank.paired_audio_memory("phase1"),
            "video_latent_stage2": memory_bank.video_latent("phase2"),
            "audio_latent_stage2": memory_bank.audio_latent("phase2"),
            "audio_segment_lengths_stage2": memory_bank.audio_segment_lengths("phase2"),
            "paired_audio_stage2": audio_memory_enabled and memory_bank.paired_audio_memory("phase2"),
            "downscale_factor": int(model.model_def.get("joyai_memory_downscale_factor", 1)),
            "return_latents": record_memory,
        }
        shot_args.pop("video_guide", None)
        if control_memory_enabled or "V" in (call_args.get("video_prompt_type", "") or ""):
            shot_args.update({"input_frames": None, "input_frames2": None, "input_masks": None, "input_masks2": None, "video_prompt_type": ""})
        if shot_idx > 0:
            shot_args.update({"image_start": None, "image_prompt_type": ""})
            if overlap_frames is None:
                shot_args.update({"input_video": None, "prefix_frames_count": 0, "input_waveform": None, "input_waveform_sample_rate": 0})
        with model.pipeline.joyai_echo_context(reference_context):
            result = single_shot_generate(**shot_args)
        if result is None:
            return None
        memory_latents = result.pop("_memory_latents", None)
        if trim_frames > 0:
            result["x"] = result["x"][:, trim_frames:].contiguous()
            result["audio"] = _trim_audio_start(result.get("audio"), trim_frames, fps, result.get("audio_sampling_rate"))
            memory_latents = _trim_memory_latents(model, memory_latents, trim_frames, frame_num)
        if record_memory:
            memory_bank.add_generation(model, memory_latents, audio_waveform=result.get("audio"), audio_sample_rate=result.get("audio_sampling_rate"))
        previous_video = result.get("x")
        previous_audio = result.get("audio")
        previous_audio_sample_rate = result.get("audio_sampling_rate")
        results.append(result)
    return merge_shot_results(results)
