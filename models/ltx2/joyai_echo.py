from __future__ import annotations

import math
import os
import re
import tempfile

import torch


JOYAI_CONTROL_MEMORY_SETTING = "joyai_control_memory_positions"
JOYAI_CONTROL_MEMORY_MAX_SECONDS = 60.0
JOYAI_AUDIO_SILENCE_DYNAMIC_RANGE_DB = 6.0
JOYAI_AUDIO_SILENCE_THRESHOLD_FRACTION = 0.35
JOYAI_DEBUG_MEMORY = True
JOYAI_ECHO_PROMPT_INFOS = """**JoyAI-Echo Prompt Format**

JoyAI-Echo keeps a small audio-video memory between sliding windows. Each memory slot is a short visual moment paired with nearby voice or sound, so prompts work best when recurring people, voices, objects, and settings are described consistently across windows. Use stable IDs such as `ID_A`, `ID_B`, `ID_OBJECT`, and `ID_PLACE`, then repeat the details that matter each time they reappear.

Write one story window per line or paragraph, depending on the selected Text Prompt processing mode. Each window should be self-contained: identity, wardrobe, voice, setting, action, camera framing, dialogue, visible lips when someone speaks, ambience, and sound effects.

**Window Commands**

WanGP removes `[/...]` commands before sending the text to the model. Brackets that do not start with `/` are preserved for other prompt syntax.

Generic window commands:

- `[/duration=121]`: this window contributes about 121 output frames.
- `[/duration=5s]`: this window contributes about 5 seconds at the generation FPS.
- `[/duration=20%]`: this window contributes 20% of the requested total frame count.
- `[/overlap]`: use the model's default overlap for this window.
- `[/overlap=9]`: use overlap frames from the previous window to make a smoother transition.
- `[/overlap=0]`: use no overlap frames, when the model supports text-to-video windows.
- `[/new_shot]`: start without overlap frames. Use it for a hard cut, a different scene, a new character introduction, or when Continue Video should be preserved in the output but not visually continued by the model.

JoyAI-Echo memory commands:

- `[/no_mem]`: generate this window without recording it into future Joy memory.
- `[/store_mem]`: store one automatic memory slot from this window.
- `[/store_mem=2]`: store or replace numeric slot 2.
- `[/store_mem=man1,man2]`: store multiple named slots from this window, sampled at different moments. Multiple slots for the same character can strengthen later reuse.
- `[/drop_mem]`: drop the latest memory slot before this window.
- `[/drop_mem=2]`: drop numeric slot 2.
- `[/drop_mem=2-4]`: drop a numeric range.
- `[/drop_mem=man1,woman1]`: drop named slots. This is useful before a new scene, cast change, or when you want to free memory for new identities.

You can combine commands in one bracket, for example `[/duration=8s,/new_shot,/store_mem=duck]`.

**Tips**

- Reuse the same ID and repeat important visible and audio traits. Memory helps, but the prompt still guides identity.
- Use named memory slots when the story has several people or objects. Names make it easier to replace or drop the right memory later.
- Use `[/new_shot]` to introduce a new person or location without visual overlap from the previous window. Stored Joy memory can still influence later windows unless you drop it.
- Use `[/no_mem]` for transitional action, abstract inserts, or shots that should not become identity reference material.
- Use a Control Video with an audio track to predefine memory before generation. Select JoyAI-Echo Control Video Memory, then leave positions empty for automatic non-silent selection or enter positions like `2s, 8s, 15s`.

**Three-Window Example**

```text
[/duration=8s,/store_mem=magician1,magician2] ID_A is a retired stage magician with silver hair, a burgundy velvet jacket, expressive eyebrows, and a dry theatrical voice. He stands inside an abandoned seaside theater at night, opens a lacquered box, reveals a glowing blue playing card, and says, "I retired from miracles because the miracles started freelancing." Keep his face and lip movement clear in a stable medium close-up. Add dusty gold footlights, torn red curtains, distant waves, old wood creaks, and soft card handling.

[/duration=8s,/new_shot,/store_mem=duck1,duck2] ID_B is a small yellow duck inspector wearing a tiny teal raincoat, round brass spectacles, polished rubber boots, and a serious waterproof satchel. ID_B waddles through the theater aisle, sees the glowing card, and quacks in a crisp subtitle-like comic rhythm, "That door has not filed the proper puddle paperwork." Keep ID_B's duck design, bill movement, spectacles, and raincoat clear in a medium shot with squeaky footsteps, distant surf, and a tiny official whistle.

[/duration=8s,/new_shot] ID_A and ID_B are together on the same seaside theater stage. ID_A is still the silver-haired retired magician in the burgundy velvet jacket with the dry theatrical voice and glowing blue card. ID_B is still the small yellow duck inspector in the teal raincoat, spectacles, boots, and serious satchel. The glowing door opens between them onto a miniature harbor full of lantern boats. ID_A says, "I told you the door was shy." ID_B replies, "Shy doors still need permits." Keep both identities, voices, scale difference, readable lip and bill movement, card glow, theater details, tiny waves, and playful overlapping laughter.
```
"""

JOYAI_ECHO_INFOS = """**JoyAI-Echo**

JoyAI-Echo is an LTX-2.3 based audio-video model for connected multi-window stories. WanGP generates one sliding window after another and keeps compact memory from previous windows so later windows can reuse people, voices, objects, and places.

**How Generation Works**

- A sliding window acts like one story beat or shot.
- Write one prompt per line or one paragraph per window, depending on the Text Prompt processing mode.
- A Start Image or Continue Video applies to the first window.
- Later windows rely on their own prompt text, optional overlap frames, and Joy memory.
- Use **1 phase** for best JoyAI-Echo performance. Two phases can refine the image, but one phase is usually faster and more reliable for connected audio-video memory.

**Memory**

- Joy memory has up to **7 slots**.
- Each slot stores a very short visual moment: about **9 frames**, roughly **360 ms at 25 fps**.
- Each visual moment is paired with about **3.8 seconds** of nearby voice or sound from the same window, centered on the selected memory moment when a frame position is provided: roughly **1.9 seconds before** and **1.9 seconds after**. If that centered audio is silent, Joy shifts to the nearest non-silent audio window.
- When slots are full, automatic storage reuses free slots first and then replaces the oldest slot.
- Memory slots can be named from the prompt, which makes it easier to replace or drop specific memories later.

Use several slots for an important character when identity matters. For example, store two or three moments of the same person from different angles or expressions, then reuse that character in later windows. You can also drop old slots before a new scene so memory is not wasted on characters or objects that no longer matter.

**Control Video Memory**

Control Video Memory can predefine memory before the first generated window. Provide a Control Video with an audio track, choose **JoyAI-Echo Control Video Memory**, and either leave positions empty for automatic non-silent selection or enter frame/second positions.

WanGP uses only the needed moments instead of treating the full control video as the generated source.

Example: to use two reference moments from the first 20 seconds of a Control Video, set **Control Video Memory Positions** to `4s, 14s`. Joy will sample a short visual memory at each target time and pair it with nearby centered audio from the same Control Video.

**Prompt Commands**

Joy supports optional `[/...]` prompt commands for per-window duration, hard cuts, overlap, and memory management. See the prompt help for the full syntax.
"""

JOYAI_ECHO_PROMPT_ENHANCER = """You are writing prompts for JoyAI-Echo, an LTX-2.3 based audio-video model for connected multi-window stories.

JoyAI-Echo works best when each window is a clear cinematic beat and later windows deliberately reuse earlier people, voices, objects, or places. It keeps compact audio-video memories across windows, but the prompt should still repeat the identity details that must remain stable.

Return only the final prompt text. Do not return JSON, bullets, headings, commentary, or code fences.

Format:
- Write 2 to 6 cinematic story windows.
- Separate windows with exactly one empty line.
- Each window is a complete paragraph.
- Preserve any useful bracket syntax the user already provided.
- You may prefix a window with `[/new_shot]` when introducing a new character, cutting to a new location, or starting a later joint scene without visual overlap from the previous window.
- Do not add other `[/...]` commands unless the user already provided them.

Content rules:
- Use stable IDs such as ID_A, ID_B, ID_OBJECT, or ID_PLACE for recurring people, objects, and settings.
- Reintroduce the key visual identity, wardrobe, voice, location, and recurring object details in every window where they matter.
- Include natural movement, camera framing, facial/lip visibility when there is speech, sound effects, ambience, and spoken lines.
- Make later windows clearly reuse earlier characters, objects, or locations so JoyAI-Echo memory has something meaningful to carry forward.
- When introducing a new character, give them a distinct silhouette, wardrobe or object, voice quality, and a readable action.
- Avoid unrelated clips stitched together without continuity.

Example:
ID_A is Martin Bell, a careful apartment superintendent in his early fifties with salt-and-pepper hair, a gray mustache, a navy work shirt, keys on his belt, and a warm gravelly voice. He stands alone beside an old apartment elevator at night. The elevator dings and announces, "Third floor: lasagna is experiencing delays." ID_A looks at the panel, sighs, and says, "I fixed the water pressure. I did not authorize culinary announcements." Keep his face and lip movement clear in a stable medium close-up. Keep background sound minimal, with only the elevator bell and a soft room hum.

[/new_shot] ID_B is Nora Chen, a sharp tenant association president in her late thirties with shoulder-length black hair, tortoiseshell glasses, a mustard cardigan, a canvas tote, and a clear quick alto voice. She stands alone in the lobby near the elevator, holding a few complaint forms. The elevator opens behind her and calmly announces, "Laundry room: socks are now al dente." ID_B raises one eyebrow and says, "That elevator owes me four quarters and a written apology." Keep her face, glasses, clothing, and lip movement clear in a medium shot. Keep sound minimal, with only the elevator doors and a soft paper rustle.

[/new_shot] ID_A and ID_B are together beside the open elevator service panel in the same apartment building. ID_A is still the gray-mustached superintendent in the navy work shirt with keys and a warm gravelly voice. ID_B is still Nora Chen with black hair, tortoiseshell glasses, mustard cardigan, canvas tote, and quick alto voice. They discover an old baby monitor inside the panel, picking up a cooking show from another apartment. ID_A says, "So the elevator is not haunted, just subscribed." ID_B replies, "Excellent. Then I am billing apartment 4B for emotional ravioli." Keep both faces and lip movement readable, preserve their distinct voices and personalities, and keep background sound minimal.
"""

_MEMORY_ID_RE = re.compile(r"^[^\s,/=\[\]]+$")


def _memory_option_items(value) -> list[str]:
    if value is True or value is None:
        return []
    items = [item.strip() for item in str(value).split(",")]
    return [item for item in items if item]


def _is_number_id(value: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(value).strip()))


def _validate_memory_name(value: str, command: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"JoyAI-Echo /{command} memory name cannot be empty.")
    if _is_number_id(value):
        raise ValueError(f"JoyAI-Echo /{command} memory names must not be pure numbers.")
    if not _MEMORY_ID_RE.fullmatch(value):
        raise ValueError(f"JoyAI-Echo /{command} memory name '{value}' contains unsupported characters.")
    return value


def parse_store_mem_option(value) -> list[int | str | None]:
    if value is True or value is None:
        return [None]
    selectors: list[int | str | None] = []
    for item in _memory_option_items(value):
        if _is_number_id(item):
            selector = int(item)
            if selector < 1:
                raise ValueError("JoyAI-Echo /store_mem slots start from 1.")
            selectors.append(selector)
        else:
            selectors.append(_validate_memory_name(item, "store_mem"))
    if not selectors:
        raise ValueError("JoyAI-Echo /store_mem requires a memory slot index or name.")
    return selectors


def parse_drop_mem_option(value) -> list[int | str] | None:
    if value is True or value is None:
        return None
    selectors: list[int | str] = []
    for item in _memory_option_items(value):
        if re.fullmatch(r"\d+\s*-\s*\d+", item):
            start, end = [int(part.strip()) for part in item.split("-", 1)]
            if start < 1 or end < 1:
                raise ValueError("JoyAI-Echo /drop_mem slots start from 1.")
            if end < start:
                raise ValueError("JoyAI-Echo /drop_mem ranges must be written from low to high, e.g. [/drop_mem=2-4].")
            selectors.extend(range(start, end + 1))
        elif _is_number_id(item):
            index = int(item)
            if index < 1:
                raise ValueError("JoyAI-Echo /drop_mem slots start from 1.")
            selectors.append(index)
        else:
            selectors.append(_validate_memory_name(item, "drop_mem"))
    if not selectors:
        raise ValueError("JoyAI-Echo /drop_mem requires a memory slot, range, or name.")
    return selectors


def _memory_labels_text(labels: list[str]) -> str:
    return ", ".join(labels) if labels else "none"


def _memory_selectors_text(selectors: list[int | str | None]) -> str:
    values = ["auto" if selector is None else str(selector) for selector in selectors]
    return ", ".join(values) if values else "none"


def _debug_memory(message: str) -> None:
    if JOYAI_DEBUG_MEMORY:
        print(f"[WAN2GP][JoyAI-Echo][memory] {message}", flush=True)


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
        self.max_size = max(0, int(max_size))
        self.num_fix_frames = max(0, int(num_fix_frames))
        self.audio_window_size = max(1, int(audio_window_size))
        self.entries: dict[int, dict] = {}
        self.created_at = 0

    def __len__(self) -> int:
        return len(self.entries)

    def _slot_items(self):
        return sorted(self.entries.items())

    def _creation_items(self):
        return sorted(self.entries.items(), key=lambda item: item[1].get("created_at", 0))

    def _entry_label(self, slot_id: int, entry: dict) -> str:
        name = entry.get("name")
        return f"{slot_id}[{name}]" if name else str(slot_id)

    def labels(self) -> list[str]:
        return [self._entry_label(slot_id, entry) for slot_id, entry in self._slot_items()]

    def _next_created_at(self) -> int:
        self.created_at += 1
        return self.created_at

    def _slot_for_name(self, name: str) -> int | None:
        for slot_id, entry in self.entries.items():
            if entry.get("name") == name:
                return slot_id
        return None

    def _oldest_slot(self) -> int | None:
        if not self.entries:
            return None
        return self._creation_items()[0][0]

    def _free_slot(self) -> int | None:
        for slot_id in range(1, self.max_size + 1):
            if slot_id not in self.entries:
                return slot_id
        return None

    def drop(self, selectors: list[int | str] | None = None) -> list[str]:
        if not self.entries:
            return []
        slots = {self._creation_items()[-1][0]}
        if selectors is not None:
            slots = set()
            for selector in selectors:
                if isinstance(selector, int):
                    if selector < 1 or selector > self.max_size:
                        raise RuntimeError(f"JoyAI-Echo /drop_mem slot {selector} is outside the valid memory range 1-{self.max_size}.")
                    if selector not in self.entries:
                        raise RuntimeError(f"JoyAI-Echo /drop_mem slot {selector} is already empty.")
                    slots.add(selector)
                else:
                    slot_id = self._slot_for_name(selector)
                    if slot_id is None:
                        raise RuntimeError(f"JoyAI-Echo /drop_mem memory name '{selector}' was not found.")
                    slots.add(slot_id)
        dropped = [self._entry_label(slot_id, self.entries[slot_id]) for slot_id in sorted(slots)]
        for slot_id in slots:
            del self.entries[slot_id]
        return dropped

    def _target_slot(self, selector: int | str | None, entry: dict) -> tuple[int | None, list[str]]:
        if self.max_size <= 0:
            return None, []
        discarded = []
        if isinstance(selector, int):
            if selector < 1 or selector > self.max_size:
                raise RuntimeError(f"JoyAI-Echo /store_mem slot {selector} is outside the valid memory range 1-{self.max_size}.")
            slot_id = selector
        elif isinstance(selector, str):
            slot_id = self._slot_for_name(selector) or self._free_slot()
            entry["name"] = selector
        else:
            slot_id = self._free_slot()
        if slot_id is None:
            slot_id = self._oldest_slot()
            if slot_id is not None:
                discarded.append(self._entry_label(slot_id, self.entries[slot_id]))
        elif slot_id in self.entries:
            discarded.append(self._entry_label(slot_id, self.entries[slot_id]))
        return slot_id, discarded

    def _store_entry(self, selector: int | str | None, entry: dict) -> tuple[str | None, list[str]]:
        slot_id, discarded = self._target_slot(selector, entry)
        if slot_id is None:
            return None, discarded
        entry["created_at"] = self._next_created_at()
        self.entries[slot_id] = entry
        return self._entry_label(slot_id, entry), discarded

    def _build_entry(self, model, phase: str, video_latent: torch.Tensor | None, audio_latent: torch.Tensor | None, audio_waveform=None, audio_sample_rate: int | None = None, center_ratio: float | None = None) -> dict | None:
        if video_latent is None:
            return None
        video_latent = video_latent.detach().cpu().contiguous()
        video_frames = int(video_latent.shape[2])
        if audio_latent is None:
            video_idx = max(0, video_frames // 2) if center_ratio is None else max(0, min(int(round(float(center_ratio) * float(max(video_frames - 1, 0)))), max(video_frames - 1, 0)))
            return {"video": {phase: video_latent[:, :, video_idx : video_idx + 1]}, "audio": {}, "audio_lengths": {}}
        audio_latent = audio_latent.detach().cpu().contiguous()
        total_audio_frames = int(audio_latent.shape[2])
        waveform = None if audio_waveform is None or audio_sample_rate is None else _normalize_waveform(audio_waveform, channels_first=False)
        center_latent = None if center_ratio is None else int(round(float(center_ratio) * float(max(total_audio_frames - 1, 0))))
        window_start, window_len = _select_audio_window_start(model, audio_latent, waveform, audio_sample_rate, self.audio_window_size, center_latent=center_latent)
        window_end = window_start + window_len
        video_idx = _video_idx_from_audio_window(video_frames, total_audio_frames, window_start, window_len)
        return {
            "video": {phase: video_latent[:, :, video_idx : video_idx + 1]},
            "audio": {phase: audio_latent[:, :, window_start:window_end]},
            "audio_lengths": {phase: int(window_len)},
        }

    def add_generation(self, model, memory_latents: dict | None, audio_waveform=None, audio_sample_rate: int | None = None, store_selectors: list[int | str | None] | None = None) -> tuple[list[str], list[str]]:
        if not memory_latents:
            return [], []
        phase_latents = memory_latents if "phase1" in memory_latents or "phase2" in memory_latents else {"phase1": memory_latents}
        store_selectors = store_selectors or [None]
        center_ratios = [None] if len(store_selectors) <= 1 else [(slot + 1) / float(len(store_selectors) + 1) for slot in range(len(store_selectors))]
        stored, discarded = [], []
        for selector, center_ratio in zip(store_selectors, center_ratios):
            entry = {"video": {}, "audio": {}, "audio_lengths": {}}
            for phase, latents in phase_latents.items():
                if not isinstance(latents, dict):
                    continue
                phase_entry = self._build_entry(model, phase, latents.get("video"), latents.get("audio"), audio_waveform, audio_sample_rate, center_ratio=center_ratio)
                if phase_entry is None:
                    continue
                entry["video"].update(phase_entry["video"])
                entry["audio"].update(phase_entry["audio"])
                entry["audio_lengths"].update(phase_entry["audio_lengths"])
            if entry["video"]:
                stored_label, discarded_labels = self._store_entry(selector, entry)
                if stored_label is not None:
                    stored.append(stored_label)
                discarded.extend(discarded_labels)
        return stored, discarded

    def add_artificial_memory(self, memory: dict) -> tuple[list[str], list[str]]:
        phase_video_latents = memory.get("video", {}) if isinstance(memory, dict) else {}
        phase_audio_slots = memory.get("audio", {}) if isinstance(memory, dict) else {}
        if not phase_video_latents:
            return [], []
        stored, discarded = [], []
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
                stored_label, discarded_labels = self._store_entry(None, entry)
                if stored_label is not None:
                    stored.append(stored_label)
                discarded.extend(discarded_labels)
        return stored, discarded

    def video_latent(self, phase: str = "phase1") -> torch.Tensor | None:
        latents = [entry["video"][phase] for _, entry in self._slot_items() if phase in entry["video"]]
        if not latents:
            return None
        return torch.cat(latents, dim=2).contiguous()

    def audio_latent(self, phase: str = "phase1") -> torch.Tensor | None:
        latents = [entry["audio"][phase] for _, entry in self._slot_items() if phase in entry["audio"]]
        if not latents:
            return None
        return torch.cat(latents, dim=2).contiguous()

    def audio_segment_lengths(self, phase: str = "phase1"):
        lengths = [entry["audio_lengths"][phase] for _, entry in self._slot_items() if phase in entry["audio_lengths"]]
        if not lengths:
            return None
        return (tuple(lengths),)

    def paired_audio_memory(self, phase: str = "phase1") -> bool:
        video_slots = sum(1 for entry in self.entries.values() if phase in entry["video"])
        audio_slots = sum(1 for entry in self.entries.values() if phase in entry["audio"])
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


def generate_joyai_echo_window(model, single_shot_generate, **call_args):
    gen_state = call_args.get("gen_state")
    if not isinstance(gen_state, dict):
        gen_state = {}
    joy_state = gen_state.setdefault("joyai_echo", {})
    memory_bank = joy_state.get("memory_bank")
    if memory_bank is None:
        memory_bank = JoyAIEchoMemoryBank(
            max_size=int(model.model_def.get("joyai_memory_max_size", 7)),
            num_fix_frames=int(model.model_def.get("joyai_memory_num_fix_frames", 3)),
            audio_window_size=int(model.model_def.get("joyai_audio_memory_window_size", 96)),
        )
        joy_state["memory_bank"] = memory_bank

    fps = float(call_args.get("fps", model.model_def.get("fps", 25)) or 25)
    custom_settings = call_args.get("custom_settings") if isinstance(call_args.get("custom_settings"), dict) else {}
    control_memory_enabled = "1" in (call_args.get("video_prompt_type", "") or "")
    control_video_path = call_args.get("video_guide")
    control_key = (control_video_path, custom_settings.get(JOYAI_CONTROL_MEMORY_SETTING, ""), int(call_args.get("height")), int(call_args.get("width")), int(call_args.get("guide_phases", call_args.get("guidance_phases", 1)) or 1))
    if control_memory_enabled and joy_state.get("control_key") != control_key:
        if not control_video_path:
            raise RuntimeError("JoyAI-Echo Control Video Memory requires the original Control Video path.")
        target_height, target_width = int(call_args.get("height")), int(call_args.get("width"))
        target_height = int(math.ceil(target_height / 64) * 64) if target_height % 64 else target_height
        target_width = int(math.ceil(target_width / 64) * 64) if target_width % 64 else target_width
        artificial_memory = build_control_video_memory(
            model,
            control_video_path,
            str(custom_settings.get(JOYAI_CONTROL_MEMORY_SETTING, "")),
            fps=fps,
            height=target_height,
            width=target_width,
            two_phase=int(call_args.get("guide_phases", call_args.get("guidance_phases", 1)) or 1) > 1,
            VAE_tile_size=call_args.get("VAE_tile_size"),
        )
        stored, discarded = memory_bank.add_artificial_memory(artificial_memory)
        _debug_memory(f"control video memory stored {len(stored)} slot(s): {_memory_labels_text(stored)}")
        if discarded:
            _debug_memory(f"slots {len(discarded)} discarded: {_memory_labels_text(discarded)}")
        joy_state["control_key"] = control_key

    window_options = call_args.get("frame_window_options") if isinstance(call_args.get("frame_window_options"), dict) else {}
    model_options = window_options.get("model_options", {}) if isinstance(window_options.get("model_options", {}), dict) else {}
    drop_mem = model_options.get("drop_mem", None)
    if drop_mem is not None:
        dropped = memory_bank.drop(parse_drop_mem_option(drop_mem))
        _debug_memory(f"/drop_mem={drop_mem}: slots {len(dropped)} discarded: {_memory_labels_text(dropped)}")
    record_memory = not bool(model_options.get("no_mem", False))
    if "no_mem" in model_options:
        _debug_memory("/no_mem: this window will not record Joy memory")
    store_mem = model_options.get("store_mem", True)
    store_mem_selectors = parse_store_mem_option(store_mem)
    if "store_mem" in model_options:
        store_action = "ignored because /no_mem is set" if not record_memory else f"will try to store {len(store_mem_selectors)} slot(s): {_memory_selectors_text(store_mem_selectors)}"
        _debug_memory(f"/store_mem={store_mem}: {store_action}")
    audio_memory_enabled = bool(model.model_def.get("joyai_audio_memory", False))
    reference_context = {
        "video_latent": memory_bank.video_latent("phase1"),
        "audio_latent": memory_bank.audio_latent("phase1"),
        "audio_segment_lengths": memory_bank.audio_segment_lengths("phase1"),
        "paired_audio": audio_memory_enabled and memory_bank.paired_audio_memory("phase1"),
        "video_latent_stage2": memory_bank.video_latent("phase2"),
        "audio_latent_stage2": memory_bank.audio_latent("phase2"),
        "audio_segment_lengths_stage2": memory_bank.audio_segment_lengths("phase2"),
        "paired_audio_stage2": audio_memory_enabled and memory_bank.paired_audio_memory("phase2"),
        "freeze_stage2_audio": True,
        "downscale_factor": int(model.model_def.get("joyai_memory_downscale_factor", 1)),
        "return_latents": record_memory,
        "debug_memory": JOYAI_DEBUG_MEMORY,
        "window_no": call_args.get("window_no", 1),
        "memory_labels": memory_bank.labels(),
    }

    shot_args = dict(call_args)
    shot_args["joyai_single_window"] = True
    if control_memory_enabled or "V" in (call_args.get("video_prompt_type", "") or ""):
        shot_args.update({"input_frames": None, "input_frames2": None, "input_masks": None, "input_masks2": None, "video_prompt_type": ""})
    print(f"[WAN2GP][JoyAI-Echo] window={call_args.get('window_no', 1)} frames={call_args.get('frame_num')} overlap={call_args.get('prefix_frames_count', 0)} memory_slots={len(memory_bank)} slots={_memory_labels_text(memory_bank.labels())} record_memory={record_memory} store_mem={len(store_mem_selectors)}", flush=True)
    if JOYAI_DEBUG_MEMORY:
        phase1_video_slots = 0 if reference_context["video_latent"] is None else reference_context["video_latent"].shape[2]
        phase2_video_slots = 0 if reference_context["video_latent_stage2"] is None else reference_context["video_latent_stage2"].shape[2]
        phase1_audio_slots = 0 if reference_context["audio_segment_lengths"] is None else len(reference_context["audio_segment_lengths"][0])
        phase2_audio_slots = 0 if reference_context["audio_segment_lengths_stage2"] is None else len(reference_context["audio_segment_lengths_stage2"][0])
        _debug_memory(f"window={call_args.get('window_no', 1)} guiding memory to inject: phase1_video={phase1_video_slots} phase1_audio={phase1_audio_slots} phase2_video={phase2_video_slots} phase2_audio={phase2_audio_slots} slots={_memory_labels_text(reference_context['memory_labels'])}")
    with model.pipeline.joyai_echo_context(reference_context):
        result = single_shot_generate(**shot_args)
    if result is None:
        return None
    memory_latents = result.pop("_memory_latents", None)
    if record_memory:
        trim_frames = max(0, int(call_args.get("prefix_frames_count", 0) or 0))
        stored, discarded = memory_bank.add_generation(
            model,
            _trim_memory_latents(model, memory_latents, trim_frames, int(call_args.get("frame_num", 0) or 0)),
            audio_waveform=_trim_audio_start(result.get("audio"), trim_frames, fps, result.get("audio_sampling_rate")),
            audio_sample_rate=result.get("audio_sampling_rate"),
            store_selectors=store_mem_selectors,
        )
        _debug_memory(f"stored {len(stored)} slot(s): {_memory_labels_text(stored)}")
        if discarded:
            _debug_memory(f"slots {len(discarded)} discarded: {_memory_labels_text(discarded)}")
    return result
