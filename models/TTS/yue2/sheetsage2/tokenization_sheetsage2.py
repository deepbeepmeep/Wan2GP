import re
import hashlib
import json

import mir_eval.chord
import numpy as np

from .durations_sheetsage2 import DURATION_TEMPLATES, duration_boundaries
from .labels_sheetsage2 import STRUCTURE_LABELS
from .schema_sheetsage2 import get_prompt_multitask_schema


PROMPT_ORDER = tuple(task.name for task in get_prompt_multitask_schema("v1").tasks)


CHROMATIC_SHARPS = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

FULL_CHORD_QUALITIES = (
    "maj",
    "min",
    "dim",
    "aug",
    "maj7",
    "min7",
    "7",
    "hdim7",
    "dim7",
    "minmaj7",
    "sus2",
    "sus4",
    "sus4(b7)",
    "maj6",
    "min6",
)

FULL_CHORD_INVERSIONS = {
    "maj": ("/2", "/3", "/5"),
    "min": ("/2", "/b3", "/5"),
    "maj7": ("/3", "/5", "/7"),
    "min7": ("/b3", "/5", "/b7"),
    "7": ("/3", "/5", "/b7"),
}


def build_full_chord_vocabulary():
    labels = ["N"]
    for quality in FULL_CHORD_QUALITIES:
        inversions = FULL_CHORD_INVERSIONS.get(quality, ())
        for root in CHROMATIC_SHARPS:
            for inversion in (*inversions, ""):
                labels.append(f"{root}:{quality}{inversion}")
    return tuple(labels)


FULL_CHORD_VOCABULARY = build_full_chord_vocabulary()


class SheetSage2Tokenizer:
    """Typed prompt and event vocabulary for prompt-conditioned transcription."""

    meter_numerators = tuple(range(1, 33))
    meter_denominators = (1, 2, 4, 8, 16, 32)
    n_eighth_positions = 256
    max_subbeat_shift = 256
    prompt_capacity = 256

    def __init__(
        self,
        audio_length_seconds=300.0,
        time_hz=100,
        schema_version="v1",
        expected_fingerprint=None,
    ):
        self.audio_length_seconds = float(audio_length_seconds)
        self.time_hz = int(time_hz)
        self.schema = get_prompt_multitask_schema(schema_version)
        self.schema_version = self.schema.version
        self.task_specs = self.schema.tasks
        self.event_field_order = self.schema.event_field_order
        self.n_time_tokens = int(round(self.audio_length_seconds * self.time_hz))
        if self.n_time_tokens <= 0:
            raise ValueError("audio_length_seconds must produce at least one time token")

        self.pad_token = 0
        self.sos_token = 1
        self.bos_token = self.sos_token
        self.eos_token = 2
        self.out_token = 3

        self.prompt_token_start = 4
        self.prompt_names = tuple(task.name for task in self.task_specs)
        if len(self.prompt_names) > self.prompt_capacity:
            raise ValueError(
                f"Schema {self.schema_version} has {len(self.prompt_names)} prompts, "
                f"exceeding immutable capacity {self.prompt_capacity}"
            )
        self.prompt_to_id = {
            name: self.prompt_token_start + index
            for index, name in enumerate(self.prompt_names)
        }
        self.prompt_token_end = self.prompt_token_start + self.prompt_capacity

        self.subbeat_shift_token_start = self.prompt_token_end
        self.n_subbeat_shift_tokens = self.max_subbeat_shift + 1
        self.subbeat_shift_token_end = (
            self.subbeat_shift_token_start + self.n_subbeat_shift_tokens
        )

        self.time_token_start = self.subbeat_shift_token_end
        self.time_token_end = self.time_token_start + self.n_time_tokens

        self.meter_pairs = tuple(
            (numerator, denominator)
            for numerator in self.meter_numerators
            for denominator in self.meter_denominators
        )
        self.meter_to_id = {meter: index for index, meter in enumerate(self.meter_pairs)}
        self.meter_token_start = self.time_token_end
        self.meter_token_end = self.meter_token_start + len(self.meter_pairs)

        self.eighth_position_token_start = self.meter_token_end
        self.eighth_position_token_end = (
            self.eighth_position_token_start + self.n_eighth_positions
        )

        self.structure_labels = tuple(STRUCTURE_LABELS)
        self.structure_token_start = self.eighth_position_token_end
        self.structure_token_end = self.structure_token_start + len(self.structure_labels)

        self.key_token_start = self.structure_token_end
        self.n_key_tokens = 24
        self.key_token_end = self.key_token_start + self.n_key_tokens

        self.majmin_chord_labels = (
            "N",
            *(f"{root}:maj" for root in CHROMATIC_SHARPS),
            *(f"{root}:min" for root in CHROMATIC_SHARPS),
        )
        self.majmin_chord_token_start = self.key_token_end
        self.majmin_chord_token_end = (
            self.majmin_chord_token_start + len(self.majmin_chord_labels)
        )

        self.full_chord_labels = FULL_CHORD_VOCABULARY
        self.full_chord_to_id = {
            label: index for index, label in enumerate(self.full_chord_labels)
        }
        self.full_chord_token_start = self.majmin_chord_token_end
        self.full_chord_token_end = (
            self.full_chord_token_start + len(self.full_chord_labels)
        )

        self.pitch_token_start = self.full_chord_token_end
        self.n_pitch_tokens = 256
        self.pitch_token_end = self.pitch_token_start + self.n_pitch_tokens

        self.duration_templates = DURATION_TEMPLATES
        self.duration_boundaries = duration_boundaries
        self.duration_token_start = self.pitch_token_end
        self.n_duration_tokens = len(self.duration_templates)
        self.duration_token_end = self.duration_token_start + self.n_duration_tokens
        self.appended_token_blocks = {}
        next_token = self.duration_token_end
        for block in self.schema.appended_token_blocks:
            start = next_token
            end = start + len(block.labels)
            self.appended_token_blocks[block.name] = {
                "start": start,
                "end": end,
                "labels": tuple(block.labels),
                "output_field": block.output_field or block.name,
            }
            next_token = end
        self.n_tokens = next_token

        self._full_chord_templates = self._build_full_chord_templates()
        self.vocab_fingerprint = self._compute_fingerprint()
        if (
            expected_fingerprint is not None
            and str(expected_fingerprint) != self.vocab_fingerprint
        ):
            raise ValueError(
                f"Tokenizer fingerprint mismatch for schema {self.schema_version}: "
                f"expected {expected_fingerprint}, got {self.vocab_fingerprint}"
            )

    def _compute_fingerprint(self):
        payload = {
            "schema_version": self.schema_version,
            "audio_length_seconds": self.audio_length_seconds,
            "time_hz": self.time_hz,
            "prompt_capacity": self.prompt_capacity,
            "prompt_names": self.prompt_names,
            "event_field_order": self.event_field_order,
            "meter_pairs": self.meter_pairs,
            "structure_labels": self.structure_labels,
            "majmin_chord_labels": self.majmin_chord_labels,
            "full_chord_labels": self.full_chord_labels,
            "duration_templates": tuple(int(value) for value in self.duration_templates),
            "appended_token_blocks": tuple(
                (name, block["labels"], block["output_field"])
                for name, block in self.appended_token_blocks.items()
            ),
            "n_tokens": self.n_tokens,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()[:16]

    def get_config(self):
        return {
            "schema_version": self.schema_version,
            "audio_length_seconds": self.audio_length_seconds,
            "time_hz": self.time_hz,
            "vocab_fingerprint": self.vocab_fingerprint,
            "n_tokens": self.n_tokens,
        }

    def normalize_prompts(self, prompts):
        names = []
        seen = set()
        for prompt in prompts:
            name = str(prompt).strip()
            if name.startswith("<|") and name.endswith("|>"):
                name = name[2:-2]
            if name not in self.prompt_to_id:
                raise ValueError(f"Unknown prompt: {prompt!r}")
            if name not in seen:
                names.append(name)
                seen.add(name)
        names.sort(key=self.prompt_names.index)
        selected_groups = {}
        task_by_name = {task.name: task for task in self.task_specs}
        for name in names:
            group = task_by_name[name].sampling_group
            previous = selected_groups.get(group)
            if previous is not None:
                raise ValueError(
                    f"Prompts {previous!r} and {name!r} are mutually exclusive "
                    f"within sampling group {group!r}"
                )
            selected_groups[group] = name
        if not names:
            raise ValueError("At least one task prompt is required")
        return tuple(names)

    def prompt_to_token(self, prompt):
        return self.prompt_to_id[self.normalize_prompts((prompt,))[0]]

    def token_to_prompt(self, token):
        token = int(token)
        for prompt, prompt_token in self.prompt_to_id.items():
            if token == prompt_token:
                return prompt
        raise ValueError(f"token {token} is not a prompt token")

    def prompt_prefix(self, prompts):
        prompts = self.normalize_prompts(prompts)
        return [
            self.sos_token,
            *(self.prompt_to_id[prompt] for prompt in prompts),
            self.out_token,
        ]

    def subbeat_shift_to_tokens(self, shift):
        shift = int(shift)
        if shift < 0:
            raise ValueError("subbeat shift must be non-negative")
        tokens = []
        while shift > self.max_subbeat_shift:
            tokens.append(self.subbeat_shift_token_start + self.max_subbeat_shift)
            shift -= self.max_subbeat_shift
        tokens.append(self.subbeat_shift_token_start + shift)
        return tokens

    def token_to_subbeat_shift(self, token):
        token = int(token)
        if not self.subbeat_shift_token_start <= token < self.subbeat_shift_token_end:
            raise ValueError(f"token {token} is not a subbeat shift token")
        return token - self.subbeat_shift_token_start

    def time_id_to_token(self, time_id):
        time_id = int(time_id)
        if not 0 <= time_id < self.n_time_tokens:
            raise ValueError(f"time id {time_id} is outside [0, {self.n_time_tokens})")
        return self.time_token_start + time_id

    def token_to_time_id(self, token):
        token = int(token)
        if not self.time_token_start <= token < self.time_token_end:
            raise ValueError(f"token {token} is not a time token")
        return token - self.time_token_start

    def meter_to_token(self, numerator, denominator):
        meter = (int(numerator), int(denominator))
        if meter not in self.meter_to_id:
            raise ValueError(f"Unsupported meter {meter[0]}/{meter[1]}")
        return self.meter_token_start + self.meter_to_id[meter]

    def eighth_position_to_token(self, position):
        position = int(position)
        if not 0 <= position < self.n_eighth_positions:
            raise ValueError(
                f"eighth-note position {position} is outside [0, {self.n_eighth_positions})"
            )
        return self.eighth_position_token_start + position

    def structure_to_token(self, label):
        label = str(label)
        if label not in self.structure_labels:
            raise ValueError(f"Unknown structure label: {label!r}")
        return self.structure_token_start + self.structure_labels.index(label)

    def key_to_token(self, key_label, pitch_shift=0):
        tonic, mode = str(key_label).split(":", 1)
        tonic_id = mir_eval.chord.pitch_class_to_semitone(tonic)
        if tonic_id < 0:
            raise ValueError(f"Invalid key tonic: {key_label!r}")
        tonic_id = (int(tonic_id) + int(pitch_shift)) % 12
        minor_modes = {"minor", "dorian", "phrygian", "locrian"}
        mode_id = 1 if mode.lower() in minor_modes else 0
        return self.key_token_start + mode_id * 12 + tonic_id

    @staticmethod
    def _canonical_chord_parts(chord_label, pitch_shift=0):
        chord_label = str(chord_label).strip()
        if chord_label in {"N", "X", ""}:
            return "N", None, None
        root, suffix = chord_label.split(":", 1)
        root_id = mir_eval.chord.pitch_class_to_semitone(root)
        if root_id < 0:
            return "N", None, None
        root_id = (int(root_id) + int(pitch_shift)) % 12
        canonical = f"{CHROMATIC_SHARPS[root_id]}:{suffix}"
        quality = re.split(r"/", suffix, maxsplit=1)[0]
        return canonical, root_id, quality

    def chord_majmin_to_token(self, chord_label, pitch_shift=0):
        canonical, root_id, quality = self._canonical_chord_parts(
            chord_label,
            pitch_shift,
        )
        if canonical == "N":
            return self.majmin_chord_token_start

        try:
            original_root, chroma, _ = mir_eval.chord.encode(str(chord_label))
            relative = mir_eval.chord.rotate_bitmap_to_root(chroma, original_root)
            has_minor_third = bool(relative[3] > 0)
            has_major_third = bool(relative[4] > 0)
        except Exception:
            has_minor_third = quality.startswith(("min", "dim", "hdim"))
            has_major_third = not has_minor_third

        if has_major_third and not has_minor_third:
            quality_id = 0
        elif has_minor_third and not has_major_third:
            quality_id = 1
        elif quality.startswith(("min", "dim", "hdim")):
            quality_id = 1
        elif has_major_third:
            quality_id = 0
        else:
            return self.majmin_chord_token_start
        return self.majmin_chord_token_start + 1 + quality_id * 12 + root_id

    @staticmethod
    def _chord_template(chord_label):
        root, chroma, bass = mir_eval.chord.encode(chord_label)
        root_chroma = np.zeros(12, dtype=np.float32)
        root_chroma[root] = 1.0
        relative_chroma = mir_eval.chord.rotate_bitmap_to_root(chroma, root).astype(
            np.float32
        )
        bass_chroma = np.zeros(12, dtype=np.float32)
        bass_chroma[(bass + root) % 12] = 1.0
        return np.concatenate([root_chroma, relative_chroma, bass_chroma])

    def _build_full_chord_templates(self):
        templates = [np.zeros(36, dtype=np.float32)]
        templates.extend(
            self._chord_template(label) for label in self.full_chord_labels[1:]
        )
        return np.stack(templates)

    def chord_full_to_token(self, chord_label, pitch_shift=0):
        canonical, _root_id, _quality = self._canonical_chord_parts(
            chord_label,
            pitch_shift,
        )
        chord_id = self.full_chord_to_id.get(canonical)
        if chord_id is None:
            if canonical == "N":
                chord_id = 0
            else:
                try:
                    target = self._chord_template(canonical)
                    distances = np.abs(self._full_chord_templates - target[None]).sum(
                        axis=1
                    )
                    chord_id = int(np.argmin(distances))
                except Exception:
                    chord_id = 0
        return self.full_chord_token_start + chord_id

    def pitch_to_token(self, pitch, track=0, full_melody=False):
        pitch = int(pitch)
        track = int(track)
        if not 0 <= pitch < 128:
            raise ValueError(f"MIDI pitch {pitch} is outside [0, 128)")
        if track not in (0, 1):
            raise ValueError(f"melody track {track} is outside [0, 2)")
        pitch_id = pitch + (128 if full_melody and track == 1 else 0)
        return self.pitch_token_start + pitch_id

    def duration_bin_to_token(self, duration_bin):
        duration_bin = int(duration_bin)
        if not 0 <= duration_bin < self.n_duration_tokens:
            raise ValueError(
                f"duration bin {duration_bin} is outside [0, {self.n_duration_tokens})"
            )
        return self.duration_token_start + duration_bin

    def token_to_duration_bin(self, token):
        token = int(token)
        if not self.duration_token_start <= token < self.duration_token_end:
            raise ValueError(f"token {token} is not a duration token")
        return token - self.duration_token_start

    def appended_label_to_token(self, block_name, label):
        block = self.appended_token_blocks.get(str(block_name))
        if block is None:
            raise ValueError(f"unknown appended token block: {block_name!r}")
        try:
            index = block["labels"].index(str(label))
        except ValueError as exc:
            raise ValueError(
                f"unknown label {label!r} for appended block {block_name!r}"
            ) from exc
        return block["start"] + index

    def token_to_appended_label(self, token):
        token = int(token)
        token_type = self.token_type(token)
        block = self.appended_token_blocks.get(token_type)
        if block is None:
            raise ValueError(f"token {token} is not from an appended token block")
        return token_type, block["labels"][token - block["start"]]

    def _token_output_field(self, token_type):
        built_in = {
            "time": "timestamp",
            "meter": "rhythm",
            "eighth_position": "rhythm",
            "structure": "structure",
            "key": "key",
            "chord_majmin": "chord",
            "chord_full": "chord",
            "pitch": "melody",
            "duration": "melody",
        }
        if token_type in built_in:
            return built_in[token_type]
        block = self.appended_token_blocks.get(token_type)
        return None if block is None else block["output_field"]

    def _decode_field(self, field, field_tokens, prompts):
        token_types = [self.token_type(token) for token in field_tokens]
        if field == "timestamp":
            return self.token_to_time_id(field_tokens[0]) / self.time_hz
        if field == "rhythm":
            rhythm = {}
            for token, token_type in zip(field_tokens, token_types):
                if token_type == "meter":
                    rhythm["meter"] = self.meter_pairs[token - self.meter_token_start]
                elif token_type == "eighth_position":
                    rhythm["eighth_position"] = token - self.eighth_position_token_start
            return rhythm
        if field == "structure":
            return self.structure_labels[field_tokens[0] - self.structure_token_start]
        if field == "key":
            key_id = field_tokens[0] - self.key_token_start
            mode = "minor" if key_id >= 12 else "major"
            return f"{CHROMATIC_SHARPS[key_id % 12]}:{mode}"
        if field == "chord":
            if token_types[0] == "chord_majmin":
                return self.majmin_chord_labels[
                    field_tokens[0] - self.majmin_chord_token_start
                ]
            return self.full_chord_labels[
                field_tokens[0] - self.full_chord_token_start
            ]
        if field == "melody":
            notes = []
            index = 0
            while index < len(field_tokens):
                pitch_id = field_tokens[index] - self.pitch_token_start
                duration_bin = 0
                if (
                    index + 1 < len(field_tokens)
                    and self.token_type(field_tokens[index + 1]) == "duration"
                ):
                    duration_bin = self.token_to_duration_bin(field_tokens[index + 1])
                    index += 2
                else:
                    index += 1
                notes.append(
                    {
                        "pitch": pitch_id % 128,
                        "track": int(pitch_id >= 128),
                        "duration_bin": duration_bin,
                        "duration_steps": int(self.duration_templates[duration_bin]),
                    }
                )
            return notes
        values = []
        for token in field_tokens:
            block_name, label = self.token_to_appended_label(token)
            values.append({"block": block_name, "label": label})
        return values[0] if len(values) == 1 else values

    def decode_sequence(self, tokens, strict=True):
        """Parse one prompt-conditioned sequence into timed, typed events."""
        if hasattr(tokens, "detach"):
            tokens = tokens.detach().cpu().tolist()
        tokens = [int(token) for token in tokens]
        while tokens and tokens[-1] == self.pad_token:
            tokens.pop()
        if not tokens or tokens[0] != self.sos_token:
            raise ValueError("sequence must begin with <|sos|>")

        try:
            out_index = tokens.index(self.out_token, 1)
        except ValueError as exc:
            raise ValueError("sequence is missing <|out|>") from exc
        prompts = tuple(self.token_to_prompt(token) for token in tokens[1:out_index])
        if strict and self.normalize_prompts(prompts) != prompts:
            raise ValueError("prompt tokens are not in canonical schema order")
        active_fields = {
            task.output_field for task in self.task_specs if task.name in prompts
        }

        events = []
        position = out_index + 1
        current_step = 0
        saw_eos = False
        while position < len(tokens):
            token = tokens[position]
            if token == self.eos_token:
                saw_eos = True
                position += 1
                break
            if self.token_type(token) != "subbeat_shift":
                raise ValueError(f"event at token index {position} has no subbeat shift")
            shift = 0
            while (
                position < len(tokens)
                and self.token_type(tokens[position]) == "subbeat_shift"
            ):
                shift += self.token_to_subbeat_shift(tokens[position])
                position += 1
            current_step += shift

            tokens_by_field = {field: [] for field in self.event_field_order}
            while position < len(tokens):
                token = tokens[position]
                token_type = self.token_type(token)
                if token_type == "subbeat_shift" or token == self.eos_token:
                    break
                field = self._token_output_field(token_type)
                if field is None or field not in tokens_by_field:
                    raise ValueError(
                        f"token {token} ({token_type}) has no field in schema "
                        f"{self.schema_version}"
                    )
                if strict and field not in active_fields:
                    raise ValueError(
                        f"token {token} belongs to inactive output field {field!r}"
                    )
                tokens_by_field[field].append(token)
                position += 1

            tokens_by_field = {
                field: values for field, values in tokens_by_field.items() if values
            }
            if not tokens_by_field:
                if strict:
                    raise ValueError(f"empty event at subbeat {current_step}")
                continue
            if strict:
                for field, values in tokens_by_field.items():
                    types = [self.token_type(value) for value in values]
                    if field == "timestamp" and types != ["time"]:
                        raise ValueError("timestamp event must contain exactly one time token")
                    if field == "rhythm":
                        if types not in (["eighth_position"], ["meter", "eighth_position"]):
                            raise ValueError(f"invalid rhythm payload: {types}")
                    if field in {"structure", "key", "chord"} and len(values) != 1:
                        raise ValueError(f"field {field!r} must contain exactly one token")
                    if field == "melody":
                        index = 0
                        while index < len(types):
                            if types[index] != "pitch":
                                raise ValueError(
                                    "melody payload must contain pitch tokens with optional duration"
                                )
                            if index + 1 < len(types) and types[index + 1] == "duration":
                                index += 2
                            else:
                                index += 1
            events.append(
                {
                    "subbeat": current_step,
                    "tokens_by_field": tokens_by_field,
                    "values": {
                        field: self._decode_field(field, values, prompts)
                        for field, values in tokens_by_field.items()
                    },
                }
            )

        if strict and not saw_eos:
            raise ValueError("sequence is missing <|eos|>")
        if strict and position != len(tokens):
            raise ValueError("non-padding tokens follow <|eos|>")
        return {
            "schema_version": self.schema_version,
            "prompts": prompts,
            "events": events,
            "has_eos": saw_eos,
        }

    def encode_decoded_sequence(self, decoded):
        """Re-encode the lossless representation returned by decode_sequence."""
        prompts = self.normalize_prompts(decoded["prompts"])
        output = self.prompt_prefix(prompts)
        previous_step = 0
        for event in decoded["events"]:
            step = int(event["subbeat"])
            if step < previous_step:
                raise ValueError("events must be sorted by non-decreasing subbeat")
            output.extend(self.subbeat_shift_to_tokens(step - previous_step))
            previous_step = step
            tokens_by_field = event["tokens_by_field"]
            for field in self.event_field_order:
                output.extend(int(token) for token in tokens_by_field.get(field, ()))
        if decoded.get("has_eos", True):
            output.append(self.eos_token)
        return output

    def token_type(self, token):
        token = int(token)
        for name, start, end in (
            ("subbeat_shift", self.subbeat_shift_token_start, self.subbeat_shift_token_end),
            ("time", self.time_token_start, self.time_token_end),
            ("meter", self.meter_token_start, self.meter_token_end),
            ("eighth_position", self.eighth_position_token_start, self.eighth_position_token_end),
            ("structure", self.structure_token_start, self.structure_token_end),
            ("key", self.key_token_start, self.key_token_end),
            ("chord_majmin", self.majmin_chord_token_start, self.majmin_chord_token_end),
            ("chord_full", self.full_chord_token_start, self.full_chord_token_end),
            ("pitch", self.pitch_token_start, self.pitch_token_end),
            ("duration", self.duration_token_start, self.duration_token_end),
        ):
            if start <= token < end:
                return name
        if token in self.prompt_to_id.values():
            return "prompt"
        for name, block in self.appended_token_blocks.items():
            if block["start"] <= token < block["end"]:
                return name
        if token == self.pad_token:
            return "pad"
        if token == self.sos_token:
            return "sos"
        if token == self.eos_token:
            return "eos"
        if token == self.out_token:
            return "out"
        raise ValueError(f"token {token} is outside vocabulary size {self.n_tokens}")

    def describe(self, token):
        token = int(token)
        token_type = self.token_type(token)
        if token_type == "prompt":
            prompt_by_id = {value: key for key, value in self.prompt_to_id.items()}
            return f"<|{prompt_by_id[token]}|>"
        if token_type == "subbeat_shift":
            return f"<subbeat_shift_{token - self.subbeat_shift_token_start}>"
        if token_type == "time":
            time_id = token - self.time_token_start
            return f"<time_{time_id / self.time_hz:.2f}s>"
        if token_type == "meter":
            numerator, denominator = self.meter_pairs[token - self.meter_token_start]
            return f"<meter_{numerator}/{denominator}>"
        if token_type == "eighth_position":
            return f"<eighth_pos_{token - self.eighth_position_token_start}>"
        if token_type == "structure":
            return f"<structure_{self.structure_labels[token - self.structure_token_start]}>"
        if token_type == "key":
            key_id = token - self.key_token_start
            mode = "minor" if key_id >= 12 else "major"
            return f"<key_{CHROMATIC_SHARPS[key_id % 12]}:{mode}>"
        if token_type == "chord_majmin":
            return f"<chord_majmin_{self.majmin_chord_labels[token - self.majmin_chord_token_start]}>"
        if token_type == "chord_full":
            return f"<chord_full_{self.full_chord_labels[token - self.full_chord_token_start]}>"
        if token_type == "pitch":
            pitch = token - self.pitch_token_start
            track = 1 if pitch >= 128 else 0
            return f"<pitch_{pitch % 128}_track_{track}>"
        if token_type == "duration":
            return f"<duration_{token - self.duration_token_start}>"
        if token_type in self.appended_token_blocks:
            block = self.appended_token_blocks[token_type]
            label = block["labels"][token - block["start"]]
            return f"<{token_type}_{label}>"
        return f"<|{token_type}|>"
