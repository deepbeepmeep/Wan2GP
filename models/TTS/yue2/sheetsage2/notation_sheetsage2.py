"""Convert timed musical events into validated two-voice ABC notation."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pretty_midi

from .io_sheetsage2 import atomic_write_text


SUBBEAT_DIVISION = 4
VOICE_IDS = ("Vocal", "Ins")
NO_CHORDS = frozenset({"N", "X", "?"})


class AbcRebuildError(ValueError):
    """Base class for deterministic reconstruction failures."""


class BeatGridError(AbcRebuildError):
    pass


class ChordSymbolError(AbcRebuildError):
    pass


class MelodyVoiceError(AbcRebuildError):
    pass


@dataclass(frozen=True)
class BeatEvent:
    time: float
    beat_id: int
    declared_numerator: int
    denominator: int
    line_no: int


@dataclass(frozen=True)
class Measure:
    index: int
    start_beat: int
    end_beat: int
    numerator: int
    denominator: int
    pickup: bool = False
    partial: bool = False
    inferred: bool = False
    notated_numerator: int | None = None
    notated_denominator: int | None = None
    pad_before: bool = False

    @property
    def beat_count(self) -> int:
        return self.end_beat - self.start_beat

    @property
    def start_t(self) -> int:
        return self.start_beat * SUBBEAT_DIVISION

    @property
    def end_t(self) -> int:
        return self.end_beat * SUBBEAT_DIVISION

    @property
    def abc_numerator(self) -> int:
        return self.notated_numerator or self.numerator

    @property
    def abc_denominator(self) -> int:
        return self.notated_denominator or self.denominator


@dataclass
class RebuiltAbcScore:
    beats: list[BeatEvent]
    measures: list[Measure]
    subbeat_times: np.ndarray
    subbeat_quarters: np.ndarray
    subbeat_denominators: np.ndarray
    key_arr: np.ndarray
    chord_arr: np.ndarray
    structure_events: list[tuple[int, str]]
    voice_arrs: dict[str, np.ndarray]
    diagnostics: list[str]
    subbeat_div: int = SUBBEAT_DIVISION


@dataclass
class MeasureGroup:
    measures: list[Measure]
    structure_labels: list[str]
    meter_changed: bool
    key_changed: bool


_QUALITY_TO_ABC = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "sus4": "sus4",
    "sus2": "sus2",
    "maj6": "6",
    "min6": "m6",
    "sus4(b7)": "7sus4",
    # abc2midi and SymMusic both accept the parenthesized major seventh.
    # Common aliases such as mmaj7/mM7 trigger abc2midi diagnostics.
    "minmaj7": "m(maj7)",
}

_NATURAL_PITCH_CLASS = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
_LETTERS = "CDEFGAB"
_SHARP_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_PITCH_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_ROOT_RE = re.compile(r"^(?P<letter>[A-G])(?P<accidental>#{0,2}|b{0,2})$")
_BASS_DEGREE_RE = re.compile(r"^(?P<accidental>#{0,2}|b{0,2})(?P<degree>[1-9]|1[0-3])$")

_KEY_SIGNATURE_ACCIDENTALS = {
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
    "F#": 6,
    "C#": 7,
    "F": -1,
    "Bb": -2,
    "Eb": -3,
    "Ab": -4,
    "Db": -5,
    "Gb": -6,
    "Cb": -7,
    "Am": 0,
    "Em": 1,
    "Bm": 2,
    "F#m": 3,
    "C#m": 4,
    "G#m": 5,
    "D#m": 6,
    "A#m": 7,
    "Dm": -1,
    "Gm": -2,
    "Cm": -3,
    "Fm": -4,
    "Bbm": -5,
    "Ebm": -6,
    "Abm": -7,
}

# Keep the standard key-relative chromatic spelling.  MIDI carries only
# pitch, not note names, so a deterministic key-based table is preferable to
# rewriting every chromatic pitch with a single sharp/flat.  In remote sharp
# and flat keys this deliberately permits musically useful double accidentals,
# for example MIDI G as F## in G# minor.
_KEY_RELATIVE_PITCH_NAMES = {
    7: ("B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B"),
    6: ("B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B"),
    5: ("B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"),
    4: ("B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"),
    3: ("B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"),
    2: ("C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"),
    1: ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"),
    0: ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"),
    -1: ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"),
    -2: ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"),
    -3: ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"),
    -4: ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"),
    -5: ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"),
    -6: ("C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"),
    -7: ("C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb"),
}


def _read_tsv(path: os.PathLike[str] | str, min_columns: int) -> list[tuple[int, list[str]]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            columns = line.split("\t")
            if len(columns) < min_columns:
                raise AbcRebuildError(
                    f"{path}:{line_no}: expected at least {min_columns} tab-separated columns"
                )
            rows.append((line_no, columns))
    return rows


def read_beats(path: os.PathLike[str] | str) -> list[BeatEvent]:
    return _parse_beats(_read_tsv(path, 3), path)


def _row_entries(rows, source):
    entries = []
    for line_no, row in enumerate(rows, 1):
        if len(row) < 3:
            raise AbcRebuildError(f"{source}:{line_no}: expected at least 3 columns")
        entries.append((line_no, [str(value) for value in row]))
    return entries


def _parse_beats(entries, path):
    beats = []
    for line_no, row in entries:
        meter_text = row[2]
        if len(row) >= 4:
            numerator_text, denominator_text = meter_text, row[3]
        elif "/" in meter_text:
            numerator_text, denominator_text = meter_text.split("/", 1)
        else:
            numerator_text, denominator_text = meter_text, "4"
        try:
            beat = BeatEvent(
                time=float(row[0]),
                beat_id=int(row[1]),
                declared_numerator=int(numerator_text),
                denominator=int(denominator_text),
                line_no=line_no,
            )
        except ValueError as exc:
            raise BeatGridError(f"{path}:{line_no}: invalid beat row {row!r}") from exc
        if beat.beat_id < 1:
            raise BeatGridError(f"{path}:{line_no}: beat ID must be positive")
        if beat.declared_numerator < 1:
            raise BeatGridError(f"{path}:{line_no}: meter numerator must be positive")
        if beat.denominator < 1 or beat.denominator & (beat.denominator - 1):
            raise BeatGridError(
                f"{path}:{line_no}: meter denominator must be a positive power of two"
            )
        if beats and beat.time <= beats[-1].time:
            raise BeatGridError(f"{path}:{line_no}: beat times must be strictly increasing")
        beats.append(beat)
    if len(beats) < 2:
        raise BeatGridError(f"{path}: at least two beat events are required")
    return beats


def read_chords(path: os.PathLike[str] | str) -> list[tuple[float, float, str]]:
    return _parse_chords(_read_tsv(path, 3), path)


def _parse_chords(entries, path):
    rows = []
    previous_end = None
    for line_no, row in entries:
        start, end, chord = float(row[0]), float(row[1]), row[2].strip()
        if end <= start:
            raise ChordSymbolError(f"{path}:{line_no}: chord end must be after start")
        if previous_end is not None and start < previous_end - 1e-6:
            raise ChordSymbolError(f"{path}:{line_no}: overlapping chord intervals")
        chord_symbol_to_abc(chord)
        rows.append((start, end, chord))
        previous_end = end
    return rows


def read_keys(path: os.PathLike[str] | str) -> list[tuple[float, float, str]]:
    return _parse_keys(_read_tsv(path, 3), path)


def _parse_keys(entries, path):
    rows = []
    previous_end = None
    for line_no, row in entries:
        start, end, key = float(row[0]), float(row[1]), row[2].strip()
        if end <= start:
            raise AbcRebuildError(f"{path}:{line_no}: key end must be after start")
        if previous_end is not None and start < previous_end - 1e-6:
            raise AbcRebuildError(f"{path}:{line_no}: overlapping key intervals")
        normalized = key_symbol_to_abc(key)
        rows.append((start, end, normalized))
        previous_end = end
    if not rows:
        raise AbcRebuildError(f"{path}: at least one key interval is required")
    return rows


def read_structures(path: os.PathLike[str] | str) -> list[tuple[float, float, str]]:
    return _parse_structures(_read_tsv(path, 3), path)


def _parse_structures(entries, path):
    rows = []
    previous_end = None
    for line_no, row in entries:
        start, end, label = float(row[0]), float(row[1]), row[2].strip()
        if end <= start:
            raise AbcRebuildError(f"{path}:{line_no}: structure end must be after start")
        if previous_end is not None and start < previous_end - 1e-6:
            raise AbcRebuildError(f"{path}:{line_no}: overlapping structure intervals")
        rows.append((start, end, label))
        previous_end = end
    return rows


def _mode_with_first_tiebreak(values: Sequence[int]) -> int:
    counts = Counter(values)
    maximum = max(counts.values())
    return next(value for value in values if counts[value] == maximum)


def infer_measures(
    beats: Sequence[BeatEvent],
    *,
    meter_conflict: str = "infer",
) -> tuple[list[Measure], list[str]]:
    """Infer self-consistent measures from actual downbeat boundaries."""

    if meter_conflict not in {"infer", "reject"}:
        raise ValueError("meter_conflict must be 'infer' or 'reject'")
    downbeat_indices = [index for index, beat in enumerate(beats) if beat.beat_id == 1]
    if not downbeat_indices:
        raise BeatGridError("No downbeat (beat ID 1) exists in the beat lab")
    spans: list[tuple[int, int, bool, bool]] = []
    if downbeat_indices[0] > 0:
        spans.append((0, downbeat_indices[0], True, False))
    spans.extend(
        (start, end, False, False)
        for start, end in zip(downbeat_indices, downbeat_indices[1:])
    )
    if downbeat_indices[-1] < len(beats) - 1:
        # Exported beat labs use their last row as the score end boundary.  If
        # that row is not a downbeat, the final bar is intentionally truncated.
        spans.append((downbeat_indices[-1], len(beats) - 1, False, True))
    if not spans:
        raise BeatGridError("No positive-length measure exists between downbeats")

    diagnostics = []
    measures = []
    for measure_index, (start, end, pickup, partial) in enumerate(spans):
        events = list(beats[start:end])
        beat_count = len(events)
        if beat_count < 1:
            raise BeatGridError(f"Measure {measure_index}: empty downbeat span")
        ids = [event.beat_id for event in events]
        expected_ids = list(range(ids[0], ids[0] + beat_count))
        if ids != expected_ids:
            line_numbers = [event.line_no for event in events]
            raise BeatGridError(
                f"Measure {measure_index} (beat rows {line_numbers[0]}-{line_numbers[-1]}): "
                f"non-consecutive beat IDs {ids!r}"
            )
        if not pickup and ids[0] != 1:
            raise BeatGridError(f"Measure {measure_index}: full measure does not start at beat ID 1")

        denominators = [event.denominator for event in events]
        denominator = _mode_with_first_tiebreak(denominators)
        declared_numerators = [event.declared_numerator for event in events]
        declared_numerator = _mode_with_first_tiebreak(declared_numerators)
        numerator_conflict = any(value != beat_count for value in declared_numerators)
        denominator_conflict = any(value != denominator for value in denominators)
        pad_final_partial = (
            partial
            and len(set(declared_numerators)) == 1
            and not denominator_conflict
            and declared_numerator >= beat_count
        )
        inferred = pickup or partial or numerator_conflict or denominator_conflict
        unresolved_numerator_conflict = (
            numerator_conflict
            and not pad_final_partial
            and not pickup
        )
        if (
            unresolved_numerator_conflict or denominator_conflict
        ) and meter_conflict == "reject":
            raise BeatGridError(
                f"Measure {measure_index}: {beat_count} actual beats conflict with declarations "
                f"{list(zip(declared_numerators, denominators))!r}"
            )
        if pad_final_partial and declared_numerator > beat_count:
            diagnostics.append(
                f"measure {measure_index}: padded final {beat_count}/{denominator} span "
                f"to declared {declared_numerator}/{denominator} with trailing rest"
            )
        elif numerator_conflict:
            diagnostics.append(
                f"measure {measure_index}: inferred {beat_count}/{denominator} from downbeat span; "
                f"declared numerators were {declared_numerators}"
            )
        if denominator_conflict:
            diagnostics.append(
                f"measure {measure_index}: placed denominator {denominator} at the measure boundary; "
                f"row declarations were {denominators}"
            )
        measures.append(
            Measure(
                index=measure_index,
                start_beat=start,
                end_beat=end,
                numerator=beat_count,
                denominator=denominator,
                pickup=pickup,
                partial=partial,
                inferred=inferred,
                notated_numerator=(
                    declared_numerator if pad_final_partial else beat_count
                ),
            )
        )
    if len(measures) >= 2:
        first = measures[0]
        following = measures[1]
        first_duration = first.numerator / first.denominator
        following_duration = (
            following.abc_numerator / following.abc_denominator
        )
        if first_duration < following_duration:
            measures[0] = replace(
                first,
                inferred=True,
                notated_numerator=following.abc_numerator,
                notated_denominator=following.abc_denominator,
                pad_before=True,
            )
            diagnostics.append(
                f"measure 0: padded leading {first.numerator}/{first.denominator} span "
                f"to {following.abc_numerator}/{following.abc_denominator} "
                f"with preceding rest"
            )
    return measures, diagnostics


def _build_grid(beats: Sequence[BeatEvent], measures: Sequence[Measure]):
    interval_denominators = np.zeros(len(beats) - 1, dtype=np.int32)
    for measure in measures:
        interval_denominators[measure.start_beat:measure.end_beat] = measure.denominator
    if np.any(interval_denominators == 0):
        raise BeatGridError("Downbeat spans do not cover every beat interval")

    subbeat_times = []
    subbeat_denominators = []
    quarter_positions = [0.0]
    current_quarter = 0.0
    for index in range(len(beats) - 1):
        start = beats[index].time
        end = beats[index + 1].time
        denominator = int(interval_denominators[index])
        times = np.linspace(start, end, SUBBEAT_DIVISION + 1)[:-1]
        subbeat_times.extend(float(value) for value in times)
        subbeat_denominators.extend([denominator] * SUBBEAT_DIVISION)
        quarter_step = 4.0 / denominator / SUBBEAT_DIVISION
        for _ in range(SUBBEAT_DIVISION):
            current_quarter += quarter_step
            quarter_positions.append(current_quarter)
    subbeat_times.append(beats[-1].time)
    subbeat_denominators.append(int(interval_denominators[-1]))
    return (
        np.asarray(subbeat_times, dtype=np.float64),
        np.asarray(quarter_positions, dtype=np.float64),
        np.asarray(subbeat_denominators, dtype=np.int32),
    )


def _subbeat_boundaries(subbeat_times: np.ndarray) -> np.ndarray:
    return (subbeat_times[:-1] + subbeat_times[1:]) / 2


def _quantize_time(time: float, subbeat_times: np.ndarray) -> int:
    return int(np.searchsorted(_subbeat_boundaries(subbeat_times), float(time)))


def _fill_intervals(rows, subbeat_times, *, default, dtype):
    result = np.full(len(subbeat_times), default, dtype=dtype)
    for start, end, value in rows:
        start_t = _quantize_time(start, subbeat_times)
        end_t = _quantize_time(end, subbeat_times)
        start_t = max(0, min(start_t, len(result) - 1))
        end_t = max(0, min(end_t, len(result) - 1))
        if end_t <= start_t:
            raise AbcRebuildError(
                f"Interval {start:.6f}-{end:.6f} ({value}) is shorter than the ABC subbeat grid"
            )
        result[start_t:end_t] = value
    if len(result) > 1:
        result[-1] = result[-2]
    return result


def _structure_events(rows, subbeat_times):
    events = []
    for start, _, label in rows:
        t = _quantize_time(start, subbeat_times)
        t = max(0, min(t, len(subbeat_times) - 1))
        events.append((t, label))
    return events


def _classify_melody_tracks(midi: pretty_midi.PrettyMIDI):
    classified = {"Vocal": [], "Ins": []}
    unknown = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        name = (instrument.name or "").strip().lower()
        if "vocal" in name:
            classified["Vocal"].append(instrument)
        elif "ins" in name or "instrument" in name:
            classified["Ins"].append(instrument)
        elif instrument.notes:
            unknown.append(instrument)
    if unknown:
        if not classified["Vocal"] and not classified["Ins"] and len(unknown) == 1:
            classified["Ins"].extend(unknown)
        else:
            names = [instrument.name or "<unnamed>" for instrument in unknown]
            raise MelodyVoiceError(
                f"Cannot map non-empty melody track(s) {names!r} to fixed Vocal/Ins voices"
            )
    return classified


def _notes_to_arr(notes, subbeat_times, voice_id):
    result = np.zeros(len(subbeat_times), dtype=np.int32)
    boundaries = _subbeat_boundaries(subbeat_times)
    for note in sorted(notes, key=lambda item: (item.start, item.end, item.pitch)):
        start_t = int(np.searchsorted(boundaries, note.start))
        end_t = int(np.searchsorted(boundaries, note.end))
        start_t = max(0, min(start_t, len(result) - 1))
        end_t = max(0, min(end_t, len(result) - 1))
        if end_t <= start_t:
            raise MelodyVoiceError(
                f"{voice_id}: MIDI note pitch={note.pitch} at {note.start:.6f}-{note.end:.6f} "
                "cannot be represented on the decoded subbeat grid"
            )
        if np.any(result[start_t:end_t] != 0):
            raise MelodyVoiceError(
                f"{voice_id}: overlapping quantized melody notes at subbeats {start_t}:{end_t}"
            )
        sustain = note.pitch * 2 + 2
        result[start_t:end_t] = sustain
        result[start_t] = sustain + 1
    return result


def _pitch_class(root: str) -> tuple[int, str, str]:
    match = _ROOT_RE.fullmatch(root)
    if match is None:
        raise ChordSymbolError(f"Invalid pitch spelling {root!r}")
    letter = match.group("letter")
    accidental = match.group("accidental")
    offset = accidental.count("#") - accidental.count("b")
    return (_NATURAL_PITCH_CLASS[letter] + offset) % 12, letter, accidental


def portable_pitch_name(root: str, *, preserve_double: bool = False) -> str:
    pitch_class, _, accidental = _pitch_class(root)
    if preserve_double or len(accidental) <= 1:
        return root
    names = _SHARP_PITCH_NAMES if accidental.startswith("#") else _FLAT_PITCH_NAMES
    return names[pitch_class]


def _bass_degree_to_pitch(root: str, degree_text: str) -> str:
    if _ROOT_RE.fullmatch(degree_text):
        return portable_pitch_name(degree_text, preserve_double=True)
    match = _BASS_DEGREE_RE.fullmatch(degree_text)
    if match is None:
        raise ChordSymbolError(f"Invalid chord bass degree {degree_text!r}")
    root_pc, root_letter, root_accidental = _pitch_class(root)
    degree = int(match.group("degree"))
    degree_accidental = match.group("accidental")
    scale_semitones = (0, 2, 4, 5, 7, 9, 11)
    interval = scale_semitones[(degree - 1) % 7] + 12 * ((degree - 1) // 7)
    interval += degree_accidental.count("#") - degree_accidental.count("b")
    target_pc = (root_pc + interval) % 12

    target_letter_index = (_LETTERS.index(root_letter) + degree - 1) % 7
    target_letter = _LETTERS[target_letter_index]
    natural_pc = _NATURAL_PITCH_CLASS[target_letter]
    difference = (target_pc - natural_pc + 6) % 12 - 6
    if difference in {-2, -1, 0, 1, 2}:
        accidental = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[difference]
        return target_letter + accidental
    names = _SHARP_PITCH_NAMES if "#" in (root_accidental + degree_accidental) else _FLAT_PITCH_NAMES
    return names[target_pc]


def chord_symbol_to_abc(chord: str) -> str | None:
    chord = chord.strip()
    if chord in NO_CHORDS:
        return None
    if ":" not in chord:
        raise ChordSymbolError(f"Chord {chord!r} is missing the ':' quality separator")
    root, descriptor = chord.split(":", 1)
    if "/" in descriptor:
        quality, bass_degree = descriptor.split("/", 1)
    else:
        quality, bass_degree = descriptor, None
    if quality not in _QUALITY_TO_ABC:
        raise ChordSymbolError(
            f"Unsupported chord quality {quality!r} in {chord!r}; refusing to rewrite it as major"
        )
    chord_root = portable_pitch_name(root, preserve_double=True)
    text = chord_root + _QUALITY_TO_ABC[quality]
    if bass_degree:
        text += "/" + _bass_degree_to_pitch(root, bass_degree)
    return text


def key_symbol_to_abc(key: str) -> str:
    key = key.strip()
    if ":" in key:
        root, mode = key.split(":", 1)
        if mode not in {"major", "minor"}:
            raise AbcRebuildError(f"Unsupported key mode {mode!r} in {key!r}")
    elif key.endswith("m"):
        root, mode = key[:-1], "minor"
    else:
        root, mode = key, "major"
    root_pc, _, accidental = _pitch_class(root)
    candidate = portable_pitch_name(root) + ("m" if mode == "minor" else "")
    if candidate in _KEY_SIGNATURE_ACCIDENTALS:
        return candidate
    names = _FLAT_PITCH_NAMES if "b" in accidental else _SHARP_PITCH_NAMES
    candidate = names[root_pc] + ("m" if mode == "minor" else "")
    if candidate not in _KEY_SIGNATURE_ACCIDENTALS:
        fallback_names = _SHARP_PITCH_NAMES if names is _FLAT_PITCH_NAMES else _FLAT_PITCH_NAMES
        candidate = fallback_names[root_pc] + ("m" if mode == "minor" else "")
    if candidate not in _KEY_SIGNATURE_ACCIDENTALS:
        raise AbcRebuildError(f"Cannot encode portable ABC key for {key!r}")
    return candidate


def get_key_accidentals(key: str) -> list[int]:
    try:
        count = _KEY_SIGNATURE_ACCIDENTALS[key]
    except KeyError as exc:
        raise AbcRebuildError(f"Unsupported ABC key signature {key!r}") from exc
    accidentals = [0] * 7
    order = "FCGDAEB" if count > 0 else "BEADGCF"
    for letter in order[:abs(count)]:
        accidentals[_LETTERS.index(letter)] = 1 if count > 0 else -1
    return accidentals


def note_to_abc(note: int, key_accidentals: Sequence[int], measure_accidentals: dict) -> str:
    """Use key-relative spelling and write only bar-state changes.

    The two target parsers propagate an accidental to the same note letter in
    every octave until the next barline. ``measure_accidentals`` is therefore
    keyed by letter and reset by the caller for every bar (and after an inline
    key change). This preserves pitches across parsers while still omitting
    repeated accidental marks.  The key-relative spelling can use double
    accidentals in remote keys; MIDI G is F## in G# minor, for example.
    """

    accidental_count = sum(key_accidentals)
    try:
        pitch_name = _KEY_RELATIVE_PITCH_NAMES[accidental_count][note % 12]
    except KeyError as exc:
        raise AbcRebuildError(
            f"Unsupported key signature accidental count {accidental_count}"
        ) from exc
    letter = pitch_name[0]
    accidental = pitch_name[1:]
    accidental_number = {"": 0, "#": 1, "##": 2, "b": -1, "bb": -2}[accidental]
    octave = (note - 60) // 12
    # Cb and B# cross the MIDI octave boundary even though their written note
    # letter does not.
    if note % 12 == 11 and accidental_number == -1:
        octave += 1
    elif note % 12 == 0 and accidental_number == 1:
        octave -= 1
    scale_index = _LETTERS.index(letter)
    current_accidental = measure_accidentals.get(
        scale_index,
        key_accidentals[scale_index],
    )
    accidental_text = ""
    if current_accidental != accidental_number:
        measure_accidentals[scale_index] = accidental_number
        accidental_text = {-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"}[
            accidental_number
        ]

    if octave > 0:
        letter = letter.lower()
        if octave > 1:
            letter += "'" * (octave - 1)
    elif octave < 0:
        letter += "," * abs(octave)
    return accidental_text + letter


def build_rebuilt_abc_score(
    melody_midi_path,
    beats_path,
    chords_path,
    keys_path,
    structures_path,
    *,
    meter_conflict: str = "infer",
    melody_only: bool = False,
) -> RebuiltAbcScore:
    beats = read_beats(beats_path)
    keys = read_keys(keys_path)
    structures = read_structures(structures_path)
    chords = [] if melody_only else read_chords(chords_path)
    midi = pretty_midi.PrettyMIDI(str(melody_midi_path))
    return _assemble_abc_score(midi, beats, keys, structures, chords,
                               meter_conflict=meter_conflict, melody_only=melody_only)


def build_rebuilt_abc_score_from_data(
    melody_midi, beats, chords, keys, structures, *, meter_conflict="infer", melody_only=False,
) -> RebuiltAbcScore:
    """Build from MIDI bytes/BytesIO/PrettyMIDI and beat/interval rows.

    BeatEvent lists are also accepted. The file and memory interfaces share
    interval validation, score construction, serialization and ABC validation.
    """
    beats = list(beats)
    if beats and isinstance(beats[0], BeatEvent):
        beat_entries = [(b.line_no, [str(b.time), str(b.beat_id), str(b.declared_numerator), str(b.denominator)]) for b in beats]
    else:
        beat_entries = _row_entries(beats, "beats")
    beats = _parse_beats(beat_entries, "beats")
    keys = _parse_keys(_row_entries(keys, "keys"), "keys")
    structures = _parse_structures(_row_entries(structures, "structures"), "structures")
    chords = [] if melody_only else _parse_chords(_row_entries(chords, "chords"), "chords")
    if isinstance(melody_midi, (bytes, bytearray)):
        melody_midi = BytesIO(melody_midi)
    if not isinstance(melody_midi, (BytesIO, pretty_midi.PrettyMIDI)):
        raise TypeError("melody_midi must be MIDI bytes, BytesIO, or PrettyMIDI")
    midi = melody_midi if isinstance(melody_midi, pretty_midi.PrettyMIDI) else pretty_midi.PrettyMIDI(melody_midi)
    return _assemble_abc_score(midi, beats, keys, structures, chords,
                               meter_conflict=meter_conflict, melody_only=melody_only)


def _assemble_abc_score(midi, beats, keys, structures, chords, *, meter_conflict, melody_only):
    measures, diagnostics = infer_measures(beats, meter_conflict=meter_conflict)
    subbeat_times, subbeat_quarters, subbeat_denominators = _build_grid(beats, measures)
    classified = _classify_melody_tracks(midi)
    voice_arrs = {}
    for voice_id in VOICE_IDS:
        notes = [
            note
            for instrument in classified[voice_id]
            for note in instrument.notes
        ]
        voice_arrs[voice_id] = _notes_to_arr(notes, subbeat_times, voice_id)

    key_arr = _fill_intervals(keys, subbeat_times, default=keys[0][2], dtype="<U16")
    if melody_only:
        # Do not even read chord labels in melody-only mode.  A constant no-chord
        # timeline removes chord-only render boundaries, allowing held notes and
        # rests to be serialized as their original semantic segments.
        chord_arr = np.full(len(subbeat_times), "N", dtype="<U64")
    else:
        chord_arr = _fill_intervals(
            chords,
            subbeat_times,
            default="N",
            dtype="<U64",
        )
    return RebuiltAbcScore(
        beats=list(beats),
        measures=measures,
        subbeat_times=subbeat_times,
        subbeat_quarters=subbeat_quarters,
        subbeat_denominators=subbeat_denominators,
        key_arr=key_arr,
        chord_arr=chord_arr,
        structure_events=_structure_events(structures, subbeat_times),
        voice_arrs=voice_arrs,
        diagnostics=diagnostics,
    )


def abc_unit_denominator(score: RebuiltAbcScore) -> int:
    values = [
        denominator * score.subbeat_div
        for measure in score.measures
        for denominator in (measure.denominator, measure.abc_denominator)
    ]
    denominator = math.lcm(*values)
    if denominator > 1024:
        raise AbcRebuildError(f"Required ABC unit length 1/{denominator} is unreasonably small")
    return denominator


def _measure_actual_units(measure: Measure, unit_denominator: int) -> int:
    return measure.numerator * unit_denominator // measure.denominator


def _measure_abc_units(measure: Measure, unit_denominator: int) -> int:
    return measure.abc_numerator * unit_denominator // measure.abc_denominator


def _measure_padding_units(measure: Measure, unit_denominator: int) -> int:
    return (
        _measure_abc_units(measure, unit_denominator)
        - _measure_actual_units(measure, unit_denominator)
    )


def _duration_units(score: RebuiltAbcScore, start_t: int, end_t: int, unit_denominator: int) -> int:
    units = 0
    for denominator in score.subbeat_denominators[start_t:end_t]:
        divisor = int(denominator) * score.subbeat_div
        if unit_denominator % divisor:
            raise AbcRebuildError(
                f"ABC L:1/{unit_denominator} cannot express a 1/{divisor} subbeat exactly"
            )
        units += unit_denominator // divisor
    return units


def estimate_tempo(score: RebuiltAbcScore) -> float:
    seconds = score.subbeat_times[-1] - score.subbeat_times[0]
    quarter_notes = score.subbeat_quarters[-1] - score.subbeat_quarters[0]
    if seconds <= 0 or quarter_notes <= 0:
        raise AbcRebuildError("Cannot estimate tempo from a zero-duration score")
    return float(quarter_notes / seconds * 60.0)


def _continues_pitch(value: int, next_value: int) -> bool:
    if value <= 0:
        return False
    pitch = value // 2 - 1
    return next_value == pitch * 2 + 2


def _same_note_segment(value: int, next_value: int) -> bool:
    if value == 0:
        return next_value == 0
    pitch = value // 2 - 1
    return next_value == pitch * 2 + 2


_SUPPORTED_DURATION_UNITS = frozenset(
    {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48}
)


def _split_duration_units(duration: int) -> list[int]:
    """Split a duration into values accepted by strict music parsers."""

    if duration <= 0:
        raise AbcRebuildError(f"Cannot serialize non-positive duration {duration}")
    result = []
    remaining = int(duration)
    while remaining:
        if remaining in _SUPPORTED_DURATION_UNITS:
            result.append(remaining)
            break
        candidates = [
            value
            for value in _SUPPORTED_DURATION_UNITS
            if value < remaining
        ]
        if not candidates:
            raise AbcRebuildError(
                f"Duration {duration} cannot be split into representable ABC values"
            )
        chunk = max(candidates)
        result.append(chunk)
        remaining -= chunk
    return result


def _duration_text(duration: int) -> str:
    return "" if duration == 1 else str(duration)


def _render_duration_tokens(
    prefix: str,
    note_text: str,
    duration: int,
    *,
    tie_out: bool,
) -> list[str]:
    chunks = _split_duration_units(duration)
    tokens = []
    for index, chunk in enumerate(chunks):
        continues = note_text != "z" and (
            index + 1 < len(chunks) or tie_out
        )
        tokens.append(
            (prefix if index == 0 else "")
            + note_text
            + _duration_text(chunk)
            + ("-" if continues else "")
        )
    return tokens


def _render_voice_measure(
    score: RebuiltAbcScore,
    voice_id: str,
    measure: Measure,
    unit_denominator: int,
) -> str:
    voice = score.voice_arrs[voice_id]
    show_chords = voice_id == "Vocal"
    measure_accidentals = {}
    current_key = str(score.key_arr[measure.start_t])
    key_accidentals = get_key_accidentals(current_key)
    parts = []
    padding = _measure_padding_units(measure, unit_denominator)
    if padding < 0:
        raise AbcRebuildError(
            f"Measure {measure.index}: notated meter is shorter than its decoded span"
        )
    leading_padding = padding if measure.pad_before else 0
    trailing_padding = 0 if measure.pad_before else padding
    t = measure.start_t
    while t < measure.end_t:
        change_points = [measure.end_t]
        for probe in range(t + 1, measure.end_t):
            if not _same_note_segment(int(voice[t]), int(voice[probe])):
                change_points.append(probe)
                break
        for probe in range(t + 1, measure.end_t):
            if score.key_arr[probe] != score.key_arr[probe - 1]:
                change_points.append(probe)
                break
        if show_chords:
            for probe in range(t + 1, measure.end_t):
                if score.chord_arr[probe] != score.chord_arr[probe - 1]:
                    change_points.append(probe)
                    break
        next_t = min(change_points)

        prefix = ""
        key = str(score.key_arr[t])
        if t > measure.start_t and key != current_key:
            current_key = key
            key_accidentals = get_key_accidentals(current_key)
            measure_accidentals = {}
            prefix += f"[K:{current_key}]"

        if show_chords and (t == measure.start_t or score.chord_arr[t] != score.chord_arr[t - 1]):
            chord = str(score.chord_arr[t])
            chord_text = chord_symbol_to_abc(chord)
            if chord_text is not None:
                prefix += f'"{chord_text}"'

        value = int(voice[t])
        if value == 0:
            note_text = "z"
        else:
            note_text = note_to_abc(value // 2 - 1, key_accidentals, measure_accidentals)
        duration = _duration_units(score, t, next_t, unit_denominator)
        if t == measure.start_t and leading_padding:
            if value == 0 and not prefix:
                duration += leading_padding
            else:
                parts.extend(
                    _render_duration_tokens(
                        "",
                        "z",
                        leading_padding,
                        tie_out=False,
                    )
                )
            leading_padding = 0
        if value == 0 and next_t == measure.end_t and trailing_padding:
            duration += trailing_padding
            trailing_padding = 0
        if duration <= 0:
            raise AbcRebuildError(f"Non-positive ABC duration at subbeats {t}:{next_t}")
        tie_out = (
            value > 0
            and next_t < len(voice)
            and _continues_pitch(value, int(voice[next_t]))
        )
        parts.extend(
            _render_duration_tokens(
                prefix,
                note_text,
                duration,
                tie_out=tie_out,
            )
        )
        t = next_t
    if leading_padding:
        raise AbcRebuildError(
            f"Measure {measure.index}: leading rest padding was not serialized"
        )
    if trailing_padding:
        parts.extend(
            _render_duration_tokens(
                "",
                "z",
                trailing_padding,
                tie_out=False,
            )
        )
    return "".join(parts)


def _is_compressible_full_rest(rendered_measure: str) -> bool:
    """Whether a rendered measure can be losslessly replaced by ABC ``Z``."""

    cursor = 0
    saw_note = False
    for match in _MUSIC_ELEMENT_RE.finditer(rendered_measure):
        if rendered_measure[cursor:match.start()]:
            return False
        cursor = match.end()
        if match.group("quoted") is not None or match.group("key") is not None:
            return False
        saw_note = True
        if match.group("note") != "z" or match.group("tie"):
            return False
    return saw_note and cursor == len(rendered_measure)


def _render_voice_group(
    score: RebuiltAbcScore,
    voice_id: str,
    measures: list[Measure],
    unit_denominator: int,
) -> str:
    rendered = [
        _render_voice_measure(
            score,
            voice_id,
            measure,
            unit_denominator,
        )
        for measure in measures
    ]
    parts = []
    index = 0
    while index < len(rendered):
        if not _is_compressible_full_rest(rendered[index]):
            parts.append(rendered[index] + "|")
            index += 1
            continue
        end = index + 1
        while (
            end < len(rendered)
            and _is_compressible_full_rest(rendered[end])
        ):
            end += 1
        count = end - index
        parts.append("Z" + (str(count) if count > 1 else "") + "|")
        index = end
    return "".join(parts)


def _sanitize_structure_label(value: str) -> str:
    return " ".join(str(value).split())


def _measure_groups(score: RebuiltAbcScore) -> list[MeasureGroup]:
    first_measure = score.measures[0]
    active_meter = (
        first_measure.abc_numerator,
        first_measure.abc_denominator,
    )
    active_key = str(score.key_arr[first_measure.start_t])
    active_structure = ""
    groups: list[MeasureGroup] = []

    for measure in score.measures:
        meter = (measure.abc_numerator, measure.abc_denominator)
        key = str(score.key_arr[measure.start_t])
        meter_changed = meter != active_meter
        key_changed = key != active_key
        new_structure_labels = []
        for t, label in score.structure_events:
            if not measure.start_t <= t < measure.end_t:
                continue
            clean_label = _sanitize_structure_label(label)
            if clean_label and clean_label != active_structure:
                new_structure_labels.append(clean_label)
                active_structure = clean_label

        start_group = (
            not groups
            or len(groups[-1].measures) >= 4
            or meter_changed
            or key_changed
            or bool(new_structure_labels)
        )
        if start_group:
            groups.append(
                MeasureGroup(
                    measures=[measure],
                    structure_labels=new_structure_labels,
                    meter_changed=meter_changed,
                    key_changed=key_changed,
                )
            )
        else:
            groups[-1].measures.append(measure)

        active_meter = meter
        active_key = str(score.key_arr[measure.end_t - 1])
    return groups


def score_to_abc(score: RebuiltAbcScore) -> str:
    unit_denominator = abc_unit_denominator(score)
    first_measure = score.measures[0]
    first_key = str(score.key_arr[first_measure.start_t])
    lines = [
        "X:1",
        "T:",
        f"M:{first_measure.abc_numerator}/{first_measure.abc_denominator}",
        f"L:1/{unit_denominator}",
        f"Q:1/4={int(round(estimate_tempo(score)))}",
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
        'V: Ins clef=treble name="Ins Melody" snm="Inst."',
        f"K:{first_key}",
    ]
    for group in _measure_groups(score):
        lines.extend(f"% {label}" for label in group.structure_labels)
        first_group_measure = group.measures[0]
        for voice_id in VOICE_IDS:
            lines.append(f"V: {voice_id}")
            if group.meter_changed:
                lines.append(
                    f"M:{first_group_measure.abc_numerator}/"
                    f"{first_group_measure.abc_denominator}"
                )
            if group.key_changed:
                lines.append(
                    f"K:{score.key_arr[first_group_measure.start_t]}"
                )
            lines.append(
                _render_voice_group(
                    score,
                    voice_id,
                    group.measures,
                    unit_denominator,
                )
            )
    text = "\n".join(lines) + "\n"
    validate_serialized_abc(text, score)
    return text


_MUSIC_ELEMENT_RE = re.compile(
    r'"(?P<quoted>[^"]*)"'
    r"|\[K:(?P<key>[^\]]+)\]"
    r"|(?P<note>[_=^]*[A-Ga-gz][,']*)(?P<duration>\d*)(?P<tie>-?)"
)


def _parse_music_measure(line: str, expected: int, context: str):
    body = line
    if body == "Z":
        return [], []
    position = 0
    cursor = 0
    quoted_events = []
    key_events = []
    for match in _MUSIC_ELEMENT_RE.finditer(body):
        gap = body[cursor:match.start()]
        if gap.strip():
            raise AbcRebuildError(f"{context}: unsupported serialized ABC tokens {gap!r}")
        cursor = match.end()
        if match.group("quoted") is not None:
            quoted_events.append((position, match.group("quoted")))
            continue
        if match.group("key") is not None:
            key_events.append((position, match.group("key")))
            continue
        note = match.group("note")
        tie = match.group("tie")
        if tie and note == "z":
            raise AbcRebuildError(f"{context}: a rest cannot be tied")
        duration_text = match.group("duration")
        duration = int(duration_text) if duration_text else 1
        if duration not in _SUPPORTED_DURATION_UNITS:
            raise AbcRebuildError(
                f"{context}: duration {duration} is not parser-representable"
            )
        position += duration
    if body[cursor:].strip():
        raise AbcRebuildError(
            f"{context}: unsupported serialized ABC tokens {body[cursor:]!r}"
        )
    if position != expected:
        raise AbcRebuildError(
            f"{context}: duration {position} does not match meter duration {expected}"
        )
    if re.search(r"(^|[\s|])-[_=^A-Ga-g]", line):
        raise AbcRebuildError(f"{context}: tie is written before its second note")
    return quoted_events, key_events


def _expected_measure_chords(
    score: RebuiltAbcScore,
    measure: Measure,
    unit_denominator: int,
) -> list[tuple[int, str]]:
    expected = []
    leading_padding = (
        _measure_padding_units(measure, unit_denominator)
        if measure.pad_before
        else 0
    )
    for t in range(measure.start_t, measure.end_t):
        if t != measure.start_t and score.chord_arr[t] == score.chord_arr[t - 1]:
            continue
        position = leading_padding + _duration_units(
            score,
            measure.start_t,
            t,
            unit_denominator,
        )
        chord = str(score.chord_arr[t])
        text = chord_symbol_to_abc(chord)
        if text is not None:
            expected.append((position, text))
    return expected


def _expected_measure_keys(
    score: RebuiltAbcScore,
    measure: Measure,
    unit_denominator: int,
) -> list[tuple[int, str]]:
    leading_padding = (
        _measure_padding_units(measure, unit_denominator)
        if measure.pad_before
        else 0
    )
    return [
        (
            leading_padding
            + _duration_units(score, measure.start_t, t, unit_denominator),
            str(score.key_arr[t]),
        )
        for t in range(measure.start_t + 1, measure.end_t)
        if score.key_arr[t] != score.key_arr[t - 1]
    ]


def _parse_voice_group(lines, cursor, voice_id, group_index):
    expected_voice_field = f"V: {voice_id}"
    if cursor >= len(lines) or lines[cursor] != expected_voice_field:
        observed = lines[cursor] if cursor < len(lines) else "<end>"
        raise AbcRebuildError(
            f"Group {group_index}: expected {expected_voice_field}, got {observed!r}"
        )
    cursor += 1
    fields = {}
    while cursor < len(lines) and (
        lines[cursor].startswith("M:")
        or lines[cursor].startswith("K:")
    ):
        name, value = lines[cursor].split(":", 1)
        if name in fields:
            raise AbcRebuildError(
                f"Group {group_index} {voice_id}: repeated {name}: field"
            )
        fields[name] = value
        cursor += 1
    if cursor >= len(lines):
        raise AbcRebuildError(
            f"Group {group_index} {voice_id}: missing music line"
        )
    music_line = lines[cursor]
    if music_line.startswith(("V:", "M:", "K:", "%")):
        raise AbcRebuildError(
            f"Group {group_index} {voice_id}: invalid music line {music_line!r}"
        )
    cursor += 1
    split_bars = music_line.split("|")
    if not split_bars or split_bars[-1].strip():
        raise AbcRebuildError(
            f"Group {group_index} {voice_id}: music line must end with a barline"
        )
    serialized_bars = [bar.strip() for bar in split_bars[:-1]]
    if any(not bar for bar in serialized_bars):
        raise AbcRebuildError(
            f"Group {group_index} {voice_id}: empty serialized measure"
        )
    bars = []
    for bar in serialized_bars:
        match = re.fullmatch(r"Z(?P<count>[1-4])?", bar)
        if match is None:
            bars.append(bar)
            continue
        if match.group("count") == "1":
            raise AbcRebuildError(
                f"Group {group_index} {voice_id}: Z1 must be written as Z"
            )
        bars.extend(["Z"] * int(match.group("count") or "1"))
    if not 1 <= len(bars) <= 4:
        raise AbcRebuildError(
            f"Group {group_index} {voice_id}: expected 1-4 semantic measures"
        )
    return cursor, fields, bars


def validate_serialized_abc(text: str, score: RebuiltAbcScore) -> None:
    """Validate invariants that must hold before any ABC is written."""

    lines = text.splitlines()
    if not lines or lines[0] != "X:1":
        raise AbcRebuildError("ABC must start with X:1")
    if len(lines) < 2 or lines[1] != "T:":
        raise AbcRebuildError("ABC title must be fixed as empty T:")
    if any(line.startswith(("%abc-", "I:abc-creator")) for line in lines):
        raise AbcRebuildError("ABC must not contain version or creator metadata")
    if "%%MIDI gchordoff" in text:
        raise AbcRebuildError("ABC must not contain %%MIDI gchordoff")
    if "% ss2" in text:
        raise AbcRebuildError("ABC must not contain % ss2 metadata")
    header_voice_ids = [
        match.group(1)
        for line in lines
        if (match := re.match(r"^V: (Vocal|Ins) ", line))
    ]
    if header_voice_ids != list(VOICE_IDS):
        raise AbcRebuildError(f"Expected fixed Vocal/Ins voice definitions, got {header_voice_ids!r}")

    header_key_index = next(
        (
            index
            for index, line in enumerate(lines)
            if index > 0
            and line.startswith("K:")
            and any(
                header_index < index
                for header_index, header_line in enumerate(lines)
                if header_line.startswith("V: Ins ")
            )
        ),
        None,
    )
    if header_key_index is None:
        raise AbcRebuildError("ABC header K: field is missing")
    first_measure = score.measures[0]
    expected_header_meter = (
        f"M:{first_measure.abc_numerator}/{first_measure.abc_denominator}"
    )
    header_meters = [
        line
        for line in lines[:header_key_index + 1]
        if line.startswith("M:")
    ]
    if header_meters != [expected_header_meter]:
        raise AbcRebuildError(
            f"ABC header meters {header_meters!r} "
            f"!= {[expected_header_meter]!r}"
        )
    expected_header_key = f"K:{score.key_arr[first_measure.start_t]}"
    header_keys = [
        line
        for line in lines[:header_key_index + 1]
        if line.startswith("K:")
    ]
    if header_keys != [expected_header_key]:
        raise AbcRebuildError(
            f"ABC header keys {header_keys!r} "
            f"!= {[expected_header_key]!r}"
        )

    unit_denominator = abc_unit_denominator(score)
    expected_groups = _measure_groups(score)
    cursor = header_key_index + 1
    for group_index, group in enumerate(expected_groups):
        structure_labels = []
        while cursor < len(lines) and lines[cursor].startswith("% "):
            structure_labels.append(lines[cursor][2:].strip())
            cursor += 1
        if structure_labels != group.structure_labels:
            raise AbcRebuildError(
                f"Group {group_index}: structure labels "
                f"{structure_labels!r} != {group.structure_labels!r}"
            )

        cursor, vocal_fields, vocal_bars = _parse_voice_group(
            lines,
            cursor,
            "Vocal",
            group_index,
        )
        cursor, ins_fields, ins_bars = _parse_voice_group(
            lines,
            cursor,
            "Ins",
            group_index,
        )
        if vocal_fields != ins_fields:
            raise AbcRebuildError(
                f"Group {group_index}: meter/key changes must be scoped to both voices"
            )
        first_measure = group.measures[0]
        expected_fields = {}
        if group.meter_changed:
            expected_fields["M"] = (
                f"{first_measure.abc_numerator}/{first_measure.abc_denominator}"
            )
        if group.key_changed:
            expected_fields["K"] = str(
                score.key_arr[first_measure.start_t]
            )
        if vocal_fields != expected_fields:
            raise AbcRebuildError(
                f"Group {group_index}: fields {vocal_fields!r} "
                f"!= required changes {expected_fields!r}"
            )
        if (
            len(vocal_bars) != len(group.measures)
            or len(ins_bars) != len(group.measures)
        ):
            raise AbcRebuildError(
                f"Group {group_index}: both voices must contain "
                f"{len(group.measures)} measures"
            )

        for bar_index, measure in enumerate(group.measures):
            expected_duration = (
                measure.abc_numerator
                * unit_denominator
                // measure.abc_denominator
            )
            for voice_id, bars in (
                ("Vocal", vocal_bars),
                ("Ins", ins_bars),
            ):
                quoted_events, key_events = _parse_music_measure(
                    bars[bar_index],
                    expected_duration,
                    f"measure {measure.index} {voice_id}",
                )
                expected_keys = _expected_measure_keys(
                    score,
                    measure,
                    unit_denominator,
                )
                if key_events != expected_keys:
                    raise AbcRebuildError(
                        f"Measure {measure.index} {voice_id}: inline keys "
                        f"{key_events!r} != {expected_keys!r}"
                    )
                if voice_id == "Vocal":
                    expected_chords = _expected_measure_chords(
                        score,
                        measure,
                        unit_denominator,
                    )
                    if quoted_events != expected_chords:
                        raise AbcRebuildError(
                            f"Measure {measure.index}: chord symbols "
                            f"{quoted_events!r} != {expected_chords!r}"
                        )
                elif quoted_events:
                    raise AbcRebuildError(
                        f"Measure {measure.index}: chords must only be in Vocal"
                    )
    if cursor != len(lines):
        raise AbcRebuildError(
            f"Unexpected trailing ABC body lines: {lines[cursor:cursor + 5]!r}"
        )


def abc_paths_from_melody(
    melody_midi_path,
    output_path=None,
    *,
    melody_only=False,
):
    melody = Path(melody_midi_path)
    if melody.name.endswith("_raw_full_melody.mid"):
        raise AbcRebuildError(f"Raw melody MIDI is not a valid ABC input: {melody}")
    if not melody.name.endswith("_melody.mid"):
        raise AbcRebuildError(f"Expected an exact *_melody.mid input, got {melody}")
    stem = melody.name[: -len("_melody.mid")]
    prefix = melody.with_name(stem)
    default_suffix = "_melody_only.abc" if melody_only else "_full.abc"
    return {
        "melody_midi": melody,
        "beats": Path(str(prefix) + "_beats.txt"),
        "chords": Path(str(prefix) + "_chords.txt"),
        "keys": Path(str(prefix) + "_keys.txt"),
        "structures": Path(str(prefix) + "_structures.txt"),
        "output": (
            Path(output_path)
            if output_path is not None
            else Path(str(prefix) + default_suffix)
        ),
    }


def preflight_exports(
    melody_midi_path,
    output_path=None,
    *,
    melody_only=False,
):
    paths = abc_paths_from_melody(
        melody_midi_path,
        output_path=output_path,
        melody_only=melody_only,
    )
    required = {"melody_midi", "beats", "keys", "structures"}
    if not melody_only:
        required.add("chords")
    missing = [
        str(path)
        for name, path in paths.items()
        if name in required and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{paths['melody_midi']}: missing required companion file(s): {', '.join(missing)}"
        )
    return paths


def generate_abc_from_exports(
    melody_midi_path,
    *,
    output_path=None,
    meter_conflict="infer",
    melody_only=False,
):
    paths = preflight_exports(
        melody_midi_path,
        output_path=output_path,
        melody_only=melody_only,
    )
    score = build_rebuilt_abc_score(
        paths["melody_midi"],
        paths["beats"],
        paths["chords"],
        paths["keys"],
        paths["structures"],
        meter_conflict=meter_conflict,
        melody_only=melody_only,
    )
    return score_to_abc(score), score, paths


def generate_abc_from_data(melody_midi, beats, chords, keys, structures, *,
                           meter_conflict="infer", melody_only=False):
    """Return validated ABC text and its score without filesystem access."""
    score = build_rebuilt_abc_score_from_data(
        melody_midi, beats, chords, keys, structures,
        meter_conflict=meter_conflict, melody_only=melody_only,
    )
    return score_to_abc(score), score
