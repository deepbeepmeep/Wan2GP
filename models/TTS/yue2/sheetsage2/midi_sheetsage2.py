"""Original-time MIDI playback and a separate score-to-audio measure map."""
import json
from io import BytesIO
from pathlib import Path

import mir_eval.chord
import numpy as np
import pretty_midi

from .io_sheetsage2 import atomic_write_text


def chord_pitches(label):
    if label in {"N", "X", "?"}:
        return []
    root, bitmap, bass = mir_eval.chord.encode(label, reduce_extended_chords=True)
    if root < 0:
        return []
    # Keep inversion bass below the chord; no ABC chord-name reinterpretation.
    upper = [48 + root + int(interval) for interval in np.flatnonzero(bitmap > 0)]
    return sorted(set([36 + (root + bass) % 12] + upper))


def measure_map(score, duration):
    rows, position = [], 0.0
    for measure in score.measures:
        length = measure.abc_numerator / measure.abc_denominator
        actual = measure.numerator / measure.denominator
        start = float(score.beats[measure.start_beat].time)
        end = float(score.beats[measure.end_beat].time)
        rows.append(dict(index=measure.index, start=start, end=min(end, duration),
                         score_start=position, score_end=position + length,
                         leading_rest=length - actual if measure.pad_before else 0.0,
                         trailing_rest=length - actual if not measure.pad_before else 0.0))
        position += length
    return rows


def midi_bytes(midi):
    """Serialize a PrettyMIDI object without touching the filesystem."""
    stream = BytesIO()
    midi.write(stream)
    return stream.getvalue()


def build_playback(melody_midi, chord_rows, score, duration):
    """Return playback metadata and MIDI bytes, preserving MIDI tick rounding."""
    if isinstance(melody_midi, pretty_midi.PrettyMIDI):
        melody_midi = midi_bytes(melody_midi)
    midi = pretty_midi.PrettyMIDI(BytesIO(melody_midi))
    chord_track = pretty_midi.Instrument(0, name="Chords")
    warnings = []
    for start, end, label in chord_rows:
        start, end = max(0.0, float(start)), min(duration, float(end))
        try:
            pitches = chord_pitches(label)
        except mir_eval.chord.InvalidChordException as exc:
            warnings.append(f"Chord playback skipped {label}: {exc}")
            continue
        # Rearticulate long chord spans at downbeats, including repeated bars.
        downbeats = [b.time for b in score.beats if b.beat_id == 1] if score is not None else []
        cuts = [start] + [time for time in downbeats if start < time < end] + [end]
        for a, b in zip(cuts, cuts[1:]):
            if b <= a:
                continue
            for pitch in pitches:
                chord_track.notes.append(pretty_midi.Note(48, pitch, a, b))
    midi.instruments.append(chord_track)
    transcription = midi_bytes(midi)
    chords = pretty_midi.PrettyMIDI(resolution=midi.resolution)
    chords.instruments.append(chord_track)
    midis = {"transcription": transcription, "chords": midi_bytes(chords)}
    # Decode the serialized bytes, so playback follows the returned MIDI exactly.
    midi = pretty_midi.PrettyMIDI(BytesIO(transcription))
    tracks = [dict(name=instrument.name, program=int(instrument.program),
                   notes=[dict(pitch=int(n.pitch), start=float(n.start), end=float(n.end), velocity=int(n.velocity))
                          for n in instrument.notes]) for instrument in midi.instruments]
    data = dict(version=1, duration=duration, midi="transcription.mid", tracks=tracks,
                measures=measure_map(score, duration) if score is not None else [], warnings=warnings)
    return data, midis


def export_playback(directory, score, duration):
    """Write playback files for an existing directory of exported annotations."""
    directory = Path(directory)
    chord_path = directory / "chord.lab"
    rows = [line.split(maxsplit=2) for line in chord_path.read_text(encoding="utf-8").splitlines()] if chord_path.exists() else []
    data, midis = build_playback((directory / "melody.mid").read_bytes(), rows, score, duration)
    for name, content in midis.items():
        (directory / f"{name}.mid").write_bytes(content)
    atomic_write_text(directory / "playback.json", json.dumps(data, indent=2))
    return data
