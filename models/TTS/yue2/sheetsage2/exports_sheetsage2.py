"""Lossless decoded events/LAB/MIDI, then independently validated ABC."""
from pathlib import Path
import json
from fractions import Fraction

import numpy as np
import pretty_midi

from .notation_sheetsage2 import generate_abc_from_data
from .io_sheetsage2 import atomic_write_text
from .generation_sheetsage2 import field_text
from .midi_sheetsage2 import build_playback, midi_bytes


def rows_text(rows):
    return "".join("\t".join(str(x) for x in row) + "\n" for row in rows)


def write_rows(path, rows):
    atomic_write_text(path, rows_text(rows))


def interval_rows(events, field, duration):
    rows = [[e["time"], 0, e["values"][field]] for e in events if field in e["values"]]
    for i, row in enumerate(rows):
        row[1] = rows[i + 1][0] if i + 1 < len(rows) else duration
    return [r for r in rows if r[1] > r[0]]


def rhythm_rows(events):
    rows, meter = [], None
    for event in events:
        rhythm = event["values"].get("rhythm", {})
        meter = rhythm.get("meter", meter)
        eighth = rhythm.get("eighth_position")
        if eighth is not None and meter is not None:
            # The model stores eighth-note position, not denominator-beat index.
            position = Fraction(int(eighth) * int(meter[1]), 8)
            if position.denominator != 1:
                raise ValueError(f"Eighth position {eighth} is off the {meter[0]}/{meter[1]} beat grid")
            if not 0 <= position < meter[0]:
                raise ValueError(f"Eighth position {eighth} is outside meter {meter}")
            rows.append([float(event["time"]), int(position) + 1, int(meter[0]), int(meter[1])])
    return rows


def _midi(notes, paper=False):
    midi = pretty_midi.PrettyMIDI(resolution=220 if paper else 960)
    names = ((0, "vocal_melody"), (1, "instrumental_melody")) if paper else ((0, "Vocal"), (1, "Ins"))
    for track, name in names:
        instrument = pretty_midi.Instrument(program=0, name=name)
        for start, end, pitch, source in notes:
            if track == source and end > start:
                instrument.notes.append(pretty_midi.Note(100, pitch, start, end))
        midi.instruments.append(instrument)
    return midi


def notation_notes(notes):
    """A monophonic notation view, retaining the exact raw prediction separately."""
    result, diagnostics = [], []
    for track in (0, 1):
        ordered = sorted((list(n) for n in notes if n[3] == track), key=lambda n: (n[0], n[2], n[1]))
        for i, note in enumerate(ordered):
            if i + 1 < len(ordered) and note[1] > ordered[i + 1][0] + 1e-6:
                note[1] = ordered[i + 1][0]
                diagnostics.append(f"notation only: clipped track {track} note at {note[0]:.3f} to next onset")
            if note[1] > note[0] + 1e-6:
                result.append(note)
    return sorted(result), diagnostics


def export_result(decoded, tokenizer, output_dir=None, duration=None, paper=False, *, melody_only=False):
    """Build ABC, MIDI and LAB in memory, optionally writing the same bytes.

    Statistics retain their file-interface names. ``payload`` contains the
    in-memory ABC text, MIDI bytes, decoded event list, LAB texts and playback.
    melody_only removes chords from notation and playback without changing
    decoded events or raw LAB annotations.
    """
    if not isinstance(melody_only, bool):
        raise ValueError("melody_only must be True or False")
    if duration is None:
        raise TypeError("duration is required")
    texts, midis, notation_midis, labs = {}, {}, {}, {}

    def keep_rows(name, rows):
        text = rows_text(rows)
        texts[name] = text
        if name.endswith(".lab"):
            labs[name[:-4]] = text

    events = decoded["events"]
    notes = []
    for event in events:
        start = float(event["time"])
        for note in event["values"].get("melody", ()):
            end = max(start + 0.04, float(note["end_time"])) if paper else min(duration, float(note["end_time"]))
            if end > start:
                notes.append([start, end, int(note["pitch"]), int(note["track"])])
    if not paper:
        notes.sort()
    texts["events.json"] = json.dumps(decoded, indent=2)
    keep_rows("events.tsv", [["time", "global_subbeat", "fields"]] +
               [[e["time"], e["global_subbeat"], field_text(e, tokenizer)] for e in events])
    midis["melody"] = midi_bytes(_midi(notes, paper=paper))
    lab_notes = [[f"{a:.6f}", f"{b:.6f}", p, t] for a, b, p, t in notes] if paper else notes
    keep_rows("melody_full.lab", lab_notes)
    for track, name in ((0, "vocal"), (1, "instrumental")):
        selected = [n for n in notes if n[3] == track]
        keep_rows(f"melody_{name}.lab", [n[:3] for n in lab_notes if n[3] == track])
        midis[f"melody_{name}"] = midi_bytes(_midi(selected, paper=paper))
    intervals = {}
    for field in ("chord", "key", "structure"):
        rows = interval_rows(events, field, duration)
        if paper:
            rows = [[float(e["time"]), None, e["values"][field]] for e in events if field in e["values"]]
            for i, row in enumerate(rows):
                row[1] = max(row[0], rows[i + 1][0]) if i + 1 < len(rows) else float(duration)
        intervals[field] = rows
        keep_rows(f"{field}.lab", rows)
    if paper:
        rows = []
        for event in events:
            rhythm, stamp = event["values"].get("rhythm"), event["values"].get("timestamp")
            if rhythm is None and stamp is None:
                continue
            meter = rhythm.get("meter") if isinstance(rhythm, dict) else None
            eighth = rhythm.get("eighth_position") if isinstance(rhythm, dict) else None
            rows.append([f"{event['time']:.6f}", "" if stamp is None else f"{float(stamp):.6f}",
                         "" if eighth is None else int(eighth), "" if meter is None else f"{meter[0]}/{meter[1]}"])
        keep_rows("beat_meter.lab", rows)
    raw_rhythm = [[e["time"], json.dumps(e["values"].get("rhythm", {}))]
                  for e in events if "rhythm" in e["values"] or "timestamp" in e["values"]]
    keep_rows("rhythm_events.lab", raw_rhythm)
    diagnostics, abc_error, score, text = [], None, None, None
    try:
        beats = rhythm_rows(events)
        keep_rows("beat.lab", beats)
        keep_rows("downbeat.lab", [[r[0]] for r in beats if r[1] == 1])
        if len(beats) < 2:
            raise ValueError("At least two decoded beats are required for ABC")
        # canonical notation's last beat is a boundary, so append it explicitly. Continue the
        # final tempo only as far as the audio/last note and let it pad the bar.
        abc_beats = [list(b) for b in beats]
        period = float(np.median(np.diff([b[0] for b in beats[-9:]])))
        if period <= 0:
            raise ValueError("Decoded beats must increase in time")
        end = max(duration, max((n[1] for n in notes), default=0))
        while abc_beats[-1][0] < end - 1e-6:
            prev = abc_beats[-1]
            abc_beats.append([prev[0] + period, prev[1] % prev[2] + 1, prev[2], prev[3]])
        keep_rows("notation/song_beats.txt", abc_beats)
        notation_intervals = {}
        for field, suffix in (("chord", "chords"), ("key", "keys"), ("structure", "structures")):
            rows = interval_rows(events, field, duration)
            # Clip to the actual beat domain: events beyond it have no ABC bin.
            rows = [[max(abc_beats[0][0], a), min(abc_beats[-1][0], b), v]
                    for a, b, v in rows if b > abc_beats[0][0] and a < abc_beats[-1][0]]
            notation_intervals[field] = rows
            keep_rows(f"notation/song_{suffix}.txt", rows)
        if not interval_rows(events, "key", duration):
            raise ValueError("No key was decoded; cannot construct a keyed ABC score")
        clean, adjustments = notation_notes(notes)
        diagnostics.extend(adjustments)
        notation_midis["notation/song_melody.mid"] = midi_bytes(_midi(clean))
        text, score = generate_abc_from_data(
            notation_midis["notation/song_melody.mid"], abc_beats,
            notation_intervals["chord"], notation_intervals["key"], notation_intervals["structure"],
            melody_only=melody_only,
        )
        diagnostics.extend(score.diagnostics)
        texts["score.abc"] = text
        abc_measures = len(score.measures)
    except (ValueError, FileNotFoundError) as exc:
        abc_error = str(exc)
        abc_measures = 0
    playback, playback_midis = build_playback(
        midis["melody"], [] if melody_only else intervals["chord"], score, duration)
    midis.update(playback_midis)
    texts["playback.json"] = json.dumps(playback, indent=2)
    diagnostics.extend(playback["warnings"])
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in texts.items():
            atomic_write_text(output_dir / name, content)
        for name, content in {**{f"{name}.mid": content for name, content in midis.items()}, **notation_midis}.items():
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if abc_error is not None:
            (output_dir / "score.abc").unlink(missing_ok=True)
        else:
            (output_dir / "score.browser.abc").unlink(missing_ok=True)
    payload = dict(abc=text, midi=midis["transcription"], midis=midis,
                   events=events, labs=labs, playback=playback)
    return dict(melody_notes=len(notes), vocal_notes=sum(n[3] == 0 for n in notes),
                instrumental_notes=sum(n[3] == 1 for n in notes), events=len(events),
                abc_measures=abc_measures, abc_error=abc_error, diagnostics=diagnostics, payload=payload)
