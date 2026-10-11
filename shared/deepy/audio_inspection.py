# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Streaming CPU measurements and bounded previews of an audio stream."""

import math
import re
import subprocess
import tempfile
from pathlib import Path

import ffmpeg
import numpy as np

from shared.deepy.video_tools import _get_ffmpeg_path


def inspect_audio(path, *, audio_track_no=1, start_time=0.0, duration=None, silence_db=-50.0, min_silence=0.2, preview_path=None, preview_seconds=10.0):
    numbers = [start_time, silence_db, min_silence, preview_seconds, *([] if duration is None else [duration])]
    if not all(math.isfinite(value) for value in numbers) or start_time < 0 or duration is not None and duration <= 0:
        raise ValueError("Times must be finite; start_time must be non-negative and duration positive.")
    if not -120 <= silence_db <= 0 or min_silence <= 0 or not 0 < preview_seconds <= 30:
        raise ValueError("silence_db must be -120..0, min_silence positive and preview_seconds in (0, 30].")
    binary = _get_ffmpeg_path()
    probe = ffmpeg.probe(str(path))
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
    if not 1 <= audio_track_no <= len(streams):
        raise ValueError(f"audio_track_no must select an available 1-based audio stream (found {len(streams)}).")
    stream = streams[audio_track_no - 1]
    rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    container = probe.get("format", {})
    audio_start = max(0.0, float(stream.get("start_time", 0)) - float(container.get("start_time", 0)))
    audio_duration = float(stream["duration"]) if stream.get("duration") not in (None, "N/A") else None
    end_time = start_time + duration if duration is not None else None
    if audio_duration is not None:
        end_time = min(end_time, audio_start + audio_duration) if end_time is not None else audio_start + audio_duration
    start_time = max(start_time, audio_start)
    duration = end_time - start_time if end_time is not None else None
    if duration is not None and duration <= 0:
        raise ValueError("The selected audio range contains no samples.")
    command = [binary, "-nostdin", "-hide_banner", "-nostats", "-threads", "1", "-ss", str(start_time), "-i", str(path), "-map", f"0:a:{audio_track_no - 1}", "-vn"]
    if duration is not None:
        command += ["-t", str(duration)]
    window = max(1, round(rate * 0.02))
    peak, squares, clipped = np.zeros(channels), np.zeros(channels), np.zeros(channels, dtype=np.int64)
    frames, silence_start, silent_seconds, interval_count, intervals = 0, None, 0.0, 0, []

    def end_silence(end):
        nonlocal silence_start, silent_seconds, interval_count
        if silence_start is not None and end - silence_start >= min_silence:
            silent_seconds += end - silence_start
            interval_count += 1
            if len(intervals) < 200:
                intervals.append({"start": start_time + silence_start, "end": start_time + end})
        silence_start = None

    # ebur128 passes the original samples through; measurements below are not loudness-normalized.
    # Trim before metering: an output -t alone can let the filter inspect later frames.
    filters = ([f"atrim=end_sample={round(duration * rate)}"] if duration is not None else []) + ["ebur128=peak=true:framelog=verbose"]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command + ["-af", ",".join(filters), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], stdout=subprocess.PIPE, stderr=log)
        try:
            while data := process.stdout.read(window * channels * 4):
                samples = np.frombuffer(data, dtype="<f4").reshape(-1, channels).astype(np.float64)
                if not np.isfinite(samples).all():
                    raise ValueError("Decoded audio contains non-finite samples.")
                absolute = np.abs(samples)
                peak = np.maximum(peak, absolute.max(axis=0))
                squares += np.square(samples).sum(axis=0)
                clipped += (absolute >= 1.0).sum(axis=0)
                quiet = np.sqrt(np.square(samples).mean()) <= 10 ** (silence_db / 20)
                if quiet and silence_start is None:
                    silence_start = frames / rate
                elif not quiet:
                    end_silence(frames / rate)
                frames += len(samples)
            code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        log.seek(0)
        diagnostics = log.read().decode("utf-8", errors="replace")
        if code:
            raise ValueError(f"Audio decoding failed: {diagnostics[-1500:]}")
    if not frames:
        raise ValueError("The selected audio range contains no samples.")
    end_silence(frames / rate)

    def db(value):
        return 20 * math.log10(value) if value > 0 else None

    summary = diagnostics.rsplit("Summary:", 1)[-1]
    integrated = re.search(r"\bI:\s*([-+\w.]+)\s*LUFS", summary)
    true_peak = re.search(r"Peak:\s*([-+\w.]+)\s*dBFS", summary)

    def measured(match):
        value = float(match[1]) if match else None
        return value if value is not None and math.isfinite(value) else None

    result = {"status": "done", "audio_track_no": audio_track_no, "sample_rate": rate, "channels": channels,
              "channel_layout": stream.get("channel_layout"), "source_duration_seconds": float(container["duration"]) if container.get("duration") not in (None, "N/A") else None,
              "audio_start_time_seconds": audio_start, "audio_duration_seconds": audio_duration,
              "coverage": {"start_time": start_time, "end_time": start_time + frames / rate, "duration_seconds": frames / rate, "samples_per_channel": frames},
              "per_channel": [{"channel": index + 1, "peak_dbfs": db(peak[index]), "rms_dbfs": db(math.sqrt(squares[index] / frames)), "full_scale_samples": int(clipped[index])} for index in range(channels)],
              "loudness": {"method": "FFmpeg EBU R128", "integrated_lufs": measured(integrated), "true_peak_dbfs": measured(true_peak)},
              "silence": {"threshold_dbfs": silence_db, "window_seconds": window / rate, "minimum_seconds": min_silence,
                          "duration_seconds": silent_seconds, "interval_count": interval_count, "intervals": intervals, "truncated": interval_count > len(intervals)},
              "notes": ["Coverage follows the selected audio stream's playable timeline when its duration is known, excluding decoder tail padding. Times are relative to the source file.", "Silence means all-channel RMS below the threshold in 20 ms windows, not necessarily zero samples or missing audio.", "Full-scale samples (absolute amplitude >= 1) indicate possible clipping, not proof of audible distortion. Null dBFS means zero amplitude; null loudness means unavailable. Measurements do not judge semantic content or musical quality."]}
    if preview_path:
        seconds = min(preview_seconds, frames / rate)
        # A compact stereo preview is deliberately bounded; measurements above retain all source channels.
        preview = command.copy()
        if duration is not None:
            preview = preview[:-2]
        completed = subprocess.run(preview + ["-t", str(seconds), "-ac", "2", "-ar", "24000", "-c:a", "pcm_s16le", "-y", str(preview_path)], capture_output=True)
        if completed.returncode:
            Path(preview_path).unlink(missing_ok=True)
            raise ValueError(completed.stderr.decode(errors="replace")[-1500:])
        result.update(output_file=str(preview_path), preview={"start_time": start_time, "duration_seconds": seconds, "sample_rate": 24000, "channels": 2})
    return result
