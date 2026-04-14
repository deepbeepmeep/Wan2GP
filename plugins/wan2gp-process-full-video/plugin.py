from __future__ import annotations

import html
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from shared.api import extract_status_phase_label
from shared.utils.audio_video import extract_audio_tracks, get_video_encode_args
from shared.utils.plugins import WAN2GPPlugin
from shared.utils.utils import get_video_info_details
from shared.utils.video_decode import decode_video_frames_ffmpeg, resolve_media_binary
from shared.utils.virtual_media import build_virtual_media_path

PlugIn_Name = "Process Full Video"
PlugIn_Id = "ProcessFullVideo"

PROCESS_CHOICES = [("Outpaint Video", "outpaint_video")]
RATIO_CHOICES = [("1:1", "1:1"), ("4:3", "4:3"), ("3:4", "3:4"), ("16:9", "16:9"), ("9:16", "9:16"), ("21:9", "21:9"), ("9:21", "9:21")]
DEFAULT_SOURCE_PATH = ""
DEFAULT_OUTPUT_PATH = ""
DEFAULT_MODEL_TYPE = "ltx2_22B_distilled"
MAX_STATUS_REFRESH_HZ = 3.0
STATUS_REFRESH_INTERVAL_SECONDS = 1.0 / MAX_STATUS_REFRESH_HZ


@dataclass(frozen=True)
class ChunkPlan:
    control_start_frame: int
    requested_frames: int
    drop_first_frame: bool

    @property
    def control_end_frame(self) -> int:
        return self.control_start_frame + self.requested_frames - 1


@dataclass(frozen=True)
class FramePlanRules:
    frame_step: int
    minimum_requested_frames: int


MODEL_FRAME_RULES = {
    DEFAULT_MODEL_TYPE: FramePlanRules(frame_step=8, minimum_requested_frames=17),
}


def _get_frame_plan_rules(model_type: str) -> FramePlanRules:
    try:
        return MODEL_FRAME_RULES[str(model_type)]
    except KeyError as exc:
        raise gr.Error(f"Unsupported frame-planning model type: {model_type}") from exc


def _align_requested_frames(frame_count: int, *, frame_step: int, round_up: bool) -> int:
    if frame_count <= 1:
        return 1
    frame_step = max(1, int(frame_step))
    if round_up:
        return int(math.ceil((frame_count - 1) / float(frame_step)) * frame_step + 1)
    return int(math.floor((frame_count - 1) / float(frame_step)) * frame_step + 1)


def _normalize_chunk_frames(chunk_seconds: float, fps_float: float, *, frame_step: int, minimum_requested_frames: int) -> int:
    minimum_requested_frames = max(1, int(minimum_requested_frames))
    target_frames = max(minimum_requested_frames, int(round(max(float(chunk_seconds), 0.1) * max(float(fps_float), 1.0))))
    below = max(minimum_requested_frames, _align_requested_frames(target_frames, frame_step=frame_step, round_up=False))
    above = max(minimum_requested_frames, _align_requested_frames(target_frames, frame_step=frame_step, round_up=True))
    return below if abs(below - target_frames) <= abs(above - target_frames) else above


def _align_total_unique_frames(total_unique_frames: int, *, frame_step: int, minimum_requested_frames: int, initial_drop_first: bool) -> int:
    total_unique_frames = max(0, int(total_unique_frames))
    if initial_drop_first:
        minimum_unique_frames = max(1, int(minimum_requested_frames) - 1)
        return 0 if total_unique_frames < minimum_unique_frames else total_unique_frames - (total_unique_frames % max(1, int(frame_step)))
    return 0 if total_unique_frames < max(1, int(minimum_requested_frames)) else ((total_unique_frames - 1) // max(1, int(frame_step))) * max(1, int(frame_step)) + 1


def _count_planned_unique_frames(plans: list[ChunkPlan]) -> int:
    return sum(max(0, int(plan.requested_frames) - (1 if plan.drop_first_frame else 0)) for plan in plans)


def _choose_resolution(budget_label: str) -> str:
    resolutions = {"540p": "720x540", "720p": "1280x720", "900p": "1200x900", "1080p": "1920x1088"}
    try:
        return resolutions[str(budget_label)]
    except KeyError as exc:
        raise gr.Error(f"Unsupported Output Resolution: {budget_label}") from exc


def _format_time_token(seconds: float | None) -> str:
    if seconds in (None, ""):
        return "end"
    total_milliseconds = max(0, int(round(float(seconds) * 1000.0)))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    minutes, seconds_only = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        token = f"{hours:02d}h{minutes:02d}m{seconds_only:02d}s"
    else:
        token = f"{minutes:02d}m{seconds_only:02d}s"
    if milliseconds > 0:
        token += f"{milliseconds:03d}ms"
    return token


def _parse_time_input(value, *, label: str, allow_empty: bool) -> float | None:
    if value is None:
        return None if allow_empty else 0.0
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None if allow_empty else 0.0
        return max(0.0, float(value))
    text = str(value).strip()
    if len(text) == 0:
        return None if allow_empty else 0.0
    if ":" not in text:
        try:
            return max(0.0, float(text))
        except ValueError as exc:
            raise gr.Error(f"{label} must be a number of seconds, MM:SS, or HH:MM:SS.") from exc
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise gr.Error(f"{label} must be a number of seconds, MM:SS, or HH:MM:SS.")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return max(0.0, minutes * 60.0 + seconds)
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return max(0.0, hours * 3600.0 + minutes * 60.0 + seconds)
    except ValueError as exc:
        raise gr.Error(f"{label} must be a number of seconds, MM:SS, or HH:MM:SS.") from exc


def _build_auto_output_path(source_path: str, ratio_text: str, output_resolution: str, start_seconds: float | None, end_seconds: float | None, output_dir: str | None = None) -> str:
    source = Path(source_path)
    ratio_suffix = str(ratio_text or "").replace(":", "x") or "ratio"
    resolution_suffix = str(output_resolution or "").strip() or "res"
    start_suffix = _format_time_token(start_seconds)
    end_suffix = _format_time_token(end_seconds)
    target_dir = source.parent if not output_dir else Path(output_dir)
    return str(target_dir / f"{source.stem}_outpaint_{ratio_suffix}_{resolution_suffix}_{start_suffix}_{end_suffix}{source.suffix}")


def _make_output_variant(output: Path) -> str:
    for index in range(2, 10000):
        candidate = output.with_name(f"{output.stem}_{index}{output.suffix}")
        if not candidate.exists():
            _plugin_info(f"Output file already exists. Using {candidate}")
            return str(candidate)
    raise gr.Error(f"Unable to find a free output filename for {output}")


def _make_continuation_output_path(output_path: str) -> str:
    output = Path(output_path)
    candidate = output.with_name(f"{output.stem}_continue{output.suffix}")
    if not candidate.exists():
        return str(candidate)
    for index in range(2, 10000):
        variant = output.with_name(f"{output.stem}_continue_{index}{output.suffix}")
        if not variant.exists():
            return str(variant)
    raise gr.Error(f"Unable to find a free continuation filename for {output}")


def _resolve_output_path(source_path: str, output_path: str, ratio_text: str, output_resolution: str, start_seconds: float | None, end_seconds: float | None, continue_enabled: bool) -> tuple[str, bool]:
    output_text = str(output_path or "").strip()
    if len(output_text) == 0:
        output = Path(_build_auto_output_path(source_path, ratio_text, output_resolution, start_seconds, end_seconds))
    elif output_text.endswith(("\\", "/")) or Path(output_text).is_dir():
        output = Path(_build_auto_output_path(source_path, ratio_text, output_resolution, start_seconds, end_seconds, output_dir=output_text))
    else:
        output = Path(output_text)
    source_suffix = Path(source_path).suffix
    if not output.suffix:
        output = output.with_suffix(source_suffix)
    if continue_enabled:
        return str(output), output.exists()
    if output.exists():
        return _make_output_variant(output), False
    return str(output), False


def _frame_to_image(frame_tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(frame_tensor.permute(1, 2, 0).cpu().numpy())


def _extract_exact_frame_image(video_path: str, frame_no: int) -> Image.Image:
    frames = decode_video_frames_ffmpeg(video_path, int(frame_no), 1, target_fps=None, bridge="torch")
    if frames.shape[0] <= 0:
        raise gr.Error(f"Unable to decode frame {frame_no} from {video_path}")
    return Image.fromarray(frames[0].cpu().numpy())


def _resolve_resume_last_frame(video_path: str, reported_frame_count: int) -> tuple[int, Image.Image | None, str]:
    candidate_count = max(0, int(reported_frame_count))
    if candidate_count <= 0:
        return 0, None, "existing output contains no decodable frame"
    for window in (1, 8, 32, 128, 512, 2048, 8192):
        window = min(candidate_count, window)
        start_frame = max(0, candidate_count - window)
        frames = decode_video_frames_ffmpeg(video_path, start_frame, window, target_fps=None, bridge="torch")
        if frames.shape[0] <= 0:
            continue
        actual_frame_count = start_frame + int(frames.shape[0])
        message = "" if actual_frame_count == candidate_count else f"Adjusted continuation point to {actual_frame_count} decodable frame(s) from the existing output."
        return actual_frame_count, Image.fromarray(frames[-1].cpu().numpy()), message
    return 0, None, f"Unable to decode a valid tail frame from {video_path}"


def _probe_existing_output_resolution(output_path: str) -> tuple[str, int, int]:
    metadata = get_video_info_details(output_path)
    width = int(metadata.get("display_width") or metadata.get("width") or 0)
    height = int(metadata.get("display_height") or metadata.get("height") or 0)
    if width <= 0 or height <= 0:
        raise gr.Error(f"Unable to read the resolution of existing output: {output_path}")
    return f"{width}x{height}", width, height


def _get_video_tensor_resolution(video_tensor_uint8: torch.Tensor) -> tuple[int, int]:
    if not torch.is_tensor(video_tensor_uint8) or video_tensor_uint8.ndim != 4:
        raise gr.Error("WanGP API returned an invalid video tensor.")
    return int(video_tensor_uint8.shape[3]), int(video_tensor_uint8.shape[2])


def _load_video_tensor_from_file(video_path: str) -> torch.Tensor:
    metadata = get_video_info_details(video_path)
    frame_count = int(metadata.get("frame_count") or 0)
    if frame_count <= 0:
        raise gr.Error(f"Unable to read the frame count of generated chunk: {video_path}")
    frames = decode_video_frames_ffmpeg(video_path, 0, frame_count, target_fps=None, bridge="torch")
    if frames.shape[0] <= 0:
        raise gr.Error(f"Unable to decode generated chunk: {video_path}")
    return frames.permute(3, 0, 1, 2).contiguous()


def _write_video_chunk(process, video_tensor_uint8: torch.Tensor, *, start_frame: int, frame_count: int) -> torch.Tensor:
    if frame_count <= 0:
        raise RuntimeError("No frames available to write.")
    end_frame = start_frame + frame_count
    batch_frames = 8
    for batch_start in range(start_frame, end_frame, batch_frames):
        batch_end = min(batch_start + batch_frames, end_frame)
        batch = video_tensor_uint8[:, batch_start:batch_end].permute(1, 2, 3, 0).contiguous()
        try:
            process.stdin.write(batch.numpy().tobytes())
            process.stdin.flush()
        except BrokenPipeError as exc:
            stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None and process.poll() is not None else ""
            raise RuntimeError(stderr or "ffmpeg stopped receiving video frames while streaming a chunk") from exc
        if process.poll() not in (None, 0):
            stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None else ""
            raise RuntimeError(stderr or "ffmpeg exited while streaming a chunk")
    return video_tensor_uint8[:, start_frame + frame_count - 1]


def _compute_selected_frame_range(metadata: dict, start_seconds: float | None, end_seconds: float | None) -> tuple[int, int, float, int]:
    fps_float = float(metadata.get("fps_float") or metadata.get("fps") or 0.0)
    total_frames = int(metadata.get("frame_count") or 0)
    if fps_float <= 0 or total_frames <= 0:
        raise gr.Error("Unable to read the source video FPS or frame count.")
    start_frame = max(0, min(total_frames - 1, int(round(float(start_seconds or 0.0) * fps_float))))
    end_frame_exclusive = total_frames if end_seconds in (None, "") else min(total_frames, max(start_frame + 1, int(round(float(end_seconds) * fps_float))))
    if end_frame_exclusive <= start_frame:
        raise gr.Error("End must be greater than Start.")
    return start_frame, end_frame_exclusive, fps_float, total_frames


def _build_chunk_plan(start_frame: int, end_frame_exclusive: int, total_source_frames: int, chunk_frames: int, *, frame_step: int, minimum_requested_frames: int, initial_drop_first: bool = False) -> list[ChunkPlan]:
    plans: list[ChunkPlan] = []
    cursor = start_frame
    total_unique_frames = _align_total_unique_frames(end_frame_exclusive - start_frame, frame_step=frame_step, minimum_requested_frames=minimum_requested_frames, initial_drop_first=initial_drop_first)
    if total_unique_frames <= 0:
        raise gr.Error("The selected range ends too close to the source video end to build a valid chunk for the current model.")
    written_unique_frames = 0
    while written_unique_frames < total_unique_frames:
        drop_first_frame = initial_drop_first if len(plans) == 0 else True
        remaining_unique = total_unique_frames - written_unique_frames
        max_unique_frames = chunk_frames - (1 if drop_first_frame else 0)
        requested_frames = chunk_frames if remaining_unique > max_unique_frames else remaining_unique + (1 if drop_first_frame else 0)
        control_start_frame = cursor - 1 if drop_first_frame else cursor
        max_available_frames = total_source_frames - control_start_frame
        if max_available_frames < requested_frames:
            raise gr.Error("The selected range ends too close to the source video end to build a valid chunk for the current model.")
        if requested_frames < max(1, int(minimum_requested_frames)):
            raise gr.Error("The selected range ends too close to the source video end to build a valid chunk for the current model.")
        plans.append(ChunkPlan(control_start_frame=control_start_frame, requested_frames=requested_frames, drop_first_frame=drop_first_frame))
        unique_frames = requested_frames - (1 if drop_first_frame else 0)
        written_unique_frames += unique_frames
        cursor += unique_frames
    return plans


def _count_completed_chunks(plans: list[ChunkPlan], completed_unique_frames: int) -> tuple[int, int]:
    completed_chunks = 0
    consumed_frames = 0
    target_frames = max(0, int(completed_unique_frames))
    for plan in plans:
        unique_frames = plan.requested_frames - (1 if plan.drop_first_frame else 0)
        if consumed_frames + unique_frames <= target_frames:
            consumed_frames += unique_frames
            completed_chunks += 1
            continue
        break
    return completed_chunks, consumed_frames


def _probe_resume_frame_count(ffprobe_path: str, output_path: str, fps_float: float) -> tuple[int, str]:
    probe = subprocess.run([ffprobe_path, "-v", "error", "-show_entries", "format_tags=comment:format_tags=COMMENT:format_tags=description:format_tags=DESCRIPTION", "-of", "json", output_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if probe.returncode == 0:
        try:
            tags = (json.loads(probe.stdout).get("format") or {}).get("tags") or {}
            for key in ("comment", "COMMENT", "description", "DESCRIPTION"):
                payload = tags.get(key)
                if not payload:
                    continue
                metadata = json.loads(payload)
                frame_count = int(metadata.get("written_unique_frames") or 0)
                if frame_count > 0:
                    return frame_count, ""
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    metadata = get_video_info_details(output_path)
    frame_count = int(metadata.get("frame_count") or 0)
    if frame_count > 0:
        return frame_count, ""
    probe = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "json", output_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if probe.returncode == 0:
        try:
            frame_count = int((((json.loads(probe.stdout).get("streams") or [{}])[0]).get("nb_read_packets")) or 0)
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            frame_count = 0
        if frame_count > 0:
            return frame_count, ""
    probe = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", output_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if probe.returncode == 0:
        try:
            frame_count = int((((json.loads(probe.stdout).get("streams") or [{}])[0]).get("nb_read_frames")) or 0)
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            frame_count = 0
        if frame_count > 0:
            return frame_count, ""
    duration = float(metadata.get("duration") or 0.0)
    if duration > 0 and fps_float > 0:
        return int(round(duration * fps_float)), ""
    stderr = (probe.stderr or "").strip()
    return 0, stderr or "existing output contains no readable frame count or duration metadata"


def _normalize_container_name(video_container: str | None) -> str:
    return str(video_container or "mp4").strip().lower() or "mp4"


def _get_live_mux_output_args(video_container: str | None) -> list[str]:
    video_container = _normalize_container_name(video_container)
    if video_container == "mkv":
        return ["-fflags", "+flush_packets", "-flush_packets", "1", "-f", "matroska", "-live", "1"]
    if video_container == "mp4":
        return ["-movflags", "+frag_keyframe+empty_moov+default_base_moof"]
    return []


def _probe_media_duration(ffprobe_path: str, media_path: str) -> float:
    result = subprocess.run([ffprobe_path, "-v", "error", "-show_entries", "format=duration", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if result.returncode != 0:
        return 0.0
    try:
        return max(0.0, float((((json.loads(result.stdout) or {}).get("format") or {}).get("duration")) or 0.0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def _probe_audio_end_time(ffprobe_path: str, media_path: str, audio_index: int) -> float:
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", f"a:{max(0, int(audio_index))}", "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if result.returncode != 0:
        return 0.0
    last_end = 0.0
    for raw_line in result.stdout.splitlines():
        fields = [field.strip() for field in str(raw_line or "").strip().split(",")]
        if len(fields) <= 0 or len(fields[0]) == 0:
            continue
        try:
            pts_time = float(fields[0])
        except (TypeError, ValueError):
            continue
        try:
            duration_time = float(fields[1]) if len(fields) > 1 and len(fields[1]) > 0 else 0.0
        except (TypeError, ValueError):
            duration_time = 0.0
        last_end = max(last_end, pts_time, pts_time + max(0.0, duration_time))
    return max(0.0, last_end)


def _probe_selected_audio_end_time(ffprobe_path: str, media_path: str, audio_track_no: int | None) -> float:
    _, audio_stream_count = _probe_media_stream_layout(ffprobe_path, media_path)
    if audio_stream_count <= 0:
        return 0.0
    if audio_track_no is None:
        audio_indices = range(audio_stream_count)
    else:
        audio_indices = [max(0, min(audio_stream_count - 1, int(audio_track_no) - 1))]
    return max((_probe_audio_end_time(ffprobe_path, media_path, audio_index) for audio_index in audio_indices), default=0.0)


def _probe_audio_stream_codecs(ffprobe_path: str, media_path: str) -> list[str]:
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index,codec_name", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if result.returncode != 0:
        raise gr.Error((result.stderr or result.stdout or f"ffprobe failed for {media_path}").strip())
    try:
        streams = list((json.loads(result.stdout) or {}).get("streams") or [])
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Unable to read audio codecs for {media_path}") from exc
    return [str(stream.get("codec_name") or "").strip().lower() for stream in streams]


def _validate_audio_copy_container(ffprobe_path: str, source_path: str, video_container: str, audio_track_no: int | None) -> None:
    if _normalize_container_name(video_container) != "mp4":
        return
    supported_codecs = {"aac", "ac3", "alac", "eac3", "mp3", "opus"}
    audio_codecs = _probe_audio_stream_codecs(ffprobe_path, source_path)
    if audio_track_no is None:
        selected_codecs = [codec for codec in audio_codecs if len(codec) > 0]
    else:
        selected_index = max(0, int(audio_track_no) - 1)
        selected_codecs = [audio_codecs[selected_index]] if selected_index < len(audio_codecs) and len(audio_codecs[selected_index]) > 0 else []
    incompatible_codecs = [codec for codec in selected_codecs if codec not in supported_codecs]
    if len(incompatible_codecs) > 0:
        track_label = f"audio track {int(audio_track_no)}" if audio_track_no is not None else "the selected audio tracks"
        codecs_text = ", ".join(sorted(set(incompatible_codecs)))
        raise gr.Error(f"MP4 output cannot packet-copy {track_label} with codec(s): {codecs_text}. Use an .mkv output path or choose a compatible track.")


def _start_video_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str) -> subprocess.Popen:
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{float(fps_float):.12g}",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
    ]
    command += get_video_encode_args(video_codec, video_container)
    command += _get_live_mux_output_args(video_container)
    command += [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)


def _start_av_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str, source_path: str, start_seconds: float, audio_track_no: int | None) -> subprocess.Popen:
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{float(fps_float):.12g}",
        "-i",
        "pipe:0",
        "-ss",
        f"{max(0.0, float(start_seconds)):.12g}",
        "-i",
        source_path,
        "-map_metadata",
        "1",
        "-map",
        "0:v:0",
    ]
    if audio_track_no is None:
        command += ["-map", "1:a?"]
    else:
        command += ["-map", f"1:a:{max(0, int(audio_track_no) - 1)}?"]
    command += get_video_encode_args(video_codec, video_container) + ["-c:a", "copy", "-shortest"]
    command += _get_live_mux_output_args(video_container)
    command += [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)


def _finalize_mux_process(process: subprocess.Popen, *, timeout_seconds: float = 30.0) -> tuple[int, str, bool]:
    if process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.close()
        except OSError:
            pass
    forced_termination = False
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        forced_termination = True
        process.kill()
        return_code = process.wait(timeout=5)
    stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None else ""
    return return_code, stderr, forced_termination


def _mux_source_audio(ffmpeg_path: str, video_only_path: str, output_path: str, source_path: str, start_seconds: float, duration_seconds: float, audio_track_no: int | None) -> None:
    temp_output_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_mux{Path(output_path).suffix}"))
    command = [ffmpeg_path, "-y", "-v", "error", "-i", video_only_path, "-ss", f"{max(0.0, float(start_seconds)):.12g}", "-t", f"{max(0.0, float(duration_seconds)):.12g}", "-i", source_path, "-map_metadata", "1", "-map", "0:v:0"]
    if audio_track_no is None:
        command += ["-map", "1:a?"]
    else:
        command += ["-map", f"1:a:{max(0, int(audio_track_no) - 1)}?"]
    command += ["-c:v", "copy", "-c:a", "copy", "-shortest"]
    if str(Path(output_path).suffix).strip().lower() == ".mp4":
        command += ["-movflags", "+faststart"]
    command += [temp_output_path]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(temp_output_path):
        if os.path.isfile(temp_output_path):
            os.remove(temp_output_path)
        raise gr.Error((result.stderr or result.stdout or "ffmpeg audio mux failed").strip())
    os.replace(temp_output_path, output_path)

def _write_output_metadata(ffmpeg_path: str, output_path: str, metadata: dict) -> None:
    if len(metadata) == 0 or not os.path.isfile(output_path):
        return
    temp_output_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_metadata{Path(output_path).suffix}"))
    command = [ffmpeg_path, "-y", "-v", "error", "-i", output_path, "-map", "0", "-c", "copy", "-metadata", f"comment={json.dumps(metadata, ensure_ascii=False)}", temp_output_path]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and os.path.isfile(temp_output_path):
        os.replace(temp_output_path, output_path)
        return
    if os.path.isfile(temp_output_path):
        os.remove(temp_output_path)
    print(f"[Process Full Video] Warning: failed to write metadata to {output_path}: {(result.stderr or result.stdout or '').strip()}")


def _make_output_temp_dir(output_path: str, prefix: str) -> str:
    return tempfile.mkdtemp(prefix=prefix, dir=str(Path(output_path).resolve().parent))


def _plugin_info(message: str) -> None:
    text = str(message or "").strip()
    if len(text) == 0:
        return
    print(f"[Process Full Video] {text}")
    gr.Info(text)


def _job_was_stopped(job_result) -> bool:
    return bool(getattr(job_result, "cancelled", False))


def _request_job_stop(job) -> None:
    stop_fn = getattr(job, "cancel", None)
    if callable(stop_fn):
        stop_fn()


def _probe_media_stream_layout(ffprobe_path: str, media_path: str) -> tuple[int, int]:
    result = subprocess.run([ffprobe_path, "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", media_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise gr.Error((result.stderr or result.stdout or f"ffprobe failed for {media_path}").strip())
    try:
        streams = list((json.loads(result.stdout) or {}).get("streams") or [])
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Unable to read media stream layout for {media_path}") from exc
    video_count = sum(1 for stream in streams if str(stream.get("codec_type") or "").lower() == "video")
    audio_count = sum(1 for stream in streams if str(stream.get("codec_type") or "").lower() == "audio")
    return video_count, audio_count


def _probe_primary_video_codec(ffprobe_path: str, media_path: str) -> str:
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if result.returncode != 0:
        raise gr.Error((result.stderr or result.stdout or f"ffprobe failed for {media_path}").strip())
    try:
        codec_name = str((((json.loads(result.stdout) or {}).get("streams") or [{}])[0]).get("codec_name") or "").strip().lower()
    except (TypeError, ValueError, json.JSONDecodeError, IndexError):
        codec_name = ""
    if len(codec_name) == 0:
        raise gr.Error(f"Unable to detect the video codec of {media_path}")
    return codec_name


def _probe_primary_video_rate(ffprobe_path: str, media_path: str) -> Fraction | None:
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if result.returncode != 0:
        raise gr.Error((result.stderr or result.stdout or f"ffprobe failed for {media_path}").strip())
    try:
        stream = (((json.loads(result.stdout) or {}).get("streams") or [{}])[0])
    except (TypeError, ValueError, json.JSONDecodeError, IndexError):
        return None
    for key in ("r_frame_rate", "avg_frame_rate"):
        rate_text = str(stream.get(key) or "").strip()
        if len(rate_text) == 0 or rate_text in ("0/0", "N/A"):
            continue
        try:
            rate = Fraction(rate_text)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    return None


def _probe_video_packet_times(ffprobe_path: str, media_path: str, *, start_seconds: float | None = None, duration_seconds: float | None = None) -> list[float]:
    command = [ffprobe_path, "-v", "error", "-select_streams", "v:0"]
    if start_seconds is not None and duration_seconds is not None and float(duration_seconds) > 0.0:
        command += ["-read_intervals", f"{max(0.0, float(start_seconds)):.6f}%+{max(0.05, float(duration_seconds)):.6f}"]
    command += ["-show_packets", "-show_entries", "packet=dts_time,pts_time", "-of", "json", media_path]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if result.returncode != 0:
        raise gr.Error((result.stderr or result.stdout or f"ffprobe failed for {media_path}").strip())
    try:
        return sorted(
            float(packet.get("dts_time") if packet.get("dts_time") is not None else packet.get("pts_time"))
            for packet in ((json.loads(result.stdout) or {}).get("packets") or [])
            if packet.get("dts_time") is not None or packet.get("pts_time") is not None
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _probe_video_frame_gap(ffprobe_path: str, media_path: str, fps_float: float, *, near_time: float | None = None, window_seconds: float = 4.0) -> tuple[float, float, float] | None:
    fps_value = float(fps_float or 0.0)
    if fps_value <= 0.0 or not os.path.isfile(media_path):
        return None
    max_delta = max(1.0 / fps_value * 1.5, 0.05)
    if near_time is None:
        packet_times = _probe_video_packet_times(ffprobe_path, media_path)
        if len(packet_times) < 2:
            return None
        for current_pts, next_pts in zip(packet_times, packet_times[1:]):
            delta = float(next_pts) - float(current_pts)
            if delta > max_delta:
                return float(current_pts), float(next_pts), float(delta)
        return None
    seam_time = max(0.0, float(near_time))
    local_window = max(1.0, float(window_seconds))
    seam_margin = max_delta * 2.0
    total_duration = max(seam_time, _probe_media_duration(ffprobe_path, media_path))
    probe_start = max(0.0, seam_time - max(0.25, seam_margin))
    probe_duration = min(max(local_window, 4.0), max(0.05, total_duration - probe_start))
    while probe_duration > 0.0:
        packet_times = _probe_video_packet_times(ffprobe_path, media_path, start_seconds=probe_start, duration_seconds=probe_duration)
        prev_candidates = [packet_time for packet_time in packet_times if packet_time <= seam_time + seam_margin]
        if len(prev_candidates) > 0:
            prev_time = prev_candidates[-1]
            next_candidates = [packet_time for packet_time in packet_times if packet_time > prev_time + 1e-9]
            if len(next_candidates) > 0:
                next_time = next_candidates[0]
                delta = float(next_time) - float(prev_time)
                return None if delta <= max_delta else (float(prev_time), float(next_time), float(delta))
        if probe_start + probe_duration >= total_duration - 1e-6:
            return None
        probe_duration = min(max(2.0 * probe_duration, local_window), max(0.05, total_duration - probe_start))
    return None


def _write_concat_list(list_path: str, media_paths: list[str]) -> None:
    with open(list_path, "w", encoding="utf-8") as handle:
        for media_path in media_paths:
            escaped_path = str(media_path).replace("'", "'\\''")
            handle.write(f"file '{escaped_path}'\n")


def _build_mp4_video_reconstruct_bsf(frame_rate: Fraction | None, fps_float: float) -> str:
    if frame_rate is not None and frame_rate.numerator > 0 and frame_rate.denominator > 0:
        frame_duration_expr = f"{int(frame_rate.denominator)}/({int(frame_rate.numerator)}*TB)"
    else:
        fps_value = max(float(fps_float or 0.0), 1.0)
        frame_duration_expr = f"1/({fps_value:.15g}*TB)"
    return (
        "setts="
        f"pts='if(eq(N,0),PTS,PREV_OUTPTS+(PTS-PREV_INPTS)-(PREV_INDURATION-DURATION))':"
        f"dts='if(eq(N,0),DTS,PREV_OUTDTS+(DTS-PREV_INDTS)-(PREV_INDURATION-DURATION))':"
        f"duration='if(eq(N,0),{frame_duration_expr},DURATION)'"
    )


def _build_mp4_video_zero_base_bsf() -> str:
    return "setts=pts=PTS-STARTPTS:dts=DTS:duration=DURATION"


def _get_mp4_video_track_timescale(frame_rate: Fraction | None, fps_float: float) -> int:
    if frame_rate is not None and frame_rate.numerator > 0:
        return int(frame_rate.numerator)
    return max(1, int(round(max(float(fps_float or 0.0), 1.0) * 1000.0)))


def _concat_video_streams_for_mp4(ffmpeg_path: str, segment_paths: list[str], output_path: str, work_dir: str, *, fps_float: float, frame_rate: Fraction | None = None) -> int:
    temp_paths: list[str] = []
    prepared_paths: list[str] = []
    list_path = os.path.join(work_dir, "video_mp4.txt")
    reconstruct_bsf = _build_mp4_video_reconstruct_bsf(frame_rate, fps_float)
    zero_base_bsf = _build_mp4_video_zero_base_bsf()
    track_timescale = _get_mp4_video_track_timescale(frame_rate, fps_float)
    try:
        for segment_no, segment_path in enumerate(segment_paths, start=1):
            reconstructed_path = os.path.join(work_dir, f"segment_{segment_no}_video_step1.mp4")
            prepared_path = os.path.join(work_dir, f"segment_{segment_no}_video.mp4")
            result = subprocess.run([ffmpeg_path, "-y", "-v", "error", "-i", segment_path, "-map", "0:v:0", "-c", "copy", "-bsf:v", reconstruct_bsf, "-video_track_timescale", str(track_timescale), reconstructed_path], capture_output=True, text=True)
            if result.returncode != 0 or not os.path.isfile(reconstructed_path):
                raise gr.Error((result.stderr or result.stdout or f"ffmpeg failed to prepare {segment_path} for MP4 concat").strip())
            temp_paths.append(reconstructed_path)
            result = subprocess.run([ffmpeg_path, "-y", "-v", "error", "-i", reconstructed_path, "-map", "0:v:0", "-c", "copy", "-bsf:v", zero_base_bsf, "-video_track_timescale", str(track_timescale), prepared_path], capture_output=True, text=True)
            if result.returncode != 0 or not os.path.isfile(prepared_path):
                raise gr.Error((result.stderr or result.stdout or f"ffmpeg failed to zero-base {segment_path} for MP4 concat").strip())
            prepared_paths.append(prepared_path)
        _write_concat_list(list_path, prepared_paths)
        result = subprocess.run([ffmpeg_path, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path, "-map", "0:v:0", "-c", "copy", "-video_track_timescale", str(track_timescale), output_path], capture_output=True, text=True)
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise gr.Error((result.stderr or result.stdout or "ffmpeg failed to concatenate MP4 continuation video").strip())
        return track_timescale
    finally:
        for temp_path in temp_paths:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
        for prepared_path in prepared_paths:
            if os.path.isfile(prepared_path):
                os.remove(prepared_path)
        if os.path.isfile(list_path):
            os.remove(list_path)


def _concat_audio_segments(ffmpeg_path: str, segment_paths: list[str], output_path: str, work_dir: str, *, segment_trim_seconds: list[float] | None = None, segment_duration_seconds: list[float | None] | None = None, audio_stream_indices: list[int] | None = None) -> None:
    extracted_paths: list[str] = []
    list_path = os.path.join(work_dir, "audio.txt")
    try:
        for segment_no, segment_path in enumerate(segment_paths, start=1):
            extracted_path = os.path.join(work_dir, f"segment_{segment_no}_audio.mka")
            trim_seconds = max(0.0, float(segment_trim_seconds[segment_no - 1])) if segment_trim_seconds is not None and segment_no - 1 < len(segment_trim_seconds) else 0.0
            duration_seconds = None
            if segment_duration_seconds is not None and segment_no - 1 < len(segment_duration_seconds) and segment_duration_seconds[segment_no - 1] is not None:
                duration_seconds = max(0.0, float(segment_duration_seconds[segment_no - 1]))
            audio_stream_index = max(0, int(audio_stream_indices[segment_no - 1])) if audio_stream_indices is not None and segment_no - 1 < len(audio_stream_indices) else 0
            command = [ffmpeg_path, "-y", "-v", "error"]
            if trim_seconds > 0.0:
                command += ["-ss", f"{trim_seconds:.12g}"]
            command += ["-i", segment_path]
            if duration_seconds is not None and duration_seconds > 0.0:
                command += ["-t", f"{duration_seconds:.12g}"]
            command += ["-map", f"0:a:{audio_stream_index}?", "-c", "copy", "-avoid_negative_ts", "make_zero", extracted_path]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.isfile(extracted_path):
                raise gr.Error((result.stderr or result.stdout or f"ffmpeg failed to extract audio from {segment_path}").strip())
            extracted_paths.append(extracted_path)
        _write_concat_list(list_path, extracted_paths)
        result = subprocess.run([ffmpeg_path, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path], capture_output=True, text=True)
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise gr.Error((result.stderr or result.stdout or "ffmpeg failed to concatenate audio streams").strip())
    finally:
        for extracted_path in extracted_paths:
            if os.path.isfile(extracted_path):
                os.remove(extracted_path)
        if os.path.isfile(list_path):
            os.remove(list_path)


def _concat_video_segments(ffmpeg_path: str, segment_paths: list[str], output_path: str, video_codec: str, video_container: str, audio_codec_key: str, *, segment_audio_trim_seconds: list[float] | None = None, fps_float: float | None = None, selected_audio_track_no: int | None = None) -> None:
    segment_paths = [str(Path(path).resolve()) for path in segment_paths if isinstance(path, str) and os.path.isfile(path)]
    if len(segment_paths) == 0:
        raise gr.Error("No output segments available to merge.")
    if len(segment_paths) == 1:
        if str(Path(segment_paths[0]).resolve()) != str(Path(output_path).resolve()):
            os.replace(segment_paths[0], output_path)
        return
    ffprobe_path = resolve_media_binary("ffprobe")
    layouts = [_probe_media_stream_layout(ffprobe_path, path) for path in segment_paths]
    if any(video_count != 1 for video_count, _ in layouts):
        raise gr.Error("All continuation segments must contain exactly one video stream.")
    audio_stream_counts = [audio_count for _, audio_count in layouts]
    has_audio = any(audio_count > 0 for audio_count in audio_stream_counts)
    if has_audio and any(audio_count <= 0 for audio_count in audio_stream_counts):
        raise gr.Error("All continuation segments must expose an audio stream.")
    fps_value = float(fps_float or 0.0)
    concat_dir = _make_output_temp_dir(output_path, "wangp_process_full_video_concat_")
    merged_video_path = os.path.join(concat_dir, "merged_video.mp4" if _normalize_container_name(video_container) == "mp4" else "merged_video.mkv")
    temp_output_path = os.path.join(concat_dir, f"{Path(output_path).stem}_merged{Path(output_path).suffix}")
    video_track_timescale = None
    try:
        if _normalize_container_name(video_container) == "mp4":
            video_frame_rate = _probe_primary_video_rate(ffprobe_path, segment_paths[0])
            video_track_timescale = _concat_video_streams_for_mp4(ffmpeg_path, segment_paths, merged_video_path, concat_dir, fps_float=fps_value, frame_rate=video_frame_rate)
        else:
            video_codec_name = _probe_primary_video_codec(ffprobe_path, segment_paths[0])
            video_bsf = "h264_mp4toannexb" if video_codec_name == "h264" else "hevc_mp4toannexb" if video_codec_name in ("hevc", "h265") else ""
            if len(video_bsf) == 0:
                raise gr.Error(f"Unsupported continuation video codec for no-reencode merge: {video_codec_name}")
            concat_ts_path = os.path.join(concat_dir, "segments.ts")
            ts_paths: list[str] = []
            for index, segment_path in enumerate(segment_paths, start=1):
                ts_path = os.path.join(concat_dir, f"segment_{index}.ts")
                result = subprocess.run([ffmpeg_path, "-y", "-v", "error", "-i", segment_path, "-map", "0:v:0", "-c", "copy", "-bsf:v", video_bsf, "-f", "mpegts", ts_path], capture_output=True, text=True)
                if result.returncode != 0 or not os.path.isfile(ts_path):
                    raise gr.Error((result.stderr or result.stdout or f"ffmpeg failed to prepare {segment_path} for concat").strip())
                ts_paths.append(ts_path)
            with open(concat_ts_path, "wb") as handle:
                for ts_path in ts_paths:
                    with open(ts_path, "rb") as ts_file:
                        shutil.copyfileobj(ts_file, handle, length=1024 * 1024)
            result = subprocess.run([ffmpeg_path, "-y", "-v", "error", "-i", concat_ts_path, "-map", "0:v:0", "-c", "copy", merged_video_path], capture_output=True, text=True)
            if result.returncode != 0 or not os.path.isfile(merged_video_path):
                raise gr.Error((result.stderr or result.stdout or "ffmpeg failed to concatenate video stream").strip())
        command = [ffmpeg_path, "-y", "-v", "error", "-i", merged_video_path]
        if has_audio:
            selected_audio_index = max(0, int(selected_audio_track_no or 1) - 1)
            audio_stream_indices = [0 if audio_count <= 1 else min(audio_count - 1, selected_audio_index) for audio_count in audio_stream_counts]
            merged_audio_path = os.path.join(concat_dir, "merged_audio.mka")
            segment_audio_duration_seconds = [None] * len(segment_paths)
            if _normalize_container_name(video_container) == "mp4" and fps_value > 0.0:
                max_overrun_seconds = max(0.25, 2.0 / fps_value)
                for segment_index, segment_path in enumerate(segment_paths):
                    segment_frame_count, _ = _probe_resume_frame_count(ffprobe_path, segment_path, fps_value)
                    if segment_frame_count <= 0:
                        continue
                    actual_video_duration = float(segment_frame_count) / fps_value
                    container_duration = _probe_media_duration(ffprobe_path, segment_path)
                    if container_duration - actual_video_duration <= max_overrun_seconds:
                        continue
                    trim_seconds = max(0.0, float(segment_audio_trim_seconds[segment_index])) if segment_audio_trim_seconds is not None and segment_index < len(segment_audio_trim_seconds) else 0.0
                    segment_audio_duration_seconds[segment_index] = max(0.0, actual_video_duration - trim_seconds)
            _concat_audio_segments(ffmpeg_path, segment_paths, merged_audio_path, concat_dir, segment_trim_seconds=segment_audio_trim_seconds, segment_duration_seconds=segment_audio_duration_seconds, audio_stream_indices=audio_stream_indices)
            command += ["-i", merged_audio_path]
        command += ["-map_metadata", "0"]
        command += ["-map", "0:v:0"]
        if has_audio:
            command += ["-map", "1:a:0"]
        command += ["-c", "copy"]
        if _normalize_container_name(video_container) == "mp4":
            command += ["-movflags", "+faststart"]
            if video_track_timescale is not None:
                command += ["-video_track_timescale", str(video_track_timescale)]
        command += [temp_output_path]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.isfile(temp_output_path):
            raise gr.Error((result.stderr or result.stdout or "ffmpeg concat failed").strip())
        timeline_gap = _probe_video_frame_gap(ffprobe_path, temp_output_path, fps_value, near_time=_probe_media_duration(ffprobe_path, segment_paths[0]))
        if timeline_gap is not None:
            gap_start, gap_end, gap_seconds = timeline_gap
            raise gr.Error(f"Merged video timeline contains a {gap_seconds:.6f}s gap near {gap_start:.3f}s -> {gap_end:.3f}s.")
        os.replace(temp_output_path, output_path)
    finally:
        shutil.rmtree(concat_dir, ignore_errors=True)


def _phase_label_from_status(status: str = "") -> str:
    return extract_status_phase_label(status)


def _phase_label_from_update(update=None, *, status: str = "", phase: str = "", raw_phase: str = "") -> str:
    status_phase = _phase_label_from_status(status or getattr(update, "status", ""))
    raw_phase_text = str(raw_phase or getattr(update, "raw_phase", "") or phase or "").strip()
    if len(status_phase) > 0:
        return status_phase
    return raw_phase_text


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds_only = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_only:02d}" if hours > 0 else f"{minutes:02d}:{seconds_only:02d}"


def _render_chunk_status_html(total_chunks: int, completed_chunks: int, current_chunk: int, phase_label: str, status_text: str, *, continued: bool = False, phase_current_step=None, phase_total_steps=None, elapsed_seconds: float | None = None, eta_seconds: float | None = None, prefer_status_phase: bool = False) -> str:
    total_chunks = max(1, int(total_chunks))
    completed_chunks = max(0, min(int(completed_chunks), total_chunks))
    current_chunk = max(1, min(int(current_chunk), total_chunks))
    top_ratio = completed_chunks / total_chunks
    top_width = f"{100.0 * top_ratio:.2f}%"
    raw_status_text = str(status_text or "").strip()
    raw_phase_text = str(phase_label or "").strip()
    if prefer_status_phase:
        derived_phase = _phase_label_from_status(raw_status_text)
        if len(derived_phase) > 0:
            raw_phase_text = derived_phase
    phase_html = html.escape(raw_phase_text or "Queued in WanGP...")
    status_html = html.escape(raw_status_text or raw_phase_text or "")
    continued_suffix = " (Continued)" if continued else ""
    has_phase_progress = isinstance(phase_current_step, int) and isinstance(phase_total_steps, int) and phase_total_steps > 0
    phase_ratio = max(0.0, min(float(phase_current_step) / float(phase_total_steps), 1.0)) if has_phase_progress else None
    phase_width = f"{100.0 * phase_ratio:.2f}%" if phase_ratio is not None else "0%"
    phase_suffix = f" ({phase_current_step} / {phase_total_steps})" if has_phase_progress else ""
    elapsed_html = html.escape(_format_elapsed(elapsed_seconds))
    eta_html = html.escape(_format_elapsed(eta_seconds))
    normalized_phase = raw_phase_text.lower()
    normalized_status = raw_status_text
    show_status_line = (not prefer_status_phase) and len(normalized_status) > 0 and (len(normalized_phase) == 0 or normalized_phase not in normalized_status.lower())
    status_line_html = f"<div style='font-size:0.9em;color:#4b5563'>{status_html}</div>" if show_status_line else ""
    return (
        "<div style='display:flex;flex-direction:column;gap:8px'>"
        f"<div style='font-weight:600'>Chunks Processed: {completed_chunks} / {total_chunks}{continued_suffix}</div>"
        "<div style='height:12px;border-radius:999px;background:#d7dce3;overflow:hidden'>"
        f"<div style='height:100%;width:{top_width};background:linear-gradient(90deg,#2f7de1,#5db0ff)'></div>"
        "</div>"
        f"<div style='font-size:0.95em'><b>Phase:</b> {phase_html}{phase_suffix}</div>"
        "<div style='height:12px;border-radius:999px;background:#d7dce3;overflow:hidden'>"
        f"<div style='height:100%;width:{phase_width};background:linear-gradient(90deg,#e37a2f,#ffb05d)'></div>"
        "</div>"
        f"<div style='font-size:0.9em;color:#4b5563'><b>Elapsed:</b> {elapsed_html} <span style='padding-left:12px'><b>ETA:</b> {eta_html}</span></div>"
        f"{status_line_html}"
        "</div>"
    )


def _render_output_file_html(output_path: str) -> str:
    value = html.escape(str(output_path or ""), quote=False)
    return (
        "<div style='display:flex;flex-direction:column;gap:6px'>"
        "<div style='font-size:var(--block-label-text-size);font-weight:var(--block-label-text-weight);line-height:var(--line-sm)'>Output File</div>"
        f"<textarea readonly onclick='this.select()' spellcheck='false' rows='1' "
        "style='width:100%;min-height:35.64px;resize:none;overflow:hidden;padding:calc(8px * var(--wangp-ui-scale)) calc(12px * var(--wangp-ui-scale));"
        "border:1px solid var(--input-border-color);border-radius:var(--input-radius);background:var(--input-background-fill);color:var(--body-text-color);"
        "font:inherit;line-height:1.5;box-sizing:border-box'>"
        f"{value}</textarea>"
        "</div>"
    )


def _delete_released_chunk_outputs(state: dict, chunk_output_paths: list[str]) -> list[str]:
    if not isinstance(state, dict):
        return chunk_output_paths
    gen = state.get("gen", {})
    if not isinstance(gen, dict):
        return chunk_output_paths
    referenced_paths = {
        str(Path(path).resolve())
        for path in list(gen.get("file_list", []) or []) + list(gen.get("audio_file_list", []) or [])
        if isinstance(path, str) and len(path.strip()) > 0
    }
    kept_paths: list[str] = []
    for path in chunk_output_paths:
        resolved = str(Path(path).resolve())
        if resolved in referenced_paths:
            kept_paths.append(resolved)
            continue
        if os.path.isfile(resolved):
            try:
                os.remove(resolved)
            except OSError:
                kept_paths.append(resolved)
    return kept_paths


class ConfigTabPlugin(WAN2GPPlugin):
    def setup_ui(self):
        self.request_global("server_config")
        self.request_component("state")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_config_ui)

    def create_config_ui(self, api_session):
        active_job = {"job": None}
        preview_state = {"image": None}
        ui_skip = object()

        def refresh_preview(_refresh_id):
            return preview_state["image"]

        def _button_update(label: str, enabled: bool | None):
            return gr.skip() if enabled is None else gr.update(value=label, interactive=bool(enabled))

        def _ui_update(status=ui_skip, output=ui_skip, preview_refresh=ui_skip, *, start_enabled: bool | None = None, abort_enabled: bool | None = None):
            status_update = gr.skip() if status is ui_skip else status
            output_update = gr.skip() if output is ui_skip else _render_output_file_html(output)
            preview_update = gr.skip() if preview_refresh is ui_skip else preview_refresh
            start_update = _button_update("Start Process", start_enabled)
            abort_update = _button_update("Stop", abort_enabled)
            return status_update, output_update, preview_update, start_update, abort_update

        def _reset_live_chunk_status(state: dict) -> None:
            gen = state.get("gen") if isinstance(state, dict) else None
            if not isinstance(gen, dict):
                return
            gen["status"] = ""
            gen["status_display"] = False
            gen["progress_args"] = None
            gen["progress_phase"] = None
            gen["progress_status"] = ""
            gen["preview"] = None

        def start_process(state, process_name, source_path, output_path, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, start_seconds, end_seconds):
            if process_name != "outpaint_video":
                raise gr.Error(f"Unsupported process: {process_name}")
            source_path = str(source_path or "").strip()
            if not os.path.isfile(source_path):
                raise gr.Error(f"Source video not found: {source_path}")
            start_seconds = _parse_time_input(start_seconds, label="Start", allow_empty=False)
            end_seconds = _parse_time_input(end_seconds, label="End", allow_empty=True)
            started_ui = False
            try:
                yield _ui_update(_render_chunk_status_html(1, 0, 1, "Initializing", "Preparing processing job..."), ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=True)
                started_ui = True
                _plugin_info("Starting Process Full Video conversion...")
                output_path, resume_existing_output = _resolve_output_path(source_path, output_path, target_ratio, str(output_resolution), start_seconds, end_seconds, bool(continue_enabled))
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                metadata = get_video_info_details(source_path)
                start_frame, end_frame_exclusive, fps_float, total_source_frames = _compute_selected_frame_range(metadata, start_seconds, end_seconds)
                audio_track_count = int(extract_audio_tracks(source_path, query_only=True))
                selected_audio_track = None
                if source_audio_track not in (None, ""):
                    source_audio_track_value = float(source_audio_track)
                    if not math.isnan(source_audio_track_value) and source_audio_track_value > 0:
                        selected_audio_track = int(source_audio_track_value)
                elif audio_track_count > 0:
                    selected_audio_track = 1
                if selected_audio_track is not None and (selected_audio_track <= 0 or selected_audio_track > audio_track_count):
                    raise gr.Error(f"Source Audio must be between 1 and {audio_track_count}.")
                model_type = DEFAULT_MODEL_TYPE
                frame_plan_rules = _get_frame_plan_rules(model_type)
                budget_resolution = _choose_resolution(str(output_resolution))
                chunk_frames = _normalize_chunk_frames(float(chunk_size_seconds or 10.0), fps_float, frame_step=frame_plan_rules.frame_step, minimum_requested_frames=frame_plan_rules.minimum_requested_frames)
                selected_unique_frames = end_frame_exclusive - start_frame
                full_plans = _build_chunk_plan(start_frame, end_frame_exclusive, total_source_frames, chunk_frames, frame_step=frame_plan_rules.frame_step, minimum_requested_frames=frame_plan_rules.minimum_requested_frames)
                requested_unique_frames = _count_planned_unique_frames(full_plans)
                dropped_tail_frames = max(0, selected_unique_frames - requested_unique_frames)
                if dropped_tail_frames > 0:
                    _plugin_info(f"Dropping the last {dropped_tail_frames} source frame(s) so the selected range fits the current model chunk shape.")
                ffmpeg_path = resolve_media_binary("ffmpeg")
                if ffmpeg_path is None:
                    raise gr.Error("ffmpeg binary not found.")
                ffprobe_path = resolve_media_binary("ffprobe")
                if ffprobe_path is None:
                    raise gr.Error("ffprobe binary not found.")
                output_container = _normalize_container_name(Path(output_path).suffix.lstrip(".") or self.server_config.get("video_container", "mp4"))
                _validate_audio_copy_container(ffprobe_path, source_path, output_container, selected_audio_track)
                mux_process = None
                stopped = False
                temp_dir = _make_output_temp_dir(output_path, "wangp_process_full_video_")
                last_frame_path = os.path.join(temp_dir, "last_frame.png")
                last_frame_image = None
                continuation_output_path = ""
                chunk_output_paths: list[str] = []
                written_unique_frames = 0
                resumed_unique_frames = 0
                completed_chunks = 0
                resolved_resolution = ""
                resolved_width = 0
                resolved_height = 0
                mux_finished = False
                merged_continuation = False
                resume_audio_trim_seconds = 0.0
                preview_state["image"] = None
                output_path_for_write = output_path
                video_only_output_path = os.path.join(temp_dir, f"{Path(output_path_for_write).stem}_videoonly{Path(output_path_for_write).suffix}")
                exact_start_seconds = start_frame / fps_float
                if resume_existing_output:
                    yield _ui_update(_render_chunk_status_html(max(1, len(full_plans)), 0, 1, "Inspecting Existing Output", f"Inspecting existing output to continue: {output_path}"), output_path, str(time.time_ns()))
                    resumed_unique_frames, resume_reason = _probe_resume_frame_count(ffprobe_path, output_path, fps_float)
                    resumed_unique_frames = max(0, min(requested_unique_frames, resumed_unique_frames))
                    if resumed_unique_frames <= 0:
                        _plugin_info(f"Unable to continue from existing output: {output_path}. {resume_reason or 'Starting a new file instead.'}")
                        output_path = _make_output_variant(Path(output_path))
                        output_path_for_write = output_path
                        video_only_output_path = os.path.join(temp_dir, f"{Path(output_path_for_write).stem}_videoonly{Path(output_path_for_write).suffix}")
                        resume_existing_output = False
                    else:
                        _plugin_info(f"Continuing existing output: {output_path}")
                        resolved_resolution, resolved_width, resolved_height = _probe_existing_output_resolution(output_path)
                        print(f"[Process Full Video] Continuing with locked output resolution {resolved_resolution}")
                        completed_chunks, _ = _count_completed_chunks(full_plans, resumed_unique_frames)
                        exact_start_seconds = (start_frame + resumed_unique_frames) / fps_float
                        if resumed_unique_frames < requested_unique_frames:
                            yield _ui_update(_render_chunk_status_html(max(1, len(full_plans)), 0, 1, "Loading Last Frame", f"Continuing existing output with {resumed_unique_frames} frame(s) already written."), output_path, str(time.time_ns()))
                            resumed_unique_frames, last_frame_image, tail_reason = _resolve_resume_last_frame(output_path, resumed_unique_frames)
                            if resumed_unique_frames <= 0 or last_frame_image is None:
                                _plugin_info(f"Unable to continue from existing output: {output_path}. {tail_reason or 'Starting a new file instead.'}")
                                output_path = _make_output_variant(Path(output_path))
                                output_path_for_write = output_path
                                video_only_output_path = os.path.join(temp_dir, f"{Path(output_path_for_write).stem}_videoonly{Path(output_path_for_write).suffix}")
                                resume_existing_output = False
                            else:
                                if tail_reason:
                                    _plugin_info(tail_reason)
                                last_frame_image.save(last_frame_path, format="PNG")
                                preview_state["image"] = last_frame_image
                                completed_chunks, _ = _count_completed_chunks(full_plans, resumed_unique_frames)
                                exact_start_seconds = (start_frame + resumed_unique_frames) / fps_float
                        if resume_existing_output and resumed_unique_frames < requested_unique_frames:
                            resume_audio_trim_seconds = max(0.0, _probe_selected_audio_end_time(ffprobe_path, output_path, selected_audio_track) - (resumed_unique_frames / fps_float))
                            if resume_audio_trim_seconds > 0.0:
                                print(f"[Process Full Video] Trimming {resume_audio_trim_seconds:.6f}s of leading audio packets from the continuation segment during final merge")
                            continuation_output_path = _make_continuation_output_path(output_path)
                            output_path_for_write = continuation_output_path
                            video_only_output_path = os.path.join(temp_dir, f"{Path(output_path_for_write).stem}_videoonly{Path(output_path_for_write).suffix}")
                            plans = _build_chunk_plan(start_frame + resumed_unique_frames, end_frame_exclusive, total_source_frames, chunk_frames, frame_step=frame_plan_rules.frame_step, minimum_requested_frames=frame_plan_rules.minimum_requested_frames, initial_drop_first=True)
                        elif resume_existing_output:
                            plans = []
                if not resume_existing_output:
                    plans = full_plans
                continued_mode = resumed_unique_frames > 0
                use_live_av_mux = True
                total_chunks_display = completed_chunks + len(plans)
                run_started_at = time.time()
                initial_completed_chunks = completed_chunks

                def _timing_kwargs(phase_current_step=None, phase_total_steps=None):
                    elapsed_seconds = max(0.0, time.time() - run_started_at)
                    run_total_chunks = max(1, len(plans))
                    run_completed_chunks = max(0, completed_chunks - initial_completed_chunks)
                    phase_ratio = 0.0
                    if isinstance(phase_current_step, int) and isinstance(phase_total_steps, int) and phase_total_steps > 0:
                        phase_ratio = max(0.0, min(float(phase_current_step) / float(phase_total_steps), 1.0))
                    overall_ratio = max(0.0, min((run_completed_chunks + phase_ratio) / float(run_total_chunks), 1.0))
                    eta_seconds = None if overall_ratio <= 0.0 or overall_ratio >= 1.0 else elapsed_seconds * (1.0 - overall_ratio) / overall_ratio
                    return {"elapsed_seconds": elapsed_seconds, "eta_seconds": eta_seconds}

                if len(plans) == 0:
                    yield _ui_update(_render_chunk_status_html(max(1, len(full_plans)), len(full_plans), len(full_plans), "Completed", "Existing output already covers the requested range.", continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()), start_enabled=True, abort_enabled=False)
                    return
                planning_text = f"Resuming from {resumed_unique_frames} frame(s) already written." if resumed_unique_frames > 0 else f"Preparing {len(plans)} chunk(s)..."
                yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Planning", planning_text, continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()))

                for chunk_index, plan in enumerate(plans, start=1):
                    class ChunkCallbacks:
                        def __init__(self) -> None:
                            self.phase_label = "Queued in WanGP..."
                            self.status_text = "Queued in WanGP..."
                            self.current_step = None
                            self.total_steps = None
                            self._last_explicit_status_at = 0.0

                        def on_status(self, status):
                            self.status_text = str(status or "").strip() or self.status_text
                            status_phase = _phase_label_from_status(self.status_text)
                            if len(status_phase) > 0:
                                if status_phase != self.phase_label:
                                    self.current_step = None
                                    self.total_steps = None
                                self.phase_label = status_phase
                                self._last_explicit_status_at = time.time()

                        def on_progress(self, update):
                            incoming_status = str(getattr(update, "status", "") or "").strip()
                            incoming_phase = _phase_label_from_update(update, status=incoming_status or self.status_text)
                            incoming_step = getattr(update, "current_step", None)
                            incoming_total = getattr(update, "total_steps", None)
                            if time.time() - self._last_explicit_status_at <= 1.0 and len(self.phase_label) > 0 and len(incoming_phase) > 0 and incoming_phase.lower() != self.phase_label.lower() and not isinstance(incoming_step, int):
                                return
                            self.status_text = incoming_status or self.status_text or "Generating..."
                            if len(incoming_phase) > 0:
                                self.phase_label = incoming_phase
                            self.current_step = incoming_step
                            self.total_steps = incoming_total

                    callbacks = ChunkCallbacks()
                    last_html = ""
                    settings = {
                        "model_type": model_type,
                        "prompt": "generate a video",
                        "resolution": resolved_resolution or budget_resolution,
                        "num_inference_steps": 8,
                        "video_length": int(plan.requested_frames),
                        "sliding_window_size": 481,
                        "sliding_window_overlap": 1,
                        "force_fps": "control",
                        "video_prompt_type": "VG",
                        "audio_prompt_type": "K",
                        "image_prompt_type": "" if not plan.drop_first_frame else "V",
                        "denoising_strength": 1,
                        "video_guide": build_virtual_media_path(source_path, start_frame=plan.control_start_frame, end_frame=plan.control_end_frame, audio_track_no=selected_audio_track),
                        "video_guide_outpainting": "0 0 0 0",
                        "video_guide_outpainting_ratio": target_ratio,
                        "_api": {"return_media": True},
                    }
                    if plan.drop_first_frame:
                        settings["video_source"] = last_frame_path
                    _reset_live_chunk_status(state)
                    job = api_session.submit_task(settings, callbacks=callbacks)
                    active_job["job"] = job
                    next_status_refresh_at = 0.0
                    while not job.done:
                        now = time.monotonic()
                        if now >= next_status_refresh_at:
                            html_value = _render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), callbacks.phase_label, callbacks.status_text, continued=continued_mode, phase_current_step=callbacks.current_step, phase_total_steps=callbacks.total_steps, prefer_status_phase=True, **_timing_kwargs(callbacks.current_step, callbacks.total_steps))
                            next_status_refresh_at = now + STATUS_REFRESH_INTERVAL_SECONDS
                            if html_value != last_html:
                                last_html = html_value
                                yield _ui_update(html_value)
                        time.sleep(0.1)
                    try:
                        result = job.result()
                    finally:
                        if active_job.get("job") is job:
                            active_job["job"] = None
                    if not result.success:
                        if _job_was_stopped(result):
                            stopped = True
                            break
                        errors = list(result.errors or [])
                        raise gr.Error(str(errors[0] if errors else f"Chunk {chunk_index} failed."))
                    chunk_output_paths.extend(
                        str(Path(path).resolve())
                        for path in result.generated_files
                        if isinstance(path, str) and len(path.strip()) > 0 and str(Path(path).resolve()) not in chunk_output_paths
                    )
                    artifact = next((item for item in result.artifacts if item.video_tensor_uint8 is not None), None)
                    if artifact is None or not torch.is_tensor(artifact.video_tensor_uint8):
                        raise gr.Error(f"Chunk {chunk_index} completed without returned video tensor data.")
                    video_tensor_uint8 = artifact.video_tensor_uint8.detach().cpu()
                    artifact_frame_count = int(video_tensor_uint8.shape[1])
                    expected_frame_count = int(plan.requested_frames)
                    if artifact_frame_count < max(1, expected_frame_count - 1):
                        video_candidates = [path for path in result.generated_files if isinstance(path, str) and os.path.isfile(path) and str(Path(path).suffix).lower() in {".mp4", ".mkv", ".mov", ".avi"}]
                        if video_candidates:
                            decoded_tensor = _load_video_tensor_from_file(video_candidates[0])
                            decoded_frame_count = int(decoded_tensor.shape[1])
                            print(f"[Process Full Video] Chunk {chunk_index} artifact returned {artifact_frame_count} frame(s), generated file has {decoded_frame_count} frame(s)")
                            if decoded_frame_count >= max(1, expected_frame_count - 1):
                                video_tensor_uint8 = decoded_tensor
                                artifact_frame_count = decoded_frame_count
                    print(f"[Process Full Video] Chunk {chunk_index} tensor frames {artifact_frame_count} expected {expected_frame_count}")
                    chunk_width, chunk_height = _get_video_tensor_resolution(video_tensor_uint8)
                    chunk_resolution = f"{chunk_width}x{chunk_height}"
                    print(f"[Process Full Video] Chunk {chunk_index} returned {chunk_resolution}")
                    if len(resolved_resolution) == 0:
                        resolved_resolution = chunk_resolution
                        resolved_width = chunk_width
                        resolved_height = chunk_height
                    elif chunk_resolution != resolved_resolution:
                        raise gr.Error(f"Chunk {chunk_index} changed output resolution from {resolved_resolution} to {chunk_resolution}.")
                    skip_frames = 1 if plan.drop_first_frame else 0
                    remaining_unique_frames = requested_unique_frames - written_unique_frames
                    frames_to_write = min(remaining_unique_frames, int(video_tensor_uint8.shape[1]) - skip_frames)
                    if frames_to_write <= 0:
                        continue
                    if mux_process is None:
                        mux_process = _start_av_mux_process(ffmpeg_path, output_path_for_write, resolved_width, resolved_height, fps_float, self.server_config.get("video_output_codec", "libx264_8"), output_container, source_path, exact_start_seconds, selected_audio_track) if use_live_av_mux else _start_video_mux_process(ffmpeg_path, video_only_output_path, resolved_width, resolved_height, fps_float, self.server_config.get("video_output_codec", "libx264_8"), output_container)
                    last_frame_tensor = _write_video_chunk(mux_process, video_tensor_uint8, start_frame=skip_frames, frame_count=frames_to_write)
                    written_unique_frames += frames_to_write
                    last_frame_image = _frame_to_image(last_frame_tensor)
                    last_frame_image.save(last_frame_path, format="PNG")
                    preview_state["image"] = last_frame_image
                    completed_chunks += 1
                    chunk_output_paths = _delete_released_chunk_outputs(state, chunk_output_paths)
                    if chunk_index < len(plans):
                        yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Starting new Chunk", f"Chunk {completed_chunks} finished with {frames_to_write} written frame(s). Preparing next chunk...", continued=continued_mode, **_timing_kwargs()), ui_skip, str(time.time_ns()))
                    else:
                        yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Chunk Completed", f"Chunk {completed_chunks} finished with {frames_to_write} written frame(s).", continued=continued_mode, **_timing_kwargs()), ui_skip, str(time.time_ns()))
                if mux_process is None:
                    if stopped and resumed_unique_frames > 0:
                        _plugin_info(f"Processing was stopped before writing a new chunk. Kept existing output at {output_path}")
                        yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Stopped", "Stopped before a new chunk was written. Existing output kept.", continued=continued_mode, **_timing_kwargs()), output_path, ui_skip, start_enabled=True, abort_enabled=False)
                        return
                    if stopped:
                        _plugin_info("Processing was stopped before any output chunk was written.")
                        yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Stopped", "Stopped before any output chunk was written.", continued=continued_mode, **_timing_kwargs()), ui_skip, ui_skip, start_enabled=True, abort_enabled=False)
                        return
                    raise gr.Error("Processing completed without creating an output file.")
                finalizing_message = "Finalizing written output before merge..." if continuation_output_path and os.path.isfile(output_path_for_write) else "Finalizing written output..."
                yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Finalizing Output", finalizing_message, continued=continued_mode, **_timing_kwargs()), output_path if os.path.isfile(output_path) else ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=False)
                return_code, stderr, forced_termination = _finalize_mux_process(mux_process)
                mux_finished = True
                if forced_termination:
                    raise gr.Error("ffmpeg did not finalize the partial output in time.")
                if return_code != 0 and not (stopped and os.path.isfile(output_path_for_write if use_live_av_mux else video_only_output_path)):
                    raise gr.Error(stderr or "ffmpeg failed while assembling the processed video.")
                if use_live_av_mux and os.path.isfile(output_path_for_write) and os.path.getsize(output_path_for_write) <= 0:
                    os.remove(output_path_for_write)
                    raise gr.Error("ffmpeg created an empty continuation file.")
                if not use_live_av_mux and os.path.isfile(video_only_output_path):
                    yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Muxing Audio", "Muxing source audio into the written video segment...", continued=continued_mode, **_timing_kwargs()), output_path if os.path.isfile(output_path) else ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=False)
                    _mux_source_audio(ffmpeg_path, video_only_output_path, output_path_for_write, source_path, exact_start_seconds, written_unique_frames / fps_float, selected_audio_track)
                if continuation_output_path and os.path.isfile(output_path_for_write):
                    try:
                        yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Merging Continuation", "Merging the continued segment into the main output...", continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()), start_enabled=False, abort_enabled=False)
                        _concat_video_segments(ffmpeg_path, [output_path, output_path_for_write], output_path, self.server_config.get("video_output_codec", "libx264_8"), output_container, self.server_config.get("audio_output_codec", "aac_128"), segment_audio_trim_seconds=[0.0, resume_audio_trim_seconds], fps_float=fps_float, selected_audio_track_no=selected_audio_track)
                        merged_continuation = True
                    except Exception as exc:
                        raise gr.Error(f"Failed to finalize continued output. Existing output kept, and continuation was preserved at {continuation_output_path}. {exc}") from exc
                    if os.path.isfile(output_path_for_write):
                        os.remove(output_path_for_write)
                total_written_unique_frames = resumed_unique_frames + written_unique_frames
                metadata_target_path = output_path if merged_continuation or not continuation_output_path else output_path_for_write
                yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Writing Metadata", "Writing final output metadata...", continued=continued_mode, **_timing_kwargs()), metadata_target_path if os.path.isfile(metadata_target_path) else output_path, str(time.time_ns()), start_enabled=False, abort_enabled=False)
                _write_output_metadata(ffmpeg_path, metadata_target_path, {
                    "plugin": PlugIn_Name,
                    "process": process_name,
                    "source_video": source_path,
                    "source_segment": build_virtual_media_path(source_path, start_frame=start_frame, end_frame=start_frame + requested_unique_frames - 1, audio_track_no=selected_audio_track),
                    "target_ratio": target_ratio,
                    "resolution_budget": str(output_resolution),
                    "resolved_output_resolution": resolved_resolution,
                    "fps": float(fps_float),
                    "chunk_size_seconds": float(chunk_size_seconds or 0.0),
                    "chunk_frames": int(chunk_frames),
                    "chunks": int(total_chunks_display),
                    "written_unique_frames": int(total_written_unique_frames),
                    "start_seconds": float(start_frame / fps_float),
                    "end_seconds": float((start_frame + total_written_unique_frames) / fps_float) if stopped else float((start_frame + requested_unique_frames) / fps_float),
                    "audio_track_no": selected_audio_track,
                    "model_type": model_type,
                    "video_prompt_type": "VG",
                    "audio_prompt_type": "K",
                    "image_prompt_type": "V",
                    "prompt": "generate a video",
                    "continued_from_existing": resumed_unique_frames > 0,
                    "resume_start_frame": int(start_frame + resumed_unique_frames),
                    "stopped": stopped,
                    "creation_date": datetime.now().isoformat(timespec="seconds"),
                })
                chunk_output_paths = _delete_released_chunk_outputs(state, chunk_output_paths)
                if stopped:
                    stopped_output_path = output_path
                    if merged_continuation:
                        _plugin_info(f"Processing was stopped. Merged continued progress into {output_path}")
                        stop_message = f"Stopped after {total_written_unique_frames} frame(s). Continued progress was merged into the output."
                    elif continuation_output_path and os.path.isfile(output_path_for_write):
                        stopped_output_path = output_path_for_write
                        _plugin_info(f"Processing was stopped. Kept existing output at {output_path} and preserved continuation clip at {output_path_for_write}")
                        stop_message = f"Stopped after {total_written_unique_frames} frame(s). Existing output kept and continuation clip preserved."
                    else:
                        _plugin_info(f"Processing was stopped. Kept partial output at {output_path}")
                        stop_message = f"Stopped after {total_written_unique_frames} frame(s). Partial output kept."
                    yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Stopped", stop_message, continued=continued_mode, **_timing_kwargs()), stopped_output_path, ui_skip, start_enabled=True, abort_enabled=False)
                    return
                yield _ui_update(_render_chunk_status_html(total_chunks_display, total_chunks_display, total_chunks_display, "Completed", f"Completed {total_chunks_display} chunk(s).", continued=continued_mode, **_timing_kwargs()), output_path, ui_skip, start_enabled=True, abort_enabled=False)
                active_job["job"] = None
                if mux_process is not None and not mux_finished and mux_process.poll() is None:
                    try:
                        _finalize_mux_process(mux_process)
                    except Exception:
                        pass
                if mux_process is not None and not stopped and mux_process.returncode not in (0, None) and os.path.isfile(output_path_for_write):
                    os.remove(output_path_for_write)
                if os.path.isfile(video_only_output_path):
                    os.remove(video_only_output_path)
                shutil.rmtree(temp_dir, ignore_errors=True)
            except gr.Error as exc:
                active_job["job"] = None
                mux_process_local = locals().get("mux_process")
                output_path_for_write_local = locals().get("output_path_for_write")
                video_only_output_path_local = locals().get("video_only_output_path")
                temp_dir_local = locals().get("temp_dir")
                stopped_local = bool(locals().get("stopped"))
                mux_finished_local = bool(locals().get("mux_finished"))
                if mux_process_local is not None and not mux_finished_local and mux_process_local.poll() is None:
                    try:
                        _finalize_mux_process(mux_process_local)
                    except Exception:
                        pass
                if mux_process_local is not None and not stopped_local and isinstance(output_path_for_write_local, str) and mux_process_local.returncode not in (0, None) and os.path.isfile(output_path_for_write_local):
                    os.remove(output_path_for_write_local)
                if isinstance(video_only_output_path_local, str) and os.path.isfile(video_only_output_path_local):
                    os.remove(video_only_output_path_local)
                if isinstance(temp_dir_local, str):
                    shutil.rmtree(temp_dir_local, ignore_errors=True)
                if started_ui:
                    total_chunks_value = max(1, int(locals().get("total_chunks_display", 1) or 1))
                    completed_value = max(0, min(int(locals().get("completed_chunks", 0) or 0), total_chunks_value))
                    current_value = max(1, min(completed_value + 1, total_chunks_value))
                    continued_value = bool(int(locals().get("resumed_unique_frames", 0) or 0) > 0)
                    status_message = str(exc).strip() or "Processing failed."
                    output_value = output_path if isinstance(locals().get("output_path"), str) and os.path.isfile(locals()["output_path"]) else ui_skip
                    yield _ui_update(_render_chunk_status_html(total_chunks_value, completed_value, current_value, "Error", status_message, continued=continued_value), output_value, ui_skip, start_enabled=True, abort_enabled=False)
                raise
            except BaseException as exc:
                active_job["job"] = None
                mux_process_local = locals().get("mux_process")
                output_path_for_write_local = locals().get("output_path_for_write")
                video_only_output_path_local = locals().get("video_only_output_path")
                temp_dir_local = locals().get("temp_dir")
                stopped_local = bool(locals().get("stopped"))
                mux_finished_local = bool(locals().get("mux_finished"))
                if mux_process_local is not None and not mux_finished_local and mux_process_local.poll() is None:
                    try:
                        _finalize_mux_process(mux_process_local)
                    except Exception:
                        pass
                if mux_process_local is not None and not stopped_local and isinstance(output_path_for_write_local, str) and mux_process_local.returncode not in (0, None) and os.path.isfile(output_path_for_write_local):
                    os.remove(output_path_for_write_local)
                if isinstance(video_only_output_path_local, str) and os.path.isfile(video_only_output_path_local):
                    os.remove(video_only_output_path_local)
                if isinstance(temp_dir_local, str):
                    shutil.rmtree(temp_dir_local, ignore_errors=True)
                if started_ui:
                    total_chunks_value = max(1, int(locals().get("total_chunks_display", 1) or 1))
                    completed_value = max(0, min(int(locals().get("completed_chunks", 0) or 0), total_chunks_value))
                    current_value = max(1, min(completed_value + 1, total_chunks_value))
                    continued_value = bool(int(locals().get("resumed_unique_frames", 0) or 0) > 0)
                    status_message = str(exc).strip() or exc.__class__.__name__
                    output_value = output_path if isinstance(locals().get("output_path"), str) and os.path.isfile(locals()["output_path"]) else ui_skip
                    yield _ui_update(_render_chunk_status_html(total_chunks_value, completed_value, current_value, "Error", status_message, continued=continued_value), output_value, ui_skip, start_enabled=True, abort_enabled=False)
                raise

        def stop_process():
            job = active_job.get("job")
            if job is not None and not job.done:
                _request_job_stop(job)
                _plugin_info("Stopping current processing job...")
                return gr.update(value="Start Process", interactive=False), gr.update(value="Stop", interactive=False)
            return gr.update(value="Start Process", interactive=True), gr.update(value="Stop", interactive=False)

        with gr.Column():
            with gr.Row():
                gr.Markdown("This PlugIn is more or less a *Super Sliding Windows* mode but without the *RAM restrictions* and no risk to explode the *Video Gallery* with huge files. You can stop a Process and Resume it later.")
                process_name = gr.Dropdown(PROCESS_CHOICES, value="outpaint_video", label="Process")
            with gr.Row():
                source_path = gr.Textbox(label="Source Video Path File", value=DEFAULT_SOURCE_PATH, scale=3)
                output_path = gr.Textbox(label="Output File Path File (None for auto, Full Name or Target Folder)", value=DEFAULT_OUTPUT_PATH, scale=3)
                continue_enabled = gr.Checkbox(label="Continue", value=True, elem_classes="cbx_bottom", scale=1)
            with gr.Row():
                output_resolution = gr.Dropdown([("1080p", "1080p"), ("900p", "900p"), ("720p", "720p"), ("540p", "540p")], value="720p", label="Output Resolution")
                target_ratio = gr.Dropdown(RATIO_CHOICES, value="4:3", label="Target Ratio")
                chunk_size_seconds = gr.Number(label="Chunk Size (seconds)", value=10.0, precision=2)
            with gr.Row():
                start_seconds = gr.Textbox(label="Start (s/MM:SS/HH:MM:SS)", value="", placeholder="seconds, MM:SS, or HH:MM:SS")
                end_seconds = gr.Textbox(label="End (s/MM:SS/HH:MM:SS)", value="", placeholder="seconds, MM:SS, or HH:MM:SS")
                source_audio_track = gr.Number(label="Source Audio Track (1-based)", value=1, precision=0, minimum=1)
            with gr.Row():
                start_btn = gr.Button("Start Process")
                abort_btn = gr.Button("Stop", interactive=False)
            status_html = gr.HTML(value=_render_chunk_status_html(1, 0, 1, "Idle", "Waiting to start..."))
            preview_image = gr.Image(label="Last Frame Preview", type="pil")
            output_file = gr.HTML(value=_render_output_file_html(""))
            preview_refresh = gr.Textbox(value="", visible=False)

        start_btn.click(
            fn=start_process,
            inputs=[self.state, process_name, source_path, output_path, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, start_seconds, end_seconds],
            outputs=[status_html, output_file, preview_refresh, start_btn, abort_btn],
            queue=False,
            show_progress="hidden",
            show_progress_on=[],
        )
        preview_refresh.change(fn=refresh_preview, inputs=[preview_refresh], outputs=[preview_image], queue=False, show_progress="hidden")
        abort_btn.click(fn=stop_process, outputs=[start_btn, abort_btn], queue=False, show_progress="hidden")
