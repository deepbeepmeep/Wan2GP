from __future__ import annotations

import base64
import hashlib
import math
import os
import shutil
import subprocess
from collections import OrderedDict
from datetime import datetime

import ffmpeg

from shared.ffmpeg_setup import download_ffmpeg
from shared.utils.audio_video import get_mp4_audio_codec_settings
from shared.utils.utils import get_video_frame, get_video_info
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
_THUMB_DATA_URL_CACHE: OrderedDict[str, str] = OrderedDict()
_THUMB_CACHE_MAX = 64


def _get_ffmpeg_path() -> str:
    download_ffmpeg()
    ffmpeg_path = os.environ.get("FFMPEG_BINARY", "") or shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is not available.")
    return ffmpeg_path


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run([_get_ffmpeg_path(), "-y", "-v", "error", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg command failed").strip())


def _format_timestamp(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    seconds = float(seconds)
    if seconds < 0:
        raise ValueError("Time values must be >= 0.")
    return f"{seconds:.6f}".rstrip("0").rstrip(".")


def _resolve_segment_args(start_time: float | int | None = None, end_time: float | int | None = None, duration: float | int | None = None) -> tuple[str | None, str | None]:
    if end_time is not None and duration is not None:
        raise ValueError("Specify either end_time or duration, not both.")
    start_str = _format_timestamp(0 if start_time is None else start_time)
    end_str = _format_timestamp(end_time)
    duration_str = _format_timestamp(duration)
    if end_str is not None and float(end_str) <= float(start_str):
        raise ValueError("end_time must be greater than start_time.")
    if duration_str is not None and float(duration_str) <= 0:
        raise ValueError("duration must be > 0.")
    return start_str, end_str if duration_str is None else duration_str


def get_video_container_extension(container: str | None) -> str:
    container = str(container or "mp4").strip().lower() or "mp4"
    return ".mkv" if container == "mkv" else ".mp4"


def get_audio_standalone_extension(codec_key: str | None) -> str:
    codec_key = str(codec_key or "wav").strip().lower() or "wav"
    if codec_key == "mp3":
        codec_key = "mp3_192"
    return ".wav" if codec_key == "wav" else ".mp3"


def _get_video_encode_args(codec_key: str | None, container: str | None) -> list[str]:
    codec_key = str(codec_key or "libx264_8").strip().lower() or "libx264_8"
    container = str(container or "mp4").strip().lower() or "mp4"
    if codec_key == "libx264_8":
        return ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"]
    if codec_key == "libx264_10":
        return ["-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p"]
    if codec_key == "libx265_28":
        return ["-c:v", "libx265", "-crf", "28", "-pix_fmt", "yuv420p", "-x265-params", "log-level=none"]
    if codec_key == "libx265_8":
        return ["-c:v", "libx265", "-crf", "8", "-pix_fmt", "yuv420p", "-x265-params", "log-level=none"]
    if codec_key == "libx264_lossless":
        if container == "mkv":
            return ["-c:v", "ffv1", "-pix_fmt", "rgb24"]
        return ["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"]
    return ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"]


def _get_mp4_audio_encode_args(codec_key: str | None) -> list[str]:
    settings = get_mp4_audio_codec_settings(codec_key)
    args = ["-c:a", settings["codec"]]
    if settings.get("bitrate"):
        args += ["-b:a", settings["bitrate"]]
    return args


def _get_standalone_audio_encode_args(codec_key: str | None) -> list[str]:
    codec_key = str(codec_key or "wav").strip().lower() or "wav"
    if codec_key == "mp3":
        codec_key = "mp3_192"
    if codec_key == "wav":
        return ["-c:a", "pcm_s16le"]
    bitrate = {"mp3_128": "128k", "mp3_192": "192k", "mp3_320": "320k"}.get(codec_key, "192k")
    return ["-c:a", "libmp3lame", "-b:a", bitrate]


def has_video_extension(path: str) -> bool:
    return os.path.splitext(str(path or "").strip())[1].lower() in VIDEO_EXTENSIONS


def get_video_thumbnail_data_url(video_path: str) -> str:
    video_path = os.path.abspath(os.path.normpath(str(video_path or "").strip()))
    if len(video_path) == 0 or not os.path.isfile(video_path) or not has_video_extension(video_path):
        return ""
    stat = os.stat(video_path)
    cache_key = hashlib.sha1(f"{os.path.abspath(video_path)}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")).hexdigest()
    cached = _THUMB_DATA_URL_CACHE.get(cache_key, "")
    if len(cached) > 0:
        _THUMB_DATA_URL_CACHE.move_to_end(cache_key)
        return cached
    result = subprocess.run(
        [
            _get_ffmpeg_path(),
            "-v",
            "error",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-vf",
            "scale=320:-1:force_original_aspect_ratio=decrease",
            "-q:v",
            "4",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ],
        capture_output=True,
    )
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError((result.stderr.decode("utf-8", errors="ignore").strip() or "ffmpeg thumbnail extraction failed").strip())
    data_url = f"data:image/jpeg;base64,{base64.b64encode(result.stdout).decode('ascii')}"
    _THUMB_DATA_URL_CACHE[cache_key] = data_url
    while len(_THUMB_DATA_URL_CACHE) > _THUMB_CACHE_MAX:
        _THUMB_DATA_URL_CACHE.popitem(last=False)
    return data_url


def merge_videos(first_video: str, second_video: str, output_path: str | None = None, *, video_codec: str | None = None, video_container: str | None = None, audio_codec: str | None = None) -> str:
    first_video = os.path.normpath(str(first_video or "").strip())
    second_video = os.path.normpath(str(second_video or "").strip())
    if not os.path.isfile(first_video):
        raise FileNotFoundError(f"First video not found: {first_video}")
    if not os.path.isfile(second_video):
        raise FileNotFoundError(f"Second video not found: {second_video}")
    first_info = _probe_video_stream(first_video)
    width, height = int(first_info["width"]), int(first_info["height"])
    first_has_audio = _has_audio_stream(first_video)
    second_has_audio = _has_audio_stream(second_video)
    output_path = output_path or _default_merged_output_path(first_video, second_video)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    filters = [
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[v0]",
        f"[1:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[v1]",
    ]
    maps = ["-map", "[v]"]
    if first_has_audio and second_has_audio:
        filters.append("[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]")
        maps += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    else:
        filters.append("[v0][v1]concat=n=2:v=1:a=0[v]")
    cmd = [
        "-i",
        first_video,
        "-i",
        second_video,
        "-filter_complex",
        ";".join(filters),
        *maps,
        *_get_video_encode_args(video_codec, video_container),
        *(["-movflags", "+faststart"] if str(video_container or "mp4").strip().lower() == "mp4" else []),
        output_path,
    ]
    if first_has_audio and second_has_audio:
        insert_at = cmd.index(output_path)
        cmd[insert_at:insert_at] = _get_mp4_audio_encode_args(audio_codec)
    _run_ffmpeg(cmd)
    return output_path


def extract_video(source_path: str, output_path: str, start_time: float | int = 0, end_time: float | int | None = None, duration: float | int | None = None, *, video_codec: str | None = None, video_container: str | None = None, audio_codec: str | None = None) -> str:
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video not found: {source_path}")
    start_str, end_or_duration_str = _resolve_segment_args(start_time, end_time, duration)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = ["-ss", start_str, "-i", source_path]
    if duration is not None:
        cmd += ["-t", end_or_duration_str]
    elif end_time is not None:
        cmd += ["-to", end_or_duration_str]
    cmd += ["-map", "0:v:0", "-map", "0:a?"]
    cmd += _get_video_encode_args(video_codec, video_container)
    cmd += _get_mp4_audio_encode_args(audio_codec)
    if str(video_container or "mp4").strip().lower() == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [output_path]
    _run_ffmpeg(cmd)
    return output_path


def extract_audio(source_path: str, output_path: str, start_time: float | int | None = None, end_time: float | int | None = None, duration: float | int | None = None, *, audio_codec: str | None = None) -> str:
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Media not found: {source_path}")
    if not _has_audio_stream(source_path):
        raise RuntimeError(f"No audio stream found in {source_path}")
    start_str, end_or_duration_str = _resolve_segment_args(start_time, end_time, duration)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = []
    if start_str is not None:
        cmd += ["-ss", start_str]
    cmd += ["-i", source_path]
    if duration is not None:
        cmd += ["-t", end_or_duration_str]
    elif end_time is not None:
        cmd += ["-to", end_or_duration_str]
    cmd += ["-map", "0:a:0", "-vn", *_get_standalone_audio_encode_args(audio_codec), output_path]
    _run_ffmpeg(cmd)
    return output_path


def extract_video_frame(source_path: str, output_path: str, *, frame_no: int | None = None, time_seconds: float | int | None = None) -> str:
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video not found: {source_path}")
    if time_seconds is None and frame_no is None:
        raise ValueError("frame_no or time_seconds is required.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_no = resolve_video_frame_no(source_path, frame_no=frame_no, time_seconds=time_seconds)
    image = get_video_frame(source_path, frame_no, return_last_if_missing=True, return_PIL=True).convert("RGB")
    image.save(output_path)
    if not os.path.isfile(output_path):
        raise RuntimeError("Frame extraction did not produce an output image.")
    return output_path


def mute_video(source_path: str, output_path: str) -> str:
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video not found: {source_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _run_ffmpeg(["-i", source_path, "-map", "0:v:0", "-c:v", "copy", "-an", output_path])
    return output_path


def replace_audio(video_path: str, audio_path: str, output_path: str, *, audio_codec: str | None = None) -> str:
    video_path = os.path.normpath(str(video_path or "").strip())
    audio_path = os.path.normpath(str(audio_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not _has_audio_stream(audio_path):
        raise RuntimeError(f"No audio stream found in {audio_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _run_ffmpeg(["-i", video_path, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *_get_mp4_audio_encode_args(audio_codec), "-shortest", output_path])
    return output_path


def resize_crop_video(source_path: str, output_path: str, *, width: int | None = None, height: int | None = None, crop_left: float = 0, crop_top: float = 0, crop_right: float = 0, crop_bottom: float = 0, crop_unit: str = "pixels", video_codec: str | None = None, video_container: str | None = None, audio_codec: str | None = None) -> str:
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video not found: {source_path}")
    stream = _probe_video_stream(source_path)
    source_width = int(stream["width"])
    source_height = int(stream["height"])
    crop_unit = str(crop_unit or "pixels").strip().lower() or "pixels"
    if crop_unit not in {"pixels", "percent"}:
        raise ValueError("crop_unit must be 'pixels' or 'percent'.")
    def resolve_crop(value: float, total: int) -> int:
        value = float(value or 0)
        if value < 0:
            raise ValueError("Crop values must be >= 0.")
        return round(total * value / 100.0) if crop_unit == "percent" else round(value)
    left = resolve_crop(crop_left, source_width)
    right = resolve_crop(crop_right, source_width)
    top = resolve_crop(crop_top, source_height)
    bottom = resolve_crop(crop_bottom, source_height)
    cropped_width = source_width - left - right
    cropped_height = source_height - top - bottom
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError("Crop values remove the whole video frame.")
    filters = [f"crop={cropped_width}:{cropped_height}:{left}:{top}"]
    if width is not None or height is not None:
        scale_w = int(width) if width is not None else -1
        scale_h = int(height) if height is not None else -1
        if width is not None and int(width) <= 0 or height is not None and int(height) <= 0:
            raise ValueError("width and height must be > 0 when provided.")
        filters.append(f"scale={scale_w}:{scale_h}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = ["-i", source_path, "-vf", ",".join(filters), "-map", "0:v:0", "-map", "0:a?"]
    cmd += _get_video_encode_args(video_codec, video_container)
    cmd += _get_mp4_audio_encode_args(audio_codec)
    if str(video_container or "mp4").strip().lower() == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [output_path]
    _run_ffmpeg(cmd)
    return output_path


def get_media_duration(path: str) -> float | None:
    probe = ffmpeg.probe(path)
    format_info = probe.get("format", {}) or {}
    duration = format_info.get("duration", None)
    try:
        return None if duration is None else float(duration)
    except Exception:
        return None


def get_precise_video_fps(path: str) -> float | None:
    try:
        stream = _probe_video_stream(path)
    except Exception:
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = str(stream.get(key, "") or "").strip()
        if len(value) == 0 or value in {"0/0", "N/A"}:
            continue
        if "/" in value:
            num, den = value.split("/", 1)
            try:
                num = float(num)
                den = float(den)
                if den != 0:
                    return num / den
            except Exception:
                continue
        else:
            try:
                return float(value)
            except Exception:
                continue
    return None


def resolve_video_frame_no(path: str, *, frame_no: int | None = None, time_seconds: float | int | None = None) -> int:
    if time_seconds is None and frame_no is None:
        raise ValueError("frame_no or time_seconds is required.")
    fps, _width, _height, frame_count = get_video_info(path)
    precise_fps = get_precise_video_fps(path)
    effective_fps = float(precise_fps) if precise_fps is not None and precise_fps > 0 else float(fps)
    max_frame = max(0, int(frame_count) - 1)
    if frame_no is None:
        if effective_fps <= 0:
            frame_no = 0
        else:
            frame_no = int(math.floor(max(0.0, float(time_seconds or 0.0)) * effective_fps + 1e-9))
    frame_no = int(frame_no)
    if frame_no < 0:
        frame_no = 0
    if frame_no > max_frame:
        frame_no = max_frame
    return frame_no


def has_audio_stream(path: str) -> bool:
    return _has_audio_stream(path)


def has_video_stream(path: str) -> bool:
    try:
        _probe_video_stream(path)
        return True
    except Exception:
        return False


def _probe_video_stream(video_path: str) -> dict:
    probe = ffmpeg.probe(video_path)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise RuntimeError(f"No video stream found in {video_path}")


def _has_audio_stream(video_path: str) -> bool:
    probe = ffmpeg.probe(video_path)
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))


def _default_merged_output_path(first_video: str, second_video: str) -> str:
    first_base = os.path.splitext(os.path.basename(first_video))[0]
    second_base = os.path.splitext(os.path.basename(second_video))[0]
    output_dir = os.path.dirname(first_video) or "outputs"
    timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
    return os.path.join(output_dir, f"{timestamp}_merged_{first_base}_{second_base}.mp4")
