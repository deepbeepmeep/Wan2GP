from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
from typing import Any


VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm"})


class VideoContractError(ValueError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    extension: str
    content_type: str
    width: int
    height: int
    frame_count: int | None
    fps: float | None
    duration_seconds: float | None
    has_audio: bool
    audio_duration_seconds: float | None


@dataclass(frozen=True)
class VideoArtifactContract:
    expected_frame_count: int | None = None
    expected_fps: float | None = None
    fps_tolerance: float = 0.01
    expected_duration_seconds: float | None = None
    duration_tolerance_seconds: float = 0.15
    require_audio: bool = False
    expected_width: int | None = None
    expected_height: int | None = None
    require_thumbnail: bool = False
    thumbnail_path: str | None = None
    allowed_extensions: frozenset[str] = VIDEO_EXTENSIONS


def probe_video_metadata(path: str | Path) -> VideoMetadata:
    video_path = Path(path)
    if not video_path.exists() or not video_path.is_file():
        raise VideoContractError(f"video artifact does not exist: {video_path}")

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise VideoContractError(f"ffprobe failed for {video_path}: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VideoContractError(f"ffprobe returned malformed JSON for {video_path}") from exc

    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not isinstance(video_stream, dict):
        raise VideoContractError(f"ffprobe found no video stream for {video_path}")
    audio_streams = [stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"]
    format_payload = payload.get("format") if isinstance(payload.get("format"), dict) else {}

    duration = _float_or_none(format_payload.get("duration")) or _float_or_none(video_stream.get("duration"))
    audio_duration = _first_float(audio_streams, "duration")
    return VideoMetadata(
        path=str(video_path),
        extension=video_path.suffix.lower(),
        content_type=_content_type_for_extension(video_path.suffix.lower()),
        width=_int_or_zero(video_stream.get("width")),
        height=_int_or_zero(video_stream.get("height")),
        frame_count=_frame_count(video_stream, duration),
        fps=_rate_to_float(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        duration_seconds=duration,
        has_audio=bool(audio_streams),
        audio_duration_seconds=audio_duration,
    )


def validate_video_artifact(path: str | Path, contract: VideoArtifactContract) -> VideoMetadata:
    video_path = Path(path)
    extension = video_path.suffix.lower()
    if extension not in contract.allowed_extensions:
        raise VideoContractError(f"video artifact extension {extension!r} is not allowed")

    metadata = probe_video_metadata(video_path)
    if contract.expected_frame_count is not None and metadata.frame_count != contract.expected_frame_count:
        raise VideoContractError(
            f"frame count mismatch: expected {contract.expected_frame_count}, got {metadata.frame_count}"
        )
    if contract.expected_fps is not None:
        if metadata.fps is None or abs(metadata.fps - contract.expected_fps) > contract.fps_tolerance:
            raise VideoContractError(f"fps mismatch: expected {contract.expected_fps}, got {metadata.fps}")
    if contract.expected_duration_seconds is not None:
        if metadata.duration_seconds is None or abs(metadata.duration_seconds - contract.expected_duration_seconds) > contract.duration_tolerance_seconds:
            raise VideoContractError(
                f"duration mismatch: expected {contract.expected_duration_seconds}, got {metadata.duration_seconds}"
            )
    if contract.require_audio and not metadata.has_audio:
        raise VideoContractError("audio stream is required but missing")
    if contract.require_audio and metadata.audio_duration_seconds is not None and metadata.duration_seconds is not None:
        if abs(metadata.audio_duration_seconds - metadata.duration_seconds) > contract.duration_tolerance_seconds:
            raise VideoContractError(
                f"audio duration mismatch: video={metadata.duration_seconds}, audio={metadata.audio_duration_seconds}"
            )
    if contract.expected_width is not None and metadata.width != contract.expected_width:
        raise VideoContractError(f"width mismatch: expected {contract.expected_width}, got {metadata.width}")
    if contract.expected_height is not None and metadata.height != contract.expected_height:
        raise VideoContractError(f"height mismatch: expected {contract.expected_height}, got {metadata.height}")
    if contract.require_thumbnail:
        if not contract.thumbnail_path:
            raise VideoContractError("thumbnail is required but no thumbnail_path was provided")
        thumbnail_path = Path(contract.thumbnail_path)
        if not thumbnail_path.exists() or not thumbnail_path.is_file():
            raise VideoContractError(f"thumbnail artifact does not exist: {thumbnail_path}")
    return metadata


def _frame_count(video_stream: dict[str, Any], duration: float | None) -> int | None:
    nb_frames = _int_or_none(video_stream.get("nb_frames"))
    if nb_frames is not None:
        return nb_frames
    fps = _rate_to_float(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    if fps is not None and duration is not None:
        return round(fps * duration)
    return None


def _rate_to_float(value: Any) -> float | None:
    if value in (None, "", "0/0"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    return float(text)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _first_float(streams: list[dict[str, Any]], key: str) -> float | None:
    for stream in streams:
        value = _float_or_none(stream.get(key))
        if value is not None:
            return value
    return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    return int(value)


def _int_or_zero(value: Any) -> int:
    return _int_or_none(value) or 0


def _content_type_for_extension(extension: str) -> str:
    if extension == ".webm":
        return "video/webm"
    if extension == ".mov":
        return "video/quicktime"
    return "video/mp4"
