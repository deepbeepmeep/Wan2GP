from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import ffmpeg
from PIL import Image

from shared.deepy.media_registry import detect_media_type
from shared.utils.video_decode import resolve_media_binary


TEXT_MAX_CHARS = 16000


def _extension_filter(value: Any) -> set[str]:
    if value is None:
        return set()
    values = re.split(r"[,;\s]+", value) if isinstance(value, str) else value
    return {f".{str(item).strip().lower().lstrip('.')}" for item in values if str(item).strip()}


def list_files(path: str, extensions: Any = None) -> dict[str, Any]:
    directory = Path(str(path or "").strip()).expanduser().resolve()
    if not directory.is_dir():
        return {"status": "error", "path": str(directory), "files": [], "count": 0, "error": "Path is not an existing directory."}
    allowed = _extension_filter(extensions)
    files = [
        {"filename": item.name, "extension": item.suffix.lower(), "size_bytes": item.stat().st_size, "path": str(item.resolve())}
        for item in sorted(directory.iterdir(), key=lambda entry: entry.name.casefold())
        if item.is_file() and (not allowed or item.suffix.lower() in allowed)
    ]
    return {"status": "done", "path": str(directory), "extensions": sorted(allowed), "files": files, "count": len(files), "error": ""}


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(value: Any) -> float | None:
    text = str(value or "").strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = _float(denominator)
        return None if not denominator_value else float(numerator) / denominator_value
    return _float(text)


def _probe_media(path: Path, media_type: str) -> dict[str, Any]:
    probe = ffmpeg.probe(str(path), cmd=resolve_media_binary("ffprobe") or "ffprobe")
    streams = probe.get("streams", [])
    format_info = probe.get("format", {})
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    duration = _float(format_info.get("duration"))
    result = {
        "status": "done",
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "file_type": media_type,
        "duration_seconds": duration,
        "has_audio": bool(audio_streams),
        "audio_track_count": len(audio_streams),
        "error": "",
    }
    if media_type == "video":
        video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
        fps = _rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
        frame_count = _integer(video_stream.get("nb_frames"))
        if frame_count is None and fps is not None and duration is not None:
            frame_count = int(round(fps * duration))
        width, height = _integer(video_stream.get("width")), _integer(video_stream.get("height"))
        result.update({"width": width, "height": height, "resolution": None if width is None or height is None else f"{width}x{height}", "frame_count": frame_count, "fps": fps})
    else:
        audio_stream = audio_streams[0] if audio_streams else {}
        result.update({"sample_rate": _integer(audio_stream.get("sample_rate")), "channels": _integer(audio_stream.get("channels"))})
    return result


def _query_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        frame_count = int(getattr(image, "n_frames", 1) or 1)
        duration = sum(float(image.seek(index) or image.info.get("duration", 0) or 0) for index in range(frame_count)) / 1000 if frame_count > 1 else None
    return {"status": "done", "path": str(path), "filename": path.name, "size_bytes": path.stat().st_size, "file_type": "image", "width": width, "height": height, "resolution": f"{width}x{height}", "frame_count": frame_count, "fps": None if not duration else frame_count / duration, "duration_seconds": duration, "has_audio": False, "audio_track_count": 0, "error": ""}


def query_file(path: str) -> dict[str, Any]:
    file_path = Path(str(path or "").strip()).expanduser().resolve()
    if not file_path.is_file():
        return {"status": "error", "path": str(file_path), "error": "Path is not an existing file."}
    try:
        media_type = detect_media_type(str(file_path))
        if media_type == "image":
            return _query_image(file_path)
        if media_type in {"video", "audio"}:
            return _probe_media(file_path, media_type)
        with file_path.open("r", encoding="utf-8") as reader:
            text = reader.read(TEXT_MAX_CHARS + 1)
        if len(text) > TEXT_MAX_CHARS:
            return {"status": "error", "path": str(file_path), "size_bytes": file_path.stat().st_size, "error": f"Text file exceeds the {TEXT_MAX_CHARS:,}-character limit."}
        return {"status": "done", "path": str(file_path), "filename": file_path.name, "size_bytes": file_path.stat().st_size, "file_type": "text", "text": text, "character_count": len(text), "error": ""}
    except (OSError, UnicodeError, ValueError, ffmpeg.Error) as exc:
        return {"status": "error", "path": str(file_path), "size_bytes": file_path.stat().st_size, "error": str(exc) or "Unable to inspect file."}


__all__ = ["TEXT_MAX_CHARS", "list_files", "query_file"]
