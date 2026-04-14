from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VirtualMediaSpec:
    source_path: str
    start_frame: int = 0
    end_frame: int | None = None
    audio_track_no: int | None = None
    extras: tuple[tuple[str, str], ...] = ()

    def as_suffix_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        if self.start_frame > 0:
            items.append(("start_frame", str(int(self.start_frame))))
        if self.end_frame is not None:
            items.append(("end_frame", str(int(self.end_frame))))
        if self.audio_track_no is not None:
            items.append(("audio_track_no", str(int(self.audio_track_no))))
        items.extend(list(self.extras))
        return items


def parse_virtual_media_path(value: Any) -> VirtualMediaSpec | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if "|" not in text:
        return None
    source_path, suffix = text.split("|", 1)
    source_path = source_path.strip()
    if len(source_path) == 0:
        return None
    values: dict[str, str] = {}
    extras: list[tuple[str, str]] = []
    for raw_item in suffix.split(","):
        item = raw_item.strip()
        if len(item) == 0:
            continue
        key, sep, raw_value = item.partition("=")
        key = key.strip().lower()
        if sep == "":
            extras.append((item, ""))
            continue
        value_text = raw_value.strip()
        if key in ("start_frame", "end_frame", "audio_track_no"):
            values[key] = value_text
        else:
            extras.append((key, value_text))
    return VirtualMediaSpec(
        source_path=source_path,
        start_frame=max(0, _parse_int(values.get("start_frame"), 0)),
        end_frame=_parse_optional_int(values.get("end_frame")),
        audio_track_no=_parse_optional_int(values.get("audio_track_no")),
        extras=tuple(extras),
    )


def strip_virtual_media_suffix(value: Any) -> Any:
    spec = parse_virtual_media_path(value)
    return spec.source_path if spec is not None else value


def build_virtual_media_path(
    source_path: str,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    audio_track_no: int | None = None,
    extras: dict[str, Any] | None = None,
) -> str:
    base_path = str(source_path or "").strip()
    if len(base_path) == 0:
        return base_path
    parts: list[str] = []
    if start_frame is not None:
        parts.append(f"start_frame={int(start_frame)}")
    if end_frame is not None:
        parts.append(f"end_frame={int(end_frame)}")
    if audio_track_no is not None:
        parts.append(f"audio_track_no={int(audio_track_no)}")
    for key, value in (extras or {}).items():
        key_text = str(key or "").strip()
        value_text = str(value or "").strip()
        if len(key_text) == 0 or len(value_text) == 0:
            continue
        parts.append(f"{key_text}={value_text}")
    return base_path if len(parts) == 0 else f"{base_path}|{','.join(parts)}"


def replace_virtual_media_source(value: Any, source_path: str) -> Any:
    spec = parse_virtual_media_path(value)
    if spec is None:
        return source_path
    extras = dict(spec.extras)
    return build_virtual_media_path(
        source_path,
        start_frame=spec.start_frame if spec.start_frame > 0 else None,
        end_frame=spec.end_frame,
        audio_track_no=spec.audio_track_no,
        extras=extras,
    )


def clamp_virtual_frame_range(spec: VirtualMediaSpec | None, total_frames: int) -> tuple[int, int | None]:
    if spec is None or total_frames <= 0:
        return 0, None
    start_frame = max(0, min(int(spec.start_frame), total_frames - 1))
    if spec.end_frame is None:
        return start_frame, total_frames - 1
    end_frame = max(start_frame, min(int(spec.end_frame), total_frames - 1))
    return start_frame, end_frame


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(str(value or "").strip())
    except (TypeError, ValueError):
        return default


def _parse_optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if len(text) == 0:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None
