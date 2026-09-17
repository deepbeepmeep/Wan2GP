from __future__ import annotations

import os
from typing import Any

from shared.deepy import media_registry
from shared.deepy.engine import get_or_create_assistant_session
from shared.utils.utils import get_video_info

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
_VIDEO_EXTENSIONS = {".mkv", ".mov", ".mp4", ".webm"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".flac", ".m4a", ".ogg", ".wma", ".opus"}


class Gallery:
    def __init__(self, deps, state: dict[str, Any]):
        self._deps = deps
        self._state = state
        self._session = get_or_create_assistant_session(state)
        self._record_views = {}

    def _gen(self) -> dict[str, Any]:
        return self._deps.get_gen_info(self._state)

    def _detect_media_type(self, path: str) -> str:
        ext = os.path.splitext(str(path or "").strip())[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            return "image"
        if ext in _VIDEO_EXTENSIONS:
            return "video"
        if ext in _AUDIO_EXTENSIONS:
            return "audio"
        return "any"

    def _resolve_lists(self, audio_only: bool) -> tuple[list[Any], list[Any]]:
        gen = self._gen()
        if audio_only:
            return gen["audio_file_list"], gen["audio_file_settings_list"]
        return gen["file_list"], gen["file_settings_list"]

    def _select_index(self, index: int, audio_only: bool) -> None:
        gen = self._gen()
        file_list, _file_settings_list = self._resolve_lists(audio_only)
        if len(file_list) == 0:
            index = -1
        else:
            index = max(0, min(int(index), len(file_list) - 1))
        if audio_only:
            gen["audio_selected"] = index
            gen["audio_last_selected"] = (index + 1) >= len(file_list)
            gen["current_gallery_source"] = "audio"
            gen["selected_video_time"] = None
        else:
            gen["selected"] = index
            gen["last_selected"] = (index + 1) >= len(file_list)
            gen["current_gallery_source"] = "video"
            selected_path = str(file_list[index] or "").strip() if 0 <= index < len(file_list) else ""
            gen["selected_video_time"] = 0.0 if self._detect_media_type(selected_path) == "video" else None

    def _register(self, path: str, settings: dict[str, Any] | None = None) -> dict[str, Any] | None:
        settings = settings if isinstance(settings, dict) else None
        client_id = "" if settings is None else str(settings.get("client_id", "") or "").strip()
        source = "deepy" if client_id.startswith("ai_") else "wangp"
        return media_registry.register_media(self._session, path, settings=settings, source=source, client_id=client_id)

    def add_path(self, raw_path: str, preferred_type: str = "any") -> dict[str, Any]:
        path = os.path.abspath(os.path.normpath(str(raw_path or "").strip().strip('"')))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        detected_type = self._detect_media_type(path)
        if detected_type == "any":
            raise ValueError("Unsupported media type. Use an image, video, or audio file.")
        preferred_type = str(preferred_type or "any").strip().lower() or "any"
        if preferred_type != "any" and detected_type != preferred_type:
            raise ValueError(f"Expected a {preferred_type} file, got a {detected_type} file.")
        configs, _any_image_or_video, any_audio = self._deps.get_settings_from_file(self._state, path, False, False, False)
        audio_only = detected_type == "audio" or bool(any_audio)
        file_list, file_settings_list = self._resolve_lists(audio_only)
        normalized_path = os.path.normcase(path)
        for index, existing_path in enumerate(file_list):
            if os.path.normcase(str(existing_path or "")) != normalized_path:
                continue
            self._select_index(index, audio_only)
            record = self._register(path, file_settings_list[index] if index < len(file_settings_list) else configs)
            return {"record": record, "added": False}
        file_list.append(path)
        file_settings_list.append(configs if isinstance(configs, dict) else None)
        self._select_index(len(file_list) - 1, audio_only)
        record = self._register(path, configs)
        return {"record": record, "added": True}

    def counts(self) -> tuple[int, int]:
        gen = self._gen()
        return len(gen["file_list"]), len(gen["audio_file_list"])

    def sync_latest_generated(self, before_counts: tuple[int, int]) -> None:
        gen = self._gen()
        before_files, before_audio = before_counts
        if len(gen["file_list"]) > before_files:
            if gen["last_selected"]:
                self._select_index(len(gen["file_list"]) - 1, False)
            self._register(str(gen["file_list"][-1] or ""), gen["file_settings_list"][-1] if gen["file_settings_list"] else None)
        if len(gen["audio_file_list"]) > before_audio:
            if gen["audio_last_selected"]:
                self._select_index(len(gen["audio_file_list"]) - 1, True)
            self._register(str(gen["audio_file_list"][-1] or ""), gen["audio_file_settings_list"][-1] if gen["audio_file_settings_list"] else None)

    def sync_refresh_path(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        path = str(payload.get("path", "") or "").strip()
        if len(path) == 0:
            return
        detected_type = self._detect_media_type(path)
        if detected_type == "audio":
            file_list = self._gen()["audio_file_list"]
            if path in file_list and self._gen()["audio_last_selected"]:
                self._select_index(file_list.index(path), True)
        elif detected_type in {"image", "video"}:
            file_list = self._gen()["file_list"]
            if path in file_list and self._gen()["last_selected"]:
                self._select_index(file_list.index(path), False)

    def _iter_records(self, audio_only: bool | None = None) -> list[tuple[bool, int, dict[str, Any], str]]:
        gen = self._gen()
        groups = []
        # Rendering a large retained workspace must not re-register every media
        # (registry insertion/search is linear). Invalidate on metadata replacement
        # or session registry reset, while retaining the original registration path.
        registered = {id(record) for record in self._session.media_registry}
        for audio in (False, True):
            if audio_only is not None and audio != audio_only:
                continue
            file_list, file_settings_list = self._resolve_lists(audio)
            for index, path in enumerate(file_list):
                settings = file_settings_list[index]
                cached = self._record_views.get(path)
                if cached is not None and cached[0] is settings and id(cached[1]) in registered:
                    record = cached[1]
                else:
                    record = self._register(path, settings)
                    if record is not None:
                        self._record_views[path] = (settings, record)
                        registered.add(id(record))
                if record is not None:
                    groups.append((audio, index, record, path))
        if audio_only is None:
            active = set(gen['file_list']) | set(gen['audio_file_list'])
            self._record_views = {path: value for path, value in self._record_views.items() if path in active}
        return groups

    def list_lines(self, media_scope: str = "all") -> list[str]:
        scope = str(media_scope or "all").strip().lower()
        if scope not in {"all", "media", "image", "video", "audio"}:
            scope = "all"
        gen = self._gen()
        lines = []
        media_items = self._iter_records(False)
        audio_items = self._iter_records(True)
        groups = []
        if scope == "all":
            groups.append(("Media", media_items))
            groups.append(("Audio", audio_items))
        elif scope == "media":
            groups.append(("Media", media_items))
        elif scope == "image":
            groups.append(("Images", [item for item in media_items if item[2].get("media_type") == "image"]))
        elif scope == "video":
            groups.append(("Videos", [item for item in media_items if item[2].get("media_type") == "video"]))
        elif scope == "audio":
            groups.append(("Audio", audio_items))
        for header, items in groups:
            if len(items) == 0:
                continue
            lines.append(f"{header}:")
            for audio_only, index, record, path in items:
                is_selected = (audio_only and gen["current_gallery_source"] == "audio" and gen["audio_selected"] == index) or (
                    (not audio_only) and gen["current_gallery_source"] == "video" and gen["selected"] == index
                )
                selected_suffix = ""
                if is_selected and record.get("media_type") == "video":
                    selected_suffix = f" @ {float(gen.get('selected_video_time', 0.0) or 0.0):.3f}s"
                marker = "*" if is_selected else " "
                label = str(record.get("label", "") or os.path.basename(path)).strip()
                description = str(record.get("prompt_summary", "") or "").strip()
                if len(description) > 0 and description.casefold() != label.casefold():
                    label = f"{label} - {description}"
                lines.append(f"{marker} {record.get('media_id', '?')} [{index + 1}] {label}{selected_suffix}")
        if len(lines) == 0:
            return ["No media available."]
        return lines

    def _resolve_by_reference(self, reference: str, media_type: str = "any") -> tuple[bool, int] | None:
        ref = str(reference or "").strip()
        if len(ref) == 0:
            return None
        normalized_media_type = media_registry.normalize_media_type(media_type)
        candidates = [
            (audio_only, index, record, path)
            for audio_only, index, record, path in self._iter_records()
            if normalized_media_type == "any" or str(record.get("media_type", "")).strip() == normalized_media_type
        ]
        if ref.isdigit():
            choice = int(ref) - 1
            if 0 <= choice < len(candidates):
                return candidates[choice][0], candidates[choice][1]
            return None
        record = media_registry.get_media_record(self._session, ref)
        if record is not None and (normalized_media_type == "any" or str(record.get("media_type", "")).strip() == normalized_media_type):
            for audio_only, index, candidate, _path in candidates:
                if candidate.get("media_id") == record.get("media_id"):
                    return audio_only, index
        normalized_ref = ref.lower()
        matches = []
        for audio_only, index, record, path in candidates:
            haystack = " ".join(
                [
                    str(record.get("media_id", "") or ""),
                    str(record.get("label", "") or ""),
                    str(record.get("prompt_summary", "") or ""),
                    os.path.basename(path),
                ]
            ).lower()
            if normalized_ref in haystack:
                matches.append((audio_only, index))
        if len(matches) == 1:
            return matches[0]
        return None

    def select(self, reference: str, media_type: str = "any") -> dict[str, Any] | None:
        resolved = self._resolve_by_reference(reference, media_type=media_type)
        if resolved is None:
            return None
        audio_only, index = resolved
        self._select_index(index, audio_only)
        file_list, file_settings_list = self._resolve_lists(audio_only)
        settings = file_settings_list[index] if index < len(file_settings_list) else None
        return self._register(str(file_list[index] or ""), settings)

    def selected_summary(self) -> str:
        gen = self._gen()
        audio_only = gen.get("current_gallery_source", "video") == "audio"
        file_list, file_settings_list = self._resolve_lists(audio_only)
        choice = gen["audio_selected"] if audio_only else gen["selected"]
        if choice is None or choice < 0 or choice >= len(file_list):
            return "No media is currently selected."
        record = self._register(str(file_list[choice] or ""), file_settings_list[choice] if choice < len(file_settings_list) else None)
        if record is None:
            return "No media is currently selected."
        summary = f"Selected {record.get('media_id')}: {record.get('label', os.path.basename(str(file_list[choice] or '')))}"
        if record.get("media_type") == "video":
            frame_text = self.selected_frame_summary()
            summary += f" @ {float(gen.get('selected_video_time', 0.0) or 0.0):.3f}s"
            if frame_text.startswith("Selected frame: "):
                summary += f" ({frame_text[len('Selected frame: '):]})"
        return summary

    def set_selected_time(self, seconds: float) -> str:
        gen = self._gen()
        file_list = gen["file_list"]
        choice = gen["selected"]
        if gen.get("current_gallery_source", "video") != "video" or choice is None or choice < 0 or choice >= len(file_list):
            return "No video is currently selected."
        path = str(file_list[choice] or "").strip()
        if self._detect_media_type(path) != "video":
            return "The selected media is not a video."
        gen["selected_video_time"] = max(0.0, float(seconds))
        return f"Selected video time set to {gen['selected_video_time']:.3f}s."

    def _get_selected_video_state(self) -> tuple[dict[str, Any] | None, str, int, str]:
        gen = self._gen()
        file_list = gen["file_list"]
        file_settings_list = gen["file_settings_list"]
        choice = gen["selected"]
        if gen.get("current_gallery_source", "video") != "video" or choice is None or choice < 0 or choice >= len(file_list):
            return None, "", 0, "No video is currently selected."
        path = str(file_list[choice] or "").strip()
        if self._detect_media_type(path) != "video":
            return None, "", 0, "The selected media is not a video."
        settings = file_settings_list[choice] if choice < len(file_settings_list) else None
        record = self._register(path, settings)
        if record is None:
            return None, "", 0, "No video is currently selected."
        return record, path, choice, ""

    def selected_video_summary(self) -> str:
        record, _path, _choice, error = self._get_selected_video_state()
        if len(error) > 0:
            return error
        return f"Selected video: {record.get('media_id')} ({record.get('label', '')})"

    def selected_frame_summary(self) -> str:
        record, path, _choice, error = self._get_selected_video_state()
        if len(error) > 0:
            return error
        fps, _width, _height, frame_count = get_video_info(path)
        if fps <= 0 or frame_count <= 0:
            return f"Unable to read frame info for {record.get('media_id')}."
        current_time = float(self._gen().get("selected_video_time", 0.0) or 0.0)
        frame_no = max(0, min(frame_count - 1, int(round(current_time * fps))))
        return f"Selected frame: {frame_no} on {record.get('media_id')} @ {current_time:.3f}s ({fps} fps, {frame_count} frames)"

    def set_selected_frame(self, frame_no: int) -> str:
        record, path, _choice, error = self._get_selected_video_state()
        if len(error) > 0:
            return error
        fps, _width, _height, frame_count = get_video_info(path)
        if fps <= 0 or frame_count <= 0:
            return f"Unable to read frame info for {record.get('media_id')}."
        frame_no = int(frame_no)
        if frame_no < 0 or frame_no >= frame_count:
            return f"Frame must be between 0 and {frame_count - 1}."
        seconds = frame_no / float(fps)
        self._gen()["selected_video_time"] = seconds
        return f"Selected frame set to {frame_no} on {record.get('media_id')} @ {seconds:.3f}s."

    def clear_media(self) -> None:
        gen = self._gen()
        gen["file_list"].clear()
        gen["file_settings_list"].clear()
        gen["audio_file_list"].clear()
        gen["audio_file_settings_list"].clear()
        gen["selected"] = -1
        gen["audio_selected"] = -1
        gen["last_selected"] = gen["audio_last_selected"] = True
        gen["current_gallery_source"] = "video"
        gen["selected_video_time"] = None
        self._session.media_registry.clear()
        with self._session.turn_lock:
            self._session.pending_chat_media.clear()
