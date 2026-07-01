from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from shared.utils import files_locator as fl


WAN_GP_SETTINGS_SUFFIXES = {".json", ".zip"}
SETTINGS_BUNDLE_ATTACHMENT_KEYS = ("image_start", "image_end", "image_refs", "image_guide", "image_mask", "video_guide", "video_mask", "video_source", "audio_guide", "audio_guide2", "audio_source", "replace_voice_sample", "replace_voice_sample2", "custom_guide")


def is_wangp_settings_filename(value: Any) -> bool:
    return Path(str(value or "").strip()).suffix.lower() in WAN_GP_SETTINGS_SUFFIXES


def _cache_root() -> Path:
    return Path(__file__).resolve().parents[2] / "settings" / "_settings_bundle_cache"


def _safe_zip_name(name: Any) -> str:
    text = str(name or "").strip().replace("\\", "/")
    if not fl.is_relative_down_path(text):
        return ""
    return text


def _extract_bundle_file(zf: zipfile.ZipFile, member_name: str, cache_dir: Path) -> str:
    safe_name = _safe_zip_name(member_name)
    if len(safe_name) == 0 or safe_name not in zf.namelist():
        return member_name
    target_name = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_name)
    target_path = cache_dir / target_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(safe_name) as source, target_path.open("wb") as target:
        target.write(source.read())
    return str(target_path.resolve())


def _extract_attachment_value(zf: zipfile.ZipFile, value: Any, cache_dir: Path) -> Any:
    if isinstance(value, str):
        if Path(value).is_absolute():
            return value
        return _extract_bundle_file(zf, value, cache_dir)
    if isinstance(value, list):
        return [_extract_attachment_value(zf, item, cache_dir) for item in value]
    return value


def load_first_settings_from_queue_zip(zip_path: str | Path, attachment_keys: list[str] | tuple[str, ...]) -> tuple[dict[str, Any] | None, int]:
    source_path = Path(zip_path).resolve()
    stat = source_path.stat()
    cache_dir = _cache_root() / f"{source_path.stem}_{int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1_000_000_000)))}"
    with zipfile.ZipFile(source_path, "r") as zf:
        if "queue.json" not in zf.namelist():
            return None, 0
        manifest = json.loads(zf.read("queue.json").decode("utf-8"))
        if not isinstance(manifest, list) or len(manifest) == 0:
            return None, 0
        task = manifest[0]
        if not isinstance(task, dict):
            return None, len(manifest)
        params = task.get("params", task)
        if not isinstance(params, dict):
            return None, len(manifest)
        payload = dict(params)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for key in attachment_keys:
            if key in payload:
                payload[key] = _extract_attachment_value(zf, payload[key], cache_dir)
        return payload, len(manifest)
