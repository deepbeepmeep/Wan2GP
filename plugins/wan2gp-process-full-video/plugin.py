from __future__ import annotations

import copy
import html
import json
import math
import os
import re
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
from shared.utils.audio_video import extract_audio_tracks, get_hdr_video_encode_args, get_video_encode_args
from shared.utils.hdr import VIDEO_PROMPT_HDR_OUTPUT_FLAG, iter_hdr_gbrpf32_frames, tonemap_hdr_tensor_to_uint8
from shared.utils.plugins import WAN2GPPlugin
from shared.utils.utils import get_video_info_details
from shared.utils.video_decode import decode_video_frames_ffmpeg, resolve_media_binary
from shared.utils.video_metadata import DEFAULT_RESERVED_VIDEO_METADATA_BYTES, read_metadata_from_video, save_video_metadata, write_reserved_video_ffmetadata
from shared.utils.virtual_media import build_virtual_media_path, clear_virtual_media_source, get_virtual_video, store_virtual_video

PlugIn_Name = "Process Full Video"
PlugIn_Id = "ProcessFullVideo"

PLUGIN_DIR = Path(__file__).resolve().parent
APP_ROOT_DIR = PLUGIN_DIR.parent.parent
APP_SETTINGS_DIR = APP_ROOT_DIR / "settings"
PROCESS_SETTINGS_DIR = PLUGIN_DIR / "settings"
PROCESS_FULL_VIDEO_SETTINGS_FILE = APP_SETTINGS_DIR / "process_full_video_settings.json"
RATIO_CHOICES = [("1:1", "1:1"), ("4:3", "4:3"), ("3:4", "3:4"), ("16:9", "16:9"), ("9:16", "9:16"), ("21:9", "21:9"), ("9:21", "9:21")]
RATIO_CHOICES_WITH_EMPTY = [("", "")] + RATIO_CHOICES
DEFAULT_SOURCE_PATH = ""
DEFAULT_OUTPUT_PATH = ""
LAUNCH_DEFAULT_PROCESS_NAME = "Outpaint Video - LTX 2.3 Distilled 1.1"
ADD_USER_SETTINGS_MODEL_TYPE = "__add_user_settings__"
ADD_USER_SETTINGS_LABEL = "<Add User Settings>"
NO_USER_SETTINGS_VALUE = "__no_user_settings__"
NO_USER_SETTINGS_LABEL = "<No choice>"
USER_SETTINGS_STORAGE_KEY = "user_settings"
USER_PROCESS_VALUE_PREFIX = "__user_settings__:"
USER_SETTINGS_HINT_HTML = "<div style='font-size:10px;line-height:1;opacity:.65;'>* user settings</div>"
MAX_STATUS_REFRESH_HZ = 3.0
STATUS_REFRESH_INTERVAL_SECONDS = 1.0 / MAX_STATUS_REFRESH_HZ
PROCESS_FULL_VIDEO_METADATA_KEY = "fill_process_video"
PROCESS_FULL_VIDEO_VSOURCE = "process_full_video"
PROCESS_FULL_VIDEO_VFILE = "last_frames.mp4"
USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE = True
TIMED_PROMPT_EXAMPLE = "00:00\nA calm cinematic opening shot.\n\n00:30\nThe mood becomes tense and dramatic."
TIMED_PROMPT_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?$")


def _load_process_definitions() -> tuple[dict[str, dict], str | None]:
    if not PROCESS_SETTINGS_DIR.is_dir():
        return {}, f"Missing process settings folder: {PROCESS_SETTINGS_DIR}"
    process_definitions: dict[str, dict] = {}
    for settings_path in sorted(PROCESS_SETTINGS_DIR.glob("*.json")):
        try:
            raw_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {}, f"Unable to read process setting file {settings_path.name}: {exc}"
        if not isinstance(raw_settings, dict):
            return {}, f"Process setting file {settings_path.name} must contain a JSON object."
        process_name = str(settings_path.stem).strip()
        model_type = str(raw_settings.get("model_type") or "").strip()
        if len(process_name) == 0:
            return {}, f"Process setting file {settings_path.name} has an empty filename stem."
        if len(model_type) == 0:
            return {}, f"Process setting file {settings_path.name} is missing model_type."
        process_definitions[process_name] = {"settings": raw_settings, "path": str(settings_path)}
    if len(process_definitions) == 0:
        return {}, f"No process setting files were found in: {PROCESS_SETTINGS_DIR}"
    return process_definitions, None


PROCESS_DEFINITIONS, PROCESS_DEFINITIONS_ERROR = _load_process_definitions()
PROCESS_CHOICES = [(process_name, process_name) for process_name in PROCESS_DEFINITIONS]
DEFAULT_PROCESS_NAME = LAUNCH_DEFAULT_PROCESS_NAME if LAUNCH_DEFAULT_PROCESS_NAME in PROCESS_DEFINITIONS else next(iter(PROCESS_DEFINITIONS), "")
DEFAULT_MODEL_TYPE = str(PROCESS_DEFINITIONS.get(DEFAULT_PROCESS_NAME, {}).get("settings", {}).get("model_type") or "")


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _coerce_float(value, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if math.isnan(result):
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def _coerce_int(value, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        result = int(round(float(value)))
    except (TypeError, ValueError):
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def _load_saved_process_full_video_settings() -> dict:
    if not PROCESS_FULL_VIDEO_SETTINGS_FILE.is_file():
        return {}
    try:
        raw_settings = json.loads(PROCESS_FULL_VIDEO_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Process Full Video] Warning: unable to read saved UI settings from {PROCESS_FULL_VIDEO_SETTINGS_FILE}: {exc}")
        return {}
    return raw_settings if isinstance(raw_settings, dict) else {}


def _save_process_full_video_settings(settings: dict) -> None:
    PROCESS_FULL_VIDEO_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESS_FULL_VIDEO_SETTINGS_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _normalize_user_settings_ref(value) -> str:
    text = str(value or "").strip().strip('"').replace("\\", "/")
    if len(text) == 0 or text.startswith(("/", "./", "../")):
        return ""
    if len(text) >= 2 and text[1] == ":":
        return ""
    parts = [part.strip() for part in text.split("/") if len(part.strip()) > 0]
    if len(parts) != 2:
        return ""
    base_model_type, filename = parts
    filename = Path(filename).name
    if not filename.lower().endswith(".json"):
        return ""
    return f"{base_model_type}/{filename}"


def _is_user_process_value(value) -> bool:
    return str(value or "").startswith(USER_PROCESS_VALUE_PREFIX)


def _user_process_value(ref: str) -> str:
    normalized = _normalize_user_settings_ref(ref)
    return f"{USER_PROCESS_VALUE_PREFIX}{normalized}" if len(normalized) > 0 else ""


def _user_process_ref_from_value(value) -> str:
    text = str(value or "").strip()
    if not text.startswith(USER_PROCESS_VALUE_PREFIX):
        return ""
    return _normalize_user_settings_ref(text[len(USER_PROCESS_VALUE_PREFIX):])


def _get_saved_user_settings_refs(settings: dict | None) -> list[str]:
    raw_refs = settings.get(USER_SETTINGS_STORAGE_KEY, []) if isinstance(settings, dict) else []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    if not isinstance(raw_refs, list):
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for raw_ref in raw_refs:
        ref = _normalize_user_settings_ref(raw_ref)
        if len(ref) == 0 or ref.casefold() in seen:
            continue
        refs.append(ref)
        seen.add(ref.casefold())
    return refs


def _store_user_settings_refs(refs: list[str]) -> None:
    saved_settings = _load_saved_process_full_video_settings()
    normalized_refs = _get_saved_user_settings_refs({USER_SETTINGS_STORAGE_KEY: refs})
    saved_settings[USER_SETTINGS_STORAGE_KEY] = normalized_refs
    _save_process_full_video_settings(saved_settings)


def _save_process_full_video_ui_settings(settings: dict) -> None:
    saved_settings = _load_saved_process_full_video_settings()
    user_refs = _get_saved_user_settings_refs(saved_settings)
    next_settings = dict(settings)
    next_settings[USER_SETTINGS_STORAGE_KEY] = user_refs
    _save_process_full_video_settings(next_settings)


def _get_error_message(exc: BaseException) -> str:
    message = getattr(exc, "message", exc)
    return str(message or "").strip()


def _get_default_process_strength(process_settings: dict) -> float:
    process_strength = process_settings.get("process_strength")
    if process_strength is None:
        process_strength = process_settings.get("loras_multipliers", 1.0)
    return _coerce_float(process_strength, 1.0)


@dataclass(frozen=True)
class ChunkPlan:
    control_start_frame: int
    requested_frames: int
    overlap_frames: int

    @property
    def control_end_frame(self) -> int:
        return self.control_start_frame + self.requested_frames - 1


@dataclass(frozen=True)
class FramePlanRules:
    frame_step: int
    minimum_requested_frames: int


def _require_process_definition(process_name: str) -> dict:
    process_definition = PROCESS_DEFINITIONS.get(str(process_name))
    if not isinstance(process_definition, dict):
        available = ", ".join(PROCESS_DEFINITIONS.keys()) or "none"
        raise gr.Error(f"Unsupported process: {process_name}. Available process settings: {available}.")
    return process_definition


def _require_model_def(model_type: str, get_model_def) -> dict:
    if not callable(get_model_def):
        raise gr.Error("WanGP model definitions are unavailable in this plugin context.")
    model_def = get_model_def(str(model_type))
    if not isinstance(model_def, dict):
        raise gr.Error(f"Unsupported model type: {model_type}")
    return model_def


def _get_frame_plan_rules(model_type: str, get_model_def) -> FramePlanRules:
    model_def = _require_model_def(model_type, get_model_def)
    return FramePlanRules(frame_step=max(1, int(model_def.get("frames_steps", 1))), minimum_requested_frames=max(1, int(model_def.get("frames_minimum", 1))))


def _get_vae_temporal_latent_size(model_type: str, get_model_def) -> int:
    model_def = _require_model_def(model_type, get_model_def)
    return max(1, int(model_def.get("latent_size", model_def.get("frames_steps", 1))))


def _get_overlap_slider_max(model_type: str, get_model_def, *, exclusive_upper_bound: int = 100) -> int:
    step = _get_vae_temporal_latent_size(model_type, get_model_def)
    last_allowed_value = max(1, int(exclusive_upper_bound) - 1)
    return 1 + ((last_allowed_value - 1) // step) * step


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


def _normalize_overlap_frames(overlap_frames: float, *, frame_step: int) -> int:
    target_frames = max(1, int(round(float(overlap_frames or 1))))
    below = max(1, _align_requested_frames(target_frames, frame_step=frame_step, round_up=False))
    above = max(1, _align_requested_frames(target_frames, frame_step=frame_step, round_up=True))
    return below if abs(below - target_frames) <= abs(above - target_frames) else above


def _align_total_unique_frames(total_unique_frames: int, *, frame_step: int, minimum_requested_frames: int, initial_overlap_frames: int) -> int:
    total_unique_frames = max(0, int(total_unique_frames))
    initial_overlap_frames = max(0, int(initial_overlap_frames))
    if initial_overlap_frames > 0:
        minimum_unique_frames = max(1, int(minimum_requested_frames) - initial_overlap_frames)
        return 0 if total_unique_frames < minimum_unique_frames else total_unique_frames - (total_unique_frames % max(1, int(frame_step)))
    return 0 if total_unique_frames < max(1, int(minimum_requested_frames)) else ((total_unique_frames - 1) // max(1, int(frame_step))) * max(1, int(frame_step)) + 1


def _count_planned_unique_frames(plans: list[ChunkPlan]) -> int:
    return sum(max(0, int(plan.requested_frames) - int(plan.overlap_frames)) for plan in plans)


def _describe_frame_range(start_frame: int, frame_count: int) -> str:
    frame_count = max(0, int(frame_count))
    if frame_count <= 0:
        return "0 frame(s)"
    start_frame = int(start_frame)
    return f"{frame_count} frame(s) [{start_frame}..{start_frame + frame_count - 1}]"


def _choose_resolution(budget_label: str) -> str:
    resolutions = {"256p": "352x256", "360p": "480x360", "480p": "640x480", "540p": "720x540", "720p": "1280x720", "900p": "1200x900", "1080p": "1920x1088"}
    try:
        return resolutions[str(budget_label)]
    except KeyError as exc:
        raise gr.Error(f"Unsupported Output Resolution: {budget_label}") from exc


def _format_time_token(seconds: float | None) -> str:
    if seconds in (None, ""):
        return "end"
    total_centiseconds = max(0, int(round(float(seconds) * 100.0)))
    total_seconds, centiseconds = divmod(total_centiseconds, 100)
    minutes, seconds_only = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    seconds_text = f"{seconds_only:02d}" if centiseconds <= 0 else f"{seconds_only:02d}.{centiseconds:02d}"
    if hours > 0:
        token = f"{hours:02d}h{minutes:02d}m{seconds_text}s"
    else:
        token = f"{minutes:02d}m{seconds_text}s"
    return token


def _format_time_hms(seconds: float | None) -> str:
    total_seconds = max(0, int(round(float(seconds or 0.0))))
    minutes, seconds_only = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_only:02d}"


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
            raise gr.Error(f"{label} must be a number of seconds, MM:SS(.xx), or HH:MM:SS(.xx).") from exc
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise gr.Error(f"{label} must be a number of seconds, MM:SS(.xx), or HH:MM:SS(.xx).")
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
        raise gr.Error(f"{label} must be a number of seconds, MM:SS(.xx), or HH:MM:SS(.xx).") from exc


def _parse_prompt_schedule(prompt_text: str) -> list[tuple[float, str]]:
    text = str(prompt_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) == 0:
        return [(0.0, "")]
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if len(block.strip()) > 0]
    first_line = text.split("\n", 1)[0].strip()
    if len(blocks) <= 1 and not TIMED_PROMPT_TIMESTAMP_RE.fullmatch(first_line):
        return [(0.0, text)]
    schedule: list[tuple[float, str]] = []
    for block in blocks:
        lines = block.split("\n")
        timestamp_line = lines[0].strip()
        if not TIMED_PROMPT_TIMESTAMP_RE.fullmatch(timestamp_line):
            raise gr.Error(
                "Timed prompts must be separated by blank lines, and each block must start with a timestamp like MM:SS(.xx) or HH:MM:SS(.xx).\n\n"
                f"Example:\n{TIMED_PROMPT_EXAMPLE}"
            )
        prompt_body = "\n".join(lines[1:]).strip()
        if len(prompt_body) == 0:
            raise gr.Error(
                "Each timed prompt block must contain prompt text after its timestamp.\n\n"
                f"Example:\n{TIMED_PROMPT_EXAMPLE}"
            )
        schedule.append((float(_parse_time_input(timestamp_line, label="Timed prompt timestamp", allow_empty=False) or 0.0), prompt_body))
    return sorted(schedule, key=lambda item: item[0])


def _resolve_prompt_for_chunk(prompt_schedule: list[tuple[float, str]], chunk_start_seconds: float, default_prompt: str) -> str:
    prompt_text = str(default_prompt or "")
    for start_seconds, scheduled_prompt in prompt_schedule:
        if float(start_seconds) <= float(chunk_start_seconds) + 1e-9:
            prompt_text = scheduled_prompt
        else:
            break
    return prompt_text


def _get_process_filename_token(process_name: str) -> str:
    words = str(process_name or "").strip().split()
    if len(words) == 0:
        return "process"
    token = "".join(char for char in words[0].lower() if char.isalnum() or char in {"-", "_"})
    return token or "process"


def _process_has_outpaint(process_name: str) -> bool:
    process_definition = PROCESS_DEFINITIONS.get(str(process_name))
    settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
    return isinstance(settings, dict) and "video_guide_outpainting" in settings


def _build_auto_output_path(source_path: str, process_name: str, ratio_text: str, output_resolution: str, start_seconds: float | None, end_seconds: float | None, output_dir: str | None = None, *, has_outpaint: bool | None = None) -> str:
    source = Path(source_path)
    process_token = _get_process_filename_token(process_name)
    resolution_suffix = str(output_resolution or "").strip() or "res"
    start_suffix = _format_time_token(start_seconds)
    end_suffix = _format_time_token(end_seconds)
    target_dir = source.parent if not output_dir else Path(output_dir)
    name_parts = [source.stem, process_token]
    if _process_has_outpaint(process_name) if has_outpaint is None else bool(has_outpaint):
        name_parts.append(str(ratio_text or "").replace(":", "x") or "ratio")
    name_parts.extend([resolution_suffix, start_suffix, end_suffix])
    return str(target_dir / f"{'_'.join(name_parts)}{source.suffix}")


def _make_output_variant(output: Path) -> str:
    for index in range(2, 10000):
        candidate = output.with_name(f"{output.stem}_{index}{output.suffix}")
        if not candidate.exists():
            _plugin_info(f"Output file already exists. Using {candidate}")
            return str(candidate)
    raise gr.Error(f"Unable to find a free output filename for {output}")


def _make_continuation_output_path(output_path: str) -> str:
    output = Path(output_path)
    existing_paths = _list_residual_continuation_paths(str(output))
    if len(existing_paths) == 0:
        return str(output.with_name(f"{output.stem}_continue{output.suffix}"))
    max_index = 1
    base_stem = f"{output.stem}_continue"
    for existing_path in existing_paths:
        existing_stem = Path(existing_path).stem
        if existing_stem == base_stem:
            max_index = max(max_index, 1)
            continue
        suffix = existing_stem[len(base_stem) + 1:]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    for index in range(max_index + 1, 10000):
        variant = output.with_name(f"{output.stem}_continue_{index}{output.suffix}")
        if not variant.exists():
            return str(variant)
    raise gr.Error(f"Unable to find a free continuation filename for {output}")


def _list_residual_continuation_paths(output_path: str) -> list[str]:
    output = Path(output_path)
    base_stem = f"{output.stem}_continue"
    candidates: list[tuple[int, str]] = []
    for child in output.parent.glob(f"{base_stem}*{output.suffix}"):
        if not child.is_file():
            continue
        if child.stem == base_stem:
            candidates.append((1, str(child)))
            continue
        prefix = base_stem + "_"
        if not child.stem.startswith(prefix):
            continue
        suffix = child.stem[len(prefix):]
        if suffix.isdigit():
            candidates.append((int(suffix), str(child)))
    return [path for _, path in sorted(candidates)]


class _ContinuationMergeOutputLockedError(PermissionError):
    def __init__(self, output_path: str) -> None:
        self.output_path = str(output_path or "")
        super().__init__(f"Unable to replace locked output file: {self.output_path}")


def _make_continuation_signature(file_path: str) -> dict | None:
    if not isinstance(file_path, str) or not os.path.isfile(file_path):
        return None
    try:
        stats = os.stat(file_path)
    except OSError:
        return None
    return {"path": str(Path(file_path).resolve()), "size": int(stats.st_size), "mtime_ns": int(getattr(stats, "st_mtime_ns", int(stats.st_mtime * 1_000_000_000)))}


def _continuation_signature_key(signature: dict) -> tuple[str, int, int] | None:
    if not isinstance(signature, dict):
        return None
    path = str(signature.get("path") or "").strip()
    if len(path) == 0:
        return None
    try:
        return str(Path(path).resolve()), max(0, int(signature.get("size"))), max(0, int(signature.get("mtime_ns")))
    except (TypeError, ValueError):
        return None


def _normalize_merged_continuation_signatures(signatures) -> list[dict]:
    normalized: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for signature in list(signatures or []):
        key = _continuation_signature_key(signature)
        if key is None or key in seen:
            continue
        seen.add(key)
        path, size, mtime_ns = key
        normalized.append({"path": path, "size": size, "mtime_ns": mtime_ns})
    return normalized


def _append_merged_continuation_signature(signatures: list[dict], signature: dict | None) -> list[dict]:
    return _normalize_merged_continuation_signatures([*list(signatures or []), *( [] if signature is None else [signature] )])


def _read_merged_continuation_signatures(output_path: str) -> list[dict]:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return []
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict):
        return []
    process_metadata = metadata.get(PROCESS_FULL_VIDEO_METADATA_KEY)
    return [] if not isinstance(process_metadata, dict) else _normalize_merged_continuation_signatures(process_metadata.get("merged_continuations"))


def _store_merged_continuation_signatures(output_path: str, signatures: list[dict], *, verbose_level: int = 0) -> None:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict) or len(metadata) == 0:
        return
    process_metadata = metadata.get(PROCESS_FULL_VIDEO_METADATA_KEY)
    process_metadata = {} if not isinstance(process_metadata, dict) else process_metadata.copy()
    process_metadata["merged_continuations"] = _normalize_merged_continuation_signatures(signatures)
    metadata[PROCESS_FULL_VIDEO_METADATA_KEY] = process_metadata
    if not save_video_metadata(output_path, metadata, allow_inplace_update=True, verbose_level=verbose_level):
        print(f"[Process Full Video] Warning: failed to store merged continuation signatures in {output_path}")


def _read_recorded_written_unique_frames(output_path: str) -> int:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return 0
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict):
        return 0
    process_metadata = metadata.get(PROCESS_FULL_VIDEO_METADATA_KEY)
    if not isinstance(process_metadata, dict):
        return 0
    try:
        return max(0, int(process_metadata.get("written_unique_frames") or 0))
    except (TypeError, ValueError):
        return 0


def _build_requested_output_path(source_path: str, output_path: str, process_name: str, ratio_text: str, output_resolution: str, start_seconds: float | None, end_seconds: float | None, *, has_outpaint: bool | None = None) -> Path:
    output_text = str(output_path or "").strip()
    if len(output_text) == 0:
        output = Path(_build_auto_output_path(source_path, process_name, ratio_text, output_resolution, start_seconds, end_seconds, has_outpaint=has_outpaint))
    elif output_text.endswith(("\\", "/")) or Path(output_text).is_dir():
        output = Path(_build_auto_output_path(source_path, process_name, ratio_text, output_resolution, start_seconds, end_seconds, output_dir=output_text, has_outpaint=has_outpaint))
    else:
        output = Path(output_text)
    source_suffix = Path(source_path).suffix
    if not output.suffix:
        output = output.with_suffix(source_suffix)
    return output


def _resolve_output_path(source_path: str, output_path: str, process_name: str, ratio_text: str, output_resolution: str, start_seconds: float | None, end_seconds: float | None, continue_enabled: bool, *, has_outpaint: bool | None = None) -> tuple[str, bool]:
    output = _build_requested_output_path(source_path, output_path, process_name, ratio_text, output_resolution, start_seconds, end_seconds, has_outpaint=has_outpaint)
    if continue_enabled:
        return str(output), output.exists()
    if output.exists():
        return _make_output_variant(output), False
    return str(output), False


def _normalize_identity_path(path_value: str) -> str:
    text = str(path_value or "").strip()
    if len(text) == 0:
        return ""
    try:
        return str(Path(text).resolve()).casefold()
    except (OSError, RuntimeError, ValueError):
        return text.casefold()


def _read_output_identity(output_path: str) -> tuple[str, str, str] | None:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return None
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict) or len(metadata) == 0:
        return None
    process_metadata = metadata.get(PROCESS_FULL_VIDEO_METADATA_KEY)
    process_metadata = process_metadata if isinstance(process_metadata, dict) else {}
    process_name = str(process_metadata.get("process") or "").strip()
    if len(process_name) == 0:
        segments = metadata.get("segments")
        first_segment = next((segment for segment in list(segments or []) if isinstance(segment, dict)), None)
        if first_segment is not None:
            process_name = str(first_segment.get("process") or "").strip()
    source_video = str(process_metadata.get("source_video") or metadata.get("source_video") or "").strip()
    source_segment = str(process_metadata.get("source_segment") or metadata.get("source_segment") or "").strip()
    return process_name, source_video, source_segment


def _get_output_identity_mismatch_message(output_path: str, *, process_name: str, source_path: str, source_segment: str) -> str | None:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return None
    identity = _read_output_identity(output_path)
    if identity is None:
        return f"Output file already exists at {output_path}, but it does not contain readable WanGP metadata. Processing was stopped."
    existing_process, existing_source_video, existing_source_segment = identity
    mismatches: list[str] = []
    if str(existing_process or "").strip() != str(process_name or "").strip():
        mismatches.append("process")
    if _normalize_identity_path(existing_source_video) != _normalize_identity_path(source_path):
        mismatches.append("source_video")
    if str(existing_source_segment or "").strip() != str(source_segment or "").strip():
        mismatches.append("source_segment")
    if len(mismatches) == 0:
        return None
    mismatch_text = ", ".join(mismatches)
    return f"Output file already exists at {output_path}, but its metadata does not match the current {mismatch_text}. Processing was stopped."


def _frame_to_image(frame_tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(frame_tensor.permute(1, 2, 0).cpu().numpy())


def _build_process_full_video_source_path(*, hdr: bool = False) -> str:
    extras = {"vsource": PROCESS_FULL_VIDEO_VSOURCE}
    if hdr:
        extras["hdr"] = "1"
    return build_virtual_media_path(PROCESS_FULL_VIDEO_VFILE, extras=extras)


def _set_process_full_video_overlap_buffer(overlap_tensor: torch.Tensor | None, fps_float: float, *, hdr: bool = False) -> None:
    if overlap_tensor is None or not torch.is_tensor(overlap_tensor) or int(overlap_tensor.shape[1]) <= 0:
        clear_virtual_media_source(PROCESS_FULL_VIDEO_VSOURCE)
        return
    store_virtual_video(PROCESS_FULL_VIDEO_VSOURCE, PROCESS_FULL_VIDEO_VFILE, overlap_tensor.contiguous(), fps_float, hdr=hdr)


def _load_process_full_video_overlap_buffer(video_path: str, overlap_frames: int, actual_frame_count: int) -> torch.Tensor | None:
    overlap_frames = max(0, int(overlap_frames))
    actual_frame_count = max(0, int(actual_frame_count))
    overlap_frames = min(overlap_frames, actual_frame_count)
    if overlap_frames <= 0 or actual_frame_count <= 0 or not os.path.isfile(video_path):
        return None
    frames = decode_video_frames_ffmpeg(build_virtual_media_path(video_path, start_frame=-overlap_frames, end_frame=-1, extras={"frame_count": actual_frame_count}), 0, overlap_frames, target_fps=None, bridge="torch")
    return None if not torch.is_tensor(frames) or int(frames.shape[0]) <= 0 else frames.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0).contiguous()


def _update_process_full_video_overlap_buffer(committed_tensor: torch.Tensor, overlap_frames: int, fps_float: float, *, hdr: bool = False) -> torch.Tensor | None:
    overlap_frames = max(0, int(overlap_frames))
    if overlap_frames <= 0 or not torch.is_tensor(committed_tensor) or int(committed_tensor.shape[1]) <= 0:
        clear_virtual_media_source(PROCESS_FULL_VIDEO_VSOURCE)
        return None
    overlap_tensor = committed_tensor.detach().cpu().to(torch.float32).contiguous()
    if not hdr:
        overlap_tensor = overlap_tensor.div_(127.5).sub_(1.0)
    previous_overlap = get_virtual_video(PROCESS_FULL_VIDEO_VSOURCE, PROCESS_FULL_VIDEO_VFILE)
    if previous_overlap is not None and int(previous_overlap.shape[1]) > 0:
        overlap_tensor = torch.cat([previous_overlap, overlap_tensor], dim=1)
    overlap_tensor = overlap_tensor[:, -min(overlap_frames, int(overlap_tensor.shape[1])):].contiguous()
    _set_process_full_video_overlap_buffer(overlap_tensor, fps_float, hdr=hdr)
    return overlap_tensor


def _load_process_full_video_hdr_overlap_buffer(output_path: str, overlap_frames: int, actual_frame_count: int) -> torch.Tensor | None:
    overlap_frames = max(0, min(int(overlap_frames), int(actual_frame_count)))
    if overlap_frames <= 0:
        return None
    frames = decode_video_frames_ffmpeg(build_virtual_media_path(output_path, start_frame=-overlap_frames, end_frame=-1, extras={"frame_count": actual_frame_count}), 0, overlap_frames, target_fps=None, bridge="torch", hdr_linear=True)
    return None if not torch.is_tensor(frames) or int(frames.shape[0]) <= 0 else frames.permute(3, 0, 1, 2).contiguous()


def _extract_exact_frame_image(video_path: str, frame_no: int) -> Image.Image:
    ffmpeg_path = resolve_media_binary("ffmpeg")
    if ffmpeg_path is None or not os.path.isfile(video_path) or int(frame_no) < 0:
        raise gr.Error(f"Unable to decode frame {frame_no} from {video_path}")
    with tempfile.TemporaryDirectory(prefix="wangp_tail_frame_") as temp_dir:
        output_path = os.path.join(temp_dir, "frame.png")
        command = [ffmpeg_path, "-v", "error", "-y", "-i", video_path, "-an", "-sn", "-vf", f"select=eq(n\\,{int(frame_no)})", "-frames:v", "1", output_path]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise gr.Error(f"Unable to decode frame {frame_no} from {video_path}")
        with Image.open(output_path) as frame_image:
            return frame_image.convert("RGB").copy()


def _resolve_resume_last_frame(video_path: str, reported_frame_count: int) -> tuple[int, Image.Image | None, str]:
    candidate_count = max(0, int(reported_frame_count))
    if candidate_count <= 0:
        return 0, None, "existing output contains no decodable frame"
    for backtrack in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256):
        frame_no = candidate_count - 1 - backtrack
        if frame_no < 0:
            continue
        try:
            frame_image = _extract_exact_frame_image(video_path, frame_no)
        except gr.Error:
            continue
        actual_frame_count = frame_no + 1
        message = "" if actual_frame_count == candidate_count else f"Adjusted continuation point to {actual_frame_count} decodable frame(s) from the existing output."
        return actual_frame_count, frame_image, message
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


def _write_hdr_video_chunk(process, video_tensor_hdr: torch.Tensor, *, start_frame: int, frame_count: int) -> torch.Tensor:
    if frame_count <= 0:
        raise RuntimeError("No HDR frames available to write.")
    chunk = video_tensor_hdr[:, start_frame:start_frame + frame_count].detach().cpu()
    try:
        for frame_bytes in iter_hdr_gbrpf32_frames(chunk):
            process.stdin.write(frame_bytes)
        process.stdin.flush()
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None and process.poll() is not None else ""
        raise RuntimeError(stderr or "ffmpeg stopped receiving HDR video frames while streaming a chunk") from exc
    if process.poll() not in (None, 0):
        stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None else ""
        raise RuntimeError(stderr or "ffmpeg exited while streaming HDR video")
    return tonemap_hdr_tensor_to_uint8(video_tensor_hdr[:, start_frame + frame_count - 1:start_frame + frame_count])[:, 0]


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


def _get_processing_fps(fps_float: float) -> float:
    return float(max(1, int(round(float(fps_float or 0.0)))))


def _build_chunk_plan(
    start_frame: int,
    end_frame_exclusive: int,
    total_source_frames: int,
    chunk_frames: int,
    *,
    frame_step: int,
    minimum_requested_frames: int,
    overlap_frames: int,
    initial_overlap_frames: int = 0,
) -> list[ChunkPlan]:
    plans: list[ChunkPlan] = []
    cursor = start_frame
    overlap_frames = max(0, int(overlap_frames))
    initial_overlap_frames = max(0, int(initial_overlap_frames))
    total_unique_frames = _align_total_unique_frames(
        end_frame_exclusive - start_frame,
        frame_step=frame_step,
        minimum_requested_frames=minimum_requested_frames,
        initial_overlap_frames=initial_overlap_frames,
    )
    if total_unique_frames <= 0:
        raise gr.Error("The selected range ends too close to the source video end to build a valid chunk for the current model.")
    written_unique_frames = 0
    while written_unique_frames < total_unique_frames:
        plan_overlap_frames = initial_overlap_frames if len(plans) == 0 else overlap_frames
        remaining_unique = total_unique_frames - written_unique_frames
        max_unique_frames = chunk_frames - plan_overlap_frames
        requested_frames = chunk_frames if remaining_unique > max_unique_frames else remaining_unique + plan_overlap_frames
        control_start_frame = cursor - plan_overlap_frames
        max_available_frames = total_source_frames - control_start_frame
        if max_available_frames < requested_frames:
            raise gr.Error("The selected range ends too close to the source video end to build a valid chunk for the current model.")
        if requested_frames < max(1, int(minimum_requested_frames)):
            raise gr.Error("The selected range ends too close to the source video end to build a valid chunk for the current model.")
        plans.append(ChunkPlan(control_start_frame=control_start_frame, requested_frames=requested_frames, overlap_frames=plan_overlap_frames))
        unique_frames = requested_frames - plan_overlap_frames
        written_unique_frames += unique_frames
        cursor += unique_frames
    return plans


def _count_completed_chunks(plans: list[ChunkPlan], completed_unique_frames: int) -> tuple[int, int]:
    completed_chunks = 0
    consumed_frames = 0
    target_frames = max(0, int(completed_unique_frames))
    for plan in plans:
        unique_frames = plan.requested_frames - int(plan.overlap_frames)
        if consumed_frames + unique_frames <= target_frames:
            consumed_frames += unique_frames
            completed_chunks += 1
            continue
        break
    return completed_chunks, consumed_frames


def _probe_resume_frame_count(ffprobe_path: str, output_path: str, fps_float: float) -> tuple[int, str]:
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
    metadata = get_video_info_details(output_path)
    frame_count = int(metadata.get("frame_count") or 0)
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


def _get_mmgp_verbose_level() -> int:
    try:
        from mmgp import offload
        return int(getattr(offload, "default_verboseLevel", 0) or 0)
    except Exception:
        return 0


def _log_existing_output_metadata(output_path: str, verbose_level: int) -> None:
    if int(verbose_level or 0) < 2 or not os.path.isfile(output_path):
        return
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict) or len(metadata) == 0:
        print(f"[Process Full Video] Existing output metadata not found in {output_path}")
        return
    creation_date = str(metadata.get("creation_date") or "unknown")
    generation_time = metadata.get("generation_time")
    generation_time_text = "unknown" if generation_time in (None, "") else str(generation_time)
    print(f"[Process Full Video] Existing output metadata found: creation_date={creation_date}, generation_time={generation_time_text}")


def _probe_media_duration(ffprobe_path: str, media_path: str) -> float:
    result = subprocess.run([ffprobe_path, "-v", "error", "-show_entries", "format=duration", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if result.returncode != 0:
        return 0.0
    try:
        return max(0.0, float((((json.loads(result.stdout) or {}).get("format") or {}).get("duration")) or 0.0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def _probe_audio_end_time(ffprobe_path: str, media_path: str, audio_index: int) -> float:
    stream_selector = f"a:{max(0, int(audio_index))}"
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", stream_selector, "-show_entries", "stream=start_time,duration", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if result.returncode == 0:
        try:
            stream = (((json.loads(result.stdout) or {}).get("streams") or [{}])[0])
            start_time = 0.0 if stream.get("start_time") in (None, "", "N/A") else max(0.0, float(stream.get("start_time")))
            duration_time = 0.0 if stream.get("duration") in (None, "", "N/A") else max(0.0, float(stream.get("duration")))
            if duration_time > 0.0:
                return start_time + duration_time
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            pass
    approximate_end = _probe_media_duration(ffprobe_path, media_path)
    if approximate_end <= 0.0:
        return 0.0
    for window_seconds in (2.0, 8.0, 32.0, 128.0):
        probe_start = max(0.0, approximate_end - window_seconds)
        probe_span = max(0.25, approximate_end - probe_start + 0.25)
        result = subprocess.run(
            [ffprobe_path, "-v", "error", "-select_streams", stream_selector, "-read_intervals", f"{probe_start:.6f}%+{probe_span:.6f}", "-show_packets", "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0", media_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if result.returncode != 0:
            continue
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
        if last_end > 0.0:
            return last_end
    return approximate_end


def _probe_selected_audio_end_time(ffprobe_path: str, media_path: str, audio_track_no: int | None) -> float:
    _, audio_stream_count = _probe_media_stream_layout(ffprobe_path, media_path)
    if audio_stream_count <= 0:
        return 0.0
    if audio_track_no is None:
        audio_indices = range(audio_stream_count)
    else:
        audio_indices = [max(0, min(audio_stream_count - 1, int(audio_track_no) - 1))]
    return max((_probe_audio_end_time(ffprobe_path, media_path, audio_index) for audio_index in audio_indices), default=0.0)


def _probe_selected_audio_overhang(ffprobe_path: str, media_path: str, audio_track_no: int | None, visible_duration_seconds: float) -> float:
    visible_duration_seconds = max(0.0, float(visible_duration_seconds or 0.0))
    _, audio_stream_count = _probe_media_stream_layout(ffprobe_path, media_path)
    if audio_stream_count <= 0:
        return 0.0
    if audio_track_no is None:
        audio_indices = range(audio_stream_count)
    else:
        audio_indices = [max(0, min(audio_stream_count - 1, int(audio_track_no) - 1))]
    video_end_seconds = _probe_primary_video_start_time(ffprobe_path, media_path) + visible_duration_seconds
    seam_start = max(0.0, video_end_seconds - 1.0)
    for probe_span in (2.0, 8.0, 32.0, 128.0):
        last_end = 0.0
        for audio_index in audio_indices:
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-select_streams",
                    f"a:{int(audio_index)}",
                    "-read_intervals",
                    f"{seam_start:.6f}%+{probe_span:.6f}",
                    "-show_packets",
                    "-show_entries",
                    "packet=pts_time,duration_time",
                    "-of",
                    "csv=p=0",
                    media_path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            if result.returncode != 0:
                continue
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
        if last_end > video_end_seconds:
            return max(0.0, last_end - video_end_seconds)
    return max(0.0, _probe_selected_audio_end_time(ffprobe_path, media_path, audio_track_no) - video_end_seconds)


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


def _start_video_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str, reserved_metadata_path: str | None = None) -> subprocess.Popen:
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
    ]
    metadata_input_index = None
    if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
        command += ["-f", "ffmetadata", "-i", reserved_metadata_path]
        metadata_input_index = 1
    if metadata_input_index is not None:
        command += ["-map_metadata", str(metadata_input_index)]
    command += ["-map", "0:v:0"]
    command += get_video_encode_args(video_codec, video_container)
    command += _get_live_mux_output_args(video_container)
    command += [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)


def _start_av_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str, source_path: str, start_seconds: float, audio_track_no: int | None, reserved_metadata_path: str | None = None) -> subprocess.Popen:
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
    ]
    metadata_input_index = None
    if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
        command += ["-f", "ffmetadata", "-i", reserved_metadata_path]
        metadata_input_index = 2
    else:
        metadata_input_index = 1
    command += ["-map_metadata", str(metadata_input_index), "-map", "0:v:0"]
    if audio_track_no is None:
        command += ["-map", "1:a?"]
    else:
        command += ["-map", f"1:a:{max(0, int(audio_track_no) - 1)}?"]
    command += get_video_encode_args(video_codec, video_container) + ["-c:a", "copy", "-shortest"]
    command += _get_live_mux_output_args(video_container)
    command += [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)


def _start_hdr_video_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str, reserved_metadata_path: str | None = None) -> subprocess.Popen:
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gbrpf32le",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{float(fps_float):.12g}",
        "-i",
        "pipe:0",
    ]
    metadata_input_index = None
    if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
        command += ["-f", "ffmetadata", "-i", reserved_metadata_path]
        metadata_input_index = 1
    if metadata_input_index is not None:
        command += ["-map_metadata", str(metadata_input_index)]
    command += ["-map", "0:v:0"]
    command += get_hdr_video_encode_args(video_codec, video_container)
    command += _get_live_mux_output_args(video_container)
    command += [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)


def _start_hdr_av_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str, source_path: str, start_seconds: float, audio_track_no: int | None, reserved_metadata_path: str | None = None) -> subprocess.Popen:
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gbrpf32le",
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
    ]
    if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
        command += ["-f", "ffmetadata", "-i", reserved_metadata_path, "-map_metadata", "2"]
    else:
        command += ["-map_metadata", "1"]
    command += ["-map", "0:v:0"]
    if audio_track_no is None:
        command += ["-map", "1:a?"]
    else:
        command += ["-map", f"1:a:{max(0, int(audio_track_no) - 1)}?"]
    command += get_hdr_video_encode_args(video_codec, video_container) + ["-c:a", "copy", "-shortest"]
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


def _mux_source_audio(ffmpeg_path: str, video_only_path: str, output_path: str, source_path: str, start_seconds: float, duration_seconds: float, audio_track_no: int | None, reserved_metadata_path: str | None = None, *, use_shortest: bool = True) -> None:
    temp_output_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_mux{Path(output_path).suffix}"))
    command = [ffmpeg_path, "-y", "-v", "error", "-i", video_only_path, "-ss", f"{max(0.0, float(start_seconds)):.12g}", "-t", f"{max(0.0, float(duration_seconds)):.12g}", "-i", source_path]
    if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
        command += ["-f", "ffmetadata", "-i", reserved_metadata_path]
        command += ["-map_metadata", "2"]
    else:
        command += ["-map_metadata", "1"]
    command += ["-map", "0:v:0"]
    if audio_track_no is None:
        command += ["-map", "1:a?"]
    else:
        command += ["-map", f"1:a:{max(0, int(audio_track_no) - 1)}?"]
    command += ["-c:v", "copy", "-c:a", "copy"]
    if use_shortest:
        command += ["-shortest"]
    if str(Path(output_path).suffix).strip().lower() == ".mp4":
        command += ["-movflags", "+faststart"]
    command += [temp_output_path]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(temp_output_path):
        if os.path.isfile(temp_output_path):
            os.remove(temp_output_path)
        raise gr.Error((result.stderr or result.stdout or "ffmpeg audio mux failed").strip())
    os.replace(temp_output_path, output_path)


def _make_output_sidecar_path(output_path: str, suffix: str) -> str:
    output = Path(output_path).resolve()
    return str(output.with_name(f"{output.name}{suffix}"))


def _make_video_only_output_path(output_path: str) -> str:
    output = Path(output_path).resolve()
    return str(output.with_name(f"{output.stem}_videoonly{output.suffix}"))


def _create_reserved_metadata_file(output_path: str) -> str:
    reserved_metadata_path = _make_output_sidecar_path(output_path, ".ffmeta")
    write_reserved_video_ffmetadata(reserved_metadata_path, DEFAULT_RESERVED_VIDEO_METADATA_BYTES)
    return reserved_metadata_path


def _delete_file_if_exists(file_path: str | None, *, label: str) -> None:
    if not isinstance(file_path, str) or not os.path.isfile(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        print(f"[Process Full Video] Warning: failed to delete {label} {file_path}: {exc}")


def _store_output_metadata(output_path: str, last_segment_path: str | None, *, source_path: str, process_name: str, source_start_seconds: float, start_frame: int, fps_float: float, selected_audio_track: int | None, total_generation_time: float, actual_frame_count: int, process_metadata: dict | None = None, verbose_level: int = 0) -> None:
    if not os.path.isfile(output_path):
        return
    if not last_segment_path or not os.path.isfile(last_segment_path):
        print(f"[Process Full Video] Warning: no segment metadata source was available for {output_path}")
        return
    metadata = read_metadata_from_video(last_segment_path)
    if not isinstance(metadata, dict) or len(metadata) == 0:
        print(f"[Process Full Video] Warning: failed to read WanGP metadata from {last_segment_path}")
        return
    final_metadata = metadata.copy()
    source_name = os.path.basename(source_path)
    end_frame = max(int(start_frame), int(start_frame) + max(0, int(actual_frame_count)) - 1)
    start_seconds = max(0.0, float(source_start_seconds or 0.0))
    end_seconds = start_seconds + max(0, int(actual_frame_count)) / float(fps_float)
    final_metadata["video_guide"] = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=end_frame, audio_track_no=selected_audio_track)
    final_metadata["video_length"] = int(actual_frame_count)
    final_metadata["frame_count"] = int(actual_frame_count)
    final_metadata["generation_time"] = max(0.0, float(total_generation_time))
    operation_comment = f'{PlugIn_Name}: {process_name} on "{source_name}" Start { _format_time_hms(start_seconds) } End { _format_time_hms(end_seconds) }'
    existing_comments = str(final_metadata.get("comments") or "").strip()
    final_metadata["comments"] = operation_comment if len(existing_comments) == 0 else f"{existing_comments}\n{operation_comment}"
    final_metadata["segments"] = [{
        "plugin": PlugIn_Name,
        "process": process_name,
        "source_file": source_name,
        "start_frame": int(start_frame),
        "end_frame": end_frame,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
    }]
    final_metadata[PROCESS_FULL_VIDEO_METADATA_KEY] = process_metadata.copy() if isinstance(process_metadata, dict) else {}
    for key in ("plugin", "process", "source_video", "source_segment", "video_source"):
        final_metadata.pop(key, None)
    final_metadata["creation_date"] = datetime.now().isoformat(timespec="seconds")
    final_metadata["creation_timestamp"] = int(time.time())
    if not save_video_metadata(output_path, final_metadata, allow_inplace_update=True, verbose_level=verbose_level):
        print(f"[Process Full Video] Warning: failed to write metadata to {output_path}")


def _read_metadata_generation_time(video_path: str | None) -> float:
    if not video_path or not os.path.isfile(video_path):
        return 0.0
    metadata = read_metadata_from_video(video_path)
    if not isinstance(metadata, dict):
        return 0.0
    try:
        return max(0.0, float(metadata.get("generation_time") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _get_last_generated_video_path(paths: list[str]) -> str | None:
    video_paths = [str(Path(path).resolve()) for path in paths if isinstance(path, str) and os.path.isfile(path) and str(Path(path).suffix).lower() in {".mp4", ".mkv"}]
    return video_paths[-1] if len(video_paths) > 0 else None


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


def _probe_primary_video_start_time(ffprobe_path: str, media_path: str) -> float:
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if result.returncode == 0:
        try:
            stream = (((json.loads(result.stdout) or {}).get("streams") or [{}])[0])
            start_time = stream.get("start_time")
            if start_time not in (None, "", "N/A"):
                return max(0.0, float(start_time))
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            pass
    packet_times = _probe_video_packet_times(ffprobe_path, media_path, start_seconds=0.0, duration_seconds=1.0)
    return max(0.0, min(packet_times)) if len(packet_times) > 0 else 0.0


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
            fine_seek_seconds = 0.0
            if trim_seconds > 0.0:
                coarse_seek_seconds = max(0.0, trim_seconds - 1.0)
                fine_seek_seconds = trim_seconds - coarse_seek_seconds
                if coarse_seek_seconds > 0.0:
                    command += ["-ss", f"{coarse_seek_seconds:.12g}"]
            command += ["-i", segment_path]
            if fine_seek_seconds > 0.0:
                command += ["-ss", f"{fine_seek_seconds:.12g}"]
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


def _concat_video_segments(
    ffmpeg_path: str,
    segment_paths: list[str],
    output_path: str,
    video_codec: str,
    video_container: str,
    audio_codec_key: str,
    *,
    segment_audio_trim_seconds: list[float] | None = None,
    segment_audio_duration_seconds: list[float | None] | None = None,
    fps_float: float | None = None,
    selected_audio_track_no: int | None = None,
    reserved_metadata_path: str | None = None,
    source_audio_path: str | None = None,
    source_audio_start_seconds: float | None = None,
    source_audio_duration_seconds: float | None = None,
    source_audio_track_no: int | None = None,
) -> None:
    segment_paths = [str(Path(path).resolve()) for path in segment_paths if isinstance(path, str) and os.path.isfile(path)]
    if len(segment_paths) == 0:
        raise gr.Error("No output segments available to merge.")
    if len(segment_paths) == 1:
        if str(Path(segment_paths[0]).resolve()) != str(Path(output_path).resolve()):
            try:
                os.replace(segment_paths[0], output_path)
            except OSError as exc:
                raise _ContinuationMergeOutputLockedError(output_path) from exc
        return
    ffprobe_path = resolve_media_binary("ffprobe")
    layouts = [_probe_media_stream_layout(ffprobe_path, path) for path in segment_paths]
    if any(video_count != 1 for video_count, _ in layouts):
        raise gr.Error("All continuation segments must contain exactly one video stream.")
    use_source_audio_merge = isinstance(source_audio_path, str) and len(str(source_audio_path).strip()) > 0
    if use_source_audio_merge:
        _validate_audio_copy_container(ffprobe_path, str(source_audio_path), video_container, source_audio_track_no)
    audio_stream_counts = [audio_count for _, audio_count in layouts]
    has_audio = use_source_audio_merge or any(audio_count > 0 for audio_count in audio_stream_counts)
    if not use_source_audio_merge and has_audio and any(audio_count <= 0 for audio_count in audio_stream_counts):
        raise gr.Error("All continuation segments must expose an audio stream.")
    fps_value = float(fps_float or 0.0)
    concat_dir = _make_output_temp_dir(output_path, "wangp_process_full_video_concat_")
    merged_video_path = os.path.join(concat_dir, "merged_video.mkv")
    temp_output_path = os.path.join(concat_dir, f"{Path(output_path).stem}_merged{Path(output_path).suffix}")
    video_track_timescale = None
    try:
        video_codec_name = _probe_primary_video_codec(ffprobe_path, segment_paths[0])
        video_bsf = "h264_mp4toannexb" if video_codec_name == "h264" else "hevc_mp4toannexb" if video_codec_name in ("hevc", "h265") else ""
        if len(video_bsf) == 0:
            raise gr.Error(f"Unsupported continuation video codec for no-reencode merge: {video_codec_name}")
        if use_source_audio_merge:
            command = [ffmpeg_path, "-y", "-v", "error", "-f", "mpegts", "-i", "pipe:0", "-ss", f"{max(0.0, float(source_audio_start_seconds or 0.0)):.12g}", "-t", f"{max(0.0, float(source_audio_duration_seconds or 0.0)):.12g}", "-i", str(source_audio_path)]
            if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
                command += ["-f", "ffmetadata", "-i", reserved_metadata_path, "-map_metadata", "2"]
            else:
                command += ["-map_metadata", "1"]
            command += ["-map", "0:v:0"]
            if source_audio_track_no is None:
                command += ["-map", "1:a?"]
            else:
                command += ["-map", f"1:a:{max(0, int(source_audio_track_no) - 1)}?"]
            command += ["-c", "copy"]
            if _normalize_container_name(video_container) == "mp4":
                command += ["-movflags", "+faststart"]
            command += [temp_output_path]
            mux_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)
            try:
                if mux_process.stdin is None:
                    raise gr.Error("ffmpeg source-audio merge did not expose a writable video pipe.")
                for segment_path in segment_paths:
                    segment_process = subprocess.Popen([ffmpeg_path, "-v", "error", "-i", segment_path, "-map", "0:v:0", "-c", "copy", "-bsf:v", video_bsf, "-f", "mpegts", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                    try:
                        if segment_process.stdout is None:
                            raise gr.Error(f"ffmpeg failed to expose the TS stream for {segment_path}.")
                        shutil.copyfileobj(segment_process.stdout, mux_process.stdin, length=1024 * 1024)
                    except Exception:
                        segment_process.kill()
                        segment_process.wait(timeout=5)
                        raise
                    finally:
                        if segment_process.stdout is not None:
                            segment_process.stdout.close()
                    segment_returncode = segment_process.wait()
                    segment_stderr = segment_process.stderr.read().decode("utf-8", errors="ignore").strip() if segment_process.stderr is not None else ""
                    if segment_returncode != 0:
                        raise gr.Error((segment_stderr or f"ffmpeg failed to stream {segment_path} for concat").strip())
                mux_returncode, mux_stderr, _ = _finalize_mux_process(mux_process)
            except Exception:
                try:
                    if mux_process.stdin is not None and not mux_process.stdin.closed:
                        mux_process.stdin.close()
                except OSError:
                    pass
                mux_process.kill()
                mux_process.wait(timeout=5)
                raise
            if mux_returncode != 0 or not os.path.isfile(temp_output_path):
                raise gr.Error((mux_stderr or "ffmpeg source-audio merge failed").strip())
        else:
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
                merged_audio_path = os.path.join(concat_dir, "merged_audio.mka")
                selected_audio_index = max(0, int(selected_audio_track_no or 1) - 1)
                audio_stream_indices = [0 if audio_count <= 1 else min(audio_count - 1, selected_audio_index) for audio_count in audio_stream_counts]
                resolved_audio_duration_seconds = [None] * len(segment_paths)
                if segment_audio_duration_seconds is not None:
                    for segment_index, duration_seconds in enumerate(segment_audio_duration_seconds[:len(segment_paths)]):
                        if duration_seconds is not None:
                            resolved_audio_duration_seconds[segment_index] = max(0.0, float(duration_seconds))
                elif fps_value > 0.0:
                    for segment_index, segment_path in enumerate(segment_paths):
                        segment_frame_count, _ = _probe_resume_frame_count(ffprobe_path, segment_path, fps_value)
                        if segment_frame_count > 0:
                            resolved_audio_duration_seconds[segment_index] = float(segment_frame_count) / fps_value
                _concat_audio_segments(ffmpeg_path, segment_paths, merged_audio_path, concat_dir, segment_trim_seconds=segment_audio_trim_seconds, segment_duration_seconds=resolved_audio_duration_seconds, audio_stream_indices=audio_stream_indices)
                command += ["-i", merged_audio_path]
            if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
                command += ["-f", "ffmetadata", "-i", reserved_metadata_path]
                command += ["-map_metadata", "2" if has_audio else "1"]
            else:
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
        try:
            os.replace(temp_output_path, output_path)
        except OSError as exc:
            raise _ContinuationMergeOutputLockedError(output_path) from exc
    finally:
        shutil.rmtree(concat_dir, ignore_errors=True)


def _merge_residual_continuations(
    ffmpeg_path: str,
    ffprobe_path: str,
    output_path: str,
    continuation_paths: list[str],
    *,
    video_codec: str,
    video_container: str,
    audio_codec_key: str,
    fps_float: float,
    selected_audio_track_no: int | None,
    source_audio_path: str | None = None,
    source_audio_start_seconds: float | None = None,
    merged_continuation_signatures: list[dict] | None = None,
) -> tuple[list[dict], list[str], list[str]]:
    known_signatures = _normalize_merged_continuation_signatures(merged_continuation_signatures)
    newly_merged_signatures: list[dict] = []
    undeleted_already_merged_paths: list[str] = []
    undeleted_newly_merged_paths: list[str] = []
    for continuation_path in continuation_paths:
        continuation_signature = _make_continuation_signature(continuation_path)
        if continuation_signature is not None and _continuation_signature_key(continuation_signature) in { _continuation_signature_key(signature) for signature in known_signatures }:
            if os.path.isfile(continuation_path):
                try:
                    os.remove(continuation_path)
                except OSError:
                    undeleted_already_merged_paths.append(continuation_path)
            continue
        completed_frames, _ = _probe_resume_frame_count(ffprobe_path, output_path, fps_float)
        continuation_frames, _ = _probe_resume_frame_count(ffprobe_path, continuation_path, fps_float)
        merged_duration_seconds = (float(completed_frames + continuation_frames) / fps_float) if completed_frames > 0 or continuation_frames > 0 else 0.0
        audio_trim_seconds = _probe_selected_audio_overhang(ffprobe_path, output_path, selected_audio_track_no, completed_frames / fps_float) if completed_frames > 0 else 0.0
        if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE:
            print(f"[Process Full Video] Residual merge: rebuilding audio from source for {merged_duration_seconds:.6f}s starting at {float(source_audio_start_seconds or 0.0):.6f}s")
        elif audio_trim_seconds > 0.0:
            print(f"[Process Full Video] Residual merge: trimming {audio_trim_seconds:.6f}s from continuation audio and clamping merged segment audio to visible video duration")
        _concat_video_segments(
            ffmpeg_path,
            [output_path, continuation_path],
            output_path,
            video_codec,
            video_container,
            audio_codec_key,
            segment_audio_trim_seconds=[0.0, audio_trim_seconds],
            segment_audio_duration_seconds=[(float(completed_frames) / fps_float) if completed_frames > 0 else None, (float(continuation_frames) / fps_float) if continuation_frames > 0 else None],
            fps_float=fps_float,
            selected_audio_track_no=selected_audio_track_no,
            source_audio_path=source_audio_path if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE else None,
            source_audio_start_seconds=source_audio_start_seconds if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE else None,
            source_audio_duration_seconds=merged_duration_seconds if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE else None,
            source_audio_track_no=selected_audio_track_no if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE else None,
        )
        if continuation_signature is not None:
            known_signatures = _append_merged_continuation_signature(known_signatures, continuation_signature)
            newly_merged_signatures = _append_merged_continuation_signature(newly_merged_signatures, continuation_signature)
        if os.path.isfile(continuation_path):
            try:
                os.remove(continuation_path)
            except OSError:
                undeleted_newly_merged_paths.append(continuation_path)
    return newly_merged_signatures, undeleted_already_merged_paths, undeleted_newly_merged_paths


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
    total_chunks = int(total_chunks)
    completed_chunks = int(completed_chunks)
    current_chunk = int(current_chunk)
    if total_chunks > 0:
        total_chunks = max(1, total_chunks)
        completed_chunks = max(0, min(completed_chunks, total_chunks))
        current_chunk = max(1, min(current_chunk, total_chunks))
        top_ratio = completed_chunks / total_chunks
        chunks_text = f"{completed_chunks} / {total_chunks}"
    else:
        completed_chunks = max(0, completed_chunks)
        current_chunk = max(0, current_chunk)
        top_ratio = 0.0
        chunks_text = "- / -"
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
        f"<div style='font-weight:600'>Chunks Processed: {chunks_text}{continued_suffix}</div>"
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
        self.request_global("get_model_def")
        self.request_global("get_lora_dir")
        self.request_global("get_base_model_type")
        self.request_global("get_current_model_settings")
        self.request_global("server_config")
        self.request_component("state")
        self.request_component("lset_name")
        self.request_component("refresh_form_trigger")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_config_ui)

    def on_tab_select(self, state: dict) -> str:
        return str(time.time_ns())

    def create_config_ui(self, api_session):
        if PROCESS_DEFINITIONS_ERROR is not None:
            with gr.Blocks() as plugin_blocks:
                gr.Markdown(f"Process settings configuration error: {html.escape(PROCESS_DEFINITIONS_ERROR)}")
            return plugin_blocks
        get_model_def = getattr(self, "get_model_def", None)
        get_lora_dir = getattr(self, "get_lora_dir", None)
        get_base_model_type = getattr(self, "get_base_model_type", None)
        get_current_model_settings = getattr(self, "get_current_model_settings", None)
        output_resolution_choices = [("1080p", "1080p"), ("900p", "900p"), ("720p", "720p"), ("540p", "540p"), ("480p", "480p"), ("360p", "360p"), ("256p", "256p")]
        output_resolution_values = {value for _, value in output_resolution_choices}
        source_audio_track_choices = [("Auto", "")] + [(f"Audio Track {track_no}", str(track_no)) for track_no in range(1, 10)]
        source_audio_track_values = {value for _, value in source_audio_track_choices}
        ratio_values = {value for _, value in RATIO_CHOICES}

        def _get_model_type_label(model_type: str) -> str:
            if len(str(model_type or "").strip()) == 0:
                return "Unknown Model"
            try:
                model_def = _require_model_def(str(model_type), get_model_def)
            except gr.Error:
                return str(model_type)
            model_block = model_def.get("model")
            if isinstance(model_block, dict):
                model_name = str(model_block.get("name") or "").strip()
                if len(model_name) > 0:
                    return model_name
            model_name = str(model_def.get("name") or "").strip()
            return model_name if len(model_name) > 0 else str(model_type)

        def _get_base_model_type_for_ref(model_type: str) -> str:
            model_type = str(model_type or "").strip()
            if callable(get_base_model_type):
                try:
                    base_model_type = str(get_base_model_type(model_type) or "").strip()
                    if len(base_model_type) > 0:
                        return base_model_type
                except Exception:
                    pass
            return model_type

        def _resolve_user_settings_ref(ref: str) -> Path | None:
            ref = _normalize_user_settings_ref(ref)
            if len(ref) == 0 or not callable(get_lora_dir):
                return None
            base_model_type, filename = ref.split("/", 1)
            try:
                lora_dir = Path(get_lora_dir(base_model_type))
            except Exception:
                return None
            settings_path = (lora_dir / Path(filename).name).resolve()
            if settings_path.is_file() and settings_path.suffix.lower() == ".json":
                return settings_path
            return None

        def _load_settings_payload(settings_path: Path) -> dict | None:
            try:
                payload = json.loads(settings_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            return payload if isinstance(payload, dict) else None

        def _build_user_process_definition(ref: str) -> dict | None:
            normalized_ref = _normalize_user_settings_ref(ref)
            settings_path = _resolve_user_settings_ref(normalized_ref)
            if settings_path is None:
                return None
            payload = _load_settings_payload(settings_path)
            if not isinstance(payload, dict):
                return None
            model_type = str(payload.get("model_type") or "").strip()
            if len(model_type) == 0:
                return None
            return {
                "settings": payload,
                "path": str(settings_path),
                "source": "user",
                "ref": normalized_ref,
                "name": settings_path.stem,
                "value": _user_process_value(normalized_ref),
            }

        def _get_user_process_definitions(user_refs: list[str]) -> dict[str, dict]:
            definitions: dict[str, dict] = {}
            for ref in _get_saved_user_settings_refs({USER_SETTINGS_STORAGE_KEY: user_refs}):
                definition = _build_user_process_definition(ref)
                if definition is None:
                    continue
                value = str(definition.get("value") or "")
                if len(value) > 0:
                    definitions[value] = definition
            return definitions

        def _get_system_process_definition(process_name: str) -> dict | None:
            process_definition = PROCESS_DEFINITIONS.get(str(process_name or "").strip())
            if not isinstance(process_definition, dict):
                return None
            return {
                "settings": process_definition.get("settings", {}),
                "path": process_definition.get("path", ""),
                "source": "system",
                "name": str(process_name or "").strip(),
                "value": str(process_name or "").strip(),
            }

        def _get_main_model_type(main_state: dict | None) -> str:
            if isinstance(main_state, dict):
                key = "model_type" if main_state.get("active_form", "add") == "add" else "edit_model_type"
                model_type = str(main_state.get(key) or main_state.get("model_type") or "").strip()
                if len(model_type) > 0:
                    return model_type
            return ""

        def _get_main_lset_name(main_state: dict | None, main_lset_name: str | None = None) -> str:
            lset_name = str(main_lset_name or "").strip()
            if len(lset_name) > 0:
                return lset_name
            if callable(get_current_model_settings) and isinstance(main_state, dict):
                try:
                    settings = get_current_model_settings(main_state)
                    if isinstance(settings, dict):
                        return str(settings.get("lset_name") or "").strip()
                except Exception:
                    pass
            return ""

        def _get_current_user_settings_filenames(main_state: dict | None) -> list[str]:
            loras_presets = main_state.get("loras_presets", []) if isinstance(main_state, dict) else []
            if not isinstance(loras_presets, list):
                return []
            filenames: list[str] = []
            seen: set[str] = set()
            for item in loras_presets:
                filename = str(item or "").strip()
                if "/" in filename or "\\" in filename or not filename.lower().endswith(".json"):
                    continue
                if filename.casefold() in seen:
                    continue
                filenames.append(filename)
                seen.add(filename.casefold())
            return sorted(filenames, key=lambda name: Path(name).stem.casefold())

        def _normalize_main_lset_selection(main_state: dict | None, main_lset_name: str | None) -> str:
            selection = str(main_lset_name or "").strip()
            filenames = _get_current_user_settings_filenames(main_state)
            if selection in filenames:
                return selection
            normalized_label = selection.replace("\u2500", "").replace(chr(160), " ").strip().casefold()
            for filename in filenames:
                if Path(filename).stem.casefold() == normalized_label:
                    return filename
            return ""

        def _resolve_current_user_settings_file(main_state: dict | None, settings_filename: str) -> Path | None:
            filename = str(settings_filename or "").strip()
            if filename not in _get_current_user_settings_filenames(main_state) or not callable(get_lora_dir):
                return None
            model_type = _get_main_model_type(main_state)
            if len(model_type) == 0:
                return None
            try:
                lora_dir = Path(get_lora_dir(model_type))
            except Exception:
                return None
            settings_path = (lora_dir / Path(filename).name).resolve()
            if settings_path.is_file() and settings_path.suffix.lower() == ".json":
                return settings_path
            return None

        def _build_current_user_settings_ref(main_state: dict | None, settings_filename: str) -> str:
            model_type = _get_main_model_type(main_state)
            base_model_type = _get_base_model_type_for_ref(model_type)
            filename = Path(str(settings_filename or "").strip()).name
            return _normalize_user_settings_ref(f"{base_model_type}/{filename}")

        def _build_candidate_user_process_definition(main_state: dict | None, settings_filename: str) -> dict | None:
            settings_path = _resolve_current_user_settings_file(main_state, settings_filename)
            if settings_path is None:
                return None
            payload = _load_settings_payload(settings_path)
            if not isinstance(payload, dict):
                return None
            model_type = str(payload.get("model_type") or _get_main_model_type(main_state)).strip()
            if len(model_type) == 0:
                return None
            ref = _build_current_user_settings_ref(main_state, settings_path.name)
            return {
                "settings": payload,
                "path": str(settings_path),
                "source": "user",
                "ref": ref,
                "name": settings_path.stem,
                "value": settings_path.name,
            }

        def _get_process_definition(process_value: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> dict | None:
            process_value = str(process_value or "").strip()
            if len(process_value) == 0 or process_value == NO_USER_SETTINGS_VALUE:
                return None
            system_definition = _get_system_process_definition(process_value)
            if system_definition is not None:
                return system_definition
            if _is_user_process_value(process_value):
                ref = _user_process_ref_from_value(process_value)
                if len(ref) == 0:
                    return None
                if user_refs is not None and ref.casefold() not in {item.casefold() for item in _get_saved_user_settings_refs({USER_SETTINGS_STORAGE_KEY: user_refs})}:
                    return None
                return _build_user_process_definition(ref)
            if process_value.lower().endswith(".json"):
                return _build_candidate_user_process_definition(main_state, process_value)
            return None

        def _process_definition_model_type(process_definition: dict | None) -> str:
            settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
            return str(settings.get("model_type") or "").strip() if isinstance(settings, dict) else ""

        def _get_process_values_by_model_type(user_refs: list[str]) -> dict[str, list[str]]:
            values_by_model_type: dict[str, list[str]] = {}
            for process_name, process_definition in PROCESS_DEFINITIONS.items():
                model_type = str(process_definition.get("settings", {}).get("model_type") or "").strip()
                if len(model_type) > 0:
                    values_by_model_type.setdefault(model_type, []).append(process_name)
            for value, definition in _get_user_process_definitions(user_refs).items():
                model_type = _process_definition_model_type(definition)
                if len(model_type) > 0:
                    values_by_model_type.setdefault(model_type, []).append(value)
            return values_by_model_type

        def _get_model_type_choices(user_refs: list[str]) -> list[tuple[str, str]]:
            model_types = sorted(_get_process_values_by_model_type(user_refs), key=lambda item: _get_model_type_label(item).casefold())
            choices = [(_get_model_type_label(model_type), model_type) for model_type in model_types]
            choices.append((ADD_USER_SETTINGS_LABEL, ADD_USER_SETTINGS_MODEL_TYPE))
            return choices

        def _normal_process_choices(model_type: str, user_refs: list[str]) -> list[tuple[str, str]]:
            model_type = str(model_type or "").strip()
            entries: list[tuple[str, str, str]] = []
            for process_name, process_definition in PROCESS_DEFINITIONS.items():
                if str(process_definition.get("settings", {}).get("model_type") or "").strip() == model_type:
                    entries.append((process_name, process_name, "system"))
            for value, definition in _get_user_process_definitions(user_refs).items():
                if _process_definition_model_type(definition) == model_type:
                    label = f"{Path(str(definition.get('name') or value)).stem} *"
                    entries.append((label, value, "user"))
            entries.sort(key=lambda item: (item[0].removesuffix(" *").casefold(), 0 if item[2] == "system" else 1))
            return [(label, value) for label, value, _source in entries]

        def _current_user_settings_choices(main_state: dict | None, main_lset_name: str | None) -> tuple[list[tuple[str, str]], str]:
            filenames = _get_current_user_settings_filenames(main_state)
            if len(filenames) == 0:
                return [(NO_USER_SETTINGS_LABEL, NO_USER_SETTINGS_VALUE)], NO_USER_SETTINGS_VALUE
            choices = [(Path(filename).stem, filename) for filename in filenames]
            selected = _normalize_main_lset_selection(main_state, main_lset_name)
            value = selected if selected in filenames else filenames[0]
            return choices, value

        def _get_process_choices(process_model_type: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str]) -> tuple[list[tuple[str, str]], str | None]:
            if str(process_model_type or "").strip() == ADD_USER_SETTINGS_MODEL_TYPE:
                return _current_user_settings_choices(main_state, main_lset_name)
            choices = _normal_process_choices(str(process_model_type or "").strip(), user_refs)
            return choices, choices[0][1] if len(choices) > 0 else None

        def _process_choices_have_user_settings(process_choices: list[tuple[str, str]]) -> bool:
            return any(_is_user_process_value(value) for _label, value in list(process_choices or []))

        def _user_settings_hint_update(process_choices: list[tuple[str, str]]):
            return gr.update(visible=_process_choices_have_user_settings(process_choices))

        def _settings_action_updates(process_model_type_value: str, process_value: str):
            add_visible = str(process_model_type_value or "").strip() == ADD_USER_SETTINGS_MODEL_TYPE
            delete_visible = _is_user_process_value(process_value)
            placeholder_visible = not add_visible and not delete_visible
            return gr.update(visible=add_visible), gr.update(visible=delete_visible), gr.update(visible=placeholder_visible)

        saved_ui_settings = _load_saved_process_full_video_settings()
        initial_user_refs = _get_saved_user_settings_refs(saved_ui_settings)
        saved_process_name = str(saved_ui_settings.get("process_name") or "").strip()
        saved_model_type = str(saved_ui_settings.get("process_model_type") or "").strip()
        saved_process_definition = _get_process_definition(saved_process_name, getattr(self.state, "value", None), initial_user_refs)
        if saved_process_definition is not None:
            saved_model_type = _process_definition_model_type(saved_process_definition) or saved_model_type
        process_names_by_model_type = _get_process_values_by_model_type(initial_user_refs)
        default_model_type = saved_model_type if saved_model_type in process_names_by_model_type else DEFAULT_MODEL_TYPE if DEFAULT_MODEL_TYPE in process_names_by_model_type else next(iter(process_names_by_model_type), DEFAULT_MODEL_TYPE)
        default_process_choices = _normal_process_choices(default_model_type, initial_user_refs)
        default_process_values = [value for _label, value in default_process_choices]
        default_process_name = saved_process_name if saved_process_name in default_process_values else DEFAULT_PROCESS_NAME if DEFAULT_PROCESS_NAME in default_process_values else (default_process_values[0] if default_process_values else DEFAULT_PROCESS_NAME)
        default_process_definition = _get_process_definition(default_process_name, getattr(self.state, "value", None), initial_user_refs) or _get_system_process_definition(DEFAULT_PROCESS_NAME)
        default_process_settings = default_process_definition.get("settings", {}) if isinstance(default_process_definition, dict) else {}
        overlap_step = _get_vae_temporal_latent_size(default_model_type, get_model_def)
        overlap_max = _get_overlap_slider_max(default_model_type, get_model_def)
        default_overlap_value = _coerce_int(saved_ui_settings.get("sliding_window_overlap"), int(default_process_settings.get("sliding_window_overlap") or 1), minimum=1)
        default_overlap_value = _normalize_overlap_frames(default_overlap_value, frame_step=overlap_step)
        default_overlap_value = min(max(1, default_overlap_value), overlap_max)
        default_source_path = str(saved_ui_settings.get("source_path") or DEFAULT_SOURCE_PATH)
        saved_process_strength = saved_ui_settings.get("process_strength", saved_ui_settings.get("control_video_strength"))
        default_process_strength = _get_default_process_strength(default_process_settings) if saved_process_strength is None else _coerce_float(saved_process_strength, _get_default_process_strength(default_process_settings))
        default_output_path = str(saved_ui_settings.get("output_path") or DEFAULT_OUTPUT_PATH)
        default_continue_enabled = _coerce_bool(saved_ui_settings.get("continue_enabled"), True)
        default_output_resolution = str(saved_ui_settings.get("output_resolution") or "720p").strip()
        default_output_resolution = default_output_resolution if default_output_resolution in output_resolution_values else "720p"
        default_target_ratio = str(saved_ui_settings.get("target_ratio") or "4:3").strip()
        default_target_ratio = default_target_ratio if default_target_ratio in ratio_values else "4:3"
        default_chunk_size_seconds = _coerce_float(saved_ui_settings.get("chunk_size_seconds"), 10.0, minimum=0.1)
        template_default_prompt = str(default_process_settings.get("prompt") or "")
        default_prompt = str(saved_ui_settings.get("prompt")) if "prompt" in saved_ui_settings else template_default_prompt
        default_start_seconds = "" if saved_ui_settings.get("start_seconds") in (None, "") else str(saved_ui_settings.get("start_seconds"))
        default_end_seconds = "" if saved_ui_settings.get("end_seconds") in (None, "") else str(saved_ui_settings.get("end_seconds"))
        default_source_audio_track = str(saved_ui_settings.get("source_audio_track") or "").strip()
        default_source_audio_track = default_source_audio_track if default_source_audio_track in source_audio_track_values else ""
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

        def _info_exit(message: str, *, output=ui_skip, total_chunks: int = 1, completed_chunks: int = 0, current_chunk: int = 1, continued: bool = False):
            gr.Info(str(message or "").strip())
            return _ui_update(_render_chunk_status_html(total_chunks, completed_chunks, current_chunk, "Info", str(message or "").strip(), continued=continued), output, ui_skip, start_enabled=True, abort_enabled=False)

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

        def _get_default_process_definition() -> dict:
            definition = _get_system_process_definition(DEFAULT_PROCESS_NAME)
            if definition is not None:
                return definition
            for process_name in PROCESS_DEFINITIONS:
                definition = _get_system_process_definition(process_name)
                if definition is not None:
                    return definition
            return {"settings": {}, "path": "", "source": "system", "name": "", "value": ""}

        def _get_process_definition_or_default(process_name: str, main_state: dict | None, user_refs: list[str] | None) -> dict:
            return _get_process_definition(process_name, main_state, user_refs) or _get_default_process_definition()

        def _has_process_outpaint(process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> bool:
            process_definition = _get_process_definition(process_name, main_state, user_refs)
            settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
            return isinstance(settings, dict) and "video_guide_outpainting" in settings

        def _get_target_ratio_update(process_name: str, main_state: dict | None, user_refs: list[str] | None, target_ratio: str | None = None):
            visible = _has_process_outpaint(process_name, main_state, user_refs)
            return gr.update(value=target_ratio if visible else "", visible=visible, choices=RATIO_CHOICES if visible else RATIO_CHOICES_WITH_EMPTY)

        def _get_process_strength_update(process_name: str, main_state: dict | None, user_refs: list[str] | None, process_strength: float | None = None):
            visible = not _has_process_outpaint(process_name, main_state, user_refs)
            value = 1.0 if not visible else _coerce_float(process_strength, 1.0)
            return gr.update(value=value, visible=visible)

        def _get_overlap_control_updates(process_name: str, main_state: dict | None, user_refs: list[str] | None):
            process_definition = _get_process_definition_or_default(process_name, main_state, user_refs)
            settings = process_definition.get("settings", {})
            model_type = str(settings.get("model_type") or "")
            step = _get_vae_temporal_latent_size(model_type, get_model_def)
            maximum = _get_overlap_slider_max(model_type, get_model_def)
            value = _coerce_int(settings.get("sliding_window_overlap"), 1, minimum=1)
            return gr.update(minimum=1, maximum=maximum, step=step, value=min(max(1, _normalize_overlap_frames(value, frame_step=step)), maximum))

        def _get_process_dropdown_update(model_type: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str]):
            process_choices, process_value = _get_process_choices(model_type, main_state, main_lset_name, user_refs)
            return gr.update(choices=process_choices, value=process_value)

        def _build_process_form_state(process_name: str, raw_state: dict | None = None, main_state: dict | None = None, user_refs: list[str] | None = None) -> dict:
            process_definition = _get_process_definition_or_default(process_name, main_state, user_refs)
            process_settings = process_definition.get("settings", {})
            model_type = str(process_settings.get("model_type") or DEFAULT_MODEL_TYPE)
            step = _get_vae_temporal_latent_size(model_type, get_model_def)
            maximum = _get_overlap_slider_max(model_type, get_model_def)
            default_state = {
                "process_model_type": model_type,
                "process_name": process_name,
                "source_path": DEFAULT_SOURCE_PATH,
                "process_strength": _get_default_process_strength(process_settings),
                "output_path": DEFAULT_OUTPUT_PATH,
                "prompt": str(process_settings.get("prompt") or ""),
                "continue_enabled": True,
                "source_audio_track": "",
                "output_resolution": "720p",
                "target_ratio": "4:3",
                "chunk_size_seconds": 10.0,
                "sliding_window_overlap": min(max(1, _normalize_overlap_frames(_coerce_int(process_settings.get("sliding_window_overlap"), 1, minimum=1), frame_step=step)), maximum),
                "start_seconds": "",
                "end_seconds": "",
            }
            raw_state = raw_state if isinstance(raw_state, dict) else {}
            default_state["source_path"] = str(raw_state.get("source_path") or default_state["source_path"])
            saved_process_strength = raw_state.get("process_strength", raw_state.get("control_video_strength"))
            if saved_process_strength is not None:
                default_state["process_strength"] = _coerce_float(saved_process_strength, default_state["process_strength"])
            default_state["output_path"] = str(raw_state.get("output_path") or default_state["output_path"])
            if "prompt" in raw_state:
                default_state["prompt"] = str(raw_state.get("prompt") or "")
            default_state["continue_enabled"] = _coerce_bool(raw_state.get("continue_enabled"), default_state["continue_enabled"])
            source_audio_track = str(raw_state.get("source_audio_track") or "").strip()
            default_state["source_audio_track"] = source_audio_track if source_audio_track in source_audio_track_values else default_state["source_audio_track"]
            output_resolution = str(raw_state.get("output_resolution") or "").strip()
            default_state["output_resolution"] = output_resolution if output_resolution in output_resolution_values else default_state["output_resolution"]
            target_ratio = str(raw_state.get("target_ratio") or "").strip()
            default_state["target_ratio"] = target_ratio if target_ratio in ratio_values else default_state["target_ratio"]
            default_state["chunk_size_seconds"] = _coerce_float(raw_state.get("chunk_size_seconds"), default_state["chunk_size_seconds"], minimum=0.1)
            overlap_value = _coerce_int(raw_state.get("sliding_window_overlap"), default_state["sliding_window_overlap"], minimum=1)
            default_state["sliding_window_overlap"] = min(max(1, _normalize_overlap_frames(overlap_value, frame_step=step)), maximum)
            default_state["start_seconds"] = "" if raw_state.get("start_seconds") in (None, "") else str(raw_state.get("start_seconds"))
            default_state["end_seconds"] = "" if raw_state.get("end_seconds") in (None, "") else str(raw_state.get("end_seconds"))
            return default_state

        def _snapshot_form_state(process_name: str, main_state: dict | None, user_refs: list[str] | None, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds) -> dict:
            return _build_process_form_state(process_name, {
                "source_path": source_path,
                "process_strength": process_strength,
                "output_path": output_path,
                "prompt": prompt_text,
                "continue_enabled": continue_enabled,
                "source_audio_track": source_audio_track,
                "output_resolution": output_resolution,
                "target_ratio": target_ratio,
                "chunk_size_seconds": chunk_size_seconds,
                "sliding_window_overlap": sliding_window_overlap,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
            }, main_state, user_refs)

        def _store_process_form_memory(memory_state: dict | None, current_process_name: str, main_state: dict | None, user_refs: list[str] | None, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds):
            updated_memory = dict(memory_state) if isinstance(memory_state, dict) else {}
            current_process_name = str(current_process_name or "").strip()
            if _get_process_definition(current_process_name, main_state, user_refs) is not None:
                updated_memory[current_process_name] = _snapshot_form_state(current_process_name, main_state, user_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds)
            return updated_memory

        def _restore_process_form_state(memory_state: dict | None, process_name: str, current_source_path: str, main_state: dict | None, user_refs: list[str] | None):
            state = _build_process_form_state(process_name, (memory_state or {}).get(process_name), main_state, user_refs)
            overlap_update = _get_overlap_control_updates(process_name, main_state, user_refs)
            target_ratio_update = _get_target_ratio_update(process_name, main_state, user_refs, state["target_ratio"])
            process_strength_update = _get_process_strength_update(process_name, main_state, user_refs, state["process_strength"])
            source_path_value = str(current_source_path or "").strip() or state["source_path"]
            return source_path_value, process_strength_update, state["output_path"], state["prompt"], state["continue_enabled"], state["source_audio_track"], state["output_resolution"], target_ratio_update, state["chunk_size_seconds"], overlap_update, state["start_seconds"], state["end_seconds"]

        def _change_process_model_type(memory_state: dict | None, current_process_name: str, next_model_type: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str] | None, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds):
            refs = _get_saved_user_settings_refs({USER_SETTINGS_STORAGE_KEY: user_refs})
            updated_memory = _store_process_form_memory(memory_state, current_process_name, main_state, refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds)
            process_choices, next_process_name = _get_process_choices(next_model_type, main_state, main_lset_name, refs)
            next_process_name = str(next_process_name or NO_USER_SETTINGS_VALUE).strip()
            restored = _restore_process_form_state(updated_memory, next_process_name, source_path, main_state, refs)
            add_update, delete_update, placeholder_update = _settings_action_updates(next_model_type, next_process_name)
            return (
                updated_memory,
                next_process_name,
                gr.update(choices=process_choices, value=next_process_name),
                _user_settings_hint_update(process_choices),
                add_update,
                delete_update,
                placeholder_update,
                *restored,
            )

        def _change_process_name(memory_state: dict | None, current_process_name: str, next_process_name: str, process_model_type_value: str, main_state: dict | None, user_refs: list[str] | None, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds):
            refs = _get_saved_user_settings_refs({USER_SETTINGS_STORAGE_KEY: user_refs})
            updated_memory = _store_process_form_memory(memory_state, current_process_name, main_state, refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds)
            next_process_name = str(next_process_name or "").strip()
            restored = _restore_process_form_state(updated_memory, next_process_name, source_path, main_state, refs)
            _add_update, delete_update, placeholder_update = _settings_action_updates(process_model_type_value, next_process_name)
            return (
                updated_memory,
                next_process_name,
                delete_update,
                placeholder_update,
                *restored,
            )

        def _refresh_process_form_from_main(_refresh_id, memory_state: dict | None, current_process_name: str, process_model_type_value: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str] | None, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds):
            refs = _get_saved_user_settings_refs({USER_SETTINGS_STORAGE_KEY: user_refs})
            model_choices = _get_model_type_choices(refs)
            process_model_type_value = str(process_model_type_value or "").strip()
            if process_model_type_value != ADD_USER_SETTINGS_MODEL_TYPE:
                valid_model_values = {value for _label, value in model_choices}
                model_value = process_model_type_value if process_model_type_value in valid_model_values else default_model_type
                return (
                    gr.update(choices=model_choices, value=model_value),
                    gr.update(),
                    gr.update(),
                    memory_state,
                    current_process_name,
                    gr.update(visible=False),
                    gr.update(visible=_is_user_process_value(current_process_name)),
                    gr.update(visible=not _is_user_process_value(current_process_name)),
                    *[gr.skip()] * 12,
                )
            updated_memory = _store_process_form_memory(memory_state, current_process_name, main_state, refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds)
            process_choices, next_process_name = _current_user_settings_choices(main_state, main_lset_name)
            restored = _restore_process_form_state(updated_memory, next_process_name, source_path, main_state, refs)
            return (
                gr.update(choices=model_choices, value=ADD_USER_SETTINGS_MODEL_TYPE),
                gr.update(choices=process_choices, value=next_process_name),
                _user_settings_hint_update(process_choices),
                updated_memory,
                next_process_name,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                *restored,
            )

        def _validate_user_process_definition(process_definition: dict | None) -> list[str]:
            settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
            if not isinstance(settings, dict):
                return ["The selected settings file could not be read."]
            model_type = str(settings.get("model_type") or "").strip()
            if len(model_type) == 0:
                return ["The selected settings file does not define a model."]
            try:
                model_def = _require_model_def(model_type, get_model_def)
            except gr.Error as exc:
                return [_get_error_message(exc) or "The selected model is not available."]
            problems: list[str] = []
            image_mode = _coerce_int(settings.get("image_mode"), 0)
            if image_mode != 0 or bool(model_def.get("audio_only", False)) or bool(model_def.get("image_outputs", False)) or bool(settings.get("image_outputs", False)):
                problems.append("The selected settings must generate a video, not images or audio only.")
            video_prompt_type = str(settings.get("video_prompt_type") or "")
            if "V" not in video_prompt_type:
                problems.append("Control Video must be enabled.")
            if "I" in video_prompt_type:
                problems.append("Reference Images must be disabled.")
            image_prompt_types_allowed = str(model_def.get("image_prompt_types_allowed", "") or "")
            if "V" not in image_prompt_types_allowed:
                problems.append("The selected model must support using a source video for continuation.")
            audio_prompt_type = str(settings.get("audio_prompt_type") or "")
            audio_features = []
            if "A" in audio_prompt_type:
                audio_features.append("Audio Source")
            if "B" in audio_prompt_type:
                audio_features.append("Audio Source #2")
            if "X" in audio_prompt_type:
                audio_features.append("two-speaker auto separation")
            if len(audio_features) > 0:
                problems.append("Disable these audio features: " + ", ".join(audio_features) + ".")
            return problems

        def _format_user_process_validation_error(process_definition: dict | None, problems: list[str]) -> str:
            name = str((process_definition or {}).get("name") or "selected settings").strip()
            if len(problems) == 0:
                return ""
            return f'Cannot add "{name}" as a full-video process:\n- ' + "\n- ".join(problems)

        def _add_user_process_link(process_value: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str] | None):
            if str(process_value or "").strip() == NO_USER_SETTINGS_VALUE:
                raise gr.Error("No user settings are available to add.")
            process_definition = _get_process_definition(process_value, main_state, user_refs)
            if process_definition is None:
                raise gr.Error("The selected user settings file could not be found.")
            problems = _validate_user_process_definition(process_definition)
            if len(problems) > 0:
                raise gr.Error(_format_user_process_validation_error(process_definition, problems))
            ref = _normalize_user_settings_ref(process_definition.get("ref"))
            if len(ref) == 0:
                raise gr.Error("The selected user settings file could not be linked.")
            refs = _get_saved_user_settings_refs({USER_SETTINGS_STORAGE_KEY: user_refs})
            model_label = _get_model_type_label(_process_definition_model_type(process_definition))
            if ref.casefold() not in {item.casefold() for item in refs}:
                refs.append(ref)
                _store_user_settings_refs(refs)
                gr.Info(f'User settings "{process_definition.get("name")}" have been added for {model_label}.')
            else:
                gr.Info(f'User settings "{process_definition.get("name")}" are already linked for {model_label}.')
            process_choices, selected = _current_user_settings_choices(main_state, main_lset_name)
            if str(process_value or "").strip() in {value for _label, value in process_choices}:
                selected = str(process_value or "").strip()
            return (
                refs,
                gr.update(choices=_get_model_type_choices(refs), value=ADD_USER_SETTINGS_MODEL_TYPE),
                gr.update(choices=process_choices, value=selected),
                _user_settings_hint_update(process_choices),
                selected,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def _select_after_user_process_delete(deleted_process_value: str, deleted_model_type: str, old_refs: list[str], new_refs: list[str]) -> tuple[str, str, list[tuple[str, str]]]:
            old_choices = _normal_process_choices(deleted_model_type, old_refs)
            deleted_index = next((index for index, (_label, value) in enumerate(old_choices) if value == deleted_process_value), -1)
            new_choices = _normal_process_choices(deleted_model_type, new_refs)
            new_values = {value for _label, value in new_choices}
            if deleted_index >= 0:
                for _label, value in old_choices[deleted_index + 1:]:
                    if _is_user_process_value(value) and value in new_values:
                        return deleted_model_type, value, new_choices
            first_system_value = next((value for _label, value in new_choices if not _is_user_process_value(value)), None)
            if first_system_value is not None:
                return deleted_model_type, first_system_value, new_choices
            for model_type in sorted(_get_process_values_by_model_type(new_refs), key=lambda item: _get_model_type_label(item).casefold()):
                choices = _normal_process_choices(model_type, new_refs)
                first_system_value = next((value for _label, value in choices if not _is_user_process_value(value)), None)
                if first_system_value is not None:
                    return model_type, first_system_value, choices
            if len(new_choices) > 0:
                return deleted_model_type, new_choices[0][1], new_choices
            return default_model_type, default_process_name, _normal_process_choices(default_model_type, new_refs)

        def _delete_user_process_link(memory_state: dict | None, process_value: str, main_state: dict | None, user_refs: list[str] | None, source_path: str):
            process_value = str(process_value or "").strip()
            ref = _user_process_ref_from_value(process_value)
            if len(ref) == 0:
                raise gr.Error("Choose a linked user settings process to remove.")
            old_refs = _get_saved_user_settings_refs({USER_SETTINGS_STORAGE_KEY: user_refs})
            deleted_definition = _build_user_process_definition(ref)
            deleted_model_type = _process_definition_model_type(deleted_definition)
            new_refs = [item for item in old_refs if item.casefold() != ref.casefold()]
            _store_user_settings_refs(new_refs)
            model_type, next_process_name, process_choices = _select_after_user_process_delete(process_value, deleted_model_type, old_refs, new_refs)
            restored = _restore_process_form_state(memory_state, next_process_name, source_path, main_state, new_refs)
            gr.Info(f'Removed user settings "{Path(ref).stem}".')
            add_update, delete_update, placeholder_update = _settings_action_updates(model_type, next_process_name)
            return (
                new_refs,
                gr.update(choices=_get_model_type_choices(new_refs), value=model_type),
                gr.update(choices=process_choices, value=next_process_name),
                _user_settings_hint_update(process_choices),
                next_process_name,
                add_update,
                delete_update,
                placeholder_update,
                *restored,
            )

        model_type_choices = _get_model_type_choices(initial_user_refs)
        initial_process_form_memory = {default_process_name: _build_process_form_state(default_process_name, {
            "source_path": default_source_path,
            "process_strength": default_process_strength,
            "output_path": default_output_path,
            "prompt": default_prompt,
            "continue_enabled": default_continue_enabled,
            "source_audio_track": default_source_audio_track,
            "output_resolution": default_output_resolution,
            "target_ratio": default_target_ratio,
            "chunk_size_seconds": default_chunk_size_seconds,
            "sliding_window_overlap": default_overlap_value,
            "start_seconds": default_start_seconds,
            "end_seconds": default_end_seconds,
        }, getattr(self.state, "value", None), initial_user_refs)}

        def start_process(state, process_name, user_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds):
            process_definition = _get_process_definition(process_name, state, user_refs)
            if process_definition is None:
                yield _info_exit(f"Unsupported process: {process_name}")
                return
            if process_definition.get("source") == "user":
                problems = _validate_user_process_definition(process_definition)
                if len(problems) > 0:
                    yield _info_exit(_format_user_process_validation_error(process_definition, problems))
                    return
            process_settings = process_definition["settings"]
            model_type = str(process_settings.get("model_type") or "")
            if len(model_type) == 0:
                yield _info_exit(f"Unsupported process: {process_name}")
                return
            process_display_name = str(process_definition.get("name") or process_name or "").strip()
            process_is_hdr = VIDEO_PROMPT_HDR_OUTPUT_FLAG in str(process_settings.get("video_prompt_type") or "")
            has_outpaint = "video_guide_outpainting" in process_settings
            is_user_process = process_definition.get("source") == "user"
            source_path = str(source_path or "").strip()
            output_path = str(output_path or "").strip()
            source_audio_track = str(source_audio_track or "").strip()
            output_resolution = str(output_resolution or "").strip()
            target_ratio = str(target_ratio or "").strip()
            prompt_text = str(prompt_text or "")
            start_seconds = "" if start_seconds in (None, "") else str(start_seconds)
            end_seconds = "" if end_seconds in (None, "") else str(end_seconds)
            active_process_strength = 1.0 if has_outpaint else _coerce_float(process_strength, 1.0)
            try:
                _save_process_full_video_ui_settings({
                    "process_model_type": model_type,
                    "process_name": str(process_name or "").strip(),
                    "source_path": source_path,
                    "process_strength": active_process_strength,
                    "output_path": output_path,
                    "prompt": prompt_text,
                    "continue_enabled": bool(continue_enabled),
                    "source_audio_track": source_audio_track,
                    "output_resolution": output_resolution,
                    "target_ratio": target_ratio,
                    "chunk_size_seconds": _coerce_float(chunk_size_seconds, 10.0, minimum=0.1),
                    "sliding_window_overlap": _coerce_int(sliding_window_overlap, int(process_settings.get("sliding_window_overlap") or 1), minimum=1),
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                })
            except OSError as exc:
                yield _info_exit(f"Unable to save plugin settings to {PROCESS_FULL_VIDEO_SETTINGS_FILE}: {exc}")
                return
            active_target_ratio = target_ratio if has_outpaint else ""
            default_prompt_text = str(process_settings.get("prompt") or "")
            if len(prompt_text.strip()) == 0:
                prompt_text = default_prompt_text
            if not os.path.isfile(source_path):
                yield _info_exit(f"Source video not found: {source_path}")
                return
            try:
                start_seconds = _parse_time_input(start_seconds, label="Start", allow_empty=False)
                end_seconds = _parse_time_input(end_seconds, label="End", allow_empty=True)
            except gr.Error as exc:
                yield _info_exit(_get_error_message(exc) or "Invalid start/end selection.")
                return
            try:
                prompt_schedule = _parse_prompt_schedule(prompt_text)
            except gr.Error as exc:
                yield _info_exit(_get_error_message(exc) or f"Invalid prompt syntax.\n\nExample:\n{TIMED_PROMPT_EXAMPLE}")
                return
            started_ui = False
            preflight_stage = True
            try:
                clear_virtual_media_source(PROCESS_FULL_VIDEO_VSOURCE)
                yield _ui_update(_render_chunk_status_html(1, 0, 1, "Initializing", "Preparing processing job..."), ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=True)
                started_ui = True
                try:
                    verbose_level = _get_mmgp_verbose_level()
                    try:
                        metadata = get_video_info_details(source_path)
                    except Exception as exc:
                        raise gr.Error(f"Unable to read source video metadata: {source_path}") from exc
                    start_frame, end_frame_exclusive, fps_float, total_source_frames = _compute_selected_frame_range(metadata, start_seconds, end_seconds)
                    processing_fps = _get_processing_fps(fps_float)
                    try:
                        audio_track_count = int(extract_audio_tracks(source_path, query_only=True))
                    except Exception as exc:
                        raise gr.Error(f"Unable to inspect source audio tracks in: {source_path}") from exc
                    selected_audio_track = None
                    source_audio_track = str(source_audio_track or "").strip()
                    if len(source_audio_track) > 0:
                        if not source_audio_track.isdigit() or int(source_audio_track) <= 0:
                            raise gr.Error("Source Audio must be Auto or a whole track number.")
                        if audio_track_count <= 0:
                            raise gr.Error("Source video contains no audio track. Leave Source Audio empty.")
                        selected_audio_track = int(source_audio_track)
                    elif audio_track_count > 0:
                        selected_audio_track = 1
                    if selected_audio_track is not None and (selected_audio_track <= 0 or selected_audio_track > audio_track_count):
                        raise gr.Error(f"Source Audio must be between 1 and {audio_track_count}.")
                    frame_plan_rules = _get_frame_plan_rules(model_type, get_model_def)
                    budget_resolution = _choose_resolution(str(output_resolution))
                    try:
                        chunk_seconds_value = float(chunk_size_seconds or 10.0)
                    except (TypeError, ValueError) as exc:
                        raise gr.Error("Chunk Size must be a number of seconds.") from exc
                    try:
                        overlap_value = float(sliding_window_overlap or 1.0)
                    except (TypeError, ValueError) as exc:
                        raise gr.Error("Sliding Window Overlap must be a number of frames.") from exc
                    chunk_frames = _normalize_chunk_frames(chunk_seconds_value, processing_fps, frame_step=frame_plan_rules.frame_step, minimum_requested_frames=frame_plan_rules.minimum_requested_frames)
                    overlap_frames = _normalize_overlap_frames(overlap_value, frame_step=frame_plan_rules.frame_step)
                    if overlap_frames >= chunk_frames:
                        raise gr.Error(f"Sliding Window Overlap must stay below the computed chunk size ({chunk_frames} frame(s)).")
                    selected_unique_frames = end_frame_exclusive - start_frame
                    full_plans = _build_chunk_plan(
                        start_frame,
                        end_frame_exclusive,
                        total_source_frames,
                        chunk_frames,
                        frame_step=frame_plan_rules.frame_step,
                        minimum_requested_frames=frame_plan_rules.minimum_requested_frames,
                        overlap_frames=overlap_frames,
                    )
                    requested_unique_frames = _count_planned_unique_frames(full_plans)
                    requested_source_segment = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=max(int(start_frame), int(start_frame) + max(0, int(requested_unique_frames)) - 1), audio_track_no=selected_audio_track)
                    requested_output_path = str(_build_requested_output_path(source_path, output_path, process_display_name, active_target_ratio, str(output_resolution), start_seconds, end_seconds, has_outpaint=has_outpaint))
                    identity_mismatch_message = _get_output_identity_mismatch_message(requested_output_path, process_name=process_display_name, source_path=source_path, source_segment=requested_source_segment)
                    if identity_mismatch_message is not None:
                        yield _info_exit(identity_mismatch_message, output=requested_output_path)
                        return
                    output_path, resume_existing_output = _resolve_output_path(source_path, output_path, process_display_name, active_target_ratio, str(output_resolution), start_seconds, end_seconds, bool(continue_enabled), has_outpaint=has_outpaint)
                    try:
                        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        raise gr.Error(f"Unable to create the output folder for: {output_path}") from exc
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
                    if selected_audio_track is not None:
                        _validate_audio_copy_container(ffprobe_path, source_path, output_container, selected_audio_track)
                    merged_continuation_signatures = _read_merged_continuation_signatures(output_path)
                except gr.Error as exc:
                    yield _info_exit(_get_error_message(exc) or "Invalid process settings.", output=output_path if isinstance(output_path, str) and os.path.isfile(output_path) else ui_skip)
                    return
                preflight_stage = False
                if resume_existing_output:
                        residual_continuation_paths = _list_residual_continuation_paths(output_path)
                        if residual_continuation_paths:
                            residual_names = ", ".join(Path(path).name for path in residual_continuation_paths)
                            known_signature_keys = { _continuation_signature_key(signature) for signature in merged_continuation_signatures }
                            known_residual_paths = []
                            for residual_path in residual_continuation_paths:
                                residual_signature = _make_continuation_signature(residual_path)
                                if residual_signature is not None and _continuation_signature_key(residual_signature) in known_signature_keys:
                                    known_residual_paths.append(residual_path)
                            all_residual_paths_are_known = len(known_residual_paths) == len(residual_continuation_paths)
                            if all_residual_paths_are_known:
                                _plugin_info(f"Found already-merged continuation file(s) still on disk: {residual_names}. Checking whether they can be deleted before continuing.")
                                yield _ui_update(_render_chunk_status_html(max(1, len(full_plans)), 0, 1, "Checking Continuations", f"Checking already-merged continuation file(s): {residual_names}"), output_path if os.path.isfile(output_path) else ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=True)
                            else:
                                _plugin_info(f"Found residual continuation file(s) from a previous unfinished merge: {residual_names}. Merging them into {Path(output_path).name} before continuing.")
                                yield _ui_update(_render_chunk_status_html(max(1, len(full_plans)), 0, 1, "Recovering Continuation", f"Recovering unfinished continuation merge: {residual_names}"), output_path if os.path.isfile(output_path) else ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=True)
                            try:
                                new_merged_signatures, undeleted_already_merged_paths, undeleted_newly_merged_paths = _merge_residual_continuations(
                                    ffmpeg_path,
                                    ffprobe_path,
                                    output_path,
                                    residual_continuation_paths,
                                    video_codec=self.server_config.get("video_output_codec", "libx264_8"),
                                    video_container=output_container,
                                    audio_codec_key=self.server_config.get("audio_output_codec", "aac_128"),
                                    fps_float=fps_float,
                                    selected_audio_track_no=selected_audio_track,
                                    source_audio_path=source_path if selected_audio_track is not None else None,
                                    source_audio_start_seconds=(start_frame / fps_float) if selected_audio_track is not None else None,
                                    merged_continuation_signatures=merged_continuation_signatures,
                                )
                                if len(new_merged_signatures) > 0:
                                    for signature in new_merged_signatures:
                                        merged_continuation_signatures = _append_merged_continuation_signature(merged_continuation_signatures, signature)
                                    _store_merged_continuation_signatures(output_path, merged_continuation_signatures, verbose_level=verbose_level)
                                if len(undeleted_already_merged_paths) > 0:
                                    undeleted_names = ", ".join(Path(path).name for path in undeleted_already_merged_paths)
                                    _plugin_info(f"Detected already-merged continuation file(s) still on disk, but they could not be deleted because they are still open: {undeleted_names}. Delete them manually when they are released.")
                                if len(undeleted_newly_merged_paths) > 0:
                                    undeleted_names = ", ".join(Path(path).name for path in undeleted_newly_merged_paths)
                                    _plugin_info(f"Merged residual continuation file(s) into {Path(output_path).name}, but these continuation file(s) could not be deleted because they are still open: {undeleted_names}. Delete them manually when they are released.")
                            except _ContinuationMergeOutputLockedError:
                                locked_message = f"{Path(output_path).name} is open, so the pending continuation merge could not replace it. Release the base file and start a process again."
                                gr.Info(locked_message)
                                yield _ui_update(_render_chunk_status_html(max(1, len(full_plans)), 0, 1, "Merge Pending", locked_message), output_path, str(time.time_ns()), start_enabled=True, abort_enabled=False)
                                return
                            except Exception as exc:
                                raise gr.Error(f"Failed to merge the residual continuation file(s) before resuming. Please close any player using {output_path} and retry. {exc}") from exc
                mux_process = None
                stopped = False
                reserved_metadata_path = None
                last_frame_image = None
                last_segment_path = None
                continuation_output_path = ""
                chunk_output_paths: list[str] = []
                written_unique_frames = 0
                resumed_unique_frames = 0
                resume_overlap_frames = 0
                completed_chunks = 0
                resolved_resolution = ""
                resolved_width = 0
                resolved_height = 0
                mux_finished = False
                merged_continuation = False
                resume_audio_trim_seconds = 0.0
                preview_state["image"] = None
                output_path_for_write = output_path
                video_only_output_path = _make_video_only_output_path(output_path_for_write)
                exact_start_seconds = start_frame / fps_float
                if resume_existing_output:
                    yield _ui_update(_render_chunk_status_html(max(1, len(full_plans)), 0, 1, "Inspecting Existing Output", f"Inspecting existing output to continue: {output_path}"), output_path, str(time.time_ns()))
                    _log_existing_output_metadata(output_path, verbose_level)
                    resumed_unique_frames, resume_reason = _probe_resume_frame_count(ffprobe_path, output_path, fps_float)
                    recorded_written_unique_frames = _read_recorded_written_unique_frames(output_path)
                    if recorded_written_unique_frames > resumed_unique_frames:
                        _plugin_info(f"Using recorded output progress from metadata: {recorded_written_unique_frames} frame(s) instead of the probed {resumed_unique_frames} frame(s).")
                        resumed_unique_frames = recorded_written_unique_frames
                    resumed_unique_frames = max(0, min(requested_unique_frames, resumed_unique_frames))
                    if resumed_unique_frames <= 0:
                        _plugin_info(f"Unable to continue from existing output: {output_path}. {resume_reason or 'Starting a new file instead.'}")
                        output_path = _make_output_variant(Path(output_path))
                        output_path_for_write = output_path
                        video_only_output_path = _make_video_only_output_path(output_path_for_write)
                        resumed_unique_frames = 0
                        resume_overlap_frames = 0
                        completed_chunks = 0
                        exact_start_seconds = start_frame / fps_float
                        resume_existing_output = False
                    else:
                        _plugin_info(f"Continuing existing output: {output_path}")
                        resolved_resolution, resolved_width, resolved_height = _probe_existing_output_resolution(output_path)
                        print(f"[Process Full Video] Continuing with locked output resolution {resolved_resolution}")
                        completed_chunks, _ = _count_completed_chunks(full_plans, resumed_unique_frames)
                        exact_start_seconds = (start_frame + resumed_unique_frames) / fps_float
                        if resumed_unique_frames < requested_unique_frames:
                            yield _ui_update(_render_chunk_status_html(max(1, len(full_plans)), 0, 1, "Loading Overlap Frames", f"Continuing existing output with {resumed_unique_frames} frame(s) already written."), output_path, str(time.time_ns()))
                            resumed_unique_frames, last_frame_image, tail_reason = _resolve_resume_last_frame(output_path, resumed_unique_frames)
                            if resumed_unique_frames <= 0 or last_frame_image is None:
                                _plugin_info(f"Unable to continue from existing output: {output_path}. {tail_reason or 'Starting a new file instead.'}")
                                output_path = _make_output_variant(Path(output_path))
                                output_path_for_write = output_path
                                video_only_output_path = _make_video_only_output_path(output_path_for_write)
                                resumed_unique_frames = 0
                                resume_overlap_frames = 0
                                completed_chunks = 0
                                preview_state["image"] = None
                                exact_start_seconds = start_frame / fps_float
                                resume_existing_output = False
                            else:
                                if tail_reason:
                                    _plugin_info(tail_reason)
                                preview_state["image"] = last_frame_image
                                completed_chunks, _ = _count_completed_chunks(full_plans, resumed_unique_frames)
                                exact_start_seconds = (start_frame + resumed_unique_frames) / fps_float
                                resume_overlap_frames = min(overlap_frames, resumed_unique_frames)
                                overlap_tensor = _load_process_full_video_hdr_overlap_buffer(output_path, resume_overlap_frames, resumed_unique_frames) if process_is_hdr else _load_process_full_video_overlap_buffer(output_path, resume_overlap_frames, resumed_unique_frames)
                                if overlap_tensor is None:
                                    _plugin_info(f"Unable to continue from existing output: {output_path}. Failed to load the overlap frames from the recorded output.")
                                    output_path = _make_output_variant(Path(output_path))
                                    output_path_for_write = output_path
                                    video_only_output_path = _make_video_only_output_path(output_path_for_write)
                                    resumed_unique_frames = 0
                                    resume_overlap_frames = 0
                                    completed_chunks = 0
                                    preview_state["image"] = None
                                    exact_start_seconds = start_frame / fps_float
                                    resume_existing_output = False
                                else:
                                    resume_overlap_frames = int(overlap_tensor.shape[1])
                                    _set_process_full_video_overlap_buffer(overlap_tensor, processing_fps, hdr=process_is_hdr)
                                    print(f"[Process Full Video] Loaded overlap buffer from existing output: {_describe_frame_range(int(start_frame) + int(resumed_unique_frames) - int(resume_overlap_frames), resume_overlap_frames)}")
                        if resume_existing_output and resumed_unique_frames < requested_unique_frames:
                            if not USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE:
                                resume_audio_trim_seconds = _probe_selected_audio_overhang(ffprobe_path, output_path, selected_audio_track, resumed_unique_frames / fps_float)
                            if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None:
                                print(f"[Process Full Video] Final merge: rebuilding audio from source starting at {start_frame / fps_float:.6f}s")
                            elif resume_audio_trim_seconds > 0.0:
                                print(f"[Process Full Video] Final merge: trimming {resume_audio_trim_seconds:.6f}s from continuation audio and clamping segment audio to visible video duration")
                            remaining_resume_unique_frames = _align_total_unique_frames(
                                end_frame_exclusive - (start_frame + resumed_unique_frames),
                                frame_step=frame_plan_rules.frame_step,
                                minimum_requested_frames=frame_plan_rules.minimum_requested_frames,
                                initial_overlap_frames=resume_overlap_frames,
                            )
                            if remaining_resume_unique_frames <= 0:
                                trailing_frames = max(0, requested_unique_frames - resumed_unique_frames)
                                _plugin_info(f"Existing output already covers the remaining valid range. The last {trailing_frames} frame(s) are too short to build another continuation chunk for the current model.")
                                resumed_unique_frames = requested_unique_frames
                                plans = []
                            else:
                                continuation_output_path = _make_continuation_output_path(output_path)
                                output_path_for_write = continuation_output_path
                                video_only_output_path = _make_video_only_output_path(output_path_for_write)
                                plans = _build_chunk_plan(
                                    start_frame + resumed_unique_frames,
                                    end_frame_exclusive,
                                    total_source_frames,
                                    chunk_frames,
                                    frame_step=frame_plan_rules.frame_step,
                                    minimum_requested_frames=frame_plan_rules.minimum_requested_frames,
                                    overlap_frames=overlap_frames,
                                    initial_overlap_frames=resume_overlap_frames,
                                )
                        elif resume_existing_output:
                            plans = []
                if not resume_existing_output:
                    plans = full_plans
                continued_mode = resumed_unique_frames > 0
                use_live_av_mux = selected_audio_track is not None
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
                    actual_done = int(resumed_unique_frames) + int(written_unique_frames)
                    actual_control_start_frame = int(start_frame) + actual_done - int(plan.overlap_frames)
                    actual_control_end_frame = actual_control_start_frame + int(plan.requested_frames) - 1
                    overlap_buffer_start_frame = int(start_frame) + actual_done - int(plan.overlap_frames)
                    model_video_length = int(plan.requested_frames) if int(plan.overlap_frames) <= 0 else max(1, int(plan.requested_frames) - int(plan.overlap_frames) + 1)
                    needs_video_source = continued_mode or int(plan.overlap_frames) > 0
                    print(
                        f"[Process Full Video] Chunk {chunk_index}: control video {_describe_frame_range(actual_control_start_frame, int(plan.requested_frames))}; "
                        + (f"overlap buffer {_describe_frame_range(overlap_buffer_start_frame, int(plan.overlap_frames))}" if needs_video_source else "overlap buffer not used")
                    )
                    settings = copy.deepcopy(process_definition["settings"])
                    chunk_prompt_start_seconds = max(0.0, float(actual_done) / float(fps_float))
                    settings["model_type"] = model_type
                    settings["prompt"] = _resolve_prompt_for_chunk(prompt_schedule, chunk_prompt_start_seconds, default_prompt_text)
                    settings["resolution"] = resolved_resolution or budget_resolution
                    settings["video_length"] = model_video_length
                    settings["sliding_window_overlap"] = max(1, int(plan.overlap_frames))
                    settings["image_prompt_type"] = "V" if needs_video_source else ""
                    settings["audio_prompt_type"] = "K"
                    if is_user_process:
                        settings["force_fps"] = "control"
                    # Keep the plugin-side control cursor tied to frames actually written.
                    # WGP applies any extra_control_frames model behavior internally, so this
                    # stays correct for models that use it and for models where it is 0.
                    settings["video_guide"] = build_virtual_media_path(source_path, start_frame=actual_control_start_frame, end_frame=actual_control_end_frame, audio_track_no=selected_audio_track)
                    if has_outpaint:
                        settings["video_guide_outpainting_ratio"] = active_target_ratio
                    else:
                        settings.pop("video_guide_outpainting_ratio", None)
                        if not is_user_process:
                            settings["loras_multipliers"] = str(active_process_strength)
                    api_settings = settings.get("_api")
                    settings["_api"] = dict(api_settings) if isinstance(api_settings, dict) else {}
                    settings["_api"]["return_media"] = True
                    if needs_video_source:
                        settings["video_source"] = _build_process_full_video_source_path(hdr=process_is_hdr)
                    else:
                        settings.pop("video_source", None)
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
                    last_segment_path = _get_last_generated_video_path(list(result.generated_files)) or last_segment_path
                    returned_video_item = next((item for item in result.artifacts if item.video_tensor_hdr is not None), None) if process_is_hdr else next((item for item in result.artifacts if item.video_tensor_uint8 is not None), None)
                    returned_tensor = None if returned_video_item is None else (returned_video_item.video_tensor_hdr if process_is_hdr else returned_video_item.video_tensor_uint8)
                    if returned_video_item is None or not torch.is_tensor(returned_tensor):
                        raise gr.Error(f"Chunk {chunk_index} completed without returned video tensor data.")
                    video_tensor_hdr = returned_tensor.detach().cpu() if process_is_hdr else None
                    video_tensor_uint8 = tonemap_hdr_tensor_to_uint8(video_tensor_hdr) if process_is_hdr else returned_tensor.detach().cpu()
                    returned_frame_count = int(video_tensor_uint8.shape[1])
                    expected_frame_count = int(plan.requested_frames)
                    if not process_is_hdr and returned_frame_count < max(1, expected_frame_count - 1):
                        video_candidates = [path for path in result.generated_files if isinstance(path, str) and os.path.isfile(path) and str(Path(path).suffix).lower() in {".mp4", ".mkv", ".mov", ".avi"}]
                        if video_candidates:
                            decoded_tensor = _load_video_tensor_from_file(video_candidates[0])
                            decoded_frame_count = int(decoded_tensor.shape[1])
                            print(f"[Process Full Video] Chunk {chunk_index}: returned video tensor has {returned_frame_count} frame(s); decoded chunk file has {decoded_frame_count} frame(s)")
                            if decoded_frame_count >= max(1, expected_frame_count - 1):
                                video_tensor_uint8 = decoded_tensor
                                returned_frame_count = decoded_frame_count
                    print(f"[Process Full Video] Chunk {chunk_index}: returned video tensor has {returned_frame_count} frame(s); control video lasts {expected_frame_count} frame(s)")
                    chunk_width, chunk_height = _get_video_tensor_resolution(video_tensor_uint8)
                    chunk_resolution = f"{chunk_width}x{chunk_height}"
                    print(f"[Process Full Video] Chunk {chunk_index}: generated chunk resolution {chunk_resolution}")
                    if len(resolved_resolution) == 0:
                        resolved_resolution = chunk_resolution
                        resolved_width = chunk_width
                        resolved_height = chunk_height
                    elif chunk_resolution != resolved_resolution:
                        raise gr.Error(f"Chunk {chunk_index} changed output resolution from {resolved_resolution} to {chunk_resolution}.")
                    skip_frames = int(plan.overlap_frames)
                    remaining_unique_frames = requested_unique_frames - (int(resumed_unique_frames) + int(written_unique_frames))
                    frames_to_write = min(remaining_unique_frames, int(video_tensor_uint8.shape[1]) - skip_frames)
                    if frames_to_write <= 0:
                        continue
                    if mux_process is None:
                        reserved_metadata_path = _create_reserved_metadata_file(output_path_for_write)
                        if process_is_hdr:
                            hdr_video_crf = self.server_config.get("hdr_video_crf", 8)
                            mux_process = _start_hdr_av_mux_process(ffmpeg_path, output_path_for_write, resolved_width, resolved_height, fps_float, hdr_video_crf, output_container, source_path, exact_start_seconds, selected_audio_track, reserved_metadata_path) if use_live_av_mux else _start_hdr_video_mux_process(ffmpeg_path, video_only_output_path, resolved_width, resolved_height, fps_float, hdr_video_crf, output_container, reserved_metadata_path)
                        else:
                            mux_process = _start_av_mux_process(ffmpeg_path, output_path_for_write, resolved_width, resolved_height, fps_float, self.server_config.get("video_output_codec", "libx264_8"), output_container, source_path, exact_start_seconds, selected_audio_track, reserved_metadata_path) if use_live_av_mux else _start_video_mux_process(ffmpeg_path, video_only_output_path, resolved_width, resolved_height, fps_float, self.server_config.get("video_output_codec", "libx264_8"), output_container, reserved_metadata_path)
                    last_frame_tensor = _write_hdr_video_chunk(mux_process, video_tensor_hdr, start_frame=skip_frames, frame_count=frames_to_write) if process_is_hdr and video_tensor_hdr is not None else _write_video_chunk(mux_process, video_tensor_uint8, start_frame=skip_frames, frame_count=frames_to_write)
                    written_unique_frames += frames_to_write
                    last_frame_image = _frame_to_image(last_frame_tensor)
                    preview_state["image"] = last_frame_image
                    overlap_source_tensor = video_tensor_hdr if process_is_hdr and video_tensor_hdr is not None else video_tensor_uint8
                    next_overlap_tensor = _update_process_full_video_overlap_buffer(overlap_source_tensor[:, skip_frames:skip_frames + frames_to_write], overlap_frames, processing_fps, hdr=process_is_hdr)
                    if next_overlap_tensor is not None and int(next_overlap_tensor.shape[1]) > 0:
                        next_overlap_count = int(next_overlap_tensor.shape[1])
                        next_overlap_start_frame = int(start_frame) + int(resumed_unique_frames) + int(written_unique_frames) - next_overlap_count
                        print(f"[Process Full Video] Chunk {chunk_index}: next overlap buffer {_describe_frame_range(next_overlap_start_frame, next_overlap_count)}")
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
                    _delete_file_if_exists(output_path_for_write, label="continuation output")
                    raise gr.Error("ffmpeg created an empty continuation file.")
                if not use_live_av_mux and os.path.isfile(video_only_output_path):
                    if selected_audio_track is None:
                        try:
                            os.replace(video_only_output_path, output_path_for_write)
                        except OSError as exc:
                            raise gr.Error(f"Unable to finalize the written video-only segment: {output_path_for_write}") from exc
                    else:
                        yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Muxing Audio", "Muxing source audio into the written video segment...", continued=continued_mode, **_timing_kwargs()), output_path if os.path.isfile(output_path) else ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=False)
                        _mux_source_audio(ffmpeg_path, video_only_output_path, output_path_for_write, source_path, exact_start_seconds, written_unique_frames / fps_float, selected_audio_track, reserved_metadata_path)
                undeleted_merged_continuation_paths: list[str] = []
                if continuation_output_path and os.path.isfile(output_path_for_write):
                    continuation_signature = _make_continuation_signature(output_path_for_write)
                    try:
                        existing_output_generation_time = _read_metadata_generation_time(output_path)
                        merged_duration_seconds = float(resumed_unique_frames + written_unique_frames) / fps_float
                        yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Merging Continuation", "Merging the continued segment into the main output...", continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()), start_enabled=False, abort_enabled=False)
                        _concat_video_segments(
                            ffmpeg_path,
                            [output_path, output_path_for_write],
                            output_path,
                            self.server_config.get("video_output_codec", "libx264_8"),
                            output_container,
                            self.server_config.get("audio_output_codec", "aac_128"),
                            segment_audio_trim_seconds=[0.0, resume_audio_trim_seconds],
                            segment_audio_duration_seconds=[(float(resumed_unique_frames) / fps_float) if resumed_unique_frames > 0 else None, (float(written_unique_frames) / fps_float) if written_unique_frames > 0 else None],
                            fps_float=fps_float,
                            selected_audio_track_no=selected_audio_track,
                            reserved_metadata_path=reserved_metadata_path,
                            source_audio_path=source_path if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                            source_audio_start_seconds=(start_frame / fps_float) if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                            source_audio_duration_seconds=merged_duration_seconds if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                            source_audio_track_no=selected_audio_track if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                        )
                        merged_continuation = True
                        if continuation_signature is not None:
                            merged_continuation_signatures = _append_merged_continuation_signature(merged_continuation_signatures, continuation_signature)
                    except _ContinuationMergeOutputLockedError:
                        locked_message = f"{Path(output_path).name} is open, so the continuation merge could not replace it. Existing output was kept and {Path(output_path_for_write).name} was preserved. Release the base file and start a process again."
                        gr.Info(locked_message)
                        _delete_file_if_exists(reserved_metadata_path, label="reserved metadata file")
                        yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Merge Pending", locked_message, continued=continued_mode, **_timing_kwargs()), output_path_for_write, str(time.time_ns()), start_enabled=True, abort_enabled=False)
                        return
                    except Exception as exc:
                        raise gr.Error(f"Failed to finalize continued output. Existing output kept, and continuation was preserved at {continuation_output_path}. {exc}") from exc
                    if os.path.isfile(output_path_for_write):
                        try:
                            os.remove(output_path_for_write)
                        except OSError:
                            undeleted_merged_continuation_paths.append(output_path_for_write)
                            _plugin_info(f"Merged continuation progress into {Path(output_path).name}, but {Path(output_path_for_write).name} could not be deleted because it is still open. Delete it manually when released.")
                else:
                    existing_output_generation_time = 0.0
                _delete_file_if_exists(reserved_metadata_path, label="reserved metadata file")
                total_written_unique_frames = resumed_unique_frames + written_unique_frames
                metadata_target_path = output_path if merged_continuation or not continuation_output_path else output_path_for_write
                yield _ui_update(_render_chunk_status_html(total_chunks_display, completed_chunks, min(completed_chunks + 1, total_chunks_display), "Writing Metadata", "Writing final output metadata...", continued=continued_mode, **_timing_kwargs()), metadata_target_path if os.path.isfile(metadata_target_path) else output_path, str(time.time_ns()), start_enabled=False, abort_enabled=False)
                metadata_source_path = last_segment_path or _get_last_generated_video_path(chunk_output_paths)
                actual_output_frames = _probe_resume_frame_count(ffprobe_path, metadata_target_path, fps_float)[0] if os.path.isfile(metadata_target_path) else 0
                actual_output_frames = int(actual_output_frames or total_written_unique_frames)
                total_generation_time = existing_output_generation_time + _read_metadata_generation_time(metadata_source_path) if merged_continuation else _read_metadata_generation_time(metadata_source_path)
                process_metadata = {
                    "process": process_display_name,
                    "written_unique_frames": int(total_written_unique_frames),
                    "chunks": int(total_chunks_display),
                    "sliding_window_overlap": int(overlap_frames),
                    "start_seconds": float(start_seconds),
                    "end_seconds": float(start_seconds + (total_written_unique_frames / float(fps_float))),
                    "source_video": source_path,
                    "source_segment": requested_source_segment,
                    "merged_continuations": _normalize_merged_continuation_signatures(merged_continuation_signatures),
                }
                if process_is_hdr:
                    process_metadata["hdr"] = True
                _store_output_metadata(metadata_target_path, metadata_source_path, source_path=source_path, process_name=process_display_name, source_start_seconds=start_seconds, start_frame=start_frame, fps_float=fps_float, selected_audio_track=selected_audio_track, total_generation_time=total_generation_time, actual_frame_count=actual_output_frames, process_metadata=process_metadata, verbose_level=verbose_level)
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
                    _delete_file_if_exists(output_path_for_write, label="continuation output")
                _delete_file_if_exists(video_only_output_path, label="video-only output")
            except gr.Error as exc:
                active_job["job"] = None
                mux_process_local = locals().get("mux_process")
                output_path_for_write_local = locals().get("output_path_for_write")
                video_only_output_path_local = locals().get("video_only_output_path")
                reserved_metadata_path_local = locals().get("reserved_metadata_path")
                stopped_local = bool(locals().get("stopped"))
                mux_finished_local = bool(locals().get("mux_finished"))
                if mux_process_local is not None and not mux_finished_local and mux_process_local.poll() is None:
                    try:
                        _finalize_mux_process(mux_process_local)
                    except Exception:
                        pass
                if mux_process_local is not None and not stopped_local and isinstance(output_path_for_write_local, str) and mux_process_local.returncode not in (0, None) and os.path.isfile(output_path_for_write_local):
                    _delete_file_if_exists(output_path_for_write_local, label="continuation output")
                _delete_file_if_exists(video_only_output_path_local, label="video-only output")
                _delete_file_if_exists(reserved_metadata_path_local, label="reserved metadata file")
                status_message = _get_error_message(exc) or "Processing failed."
                preflight_failure = bool(locals().get("preflight_stage", False))
                if not started_ui:
                    gr.Info(status_message)
                    return
                if started_ui:
                    total_chunks_value = max(1, int(locals().get("total_chunks_display", 1) or 1))
                    completed_value = max(0, min(int(locals().get("completed_chunks", 0) or 0), total_chunks_value))
                    current_value = max(1, min(completed_value + 1, total_chunks_value))
                    continued_value = bool(int(locals().get("resumed_unique_frames", 0) or 0) > 0)
                    output_value = output_path if isinstance(locals().get("output_path"), str) and os.path.isfile(locals()["output_path"]) else ui_skip
                    if preflight_failure:
                        gr.Info(status_message)
                    yield _ui_update(_render_chunk_status_html(total_chunks_value, completed_value, current_value, "Info" if preflight_failure else "Error", status_message, continued=continued_value), output_value, ui_skip, start_enabled=True, abort_enabled=False)
                return
            except BaseException as exc:
                active_job["job"] = None
                mux_process_local = locals().get("mux_process")
                output_path_for_write_local = locals().get("output_path_for_write")
                video_only_output_path_local = locals().get("video_only_output_path")
                reserved_metadata_path_local = locals().get("reserved_metadata_path")
                stopped_local = bool(locals().get("stopped"))
                mux_finished_local = bool(locals().get("mux_finished"))
                if mux_process_local is not None and not mux_finished_local and mux_process_local.poll() is None:
                    try:
                        _finalize_mux_process(mux_process_local)
                    except Exception:
                        pass
                if mux_process_local is not None and not stopped_local and isinstance(output_path_for_write_local, str) and mux_process_local.returncode not in (0, None) and os.path.isfile(output_path_for_write_local):
                    _delete_file_if_exists(output_path_for_write_local, label="continuation output")
                _delete_file_if_exists(video_only_output_path_local, label="video-only output")
                _delete_file_if_exists(reserved_metadata_path_local, label="reserved metadata file")
                if started_ui:
                    total_chunks_value = max(1, int(locals().get("total_chunks_display", 1) or 1))
                    completed_value = max(0, min(int(locals().get("completed_chunks", 0) or 0), total_chunks_value))
                    current_value = max(1, min(completed_value + 1, total_chunks_value))
                    continued_value = bool(int(locals().get("resumed_unique_frames", 0) or 0) > 0)
                    status_message = _get_error_message(exc) or exc.__class__.__name__
                    output_value = output_path if isinstance(locals().get("output_path"), str) and os.path.isfile(locals()["output_path"]) else ui_skip
                    yield _ui_update(_render_chunk_status_html(total_chunks_value, completed_value, current_value, "Error", status_message, continued=continued_value), output_value, ui_skip, start_enabled=True, abort_enabled=False)
                raise
            finally:
                clear_virtual_media_source(PROCESS_FULL_VIDEO_VSOURCE)

        def stop_process():
            job = active_job.get("job")
            if job is not None and not job.done:
                _request_job_stop(job)
                _plugin_info("Stopping current processing job...")
                return gr.update(value="Start Process", interactive=False), gr.update(value="Stop", interactive=False)
            return gr.update(value="Start Process", interactive=True), gr.update(value="Stop", interactive=False)

        process_form_memory = gr.State(initial_process_form_memory)
        active_process_name_state = gr.State(default_process_name)
        user_process_refs = gr.State(initial_user_refs)
        with gr.Column():
            gr.HTML(
                """
                <style>
                #process-full-video-settings-actions {
                    align-self: flex-end !important;
                    margin-bottom: 1px;
                    padding-bottom: 4px !important;
                    gap: 4px;
                    width: 34px !important;
                    min-width: 34px !important;
                    max-width: 34px !important;
                }
                #process-full-video-settings-actions > .form {
                    padding: 0 !important;
                    border: 0 !important;
                    background: transparent !important;
                    box-shadow: none !important;
                }
                #process-full-video-settings-actions button {
                    width: 34px !important;
                    min-width: 34px !important;
                    max-width: 34px !important;
                    height: 34px;
                    min-height: 34px;
                    padding: 0 !important;
                }
                #process-full-video-settings-actions .process-full-video-settings-action-placeholder {
                    width: 34px;
                    min-width: 34px;
                    max-width: 34px;
                    height: 34px;
                    min-height: 34px;
                }
                #process-full-video-user-settings-hint-row {
                    height: 12px !important;
                    min-height: 0 !important;
                    max-height: 12px !important;
                    margin-top: -10px !important;
                    margin-bottom: -4px !important;
                    padding: 0 !important;
                    overflow: visible !important;
                }
                #process-full-video-user-settings-hint-row > .form {
                    padding: 0 !important;
                    border: 0 !important;
                    background: transparent !important;
                    box-shadow: none !important;
                    min-height: 0 !important;
                    overflow: visible !important;
                }
                #process-full-video-user-settings-hint-row .block,
                #process-full-video-user-settings-hint-row .html-container,
                #process-full-video-user-settings-hint-row .prose {
                    height: auto !important;
                    margin: 0 !important;
                    min-height: 0 !important;
                    padding: 0 !important;
                    overflow: visible !important;
                }
                </style>
                """
            )
            with gr.Row():
                gr.Markdown("This PlugIn is a *Super Sliding Windows* mode with *Low RAM requirements*, lossless Audio Copy and no risk to explode your Web Browser and the *Video Gallery* with huge files. You can stop a Process and Resume it later. You can define different prompts for different time range. However quite often the prompt should have little impact on the ouput.")
            with gr.Row():
                process_model_type = gr.Dropdown(model_type_choices, value=default_model_type, label="Model", scale=1)
                process_name = gr.Dropdown(default_process_choices, value=default_process_name, label="Process", scale=3)
                with gr.Column(scale=0, min_width=34, elem_id="process-full-video-settings-actions"):
                    add_user_settings_btn = gr.Button("\u2795", size="sm", min_width=1, visible=default_model_type == ADD_USER_SETTINGS_MODEL_TYPE, elem_classes=["wangp-assistant-chat__template-tool-icon-btn"])
                    delete_user_settings_btn = gr.Button("\U0001F5D1\uFE0F", size="sm", min_width=1, visible=_is_user_process_value(default_process_name), elem_classes=["wangp-assistant-chat__template-tool-icon-btn", "wangp-assistant-chat__template-tool-icon-btn--danger"])
                    settings_actions_placeholder = gr.HTML("<div class='process-full-video-settings-action-placeholder'></div>", visible=default_model_type != ADD_USER_SETTINGS_MODEL_TYPE and not _is_user_process_value(default_process_name))
            with gr.Row(visible=_process_choices_have_user_settings(default_process_choices), elem_id="process-full-video-user-settings-hint-row") as process_user_settings_hint_row:
                gr.HTML(value=USER_SETTINGS_HINT_HTML)
            with gr.Row():
                source_path = gr.Textbox(label="Source Video Path File", value=default_source_path, scale=3)
            with gr.Row():
                output_path = gr.Textbox(label="Output File Path File (None for auto, Full Name or Target Folder)", value=default_output_path, scale=3)
                continue_enabled = gr.Checkbox(label="Continue", value=default_continue_enabled, elem_classes="cbx_bottom", scale=1)
            with gr.Row():
                output_resolution = gr.Dropdown(output_resolution_choices, value=default_output_resolution, label="Output Resolution")
                target_ratio = gr.Dropdown(RATIO_CHOICES if _has_process_outpaint(default_process_name) else RATIO_CHOICES_WITH_EMPTY, value=default_target_ratio if _has_process_outpaint(default_process_name) else "", label="Target Ratio", visible=_has_process_outpaint(default_process_name))
                process_strength = gr.Slider(label="Process Strength (LoRA Multiplier)", minimum=0.0, maximum=3.0, step=0.01, value=1.0 if _has_process_outpaint(default_process_name) else default_process_strength, visible=not _has_process_outpaint(default_process_name))
            with gr.Row():
                chunk_size_seconds = gr.Number(label="Chunk Size (seconds)", value=default_chunk_size_seconds, precision=2)
                sliding_window_overlap = gr.Slider(label="Sliding Window Overlap", minimum=1, maximum=overlap_max, step=overlap_step, value=default_overlap_value)
            with gr.Row():
                start_seconds = gr.Textbox(label="Start (s/MM:SS(.xx)/HH:MM:SS(.xx))", value=default_start_seconds, placeholder="seconds, MM:SS(.xx), or HH:MM:SS(.xx)")
                end_seconds = gr.Textbox(label="End (s/MM:SS(.xx)/HH:MM:SS(.xx))", value=default_end_seconds, placeholder="seconds, MM:SS(.xx), or HH:MM:SS(.xx)")
                source_audio_track = gr.Dropdown(source_audio_track_choices, value=default_source_audio_track, label="Source Audio Track")
            with gr.Row():
                prompt_text = gr.Textbox(
                    label="Prompt (timed blocks supported: MM:SS(.xx) / HH:MM:SS(.xx))",
                    value=default_prompt,
                    lines=1,
                    placeholder=TIMED_PROMPT_EXAMPLE,
                )
            with gr.Row():
                start_btn = gr.Button("Start Process")
                abort_btn = gr.Button("Stop", interactive=False)
            status_html = gr.HTML(value=_render_chunk_status_html(0, 0, 0, "Idle", "Waiting to start..."))
            preview_image = gr.Image(label="Last Frame Preview", type="pil")
            output_file = gr.HTML(value=_render_output_file_html(""))
            preview_refresh = gr.Textbox(value="", visible=False)
            tab_refresh_trigger = gr.Textbox(value="", visible=False)

        self.on_tab_outputs = [tab_refresh_trigger]

        gr.on(
            [
                source_path.change,
                process_strength.change,
                output_path.change,
                prompt_text.change,
                continue_enabled.change,
                source_audio_track.change,
                output_resolution.change,
                target_ratio.change,
                chunk_size_seconds.change,
                sliding_window_overlap.change,
                start_seconds.change,
                end_seconds.change,
            ],
            fn=_store_process_form_memory,
            inputs=[process_form_memory, active_process_name_state, self.state, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            outputs=[process_form_memory],
            queue=False,
            show_progress="hidden",
        )
        process_model_type.change(
            fn=_change_process_model_type,
            inputs=[process_form_memory, active_process_name_state, process_model_type, self.state, self.lset_name, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            outputs=[process_form_memory, active_process_name_state, process_name, process_user_settings_hint_row, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            queue=False,
            show_progress="hidden",
        )
        process_name.change(
            fn=_change_process_name,
            inputs=[process_form_memory, active_process_name_state, process_name, process_model_type, self.state, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            outputs=[process_form_memory, active_process_name_state, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            queue=False,
            show_progress="hidden",
        )
        add_user_settings_btn.click(
            fn=_add_user_process_link,
            inputs=[process_name, self.state, self.lset_name, user_process_refs],
            outputs=[user_process_refs, process_model_type, process_name, process_user_settings_hint_row, active_process_name_state, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder],
            show_progress="hidden",
        )
        delete_user_settings_btn.click(
            fn=_delete_user_process_link,
            inputs=[process_form_memory, process_name, self.state, user_process_refs, source_path],
            outputs=[user_process_refs, process_model_type, process_name, process_user_settings_hint_row, active_process_name_state, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            show_progress="hidden",
        )
        self.refresh_form_trigger.change(
            fn=_refresh_process_form_from_main,
            inputs=[self.refresh_form_trigger, process_form_memory, active_process_name_state, process_model_type, self.state, self.lset_name, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            outputs=[process_model_type, process_name, process_user_settings_hint_row, process_form_memory, active_process_name_state, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            queue=False,
            show_progress="hidden",
        )
        tab_refresh_trigger.change(
            fn=_refresh_process_form_from_main,
            inputs=[tab_refresh_trigger, process_form_memory, active_process_name_state, process_model_type, self.state, self.lset_name, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            outputs=[process_model_type, process_name, process_user_settings_hint_row, process_form_memory, active_process_name_state, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            queue=False,
            show_progress="hidden",
        )
        start_btn.click(
            fn=start_process,
            inputs=[self.state, process_name, user_process_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
            outputs=[status_html, output_file, preview_refresh, start_btn, abort_btn],
            queue=False,
            show_progress="hidden",
            show_progress_on=[],
        )
        preview_refresh.change(fn=refresh_preview, inputs=[preview_refresh], outputs=[preview_image], queue=False, show_progress="hidden")
        abort_btn.click(fn=stop_process, outputs=[start_btn, abort_btn], queue=False, show_progress="hidden")
