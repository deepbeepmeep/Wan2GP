from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from shared.gradio.local_file_picker import IMAGE_FILE_EXTENSIONS, VIDEO_FILE_EXTENSIONS

from . import constants


BATCH_STATE_VERSION = 1


@dataclass(frozen=True)
class BatchItem:
    source_path: str
    output_path: str
    output_is_directory: bool = False


@dataclass(frozen=True)
class BatchFiles:
    state_path: Path
    log_path: Path
    state: dict | None
    resumed: bool
    renamed: bool
    batch_name: str
    completed: bool = False


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def parse_lines(value: str) -> list[str]:
    return [line.strip().strip('"') for line in str(value or "").splitlines() if len(line.strip().strip('"')) > 0]


def _normalize_local_path(path_text: str) -> str:
    return os.path.abspath(os.path.normpath(os.path.expanduser(str(path_text or "").strip())))


def _extensions_for_media(media_kind: str) -> set[str]:
    return IMAGE_FILE_EXTENSIONS if str(media_kind or "").strip().lower() == "image" else VIDEO_FILE_EXTENSIONS


def _media_label(media_kind: str) -> str:
    return "image" if str(media_kind or "").strip().lower() == "image" else "video"


def _is_supported_source(path: str, media_kind: str) -> bool:
    return Path(path).suffix.lower() in _extensions_for_media(media_kind)


def _has_wildcards(value: str) -> bool:
    return "*" in str(value or "")


def _compile_wildcard_pattern(pattern: str) -> re.Pattern:
    escaped_parts = [re.escape(part) for part in pattern.split("*")]
    regex = "^" + "(.*?)".join(escaped_parts) + "$"
    return re.compile(regex, re.IGNORECASE if os.name == "nt" else 0)


def _capture_wildcards(pattern: str, path: str) -> tuple[str, ...]:
    match = _compile_wildcard_pattern(pattern).match(path)
    return tuple(match.groups()) if match else ()


def _expand_source_line(line: str, media_kind: str) -> list[tuple[str, tuple[str, ...]]]:
    pattern = _normalize_local_path(line)
    media_label = _media_label(media_kind)
    if not _has_wildcards(pattern):
        if os.path.isdir(pattern):
            paths = [
                _normalize_local_path(str(path))
                for path in Path(pattern).iterdir()
                if path.is_file() and _is_supported_source(str(path), media_kind)
            ]
            paths = sorted(dict.fromkeys(paths), key=str.casefold)
            if len(paths) == 0:
                supported_text = ", ".join(sorted(_extensions_for_media(media_kind)))
                raise gr.Error(f"Source folder contains no supported {media_label}s ({supported_text}): {pattern}")
            return [(path, ()) for path in paths]
        if not os.path.isfile(pattern):
            raise gr.Error(f"Source {media_label} not found: {pattern}")
        if not _is_supported_source(pattern, media_kind):
            supported_text = ", ".join(sorted(_extensions_for_media(media_kind)))
            raise gr.Error(f"Source {media_label} must use one of these extensions: {supported_text}.")
        return [(pattern, ())]
    paths = [
        _normalize_local_path(path)
        for path in glob.glob(pattern)
        if os.path.isfile(path) and _is_supported_source(path, media_kind)
    ]
    paths = sorted(dict.fromkeys(paths), key=str.casefold)
    if len(paths) == 0:
        raise gr.Error(f"No supported source {media_label}s matched: {line}")
    return [(path, _capture_wildcards(pattern, path)) for path in paths]


def _fill_output_pattern(pattern: str, captures: tuple[str, ...], media_kind: str) -> str:
    if pattern.count("*") != len(captures):
        raise gr.Error("Input and output wildcard patterns must contain the same number of * tokens.")
    parts = pattern.split("*")
    output = []
    for index, part in enumerate(parts):
        output.append(part)
        if index < len(captures):
            output.append(captures[index])
    return _validate_output_path(_normalize_local_path("".join(output)), media_kind)


def _is_output_directory_line(line: str) -> bool:
    return len(line) > 0 and not _has_wildcards(line) and (line.endswith(("\\", "/")) or Path(_normalize_local_path(line)).is_dir())


def output_path_for_process(item: BatchItem) -> str:
    if not item.output_path or not item.output_is_directory or item.output_path.endswith(("\\", "/")):
        return item.output_path
    return item.output_path + os.sep


def _validate_output_path(path: str, media_kind: str = "video") -> str:
    if _media_label(media_kind) == "image":
        return path
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix and suffix not in constants.SUPPORTED_OUTPUT_CONTAINERS:
        supported_text = ", ".join(f".{container}" for container in sorted(constants.SUPPORTED_OUTPUT_CONTAINERS))
        raise gr.Error(f"Batch output files must use one of these container extensions: {supported_text}.")
    return path


def expand_batch_items(source_text: str, output_text: str, *, media_kind: str = "video") -> list[BatchItem]:
    source_lines = parse_lines(source_text)
    if len(source_lines) == 0:
        raise gr.Error(f"Batch Source {_media_label(media_kind).title()} Paths must contain at least one source path or wildcard.")
    expanded_by_line = [_expand_source_line(line, media_kind) for line in source_lines]
    output_lines = parse_lines(output_text)

    items: list[BatchItem] = []
    if len(output_lines) == 0:
        for source_group in expanded_by_line:
            items.extend(BatchItem(source_path=source_path, output_path="") for source_path, _captures in source_group)
        return items
    if len(output_lines) == 1 and _is_output_directory_line(output_lines[0]):
        output_dir = _normalize_local_path(output_lines[0])
        for source_group in expanded_by_line:
            items.extend(BatchItem(source_path=source_path, output_path=output_dir, output_is_directory=True) for source_path, _captures in source_group)
        return items
    if len(output_lines) != len(source_lines):
        raise gr.Error("Batch output paths must be empty, a single output folder, or have one line for each source path line.")

    for source_line, source_group, output_line in zip(source_lines, expanded_by_line, output_lines):
        output_line = output_line.strip()
        source_has_wildcards = _has_wildcards(source_line)
        if source_has_wildcards:
            if _is_output_directory_line(output_line):
                output_dir = _normalize_local_path(output_line)
                items.extend(BatchItem(source_path=source_path, output_path=output_dir, output_is_directory=True) for source_path, _captures in source_group)
                continue
            if not _has_wildcards(output_line):
                raise gr.Error("A wildcard source line needs a matching wildcard output line, an output folder, or an empty output field.")
            for source_path, captures in source_group:
                items.append(BatchItem(source_path=source_path, output_path=_fill_output_pattern(output_line, captures, media_kind)))
            continue
        if _has_wildcards(output_line):
            raise gr.Error("Wildcard output paths require a wildcard source path on the same line.")
        if len(source_group) > 1 and not _is_output_directory_line(output_line):
            raise gr.Error("A source folder needs an empty output field or an output folder.")
        output_is_directory = _is_output_directory_line(output_line)
        output_path = _normalize_local_path(output_line) if output_is_directory else _validate_output_path(_normalize_local_path(output_line), media_kind)
        for source_path, _captures in source_group:
            items.append(BatchItem(source_path=source_path, output_path=output_path, output_is_directory=output_is_directory))
    return items


def _sanitize_name(value: str) -> str:
    text = str(value or "").strip()
    token = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return token.strip("._") or "batch"


def _batch_target_dir(item: BatchItem) -> Path:
    if not item.output_path:
        return Path(item.source_path).resolve().parent
    output = Path(item.output_path).resolve()
    return output if item.output_is_directory else output.parent


def _batch_base_path(batch_name: str, first_item: BatchItem) -> Path:
    return _batch_target_dir(first_item) / _sanitize_name(batch_name)


def build_signature(request, items: list[BatchItem], *, batch_name: str, process_model_type: str = "", media_kind: str = "video") -> dict:
    return {
        "batch_name": str(batch_name or "").strip(),
        "media_kind": _media_label(media_kind),
        "process_model_type": str(process_model_type or "").strip(),
        "process_name": request.process_name,
        "process_strength": str(request.process_strength),
        "prompt_text": request.prompt_text,
        "source_audio_track": request.source_audio_track,
        "output_resolution": request.output_resolution,
        "target_ratio": request.target_ratio,
        "chunk_size_seconds": str(request.chunk_size_seconds),
        "sliding_window_overlap": str(request.sliding_window_overlap),
        "start_seconds": request.start_seconds,
        "end_seconds": request.end_seconds,
        "items": [{"source_path": item.source_path, "output_path": item.output_path} for item in items],
    }


def signature_hash(signature: dict) -> str:
    payload = json.dumps(signature, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_state(batch_name: str, signature: dict, items: list[BatchItem]) -> dict:
    created_at = now_iso()
    return {
        "version": BATCH_STATE_VERSION,
        "batch_name": str(batch_name or "").strip(),
        "signature": signature,
        "signature_hash": signature_hash(signature),
        "created_at": created_at,
        "updated_at": created_at,
        "status": "running",
        "next_index": 0,
        "items": [
            {
                "index": index,
                "source_path": item.source_path,
                "output_path": item.output_path,
                "actual_output_path": "",
                "status": "pending",
                "started_at": "",
                "ended_at": "",
                "elapsed_seconds": 0.0,
                "chunks_completed": 0,
                "chunks_total": 0,
                "error": "",
            }
            for index, item in enumerate(items)
        ],
    }


def read_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _state_matches(state: dict | None, current_hash: str) -> bool:
    if not isinstance(state, dict):
        return False
    if state.get("signature_hash") == current_hash:
        return True
    signature = state.get("signature")
    if not isinstance(signature, dict):
        return False
    normalized_signature = dict(signature)
    normalized_signature.pop("continue_enabled", None)
    return signature_hash(normalized_signature) == current_hash


def _normalize_resumed_state_signature(state: dict, current_hash: str) -> None:
    state["signature_hash"] = current_hash
    signature = state.get("signature")
    if isinstance(signature, dict):
        signature.pop("continue_enabled", None)


def _is_interrupted(state: dict | None) -> bool:
    return isinstance(state, dict) and state.get("status") in {"running", "stopped"} and int(state.get("next_index") or 0) < len(state.get("items") or [])


def _is_completed(state: dict | None) -> bool:
    return isinstance(state, dict) and int(state.get("next_index") or 0) >= len(state.get("items") or [])


def _variant_batch_name(batch_name: str, first_item: BatchItem) -> tuple[str, Path]:
    base_name = str(batch_name or "").strip() or "batch"
    for index in range(2, 10000):
        candidate_name = f"{base_name}_{index}"
        candidate_path = _batch_base_path(candidate_name, first_item)
        if not candidate_path.with_suffix(".json").exists() and not candidate_path.with_suffix(".log").exists() and not candidate_path.with_suffix(".jsonl").exists():
            return candidate_name, candidate_path
    raise gr.Error(f"Unable to find a free batch filename near {_batch_target_dir(first_item)}")


def _find_resumable_batch_file(base_path: Path, current_signature_hash: str, batch_name: str) -> BatchFiles | None:
    state_path = base_path.with_suffix(".json")
    state = read_state(state_path)
    if not _state_matches(state, current_signature_hash):
        return None
    if _is_interrupted(state):
        _normalize_resumed_state_signature(state, current_signature_hash)
        return BatchFiles(state_path=state_path, log_path=state_path.with_suffix(".log"), state=state, resumed=True, renamed=False, batch_name=str(state.get("batch_name") or batch_name).strip())
    if _is_completed(state):
        _normalize_resumed_state_signature(state, current_signature_hash)
        return BatchFiles(state_path=state_path, log_path=state_path.with_suffix(".log"), state=state, resumed=False, renamed=False, batch_name=str(state.get("batch_name") or batch_name).strip(), completed=True)
    return None


def resolve_batch_files(batch_name: str, first_item: BatchItem, current_signature_hash: str, continue_enabled: bool) -> BatchFiles:
    batch_name = str(batch_name or "").strip()
    base_path = _batch_base_path(batch_name, first_item)
    state_path = base_path.with_suffix(".json")
    log_path = base_path.with_suffix(".log")
    if continue_enabled:
        resumable = _find_resumable_batch_file(base_path, current_signature_hash, batch_name)
        if resumable is not None:
            return resumable
    if state_path.exists() or log_path.exists():
        renamed_batch_name, variant = _variant_batch_name(batch_name, first_item)
        return BatchFiles(state_path=variant.with_suffix(".json"), log_path=variant.with_suffix(".log"), state=None, resumed=False, renamed=True, batch_name=renamed_batch_name)
    return BatchFiles(state_path=state_path, log_path=log_path, state=None, resumed=False, renamed=False, batch_name=batch_name)


def _format_log_value(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _format_log_duration(seconds) -> str:
    try:
        total_seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return ""
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds_only = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_only:02d}" if hours > 0 else f"{minutes:02d}:{seconds_only:02d}"


def _format_log_chunks(completed, total) -> str:
    try:
        return f"{int(completed)} / {int(total)}"
    except (TypeError, ValueError):
        return ""


def _human_log_message(event: str, payload: dict) -> str:
    event = str(event or "").strip()
    batch_name = _format_log_value(payload.get("batch_name"))
    if event in {"started", "resumed"}:
        action = "Resumed" if event == "resumed" else "Started"
        total_files = _format_log_value(payload.get("total_files"))
        renamed = " Auto-renamed from requested batch name." if payload.get("renamed") else ""
        state_file = _format_log_value(payload.get("state_file"))
        total_text = f" with {total_files} file(s)" if total_files else ""
        state_text = f" State file: {state_file}." if state_file else ""
        return f'{action} batch "{batch_name}"{total_text}.{renamed}{state_text}'
    if event == "stopped":
        next_index = _format_log_value(payload.get("next_index"))
        if next_index.isdigit():
            return f"Batch stopped. Next file to process: {int(next_index) + 1}."
        return f"Batch stopped. Next file to process: {next_index}." if next_index else "Batch stopped."
    if event == "completed":
        completed_files = _format_log_value(payload.get("completed_files"))
        total_files = _format_log_value(payload.get("total_files"))
        failed_files = _format_log_value(payload.get("failed_files"))
        failures_text = ""
        if failed_files.isdigit() and int(failed_files) > 0:
            failures_text = f" ({failed_files} {'failure' if int(failed_files) == 1 else 'failures'})"
        return f"Batch completed. Files processed: {completed_files} / {total_files}{failures_text}."
    if event == "file_started":
        index = _format_log_value(payload.get("index"))
        source_path = _format_log_value(payload.get("source_path"))
        output_path = _format_log_value(payload.get("output_path"))
        file_text = f"File {int(index) + 1}" if index.isdigit() else "File"
        output_text = f" Output: {output_path}." if output_path else ""
        return f"{file_text} started. Source: {source_path}.{output_text}"
    if event in {"file_finished", "file_stopped"}:
        index = _format_log_value(payload.get("index"))
        source_path = _format_log_value(payload.get("source_path"))
        output_path = _format_log_value(payload.get("output_path"))
        file_text = f"File {int(index) + 1}" if index.isdigit() else "File"
        if event == "file_stopped":
            status_text = "stopped"
            detail = _format_log_value(payload.get("message"))
        else:
            success = bool(payload.get("success"))
            status_text = "succeeded" if success else "failed"
            detail = _format_log_value(payload.get("error"))
        duration = _format_log_duration(payload.get("elapsed_seconds"))
        chunks = _format_log_chunks(payload.get("chunks_completed"), payload.get("chunks_total"))
        parts = [f"{file_text} {status_text}."]
        if duration:
            parts.append(f"Duration: {duration}.")
        if chunks:
            parts.append(f"Chunks: {chunks}.")
        if source_path:
            parts.append(f"Source: {source_path}.")
        if output_path:
            parts.append(f"Output: {output_path}.")
        if detail:
            parts.append(f"Message: {detail}.")
        return " ".join(parts)
    details = ", ".join(f"{key}: {_format_log_value(value)}" for key, value in payload.items())
    return f"{event}. {details}" if details else event


def append_log(path: Path, event: str, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{now_iso()} - {_human_log_message(event, payload)}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
