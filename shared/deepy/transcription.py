from __future__ import annotations

import gc
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
import whisper
from safetensors.torch import load_file as load_safetensors_file

from shared.deepy import video_tools as deepy_video_tools
from shared.ffmpeg_setup import download_ffmpeg
from shared.utils import files_locator as fl


WHISPER_MEDIUM_FOLDER = "whisper_medium"
WHISPER_MEDIUM_REPO = "DeepBeepMeep/Wan2.1"
WHISPER_MEDIUM_CONFIG_FILENAME = "config.json"
WHISPER_MEDIUM_WEIGHTS_FILENAME = "model.safetensors"
_TEMP_ROOT = Path(__file__).resolve().parents[2] / "_temp_codex" / "deepy_transcribe"
_TIMESTAMP_TYPE_ALIASES = {
    "none": None,
    "off": None,
    "disabled": None,
    "segment": "segment",
    "segments": "segment",
    "word": "word",
    "words": "word",
}
_WHISPER_MEDIUM_REQUIRED_FILES = (WHISPER_MEDIUM_CONFIG_FILENAME, WHISPER_MEDIUM_WEIGHTS_FILENAME)


def normalize_timestamp_type(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if len(normalized) == 0:
        return "segment"
    if normalized not in _TIMESTAMP_TYPE_ALIASES:
        raise ValueError("timestamp_type must be 'segment', 'word', or 'none'.")
    return _TIMESTAMP_TYPE_ALIASES[normalized]


def _get_main_callable(name: str) -> Any:
    main_module = sys.modules.get("__main__")
    return None if main_module is None else getattr(main_module, str(name or "").strip(), None)


def _whisper_medium_files_present(model_dir: Path | None) -> bool:
    if model_dir is None or not model_dir.is_dir():
        return False
    return all((model_dir / filename).is_file() for filename in _WHISPER_MEDIUM_REQUIRED_FILES)


def _ensure_whisper_medium_assets(model_dir: Path | None = None) -> None:
    if _whisper_medium_files_present(model_dir):
        return
    process_files_def = _get_main_callable("process_files_def")
    if callable(process_files_def):
        process_files_def(repoId=WHISPER_MEDIUM_REPO, sourceFolderList=[WHISPER_MEDIUM_FOLDER], fileList=[list(_WHISPER_MEDIUM_REQUIRED_FILES)])


def _whisper_medium_dir() -> Path:
    located = fl.locate_folder(WHISPER_MEDIUM_FOLDER, error_if_none=False)
    located_path = None if located is None else Path(located).resolve()
    _ensure_whisper_medium_assets(located_path)
    located = fl.locate_folder(WHISPER_MEDIUM_FOLDER, error_if_none=False)
    if located is not None:
        resolved = Path(located).resolve()
        if _whisper_medium_files_present(resolved):
            return resolved
    fallback = Path("e:/ml/wan2gp/ckpts") / WHISPER_MEDIUM_FOLDER
    if fallback.is_dir() and _whisper_medium_files_present(fallback):
        return fallback.resolve()
    raise FileNotFoundError(
        f"Unable to locate the Whisper medium folder '{WHISPER_MEDIUM_FOLDER}' in the configured checkpoints paths."
    )


def _load_whisper_medium(device: torch.device) -> whisper.Whisper:
    model_dir = _whisper_medium_dir()
    config_path = model_dir / WHISPER_MEDIUM_CONFIG_FILENAME
    weights_path = model_dir / WHISPER_MEDIUM_WEIGHTS_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Whisper config file not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Whisper weights file not found: {weights_path}")
    with config_path.open("r", encoding="utf-8") as reader:
        config = json.load(reader)
    dims = whisper.model.ModelDimensions(**dict(config.get("dims", {}) or {}))
    model = whisper.model.Whisper(dims)
    model.load_state_dict(load_safetensors_file(str(weights_path), device="cpu"))
    alignment_heads = str(config.get("alignment_heads", "") or "").strip()
    if len(alignment_heads) > 0:
        model.set_alignment_heads(alignment_heads.encode("ascii"))
    model.eval()
    if device.type == "cuda":
        return model.to(device=device)
    return model.to(device=device, dtype=torch.float32)


def _make_temp_audio_path() -> Path:
    _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return (_TEMP_ROOT / f"{uuid.uuid4().hex}.wav").resolve()


def _prepare_audio_input(source_path: str, audio_track_no: int | None = None) -> tuple[str, list[Path]]:
    download_ffmpeg()
    temp_audio_path = _make_temp_audio_path()
    deepy_video_tools.extract_audio(source_path, str(temp_audio_path), audio_track_no=audio_track_no, audio_codec="wav")
    return str(temp_audio_path), [temp_audio_path]


def _round_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except Exception:
        return None


def _serialize_segments(segments: list[dict[str, Any]], timestamp_type: str | None) -> list[dict[str, Any]]:
    serialized = []
    include_words = timestamp_type == "word"
    for segment in segments:
        item = {
            "start": _round_timestamp(segment.get("start", None)),
            "end": _round_timestamp(segment.get("end", None)),
            "text": str(segment.get("text", "") or "").strip(),
        }
        if include_words:
            words = []
            for word in list(segment.get("words", []) or []):
                words.append(
                    {
                        "word": str(word.get("word", "") or ""),
                        "start": _round_timestamp(word.get("start", None)),
                        "end": _round_timestamp(word.get("end", None)),
                        "probability": None if word.get("probability", None) is None else round(float(word["probability"]), 4),
                    }
                )
            if len(words) > 0:
                item["words"] = words
        serialized.append(item)
    return serialized


def transcribe_media(source_path: str, *, timestamp_type: str | None = None, audio_track_no: int | None = None) -> dict[str, Any]:
    normalized_timestamp_type = normalize_timestamp_type(timestamp_type)
    source_path = str(source_path or "").strip()
    if len(source_path) == 0 or not os.path.isfile(source_path):
        raise FileNotFoundError(f"Media file not found: {source_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_path, temporary_paths = _prepare_audio_input(source_path, audio_track_no=audio_track_no)
    model = None
    try:
        model = _load_whisper_medium(device)
        raw_result = model.transcribe(
            audio_path,
            verbose=None,
            fp16=device.type == "cuda",
            word_timestamps=normalized_timestamp_type == "word",
        )
    finally:
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                pass
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    segments = list(raw_result.get("segments", []) or [])
    payload = {
        "text": str(raw_result.get("text", "") or "").strip(),
        "language": str(raw_result.get("language", "") or "").strip(),
        "segment_count": int(len(segments)),
    }
    if normalized_timestamp_type is not None:
        payload["timestamp_type"] = normalized_timestamp_type
        payload["segments"] = _serialize_segments(segments, normalized_timestamp_type)
    return payload
