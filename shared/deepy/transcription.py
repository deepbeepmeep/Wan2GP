from __future__ import annotations

import gc
import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch
import whisper
from mmgp import offload

from shared.deepy import video_tools as deepy_video_tools
from shared.deepy.assets import (
    WHISPER_MEDIUM_CONFIG_FILENAME,
    WHISPER_MEDIUM_FOLDER,
    WHISPER_MEDIUM_REPO,
    WHISPER_MEDIUM_REQUIRED_FILES,
    WHISPER_MEDIUM_WEIGHTS_FILENAME,
    WHISPER_LARGE_V3_FOLDER,
    WHISPER_LARGE_V3_REPO,
    query_deepy_download_defs,
)
from shared.ffmpeg_setup import download_ffmpeg
from shared.utils import files_locator as fl


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
_WHISPER_MEDIUM_REQUIRED_FILES = WHISPER_MEDIUM_REQUIRED_FILES


def normalize_timestamp_type(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if len(normalized) == 0:
        return "segment"
    if normalized not in _TIMESTAMP_TYPE_ALIASES:
        raise ValueError("timestamp_type must be 'segment', 'word', or 'none'.")
    return _TIMESTAMP_TYPE_ALIASES[normalized]


def _whisper_medium_files_present(model_dir: Path | None) -> bool:
    if model_dir is None or not model_dir.is_dir():
        return False
    return all((model_dir / filename).is_file() for filename in _WHISPER_MEDIUM_REQUIRED_FILES)


def _ensure_whisper_medium_assets(model_dir: Path | None = None) -> None:
    if _whisper_medium_files_present(model_dir):
        return
    from shared.utils.download import process_files_def
    for download_def in query_deepy_download_defs():
        process_files_def(**download_def)


def _whisper_medium_dir() -> Path:
    located = fl.locate_folder(WHISPER_MEDIUM_FOLDER, error_if_none=False)
    located_path = None if located is None else Path(located).resolve()
    _ensure_whisper_medium_assets(located_path)
    located = fl.locate_folder(WHISPER_MEDIUM_FOLDER, error_if_none=False)
    if located is not None:
        resolved = Path(located).resolve()
        if _whisper_medium_files_present(resolved):
            return resolved
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
    alignment_heads = str(config.get("alignment_heads", "") or "").strip()
    return _load_whisper(config["dims"], weights_path, device, alignment_heads.encode("ascii") if alignment_heads else None)


def _load_whisper(dims, weights_path, device, alignment_heads, preprocess_sd=None):
    model = whisper.model.Whisper(whisper.model.ModelDimensions(**dims))
    # Whisper keeps normalization weights in float32 and casts projections during inference.
    offload.load_model_data(model, str(weights_path), writable_tensors=False, default_dtype=torch.float32, preprocess_sd=preprocess_sd)
    if alignment_heads:
        model.set_alignment_heads(alignment_heads)
    model.eval()
    if device.type == "cuda":
        return model.to(device=device)
    return model.to(device=device, dtype=torch.float32)


def _large_v3_state_dict(state_dict):
    replacements = (
        ("model.", ""), (".layers.", ".blocks."), (".self_attn_layer_norm.", ".attn_ln."), (".encoder_attn_layer_norm.", ".cross_attn_ln."),
        (".self_attn.", ".attn."), (".encoder_attn.", ".cross_attn."), (".q_proj.", ".query."), (".k_proj.", ".key."), (".v_proj.", ".value."), (".out_proj.", ".out."),
        (".fc1.", ".mlp.0."), (".fc2.", ".mlp.2."), (".final_layer_norm.", ".mlp_ln."), ("encoder.layer_norm.", "encoder.ln_post."), ("decoder.layer_norm.", "decoder.ln."),
        (".embed_positions.weight", ".positional_embedding"), (".embed_tokens.", ".token_embedding."),
    )
    converted = {}
    for name, tensor in state_dict.items():
        for old, new in replacements:
            name = name.replace(old, new)
        converted[name] = tensor
    return converted


def _load_whisper_large_v3(device):
    from shared.utils.download import process_files_def

    folder = fl.locate_folder(WHISPER_LARGE_V3_FOLDER, error_if_none=False)
    if not _whisper_medium_files_present(Path(folder) if folder else None):
        process_files_def(repoId=WHISPER_LARGE_V3_REPO, sourceFolderList=[WHISPER_LARGE_V3_FOLDER], fileList=[["config.json", "model.safetensors"]])
    config_path = fl.locate_file(f"{WHISPER_LARGE_V3_FOLDER}/config.json")
    weights_path = fl.locate_file(f"{WHISPER_LARGE_V3_FOLDER}/model.safetensors")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    dims = dict(
        n_mels=config["num_mel_bins"], n_audio_ctx=config["max_source_positions"], n_audio_state=config["d_model"], n_audio_head=config["encoder_attention_heads"], n_audio_layer=config["encoder_layers"],
        n_vocab=config["vocab_size"], n_text_ctx=config["max_target_positions"], n_text_state=config["d_model"], n_text_head=config["decoder_attention_heads"], n_text_layer=config["decoder_layers"],
    )
    return _load_whisper(dims, weights_path, device, whisper._ALIGNMENT_HEADS["large-v3"], _large_v3_state_dict)


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


def transcribe_media(source_path: str, *, timestamp_type: str | None = None, audio_track_no: int | None = None, device: str | None = None, model_name: str = "medium", language: str | None = None) -> dict[str, Any]:
    normalized_timestamp_type = normalize_timestamp_type(timestamp_type)
    source_path = str(source_path or "").strip()
    if len(source_path) == 0 or not os.path.isfile(source_path):
        raise FileNotFoundError(f"Media file not found: {source_path}")
    device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
    audio_path, temporary_paths = _prepare_audio_input(source_path, audio_track_no=audio_track_no)
    model = None
    try:
        model = {"medium": _load_whisper_medium, "large-v3": _load_whisper_large_v3}[model_name](device)
        raw_result = model.transcribe(
            audio_path,
            verbose=None,
            fp16=device.type == "cuda",
            word_timestamps=normalized_timestamp_type == "word",
            language=language,
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
