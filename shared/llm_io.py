from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import os
import threading
from typing import Any


_LOCK = threading.RLock()
_LOG_PATH: Path | None = None
_DATA_URI_PREFIX = "data:"


def configure_llm_io(folder: str | os.PathLike[str] | None) -> Path | None:
    global _LOG_PATH
    with _LOCK:
        if not folder:
            _LOG_PATH = None
            return None
        target = Path(folder).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        _LOG_PATH = target / f"llm_io_{stamp}_{os.getpid()}.log"
        _LOG_PATH.write_text(
            "WanGP LLM I/O transcript\n"
            "Text is preserved verbatim. Non-text content is written using its known type/name, token IDs, or numeric values.\n"
            "Binary media is represented by source, media type, dimensions, and size; encoded binary bodies are not dumped.\n\n",
            encoding="utf-8",
        )
        print(f"[LLM I/O] Logging LLM traffic to {_LOG_PATH}")
        return _LOG_PATH


def get_llm_io_path() -> Path | None:
    with _LOCK:
        return _LOG_PATH


def llm_io_enabled() -> bool:
    with _LOCK:
        return _LOG_PATH is not None


def log_llm_io(direction: str, engine: str, stream: str, payload: Any, **metadata: Any) -> None:
    with _LOCK:
        if _LOG_PATH is None:
            return
        direction_key = str(direction or "").strip().upper()
        direction_label = "OUT → LLM" if direction_key == "OUT" else "IN ← LLM" if direction_key == "IN" else direction_key
        header = f"[{datetime.now().astimezone().isoformat(timespec='milliseconds')}] [{direction_label}] engine={engine} stream={stream}"
        lines = ["=" * 100, header, "-" * 100]
        if metadata:
            lines.extend(_render(metadata))
            lines.append("-" * 100)
        lines.extend(_render(payload))
        with _LOG_PATH.open("a", encoding="utf-8", newline="\n") as writer:
            writer.write("\n".join(lines) + "\n\n")


def media_descriptor(media: Any) -> dict[str, Any]:
    if hasattr(media, "size") and hasattr(media, "mode") and not isinstance(media, (str, os.PathLike)):
        width, height = media.size
        return {
            "source": str(getattr(media, "filename", "") or "<in-memory image>"),
            "media_type": str(getattr(media, "format", "") or type(media).__name__),
            "width": int(width),
            "height": int(height),
            "mode": str(media.mode),
        }
    source = Path(media).resolve()
    descriptor: dict[str, Any] = {"source": str(source), "bytes": source.stat().st_size}
    try:
        from PIL import Image

        with Image.open(source) as image:
            descriptor.update({"media_type": image.get_format_mimetype() or image.format or "image", "width": image.width, "height": image.height, "mode": image.mode})
    except (ImportError, OSError):
        descriptor["media_type"] = "file"
    return descriptor


def known_token_ids(tokenizer: Any) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ("bos", "eos", "pad", "unk", "sep", "cls", "mask"):
        token_id = getattr(tokenizer, f"{name}_token_id", None)
        if token_id is None:
            continue
        token_name = getattr(tokenizer, f"{name}_token", None)
        if token_name is None and hasattr(tokenizer, "convert_ids_to_tokens"):
            token_name = tokenizer.convert_ids_to_tokens(int(token_id))
        result[name] = {"id": int(token_id), "name": str(token_name or name)}
    return result


def token_id_descriptor(tokenizer: Any, token_id: int | None) -> dict[str, Any] | None:
    if token_id is None:
        return None
    token_name = tokenizer.convert_ids_to_tokens(int(token_id)) if hasattr(tokenizer, "convert_ids_to_tokens") else ""
    return {"id": int(token_id), "name": str(token_name or "<unknown>")}


def _render(value: Any, indent: int = 0, seen: set[int] | None = None) -> list[str]:
    prefix = " " * indent
    seen = set() if seen is None else seen
    value = _plain_value(value)
    if isinstance(value, str):
        return _render_text(value, prefix)
    if value is None or isinstance(value, (bool, int, float)):
        return [prefix + str(value)]
    identity = id(value)
    if identity in seen:
        return [prefix + f"<{type(value).__name__}: recursive reference>"]
    if isinstance(value, Mapping):
        seen.add(identity)
        lines = []
        for key, item in value.items():
            item = _binary_descriptor(key, item)
            label = prefix + str(key) + ":"
            if _is_scalar(item):
                rendered = _render(item, 0, seen)
                if len(rendered) == 1:
                    lines.append(label + " " + rendered[0])
                else:
                    lines.append(label + " |")
                    lines.extend(" " * (indent + 2) + line for line in rendered)
            else:
                lines.append(label)
                lines.extend(_render(item, indent + 2, seen))
        seen.remove(identity)
        return lines or [prefix + "<empty mapping>"]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seen.add(identity)
        lines = []
        for item in value:
            item_lines = _render(item, indent + 2, seen)
            lines.append(prefix + "-" + ((" " + item_lines[0].lstrip()) if item_lines else ""))
            lines.extend(item_lines[1:])
        seen.remove(identity)
        return lines or [prefix + "<empty sequence>"]
    return [prefix + str(value)]


def _render_text(value: str, prefix: str) -> list[str]:
    if "\n" not in value:
        return [prefix + value]
    return [prefix + line for line in value.splitlines()] or [prefix]


def _plain_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name} ({value.value})"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{type(value).__name__}: {len(value):,} bytes>"
    if is_dataclass(value) and not isinstance(value, type):
        return {"content_type": type(value).__name__, **{field.name: getattr(value, field.name) for field in fields(value)}}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        descriptor = {"content_type": type(value).__name__, "shape": tuple(value.shape), "dtype": str(value.dtype)}
        if hasattr(value, "device"):
            descriptor["device"] = str(value.device)
        return descriptor
    if hasattr(value, "__dict__"):
        return {"content_type": type(value).__name__, **{key: item for key, item in vars(value).items() if not key.startswith("_")}}
    return value


def _binary_descriptor(key: Any, value: Any) -> Any:
    key_name = str(key).casefold()
    if isinstance(value, str) and value.startswith(_DATA_URI_PREFIX):
        media_header, _, encoded = value.partition(",")
        return f"<{media_header}; encoded characters={len(encoded):,}>"
    if key_name == "data" and isinstance(value, str) and len(value) > 256 and _looks_like_encoded_binary(value):
        return f"<encoded content: {len(value):,} characters>"
    return value


def _looks_like_encoded_binary(value: str) -> bool:
    compact = value.replace("\r", "").replace("\n", "")
    return len(compact) % 4 == 0 and all(character.isalnum() or character in "+/=" for character in compact)


def _is_scalar(value: Any) -> bool:
    value = _plain_value(value)
    return value is None or isinstance(value, (str, bool, int, float))
