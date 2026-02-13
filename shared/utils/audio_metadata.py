import struct
from typing import Optional
import json
import os
import re
from datetime import datetime

def write_wav_text_chunk(in_path: str, out_path: str, text: str,
                         fourcc: bytes = b'json', encoding: str = 'utf-8') -> None:
    """
    Insert (or replace) a custom RIFF chunk in a WAV file to hold an arbitrary string.
    - in_path:  source WAV path
    - out_path: destination WAV path (can be the same as in_path for in-place write)
    - text:     the string to store (e.g., JSON)
    - fourcc:   4-byte chunk ID; default b'json'
    - encoding: encoding for the string payload; default 'utf-8'

    Notes:
      * Keeps all original chunks as-is; if a chunk with the same fourcc exists,
        its payload is replaced; otherwise a new chunk is appended at the end.
      * Pads the chunk to even length per RIFF rules.
      * Supports standard little-endian RIFF/WAVE (not RF64 or RIFX).
    """
    data = open(in_path, 'rb').read()
    if len(data) < 12 or data[:4] not in (b'RIFF',) or data[8:12] != b'WAVE':
        raise ValueError("Not a standard little-endian RIFF/WAVE file (RF64/RIFX not supported).")
    if len(fourcc) != 4 or not all(32 <= b <= 126 for b in fourcc):
        raise ValueError("fourcc must be 4 printable ASCII bytes (e.g., b'json').")

    payload = text.encode(encoding)

    # Parse existing chunks
    pos = 12  # after 'RIFF' + size (4+4) and 'WAVE' (4)
    n = len(data)
    chunks = []  # list[(cid: bytes, payload: bytes)]
    while pos + 8 <= n:
        cid = data[pos:pos+4]
        size = struct.unpack_from('<I', data, pos+4)[0]
        start, end = pos + 8, pos + 8 + size
        if end > n:
            raise ValueError("Corrupt WAV: chunk size exceeds file length.")
        chunks.append((cid, data[start:end]))
        pos = end + (size & 1)  # pad to even

    # Replace existing or append new
    replaced = False
    new_chunks = []
    for cid, cdata in chunks:
        if cid == fourcc and not replaced:
            new_chunks.append((cid, payload))
            replaced = True
        else:
            new_chunks.append((cid, cdata))
    if not replaced:
        new_chunks.append((fourcc, payload))  # append at the end (often after 'data')

    # Rebuild RIFF body
    out_parts = [b'WAVE']
    for cid, cdata in new_chunks:
        out_parts.append(cid)
        out_parts.append(struct.pack('<I', len(cdata)))
        out_parts.append(cdata)
        if len(cdata) & 1:
            out_parts.append(b'\x00')  # pad to even size
    body = b''.join(out_parts)
    riff = b'RIFF' + struct.pack('<I', len(body)) + body

    with open(out_path, 'wb') as f:
        f.write(riff)


def read_wav_text_chunk(path: str, fourcc: bytes = b'json', encoding: str = 'utf-8') -> Optional[str]:
    """
    Read and return the string stored in a custom RIFF chunk from a WAV file.
    Returns None if the chunk isn't present.

    - path:     WAV file path
    - fourcc:   4-byte chunk ID to look for (default b'json')
    - encoding: decoding used for the stored bytes (default 'utf-8')
    """
    data = open(path, 'rb').read()
    if len(data) < 12 or data[:4] not in (b'RIFF',) or data[8:12] != b'WAVE':
        raise ValueError("Not a standard little-endian RIFF/WAVE file (RF64/RIFX not supported).")
    if len(fourcc) != 4:
        raise ValueError("fourcc must be 4 bytes.")

    pos = 12
    n = len(data)
    while pos + 8 <= n:
        cid = data[pos:pos+4]
        size = struct.unpack_from('<I', data, pos+4)[0]
        start, end = pos + 8, pos + 8 + size
        if end > n:
            raise ValueError("Corrupt WAV: chunk size exceeds file length.")
        if cid == fourcc:
            raw = data[start:end]
            return raw.decode(encoding, errors='strict')
        pos = end + (size & 1)

    return None

def _write_mp3_text_tag(path: str, text: str, tag_key: str = "WanGP") -> None:
    try:
        from mutagen.id3 import ID3, ID3NoHeaderError, TXXX
    except Exception as exc:
        raise RuntimeError("mutagen is required for mp3 metadata") from exc
    try:
        tag = ID3(path)
    except ID3NoHeaderError:
        tag = ID3()
    for key in list(tag.keys()):
        frame = tag.get(key)
        if isinstance(frame, TXXX) and frame.desc == tag_key:
            del tag[key]
    tag.add(TXXX(encoding=3, desc=tag_key, text=[text]))
    tag.save(path)


def _read_mp3_text_tag(path: str, tag_key: str = "WanGP") -> Optional[str]:
    try:
        from mutagen.id3 import ID3, ID3NoHeaderError, TXXX, COMM
    except Exception:
        return None
    try:
        tag = ID3(path)
    except ID3NoHeaderError:
        return None
    for frame in tag.getall("TXXX"):
        if isinstance(frame, TXXX) and frame.desc == tag_key:
            if frame.text:
                return frame.text[0]
    for frame in tag.getall("COMM"):
        if isinstance(frame, COMM) and frame.desc == tag_key:
            return frame.text[0] if frame.text else None
    return None


def save_audio_metadata(path, configs):
    ext = os.path.splitext(path)[1].lower()
    payload = json.dumps(configs)
    if ext == ".mp3":
        _write_mp3_text_tag(path, payload)
    elif ext == ".wav":
        write_wav_text_chunk(path, path, payload)
    else:
        raise ValueError(f"Unsupported audio metadata format: {ext}")


def read_audio_metadata(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        raw = _read_mp3_text_tag(path)
    elif ext == ".wav":
        raw = read_wav_text_chunk(path)
    else:
        return None
    if not raw:
        return None
    return json.loads(raw)


_CREATION_KEYS = (
    "creation_date",
    "creation_datetime",
    "created_at",
    "created_on",
    "creation_timestamp",
    "created_timestamp",
)
_DATE_KEY_PARTS = ("date", "time", "created", "timestamp")
_DATE_KEY_EXCLUDE = ("generation_time", "pause_seconds", "duration_seconds", "video_length")


def _parse_datetime_value(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        if 1900 <= value <= 3000:
            try:
                return datetime(int(value), 1, 1)
            except Exception:
                return None
        if value <= 0:
            return None
        try:
            return datetime.fromtimestamp(float(value))
        except Exception:
            return None

    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{10}(\.\d+)?", text) or re.fullmatch(r"\d{13}", text):
        try:
            ts = float(text)
            if ts > 1_000_000_000_000:
                ts /= 1000.0
            return datetime.fromtimestamp(ts)
        except Exception:
            pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if re.match(r"^\d{4}:\d{2}:\d{2}\s", text):
        text = text.replace(":", "-", 2)

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y-%m-%d-%Hh%Mm%Ss",
        "%Y%m%d",
        "%Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _iter_tag_values(value):
    if value is None:
        return
    if hasattr(value, "text"):
        txt = value.text
        if isinstance(txt, (list, tuple)):
            for item in txt:
                yield item
        else:
            yield txt
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_tag_values(item)
        return
    yield value


def extract_creation_datetime_from_metadata(metadata):
    if not isinstance(metadata, dict):
        return None
    for key in _CREATION_KEYS:
        dt = _parse_datetime_value(metadata.get(key))
        if dt is not None:
            return dt

    extra_info = metadata.get("extra_info")
    if isinstance(extra_info, dict):
        for key in _CREATION_KEYS:
            dt = _parse_datetime_value(extra_info.get(key))
            if dt is not None:
                return dt

    for source in (metadata, extra_info if isinstance(extra_info, dict) else {}):
        for key, value in source.items():
            lkey = str(key).strip().lower()
            if any(part in lkey for part in _DATE_KEY_EXCLUDE):
                continue
            if not any(part in lkey for part in _DATE_KEY_PARTS):
                continue
            dt = _parse_datetime_value(value)
            if dt is not None:
                return dt
    return None


def _extract_native_audio_datetime(path):
    try:
        from mutagen import File
    except Exception:
        return None
    try:
        audio = File(path, easy=False)
    except Exception:
        return None
    tags = None if audio is None else getattr(audio, "tags", None)
    if tags is None:
        return None

    if hasattr(tags, "getall"):
        for frame_name in ("TDRC", "TDEN", "TORY", "TYER", "TDAT", "TIME"):
            try:
                frames = tags.getall(frame_name)
            except Exception:
                frames = []
            for frame in frames:
                for item in _iter_tag_values(frame):
                    dt = _parse_datetime_value(item)
                    if dt is not None:
                        return dt
        try:
            txxx_frames = tags.getall("TXXX")
        except Exception:
            txxx_frames = []
        for frame in txxx_frames:
            desc = str(getattr(frame, "desc", "")).lower()
            if not any(part in desc for part in _DATE_KEY_PARTS):
                continue
            for item in _iter_tag_values(frame):
                dt = _parse_datetime_value(item)
                if dt is not None:
                    return dt

    items = tags.items() if hasattr(tags, "items") else []
    for key, value in items:
        lkey = str(key).lower()
        if any(part in lkey for part in _DATE_KEY_EXCLUDE):
            continue
        if not any(part in lkey for part in _DATE_KEY_PARTS + ("icrd", "\xa9day", "year")):
            continue
        for item in _iter_tag_values(value):
            dt = _parse_datetime_value(item)
            if dt is not None:
                return dt
    return None


def _get_file_creation_datetime(path):
    # For uploaded files, preserving browser-provided lastModified maps naturally to mtime.
    return datetime.fromtimestamp(os.path.getmtime(path))


def resolve_audio_creation_datetime(path, wangp_metadata=None):
    metadata = wangp_metadata
    if metadata is None:
        try:
            metadata = read_audio_metadata(path)
        except Exception:
            metadata = None

    dt = extract_creation_datetime_from_metadata(metadata)
    if dt is not None:
        return dt

    dt = _extract_native_audio_datetime(path)
    if dt is not None:
        return dt

    return _get_file_creation_datetime(path)
