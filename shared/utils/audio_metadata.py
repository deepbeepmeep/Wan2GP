import struct
from typing import Optional
import json

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

def save_audio_metadata(path, configs):
    write_wav_text_chunk(path, path, json.dumps(configs))

def read_audio_metadata(path):
    return  json.loads(read_wav_text_chunk(path))