# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Texture image helpers: PNG/JPEG encoding, decoding to uint8 arrays and header-only size probing."""
from __future__ import annotations

import io
import struct

import numpy as np
from PIL import Image

PNG_COMPRESS_LEVEL = 3   # zlib level: 4096^2 textures encode several times faster than at PIL's default 6, ~5% larger
JPEG_QUALITY = 95
MIME_TYPES = {"png": "image/png", "jpeg": "image/jpeg"}


def encode_image(array: np.ndarray, fmt: str = "png") -> bytes:
    """Encodes a uint8 [H, W], [H, W, 1|3|4] array (rows top to bottom) as PNG or JPEG (JPEG needs 1 or 3 channels)."""
    array = np.ascontiguousarray(array[..., 0] if array.ndim == 3 and array.shape[2] == 1 else array)
    buffer = io.BytesIO()
    if fmt == "png":
        Image.fromarray(array).save(buffer, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    elif fmt == "jpeg":
        if array.ndim == 3 and array.shape[2] == 4:
            raise ValueError("JPEG cannot store an alpha channel")
        Image.fromarray(array).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    else:
        raise ValueError(f"Unsupported image format {fmt!r}")
    return buffer.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    """Decodes an image file to a uint8 array [H, W, C] with C = 1 (grey), 2 (grey + alpha), 3 or 4 channels."""
    with Image.open(io.BytesIO(data)) as image:
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            array = (np.asarray(image, dtype=np.int64) >> 8).clip(0, 255).astype(np.uint8)
        else:
            if image.mode not in ("L", "LA", "RGB", "RGBA"):
                image = image.convert("RGBA" if "A" in image.mode or "transparency" in image.info else "RGB")
            array = np.array(image)
    return array[..., None] if array.ndim == 2 else array


def to_channels(array: np.ndarray, channels: int) -> np.ndarray:
    """Converts a decoded [H, W, 1..4] uint8 image to grey/RGB (channels 1 or 3) or RGBA (4), dropping alpha when asked."""
    color = np.repeat(array[..., :1], 3, axis=2) if array.shape[2] < 3 else array[..., :3]
    if channels == 1:
        return array[..., :1] if array.shape[2] < 3 else color[..., :1]
    if channels == 3:
        return np.ascontiguousarray(color)
    alpha = array[..., -1:] if array.shape[2] in (2, 4) else np.full_like(array[..., :1], 255)
    return np.concatenate([color, alpha], axis=2)


def pack_textures(arrays: list[np.ndarray], padding: int = 2) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Packs uint8 [h, w, C] images (same C) into one atlas by shelves (rows of decreasing height), each surrounded by
    `padding` replicated edge texels against filtering bleed. Returns the atlas and the (x, y, w, h) box of each image
    in it (x, y = column, row of its first texel)."""
    sizes = [(a.shape[0] + 2 * padding, a.shape[1] + 2 * padding) for a in arrays]
    width = max(max(w for _, w in sizes), int(np.ceil(np.sqrt(sum(h * w for h, w in sizes)))))
    x = y = shelf = 0
    corners = {}
    for i in sorted(range(len(arrays)), key=lambda i: -sizes[i][0]):
        h, w = sizes[i]
        if x + w > width:
            x, y, shelf = 0, y + shelf, 0
        corners[i] = (x, y)
        x, shelf = x + w, max(shelf, h)
    atlas = np.zeros((y + shelf, width, arrays[0].shape[2]), np.uint8)
    if atlas.shape[2] in (2, 4):
        atlas[..., -1] = 255  # unused areas opaque, so that an opaque set stays opaque
    boxes = []
    for i, array in enumerate(arrays):
        x, y = corners[i]
        atlas[y:y + sizes[i][0], x:x + sizes[i][1]] = np.pad(array, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
        boxes.append((x + padding, y + padding, array.shape[1], array.shape[0]))
    return atlas, boxes


def remap_uvs(uvs: np.ndarray, box: tuple[int, int, int, int], atlas_shape) -> np.ndarray:
    """glTF uvs of an image moved to `box` of an atlas of shape (H, W, ...). uvs outside [0, 1] are wrapped first
    (repeat), which is exact for vertices but not for triangles crossing a repeat boundary."""
    uvs = np.where((uvs < 0) | (uvs > 1), uvs - np.floor(uvs), uvs)
    x, y, w, h = box
    return (uvs * [w, h] + [x, y]) / [atlas_shape[1], atlas_shape[0]]


def image_size(read) -> tuple[int, int] | None:
    """(width, height) of a PNG, JPEG or WebP image from its header only. read(offset, size) returns the bytes of the
    image file at that offset (fewer at its end). Returns None for other formats or a malformed header."""
    head = read(0, 30)
    if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 24:
        return struct.unpack(">II", head[16:24])
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP" and len(head) >= 30:
        kind = head[12:16]
        if kind == b"VP8 ":
            w, h = struct.unpack("<HH", head[26:30])
            return w & 0x3FFF, h & 0x3FFF
        if kind == b"VP8L":
            bits = int.from_bytes(head[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if kind == b"VP8X":
            return int.from_bytes(head[24:27], "little") + 1, int.from_bytes(head[27:30], "little") + 1
        return None
    if head[:2] != b"\xff\xd8":
        return None
    pos = 2
    while True:
        segment = read(pos, 9)
        if len(segment) < 2 or segment[0] != 0xFF:
            return None
        marker = segment[1]
        if marker == 0xFF:                                   # fill byte
            pos += 1
        elif marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # markers without a length
            pos += 2
        elif len(segment) < 9:
            return None
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # start of frame
            height, width = struct.unpack(">HH", segment[5:9])
            return width, height
        else:
            pos += 2 + struct.unpack(">H", segment[2:4])[0]


def bytes_reader(data: bytes, start: int = 0, end: int | None = None):
    """read(offset, size) over data[start:end], for image_size."""
    end = len(data) if end is None else end
    return lambda offset, size: bytes(data[start + offset:min(start + offset + size, end)])


def file_reader(handle, start: int = 0, end: int | None = None):
    """read(offset, size) over an open binary file between start and end, for image_size."""
    def read(offset, size):
        handle.seek(start + offset)
        return handle.read(size if end is None else max(0, min(size, end - start - offset)))
    return read
