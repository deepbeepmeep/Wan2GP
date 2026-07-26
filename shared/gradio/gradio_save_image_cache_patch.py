from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from threading import RLock
from typing import Any
import weakref


WANGP_GRADIO_IMAGE_DEBUG = False

_installed = False
_lock = RLock()
_cache: dict[tuple[int, str, str, str, tuple[int, int]], tuple[weakref.ReferenceType[Any], str]] = {}
_content_cache: OrderedDict[tuple[str, str, str, tuple[int, int], str], str] = OrderedDict()
_CONTENT_CACHE_LIMIT = 256


def _debug(message: str) -> None:
    if WANGP_GRADIO_IMAGE_DEBUG:
        print(f"[WanGP gradio-image-cache] {message}", flush=True)


def _content_key(y, cache_dir, format):
    digest = hashlib.blake2b(y.tobytes(), digest_size=16).hexdigest()
    return (str(cache_dir), format, y.mode, y.size, digest)


def _effective_format(image, format):
    if "A" in image.getbands() and image.getchannel("A").getextrema()[0] < 255:
        return "png"
    return format


def _existing_cache_file(image, cache_dir):
    filename = getattr(image, "filename", None)
    if not filename:
        return None
    path = Path(filename).resolve()
    cache_root = Path(cache_dir).resolve()
    if path.exists() and (path == cache_root or cache_root in path.parents):
        return str(path)
    return None


def _encode_pil_to_bytes(image, format):
    from gradio import processing_utils

    if format.lower() == "webp":
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="WEBP", lossless=True, method=6)
        return buffer.getvalue(), "webp"

    try:
        return processing_utils.encode_pil_to_bytes(image, format), format
    except (KeyError, ValueError):
        return processing_utils.encode_pil_to_bytes(image, "png"), "png"


def _save_pil_to_mode_aware_cache(image, cache_dir, format):
    bytes_data, suffix = _encode_pil_to_bytes(image, format)
    digest = hashlib.sha256()
    digest.update(f"{suffix}:{image.mode}:{image.size[0]}x{image.size[1]}:".encode())
    digest.update(bytes_data)
    temp_dir = Path(cache_dir) / digest.hexdigest()
    temp_dir.mkdir(exist_ok=True, parents=True)
    path = (temp_dir / f"image.{suffix}").resolve()
    if not path.exists() or path.read_bytes() != bytes_data:
        path.write_bytes(bytes_data)
    return str(path)


def _remove_key(key):
    with _lock:
        _cache.pop(key, None)


def _store_identity(key, y, path, format):
    try:
        ref = weakref.ref(y, lambda _ref, key=key: _remove_key(key))
    except TypeError:
        _debug(f"cache_skip_no_weakref id={id(y)} mode={y.mode} size={y.size} format={format} path={path}")
        return
    with _lock:
        _cache[key] = (ref, path)


def _store_content(key, path):
    with _lock:
        _content_cache[key] = path
        _content_cache.move_to_end(key)
        while len(_content_cache) > _CONTENT_CACHE_LIMIT:
            _content_cache.popitem(last=False)


def install() -> bool:
    global _installed
    if _installed:
        return True

    import PIL.Image
    from gradio import image_utils

    original_save_image = image_utils.save_image
    if getattr(original_save_image, "_wangp_save_image_cache_installed", False):
        _installed = True
        return True

    def patched_save_image(y, cache_dir: str, format: str = "webp"):
        if isinstance(y, (str, Path)):
            _debug(f"already_file path={y}")
            return original_save_image(y, cache_dir, format)

        if not isinstance(y, PIL.Image.Image):
            return original_save_image(y, cache_dir, format)

        existing_path = _existing_cache_file(y, cache_dir)
        if existing_path is not None:
            _debug(f"already_file_image id={id(y)} mode={y.mode} size={y.size} path={existing_path}")
            return existing_path

        effective_format = _effective_format(y, format)
        format_label = f"{format}->{effective_format}" if effective_format != format else format

        key = (id(y), str(cache_dir), effective_format, y.mode, y.size)
        with _lock:
            entry = _cache.get(key)
            if entry is not None:
                ref, path = entry
                if ref() is y and Path(path).exists():
                    _debug(f"cache_hit id={id(y)} mode={y.mode} size={y.size} format={format_label} path={path}")
                    return path
                _cache.pop(key, None)

        raw_key = _content_key(y, cache_dir, effective_format)
        with _lock:
            content_path = _content_cache.get(raw_key)
            if content_path is not None and Path(content_path).exists():
                _content_cache.move_to_end(raw_key)
                _debug(f"content_cache_hit id={id(y)} mode={y.mode} size={y.size} format={format_label} path={content_path}")
                _store_identity(key, y, content_path, effective_format)
                return content_path
            if content_path is not None:
                _content_cache.pop(raw_key, None)

        path = _save_pil_to_mode_aware_cache(y, cache_dir, effective_format)
        _store_identity(key, y, path, effective_format)
        _store_content(raw_key, path)
        _debug(f"cache_store id={id(y)} mode={y.mode} size={y.size} format={format_label} path={path}")
        return path

    patched_save_image._wangp_save_image_cache_installed = True
    patched_save_image._wangp_original_save_image = original_save_image
    image_utils.save_image = patched_save_image
    _installed = True
    return True


def clear() -> None:
    with _lock:
        _cache.clear()
        _content_cache.clear()


__all__ = ["WANGP_GRADIO_IMAGE_DEBUG", "clear", "install"]
