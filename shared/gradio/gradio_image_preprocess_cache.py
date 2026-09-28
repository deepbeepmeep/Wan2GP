"""Avoid decoding unchanged file-backed PIL inputs on every UI callback."""
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
from pathlib import Path
from threading import RLock

from PIL import Image


_cache = OrderedDict()
_lock = RLock()
_bytes = 0
_MAX_BYTES = 96 * 1024 * 1024
_MAX_ENTRIES = 32


def clear():
    global _bytes
    with _lock:
        _cache.clear()
        _bytes = 0


def _copy(image):
    # Callbacks may edit pixels or metadata; cached pixels never leave here.
    result = image.copy()
    result.info = deepcopy(image.info)
    return result


def install():
    from gradio import image_utils

    original = image_utils.preprocess_image
    if getattr(original, '_wangp_input_cache', False):
        return

    @wraps(original)
    def preprocess(payload, cache_dir, format, image_mode, type):
        global _bytes
        # Preserve native lazy/animated, streaming, numpy and filepath behavior.
        if payload is None or type != 'pil' or image_mode not in ('RGB', 'RGBA', 'L') or not payload.path or (payload.url or '').startswith('data:') or Path(payload.orig_name or '').suffix.lower() in ('.gif', '.svg'):
            return original(payload, cache_dir, format, image_mode, type)
        path = Path(payload.path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, payload.orig_name, image_mode)
        with _lock:
            entry = _cache.get(key)
            if entry is not None:
                _cache.move_to_end(key)
        if entry is not None:
            return _copy(entry[0])
        result = original(payload, cache_dir, format, image_mode, type)
        if not isinstance(result, Image.Image):
            return result
        # Keep the uploaded file available when an app passes these unchanged
        # pixels into another image component, including an ImageEditor.
        from .gradio_save_image_cache_patch import remember_source_image
        remember_source_image(result, payload, cache_dir, format, cache_identity=False)
        # Pillow uses four bytes per RGB/RGBA pixel internally.
        size = result.width * result.height * (1 if result.mode == 'L' else 4)
        if size <= _MAX_BYTES:
            saved = _copy(result)
            with _lock:
                previous = _cache.pop(key, None)
                if previous is not None:
                    _bytes -= previous[1]
                _cache[key] = (saved, size)
                _bytes += size
                while _bytes > _MAX_BYTES or len(_cache) > _MAX_ENTRIES:
                    _, (_, removed_size) = _cache.popitem(last=False)
                    _bytes -= removed_size
        return result

    preprocess._wangp_input_cache = True
    image_utils.preprocess_image = preprocess
