"""Bounded, lazy image previews served through Gradio's authorized file route."""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs

from PIL import Image, ImageOps
from starlette.responses import Response

from shared.utils.http_disconnect import DisconnectAwareFileResponse


_workers = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Gallery image preview')
_lock = RLock()
_cache = OrderedDict()
_cache_bytes = 0
_MAX_BYTES = 64 * 1024 * 1024
_MAX_ENTRIES = 256


def _cached(key):
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return True, _cache[key]
    return False, None


def _preview(key):
    global _cache_bytes
    # One worker bounds decode memory and coalesces duplicate queued requests.
    found, data = _cached(key)
    if found:
        return data
    path, _, *version = key
    previews = {}
    with Image.open(path) as source:
        # Animated images and already-small originals retain native playback
        # and pixels; only oversized still images need a derived preview.
        source_edge = max(source.size)
        animated = getattr(source, 'is_animated', False)
        if not animated and source_edge > 320:
            source.draft(source.mode, (1600, 1600))
            image = ImageOps.exif_transpose(source)
            if image.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
                image = image.convert('RGB')
        # Decode once and prepare both sizes while the pixels are available.
        for edge in (1600, 320):
            data = None
            if not animated and source_edge > edge:
                image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                output = BytesIO()
                image.save(output, format='PNG', compress_level=3)
                data = output.getvalue()
            previews[(path, edge, *version)] = data
    with _lock:
        for cache_key, data in previews.items():
            previous = _cache.pop(cache_key, None)
            _cache_bytes -= len(previous) if previous is not None else 0
            _cache[cache_key] = data
            _cache_bytes += len(data) if data is not None else 0
        while _cache_bytes > _MAX_BYTES or len(_cache) > _MAX_ENTRIES:
            _, removed = _cache.popitem(last=False)
            _cache_bytes -= len(removed) if removed is not None else 0
        return previews[key]


class GalleryFileResponse(DisconnectAwareFileResponse):
    async def __call__(self, scope, receive, send):
        edge = parse_qs(scope['query_string'].decode()).get('__wangp_gallery_preview')
        if edge is None:
            return await super().__call__(scope, receive, send)
        if edge not in (['320'], ['1600']):
            return await Response('Invalid gallery preview size.', status_code=400)(scope, receive, send)
        path = Path(self.path)
        if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff', '.gif'):
            return await super().__call__(scope, receive, send)
        stat = path.stat()
        key = (str(path), int(edge[0]), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        etag = '"' + sha256(repr(key).encode()).hexdigest() + '"'
        headers = {'ETag': etag, 'Cache-Control': 'private, no-cache'}
        request_headers = dict(scope['headers'])
        if request_headers.get(b'if-none-match') == etag.encode():
            return await Response(status_code=304, headers=headers)(scope, receive, send)
        # Cache hits never queue behind another image's decode/resize.
        found, data = _cached(key)
        if not found:
            data = await asyncio.get_running_loop().run_in_executor(_workers, _preview, key)
        if data is None:
            self.headers.update(headers)
            return await super().__call__(scope, receive, send)
        headers['Content-Length'] = str(len(data))
        return await Response(b'' if scope['method'] == 'HEAD' else data, media_type='image/png', headers=headers)(scope, receive, send)
