"""Small gallery previews, generated only when the browser requests them."""
import asyncio
import base64
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from shared.deepy.gallery import _VIDEO_EXTENSIONS
from shared.deepy.video_tools import get_video_thumbnail_data_url


# Preview work cannot occupy all Gradio/API workers or spawn dozens of ffmpeg
# processes while session changes and inference are running.
_workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix='Media preview')


async def render_thumbnail(path):
    return await asyncio.get_running_loop().run_in_executor(_workers, thumbnail, path)


def thumbnail(path):
    path = os.path.normcase(os.path.abspath(path))
    stat = Path(path).stat()
    return _thumbnail(path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _thumbnail(path, mtime_ns, size):
    if Path(path).suffix.lower() in _VIDEO_EXTENSIONS:
        data = get_video_thumbnail_data_url(path)
        return base64.b64decode(data.split(',', 1)[1]) if data else None
    with Image.open(path) as source:
        source.draft('RGB', (512, 384))
        image = ImageOps.exif_transpose(source)
        image.thumbnail((512, 384))
        output = BytesIO()
        image.convert('RGB').save(output, format='JPEG', quality=80)
        return output.getvalue()
