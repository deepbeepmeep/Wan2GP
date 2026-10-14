# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
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
from shared.utils.utils import has_mesh_file_extension


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
    if has_mesh_file_extension(path):
        path = os.path.splitext(path)[0] + '.png'  # a 3D asset is previewed by the render saved beside it
    with Image.open(path) as source:
        source.draft('RGB', (512, 384))
        image = ImageOps.exif_transpose(source)
        image.thumbnail((512, 384))
        if image.mode in ('RGBA', 'LA', 'P'):
            image = Image.alpha_composite(Image.new('RGBA', image.size, (128, 128, 128, 255)), image.convert('RGBA'))
        output = BytesIO()
        image.convert('RGB').save(output, format='JPEG', quality=80)
        return output.getvalue()
