"""The default rembg checkpoint, fetched before reference-image processing."""

import hashlib
import os

from shared.utils import files_locator as fl
from shared.utils.download import DownloadError, download_url
from shared.utils.download_progress import check_download_cancelled, resolve_download_gen


def query_download_def():
    # Preserve rembg's upstream asset and checksum; it is not mirrored in Wan2.1.
    return {"url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx", "filename": "rembg/u2net.onnx", "md5": "60024c5c889badc19c04ad937298a77b"}


def ensure_assets(gen=None):
    gen = resolve_download_gen(gen)
    check_download_cancelled(gen)
    definition = query_download_def()
    path = fl.locate_file(definition["filename"], error_if_none=False)
    if path is not None:
        return path
    path = fl.get_download_location(definition["filename"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        download_url(definition["url"], path, gen=gen)
    except OSError as exc:
        raise DownloadError(f"Unable to Download Background Removal Model: {exc}") from exc
    with open(path, "rb") as source:
        digest = hashlib.file_digest(source, "md5").hexdigest()
    if digest != definition["md5"]:
        os.remove(path)
        raise DownloadError("Background Removal Model Checksum Mismatch; Please Retry")
    return path
