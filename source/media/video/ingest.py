"""Video-ingest helpers with injectable download seams for tests/runtime."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import httpx
import requests

from source.utils.download_utils import _get_unique_target_path


class DownloadError(RuntimeError):
    """Raised when a remote video download fails."""


def _default_stream_download(url: str, destination_path: Path, timeout: int) -> None:
    with requests.Session() as session:
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(destination_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)


def download_file_if_url(
    file_url_or_path: str,
    download_target_dir: Path | str | None,
    *,
    task_id_for_logging: str | None = "generic_task",
    descriptive_name: str | None = None,
    default_extension: str = ".mp4",
    default_stem: str = "input",
    file_type_label: str = "video",
    timeout: int = 300,
    stream_download_impl=None,
) -> str:
    """Download a URL to a local file, or return the original path unchanged."""
    del task_id_for_logging, file_type_label  # Kept for compatibility.

    if not file_url_or_path:
        return file_url_or_path

    parsed = urlparse(file_url_or_path)
    if parsed.scheme not in {"http", "https"} or not download_target_dir:
        return file_url_or_path

    target_dir = Path(download_target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(parsed.path).name
    extension = Path(original_name).suffix or default_extension
    stem = descriptive_name or Path(original_name).stem or default_stem
    destination_path = _get_unique_target_path(target_dir, stem[:50], extension)

    downloader = stream_download_impl or _default_stream_download
    try:
        downloader(file_url_or_path, destination_path, timeout)
    except httpx.HTTPError as exc:
        raise DownloadError(str(exc)) from exc
    except OSError as exc:
        raise DownloadError(str(exc)) from exc

    return str(destination_path)
