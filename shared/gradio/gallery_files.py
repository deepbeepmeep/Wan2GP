from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import gradio as gr
from gradio import processing_utils
from gradio_client import utils as client_utils

from shared.deepy.media_registry import detect_media_type


_gallery_paths: set[str] = set()
_lock = threading.Lock()
_installed = False
_file_access_policy = None
_original_is_static_file = processing_utils.utils.is_static_file


def _path_key(path: Any) -> str:
    return os.path.normcase(str(Path(os.fspath(path)).resolve()))


def _is_exposed_gallery_path(path: Any) -> bool:
    try:
        key = _path_key(path)
    except (OSError, TypeError, ValueError):
        return False
    with _lock:
        return key in _gallery_paths


def _is_authorized_gallery_path(path: Any) -> bool:
    path = getattr(path, "path", path)
    if _is_exposed_gallery_path(path):
        return True
    try:
        candidate = Path(os.fspath(path)).resolve()
    except (OSError, TypeError, ValueError):
        return False
    with _lock:
        policy = _file_access_policy
    return candidate.is_file() and detect_media_type(str(candidate)) in {"image", "video", "audio"} and policy is not None and policy.can_read(candidate)


def _is_gallery_static_file(path: Any) -> bool:
    return _original_is_static_file(path) or _is_authorized_gallery_path(path)


def _check_gallery_event_files(data) -> None:
    def check(file_data: dict) -> None:
        path = file_data.get("path", "")
        if path and not client_utils.is_http_url_like(path) and not processing_utils.is_in_or_equal(path, processing_utils.get_upload_folder()) and not _is_authorized_gallery_path(path):
            raise gr.Error(f"File {path} is not in the cache folder and cannot be accessed.")

    client_utils.traverse(data, check, client_utils.is_file_obj)


def install(file_access_policy=None) -> None:
    global _file_access_policy, _installed
    if file_access_policy is not None:
        with _lock:
            _file_access_policy = file_access_policy
    if _installed:
        return
    processing_utils.utils.is_static_file = _is_gallery_static_file
    processing_utils.check_all_files_in_cache = _check_gallery_event_files
    _installed = True


def expose_gallery_files(paths: list[str]) -> None:
    files = [Path(path).resolve() for path in paths]
    with _lock:
        _gallery_paths.update(_path_key(path) for path in files)
    install()
    gr.set_static_paths(files)


__all__ = ["expose_gallery_files", "install"]
