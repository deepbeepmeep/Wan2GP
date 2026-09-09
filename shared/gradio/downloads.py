import json
import mimetypes
import os
import queue as queue_module
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote


_download_jobs = {}
_file_downloads = {}
_file_download_tokens = {}
_gallery_downloads = {}
_download_jobs_lock = threading.Lock()
_download_routes_installed = False
_download_original_create_app = None


def _cleanup_download_jobs(ttl_seconds=600):
    now = time.time()
    with _download_jobs_lock:
        for token, job in list(_download_jobs.items()):
            if now - job.get("created", now) > ttl_seconds:
                _download_jobs.pop(token, None)


def _content_disposition(filename):
    fallback = re.sub(r'[^A-Za-z0-9._ -]+', '_', filename).strip() or "download"
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"


def register_download(filename, mime_type, iterator_factory):
    _cleanup_download_jobs()
    token = uuid.uuid4().hex
    with _download_jobs_lock:
        _download_jobs[token] = {
            "created": time.time(),
            "filename": filename,
            "mime_type": mime_type,
            "iterator_factory": iterator_factory,
        }
    return json.dumps({"url": f"/wangp_api/download/{token}", "filename": filename})


def register_file_download(path, mime_type=None):
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Download file does not exist: {file_path}")
    key = os.path.normcase(str(file_path))
    with _download_jobs_lock:
        token = _file_download_tokens.get(key)
        if token is None:
            token = uuid.uuid4().hex
            _file_download_tokens[key] = token
            _file_downloads[token] = {"path": str(file_path), "filename": file_path.name, "mime_type": mime_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"}
    return {"url": f"/wangp_api/download/{token}", "filename": file_path.name, "size_bytes": file_path.stat().st_size}


def register_gallery_download(media_id, path, mime_type=None):
    media_id = str(media_id or "").strip().casefold()
    if re.fullmatch(r"(?:visual|audio):[a-f0-9]{12}", media_id) is None:
        raise ValueError(f"Invalid Gallery media id: {media_id}")
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Gallery file does not exist: {file_path}")
    with _download_jobs_lock:
        _gallery_downloads[media_id] = {"path": str(file_path), "mime_type": mime_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"}
    return {"url": f"/wangp_api/gallery/media/{quote(media_id, safe='')}", "filename": file_path.name, "size_bytes": file_path.stat().st_size}


def stream_bytes(data, chunk_size=1024 * 1024):
    for offset in range(0, len(data), chunk_size):
        yield data[offset:offset + chunk_size]


def stream_writer(write_fn):
    chunks = queue_module.Queue(maxsize=8)
    stop_event = threading.Event()
    sentinel = object()

    def push(item):
        while not stop_event.is_set():
            try:
                chunks.put(item, timeout=0.5)
                return True
            except queue_module.Full:
                pass
        return False

    class StreamingWriter:
        def write(self, data):
            if not data:
                return 0
            data = bytes(data)
            if push(data):
                return len(data)
            raise BrokenPipeError("Download stream closed")

        def flush(self):
            pass

    def produce():
        try:
            if write_fn(StreamingWriter()) is False:
                raise RuntimeError("Failed to create download stream")
        except Exception as e:
            push(e)
        finally:
            push(sentinel)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = chunks.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()


def _pop_download_job(token):
    with _download_jobs_lock:
        return _download_jobs.pop(token, None)


def _get_file_download(token):
    with _download_jobs_lock:
        return _file_downloads.get(token)


def _get_gallery_download(media_id):
    with _download_jobs_lock:
        return _gallery_downloads.get(str(media_id or "").strip().casefold())


def _install_routes_on_app(fastapi_app):
    if getattr(fastapi_app, "_wangp_download_routes_installed", False):
        return

    @fastapi_app.get("/wangp_api/download/{token}")
    async def _wangp_download(token: str):
        from fastapi import Response
        from fastapi.responses import FileResponse, StreamingResponse
        job = _pop_download_job(token)
        if job is not None:
            headers = {"Content-Disposition": _content_disposition(job["filename"])}
            return StreamingResponse(job["iterator_factory"](), media_type=job["mime_type"], headers=headers)
        file_download = _get_file_download(token)
        if file_download is None or not Path(file_download["path"]).is_file():
            return Response("Download expired or not found", status_code=404)
        return FileResponse(file_download["path"], filename=file_download["filename"], media_type=file_download["mime_type"])

    @fastapi_app.get("/wangp_api/gallery/media/{media_id}")
    async def _wangp_gallery_media(media_id: str):
        from fastapi import Response
        from fastapi.responses import FileResponse
        gallery_download = _get_gallery_download(media_id)
        if gallery_download is None or not Path(gallery_download["path"]).is_file():
            return Response("Gallery media expired or not found", status_code=404)
        return FileResponse(gallery_download["path"], media_type=gallery_download["mime_type"])

    fastapi_app._wangp_download_routes_installed = True


def install_routes():
    global _download_routes_installed, _download_original_create_app
    if _download_routes_installed:
        return
    from gradio.routes import App
    _download_original_create_app = App.create_app

    def _patched_create_app(*args, **kwargs):
        fastapi_app = _download_original_create_app(*args, **kwargs)
        _install_routes_on_app(fastapi_app)
        return fastapi_app

    App.create_app = staticmethod(_patched_create_app)
    _download_routes_installed = True


__all__ = ["install_routes", "register_download", "register_file_download", "register_gallery_download", "stream_bytes", "stream_writer"]
