"""Shared microphone upload/transcription endpoint for Gradio and Deepy web."""
from __future__ import annotations

import threading
import tempfile
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

_voice_lock = threading.Lock()
_installed = False
VOICE_LIMIT = 32 * 1024 * 1024


async def save_upload(upload: UploadFile, directory: Path, *, limit: int, extensions: set[str]) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in extensions:
        raise HTTPException(415, "Unsupported file type.")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + suffix)
    size = 0
    try:
        with path.open("wb") as writer:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "File is too large.")
                writer.write(chunk)
        if not size:
            raise HTTPException(400, "The recording is empty.")
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def transcribe_recording(path, *, language=None):
    from shared.deepy.transcription import transcribe_media

    # Dictation must not compete with a running generation for VRAM.
    with _voice_lock:
        return transcribe_media(str(path), timestamp_type="none", device="cpu", model_name="large-v3", language=language)["text"]


def mount_voice_routes(app, *, dependencies=None, language=None):
    from fastapi import File
    from starlette.concurrency import run_in_threadpool

    @app.get("/deepy_api/voice", dependencies=dependencies or [])
    def voice_status():
        from shared.deepy.transcription import _whisper_medium_files_present
        from shared.deepy.assets import WHISPER_LARGE_V3_FOLDER
        from shared.utils import files_locator

        directory = files_locator.locate_folder(WHISPER_LARGE_V3_FOLDER, error_if_none=False)
        return {"download_required": not _whisper_medium_files_present(Path(directory) if directory else None), "model": "large-v3", "language": language}

    @app.post("/deepy_api/transcribe", dependencies=dependencies or [])
    async def transcribe(file: UploadFile = File(...)):
        directory = Path(tempfile.gettempdir()) / "wangp" / "deepy_voice"
        path = await save_upload(file, directory, limit=VOICE_LIMIT, extensions={".webm", ".mp4", ".m4a", ".wav", ".ogg", ".mp3"})
        try:
            text = await run_in_threadpool(transcribe_recording, path, language=language)
            return {"text": text}
        except Exception as exc:
            raise HTTPException(500, f"Transcription failed: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)


def install_gradio_routes(*, language=None):
    global _installed
    if _installed:
        return
    from fastapi import Depends
    from gradio.routes import API_PREFIX, App

    original = App.create_app

    def create_app(*args, **kwargs):
        app = original(*args, **kwargs)
        login_check = next(route.endpoint for route in app.routes if route.path == f"{API_PREFIX}/login_check")
        mount_voice_routes(app, dependencies=[Depends(login_check)], language=language)
        return app

    App.create_app = staticmethod(create_app)
    _installed = True
