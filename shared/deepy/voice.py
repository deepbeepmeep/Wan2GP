"""Shared microphone upload/transcription endpoint for Gradio and Deepy web."""
from __future__ import annotations

import threading
import time
import tempfile
import uuid
import gc
import asyncio
from pathlib import Path
from shared.utils.media_imports import open_import_file

from fastapi import HTTPException, Request, UploadFile
from shared.deepy.config import DEEPY_VOICE_LANGUAGE_KEY, normalize_deepy_voice_language

_voice_lock = threading.Lock()
_installed = False
_preparations = {}
_preparations_lock = threading.Lock()
VOICE_LIMIT = 32 * 1024 * 1024
VOICE_MODE_KEY = "voice_transcription_mode"
VOICE_MODE_CHOICES = [("Disabled", "disabled"), ("CPU (Always available as runs on CPU but may be slower)", "cpu"), ("Cuda (will need to wait GPU is available but will be faster)", "gpu"), ("Auto (will pick CPU or GPU depending on availability)", "auto")]


def voice_mode(config):
    mode = config.get(VOICE_MODE_KEY, "auto")
    if mode not in {value for _, value in VOICE_MODE_CHOICES}:
        raise ValueError(f"Unknown voice transcription mode: {mode}")
    return mode


async def save_upload(upload: UploadFile, directory: Path, *, limit: int, extensions: set[str], preserve_filename: bool = False) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in extensions:
        raise HTTPException(415, "Unsupported file type.")
    path, writer = open_import_file(directory, upload.filename if preserve_filename else uuid.uuid4().hex + suffix)
    size = 0
    try:
        with writer:
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


def transcribe_recording(path, *, service, language=None, prepared_model=None, check_cancelled=lambda: None, report_status=lambda value: None):
    import torch
    from shared.deepy.transcription import transcribe_media
    from shared.utils.process_locks import try_acquire_GPU_ressources, release_GPU_ressources

    with _voice_lock:
        check_cancelled()
        mode = voice_mode(service._deps.get_server_config())
        if mode == "disabled":
            raise HTTPException(403, "Microphone transcription is disabled in Configuration.")
        cuda_available = torch.cuda.is_available()
        if mode == "gpu" and not cuda_available:
            raise HTTPException(409, "CUDA is not available. Select CPU or Auto in Configuration.")
        device = "cpu"
        waiting = time.perf_counter()
        if cuda_available and mode in {"gpu", "auto"}:
            while True:
                check_cancelled()
                with service._mutation_lock:
                    session = service._session
                    restoring = service._restoration is not None and service._restoration["pending"]
                    busy = service._threads or session.worker_active or session.queued_job_count or restoring or service.generation_running
                    if not busy and try_acquire_GPU_ressources(service._state, "deepy_voice", "Voice transcription"):
                        device = "cuda"
                        break
                if mode == "auto":
                    break
                if time.perf_counter() - waiting >= 1:
                    report_status("waiting_gpu")
                time.sleep(.1)
        try:
            report_status("transcribing")
            print(f"[Voice] Whisper large-v3: {device.upper()}; mode={mode}; language={language or 'Auto'}; GPU wait={time.perf_counter() - waiting:.2f}s.", flush=True)
            prepared = {} if prepared_model is None else {"prepared_model": prepared_model, "check_cancelled": check_cancelled}
            check_cancelled()
            return transcribe_media(str(path), timestamp_type="none", device=device, model_name="large-v3", language=language, **prepared)["text"]
        finally:
            if device == "cuda":
                release_GPU_ressources(service._state, "deepy_voice")


class RecordingPreparation:
    """One bounded, cancellable CPU preload shared by both browser interfaces."""
    def __init__(self, recording_id, service):
        self.id, self.service = recording_id, service
        self.cancelled = threading.Event()
        self.audio_ready = threading.Event()
        self.ready = threading.Event()
        self.done = threading.Event()
        self.created = time.monotonic()
        self.path = None
        self.language = None
        self.text = ""
        self.error = ""
        self.submitted = False
        self.model = []
        self.status = "loading"

    def report_status(self, value):
        self.status = value

    def check_cancelled(self):
        # The browser stops recording after three minutes. Reclaim abandoned
        # preparations even when pagehide/cancel never reaches the server.
        if self.cancelled.is_set() or (not self.audio_ready.is_set() and time.monotonic() - self.created > 240):
            raise InterruptedError("Voice recording cancelled.")

    def run(self):
        from shared.deepy.transcription import _load_whisper_large_v3
        import torch
        try:
            while not _voice_lock.acquire(timeout=.1):
                self.check_cancelled()
            try:
                self.check_cancelled()
                self.model.append(_load_whisper_large_v3(torch.device("cpu"), check_cancelled=self.check_cancelled))
                self.status = "ready"
                self.ready.set()
            finally:
                _voice_lock.release()
            while not self.audio_ready.wait(.1):
                self.check_cancelled()
            self.check_cancelled()
            self.text = transcribe_recording(self.path, service=self.service, language=self.language, prepared_model=self.model, check_cancelled=self.check_cancelled, report_status=self.report_status)
        except InterruptedError:
            self.error = "Voice recording cancelled."
        except Exception as exc:
            self.error = str(exc)
        finally:
            if self.model or self.error:
                self.model.clear()
                gc.collect()
            with _preparations_lock:
                if self.path is not None:
                    self.path.unlink(missing_ok=True)
                self.done.set()
                self.ready.set()
                if _preparations.get(self.id) is self:
                    del _preparations[self.id]


def mount_voice_routes(app, *, get_service, dependencies=None, language=None):
    from fastapi import File
    from starlette.concurrency import run_in_threadpool

    def recording_language():
        configured = get_service()._deps.get_server_config().get(DEEPY_VOICE_LANGUAGE_KEY, "auto")
        value = normalize_deepy_voice_language(configured if language is None else language)
        return None if value == "auto" else value

    @app.get("/deepy_api/voice", dependencies=dependencies or [])
    def voice_status():
        from shared.deepy.transcription import _whisper_medium_files_present
        from shared.deepy.assets import WHISPER_LARGE_V3_FOLDER
        from shared.utils import files_locator

        directory = files_locator.locate_folder(WHISPER_LARGE_V3_FOLDER, error_if_none=False)
        return {"download_required": not _whisper_medium_files_present(Path(directory) if directory else None), "model": "large-v3", "language": recording_language(), "mode": voice_mode(get_service()._deps.get_server_config())}

    @app.post("/deepy_api/voice/{recording_id}/prepare", dependencies=dependencies or [])
    async def prepare(recording_id: uuid.UUID, request: Request):
        service = get_service()
        if voice_mode(service._deps.get_server_config()) == "disabled":
            raise HTTPException(403, "Microphone transcription is disabled in Configuration.")
        key = str(recording_id)
        with _preparations_lock:
            if _preparations:
                raise HTTPException(409, "Another voice recording is being prepared or transcribed. Please wait.")
            job = RecordingPreparation(key, service)
            _preparations[key] = job
            threading.Thread(target=job.run, name="wangp-voice-prepare", daemon=True).start()
        try:
            while not job.ready.is_set():
                if await request.is_disconnected():
                    job.cancelled.set()
                else:
                    # A first download can outlast the recording limit while
                    # the connected browser is still waiting to submit audio.
                    job.created = time.monotonic()
                await asyncio.sleep(.1)
            if job.error:
                raise HTTPException(409 if job.cancelled.is_set() else 500, job.error)
        except BaseException:
            job.cancelled.set()
            raise
        return {"id": key}

    @app.post("/deepy_api/voice/{recording_id}/cancel", dependencies=dependencies or [])
    def cancel(recording_id: uuid.UUID):
        with _preparations_lock:
            job = _preparations.get(str(recording_id))
            if job is not None and job.service is get_service():
                job.cancelled.set()
        return {"cancelled": True}

    @app.get("/deepy_api/voice/{recording_id}", dependencies=dependencies or [])
    def preparation_status(recording_id: uuid.UUID):
        with _preparations_lock:
            job = _preparations.get(str(recording_id))
            if job is None or job.service is not get_service():
                raise HTTPException(410, "Voice preparation ended.")
            return {"status": job.status}

    @app.post("/deepy_api/voice/{recording_id}/transcribe", dependencies=dependencies or [])
    async def transcribe_prepared(recording_id: uuid.UUID, request: Request, file: UploadFile = File(...)):
        with _preparations_lock:
            job = _preparations.get(str(recording_id))
            if job is None or job.service is not get_service():
                raise HTTPException(410, "Voice preparation ended. Please start recording again.")
            if job.submitted:
                raise HTTPException(409, "This recording has already been submitted.")
            job.submitted = True
        try:
            path = await save_upload(file, Path(tempfile.gettempdir()) / "wangp" / "deepy_voice", limit=VOICE_LIMIT, extensions={".webm", ".mp4", ".m4a", ".wav", ".ogg", ".mp3"})
            with _preparations_lock:
                if job.done.is_set():
                    path.unlink(missing_ok=True)
                    raise HTTPException(410, job.error or "Voice preparation ended.")
                job.language = recording_language()
                job.path = path
                job.audio_ready.set()
            while not job.done.is_set():
                if await request.is_disconnected():
                    job.cancelled.set()
                await asyncio.sleep(.1)
            if job.error:
                raise HTTPException(409 if job.cancelled.is_set() else 500, job.error)
            return {"text": job.text}
        except BaseException:
            job.cancelled.set()
            raise

    @app.post("/deepy_api/transcribe", dependencies=dependencies or [])
    async def transcribe(file: UploadFile = File(...)):
        if voice_mode(get_service()._deps.get_server_config()) == "disabled":
            await file.close()
            raise HTTPException(403, "Microphone transcription is disabled in Configuration.")
        directory = Path(tempfile.gettempdir()) / "wangp" / "deepy_voice"
        path = await save_upload(file, directory, limit=VOICE_LIMIT, extensions={".webm", ".mp4", ".m4a", ".wav", ".ogg", ".mp3"})
        try:
            text = await run_in_threadpool(transcribe_recording, path, service=get_service(), language=recording_language())
            return {"text": text}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Transcription failed: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)


def install_gradio_routes(*, get_service, language=None):
    global _installed
    if _installed:
        return
    from fastapi import Depends
    from gradio.routes import API_PREFIX, App
    from shared.utils.fastapi_routes import iter_app_routes

    original = App.create_app

    def create_app(*args, **kwargs):
        app = original(*args, **kwargs)
        login_check = next(route.endpoint for route in iter_app_routes(app) if getattr(route, 'path', None) == f"{API_PREFIX}/login_check")
        mount_voice_routes(app, dependencies=[Depends(login_check)], get_service=get_service, language=language)
        return app

    App.create_app = staticmethod(create_app)
    _installed = True
